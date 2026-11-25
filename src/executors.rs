use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use log::{info, trace, warn};
use serde::Serialize;
use tokio::sync::mpsc::{Receiver, Sender, UnboundedSender};
use tokio::sync::{broadcast, Mutex};
use tokio::task::JoinHandle;

use crate::requests::{
    TextGenerationAggregatedResponse, TextGenerationBackend, TextGenerationRequest,
    TextRequestGenerator,
};

#[serde_with::serde_as]
#[derive(Clone, Serialize)]
pub struct ExecutorConfig {
    pub max_vus: u64,
    #[serde(rename = "duration_secs")]
    #[serde_as(as = "serde_with::DurationSeconds<u64>")]
    pub duration: Duration,
    pub rate: Option<f64>,
}

#[async_trait]
pub trait Executor {
    async fn run(
        &self,
        requests: Arc<Mutex<dyn TextRequestGenerator + Send>>,
        responses_tx: UnboundedSender<TextGenerationAggregatedResponse>,
        stop_sender: broadcast::Sender<()>,
    );
}

pub struct ConstantVUsExecutor {
    config: ExecutorConfig,
    backend: Box<dyn TextGenerationBackend + Send + Sync>,
}

impl ConstantVUsExecutor {
    pub fn new(
        backend: Box<dyn TextGenerationBackend + Send + Sync>,
        max_vus: u64,
        duration: Duration,
    ) -> ConstantVUsExecutor {
        Self {
            backend,
            config: ExecutorConfig {
                max_vus,
                duration,
                rate: None,
            },
        }
    }
}

#[async_trait]
impl Executor for ConstantVUsExecutor {
    async fn run(
        &self,
        requests: Arc<Mutex<dyn TextRequestGenerator + Send>>,
        responses_tx: UnboundedSender<TextGenerationAggregatedResponse>,
        stop_sender: broadcast::Sender<()>,
    ) {
        let start = std::time::Instant::now();
        // channel to handle ending VUs
        let (end_tx, mut end_rx): (Sender<bool>, Receiver<bool>) =
            tokio::sync::mpsc::channel(self.config.max_vus as usize);
        let active_vus = Arc::new(AtomicI64::new(0));
        // start all VUs

        let (timeout_tx, _) = tokio::sync::broadcast::channel::<()>(self.config.max_vus as usize);
        let timed_out = Arc::new(AtomicBool::new(false));
        // sleep for duration and send timeout signal
        {
            let timeout_duration = self.config.duration;
            let timeout_tx = timeout_tx.clone();
            let timed_out = timed_out.clone();
            tokio::spawn(async move {
                tokio::time::sleep(timeout_duration).await;
                timed_out.store(true, Ordering::SeqCst);
                // send timeout signal to all VUs
                let _ = timeout_tx.send(());
            });
        }

        for _ in 0..self.config.max_vus {
            let mut requests_guard = requests.lock().await;
            let request = Arc::from(requests_guard.generate_request());
            drop(requests_guard);
            start_vu(
                self.backend.clone(),
                request,
                responses_tx.clone(),
                end_tx.clone(),
                stop_sender.clone(),
                Some(timeout_tx.subscribe()),
            )
            .await;
            active_vus.fetch_add(1, Ordering::SeqCst);
        }
        let mut stop_receiver = stop_sender.subscribe();
        tokio::select! {
            _ = stop_receiver.recv() => {
                return;
            },
            _ = async {
                // replenish VUs as they finish
                while end_rx.recv().await.is_some() {
                    active_vus.fetch_sub(1, Ordering::SeqCst);
                    if timed_out.load(Ordering::SeqCst) || start.elapsed() >= self.config.duration {
                        info!("Duration reached, waiting for all VUs to finish...");
                        if active_vus.load(Ordering::SeqCst) == 0 {
                            // Signal end only after every VU has flushed its final/partial tokens.
                            let _ = responses_tx.send(TextGenerationAggregatedResponse::new_as_ended());
                            break;
                        }
                    } else {
                        let mut requests_guard = requests.lock().await;
                        let request = Arc::from(requests_guard.generate_request());
                        drop(requests_guard);
                        active_vus.fetch_add(1, Ordering::SeqCst);
                        start_vu(self.backend.clone(), request, responses_tx.clone(), end_tx.clone(), stop_sender.clone(), Some(timeout_tx.subscribe())).await;
                    }
                }
            }=>{}
        }
    }
}

async fn start_vu(
    backend: Box<dyn TextGenerationBackend + Send + Sync>,
    request: Arc<TextGenerationRequest>,
    responses_tx: UnboundedSender<TextGenerationAggregatedResponse>,
    end_tx: Sender<bool>,
    stop_sender: broadcast::Sender<()>,
    mut timeout_stop_receiver: Option<broadcast::Receiver<()>>,
) -> JoinHandle<()> {
    let mut stop_receiver = stop_sender.subscribe();
    tokio::spawn(async move {
        let (tx, mut rx): (
            Sender<TextGenerationAggregatedResponse>,
            Receiver<TextGenerationAggregatedResponse>,
        ) = tokio::sync::mpsc::channel(1);
        let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel();
        trace!("VU started with request: {:?}", request);
        let mut req_thread = tokio::spawn(async move {
            backend.generate_with_cancel(request.clone(), tx, cancel_rx).await;
        });
        let mut send_thread = tokio::spawn(async move {
            while let Some(response) = rx.recv().await {
                let _ = responses_tx.send(response);
            }
        });

        let cancelled = tokio::select! {
            _ = stop_receiver.recv() => true,
            _ = async {
                if let Some(ref mut receiver) = timeout_stop_receiver {
                    let _ = receiver.recv().await;
                } else {
                    std::future::pending::<()>().await; // wait forever if no timeout receiver is provided
                }
            } => true,
            _ = async {
                let _ = (&mut req_thread).await;
                let _ = (&mut send_thread).await;
            } => false,
        };

        if cancelled {
            // Ask the backend to close its SSE stream and flush any tokens
            // already streamed before timeout; abort if the backend ignores
            // cancellation so no HTTP socket leaks into subsequent stages.
            let _ = cancel_tx.send(());
            if tokio::time::timeout(Duration::from_millis(200), &mut req_thread)
                .await
                .is_err()
            {
                req_thread.abort();
                let _ = req_thread.await;
            }
            if tokio::time::timeout(Duration::from_millis(200), &mut send_thread)
                .await
                .is_err()
            {
                send_thread.abort();
                let _ = send_thread.await;
            }
        }
        // signal that the VU work is done
        let _ = end_tx.send(true).await;
    })
}

pub struct ConstantArrivalRateExecutor {
    config: ExecutorConfig,
    backend: Box<dyn TextGenerationBackend + Send + Sync>,
}

impl ConstantArrivalRateExecutor {
    pub fn new(
        backend: Box<dyn TextGenerationBackend + Send + Sync>,
        max_vus: u64,
        duration: Duration,
        rate: f64,
    ) -> ConstantArrivalRateExecutor {
        Self {
            backend,
            config: ExecutorConfig {
                max_vus,
                duration,
                rate: Some(rate),
            },
        }
    }
}

#[async_trait]
impl Executor for ConstantArrivalRateExecutor {
    async fn run(
        &self,
        requests: Arc<Mutex<dyn TextRequestGenerator + Send>>,
        responses_tx: UnboundedSender<TextGenerationAggregatedResponse>,
        stop_sender: broadcast::Sender<()>,
    ) {
        let start = std::time::Instant::now();
        let active_vus = Arc::new(AtomicI64::new(0));
        // channel to handle ending VUs
        let (end_tx, mut end_rx): (Sender<bool>, Receiver<bool>) =
            tokio::sync::mpsc::channel(self.config.max_vus as usize);
        let rate = self.config.rate.expect("checked in new()");
        // spawn new VUs every `tick_ms` to reach the expected `rate` per second, until the duration is reached
        let tick_ms = 10;
        let mut interval = tokio::time::interval(Duration::from_millis(tick_ms));

        let backend = self.backend.clone();
        let duration = self.config.duration;
        let max_vus = self.config.max_vus;
        let active_vus_thread = active_vus.clone();
        let mut stop_receiver_signal = stop_sender.subscribe();
        let vu_thread = tokio::spawn(async move {
            tokio::select! {
                _ = stop_receiver_signal.recv() => {},
                _= async {
                    let mut spawn_queue = 0.; // start with at least one VU
                    while start.elapsed() < duration {
                        spawn_queue += rate * (tick_ms as f64) / 1000.0;
                        // delay spawning if we can't spawn a full VU yet
                        if spawn_queue < 1.0 {
                            interval.tick().await;
                            continue;
                        }
                        // spawn VUs, keep track of the fraction of VU to spawn for the next iteration
                        let to_spawn = spawn_queue.floor() as u64;
                        spawn_queue -= to_spawn as f64;
                        for _ in 0..to_spawn {
                            if active_vus_thread.load(std::sync::atomic::Ordering::SeqCst) < max_vus as i64 {
                                let mut requests_guard = requests.lock().await;
                                let request = Arc::from(requests_guard.generate_request());
                                start_vu(backend.clone(), request.clone(), responses_tx.clone(), end_tx.clone(), stop_sender.clone(), None).await;
                                active_vus_thread.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            } else {
                                warn!("Max VUs reached, skipping request");
                                break;
                            }
                        }
                        interval.tick().await;
                    }
                    // signal that the VU work is done
                    info!("Duration reached, waiting for all VUs to finish...");
                    let _ = responses_tx.send(TextGenerationAggregatedResponse::new_as_ended());
                }=>{}
            }
        });
        while end_rx.recv().await.is_some() {
            active_vus.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            // wait for all VUs to finish
            if start.elapsed() > self.config.duration
                && active_vus.load(std::sync::atomic::Ordering::SeqCst) == 0
            {
                break;
            }
        }
        // wait for the VU thread to finish
        vu_thread.await.unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::requests::DummyTextRequestGenerator;
    use std::sync::atomic::AtomicUsize;

    #[derive(Clone)]
    struct HangingBackend {
        in_flight: Arc<AtomicUsize>,
    }

    struct DropGuard(Arc<AtomicUsize>);
    impl Drop for DropGuard {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }

    #[async_trait]
    impl TextGenerationBackend for HangingBackend {
        async fn generate(
            &self,
            _request: Arc<TextGenerationRequest>,
            _tx: Sender<TextGenerationAggregatedResponse>,
        ) {
            self.in_flight.fetch_add(1, Ordering::SeqCst);
            let _guard = DropGuard(self.in_flight.clone());
            std::future::pending::<()>().await;
        }
    }

    #[tokio::test]
    async fn constant_vus_aborts_inflight_requests_on_timeout() {
        let in_flight = Arc::new(AtomicUsize::new(0));
        let backend = Box::new(HangingBackend {
            in_flight: in_flight.clone(),
        });
        let executor = ConstantVUsExecutor::new(backend, 2, Duration::from_millis(50));
        let requests = Arc::new(Mutex::new(DummyTextRequestGenerator::new()));
        let (responses_tx, _responses_rx) = tokio::sync::mpsc::unbounded_channel();
        let (stop_tx, _) = broadcast::channel(1);

        executor.run(requests, responses_tx, stop_tx).await;

        assert_eq!(
            in_flight.load(Ordering::SeqCst),
            0,
            "in-flight backend tasks must be aborted and dropped before executor returns"
        );
    }

    #[tokio::test]
    async fn constant_vus_counts_partial_streamed_tokens_on_timeout() {
        use crate::requests::DummyTextGenerationBackend;
        use crate::results::BenchmarkResults;
        use crate::scheduler::ExecutorType;

        // 10 tokens per 200ms (50 tok/s); run for 300ms so req #1 completes
        // (10 tokens) and req #2 streams ~4-5 partial tokens before timeout.
        let backend = Box::new(DummyTextGenerationBackend::new(Duration::from_millis(200)));
        let cfg = ExecutorConfig {
            max_vus: 1,
            duration: Duration::from_millis(300),
            rate: None,
        };
        let executor = ConstantVUsExecutor::new(backend, cfg.max_vus, cfg.duration);
        let requests = Arc::new(Mutex::new(DummyTextRequestGenerator::new()));
        let (responses_tx, mut responses_rx) = tokio::sync::mpsc::unbounded_channel();
        let (stop_tx, _) = broadcast::channel(1);

        executor.run(requests, responses_tx, stop_tx).await;

        let mut results = BenchmarkResults::new("throughput".to_string(), ExecutorType::ConstantVUs, cfg);
        while let Some(resp) = responses_rx.recv().await {
            if resp.ended {
                break;
            }
            results.add_response(resp);
        }

        assert_eq!(results.successful_requests(), 1);
        assert_eq!(results.decoded_tokens_avg().unwrap(), 10.0);
        assert!(
            results.total_tokens() > 10,
            "expected partial streamed tokens > 10, got {}",
            results.total_tokens()
        );
        let tput = results.token_throughput_secs().unwrap();
        assert!(
            tput > 35.0 && tput < 65.0,
            "expected steady throughput ~50 tok/s with partial tokens, got {tput:.2}"
        );
    }
}
