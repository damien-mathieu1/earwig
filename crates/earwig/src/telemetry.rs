//! Logs (human text, or JSON lines for log collectors) and Prometheus metrics (`GET /metrics`).

use std::{sync::OnceLock, time::Instant};

use axum::{
    extract::{MatchedPath, Request},
    middleware::Next,
    response::Response,
};
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};
use tracing_subscriber::EnvFilter;

/// Logs to stderr; level from `RUST_LOG` (default `info`, ONNX Runtime's chatter at `warn`).
pub fn init_logging(json: bool) {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,ort=warn"));
    let fmt = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr);
    // a second init (tests) is a no-op
    let _ = if json {
        fmt.json().flatten_event(true).try_init()
    } else {
        fmt.try_init()
    };
}

/// The process-wide Prometheus recorder, installed on first use.
pub fn prometheus() -> &'static PrometheusHandle {
    static HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();
    HANDLE.get_or_init(|| {
        let recorder = PrometheusBuilder::new()
            .set_buckets_for_metric(
                Matcher::Suffix("_seconds".into()),
                &[
                    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
                    30.0,
                ],
            )
            .expect("non-empty buckets")
            .build_recorder();
        let handle = recorder.handle();
        // fails only if another recorder was installed first, then metrics go there
        let _ = metrics::set_global_recorder(recorder);
        handle
    })
}

/// Per request: a log line and the `earwig_http_*` metrics, labelled by route template (not
/// by raw path, which would explode the label cardinality).
pub async fn observe(req: Request, next: Next) -> Response {
    let route = req
        .extensions()
        .get::<MatchedPath>()
        .map_or_else(|| "unmatched".to_string(), |p| p.as_str().to_string());
    let method = req.method().to_string();
    let t0 = Instant::now();
    let res = next.run(req).await;
    let secs = t0.elapsed().as_secs_f64();
    let status = res.status().as_u16();
    metrics::counter!("earwig_http_requests_total",
        "route" => route.clone(), "method" => method.clone(), "status" => status.to_string())
    .increment(1);
    metrics::histogram!("earwig_http_request_duration_seconds", "route" => route.clone())
        .record(secs);
    // probes and scrapes every few seconds would drown the useful lines
    if route.starts_with("/api") {
        tracing::info!(%method, %route, status, ms = secs * 1e3, "request");
    } else {
        tracing::debug!(%method, %route, status, ms = secs * 1e3, "request");
    }
    res
}
