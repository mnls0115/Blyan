# Pipeline Operations Guide

This document summarizes how to operate the experimental pipeline-parallel training subsystem in production.

## Environment Variables

- `BLYAN_PIPELINE_TRANSPORT` = `http` | `grpc` (default `http`)
- `BLYAN_PIPELINE_TIMEOUT_S` = per-RPC timeout seconds (float)
- `BLYAN_PIPELINE_MAX_RETRIES` = max retry attempts per request
- `BLYAN_PIPELINE_BACKOFF_BASE_S` = base backoff (exponential: base * 2^attempt)
- `BLYAN_PIPELINE_BREAKER_THRESHOLD` = failures before circuit opens
- `BLYAN_PIPELINE_BREAKER_RESET_S` = seconds before half-open
- `BLYAN_PIPELINE_CHUNK_BYTES` = activation/gradient chunk size in bytes (default 1MiB)
- `BLYAN_PIPELINE_COMPRESSION` = `none` | `gzip` (default `none`)
- `BLYAN_PIPELINE_MAX_BUFFER_MB` = server buffer watermark; on exceed, HTTP returns 429 to apply backpressure
- `BLYAN_ROUND_MAX_FAILURES` = failures before plan fallback
- `BLYAN_PIPELINE_ROUND_INTERVAL` = seconds between rounds
- `TRAINING_ENABLE` = `1` to run actual micro-step training
- `USE_DDP`, `USE_ZERO1`, `TRAINING_*` = standard training knobs
- TLS/mTLS: `BLYAN_TLS_CERT`, `BLYAN_TLS_CLIENT_CERT`, `BLYAN_TLS_CLIENT_KEY`

## Transport Switching and Benchmarking

1. Set `BLYAN_PIPELINE_TRANSPORT=http|grpc` on both coordinator and worker nodes.
2. For HTTP+TLS, set `BLYAN_TLS_CERT` (and optionally client cert+key) on the coordinator.
3. Collect Prometheus metrics at `/metrics` to compare latency and error rates.

## Plan Lifecycle via CLI

```
python scripts/plan_cli.py snapshot --epoch E1 --round round_0 --stages n1:0:11 n2:12:23 --zero1
python scripts/plan_cli.py validate --file data/partition_plans/draft_plan.json
python scripts/plan_cli.py promote --file data/partition_plans/draft_plan.json
```

## Single-Node Fallback Alerts

If the system collapses to a single stage (fallback), it records a security event `throughput_degraded_single_node` and sets metrics:
- `blyan_pipeline_fallback_mode_active` = 1
- `blyan_pipeline_current_stage_count` = 1

## Node Profiles and Heartbeats

- Nodes register in `ExpertNodeRegistry` with `device_profile` that includes `vram_gb` and `tflops_est`.
- Reputation and health are monitored by `NodeReputationManager`; staleness threshold is `BLYAN_HEARTBEAT_TIMEOUT`.
- Exported metrics include `device_profile_staleness_seconds{node="..."}`.

## TLS Certificate Automation

Use `scripts/ssl_manager.py` to obtain and renew certificates with certbot and nginx. To distribute certs to workers, use `scripts/distribute_tls_to_workers.py`.

Setup automated cron jobs: `python scripts/setup_pipeline_cron.py`

## Failure Injection (for recovery testing)

- Node down: stop the worker process or `kill -9 <pid>`; the coordinator should reallocate or collapse to single-node fallback.
- Network timeout: Linux `tc` netem example
  - Add 200ms delay: `sudo tc qdisc add dev eth0 root netem delay 200ms`
  - Remove: `sudo tc qdisc del dev eth0 root`
- Drop traffic temporarily: `sudo iptables -A INPUT -p tcp --dport 8001 -j DROP` (remove with `-D`).

