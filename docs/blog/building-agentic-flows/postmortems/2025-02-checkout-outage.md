# Postmortem: 2025-02 Checkout Outage

**Date:** 2025-02-12 · **Duration:** 47 minutes · **Severity:** SEV-1
**Author:** Jordan Park (SRE) · **Status:** Action items complete

## Summary

On February 12, 2025 at 14:23 UTC, the checkout service began returning HTTP
503 errors to approximately 35% of customers attempting to complete purchases.
The incident lasted 47 minutes until a configuration fix was deployed at 15:10
UTC.  Estimated revenue impact: ~$280K in lost or delayed orders.

## Root Cause

**Connection pool exhaustion in the payments service due to missing timeout
configuration.**

The payments service (`payments-svc`) connects to the Stripe API through an
internal connection pool (HikariCP, max pool size = 20).  A Stripe API
degradation at 14:20 UTC caused response times to increase from ~200ms to
~8 seconds.  Because the pool had **no connection timeout configured** (default:
infinite wait), threads waiting for a pool connection blocked indefinitely.

Within 3 minutes, all 20 connections were occupied by slow Stripe calls, and
new checkout requests queued behind them.  The queue grew until the service
hit its thread limit (200 threads), at which point Kubernetes health checks
started failing and pods entered CrashLoopBackOff.

The checkout service depends on payments-svc synchronously — when payments
became unavailable, checkout returned 503.

## Timeline

| Time (UTC) | Event |
|------------|-------|
| 14:20 | Stripe API latency increases (p99 200ms → 8s) |
| 14:23 | Payments-svc connection pool saturates; checkout errors begin |
| 14:25 | PagerDuty fires `checkout-error-rate` alert |
| 14:28 | On-call SRE acknowledges; begins investigation |
| 14:35 | Root cause identified: payments-svc thread dump shows all threads blocked on HikariCP `getConnection()` |
| 14:42 | Attempted fix: increase pool size to 50 — did not help (Stripe still slow) |
| 14:55 | Correct fix identified: add `connectionTimeout=5000` to HikariCP config |
| 15:02 | Config change deployed via ConfigMap update + rolling restart |
| 15:10 | All pods healthy; error rate returns to baseline |
| 15:15 | Incident resolved; monitoring confirmed stable |

## Contributing Factors

1. **No connection timeout** — HikariCP defaults to 30 seconds, but our config
   explicitly set it to `0` (infinite) based on a years-old tuning guide that
   prioritized throughput over resilience.
2. **No circuit breaker** — The payments service had no circuit breaker on the
   Stripe integration, so it kept sending requests to a degraded upstream.
3. **Synchronous dependency** — Checkout blocks on payments; there is no async
   fallback or queue-based decoupling.
4. **Monitoring gap** — We had alerts on checkout error rate but not on
   payments-svc connection pool utilization.

## Action Items

| # | Action | Owner | Status |
|---|--------|-------|--------|
| 1 | Set `connectionTimeout=5000` and `maximumPoolSize=30` on all HikariCP pools | Platform Team | ✅ Done |
| 2 | Add circuit breaker (Resilience4j) to Stripe integration in payments-svc | Payments Team | ✅ Done |
| 3 | Add Grafana alert on HikariCP active connections > 80% of pool size | SRE | ✅ Done |
| 4 | Evaluate async checkout flow (publish to SQS, process payment async) | Checkout Team | 🔄 In progress (Q2 target) |
| 5 | Audit all services for missing timeout configurations | Platform Team | ✅ Done |

## Lessons Learned

- **Timeouts are not optional.** Every connection pool, HTTP client, and RPC
  call must have an explicit timeout.  "Infinite" is never the right default
  for production.
- **Pool exhaustion cascades fast.** A 20-connection pool with no timeout can
  go from healthy to fully blocked in under 3 minutes during an upstream
  degradation.
- **Monitor pool internals, not just request outcomes.** We caught the error
  rate spike quickly, but could have caught the pool saturation 2 minutes
  earlier if we'd been monitoring HikariCP metrics.
