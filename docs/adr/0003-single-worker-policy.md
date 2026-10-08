# ADR 0003: Single Worker Production Policy

- Status: Accepted
- Date: 2026-10-08

## Context

Current AI Core operational state is held in memory. This includes the inbound
token bucket and active stream count, provider health and circuit state,
telemetry aggregates, and provider concurrency semaphores.

Running multiple workers would create independent copies of that state. Effective
rate and concurrency limits would multiply, health decisions would diverge, and
telemetry would be fragmented.

## Decision

Current production deployment uses exactly one process and one Uvicorn worker.

Docker, Compose, Nginx, and operational documentation must preserve this policy.
Horizontal or multi-worker operation is not supported until shared-state
readiness is implemented and validated.

Future multi-worker operation requires a state abstraction and distributed
backend for every state domain that needs cross-worker consistency. In-memory
state remains supported for local and single-worker deployments.

## Consequences

### Positive

- Rate limits, circuits, telemetry, and concurrency controls have coherent state
- Current behavior is simple, deterministic, and testable
- No premature distributed-system dependency is required

### Trade-offs

- A single instance has finite vertical capacity
- Process restart resets operational state
- Horizontal scaling is deferred until distributed-state work is complete

Increasing `--workers` is an architecture change, not an operational tuning
shortcut.
