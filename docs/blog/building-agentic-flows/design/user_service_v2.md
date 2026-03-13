# User Service v2 — Design Document

**Author:** Platform Team · **Status:** Approved · **Last updated:** 2025-01-15

## Overview

User Service v2 is the central identity and profile service for the Shopwave
e-commerce platform.  It owns user registration, login, profile management,
and session lifecycle.  v2 replaces the monolithic `accounts` module that lived
inside the Rails checkout app.

## Authentication

All authentication flows go through the **auth gateway** (`gateway.shopwave.internal`).
The gateway issues **JWT tokens signed with RS256** (RSA 2048-bit keys rotated
quarterly).  Access tokens have a 15-minute TTL; refresh tokens last 30 days and
are stored in an HTTP-only secure cookie.

Token verification is handled by a shared middleware library (`@shopwave/jwt-verify`)
that fetches the public key set from the gateway's `/.well-known/jwks.json`
endpoint and caches it for 5 minutes.

### Token claims

| Claim   | Description                          |
|---------|--------------------------------------|
| `sub`   | User UUID                            |
| `email` | Verified email address               |
| `roles` | Array of role strings (`customer`, `admin`, `support`) |
| `org`   | Merchant organization ID (multi-tenant) |

## Data Model

User records live in a PostgreSQL 16 cluster (`users-primary.db.shopwave.internal`).
The schema is straightforward:

- `users` — core identity (uuid, email, hashed_password, created_at)
- `profiles` — display name, avatar URL, locale, timezone
- `sessions` — active refresh tokens with device fingerprint and IP
- `audit_log` — immutable append-only log of login, logout, and password-change events

We use row-level security (RLS) so each merchant organization can only see its
own users.  The `org` claim in the JWT maps directly to the RLS policy.

## API Surface

The service exposes a gRPC API internally and an OpenAPI REST gateway for the
storefront.  Key endpoints:

| Method | Path                          | Description              |
|--------|-------------------------------|--------------------------|
| POST   | `/v2/auth/register`           | Create account           |
| POST   | `/v2/auth/login`              | Issue tokens             |
| POST   | `/v2/auth/refresh`            | Rotate access token      |
| GET    | `/v2/users/{id}/profile`      | Read profile             |
| PATCH  | `/v2/users/{id}/profile`      | Update profile           |
| DELETE | `/v2/users/{id}`              | GDPR deletion request    |

Rate limits: 20 requests/second per IP on auth endpoints, 100 req/s on profile
reads.

## Deployment

User Service v2 runs as a Kubernetes Deployment in the `platform` namespace
(`us-east-1` and `eu-west-1` regions).  Each region has 3 replicas behind an
internal ALB.  The Docker image is built in CI and pushed to our private ECR
registry.

Health checks:
- **Liveness:** `/healthz` (checks process is up)
- **Readiness:** `/readyz` (checks DB connection pool + auth gateway reachability)

## Dependencies

- PostgreSQL 16 (RDS Multi-AZ)
- Redis 7 (ElastiCache) for session caching and rate limiting
- Auth Gateway (internal, runs in the same cluster)
- Kafka (`user-events` topic) for publishing registration and deletion events

## Open Questions

- Should we migrate to passkeys (WebAuthn) for passwordless login?  Currently
  scoped for Q3 2025.
- Connection pool sizing needs revisiting after the February checkout outage
  (see postmortem).
