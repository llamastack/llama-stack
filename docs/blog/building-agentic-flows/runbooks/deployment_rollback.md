# Deployment Rollback Runbook

**Owner:** Platform SRE · **Last updated:** 2025-02-28

## When to Use This Runbook

Use this procedure when a production deployment causes user-facing issues and
you need to revert to the previous known-good state.  Common triggers:

- Error rate spikes above 1% on any service (PagerDuty alert `svc-error-rate`)
- Latency p99 exceeds SLO for more than 5 minutes
- Canary deployment fails automated smoke tests

## Prerequisites

- `kubectl` configured for the target cluster (`shopwave-prod-us-east-1` or
  `shopwave-prod-eu-west-1`)
- Membership in the `sre-oncall` or `platform-eng` RBAC group
- Access to the `#deploy` Slack channel for coordination

## Rollback Procedure

### Step 1 — Confirm the Bad Deployment

```bash
# See current and previous revisions
kubectl rollout history deployment/<service-name> -n <namespace>
```

Verify that the most recent revision matches the deployment you want to revert.

### Step 2 — Revert the Kubernetes Deployment

Revert the Kubernetes deployment to the previous revision using `kubectl rollout
undo`:

```bash
kubectl rollout undo deployment/<service-name> -n <namespace>
```

To roll back to a specific revision (not just the previous one):

```bash
kubectl rollout undo deployment/<service-name> -n <namespace> --to-revision=<N>
```

### Step 3 — Verify the Rollback

```bash
# Watch rollout progress
kubectl rollout status deployment/<service-name> -n <namespace>

# Confirm the running image
kubectl get deployment/<service-name> -n <namespace> \
  -o jsonpath='{.spec.template.spec.containers[0].image}'
```

Check the service's `/readyz` endpoint returns 200 and error rates are
returning to baseline in Grafana.

### Step 4 — Notify the Team

Post in `#deploy`:

> 🚨 Rolled back `<service-name>` in `<region>` from revision N to revision N-1.
> Reason: <brief description>.  Investigating root cause.

Tag the on-call engineer and link to the relevant PagerDuty incident.

## Database Migrations

If the bad deployment included a database migration, rolling back the
Kubernetes deployment alone is **not sufficient**.  You must also revert the
migration:

1. Check `schema_migrations` for the most recent migration version.
2. Run the down migration: `python manage.py migrate <app> <previous_version>`
3. Verify schema state matches the reverted application code.

⚠️ **Irreversible migrations** (e.g., column drops) cannot be rolled back this
way.  If the migration was destructive, escalate to the database team
immediately.

## Post-Rollback

- File a postmortem if the incident lasted more than 15 minutes or affected
  more than 0.1% of requests.
- Update the deployment ticket in Jira with the rollback details.
- Schedule a blameless review within 48 hours.

## Contacts

| Role              | Slack handle      |
|-------------------|-------------------|
| SRE on-call       | `@sre-oncall`     |
| Platform lead     | `@maria.chen`     |
| Database team     | `@db-oncall`      |
| Incident commander| Rotating — check PagerDuty |
