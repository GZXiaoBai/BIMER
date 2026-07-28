# ADR 0002: Protocol-owned experiment lifecycle

Status: Accepted

## Context

V2, V3, and V4 scripts implemented overlapping rules for screening, formal
training, resume markers, failure evidence, and one-time test access.

## Decision

`ProtocolSpec` validates stage, seed, and test-access policy.
`ExperimentProtocolRunner` atomically records running, failed, and completed
states and resumes only a verified completed result. Guarded exploratory tests
require a frozen selection and create a non-overwritable `TEST_EVALUATED`
marker.

## Consequences

- Screening and formal stages fail before loading data when test access is
  requested.
- Interrupted jobs leave machine-readable evidence and may be resumed.
- V2/V3/V4 result files remain unchanged; lifecycle state is additive.
- V5 uses the same lifecycle rather than introducing another shell-only policy.
