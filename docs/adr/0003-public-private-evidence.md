# ADR 0003: Separate public software from private evidence

Status: Accepted

## Context

EmotionTalk, MELD media, checkpoints, cached features, external videos, and
human annotations have redistribution or privacy constraints. Reproducibility
still requires an auditable link between the public release and private
defense package.

## Decision

The public repository contains code, protocol, aggregate results, and portable
hash manifests. Private evidence packages contain restricted assets and
per-sample records. Deployment manifests use paths relative to an
`artifact_root`; public documents identify private artifacts by SHA-256 only.

## Consequences

- A public clone can run tests but cannot claim to be a complete offline
  deployment without private assets.
- `bimer doctor --offline` fails before analysis when any required asset is
  absent or mismatched.
- Historical private packages are preserved as superseded archives; one
  explicit package is the current launch target.
- External annotations are never synthesized to fill a human evidence gap.
