# BIMER domain context

This document fixes the vocabulary used by code, experiments, reports, and the
defense system. New code should preserve these meanings.

## Data identity

- `sample_id` identifies one dataset utterance and never changes when a feature,
  transcription, or corruption view changes.
- `dialogue_id` is the source dataset identifier. It is retained for provenance
  and must not be rewritten to repair context grouping.
- `context_id` identifies the complete conversation used by the contextual
  model. MELD uses the source dialogue; EmotionTalk joins both speaker tracks of
  the same scene.
- An utterance is the atomic labelled unit. A context is the ordered sequence of
  utterances. Training windows are derived views, not new samples.

## Modalities and quality

- The modality order is always text, audio, vision.
- Availability is a hard mask. An unavailable modality contributes neither
  features nor a learned gate.
- 模态质量 is a continuous four-value observation for each modality. A low
  quality available modality is not equivalent to a missing modality.
- Human and Whisper text keep the same `sample_id`; `text_source` and ASR
  confidence distinguish the view.

## Research evidence

- 确认性实验 means the frozen V2 protocol, official splits, three seeds, and
  predeclared comparisons. It supplies the thesis main result.
- 探索性实验 means V3, V4, V5, or later post-hoc work. It must never replace
  V2 silently or be described as a fresh unbiased confirmation.
- Screening and formal training never access official test data. A frozen
  selection may use the guarded exploratory test entry once.
- Failed predeclared criteria are retained as negative evidence; thresholds are
  not relaxed after results are observed.
- V5 `ASRConsistentTextAdapter` is an exploratory residual over frozen feature
  inputs. A V5 result never changes the identity of V2 confirmatory evidence.

## Deployment boundary

- `DeploymentManifest` is the sole identity of a deployable model and its
  offline assets.
- `RuntimeSession` owns an analyzer, encoder lifecycle, cache access, request
  context, verification, and shutdown.
- `analyze_dialogue()` remains the stable public Python API. CLI, Gradio, and
  acceptance scripts are adapters rather than alternate runtimes.
- V2 `quality_lagf` seed 42 is the deployment model until a later release
  explicitly changes the manifest.

## Public and private evidence

- Code, configuration, tests, documentation, and aggregate results may be
  public.
- Restricted dataset media, cached features, checkpoints, per-sample
  predictions, private videos, and human labels stay in the private evidence
  package.
- Public documents may record cryptographic hashes of private artifacts but not
  private paths or payloads.
