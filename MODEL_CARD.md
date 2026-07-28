# BIMER V2 model card

## Model identity

- System model: `v2_quality_lagf`
- Deployment checkpoint: V2 `quality_lagf`, seed 42
- Task: seven-class utterance emotion recognition in Chinese and English
- Labels: `neutral`, `joy`, `sadness`, `anger`, `surprise`, `fear`, `disgust`
- Context: up to 32 utterances with an overlap of 8
- Status: frozen V2 deployment model. V3 is not deployed. V4 is not deployed.
  Exploratory V5 is not deployed.

The public deployment manifest records the expected paths, hashes, encoder IDs,
revisions, calibration parameters and runtime policy. The checkpoint and
pretrained encoder files remain private because of size and redistribution
constraints.

## Architecture

Frozen XLM-R, Wav2Vec2 XLS-R and R3D-18 encoders produce text, audio and visual
features. The fusion model projects each modality to 256 dimensions, combines
modality, language and four-dimensional quality signals, applies masked quality
gating and a cross-modal Transformer, then uses a bidirectional GRU for dialogue
context before seven-class prediction.

Visual features use YuNet face detection. Fewer than four detected face frames
marks vision unavailable. Unavailable modalities are hard-masked; low-quality
available modalities receive continuous quality inputs.

## Training and evaluation

- MELD official splits: 9,989 / 1,109 / 2,610 utterances.
- EmotionTalk official splits: 15,413 / 1,908 / 1,929 utterances.
- Total audited records and cached features: 32,958.
- Official splits were preserved; utterances were not randomly redistributed.
- Formal experiments used seeds 42, 123 and 2026.
- Reported standard deviations use `ddof=1`.
- Pairwise intervals use 2,000 complete-dialogue paired cluster-bootstrap
  replicates.

The three-seed mean is the confirmatory research result. Seed 42 is used for
deployment because it was the predeclared standard seed, not because it was the
best official-test run.

## Results

| Dataset | weighted-F1 | macro-F1 | accuracy |
|---|---:|---:|---:|
| MELD | 58.620% ± 0.830% | 39.591% ± 0.488% | 59.106% ± 1.479% |
| EmotionTalk | 61.675% ± 1.423% | 54.830% ± 0.776% | 61.051% ± 1.497% |
| Bilingual average | **60.148% ± 1.124%** | 47.210% ± 0.630% | 60.078% ± 1.474% |

Compared with Early MLP under identical features and evaluation, bilingual
weighted-F1 improves by 1.493 percentage points (95% CI 0.669 to 2.200).
Context removal reduces it by 1.385 points, and removing modality dropout
reduces it by 0.749 points; both intervals exclude zero.

See [RESULTS.md](RESULTS.md) and `results/` for the complete public aggregate
evidence.

## Intended use

- academic reproduction of the fusion and context experiments;
- non-consequential classroom demonstrations;
- analysis of aggregate robustness under missing or corrupted modalities.

## Out-of-scope use

- medical or psychological diagnosis;
- monitoring, grading, hiring, policing or other consequential decisions;
- covert analysis of people without permission;
- claims about internal mental state or universal cross-cultural emotion.

## Known limitations

- The model is not state of the art on either individual dataset.
- Language embedding ablation did not support a benefit.
- Clean-set quality-gating gain over a no-gate contextual model is not
  statistically established.
- Video degradation shows a targeted benefit, but audio corruption can favor
  the no-gate baseline.
- Whisper text has the largest tested corruption loss.
- Rare MELD classes, especially `fear` and `disgust`, have low F1.

## V3 negative result

Balanced Softmax and Focal Loss failed the predeclared loss-screening criteria.
Gate-ranking candidates at λ = 0.05, 0.10 and 0.20 all failed the required
0.5-point mean corrupted-validation improvement. They did reduce corrupted
audio and visual gate weights, but this did not translate into sufficient
classification gain. Training stopped according to protocol and V2 remained
the deployment model.

## V4 and V5 exploratory boundary

V4 did not pass all predeclared validation criteria and never accessed the
official test set. V5 targets the observed human-to-Whisper transcription
degradation with a quality-conditioned residual adapter and paired
prediction-consistency loss. V5 remains post-hoc exploratory even if it passes
validation; v1.1.0 continues to deploy V2. See
[docs/v5_protocol.md](docs/v5_protocol.md).

## Runtime behavior

The system exposes raw and calibrated probabilities, uncertainty status,
modality availability, quality signals and stage runtimes. Whisper executes in
an isolated subprocess to avoid PyAV/OpenCV FFmpeg-library conflicts. The
runtime prefers Apple MPS for encoders and fusion and explicitly falls back to
CPU when required.

The deployment calibration was fitted only on the official validation splits.
English uses temperature 1.193 and uncertainty threshold 0.45; Chinese uses
temperature 1.391 and threshold 0.55. Validation ECE fell from 6.548% to 3.923%
for English and from 11.728% to 3.514% for Chinese, while NLL and Brier score
also improved. Temperature scaling preserves the predicted class, so it does
not alter accuracy or F1. Full aggregate values and reliability curves are in
[RESULTS.md](RESULTS.md).
