# Ethics and responsible use

BIMER is an academic prototype for studying multimodal dialogue emotion
recognition. It is not a psychological, medical, employment, educational,
credit, policing or surveillance assessment tool.

## Limitations

- Seven discrete labels cannot represent the full ambiguity and cultural
  context of human emotion.
- Performance differs substantially by class, language, recording quality and
  modality availability.
- MELD `fear` and `disgust` remain particularly difficult.
- Automatic transcription errors can materially reduce performance.
- A visible face, loud voice or high confidence score does not establish a
  person's internal mental state.
- The quality-aware gate is not universally better under every corruption.

## Human impact

Do not use BIMER to make consequential decisions about individuals. Obtain
informed permission before analyzing private recordings. Avoid retaining raw
video or speech longer than required. Where results are shown, display
uncertainty and quality warnings rather than presenting predictions as facts.

## Dataset considerations

MELD and EmotionTalk have their own collection context, demographic coverage
and licensing constraints. Dataset performance does not imply equal performance
for all speakers, accents, ages or cultures. The external video evaluation is a
small engineering validation, not a population-level fairness audit.

## Reporting

Published claims are intentionally limited to effects supported by the fixed
protocol. Language embeddings were not supported by ablation. V3 ranking
supervision failed its validation threshold and is reported as a negative
result. The project does not claim state-of-the-art performance.
