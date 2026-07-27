# Data, models and licensing

## Public repository

BIMER source code, configuration, tests, documentation and the aggregate result
tables under `results/` are released under Apache-2.0 unless a file states
otherwise.

Apache-2.0 does **not** grant permission to redistribute third-party datasets,
pretrained models, derived restricted records or private videos.

## MELD

MELD is obtained from its official project and must be used under the terms
specified by its authors. This repository does not redistribute MELD media,
utterance-level annotations, extracted features or reconstructed clips.

Official project: <https://github.com/declare-lab/MELD>

## EmotionTalk

EmotionTalk requires a Hugging Face account and acceptance of its academic-use
terms. This repository does not redistribute its media, annotations,
utterance-level predictions or derived feature cache.

Official dataset page: <https://huggingface.co/datasets/BAAI/Emotiontalk>

## Pretrained assets

The private offline defense package uses separately downloaded XLM-R,
Wav2Vec2 XLS-R, faster-whisper, torchvision R3D-18 and YuNet assets. Each asset
retains its upstream license and model-card conditions. The deployment manifest
pins identifiers, revisions and hashes but the files are not included here.

## External evaluation videos

Only videos recorded by the project author or accompanied by explicit
permission may enter the private external-evaluation set. They must not be
committed to the public repository. Public reports contain aggregate statistics
and de-identified failure descriptions only.

### Chinese face-video defense sample

The private defense package includes a 50-second excerpt from **“Ma Jian VOA
interview 20181112”**, authored by Voice of America Chinese. The
[Wikimedia Commons file page](https://commons.wikimedia.org/wiki/File:Ma_Jian_VOA_interview_20181112.webm)
identifies the VOA-only work as public domain in the United States.

- Original file: 68.409-second WebM, SHA-256
  `ba891b1d4c350e7310efba8e03f5fbda86e46d02bc261da31cff89254d99a17d`.
- Defense excerpt: seconds 5.000–55.000, transcoded to H.264/AAC without
  changing the spoken content, SHA-256
  `d9a8831cfa978bd9d84f4bfb52d4472effa4a0e25678d2ee1626edd7053cb78c`.
- Use: non-commercial academic system acceptance and defense demonstration.
  The excerpt must not be presented as endorsement by the speaker or VOA.

## Research outputs

The public `results/` directory contains means, sample standard deviations,
paired cluster-bootstrap intervals and condition-level aggregate metrics. The
private evidence archive retains per-sample probabilities and predictions for
audit under the source datasets' restrictions.
