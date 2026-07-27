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

The private defense package includes a 50-second excerpt from **“VOA专访中国
玻璃大王曹德旺”**, authored by Voice of America Chinese. The
[Wikimedia Commons file page](https://commons.wikimedia.org/wiki/File:VOA专访中国玻璃大王曹德旺.webm)
identifies the VOA-only work as public domain in the United States. The page
also notes that the imported upload has not received an additional Commons
administrator review; the defense record therefore preserves the exact source
page and hashes rather than making a broader copyright claim.

- Original file: 625.158-second WebM, SHA-256
  `c823deed1ba7645075067db4651c4b6484d9032f189c98d4b1a933cad30fd4b3`.
- Defense excerpt: seconds 70.000–120.000, transcoded to H.264/AAC without
  changing the spoken content, SHA-256
  `ff67739bda750d71d4763f2d274ac65e9888f7598722b063c348a4cb01d59af1`.
- Use: non-commercial academic system acceptance and defense demonstration.
  The excerpt must not be presented as endorsement by the speaker or VOA.

## Research outputs

The public `results/` directory contains means, sample standard deviations,
paired cluster-bootstrap intervals and condition-level aggregate metrics. The
private evidence archive retains per-sample probabilities and predictions for
audit under the source datasets' restrictions.
