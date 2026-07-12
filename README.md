# BIMER：中英文多模态对话情感识别

毕业设计题目：**《基于语言感知跨模态融合的中英文对话情感识别研究与系统实现》**。

BIMER 联合文本、语音和视频特征，对中英文对话进行七分类情感识别，并输出逐句概率、模态门控权重和整段情绪时间线。项目支持 MELD、EmotionTalk、单模态/融合基线、消融实验、缺失模态、音频噪声、视频丢帧、跨语言测试及 Gradio 演示。

> 本仓库不包含受许可约束的数据集或训练权重。EmotionTalk 下载前必须在 Hugging Face 接受其学术使用条款。模型结果仅用于研究，不构成心理或医疗判断。

## 当前实现

- 统一七类标签：`neutral, joy, sadness, anger, surprise, fear, disgust`。
- MELD 与 EmotionTalk 官方划分适配及跨划分泄漏检查。
- XLM-RoBERTa、Wav2Vec2 XLS-R、YuNet + R3D-18 冻结特征入口。
- 1024条分片的无 pickle `.npz` 特征缓存。
- 多数类、三种单模态、Early Fusion MLP、Early Fusion BiGRU 基线。
- `LanguageAwareGatedFusion`：语言嵌入、可靠性门控、跨模态 Transformer、BiGRU 上下文和模态随机屏蔽。
- 双数据集 weighted-F1 选模、早停、三随机种子、95% bootstrap 区间。
- Whisper 自动切句、人工修订转写、逐句推理、时间线、JSON/CSV导出。
- 缺失模态、10/20 dB噪声、25%/50%丢帧和中英跨语言实验入口。

## 环境安装

建议 Python 3.11：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,inference]'
```

严格复现实验可使用直接依赖锁：

```bash
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
```

系统还需要 `ffmpeg` 和 `ffprobe`。macOS 可执行 `brew install ffmpeg`。

下载 YuNet：

```bash
./scripts/download_yunet.sh
```

## 数据流程

### 1. 生成统一清单

MELD：

```bash
bimer prepare-meld \
  --train-csv data/raw/meld/train_sent_emo.csv \
  --dev-csv data/raw/meld/dev_sent_emo.csv \
  --test-csv data/raw/meld/test_sent_emo.csv \
  --train-media data/raw/meld/train \
  --dev-media data/raw/meld/dev \
  --test-media data/raw/meld/test \
  --output data/processed/meld.jsonl
```

EmotionTalk：

```bash
bimer prepare-emotiontalk-official \
  --labels-csv data/raw/emotiontalk-official/EmotionTalk/dataset/mm-process/mm.csv \
  --transcriptions-csv data/raw/emotiontalk-official/EmotionTalk/dataset/mm-process/transcription.csv \
  --media-root data/raw/emotiontalk \
  --output data/processed/emotiontalk.jsonl
```

上述命令对应 EmotionTalk 当前官方发布格式，会按官方演员组划分训练、验证和测试集。如果使用自行转换的三个 JSON 文件，仍可使用兼容入口：

```bash
bimer prepare-emotiontalk \
  --train-json data/raw/emotiontalk/train.json \
  --validation-json data/raw/emotiontalk/validation.json \
  --test-json data/raw/emotiontalk/test.json \
  --media-root data/raw/emotiontalk \
  --output data/processed/emotiontalk.jsonl
```

合并两个 JSONL 文件后严格校验：

```bash
cat data/processed/meld.jsonl data/processed/emotiontalk.jsonl > data/processed/all.jsonl
bimer validate --manifest data/processed/all.jsonl --official-counts
```

### 2. 提取冻结特征

```bash
bimer extract-features \
  --manifest data/processed/all.jsonl \
  --features artifacts/features/standard \
  --yunet-model artifacts/models/face_detection_yunet_2023mar.onnx \
  --mode parallel \
  --text-audio-device cuda:0 \
  --vision-device cuda:1
```

双 GPU 模式会独立缓存三个模态并按样本 ID 合并；已有合法分片可断点续跑。单 GPU 环境可使用 `--mode serial --device cuda`。Kaggle 的分阶段运行、监控和持久化方法见 [docs/kaggle.md](docs/kaggle.md)。

EmotionTalk train 的15,413条样本使用全局 shard 范围跨 Kaggle Session 提取。`--start-shard` 与 `--end-shard` 保留全局文件编号，`bimer verify-features` 在每段 Quick Save 前校验ID、维度、掩码和有限值；八段区间及恢复命令见 [docs/kaggle.md](docs/kaggle.md)。

### 3. 训练与评估

```bash
bimer train \
  --manifest data/processed/all.jsonl \
  --features artifacts/features/standard \
  --output artifacts/experiments \
  --model lagf --training-scope joint --seed 42 --device cuda
```

缺失视频模态评估：

```bash
bimer evaluate \
  --manifest data/processed/all.jsonl \
  --features artifacts/features/standard \
  --checkpoint artifacts/experiments/lagf/joint/seed-42/best.pt \
  --missing vision \
  --output artifacts/experiments/robustness/missing-vision.json
```

全模型、三随机种子、消融和跨语言矩阵：

```bash
make full-experiment \
  MANIFEST=data/processed/all.jsonl \
  FEATURES=artifacts/features/standard \
  OUTPUT=artifacts/experiments \
  DEVICE=cuda
```

完整实验口径见 [docs/experiment_protocol.md](docs/experiment_protocol.md)。

## 演示系统

```bash
bimer serve \
  --checkpoint artifacts/experiments/lagf/joint/seed-42/best.pt \
  --yunet-model artifacts/models/face_detection_yunet_2023mar.onnx \
  --device auto
```

命令行分析：

```bash
bimer analyze \
  --video demo/dialogue.mp4 \
  --checkpoint artifacts/experiments/lagf/joint/seed-42/best.pt \
  --yunet-model artifacts/models/face_detection_yunet_2023mar.onnx \
  --language auto \
  --output artifacts/exports/demo
```

## 测试

```bash
python -m pytest
```

测试使用合成特征，不需要下载模型和数据集。

## 目录

```text
src/bimer/          数据、模型、训练、评估、推理和界面
tests/              单元及端到端合成烟雾测试
scripts/            一键实验与模型下载脚本
docs/               Kaggle、实验协议和论文结构
data/               本地清单；原始数据默认不纳入Git
artifacts/          特征、检查点、实验结果和导出文件
```
