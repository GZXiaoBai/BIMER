# BIMER：中英文多模态对话情感识别

毕业设计题目：**《基于质量感知与对话上下文建模的中英文多模态情感识别研究与系统实现》**。

BIMER 联合文本、语音和视频特征，对中英文对话进行七分类情感识别，并输出逐句概率、模态门控权重和整段情绪时间线。项目支持 MELD、EmotionTalk、单模态/融合基线、消融实验、缺失模态、音频噪声、视频丢帧、跨语言测试及 Gradio 演示。

正式系统固定采用 V2 `quality_lagf` seed 42。V3 的类别损失与配对门控排序改进均未通过预先声明的验证集门槛，作为负结果保留。V4 轻量文本适配在验证集上取得明显提升，但没有通过全部少数类稳定性门槛，因此没有访问官方测试集，也没有替换 V2。完整数字、统计边界与可复核聚合文件见 [RESULTS.md](RESULTS.md)。

V5 仅针对已确认的 Whisper 转写退化进行事后探索，采用质量条件文本残差适配器与人工/Whisper 配对一致性损失。其验证门槛、一次性测试保护和10 GPU小时成本上限见 [docs/v5_protocol.md](docs/v5_protocol.md)；无论筛选结果如何，v1.1.0答辩系统仍部署V2。

> 本仓库不包含受许可约束的数据集或训练权重。EmotionTalk 下载前必须在 Hugging Face 接受其学术使用条款。模型结果仅用于研究，不构成心理或医疗判断。

## 研究结论

- V2 完整模型的双语平均 weighted-F1 为 **60.148% ± 1.124%**。
- 相同特征与测试口径下，相比 Early MLP 提高 **1.493 个百分点**；完整对话配对 cluster bootstrap 95% CI 为 **[0.669, 2.200]** 个百分点。
- 消融明确支持对话上下文与模态随机屏蔽。
- 质量机制在视频丢帧条件下具有针对性收益，但不应表述为对所有退化均有效。
- 语言嵌入没有得到消融支持；项目不宣称达到 SOTA，也不宣称超过原数据集论文的最佳单数据集结果。
- V2 置信度仅用官方验证集拟合温度：英文 ECE 从 6.548% 降至 3.923%，中文从 11.728% 降至 3.514%；系统据此启用分语言温度缩放和不确定提示。

项目限制、适用范围和伦理边界见 [MODEL_CARD.md](MODEL_CARD.md)、[DATA_AND_LICENSES.md](DATA_AND_LICENSES.md) 与 [ETHICS.md](ETHICS.md)。
最终 20 段外部视频测试的素材、授权、锁定和双人标注流程见 [docs/external_test_protocol.md](docs/external_test_protocol.md)。

模型结构图：[可编辑 SVG](diagram/bimer-architecture/bimer-model-architecture.svg) /
[2× PNG](diagram/bimer-architecture/bimer-model-architecture@2x.png)。

## 当前实现

- 统一七类标签：`neutral, joy, sadness, anger, surprise, fear, disgust`。
- MELD 与 EmotionTalk 官方划分适配及跨划分泄漏检查。
- XLM-RoBERTa、Wav2Vec2 XLS-R、YuNet + R3D-18 冻结特征入口。
- 1024条分片的无 pickle `.npz` 特征缓存。
- 多数类、三种单模态、Early Fusion MLP、Early Fusion BiGRU 基线。
- `QualityAwareLanguageGatedFusion`：连续质量输入、语言嵌入、跨模态 Transformer、BiGRU 上下文和严格单模态随机屏蔽。
- EmotionTalk 以完整场景 `context_id` 建模，官方 19,250 条语句组成 742 段对话，同时保留原 `sample_id` 读取旧特征。
- 训练集逐维归一化、逐轮确定性洗牌、最少15轮早停、类别分布/梯度/门控坍缩诊断。
- 双数据集 weighted-F1 选模、三随机种子样本标准差、完整对话配对 cluster bootstrap。
- Whisper 自动切句、32/8滑窗、人工修订转写、质量警告、时间线、JSON/CSV/PNG导出。
- 缺失模态、10/20 dB噪声、25%/50%丢帧和中英跨语言实验入口。

## 环境安装

建议 Python 3.11：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,inference]'
```

严格复现使用已提交的 Python 3.11 与 `uv.lock`：

```bash
python3.11 -m pip install uv
uv sync --extra dev --extra inference --frozen
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

先生成 v2 连续质量和三种真实损坏训练视图（需原始媒体、YuNet、GPU编码器）：

```bash
./scripts/prepare_v2_quality_views.sh
```

学习率或结构筛选必须使用 `--skip-test`：

```bash
bimer train \
  --manifest data/processed/all.jsonl \
  --features artifacts/features/standard \
  --output artifacts/experiments \
  --model quality_lagf --training-scope joint --seed 42 --device cuda \
  --skip-test
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

可续跑的 v2 学习率、正式三种子和消融矩阵：

```bash
python scripts/run_v2_experiments.py \
  --stage all \
  --quality-features artifacts/features/bilingual-v2-quality \
  --augmentation-manifest data/processed/v2/corruption-train-10pct.jsonl \
  --augmentation-features artifacts/features/v2-corruption-audio10 \
  --augmentation-manifest data/processed/v2/corruption-train-10pct.jsonl \
  --augmentation-features artifacts/features/v2-corruption-video50 \
  --augmentation-manifest data/processed/v2/corruption-train-10pct-asr.jsonl \
  --augmentation-features artifacts/features/v2-corruption-whisper \
  --device cuda
```

完整实验口径见 [docs/experiment_protocol.md](docs/experiment_protocol.md)。

## 演示系统

最终系统只通过唯一的部署清单选择权重、编码器版本和运行参数。答辩前先执行离线预检：

```bash
bimer doctor \
  --deployment configs/deployment-v2.json \
  --artifact-root . \
  --offline
```

预检通过后启动系统：

```bash
bimer serve \
  --deployment configs/deployment-v2.json \
  --artifact-root .
```

命令行分析：

```bash
bimer analyze \
  --deployment configs/deployment-v2.json \
  --artifact-root . \
  --video artifacts/demo/dialogue.mp4 \
  --language auto \
  --output artifacts/exports/demo
```

部署清单固定使用V2 `quality_lagf` seed 42。该种子是预声明的标准部署种子，不是根据正式测试集成绩挑选。受数据许可限制，公开仓库不提供检查点和编码器文件；私有答辩包需按照清单中的相对路径放置这些资产。

## 测试

```bash
uv sync --extra dev --extra inference --frozen
uv run python scripts/check_public_tree.py --root .
uv run pytest --cov=bimer --cov-fail-under=85
```

测试使用合成特征，不需要下载模型和数据集。

公开仓库不附带受限资产，因此演示命令必须先按
[configs/deployment-v2.json](configs/deployment-v2.json) 配置私有 `artifact_root`，并确保：

```bash
bimer doctor --deployment configs/deployment-v2.json --artifact-root . --offline
bimer serve --deployment configs/deployment-v2.json --artifact-root .
```

贡献方式见 [CONTRIBUTING.md](CONTRIBUTING.md)，引用格式见
[CITATION.cff](CITATION.cff)。代码采用 Apache-2.0；数据集、模型缓存、训练权重和私人样例不因本代码许可证获得再分发许可。

## 目录

```text
src/bimer/          数据、模型、训练、评估、推理和界面
tests/              单元及端到端合成烟雾测试
scripts/            一键实验与模型下载脚本
docs/               Kaggle、实验协议和论文结构
results/            可公开复核的聚合结果，不含逐样本记录
data/               本地清单；原始数据默认不纳入Git
artifacts/          特征、检查点、实验结果和导出文件
```

最终交付前请同时核对：

- [学校论文模板迁移清单](docs/school-template-mapping.md)
- [v1.1.0 Release Checklist](docs/releases/v1.1.0-checklist.md)
- [最终交付状态](docs/final_delivery_status.md)
