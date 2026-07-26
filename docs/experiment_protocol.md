# v2 实验协议

## 固定口径

- MELD：官方 `train/dev/test`。
- EmotionTalk：官方 `train/validation/test`。
- 内部标签顺序：neutral、joy、sadness、anger、surprise、fear、disgust。
- 主指标：weighted-F1；同时报告 macro-F1、accuracy、每类F1、混淆矩阵。
- 所有调参只使用验证集，测试集仅在模型固定后运行。
- 旧代码与103份旧结果只作为 `v1-preaudit` 保存，不与 v2 汇总。
- 所有学习率和结构筛选必须传入 `--skip-test`；该模式不加载测试特征，也不生成测试预测。
- 随机种子：42、123、2026；报告均值和样本标准差（`ddof=1`）。
- 模型差值置信区间按完整 `context_id` 做配对 cluster bootstrap，禁止平均多个独立CI端点。

## 上下文与训练视图

- MELD 的 `context_id=dialogue_id`。
- EmotionTalk 将演员轨道 ID（如 `G00006_58_07`）映射为场景 ID `G00006_58`；修正后共有742段完整对话。
- 训练窗口最多32句、重叠8句。联合训练按中英文窗口1:1采样，并在每轮用 `seed+epoch` 重新洗牌。
- 训练集10%对话按数据集和主导情感分层抽取，整段生成10 dB音频、50%视频丢帧与Whisper文本视图。
- 所有模型使用同一训练集逐维 `InputNormalizer`；统计量随检查点保存。

## 模型矩阵

正式矩阵运行：多数类、文本、语音、视频、Early MLP、Early BiGRU、无门控上下文模型、原LAGF与质量感知LAGF。质量模型消融包括去语言、去门控、去上下文、去质量输入、去扰动训练和去模态屏蔽。

`--training-scope` 定义：

- `joint`：在对话上下文窗口层面对中英文做1:1采样，以两验证集 weighted-F1 平均值选模。
- `meld`：仅英文训练，只用MELD验证集选模，同时测试中英文，用于英→中迁移。
- `emotiontalk`：仅中文训练，只用EmotionTalk验证集选模，同时测试中英文，用于中→英迁移。

单语训练时，训练和验证数据严格限定为源语言。只有结构与超参冻结后的正式运行才评估两个测试集；跨语言实验必须传入联合 manifest 与特征缓存。

验证筛选示例：

```bash
bimer train ... --model audio --learning-rate 3e-4 --skip-test
python scripts/run_v2_experiments.py --stage fusion-screen --device cuda
```

所有长任务写入 `.done.json`/`.failed.json` 状态，可原命令续跑。AutoDL 使用 `scripts/run_v2_autodl.sh`；仅在成功打包并生成 SHA-256 后写入 `DOWNLOAD_READY`，设置 `AUTODL_AUTO_SHUTDOWN=1` 可在成功或失败后自动关机。

## 缺失模态

```bash
bimer evaluate ... --missing text
bimer evaluate ... --missing audio
bimer evaluate ... --missing vision
bimer evaluate ... --missing text --missing audio
bimer evaluate ... --missing text --missing vision
bimer evaluate ... --missing audio --missing vision
```

被屏蔽模态的特征置零且 mask 置假，不改变其他输入。可重复传入 `--missing` 以同时屏蔽两种模态；不允许屏蔽全部三种模态。

## 音频噪声

分别生成10 dB和20 dB缓存，噪声在 XLS-R 编码前加入：

```bash
bimer extract-features ... --features artifacts/features/snr10 --audio-snr 10
bimer extract-features ... --features artifacts/features/snr20 --audio-snr 20
bimer evaluate ... --features artifacts/features/snr10 --output artifacts/experiments/robustness/snr10.json
```

## 视频丢帧

```bash
bimer extract-features ... --features artifacts/features/drop25 --frame-drop 0.25
bimer extract-features ... --features artifacts/features/drop50 --frame-drop 0.50
```

随机种子固定为42，被丢弃帧在 R3D-18 编码前置零。

## Whisper文本对照

保持标签、样本ID和官方划分不变，只替换文本：

```bash
bimer asr-manifest \
  --manifest data/processed/all.jsonl \
  --output data/processed/all-asr.jsonl

bimer extract-features \
  --manifest data/processed/all-asr.jsonl \
  --features artifacts/features/asr \
  --yunet-model artifacts/models/face_detection_yunet_2023mar.onnx
```

用同一检查点评估 `standard` 与 `asr` 两个特征根目录，比较文字识别误差带来的性能变化。

## 统计复核

正式运行保存逐样本 `sample_ids、context_ids、truth、prediction、probabilities、gates、modality_quality、modality_available`。模型差值使用：

```bash
python scripts/compare_v2_predictions.py \
  --baseline baseline.npz --candidate candidate.npz \
  --output comparison.json
```

## 完成判定

- 语音验证集至少预测4类，且高于多数类至少3个百分点。
- 主模型双语验证平均 weighted-F1 高于修正后的 Early MLP。
- 相比无门控上下文至少提高0.5个百分点，或干净集下降不超过0.5个百分点且明显改善视频损坏鲁棒性。
- 50%视频丢帧损失不得明显大于完全缺失视觉的损失。
- 若质量门控不满足标准，系统采用无门控上下文模型，并将门控作为完整负结果报告，不改测试口径。
