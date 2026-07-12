# 实验协议

## 固定口径

- MELD：官方 `train/dev/test`。
- EmotionTalk：官方 `train/validation/test`。
- 内部标签顺序：neutral、joy、sadness、anger、surprise、fear、disgust。
- 主指标：weighted-F1；同时报告 macro-F1、accuracy、每类F1、混淆矩阵。
- 所有调参只使用验证集，测试集仅在模型固定后运行。
- 随机种子：42、123、2026；报告均值、样本标准差及单次测试集 bootstrap 95%区间。

## 模型矩阵

一键脚本运行：多数类、文本、语音、视频、Early MLP、Early BiGRU、完整LAGF；随后运行去语言嵌入、去门控、去上下文、去模态屏蔽四组消融。

`--training-scope` 定义：

- `joint`：中英文1:1采样，以两验证集 weighted-F1 平均值选模。
- `meld`：仅英文训练，只用MELD验证集选模，同时测试中英文，用于英→中迁移。
- `emotiontalk`：仅中文训练，只用EmotionTalk验证集选模，同时测试中英文，用于中→英迁移。

## 缺失模态

```bash
bimer evaluate ... --missing text
bimer evaluate ... --missing audio
bimer evaluate ... --missing vision
```

被屏蔽模态的特征置零且 mask 置假，不改变其他输入。

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

## 完成判定

完整模型应争取超过 Early MLP 的双语平均 weighted-F1。若未超过，不改变测试口径；报告门控分布、各类错误、语言域差异和消融结果，并将负面结果写入论文。

