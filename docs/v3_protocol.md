# BIMER V3 实验与成品协议

V2 `quality_lagf` 是确认性主结果和安全回退版本。V3 是事后探索性改进，筛选阶段只访问 MELD dev 与 EmotionTalk validation；官方 test 只能在冻结选择配置后通过单次受保护入口访问。

## 1. 验证扰动视图

```bash
BIMER_V3_STAGE=preprocess AUTODL_AUTO_SHUTDOWN=1 \
  bash scripts/run_v3_autodl.sh
```

该阶段建立 `audio-10db`、`video-50` 和 `whisper` 三个验证视图，保留 `sample_id、context_id、label`，状态写入 `artifacts/features/v3-validation/VALIDATION_VIEWS_READY`。最长运行 12 小时。

## 2. 类别损失筛选

训练扰动参数必须按顺序提供三组：音频 10 dB、视频丢帧 50%、Whisper 文本。运行 `scripts/run_v3_experiments.py --stage loss-screen`，随后用 `scripts/summarize_v3_screen.py loss` 生成损失决策。所有筛选命令自动包含 `--seed 42 --skip-test --v3-screen`。

## 3. 排序权重筛选与冻结

使用损失筛选选中的目标运行：

```bash
python scripts/run_v3_experiments.py \
  --stage ranking-screen \
  --classification-loss <selected-loss> \
  <三组配对扰动参数>
```

用 `scripts/summarize_v3_screen.py ranking` 计算干净集、三个验证扰动和 2,000 次对话级门控差值 bootstrap。再执行：

```bash
python scripts/freeze_v3_selection.py \
  --classification-loss <selected-loss> \
  --gate-ranking-weight <selected-lambda> \
  --loss-decision <loss-decision.json> \
  --ranking-decision <ranking-decision.json> \
  --output configs/experiment-v3-selection.json
```

若没有排序候选通过，停止 V3；系统继续使用 V2，并将负结果写入论文。

## 4. 正式训练与单次探索性测试

冻结后运行两个变体、三个种子。正式训练本身仍使用 `--skip-test`：

```bash
python scripts/run_v3_experiments.py --stage formal <三组配对扰动参数>
```

完成后只运行一次：

```bash
python scripts/run_v3_experiments.py --stage test
```

成功后生成 `artifacts/experiments/v3/exploratory-test/TEST_EVALUATED`。正常流程没有覆盖或删除标记的参数。

## 5. 校准、外部视频与 M2

```bash
python scripts/fit_v3_calibration.py \
  --predictions <final-checkpoint-validation_predictions> \
  --profile artifacts/calibration/v3.json \
  --report artifacts/calibration/v3-report.json \
  --figure artifacts/calibration/v3-reliability.png
```

外部视频操作见 `docs/v3_external_test_protocol.md`。M2 最终验收使用 `scripts/run_m2_acceptance.sh`，强制检查首次推理 120 秒、文本修改 15 秒、峰值 6.5 GiB、交换不增长、无人脸关闭视觉以及 JSON/CSV/PNG 导出。

## 6. 成本和归档

`scripts/run_v3_autodl.sh` 将累计 GPU 时间记录在 `_status/GPU_SECONDS_USED`，总上限 18 小时；预处理单次 12 小时、训练单次 8 小时。无论成功还是失败，脚本都会归档日志与结果、生成 SHA-256，并默认自动关机。必须在本地下载并校验归档哈希后，才清理云端文件。
