# BIMER V2 最终系统外部视频测试规程

外部测试在 V2 部署模型、校准参数和不确定阈值冻结后执行，视频不得参与训练或验证筛选。V3 和 V4 均已按预声明规则在验证阶段停止，不参与最终外部测试或上线判定。

1. 按 `data/templates/external-video-plan.csv` 准备中文、英文各 10 段 30–60 秒视频。每种语言在 `normal_face`、`no_face`、`background_noise`、`multi_cut`、`accent_fast_change` 五个条件下各准备 2 段。只使用本人拍摄或具有明确授权的素材。`authorization_basis` 只允许 `self_recorded`、`written_permission` 或 `open_license`，`authorization_reference` 必须对应私有授权记录或许可页面存档。
2. 运行 `scripts/lock_external_video_plan.py` 生成含绝对路径、SHA-256 和时长的锁定清单。锁定后不得替换视频；必须替换时作废整轮外部测试并重新锁定。
3. 两名标注者分别复制 `data/templates/external-annotations.csv`，独立完成逐句七分类标注，禁止互看结果。记录原始一致率和 Cohen’s kappa。
4. 若 kappa 小于 0.60，统一标注说明后重新独立标注一次。之后生成仲裁文件，仲裁文件只解决两人不一致的条目。
5. 在运行 V2 前确认视频清单哈希未变化，再导出预测 CSV。运行 `scripts/evaluate_external_videos.py`，输出总体指标、五类条件指标、视频级 cluster bootstrap 置信区间、置信度覆盖率和逐阶段耗时。最终流程只要求 `--v2-predictions`；可选的 `--comparison-predictions` 仅用于已经冻结、未参与外部测试调参的候选。
6. 论文保留三个成功案例和三个失败案例；失败案例不得删除或换成更容易的视频。

七类标签固定为：`neutral、joy、sadness、anger、surprise、fear、disgust`。
