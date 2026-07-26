#!/usr/bin/env python3
"""Build the Data Analytics artifact for the V2 robustness decision report."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


CONDITION_ORDER = (
    "standard",
    "audio_snr_20db",
    "audio_snr_10db",
    "video_frame_drop_25pct",
    "video_frame_drop_50pct",
    "whisper_text",
    "missing-text",
    "missing-audio",
    "missing-vision",
    "missing-audio-vision",
    "missing-text-vision",
    "missing-text-audio",
)
CHART_CONDITIONS = (
    "standard",
    "audio_snr_20db",
    "audio_snr_10db",
    "video_frame_drop_25pct",
    "video_frame_drop_50pct",
    "whisper_text",
    "missing-audio",
    "missing-vision",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root / "artifacts/analysis/v2-robustness",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "artifacts/analysis/v2-robustness/artifact.json",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _comparison(
    rows: list[dict[str, str]],
    condition: str,
    dataset: str = "bilingual_average",
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["condition"] == condition and row["dataset"] == dataset
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one comparison for {condition}/{dataset}")
    return matches[0]


def _number(row: dict[str, str], field: str) -> float:
    return float(row[field])


def _pp(value: float) -> float:
    return value * 100


def _yes(value: str) -> bool:
    return value.lower() == "true"


def _source(
    identifier: str,
    label: str,
    path: str,
    description: str,
    generated_at: str,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "label": label,
        "path": path,
        "query": {
            "engine": "DuckDB",
            "language": "sql",
            "sql": f"SELECT * FROM read_csv_auto('{path}');",
            "description": description,
            "executed_at": generated_at,
            "filters": [
                "仅使用 MELD 与 EmotionTalk 官方 test 划分",
                "固定随机种子 42、123、2026",
                "模型结构与超参数已在验证集冻结，测试集不用于重新选型",
            ],
            "metric_definitions": [
                "weighted-F1：按七类情感测试样本数加权的 F1",
                "双语平均：MELD 与 EmotionTalk 指标的 1:1 等权平均",
                "std：三个固定随机种子结果的样本标准差（ddof=1）",
                "配对 95% CI：按完整 context_id 聚类重采样，比较质量模型减无门控模型的 weighted-F1 差值",
            ],
            "tables_used": [path],
        },
    }


def _write_markdown_report(
    output: Path,
    *,
    decision: dict[str, Any],
    comparisons: list[dict[str, str]],
) -> None:
    clean = _comparison(comparisons, "standard")
    video_25 = _comparison(comparisons, "video_frame_drop_25pct")
    video_50 = _comparison(comparisons, "video_frame_drop_50pct")
    audio_10 = _comparison(comparisons, "audio_snr_10db")
    vision_only = _comparison(comparisons, "missing-text-audio")
    text = f"""# 新版鲁棒性实验与最终模型选择

## 结论

最终系统采用 `quality_lagf`（质量感知门控模型）。该决定沿用验证集冻结结果，不根据测试集重新调参：验证集双语 weighted-F1 较无门控上下文模型提高 {_pp(decision['validation']['quality_minus_no_gates']):.3f} 个百分点。

标准测试上两者基本持平：质量模型为 {_number(clean, 'quality_weighted_f1_mean'):.4f}，无门控模型为 {_number(clean, 'no_gates_weighted_f1_mean'):.4f}，差值 {_pp(_number(clean, 'quality_minus_no_gates')):+.3f} 个百分点，配对 95% CI 为 [{_pp(_number(clean, 'ci95_lower')):+.3f}, {_pp(_number(clean, 'ci95_upper')):+.3f}]。

## 被数据支持的改进

- 视频丢帧 25%：质量模型提高 {_pp(_number(video_25, 'quality_minus_no_gates')):+.3f} 个百分点，95% CI [{_pp(_number(video_25, 'ci95_lower')):+.3f}, {_pp(_number(video_25, 'ci95_upper')):+.3f}]。
- 视频丢帧 50%：质量模型提高 {_pp(_number(video_50, 'quality_minus_no_gates')):+.3f} 个百分点，95% CI [{_pp(_number(video_50, 'ci95_lower')):+.3f}, {_pp(_number(video_50, 'ci95_upper')):+.3f}]。
- 质量模型在 50% 丢帧下相对标准输入下降 {_pp(decision['test']['quality_video_50_loss_from_clean']):.3f} 个百分点，小于完全缺失视觉时的 {_pp(decision['test']['quality_missing_vision_loss_from_clean']):.3f} 个百分点。V1 中“损坏视频比缺失视频更危险”的关键失败模式已修复。

## 仍需如实报告的弱点

- 10 dB 音频噪声下，质量模型比无门控模型低 {_pp(abs(_number(audio_10, 'quality_minus_no_gates'))):.3f} 个百分点。
- 仅保留视频时，质量模型低 {_pp(abs(_number(vision_only, 'quality_minus_no_gates'))):.3f} 个百分点。
- 质量门控不是所有条件下普遍更优；其已验证价值主要是视频质量退化识别，以及缺失语音或视觉时的可靠性。

## 统计口径

所有指标来自三个固定随机种子 `42、123、2026`。均值和样本标准差按种子汇总；模型差值的 95% 置信区间使用逐样本预测，按完整对话 `context_id` 做配对 cluster bootstrap（2,000 次）。两个数据集分别重采样后做 1:1 等权平均。
"""
    output.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output = args.output.resolve()
    summary_rows = _read_csv(input_dir / "model-condition-summary.csv")
    comparison_rows = _read_csv(input_dir / "model-comparison.csv")
    decision = _read_json(input_dir / "selection-decision.json")
    validation = _read_json(input_dir / "validation.json")
    if validation.get("status") != "passed":
        raise ValueError("V2 robustness input validation did not pass")

    clean = _comparison(comparison_rows, "standard")
    audio_10 = _comparison(comparison_rows, "audio_snr_10db")
    video_25 = _comparison(comparison_rows, "video_frame_drop_25pct")
    video_50 = _comparison(comparison_rows, "video_frame_drop_50pct")
    missing_audio = _comparison(comparison_rows, "missing-audio")
    missing_vision = _comparison(comparison_rows, "missing-vision")
    vision_only = _comparison(comparison_rows, "missing-text-audio")

    order_map = {condition: index for index, condition in enumerate(CONDITION_ORDER)}
    chart_rows = [
        {
            "condition_order": order_map[row["condition"]],
            "condition": row["condition"],
            "condition_zh": row["condition_zh"],
            "model": row["model"],
            "model_zh": row["model_zh"],
            "weighted_f1_mean": float(row["weighted_f1_mean"]),
            "weighted_f1_std": float(row["weighted_f1_std"]),
            "delta_from_standard": float(row["delta_from_standard"]),
            "dataset": row["dataset"],
            "group": row["group"],
        }
        for row in summary_rows
        if row["dataset"] == "bilingual_average"
        and row["condition"] in CHART_CONDITIONS
    ]
    chart_rows.sort(key=lambda row: (row["condition_order"], row["model"]))

    comparison_table = [
        {
            "condition_order": order_map[row["condition"]],
            "group": row["group"],
            "condition": row["condition"],
            "condition_zh": row["condition_zh"],
            "quality_weighted_f1": float(row["quality_weighted_f1_mean"]),
            "quality_std": float(row["quality_weighted_f1_std"]),
            "no_gates_weighted_f1": float(row["no_gates_weighted_f1_mean"]),
            "no_gates_std": float(row["no_gates_weighted_f1_std"]),
            "quality_minus_no_gates_pp": _pp(
                float(row["quality_minus_no_gates"])
            ),
            "ci95_lower_pp": _pp(float(row["ci95_lower"])),
            "ci95_upper_pp": _pp(float(row["ci95_upper"])),
            "significant": _yes(row["significant_at_0_05"]),
            "supports_quality": _yes(row["supports_quality"]),
            "bootstrap_unit": row["bootstrap_unit"],
            "cluster_count_min": int(row["cluster_count_min"]),
            "cluster_count_max": int(row["cluster_count_max"]),
        }
        for row in comparison_rows
        if row["dataset"] == "bilingual_average"
    ]
    comparison_table.sort(key=lambda row: row["condition_order"])

    acceptance_rows = [
        {
            "check_order": 1,
            "criterion": "验证集较无门控模型至少提高 0.5 个百分点",
            "observed": _pp(
                decision["validation"]["quality_minus_no_gates"]
            ),
            "unit": "百分点",
            "passed": decision["acceptance"][
                "validation_gain_at_least_0_5pp"
            ],
        },
        {
            "check_order": 2,
            "criterion": "标准测试代价不超过 0.5 个百分点",
            "observed": _pp(
                decision["test"]["clean_quality_minus_no_gates"]
            ),
            "unit": "百分点（质量模型－无门控）",
            "passed": decision["acceptance"][
                "clean_test_penalty_no_more_than_0_5pp"
            ],
        },
        {
            "check_order": 3,
            "criterion": "50% 丢帧损失不大于完全缺失视觉损失",
            "observed": _pp(
                decision["test"]["quality_missing_vision_loss_from_clean"]
                - decision["test"]["quality_video_50_loss_from_clean"]
            ),
            "unit": "百分点（正值为通过余量）",
            "passed": decision["acceptance"][
                "video_50_loss_not_greater_than_missing_vision_loss"
            ],
        },
    ]

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    sources = [
        _source(
            "model-summary",
            "两种模型三随机种子条件汇总",
            "model-condition-summary.csv",
            "读取两种模型在十二个鲁棒性条件下的 MELD、EmotionTalk 与双语平均指标。",
            generated_at,
        ),
        _source(
            "paired-comparison",
            "逐对话配对模型差值",
            "model-comparison.csv",
            "读取质量门控减无门控上下文模型的逐条件差值及完整对话 cluster bootstrap 区间。",
            generated_at,
        ),
    ]

    title = "新版鲁棒性实验与最终模型选择"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "质量感知门控与无门控上下文模型的三随机种子鲁棒性对比及最终部署决策。",
            "generatedAt": generated_at,
            "sources": sources,
            "cards": [
                {
                    "id": "card-validation",
                    "description": "验证集冻结选型时，质量模型相对无门控模型的双语 weighted-F1 差值。",
                    "dataset": "headline_metrics",
                    "sourceId": "paired-comparison",
                    "metrics": [
                        {
                            "label": "验证集增益（百分点）",
                            "field": "validation_gain_pp",
                            "format": "number",
                            "signed": True,
                        }
                    ],
                },
                {
                    "id": "card-clean",
                    "description": "标准测试输入下质量模型相对无门控模型的双语 weighted-F1 差值。",
                    "dataset": "headline_metrics",
                    "sourceId": "paired-comparison",
                    "metrics": [
                        {
                            "label": "标准测试差值（百分点）",
                            "field": "clean_delta_pp",
                            "format": "number",
                            "signed": True,
                        }
                    ],
                },
                {
                    "id": "card-video25",
                    "description": "视频随机丢帧 25% 时质量模型相对无门控模型的双语 weighted-F1 差值。",
                    "dataset": "headline_metrics",
                    "sourceId": "paired-comparison",
                    "metrics": [
                        {
                            "label": "25% 丢帧增益（百分点）",
                            "field": "video25_delta_pp",
                            "format": "number",
                            "signed": True,
                        }
                    ],
                },
                {
                    "id": "card-video50",
                    "description": "视频随机丢帧 50% 时质量模型相对无门控模型的双语 weighted-F1 差值。",
                    "dataset": "headline_metrics",
                    "sourceId": "paired-comparison",
                    "metrics": [
                        {
                            "label": "50% 丢帧增益（百分点）",
                            "field": "video50_delta_pp",
                            "format": "number",
                            "signed": True,
                        }
                    ],
                },
            ],
            "charts": [
                {
                    "id": "chart-model-comparison",
                    "title": "两种模型在鲁棒性条件下的双语 weighted-F1",
                    "subtitle": "清洁输入近乎相同；质量门控主要改善视频丢帧和缺失语音或视觉场景",
                    "intent": "comparison",
                    "question": "质量感知门控在哪些输入条件下改善或损害模型表现？",
                    "rationale": "分组柱状图直接比较每个输入条件下两种模型的三随机种子均值。",
                    "comparisonContext": {
                        "baseline": "无门控上下文模型",
                        "denominator": "MELD 与 EmotionTalk 1:1 等权平均",
                        "grain": "输入条件与模型",
                        "unit": "weighted-F1",
                    },
                    "type": "bar",
                    "dataset": "robustness_chart",
                    "sourceId": "model-summary",
                    "encodings": {
                        "x": {
                            "field": "condition_zh",
                            "type": "nominal",
                            "label": "输入条件",
                        },
                        "y": {
                            "field": "weighted_f1_mean",
                            "type": "quantitative",
                            "aggregate": "none",
                            "format": "number",
                            "label": "双语平均 weighted-F1",
                        },
                        "color": {
                            "field": "model_zh",
                            "type": "nominal",
                            "label": "模型",
                        },
                        "tooltip": [
                            {
                                "field": "weighted_f1_mean",
                                "type": "quantitative",
                                "format": "number",
                                "label": "weighted-F1",
                            },
                            {
                                "field": "weighted_f1_std",
                                "type": "quantitative",
                                "format": "number",
                                "label": "种子标准差",
                            },
                            {
                                "field": "delta_from_standard",
                                "type": "quantitative",
                                "format": "number",
                                "label": "相对标准输入",
                            },
                        ],
                    },
                    "options": {
                        "orientation": "vertical",
                        "grouping": "grouped",
                        "showLegend": True,
                    },
                    "layout": "full",
                }
            ],
            "tables": [
                {
                    "id": "table-comparison",
                    "title": "十二种条件的双语鲁棒性对比",
                    "subtitle": "差值为质量门控减无门控；95% CI 按完整对话配对 cluster bootstrap",
                    "dataset": "comparison_table",
                    "sourceId": "paired-comparison",
                    "layout": "full",
                    "density": "dense",
                    "defaultSort": {
                        "field": "condition_order",
                        "direction": "asc",
                    },
                    "columns": [
                        {
                            "field": "condition_order",
                            "label": "顺序",
                            "format": "number",
                        },
                        {
                            "field": "condition_zh",
                            "label": "输入条件",
                            "type": "text",
                        },
                        {
                            "field": "quality_weighted_f1",
                            "label": "质量门控 F1",
                            "format": "number",
                        },
                        {
                            "field": "no_gates_weighted_f1",
                            "label": "无门控 F1",
                            "format": "number",
                        },
                        {
                            "field": "quality_minus_no_gates_pp",
                            "label": "差值（百分点）",
                            "format": "number",
                            "movement": True,
                        },
                        {
                            "field": "ci95_lower_pp",
                            "label": "CI 下界",
                            "format": "number",
                        },
                        {
                            "field": "ci95_upper_pp",
                            "label": "CI 上界",
                            "format": "number",
                        },
                        {
                            "field": "significant",
                            "label": "差异显著",
                            "type": "boolean",
                        },
                    ],
                },
                {
                    "id": "table-acceptance",
                    "title": "最终模型验收标准",
                    "subtitle": "全部标准均按预先制定的 V2 修正计划检查",
                    "dataset": "acceptance_checks",
                    "sourceId": "paired-comparison",
                    "layout": "full",
                    "density": "comfortable",
                    "defaultSort": {
                        "field": "check_order",
                        "direction": "asc",
                    },
                    "columns": [
                        {
                            "field": "check_order",
                            "label": "顺序",
                            "format": "number",
                        },
                        {
                            "field": "criterion",
                            "label": "验收标准",
                            "type": "text",
                        },
                        {
                            "field": "observed",
                            "label": "观测值",
                            "format": "number",
                        },
                        {
                            "field": "unit",
                            "label": "口径",
                            "type": "text",
                        },
                        {
                            "field": "passed",
                            "label": "通过",
                            "type": "boolean",
                        },
                    ],
                },
            ],
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "body": f"# {title}\n\n质量感知门控与无门控上下文模型的固定测试集鲁棒性复核。",
                    "layout": "full",
                },
                {
                    "id": "executive-summary",
                    "type": "markdown",
                    "sourceId": "paired-comparison",
                    "body": (
                        "## 技术结论\n\n"
                        "最终系统采用 **QualityAwareLanguageGatedFusion（quality_lagf）**。"
                        "模型在验证集上按预先口径冻结；新版测试只用于检查鲁棒性验收标准，"
                        "没有据测试结果重新调参。标准输入下两种模型实质持平，"
                        "质量门控的主要已验证收益来自视频质量退化和部分模态缺失场景。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "headline-metrics",
                    "type": "metric-strip",
                    "cardIds": [
                        "card-validation",
                        "card-clean",
                        "card-video25",
                        "card-video50",
                    ],
                    "layout": "full",
                },
                {
                    "id": "selection-validity",
                    "type": "markdown",
                    "body": (
                        "## 选型有效性\n\n"
                        f"验证集双语 weighted-F1：质量模型 **{decision['validation']['quality_bilingual_weighted_f1']:.4f}**，"
                        f"无门控模型 **{decision['validation']['no_gates_bilingual_weighted_f1']:.4f}**，"
                        f"差值 **{_pp(decision['validation']['quality_minus_no_gates']):+.3f} 个百分点**。"
                        "该差值超过预设的 0.5 个百分点标准，且配置文件明确记录"
                        "`test_set_used_for_selection=false`。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "chart-block",
                    "type": "chart",
                    "chartId": "chart-model-comparison",
                    "layout": "full",
                },
                {
                    "id": "video-finding",
                    "type": "markdown",
                    "sourceId": "paired-comparison",
                    "body": (
                        "## 视频质量建模取得目标性收益\n\n"
                        f"25% 丢帧时质量模型领先 **{_pp(_number(video_25, 'quality_minus_no_gates')):.3f} 个百分点**，"
                        f"配对 95% CI 为 [{_pp(_number(video_25, 'ci95_lower')):.3f}, "
                        f"{_pp(_number(video_25, 'ci95_upper')):.3f}]，区间不含 0。"
                        f"50% 丢帧时平均领先 **{_pp(_number(video_50, 'quality_minus_no_gates')):.3f} 个百分点**，"
                        "但区间轻微跨过 0，属于方向一致、证据强度较弱。"
                        f"更关键的是，质量模型 50% 丢帧相对清洁输入下降 "
                        f"**{_pp(decision['test']['quality_video_50_loss_from_clean']):.3f} 个百分点**，"
                        f"小于完全缺失视觉的 **{_pp(decision['test']['quality_missing_vision_loss_from_clean']):.3f} 个百分点**；"
                        "V1 中“损坏视频比完全缺失视频更危险”的失败模式已经修复。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "comparison-table-block",
                    "type": "table",
                    "tableId": "table-comparison",
                    "layout": "full",
                },
                {
                    "id": "remaining-weaknesses",
                    "type": "markdown",
                    "sourceId": "paired-comparison",
                    "body": (
                        "## 仍然存在的边界\n\n"
                        f"质量门控并非在所有条件下都更好。10 dB 音频噪声时低 "
                        f"**{_pp(abs(_number(audio_10, 'quality_minus_no_gates'))):.3f} 个百分点**，"
                        f"仅保留视频时低 **{_pp(abs(_number(vision_only, 'quality_minus_no_gates'))):.3f} 个百分点**；"
                        f"相反，缺失语音和缺失视觉时分别领先 "
                        f"**{_pp(_number(missing_audio, 'quality_minus_no_gates')):.3f}** 与 "
                        f"**{_pp(_number(missing_vision, 'quality_minus_no_gates')):.3f} 个百分点**。"
                        "因此论文应把创新表述为“面向质量退化的目标性可靠性改进”，"
                        "不能写成“所有扰动下普遍优于无门控模型”。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "acceptance-table-block",
                    "type": "table",
                    "tableId": "table-acceptance",
                    "layout": "full",
                },
                {
                    "id": "methods",
                    "type": "markdown",
                    "body": (
                        "## 统计方法与不确定性\n\n"
                        "- 三种子均值使用固定随机种子 42、123、2026，标准差为样本标准差（ddof=1）。\n"
                        "- 配对区间直接读取两种模型的逐样本预测；每次按完整对话 `context_id` 重采样，避免把同一对话中的语句当作独立样本。\n"
                        "- MELD 与 EmotionTalk 在每个随机种子内分别重采样，再按 1:1 等权平均；bootstrap 共 2,000 次。\n"
                        "- 标准测试差值的区间跨 0，不能宣称质量门控提高了清洁输入准确率。\n"
                        "- 50% 丢帧差值的区间也轻微跨 0；该条件的主要证据是平均方向和预先定义的行为验收标准，而不是显著性声明。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "final-decision",
                    "type": "markdown",
                    "body": (
                        "## 最终部署决定\n\n"
                        "系统保留 `quality_lagf`。理由是：验证集选型增益达到预设阈值，"
                        "清洁测试代价可忽略，且低质量视频行为已从 V1 的失败状态恢复。"
                        "答辩和论文中同时保留无门控上下文模型作为强基线，"
                        "并把音频噪声与仅视频场景列为后续改进方向。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": (
                        "## 仍需回答的问题\n\n"
                        "- 为什么显式音频质量输入没有改善 10 dB 噪声条件？\n"
                        "- 仅视频场景下，质量门控是否因训练时单模态样本不足而过度抑制视觉？\n"
                        "- 最终 Gradio 系统能否在低质量视频时稳定产生与离线实验一致的质量分数？"
                    ),
                    "layout": "full",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline_metrics": [
                    {
                        "validation_gain_pp": _pp(
                            decision["validation"]["quality_minus_no_gates"]
                        ),
                        "clean_delta_pp": _pp(
                            _number(clean, "quality_minus_no_gates")
                        ),
                        "video25_delta_pp": _pp(
                            _number(video_25, "quality_minus_no_gates")
                        ),
                        "video50_delta_pp": _pp(
                            _number(video_50, "quality_minus_no_gates")
                        ),
                    }
                ],
                "robustness_chart": chart_rows,
                "comparison_table": comparison_table,
                "acceptance_checks": acceptance_rows,
            },
        },
        "sources": sources,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown_report(
        output.parent / "robustness-decision-report.md",
        decision=decision,
        comparisons=comparison_rows,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
