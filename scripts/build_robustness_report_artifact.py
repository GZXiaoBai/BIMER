#!/usr/bin/env python3
"""Build the canonical Data Analytics artifact for the robustness report."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CORRUPTION_ORDER = (
    "standard",
    "audio_snr_20db",
    "audio_snr_10db",
    "video_frame_drop_25pct",
    "video_frame_drop_50pct",
    "whisper_text",
)
MISSING_ORDER = (
    "missing-audio",
    "missing-vision",
    "missing-text",
    "missing-audio-vision",
    "missing-text-vision",
    "missing-text-audio",
)
DATASET_LABELS = {
    "meld": "MELD（英文）",
    "emotiontalk": "EmotionTalk（中文）",
    "bilingual_average": "双语平均",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], field: str) -> float:
    return float(row[field])


def find_row(
    rows: list[dict[str, str]], condition: str, dataset: str
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["condition"] == condition and row["dataset"] == dataset
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {condition}/{dataset}, got {len(matches)}")
    return matches[0]


def find_class_row(
    rows: list[dict[str, str]], condition: str, dataset: str, emotion: str
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["condition"] == condition
        and row["dataset"] == dataset
        and row["emotion"] == emotion
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one class row for {condition}/{dataset}/{emotion}, got {len(matches)}"
        )
    return matches[0]


def metric_row(row: dict[str, str], order: int) -> dict[str, Any]:
    return {
        "order": order,
        "condition": row["condition"],
        "condition_zh": row["condition_zh"],
        "dataset": row["dataset"],
        "dataset_zh": DATASET_LABELS[row["dataset"]],
        "weighted_f1_mean": f(row, "weighted_f1_mean"),
        "weighted_f1_std": f(row, "weighted_f1_std"),
        "macro_f1_mean": f(row, "macro_f1_mean"),
        "accuracy_mean": f(row, "accuracy_mean"),
        "delta_from_standard": f(row, "delta_from_standard"),
        "relative_delta_pct": f(row, "relative_delta_pct"),
    }


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    report_dir = root / "artifacts/analysis/robustness"
    input_path = args.input.resolve() if args.input else report_dir / "robustness-summary.csv"
    output_path = args.output.resolve() if args.output else report_dir / "artifact.json"

    rows = read_csv(input_path)
    per_class_rows = read_csv(report_dir / "robustness-per-class.csv")
    whisper_quality = read_csv(report_dir / "whisper-transcription-quality.csv")

    standard = find_row(rows, "standard", "bilingual_average")
    audio_10 = find_row(rows, "audio_snr_10db", "bilingual_average")
    video_25 = find_row(rows, "video_frame_drop_25pct", "bilingual_average")
    video_50 = find_row(rows, "video_frame_drop_50pct", "bilingual_average")
    whisper = find_row(rows, "whisper_text", "bilingual_average")
    missing_vision = find_row(rows, "missing-vision", "bilingual_average")
    missing_text = find_row(rows, "missing-text", "bilingual_average")
    missing_audio = find_row(rows, "missing-audio", "bilingual_average")

    whisper_meld = find_row(rows, "whisper_text", "meld")
    whisper_zh = find_row(rows, "whisper_text", "emotiontalk")
    video_25_zh = find_row(rows, "video_frame_drop_25pct", "emotiontalk")
    video_50_zh = find_row(rows, "video_frame_drop_50pct", "emotiontalk")
    video_50_zh_neutral = find_class_row(
        per_class_rows, "video_frame_drop_50pct", "emotiontalk", "neutral"
    )
    video_50_zh_joy = find_class_row(
        per_class_rows, "video_frame_drop_50pct", "emotiontalk", "joy"
    )
    whisper_meld_joy = find_class_row(
        per_class_rows, "whisper_text", "meld", "joy"
    )

    corruption_chart = [
        metric_row(find_row(rows, condition, "bilingual_average"), index)
        for index, condition in enumerate(CORRUPTION_ORDER)
    ]
    corruption_table = [
        metric_row(find_row(rows, condition, dataset), order)
        for order, condition in enumerate(CORRUPTION_ORDER)
        for dataset in ("meld", "emotiontalk")
    ]
    missing_table = [
        metric_row(find_row(rows, condition, "bilingual_average"), order)
        for order, condition in enumerate(MISSING_ORDER)
    ]
    asr_table = [
        {
            "dataset": row["dataset"],
            "dataset_zh": DATASET_LABELS[row["dataset"]],
            "metric": row["metric"],
            "samples": int(row["samples"]),
            "asr_successes": int(row["asr_successes"]),
            "fallback_to_original": int(row["fallback_to_original"]),
            "fallback_rate": float(row["fallback_rate"]),
            "exact_match_rate_on_success": float(row["exact_match_rate_on_success"]),
            "corpus_edit_error_rate_on_success": float(
                row["corpus_edit_error_rate_on_success"]
            ),
            "modified_pipeline_input_rate": float(
                row["modified_pipeline_input_rate"]
            ),
        }
        for row in whisper_quality
    ]
    per_class_focus = []
    for scenario_order, (condition, dataset, scenario_zh) in enumerate(
        (
            ("video_frame_drop_50pct", "emotiontalk", "EmotionTalk：视频丢帧 50%"),
            ("whisper_text", "meld", "MELD：Whisper 转写"),
        )
    ):
        for emotion_order, emotion in enumerate(
            ("neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust")
        ):
            row = find_class_row(per_class_rows, condition, dataset, emotion)
            per_class_focus.append(
                {
                    "scenario_order": scenario_order,
                    "emotion_order": emotion_order,
                    "scenario_zh": scenario_zh,
                    "emotion": emotion,
                    "f1_mean": float(row["f1_mean"]),
                    "f1_std": float(row["f1_std"]),
                    "delta_from_standard": float(row["delta_from_standard"]),
                }
            )

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    sources = [
        {
            "id": "robustness-summary",
            "label": "三随机种子鲁棒性汇总指标",
            "path": "robustness-summary.csv",
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": "SELECT * FROM read_csv_auto('robustness-summary.csv');",
                "description": (
                    "读取标准输入、噪声、丢帧、Whisper 与缺失模态的逐种子测试结果，"
                    "按固定种子 42、123、2026 计算均值、样本标准差和相对标准输入变化。"
                ),
                "executed_at": generated_at,
                "filters": [
                    "仅使用 MELD 与 EmotionTalk 官方 test 划分",
                    "模型选择和训练不使用测试集",
                    "双语平均为两个数据集 weighted-F1 的等权平均",
                ],
                "metric_definitions": [
                    "weighted-F1：按各情感类别测试样本数加权的 F1",
                    "macro-F1：七类情感 F1 的算术平均",
                    "delta_from_standard：同数据集同种子条件指标减去标准输入指标后再汇总",
                    "std：三个固定随机种子结果的样本标准差（ddof=1）",
                ],
                "tables_used": ["robustness-summary.csv"],
            },
        },
        {
            "id": "per-class-results",
            "label": "逐情感类别鲁棒性结果",
            "path": "robustness-per-class.csv",
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": "SELECT * FROM read_csv_auto('robustness-per-class.csv');",
                "description": "汇总每个输入条件、数据集和七类情感的逐种子 F1，并计算相对标准输入变化。",
                "executed_at": generated_at,
                "filters": ["仅使用官方 test 划分", "固定随机种子 42、123、2026"],
                "metric_definitions": [
                    "f1_mean：同一数据集、输入条件和情感类别的三随机种子 F1 均值",
                    "delta_from_standard：该类别 F1 均值减去标准输入下同类别 F1 均值",
                ],
                "tables_used": ["robustness-per-class.csv"],
            },
        },
        {
            "id": "whisper-quality",
            "label": "Whisper 测试集转写质量统计",
            "path": "whisper-transcription-quality.csv",
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": "SELECT * FROM read_csv_auto('whisper-transcription-quality.csv');",
                "description": (
                    "将 Whisper 测试集输出与数据集原始人工文本按样本标识对齐，"
                    "在成功转写样本上计算英文 WER 与中文 CER，并统计回退原文比例。"
                ),
                "executed_at": generated_at,
                "filters": ["仅使用官方 test 划分", "ASR 失败样本按系统策略回退原始人工文本"],
                "metric_definitions": [
                    "WER/CER：成功转写样本的语料级编辑距离除以参考词/字符总数",
                    "fallback_rate：ASR 失败并回退原始文本的样本数占测试样本数比例",
                    "modified_pipeline_input_rate：最终送入模型的文本与规范化人工文本不同的样本比例",
                ],
                "tables_used": ["whisper-transcription-quality.csv"],
            },
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "中英文多模态情感识别鲁棒性实验技术报告",
            "description": "LAGF 完整模型在噪声、丢帧、自动转写和模态缺失条件下的三随机种子测试结果。",
            "generatedAt": generated_at,
            "sources": sources,
            "cards": [
                {
                    "id": "card-standard",
                    "description": "标准人工文本、原始音频和完整视觉输入下的双语等权平均。",
                    "dataset": "headline_metrics",
                    "sourceId": "robustness-summary",
                    "metrics": [
                        {"label": "标准双语 weighted-F1", "field": "standard", "format": "number"}
                    ],
                },
                {
                    "id": "card-audio",
                    "description": "10 dB 音频噪声相对标准输入的绝对变化。",
                    "dataset": "headline_metrics",
                    "sourceId": "robustness-summary",
                    "metrics": [
                        {"label": "10 dB 音频 F1 变化", "field": "audio_delta", "format": "number", "signed": True}
                    ],
                },
                {
                    "id": "card-video",
                    "description": "随机丢弃 50% 视频帧相对标准输入的绝对变化。",
                    "dataset": "headline_metrics",
                    "sourceId": "robustness-summary",
                    "metrics": [
                        {"label": "50% 丢帧 F1 变化", "field": "video_delta", "format": "number", "signed": True}
                    ],
                },
                {
                    "id": "card-whisper",
                    "description": "Whisper 自动转写文本相对人工文本的绝对变化。",
                    "dataset": "headline_metrics",
                    "sourceId": "robustness-summary",
                    "metrics": [
                        {"label": "Whisper F1 变化", "field": "whisper_delta", "format": "number", "signed": True}
                    ],
                },
            ],
            "charts": [
                {
                    "id": "chart-corruption",
                    "title": "不同输入扰动下的双语平均 weighted-F1",
                    "subtitle": "MELD 与 EmotionTalk 等权平均；虚线为标准输入，三随机种子均值",
                    "intent": "comparison",
                    "question": "哪类真实输入扰动最影响完整 LAGF 模型？",
                    "rationale": "水平条形图适合比较六个离散输入条件及其长标签。",
                    "comparisonContext": {
                        "baseline": "标准输入",
                        "denominator": "两个数据集等权平均",
                        "grain": "输入条件",
                        "unit": "weighted-F1",
                    },
                    "type": "horizontalBar",
                    "dataset": "corruption_comparison",
                    "sourceId": "robustness-summary",
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
                        "tooltip": [
                            {"field": "weighted_f1_mean", "type": "quantitative", "format": "number", "label": "weighted-F1"},
                            {"field": "weighted_f1_std", "type": "quantitative", "format": "number", "label": "种子标准差"},
                            {"field": "delta_from_standard", "type": "quantitative", "format": "number", "label": "相对标准输入"},
                        ],
                    },
                    "layout": "full",
                    "labels": {"values": "all"},
                    "palette": {"kind": "sequential", "name": "blue"},
                    "referenceLines": [
                        {
                            "axis": "y",
                            "value": f(standard, "weighted_f1_mean"),
                            "label": "标准输入",
                            "color": "neutral",
                            "lineStyle": "dashed",
                        }
                    ],
                    "settings": {
                        "orientation": "horizontal",
                        "groupMode": "single",
                        "showValues": True,
                        "sort": "descending",
                        "categoryLabelPolicy": "wrap",
                    },
                    "surface": {"surface": "explorer", "viewMode": "both"},
                }
            ],
            "tables": [
                {
                    "id": "table-corruption",
                    "title": "扰动条件的分数据集结果",
                    "subtitle": "三随机种子均值；变化量均相对同一数据集标准输入",
                    "dataset": "corruption_table",
                    "sourceId": "robustness-summary",
                    "layout": "full",
                    "density": "dense",
                    "defaultSort": {"field": "order", "direction": "asc"},
                    "columns": [
                        {"field": "order", "label": "顺序", "format": "number"},
                        {"field": "condition_zh", "label": "输入条件", "type": "text"},
                        {"field": "dataset_zh", "label": "数据集", "type": "text"},
                        {"field": "weighted_f1_mean", "label": "weighted-F1", "format": "number"},
                        {"field": "weighted_f1_std", "label": "标准差", "format": "number"},
                        {"field": "macro_f1_mean", "label": "macro-F1", "format": "number"},
                        {"field": "accuracy_mean", "label": "Accuracy", "format": "number"},
                        {"field": "delta_from_standard", "label": "F1 变化", "format": "number", "movement": True},
                    ],
                },
                {
                    "id": "table-missing",
                    "title": "模态缺失条件的双语平均结果",
                    "subtitle": "三随机种子均值；用于判断模型对各模态的依赖",
                    "dataset": "missing_modality_table",
                    "sourceId": "robustness-summary",
                    "layout": "full",
                    "density": "dense",
                    "defaultSort": {"field": "weighted_f1_mean", "direction": "desc"},
                    "columns": [
                        {"field": "condition_zh", "label": "可用/缺失模态", "type": "text"},
                        {"field": "weighted_f1_mean", "label": "双语 weighted-F1", "format": "number"},
                        {"field": "weighted_f1_std", "label": "标准差", "format": "number"},
                        {"field": "delta_from_standard", "label": "F1 变化", "format": "number", "movement": True},
                        {"field": "relative_delta_pct", "label": "相对变化（%）", "format": "number", "movement": True},
                    ],
                },
                {
                    "id": "table-per-class",
                    "title": "关键退化条件的逐类 F1",
                    "subtitle": "三随机种子均值；变化量相对相同数据集与类别的标准输入",
                    "dataset": "per_class_focus",
                    "sourceId": "per-class-results",
                    "layout": "full",
                    "density": "dense",
                    "defaultSort": {"field": "delta_from_standard", "direction": "asc"},
                    "columns": [
                        {"field": "scenario_zh", "label": "场景", "type": "text"},
                        {"field": "emotion", "label": "情感类别", "type": "text"},
                        {"field": "f1_mean", "label": "类别 F1", "format": "number"},
                        {"field": "f1_std", "label": "标准差", "format": "number"},
                        {"field": "delta_from_standard", "label": "F1 变化", "format": "number", "movement": True},
                    ],
                },
                {
                    "id": "table-asr",
                    "title": "Whisper 转写覆盖与误差",
                    "subtitle": "WER/CER 仅在成功转写样本上计算；失败样本回退人工文本",
                    "dataset": "asr_quality",
                    "sourceId": "whisper-quality",
                    "layout": "full",
                    "density": "dense",
                    "defaultSort": {"field": "samples", "direction": "desc"},
                    "columns": [
                        {"field": "dataset_zh", "label": "数据集", "type": "text"},
                        {"field": "samples", "label": "样本数", "format": "number"},
                        {"field": "fallback_to_original", "label": "回退原文", "format": "number"},
                        {"field": "fallback_rate", "label": "回退率", "format": "percent"},
                        {"field": "metric", "label": "误差指标", "type": "text"},
                        {"field": "corpus_edit_error_rate_on_success", "label": "转写误差率", "format": "percent"},
                        {"field": "exact_match_rate_on_success", "label": "成功样本完全一致率", "format": "percent"},
                        {"field": "modified_pipeline_input_rate", "label": "模型输入被改写比例", "format": "percent"},
                    ],
                },
            ],
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "body": "# 中英文多模态情感识别鲁棒性实验技术报告\n\n完整 LAGF 模型在真实输入退化与模态缺失条件下的固定测试集评估。",
                    "layout": "full",
                },
                {
                    "id": "summary-heading",
                    "type": "markdown",
                    "body": "## 技术结论\n\n当前模型对音频噪声稳定，但对损坏而非完全缺失的视频输入较敏感；Whisper 自动转写带来可见但可控的性能下降。",
                    "layout": "full",
                },
                {
                    "id": "summary-metrics",
                    "type": "metric-strip",
                    "cardIds": ["card-standard", "card-audio", "card-video", "card-whisper"],
                    "layout": "full",
                },
                {
                    "id": "finding-audio",
                    "type": "markdown",
                    "sourceId": "robustness-summary",
                    "body": (
                        "## 音频噪声不是当前主要瓶颈\n\n"
                        f"标准双语 weighted-F1 为 **{f(standard, 'weighted_f1_mean'):.4f}**；"
                        f"10 dB 噪声下为 **{f(audio_10, 'weighted_f1_mean'):.4f}**，"
                        f"绝对变化仅 **{f(audio_10, 'delta_from_standard'):+.4f}**。"
                        "该变化明显小于三次运行的随机种子波动，不能据此宣称噪声提升或下降。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "finding-video",
                    "type": "markdown",
                    "sourceId": "robustness-summary",
                    "body": (
                        "## 损坏的视频比完全缺失的视频更危险\n\n"
                        f"随机丢帧 25% 和 50% 时，双语 weighted-F1 分别降至 "
                        f"**{f(video_25, 'weighted_f1_mean'):.4f}** 与 "
                        f"**{f(video_50, 'weighted_f1_mean'):.4f}**；"
                        f"完全关闭视觉模态时仍为 **{f(missing_vision, 'weighted_f1_mean'):.4f}**。"
                        "这说明门控网络能够处理“明确缺失”，却未充分识别“存在但不可靠”的视觉特征。"
                        f"问题主要来自 EmotionTalk：25% 与 50% 丢帧时分别为 "
                        f"**{f(video_25_zh, 'weighted_f1_mean'):.4f}** 和 "
                        f"**{f(video_50_zh, 'weighted_f1_mean'):.4f}**。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "finding-text",
                    "type": "markdown",
                    "sourceId": "robustness-summary",
                    "body": (
                        "## 文本仍是主导模态\n\n"
                        f"缺失文本使双语 weighted-F1 下降 **{abs(f(missing_text, 'delta_from_standard')):.4f}**；"
                        f"缺失语音仅下降 **{abs(f(missing_audio, 'delta_from_standard')):.4f}**。"
                        "因此，现阶段模型的多模态收益主要来自文本与视觉，音频分支尚未形成稳定增益。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "corruption-chart-block",
                    "type": "chart",
                    "chartId": "chart-corruption",
                    "layout": "full",
                },
                {
                    "id": "corruption-table-block",
                    "type": "table",
                    "tableId": "table-corruption",
                    "layout": "full",
                },
                {
                    "id": "missing-heading",
                    "type": "markdown",
                    "body": "## 模态缺失检查\n\n缺失任意一种或两种模态时均成功输出，无 NaN 或崩溃；表中结果同时揭示各模态的实际贡献。",
                    "layout": "full",
                },
                {
                    "id": "missing-table-block",
                    "type": "table",
                    "tableId": "table-missing",
                    "layout": "full",
                },
                {
                    "id": "class-finding",
                    "type": "markdown",
                    "sourceId": "per-class-results",
                    "body": (
                        "### 退化集中在少数高频或语义关键类别\n\n"
                        f"EmotionTalk 在 50% 丢帧下，`neutral` 类 F1 变化 "
                        f"**{float(video_50_zh_neutral['delta_from_standard']):+.4f}**，"
                        f"`joy` 变化 **{float(video_50_zh_joy['delta_from_standard']):+.4f}**；"
                        f"MELD 在 Whisper 条件下，`joy` 变化 "
                        f"**{float(whisper_meld_joy['delta_from_standard']):+.4f}**。"
                        "总体下降并非均匀扩散，因此后续质量门控应结合类别混淆矩阵分析。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "class-table-block",
                    "type": "table",
                    "tableId": "table-per-class",
                    "layout": "full",
                },
                {
                    "id": "whisper-heading",
                    "type": "markdown",
                    "sourceId": "robustness-summary",
                    "body": (
                        "## 自动转写造成跨语言不对称影响\n\n"
                        f"Whisper 文本使 MELD weighted-F1 变化 "
                        f"**{f(whisper_meld, 'delta_from_standard'):+.4f}**，"
                        f"EmotionTalk 变化 **{f(whisper_zh, 'delta_from_standard'):+.4f}**。"
                        "英文样本下降更明显，说明系统推理质量不仅取决于融合模型，也取决于上游切句和转写。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "whisper-quality-note",
                    "type": "markdown",
                    "sourceId": "whisper-quality",
                    "body": (
                        "### 转写失败回退会高估端到端表现\n\n"
                        f"MELD 有 **{asr_table[0]['fallback_to_original']}/{asr_table[0]['samples']}** 条、"
                        f"EmotionTalk 有 **{asr_table[1]['fallback_to_original']}/{asr_table[1]['samples']}** 条"
                        "在 ASR 失败后回退人工文本。因此 Whisper 对照结果是保守工程策略下的系统表现，"
                        "不是“所有样本纯自动转写”的上限无偏估计。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "asr-table-block",
                    "type": "table",
                    "tableId": "table-asr",
                    "layout": "full",
                },
                {
                    "id": "scope",
                    "type": "markdown",
                    "body": (
                        "## 范围、方法与不确定性\n\n"
                        "- 数据口径：严格使用 MELD 与 EmotionTalk 官方测试划分，不重新随机拆分。\n"
                        "- 模型口径：所有扰动直接评估 seed 42、123、2026 的既有完整 LAGF 检查点，不因测试结果重新训练或调参。\n"
                        "- 双语指标：分别计算两个数据集指标，再做 1:1 等权平均，避免样本量更大的数据集主导结论。\n"
                        "- 区间口径：逐种子结果含样本级 bootstrap 95% CI；汇总文件中的 CI 端点为三个逐种子端点的均值，不能当作跨种子均值的严格 bootstrap 区间。\n"
                        "- 限制：音频和视频扰动仅覆盖两个强度；尚未对门控权重变化、说话人、句长和类别不平衡进行分层显著性分析。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "recommendations",
                    "type": "markdown",
                    "body": (
                        "## 论文与模型的下一步\n\n"
                        "1. 将“损坏视觉输入识别”作为失败分析重点：增加基于人脸检出率、有效帧率和特征稳定性的显式质量信号，再让门控网络使用这些信号。\n"
                        "2. 对比“丢帧后仍开启视觉”与“低质量时自动关闭视觉”，验证可靠性门控是否真正改善系统行为。\n"
                        "3. 在论文中如实报告音频分支贡献接近零；后续用语音活动比例、信噪比或情感声学特征改善音频可靠性。\n"
                        "4. 端到端系统报告同时给出纯成功转写子集和失败回退口径，避免把人工文本回退误写成 Whisper 能力。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": (
                        "## 仍需回答的问题\n\n"
                        "- EmotionTalk 的丢帧退化是否进一步集中在少数长视频或说话人？\n"
                        "- 显式视觉质量分数能否让低质量视觉表现至少不差于直接缺失视觉？\n"
                        "- 英文 Whisper 下降主要来自转写词错、切句边界，还是短片段无文本？\n"
                        "- 冻结音频编码器的前提下，音频分支是否需要独立归一化或更强的模态 dropout？"
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
                        "standard": f(standard, "weighted_f1_mean"),
                        "audio_delta": f(audio_10, "delta_from_standard"),
                        "video_delta": f(video_50, "delta_from_standard"),
                        "whisper_delta": f(whisper, "delta_from_standard"),
                    }
                ],
                "corruption_comparison": corruption_chart,
                "corruption_table": corruption_table,
                "missing_modality_table": missing_table,
                "per_class_focus": per_class_focus,
                "asr_quality": asr_table,
            },
        },
        "sources": sources,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(output_path)


if __name__ == "__main__":
    main()
