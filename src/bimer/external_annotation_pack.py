from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .external_evaluation import annotation_agreement
from .labels import EMOTION_LABELS

ANNOTATION_COLUMNS = (
    "video_id",
    "segment_id",
    "start_seconds",
    "end_seconds",
    "text",
    "asr_confidence",
    "label",
    "notes",
)


@dataclass(frozen=True, slots=True)
class AnnotationSegment:
    start_seconds: float
    end_seconds: float
    text: str
    asr_confidence: float | None = None


def build_annotation_rows(
    segments_by_video: Mapping[str, Sequence[AnnotationSegment]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for video_id in sorted(segments_by_video):
        for segment_id, segment in enumerate(segments_by_video[video_id]):
            rows.append(
                {
                    "video_id": video_id,
                    "segment_id": str(segment_id),
                    "start_seconds": f"{segment.start_seconds:.3f}",
                    "end_seconds": f"{segment.end_seconds:.3f}",
                    "text": segment.text.strip(),
                    "asr_confidence": (
                        "" if segment.asr_confidence is None else f"{segment.asr_confidence:.6f}"
                    ),
                    "label": "",
                    "notes": "",
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_annotation_handoff(
    rows: Sequence[Mapping[str, str]],
    *,
    output_dir: Path | str,
) -> dict[str, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    outputs = {
        "segments": target / "00-segments-and-asr.csv",
        "annotator_one": target / "01-annotator-one.csv",
        "annotator_two": target / "02-annotator-two.csv",
        "adjudication": target / "03-adjudication.csv",
        "instructions": target / "README-双人标注说明.md",
    }
    for name in ("segments", "annotator_one", "annotator_two", "adjudication"):
        _write_csv(outputs[name], rows)
    outputs["instructions"].write_text(
        """# BIMER 外部视频双人标注说明

1. 两名标注者分别填写 `01-annotator-one.csv` 与 `02-annotator-two.csv`。
2. 只填写 `label` 和可选的 `notes`；七类标签固定为
   `neutral、joy、sadness、anger、surprise、fear、disgust`。
3. 标注者必须独立判断、禁止互看另一份结果，也不要查看模型预测。
4. `text` 是 Whisper 预转写，仅用于帮助理解；情感判断应同时观看对应视频。
5. 两份文件完成后计算原始一致率与 Cohen's kappa。kappa 低于 0.60 时，
   统一规则后重新独立标注；达标后只在 `03-adjudication.csv` 仲裁分歧项。
6. 空白标签不是测试结果。未完成两份人工标注和仲裁前，不得在论文中报告外测指标。

## 七类标签的操作性定义

- `neutral`：没有清晰占主导的正向或负向情感，包括平静陈述、解释和礼貌回应。
- `joy`：明显的愉悦、喜爱、兴奋、轻松或积极满足。
- `sadness`：明显的悲伤、失落、遗憾、无助或低落。
- `anger`：明显的愤怒、恼火、指责、敌意或强烈不满。
- `surprise`：由意外信息引起的惊讶、震惊或突然反应；不能只因语调升高就选择。
- `fear`：明显的害怕、焦虑、担忧、紧张或对威胁的回避。
- `disgust`：明显的厌恶、反感、嫌弃、鄙视或排斥。

## 统一判定规则

1. 以整段中占主导的可观察情感为准，综合文本语义、声音和面部/身体表现。
2. 不要因为 ASR 文本中的情感词直接定类；ASR 可能出错，必须回看对应时间段。
3. 多种情感同时存在时，选择表达强度更高且持续更久的一类；仍无法区分时选
   `neutral`，并在 `notes` 写明候选类别。
4. 讽刺、反问或礼貌用语以实际表达态度为准，不按字面正负性判断。
5. 只标注当前时间段，不根据说话者身份、视频来源或后续情节反推标签。
6. 无人脸或画面无关时，依据可用文本与语音判断；音频不清楚时在 `notes`
   记录 `low_audio_quality`，但仍按可观察证据给出一个标签。
""",
        encoding="utf-8",
    )
    return outputs


def prepare_adjudication_rows(
    annotator_one: Sequence[Mapping[str, str]],
    annotator_two: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, str]], dict[str, float | bool | int]]:
    def keyed(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str], Mapping[str, str]]:
        values = {(row["video_id"], row["segment_id"]): row for row in rows}
        if len(values) != len(rows):
            raise ValueError("annotation rows contain duplicate video_id/segment_id keys")
        return values

    first = keyed(annotator_one)
    second = keyed(annotator_two)
    if set(first) != set(second) or not first:
        raise ValueError("annotator files must contain the same non-empty segment set")
    ordered = sorted(first)
    for source in (first, second):
        invalid = [
            key for key in ordered if source[key].get("label", "").strip() not in EMOTION_LABELS
        ]
        if invalid:
            raise ValueError("all human annotation labels must use the fixed seven classes")
    first_labels = [first[key]["label"].strip() for key in ordered]
    second_labels = [second[key]["label"].strip() for key in ordered]
    report = {
        **annotation_agreement(first_labels, second_labels),
        "segments": len(ordered),
        "disagreements": sum(a != b for a, b in zip(first_labels, second_labels, strict=True)),
    }
    rows: list[dict[str, str]] = []
    for key, first_label, second_label in zip(
        ordered,
        first_labels,
        second_labels,
        strict=True,
    ):
        row = {column: str(first[key].get(column, "")) for column in ANNOTATION_COLUMNS}
        row["label"] = first_label if first_label == second_label else ""
        row["notes"] = (
            ""
            if first_label == second_label
            else f"annotator_one={first_label}; annotator_two={second_label}"
        )
        rows.append(row)
    return rows, report
