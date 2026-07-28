from __future__ import annotations

import csv
from pathlib import Path

import pytest

from bimer.external_annotation_pack import (
    AnnotationSegment,
    audit_annotation_readiness,
    build_annotation_rows,
    prepare_adjudication_rows,
    write_annotation_handoff,
)


def test_build_annotation_rows_keeps_text_and_leaves_human_label_blank():
    rows = build_annotation_rows(
        {
            "zh-normal-01": [
                AnnotationSegment(0.0, 2.5, "你好", 0.91),
                AnnotationSegment(2.5, 5.0, "今天很高兴", 0.82),
            ]
        }
    )

    assert rows == [
        {
            "video_id": "zh-normal-01",
            "segment_id": "0",
            "start_seconds": "0.000",
            "end_seconds": "2.500",
            "text": "你好",
            "asr_confidence": "0.910000",
            "label": "",
            "notes": "",
        },
        {
            "video_id": "zh-normal-01",
            "segment_id": "1",
            "start_seconds": "2.500",
            "end_seconds": "5.000",
            "text": "今天很高兴",
            "asr_confidence": "0.820000",
            "label": "",
            "notes": "",
        },
    ]


def test_write_annotation_handoff_creates_independent_blank_copies(tmp_path: Path):
    rows = build_annotation_rows(
        {"en-normal-01": [AnnotationSegment(0.0, 3.0, "Hello there", None)]}
    )

    outputs = write_annotation_handoff(rows, output_dir=tmp_path)

    assert set(outputs) == {
        "segments",
        "annotator_one",
        "annotator_two",
        "adjudication",
        "instructions",
    }
    for name in ("annotator_one", "annotator_two", "adjudication"):
        with outputs[name].open(encoding="utf-8-sig", newline="") as handle:
            saved = list(csv.DictReader(handle))
        assert saved[0]["label"] == ""
        assert saved[0]["text"] == "Hello there"
    instructions = outputs["instructions"].read_text(encoding="utf-8")
    assert "禁止互看" in instructions
    assert "以整段中占主导的可观察情感为准" in instructions
    assert "neutral" in instructions
    assert "讽刺" in instructions
    assert "不要因为 ASR 文本中的情感词直接定类" in instructions


def test_prepare_adjudication_rows_keeps_agreements_and_blanks_disagreements():
    base = {
        "video_id": "en-normal-01",
        "start_seconds": "0.000",
        "end_seconds": "3.000",
        "text": "Hello",
        "asr_confidence": "0.9",
        "notes": "",
    }
    first = [
        {**base, "segment_id": "0", "label": "neutral"},
        {**base, "segment_id": "1", "label": "joy"},
    ]
    second = [
        {**base, "segment_id": "0", "label": "neutral"},
        {**base, "segment_id": "1", "label": "sadness"},
    ]

    rows, report = prepare_adjudication_rows(first, second)

    assert report["raw_agreement"] == 0.5
    assert rows[0]["label"] == "neutral"
    assert rows[1]["label"] == ""
    assert "annotator_one=joy" in rows[1]["notes"]


def test_annotation_readiness_keeps_blank_handoff_explicitly_blocked():
    rows = build_annotation_rows({"en-normal-01": [AnnotationSegment(0.0, 3.0, "Hello", 0.9)]})

    report = audit_annotation_readiness(rows, rows, rows)

    assert report["status"] == "blocked_human_annotation"
    assert report["segments"] == 1
    assert report["agreement_calculated"] is False


def test_annotation_readiness_requires_independence_attestation_before_kappa():
    master = build_annotation_rows(
        {
            "en-normal-01": [
                AnnotationSegment(0.0, 3.0, "Hello", 0.9),
                AnnotationSegment(3.0, 6.0, "Goodbye", 0.8),
            ]
        }
    )
    first = [{**master[0], "label": "joy"}, {**master[1], "label": "sadness"}]
    second = [{**master[0], "label": "joy"}, {**master[1], "label": "sadness"}]

    awaiting = audit_annotation_readiness(master, first, second)
    ready = audit_annotation_readiness(
        master,
        first,
        second,
        independence_confirmed=True,
    )

    assert awaiting["status"] == "awaiting_independence_attestation"
    assert awaiting["exact_label_match"] is True
    assert awaiting["agreement_calculated"] is False
    assert ready["status"] == "ready_for_adjudication"
    assert ready["agreement_calculated"] is True
    assert ready["cohen_kappa"] == 1.0


def test_annotation_readiness_rejects_edited_segments_and_illegal_labels():
    master = build_annotation_rows({"en-normal-01": [AnnotationSegment(0.0, 3.0, "Hello", 0.9)]})
    edited = [{**master[0], "text": "changed", "label": "joy"}]
    with pytest.raises(ValueError, match="immutable segment fields"):
        audit_annotation_readiness(master, edited, edited)

    invalid = [{**master[0], "label": "happy"}]
    with pytest.raises(ValueError, match="fixed seven classes"):
        audit_annotation_readiness(master, invalid, invalid)
