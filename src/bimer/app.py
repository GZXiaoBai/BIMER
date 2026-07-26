from __future__ import annotations

import uuid
from dataclasses import replace
from html import escape
from pathlib import Path
from typing import Any, Sequence
import time

import matplotlib.pyplot as plt

from .export import export_analysis_csv, export_analysis_figure, export_analysis_json
from .inference import DialogueAnalyzer, TranscriptSegment
from .schema import AnalysisResult

_COLORS = {
    "neutral": "#94A3B8",
    "joy": "#22C55E",
    "sadness": "#3B82F6",
    "anger": "#EF4444",
    "surprise": "#F59E0B",
    "fear": "#8B5CF6",
    "disgust": "#14B8A6",
}


def transcript_rows(segments: Sequence[TranscriptSegment]) -> list[list[object]]:
    return [
        [segment.start_seconds, segment.end_seconds, segment.text]
        for segment in segments
    ]


def analysis_rows(result: AnalysisResult) -> list[list[object]]:
    rows: list[list[object]] = []
    for segment in result.segments:
        confidence = max(segment.probabilities.values(), default=0.0)
        quality_means = []
        for name in ("text", "audio", "vision"):
            values = list(segment.modality_quality.get(name, {}).values())
            quality_means.append(sum(values) / len(values) if values else 0.0)
        rows.append(
            [
                segment.start_seconds,
                segment.end_seconds,
                segment.text,
                segment.emotion,
                confidence,
                segment.modality_gates.get("text", 0.0),
                segment.modality_gates.get("audio", 0.0),
                segment.modality_gates.get("vision", 0.0),
                *[
                    segment.modality_available.get(name, False)
                    for name in ("text", "audio", "vision")
                ],
                *quality_means,
                ", ".join(segment.quality_warnings),
                segment.confidence_status,
            ]
        )
    return rows


def timeline_figure(result: AnalysisResult):
    figure, axis = plt.subplots(figsize=(12, 2.8))
    for index, segment in enumerate(result.segments):
        width = segment.end_seconds - segment.start_seconds
        axis.barh(
            [0],
            [width],
            left=[segment.start_seconds],
            height=0.45,
            color=_COLORS[str(segment.emotion)],
            edgecolor="white",
        )
        axis.text(
            segment.start_seconds + width / 2,
            0,
            str(segment.emotion),
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            clip_on=True,
        )
    axis.set_yticks([])
    axis.set_xlabel("Time (seconds)")
    axis.set_title(f"Dialogue emotion timeline · {result.language}")
    axis.set_xlim(left=0)
    figure.tight_layout()
    return figure


def distribution_figure(result: AnalysisResult):
    figure, axis = plt.subplots(figsize=(8, 3.2))
    labels = list(_COLORS)
    axis.bar(
        labels,
        [result.global_distribution[label] for label in labels],
        color=[_COLORS[label] for label in labels],
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Mean probability")
    axis.set_title("Global emotion distribution")
    figure.tight_layout()
    return figure


def modality_quality_figure(result: AnalysisResult):
    figure, axis = plt.subplots(figsize=(8, 3.2))
    names = ("text", "audio", "vision")
    gate_means = [
        sum(segment.modality_gates.get(name, 0.0) for segment in result.segments)
        / max(1, len(result.segments))
        for name in names
    ]
    quality_means = []
    for name in names:
        values = [
            value
            for segment in result.segments
            for value in segment.modality_quality.get(name, {}).values()
        ]
        quality_means.append(sum(values) / max(1, len(values)))
    positions = list(range(3))
    axis.bar([value - 0.18 for value in positions], gate_means, 0.36, label="gate")
    axis.bar([value + 0.18 for value in positions], quality_means, 0.36, label="quality")
    axis.set_xticks(positions, names)
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Modality contribution and quality")
    axis.legend()
    figure.tight_layout()
    return figure


def timeline_html(result: AnalysisResult) -> str:
    buttons = []
    for segment in result.segments:
        color = _COLORS[str(segment.emotion)]
        title = escape(f"{segment.start_seconds:.1f}s · {segment.emotion} · {segment.text}")
        script = (
            "const v=document.querySelector('#dialogue-video video');"
            f"if(v){{v.currentTime={segment.start_seconds};v.play();}}"
            "return false;"
        )
        buttons.append(
            f'<button title="{title}" onclick="{script}" '
            f'style="background:{color};color:white;border:0;border-radius:6px;'
            'padding:8px 10px;margin:3px;cursor:pointer">'
            f'{segment.start_seconds:.1f}s {escape(str(segment.emotion))}</button>'
        )
    return (
        '<div aria-label="clickable emotion timeline" style="display:flex;flex-wrap:wrap">'
        + "".join(buttons)
        + "</div>"
    )


def _rows_to_segments(rows: Any) -> list[TranscriptSegment]:
    if hasattr(rows, "values"):
        rows = rows.values.tolist()
    return [
        TranscriptSegment(float(row[0]), float(row[1]), str(row[2]).strip())
        for row in rows
        if len(row) >= 3 and str(row[2]).strip()
    ]


def create_app(
    analyzer: DialogueAnalyzer,
    *,
    export_root: Path | str = "artifacts/exports",
):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install bimer[inference] to launch the Gradio app") from exc

    export_directory = Path(export_root)

    def run_transcription(video: str, requested_language: str):
        if not video:
            raise gr.Error("请先上传视频")
        detected, segments = analyzer.transcribe(Path(video), requested_language)
        return transcript_rows(segments), detected

    def run_analysis(video: str, detected_language: str, rows: Any):
        if not video:
            raise gr.Error("请先上传视频")
        segments = _rows_to_segments(rows)
        result = analyzer.analyze_segments(
            Path(video),
            detected_language=detected_language,
            segments=segments,
        )
        run_id = uuid.uuid4().hex
        export_started = time.perf_counter()
        csv_path = export_analysis_csv(result, export_directory / f"{run_id}.csv")
        figure_path = export_analysis_figure(result, export_directory / f"{run_id}.png")
        runtime_profile = dict(result.runtime_profile)
        runtime_profile["export"] = time.perf_counter() - export_started
        result = replace(result, runtime_profile=runtime_profile)
        json_path = export_analysis_json(result, export_directory / f"{run_id}.json")
        return (
            analysis_rows(result),
            timeline_figure(result),
            distribution_figure(result),
            modality_quality_figure(result),
            timeline_html(result),
            result.to_dict(),
            str(json_path),
            str(csv_path),
            str(figure_path),
        )

    def clear_runtime_cache():
        cache = getattr(analyzer.feature_pipeline, "cache", None)
        if cache is None:
            return "运行时缓存未启用"
        count = cache.clear()
        return f"已清除 {count} 个缓存文件"

    with gr.Blocks(title="BIMER 中英文多模态情感识别") as demo:
        gr.Markdown(
            "# 中英文多模态对话情感识别\n"
            "上传不超过3分钟的 MP4/MOV。结果仅用于学术研究，不构成心理或医疗判断。"
        )
        detected_state = gr.State("en")
        with gr.Row():
            video = gr.Video(label="对话视频", sources=["upload"], elem_id="dialogue-video")
            requested_language = gr.Radio(
                choices=[("自动检测", "auto"), ("中文", "zh"), ("英文", "en")],
                value="auto",
                label="语言",
            )
        transcribe_button = gr.Button("1. 转写并切句", variant="primary")
        transcript = gr.Dataframe(
            headers=["start_seconds", "end_seconds", "text"],
            datatype=["number", "number", "str"],
            interactive=True,
            label="可编辑转写",
        )
        analyze_button = gr.Button("2. 分析情感", variant="primary")
        with gr.Row():
            clear_cache_button = gr.Button("清除24小时特征缓存")
            cache_status = gr.Textbox(label="缓存状态", interactive=False)
        analysis = gr.Dataframe(
            headers=[
                "start_seconds",
                "end_seconds",
                "text",
                "emotion",
                "confidence",
                "gate_text",
                "gate_audio",
                "gate_vision",
                "available_text",
                "available_audio",
                "available_vision",
                "quality_text_mean",
                "quality_audio_mean",
                "quality_vision_mean",
                "quality_warnings",
                "confidence_status",
            ],
            interactive=False,
            label="逐句结果",
        )
        timeline = gr.Plot(label="情绪时间线")
        distribution = gr.Plot(label="全局情感分布")
        modality_quality = gr.Plot(label="模态贡献与质量")
        clickable_timeline = gr.HTML(label="点击跳转到对应片段")
        raw_json = gr.JSON(label="结构化结果")
        with gr.Row():
            json_file = gr.File(label="JSON导出")
            csv_file = gr.File(label="CSV导出")
            figure_file = gr.File(label="结果图导出")

        transcribe_button.click(
            run_transcription,
            inputs=[video, requested_language],
            outputs=[transcript, detected_state],
        )
        analyze_button.click(
            run_analysis,
            inputs=[video, detected_state, transcript],
            outputs=[
                analysis,
                timeline,
                distribution,
                modality_quality,
                clickable_timeline,
                raw_json,
                json_file,
                csv_file,
                figure_file,
            ],
        )
        clear_cache_button.click(clear_runtime_cache, outputs=[cache_status])
    return demo
