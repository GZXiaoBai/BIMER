from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .app import create_app
from .data_adapters import (
    check_official_split_counts,
    count_records,
    load_emotiontalk_manifest,
    load_meld_csv,
)
from .experiment import ExperimentConfig, evaluate_checkpoint, resolve_device, run_experiment
from .export import export_analysis_csv, export_analysis_json
from .feature_extraction_runner import DatasetFeatureExtractionRunner, load_full_waveform
from .feature_extractors import (
    AudioFeatureExtractor,
    TextFeatureExtractor,
    VisionFeatureExtractor,
    YuNetFaceCropper,
)
from .feature_store import FeatureStore
from .inference import (
    DialogueAnalyzer,
    FasterWhisperTranscriber,
    PretrainedFeaturePipeline,
)
from .manifest import read_manifest, write_manifest
from .model_factory import build_model
from .robustness import add_noise_at_snr
from .validation import validate_dataset_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bimer")
    commands = parser.add_subparsers(dest="command", required=True)

    meld = commands.add_parser("prepare-meld")
    meld.add_argument("--train-csv", required=True)
    meld.add_argument("--dev-csv", required=True)
    meld.add_argument("--test-csv", required=True)
    meld.add_argument("--train-media", required=True)
    meld.add_argument("--dev-media", required=True)
    meld.add_argument("--test-media", required=True)
    meld.add_argument("--output", required=True)
    meld.add_argument("--allow-partial", action="store_true")

    emotiontalk = commands.add_parser("prepare-emotiontalk")
    emotiontalk.add_argument("--train-json", required=True)
    emotiontalk.add_argument("--validation-json", required=True)
    emotiontalk.add_argument("--test-json", required=True)
    emotiontalk.add_argument("--media-root", required=True)
    emotiontalk.add_argument("--output", required=True)
    emotiontalk.add_argument("--allow-partial", action="store_true")

    validate = commands.add_parser("validate")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--official-counts", action="store_true")

    extract = commands.add_parser("extract-features")
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--features", required=True)
    extract.add_argument("--yunet-model", required=True)
    extract.add_argument("--dataset", choices=["meld", "emotiontalk"])
    extract.add_argument("--split")
    extract.add_argument("--device", default="auto")
    extract.add_argument("--shard-size", type=int, default=1024)
    extract.add_argument("--audio-snr", type=float)
    extract.add_argument("--frame-drop", type=float, default=0.0)

    train = commands.add_parser("train")
    train.add_argument("--manifest", required=True)
    train.add_argument("--features", required=True)
    train.add_argument("--output", required=True)
    train.add_argument(
        "--model",
        choices=["majority", "text", "audio", "vision", "early_mlp", "early_context", "lagf"],
        default="lagf",
    )
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--max-epochs", type=int, default=50)
    train.add_argument("--patience", type=int, default=7)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=1e-2)
    train.add_argument("--device", default="auto")
    train.add_argument("--no-language", action="store_true")
    train.add_argument("--no-gates", action="store_true")
    train.add_argument("--no-context", action="store_true")
    train.add_argument("--no-modality-dropout", action="store_true")

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--features", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--missing", choices=["text", "audio", "vision"])
    evaluate.add_argument("--device", default="auto")

    analyze = commands.add_parser("analyze")
    analyze.add_argument("--video", required=True)
    analyze.add_argument("--checkpoint", required=True)
    analyze.add_argument("--yunet-model", required=True)
    analyze.add_argument("--language", choices=["auto", "zh", "en"], default="auto")
    analyze.add_argument("--output", default="artifacts/exports")
    analyze.add_argument("--device", default="auto")

    serve = commands.add_parser("serve")
    serve.add_argument("--checkpoint", required=True)
    serve.add_argument("--yunet-model", required=True)
    serve.add_argument("--device", default="auto")
    serve.add_argument("--share", action="store_true")
    return parser


def _runtime_analyzer(checkpoint_path: str, yunet_model: str, device_name: str) -> DialogueAnalyzer:
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("metadata", {}).get("model_config")
    if not model_config:
        raise ValueError("checkpoint does not contain model_config metadata")
    model = build_model(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    extractor_device = "cuda" if device.type == "cuda" else "cpu"
    face_cropper = YuNetFaceCropper(yunet_model)
    pipeline = PretrainedFeaturePipeline(
        text_extractor=TextFeatureExtractor(device=extractor_device),
        audio_extractor=AudioFeatureExtractor(device=extractor_device),
        vision_extractor=VisionFeatureExtractor(device=extractor_device),
        face_cropper=face_cropper,
    )
    return DialogueAnalyzer(
        transcriber=FasterWhisperTranscriber(device=extractor_device),
        feature_pipeline=pipeline,
        model=model,
        device=device,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-meld":
        records = []
        for split, csv_path, media_root in (
            ("train", args.train_csv, args.train_media),
            ("dev", args.dev_csv, args.dev_media),
            ("test", args.test_csv, args.test_media),
        ):
            records.extend(load_meld_csv(csv_path, media_root=media_root, split=split))
        if not args.allow_partial:
            check_official_split_counts("meld", count_records(records))
        write_manifest(records, args.output)
        return 0

    if args.command == "prepare-emotiontalk":
        records = []
        for split, source in (
            ("train", args.train_json),
            ("validation", args.validation_json),
            ("test", args.test_json),
        ):
            records.extend(
                load_emotiontalk_manifest(source, media_root=args.media_root, split=split)
            )
        if not args.allow_partial:
            check_official_split_counts("emotiontalk", count_records(records))
        write_manifest(records, args.output)
        return 0

    if args.command == "validate":
        records = read_manifest(args.manifest)
        report = validate_dataset_records(records)
        if args.official_counts:
            for dataset in {record.dataset for record in records}:
                check_official_split_counts(
                    dataset,
                    count_records(record for record in records if record.dataset == dataset),
                )
        print(json.dumps({
            "is_valid": report.is_valid,
            "split_counts": report.split_counts,
            "label_counts": report.label_counts,
            "duplicate_sample_ids": report.duplicate_sample_ids,
            "cross_split_media": report.cross_split_media,
        }, ensure_ascii=False, indent=2))
        return 0 if report.is_valid else 1

    if args.command == "extract-features":
        device = resolve_device(args.device)
        extractor_device = "cuda" if device.type == "cuda" else "cpu"
        records = [
            record
            for record in read_manifest(args.manifest)
            if (not args.dataset or record.dataset == args.dataset)
            and (not args.split or str(record.split) == args.split)
        ]
        text = TextFeatureExtractor(device=extractor_device)
        audio = AudioFeatureExtractor(device=extractor_device)
        vision = VisionFeatureExtractor(device=extractor_device)
        cropper = YuNetFaceCropper(args.yunet_model)

        def waveform_loader(path: Path):
            waveform = load_full_waveform(path)
            if args.audio_snr is not None and waveform.size:
                waveform = add_noise_at_snr(waveform, snr_db=args.audio_snr, seed=42)
            return waveform

        runner = DatasetFeatureExtractionRunner(
            text_extractor=text,
            audio_extractor=audio,
            waveform_loader=waveform_loader,
            vision_loader=lambda path: vision.encode_video(
                path,
                face_cropper=cropper,
                frame_drop_fraction=args.frame_drop,
                seed=42,
            ),
        )
        for dataset in sorted({record.dataset for record in records}):
            for split in sorted({str(record.split) for record in records if record.dataset == dataset}):
                group = [
                    record for record in records
                    if record.dataset == dataset and str(record.split) == split
                ]
                runner.run(group, FeatureStore(args.features), shard_size=args.shard_size)
        return 0

    if args.command == "train":
        result = run_experiment(
            manifest_path=args.manifest,
            feature_root=args.features,
            output_directory=args.output,
            config=ExperimentConfig(
                model=args.model,
                seed=args.seed,
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                patience=args.patience,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                use_language_embedding=not args.no_language,
                use_reliability_gates=not args.no_gates,
                use_context=not args.no_context,
                modality_dropout=0.0 if args.no_modality_dropout else 0.2,
            ),
            device_name=args.device,
        )
        print(result)
        return 0

    if args.command == "evaluate":
        result = evaluate_checkpoint(
            manifest_path=args.manifest,
            feature_root=args.features,
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            missing_modality=args.missing,
            device_name=args.device,
        )
        print(result)
        return 0

    analyzer = _runtime_analyzer(args.checkpoint, args.yunet_model, args.device)
    if args.command == "analyze":
        result = analyzer.analyze(Path(args.video), args.language)
        output = Path(args.output)
        export_analysis_json(result, output / "analysis.json")
        export_analysis_csv(result, output / "analysis.csv")
        print(output / "analysis.json")
        return 0
    if args.command == "serve":
        create_app(analyzer).launch(share=args.share)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
