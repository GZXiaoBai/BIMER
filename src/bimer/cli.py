from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path

import torch

from .app import create_app
from .asr_manifest import write_asr_manifest_incrementally
from .corruption_sampling import (
    materialize_feature_subset,
    select_stratified_context_records,
)
from .data_adapters import (
    check_official_split_counts,
    count_records,
    load_emotiontalk_manifest,
    load_emotiontalk_official_csv,
    load_meld_csv,
)
from .deployment import DeploymentManifest, verify_deployment
from .experiment import ExperimentConfig, evaluate_checkpoint, resolve_device, run_experiment
from .export import export_analysis_csv, export_analysis_figure, export_analysis_json
from .feature_extraction_runner import DatasetFeatureExtractionRunner, load_full_waveform
from .feature_extractors import (
    AudioFeatureExtractor,
    TextFeatureExtractor,
    VisionFeatureExtractor,
    YuNetFaceCropper,
)
from .feature_statistics import compute_feature_statistics, write_feature_statistics
from .feature_store import FeatureStore
from .feature_verification import verify_feature_range, write_range_completion
from .inference import (
    FasterWhisperTranscriber,
)
from .integrity import verify_sha256_manifest
from .manifest import read_manifest, write_manifest
from .modality_store import seed_staging_from_base_shard
from .overfit_smoke import run_unimodal_overfit_smoke, write_overfit_smoke
from .parallel_feature_extraction import (
    ParallelFeatureExtractionConfig,
    ParallelFeatureExtractionRunner,
    initialize_vision_worker,
    load_waveform_worker,
    measure_audio_quality_worker,
    measure_video_quality_worker,
    prepare_video_quality_worker,
    record_shards,
)
from .quality_attachment import QualityAttachmentRunner
from .robustness import add_noise_at_snr, write_condition_provenance
from .runtime import build_legacy_runtime, build_runtime
from .shard_ranges import slice_shard_range
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

    emotiontalk_official = commands.add_parser("prepare-emotiontalk-official")
    emotiontalk_official.add_argument("--labels-csv", required=True)
    emotiontalk_official.add_argument("--transcriptions-csv", required=True)
    emotiontalk_official.add_argument("--media-root", required=True)
    emotiontalk_official.add_argument("--output", required=True)
    emotiontalk_official.add_argument("--allow-partial", action="store_true")

    asr_manifest = commands.add_parser("asr-manifest")
    asr_manifest.add_argument("--manifest", required=True)
    asr_manifest.add_argument("--output", required=True)
    asr_manifest.add_argument("--dataset", choices=["meld", "emotiontalk"])
    asr_manifest.add_argument("--split")
    asr_manifest.add_argument("--device", default="cpu")
    asr_manifest.add_argument("--keep-original-on-error", action="store_true")
    asr_manifest.add_argument("--error-log")

    validate = commands.add_parser("validate")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--official-counts", action="store_true")

    verify_features = commands.add_parser("verify-features")
    verify_features.add_argument("--manifest", required=True)
    verify_features.add_argument("--features", required=True)
    verify_features.add_argument("--dataset", choices=["meld", "emotiontalk"], required=True)
    verify_features.add_argument("--split", required=True)
    verify_features.add_argument("--shard-size", type=int, required=True)
    verify_features.add_argument("--start-shard", type=int)
    verify_features.add_argument("--end-shard", type=int)
    verify_features.add_argument("--write-completion", action="store_true")

    feature_stats = commands.add_parser("feature-stats")
    feature_stats.add_argument("--manifest", required=True)
    feature_stats.add_argument("--features", required=True)
    feature_stats.add_argument("--dataset", choices=["meld", "emotiontalk"], required=True)
    feature_stats.add_argument("--split", required=True)
    feature_stats.add_argument("--output", required=True)

    overfit_smoke = commands.add_parser("overfit-smoke")
    overfit_smoke.add_argument("--manifest", required=True)
    overfit_smoke.add_argument("--features", required=True)
    overfit_smoke.add_argument("--dataset", choices=["meld", "emotiontalk"], required=True)
    overfit_smoke.add_argument("--split", required=True)
    overfit_smoke.add_argument("--output", required=True)
    overfit_smoke.add_argument("--modality", choices=["text", "audio", "vision"], action="append")
    overfit_smoke.add_argument("--sample-count", type=int, default=16)
    overfit_smoke.add_argument("--max-epochs", type=int, default=200)
    overfit_smoke.add_argument("--learning-rate", type=float, default=1e-2)
    overfit_smoke.add_argument("--target-accuracy", type=float, default=0.95)
    overfit_smoke.add_argument("--hidden-dim", type=int, default=64)
    overfit_smoke.add_argument("--seed", type=int, default=42)
    overfit_smoke.add_argument("--device", default="auto")

    extract = commands.add_parser("extract-features")
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--features", required=True)
    extract.add_argument("--base-features")
    extract.add_argument("--yunet-model", required=True)
    extract.add_argument("--dataset", choices=["meld", "emotiontalk"])
    extract.add_argument("--split")
    extract.add_argument("--device", default="auto")
    extract.add_argument("--shard-size", type=int, default=1024)
    extract.add_argument("--audio-snr", type=float)
    extract.add_argument("--frame-drop", type=float, default=0.0)
    extract.add_argument("--mode", choices=["serial", "parallel"], default="serial")
    extract.add_argument("--text-audio-device", default="cuda:0")
    extract.add_argument("--vision-device", default="cuda:1")
    extract.add_argument("--text-batch-size", type=int, default=64)
    extract.add_argument("--audio-batch-size", type=int, default=8)
    extract.add_argument("--vision-batch-size", type=int, default=8)
    extract.add_argument("--audio-workers", type=int, default=4)
    extract.add_argument("--vision-workers", type=int, default=4)
    extract.add_argument("--queue-capacity", type=int, default=8)
    extract.add_argument("--staging")
    extract.add_argument("--only-modality", choices=["text", "audio", "vision"])
    extract.add_argument("--condition-name")
    extract.add_argument("--start-shard", type=int)
    extract.add_argument("--end-shard", type=int)

    train = commands.add_parser("train")
    train.add_argument("--manifest", required=True)
    train.add_argument("--features", required=True)
    train.add_argument("--output", required=True)
    train.add_argument(
        "--model",
        choices=[
            "majority",
            "text",
            "audio",
            "vision",
            "early_mlp",
            "early_context",
            "lagf",
            "quality_lagf",
        ],
        default="lagf",
    )
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--max-epochs", type=int, default=50)
    train.add_argument("--min-epochs", type=int, default=15)
    train.add_argument("--patience", type=int, default=7)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=1e-2)
    train.add_argument("--device", default="auto")
    train.add_argument(
        "--training-scope",
        choices=["joint", "meld", "emotiontalk"],
        default="joint",
    )
    train.add_argument("--no-language", action="store_true")
    train.add_argument("--no-gates", action="store_true")
    train.add_argument("--no-context", action="store_true")
    train.add_argument("--no-quality", action="store_true")
    train.add_argument("--no-modality-dropout", action="store_true")
    train.add_argument("--augmentation-manifest", action="append", default=[])
    train.add_argument("--augmentation-features", action="append", default=[])
    train.add_argument(
        "--augmentation-modality",
        action="append",
        choices=["text", "audio", "vision"],
        default=[],
    )
    train.add_argument(
        "--augmentation-severity",
        action="append",
        type=float,
        default=[],
    )
    train.add_argument(
        "--classification-loss",
        choices=["weighted_ce", "balanced_softmax", "focal"],
        default="weighted_ce",
    )
    train.add_argument("--focal-gamma", type=float, default=2.0)
    train.add_argument("--corrupted-classification-weight", type=float, default=0.5)
    train.add_argument("--gate-ranking-weight", type=float, default=0.0)
    train.add_argument("--gate-ranking-margin", type=float, default=0.10)
    train.add_argument(
        "--v3-screen",
        action="store_true",
        help="seed-42 validation screening; requires --skip-test",
    )
    train.add_argument(
        "--v3-formal",
        action="store_true",
        help="frozen V3 formal training; test evaluation remains a separate guarded stage",
    )
    train.add_argument(
        "--skip-test",
        action="store_true",
        help="validation-only run for model selection; never load or evaluate test splits",
    )

    corruption = commands.add_parser("sample-corruption-manifest")
    corruption.add_argument("--manifest", required=True)
    corruption.add_argument("--output-manifest", required=True)
    corruption.add_argument("--base-features", required=True)
    corruption.add_argument("--output-features", required=True)
    corruption.add_argument("--fraction", type=float, default=0.1)
    corruption.add_argument("--seed", type=int, default=42)
    corruption.add_argument("--shard-size", type=int, default=1024)
    corruption.add_argument(
        "--dataset",
        choices=["meld", "emotiontalk"],
        help="optionally build a dataset-local corruption subset",
    )

    attach_quality = commands.add_parser("attach-quality")
    attach_quality.add_argument("--manifest", required=True)
    attach_quality.add_argument("--base-features", required=True)
    attach_quality.add_argument("--output-features", required=True)
    attach_quality.add_argument("--yunet-model", required=True)
    attach_quality.add_argument("--dataset", choices=["meld", "emotiontalk"], required=True)
    attach_quality.add_argument("--split", required=True)
    attach_quality.add_argument("--workers", type=int, default=4)
    attach_quality.add_argument("--queue-capacity", type=int, default=8)
    attach_quality.add_argument("--start-shard", type=int)
    attach_quality.add_argument("--end-shard", type=int)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--features", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument(
        "--missing",
        choices=["text", "audio", "vision"],
        action="append",
    )
    evaluate.add_argument("--condition-name")
    evaluate.add_argument(
        "--role",
        choices=["validation", "test"],
        default="test",
    )
    evaluate.add_argument("--device", default="auto")

    analyze = commands.add_parser("analyze")
    analyze.add_argument("--video", required=True)
    analyze.add_argument("--deployment")
    analyze.add_argument("--artifact-root", default=".")
    analyze.add_argument("--online", action="store_true")
    analyze.add_argument("--checkpoint")
    analyze.add_argument("--yunet-model")
    analyze.add_argument("--language", choices=["auto", "zh", "en"], default="auto")
    analyze.add_argument("--output", default="artifacts/exports")
    analyze.add_argument("--device", default="auto")
    analyze.add_argument("--text-model", default="xlm-roberta-base")
    analyze.add_argument("--audio-model", default="facebook/wav2vec2-xls-r-300m")
    analyze.add_argument("--whisper-model", default="small")
    analyze.add_argument("--calibration")
    analyze.add_argument("--cache-dir", default="artifacts/runtime-cache")
    analyze.add_argument("--model-version", default="auto")

    serve = commands.add_parser("serve")
    serve.add_argument("--deployment")
    serve.add_argument("--artifact-root", default=".")
    serve.add_argument("--online", action="store_true")
    serve.add_argument("--checkpoint")
    serve.add_argument("--yunet-model")
    serve.add_argument("--device", default="auto")
    serve.add_argument("--text-model", default="xlm-roberta-base")
    serve.add_argument("--audio-model", default="facebook/wav2vec2-xls-r-300m")
    serve.add_argument("--whisper-model", default="small")
    serve.add_argument("--calibration")
    serve.add_argument("--cache-dir", default="artifacts/runtime-cache")
    serve.add_argument("--model-version", default="auto")
    serve.add_argument("--share", action="store_true")

    doctor = commands.add_parser("doctor")
    doctor.add_argument("--deployment", required=True)
    doctor.add_argument("--artifact-root", default=".")
    doctor.add_argument("--offline", action="store_true")

    verify_evidence = commands.add_parser("verify-evidence")
    verify_evidence.add_argument("--manifest", required=True)
    verify_evidence.add_argument("--root", default=".")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "doctor":
        manifest = DeploymentManifest.load(args.deployment)
        report = verify_deployment(
            manifest,
            artifact_root=Path(args.artifact_root),
            offline=args.offline,
        )
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return 0 if report.ok else 1

    if args.command == "verify-evidence":
        result = verify_sha256_manifest(
            manifest=Path(args.manifest),
            root=Path(args.root),
        )
        print(
            json.dumps(
                {
                    "ok": result.ok,
                    "missing": list(result.missing),
                    "mismatched": list(result.mismatched),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if result.ok else 1

    if args.command == "sample-corruption-manifest":
        training_records = [
            record
            for record in read_manifest(args.manifest)
            if str(record.split) == "train"
            and (args.dataset is None or record.dataset == args.dataset)
        ]
        selected = select_stratified_context_records(
            training_records,
            fraction=args.fraction,
            seed=args.seed,
        )
        write_manifest(selected, args.output_manifest)
        materialize_feature_subset(
            selected,
            FeatureStore(args.base_features),
            FeatureStore(args.output_features),
            shard_size=args.shard_size,
        )
        print(args.output_manifest)
        return 0

    if args.command == "attach-quality":
        if (args.start_shard is None) != (args.end_shard is None):
            raise ValueError("start-shard and end-shard must be supplied together")
        records = [
            record
            for record in read_manifest(args.manifest)
            if record.dataset == args.dataset and str(record.split) == args.split
        ]
        spawn_context = multiprocessing.get_context("spawn")
        runner = QualityAttachmentRunner(
            audio_quality_loader=measure_audio_quality_worker,
            vision_quality_loader=measure_video_quality_worker,
            audio_executor_factory=partial(
                ProcessPoolExecutor,
                mp_context=spawn_context,
            ),
            vision_executor_factory=partial(
                ProcessPoolExecutor,
                initializer=initialize_vision_worker,
                initargs=(Path(args.yunet_model), 0.0, 42),
                mp_context=spawn_context,
            ),
            workers=args.workers,
            queue_capacity=args.queue_capacity,
        )
        written = runner.run(
            records,
            FeatureStore(args.base_features),
            FeatureStore(args.output_features),
            start_shard=args.start_shard,
            end_shard=args.end_shard,
        )
        print(f"quality shards: {len(written)}")
        return 0
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

    if args.command == "prepare-emotiontalk-official":
        records = load_emotiontalk_official_csv(
            args.labels_csv,
            args.transcriptions_csv,
            media_root=args.media_root,
        )
        if not args.allow_partial:
            check_official_split_counts("emotiontalk", count_records(records))
        write_manifest(records, args.output)
        return 0

    if args.command == "asr-manifest":
        transcriber = FasterWhisperTranscriber(device=args.device)
        records = [
            record
            for record in read_manifest(args.manifest)
            if (not args.dataset or record.dataset == args.dataset)
            and (not args.split or str(record.split) == args.split)
        ]
        error_options = {}
        if args.keep_original_on_error:
            error_options = {
                "keep_original_on_error": True,
                "error_path": args.error_log,
            }
        write_asr_manifest_incrementally(
            records,
            transcriber,
            args.output,
            **error_options,
        )
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
        print(
            json.dumps(
                {
                    "is_valid": report.is_valid,
                    "split_counts": report.split_counts,
                    "label_counts": report.label_counts,
                    "duplicate_sample_ids": report.duplicate_sample_ids,
                    "cross_split_media": report.cross_split_media,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if report.is_valid else 1

    if args.command == "verify-features":
        group = [
            record
            for record in read_manifest(args.manifest)
            if record.dataset == args.dataset and str(record.split) == args.split
        ]
        selected, resolved = slice_shard_range(
            group,
            args.shard_size,
            args.start_shard,
            args.end_shard,
        )
        result = verify_feature_range(
            selected,
            FeatureStore(args.features),
            shard_size=args.shard_size,
            shard_index_offset=resolved.start,
            total_shards=resolved.total_shards,
        )
        if args.write_completion:
            write_range_completion(result, args.features)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    if args.command == "feature-stats":
        report = compute_feature_statistics(
            read_manifest(args.manifest),
            FeatureStore(args.features),
            dataset=args.dataset,
            split=args.split,
        )
        write_feature_statistics(report, args.output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "overfit-smoke":
        report = run_unimodal_overfit_smoke(
            read_manifest(args.manifest),
            FeatureStore(args.features),
            dataset=args.dataset,
            split=args.split,
            modalities=tuple(args.modality or ("text", "audio", "vision")),
            sample_count=args.sample_count,
            max_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            target_accuracy=args.target_accuracy,
            hidden_dim=args.hidden_dim,
            seed=args.seed,
            device=resolve_device(args.device),
        )
        write_overfit_smoke(report, args.output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["all_passed"] else 1

    if args.command == "extract-features":
        if bool(args.only_modality) != bool(args.base_features):
            raise ValueError("only-modality and base-features must be supplied together")
        if bool(args.condition_name) != bool(args.only_modality):
            raise ValueError("condition-name is required for single-modality replacement")
        if args.only_modality and args.mode != "parallel":
            raise ValueError("single-modality replacement requires parallel mode")
        if (
            args.base_features
            and Path(args.base_features).resolve() == Path(args.features).resolve()
        ):
            raise ValueError("base-features and features must be different roots")
        range_requested = args.start_shard is not None or args.end_shard is not None
        if range_requested:
            if args.start_shard is None or args.end_shard is None:
                raise ValueError("start-shard and end-shard must be supplied together")
            if args.mode != "parallel":
                raise ValueError("shard ranges require parallel mode")
            if not args.dataset or not args.split:
                raise ValueError("shard ranges require explicit dataset and split")
        records = [
            record
            for record in read_manifest(args.manifest)
            if (not args.dataset or record.dataset == args.dataset)
            and (not args.split or str(record.split) == args.split)
        ]
        if args.mode == "parallel":
            requested_cuda_indices = []
            requested_devices = []
            if args.only_modality in (None, "text", "audio"):
                requested_devices.append(args.text_audio_device)
            if args.only_modality in (None, "vision"):
                requested_devices.append(args.vision_device)
            for device_name in requested_devices:
                if device_name.startswith("cuda:"):
                    requested_cuda_indices.append(int(device_name.split(":", 1)[1]))
            if requested_cuda_indices and torch.cuda.device_count() <= max(requested_cuda_indices):
                raise RuntimeError(
                    "parallel extraction requested unavailable CUDA devices: "
                    f"{args.text_audio_device}, {args.vision_device}"
                )
            spawn_context = multiprocessing.get_context("spawn")
            audio_executor_factory = partial(
                ProcessPoolExecutor,
                mp_context=spawn_context,
            )
            vision_executor_factory = partial(
                ProcessPoolExecutor,
                initializer=initialize_vision_worker,
                initargs=(Path(args.yunet_model), args.frame_drop, 42),
                mp_context=spawn_context,
            )
            final_store = FeatureStore(args.features)
            base_store = FeatureStore(args.base_features) if args.base_features else None
            for dataset in sorted({record.dataset for record in records}):
                for split in sorted(
                    {str(record.split) for record in records if record.dataset == dataset}
                ):
                    group = [
                        record
                        for record in records
                        if record.dataset == dataset and str(record.split) == split
                    ]
                    if args.only_modality:
                        write_condition_provenance(
                            Path(args.features) / dataset / split,
                            {
                                "condition": args.condition_name,
                                "recompute_modality": args.only_modality,
                                "manifest": str(Path(args.manifest).resolve()),
                                "base_features": str(Path(args.base_features).resolve()),
                                "dataset": dataset,
                                "split": split,
                                "audio_snr": args.audio_snr,
                                "frame_drop": args.frame_drop,
                                "perturbation_seed": 42,
                            },
                        )
                    selected, resolved = slice_shard_range(
                        group,
                        args.shard_size,
                        args.start_shard,
                        args.end_shard,
                    )
                    config = ParallelFeatureExtractionConfig(
                        shard_size=args.shard_size,
                        text_batch_size=args.text_batch_size,
                        audio_batch_size=args.audio_batch_size,
                        vision_batch_size=args.vision_batch_size,
                        audio_workers=args.audio_workers,
                        vision_workers=args.vision_workers,
                        queue_capacity=args.queue_capacity,
                        shard_index_offset=resolved.start,
                    )
                    staging_root = Path(args.staging or args.features)
                    if base_store is not None:
                        for shard_index, _, sample_ids in record_shards(
                            selected,
                            args.shard_size,
                            resolved.start,
                        ):
                            seed_staging_from_base_shard(
                                base_store=base_store,
                                staging_root=staging_root,
                                dataset=dataset,
                                split=split,
                                shard_index=shard_index,
                                expected_sample_ids=sample_ids,
                                recompute_modality=args.only_modality,
                            )
                    runner = ParallelFeatureExtractionRunner(
                        staging_root=staging_root,
                        config=config,
                        text_extractor_factory=lambda: TextFeatureExtractor(
                            device=args.text_audio_device
                        ),
                        audio_extractor_factory=lambda: AudioFeatureExtractor(
                            device=args.text_audio_device
                        ),
                        vision_extractor_factory=lambda: VisionFeatureExtractor(
                            device=args.vision_device
                        ),
                        waveform_loader=partial(
                            load_waveform_worker,
                            audio_snr=args.audio_snr,
                            seed=42,
                        ),
                        prepared_loader=prepare_video_quality_worker,
                        audio_executor_factory=audio_executor_factory,
                        vision_executor_factory=vision_executor_factory,
                    )
                    runner.run(selected, final_store)
            return 0

        device = resolve_device(args.device)
        extractor_device = "cuda" if device.type == "cuda" else "cpu"
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
            vision_loader=lambda path: vision.encode_video_with_quality(
                path,
                face_cropper=cropper,
                frame_drop_fraction=args.frame_drop,
                seed=42,
            ),
        )
        for dataset in sorted({record.dataset for record in records}):
            for split in sorted(
                {str(record.split) for record in records if record.dataset == dataset}
            ):
                group = [
                    record
                    for record in records
                    if record.dataset == dataset and str(record.split) == split
                ]
                runner.run(group, FeatureStore(args.features), shard_size=args.shard_size)
        return 0

    if args.command == "train":
        if args.v3_screen and args.v3_formal:
            raise ValueError("v3-screen and v3-formal are mutually exclusive")
        result = run_experiment(
            manifest_path=args.manifest,
            feature_root=args.features,
            output_directory=args.output,
            config=ExperimentConfig(
                model=args.model,
                seed=args.seed,
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                min_epochs=args.min_epochs,
                patience=args.patience,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                use_language_embedding=not args.no_language,
                use_reliability_gates=not args.no_gates,
                use_context=not args.no_context,
                use_quality_input=not args.no_quality,
                modality_dropout=0.0 if args.no_modality_dropout else 0.2,
                training_scope=args.training_scope,
                evaluate_test=not args.skip_test,
                augmentation_manifests=tuple(args.augmentation_manifest),
                augmentation_feature_roots=tuple(args.augmentation_features),
                classification_loss=args.classification_loss,
                focal_gamma=args.focal_gamma,
                augmentation_modalities=tuple(args.augmentation_modality),
                augmentation_severities=tuple(args.augmentation_severity),
                corrupted_classification_weight=args.corrupted_classification_weight,
                gate_ranking_weight=args.gate_ranking_weight,
                gate_ranking_margin=args.gate_ranking_margin,
                protocol_stage=(
                    "v3_screen"
                    if args.v3_screen
                    else ("v3_formal" if args.v3_formal else "standard")
                ),
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
            condition_name=args.condition_name,
            device_name=args.device,
            evaluation_role=args.role,
        )
        print(result)
        return 0

    if args.deployment:
        analyzer = build_runtime(
            args.deployment,
            artifact_root=args.artifact_root,
            device_name=args.device,
            offline=not args.online,
        )
    else:
        if not args.checkpoint or not args.yunet_model:
            raise SystemExit(
                "analyze/serve requires --deployment or both --checkpoint and --yunet-model"
            )
        analyzer = build_legacy_runtime(
            checkpoint_path=args.checkpoint,
            yunet_path=args.yunet_model,
            device_name=args.device,
            text_model=args.text_model,
            audio_model=args.audio_model,
            whisper_model=args.whisper_model,
            calibration_path=args.calibration,
            cache_directory=args.cache_dir,
            model_version=args.model_version,
        )
    if args.command == "analyze":
        result = analyzer.analyze(Path(args.video), args.language)
        output = Path(args.output)
        export_started = time.perf_counter()
        export_analysis_csv(result, output / "analysis.csv")
        export_analysis_figure(result, output / "analysis.png")
        runtime_profile = dict(result.runtime_profile)
        runtime_profile["export"] = time.perf_counter() - export_started
        result = replace(result, runtime_profile=runtime_profile)
        export_analysis_json(result, output / "analysis.json")
        print(output / "analysis.json")
        return 0
    if args.command == "serve":
        create_app(analyzer).launch(share=args.share)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
