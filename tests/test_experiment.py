import json
from dataclasses import replace

import numpy as np
import pytest

from bimer.experiment import (
    ExperimentConfig,
    _without_modalities,
    aggregate_seed_results,
    evaluate_checkpoint,
    run_experiment,
)
from bimer.feature_store import FeatureShard, FeatureStore
from bimer.manifest import read_manifest, write_manifest
from bimer.schema import UtteranceRecord
from bimer.training import DialogueExample


def _write_tiny_joint_data(tmp_path):
    records = []
    store = FeatureStore(tmp_path / "features")
    split_names = {
        "meld": ("train", "dev"),
        "emotiontalk": ("train", "validation"),
    }
    for dataset, splits in split_names.items():
        for split in splits:
            group = [
                UtteranceRecord(
                    dataset=dataset,
                    split=split,
                    dialogue_id=f"{dataset}-{split}",
                    utterance_id=index,
                    text="line",
                    emotion="neutral" if index == 0 else "joy",
                    language="en" if dataset == "meld" else "zh",
                    start_seconds=float(index),
                    end_seconds=float(index + 1),
                )
                for index in range(2)
            ]
            records.extend(group)
            store.write(
                dataset,
                split,
                0,
                FeatureShard(
                    sample_ids=np.asarray([record.sample_id for record in group]),
                    text=np.ones((2, 4), np.float32),
                    audio=np.ones((2, 6), np.float32),
                    vision=np.ones((2, 5), np.float32),
                    modality_mask=np.ones((2, 3), np.bool_),
                ),
            )
    return write_manifest(records, tmp_path / "manifest.jsonl"), tmp_path / "features"


def test_without_modalities_zeros_features_and_preserves_remaining_modality():
    example = DialogueExample(
        dataset="meld",
        sample_ids=("sample-1",),
        text=np.ones((1, 4), dtype=np.float32),
        audio=np.full((1, 6), 2.0, dtype=np.float32),
        vision=np.full((1, 5), 3.0, dtype=np.float32),
        modality_mask=np.ones((1, 3), dtype=np.bool_),
        labels=np.zeros(1, dtype=np.int64),
        language_id=0,
    )

    updated = _without_modalities([example], ("vision", "text"))[0]

    assert not updated.text.any()
    assert not updated.vision.any()
    np.testing.assert_array_equal(updated.audio, example.audio)
    np.testing.assert_array_equal(
        updated.modality_mask,
        np.array([[False, True, False]], dtype=np.bool_),
    )


def test_majority_experiment_runs_end_to_end_on_cached_features(tmp_path):
    records = []
    store = FeatureStore(tmp_path / "features")
    split_names = {
        "meld": ("train", "dev", "test"),
        "emotiontalk": ("train", "validation", "test"),
    }
    for dataset, splits in split_names.items():
        for split in splits:
            group = []
            for index in range(2):
                group.append(
                    UtteranceRecord(
                        dataset=dataset,
                        split=split,
                        dialogue_id=f"{dataset}-{split}",
                        utterance_id=index,
                        text="line",
                        emotion="neutral" if index == 0 else "joy",
                        language="en" if dataset == "meld" else "zh",
                        start_seconds=float(index),
                        end_seconds=float(index + 1),
                    )
                )
            records.extend(group)
            shard = FeatureShard(
                sample_ids=np.array([record.sample_id for record in group]),
                text=np.ones((2, 4), np.float32),
                audio=np.ones((2, 6), np.float32),
                vision=np.ones((2, 5), np.float32),
                modality_mask=np.ones((2, 3), np.bool_),
            )
            store.write(dataset, split, 0, shard)

    manifest = write_manifest(records, tmp_path / "manifest.jsonl")
    result_path = run_experiment(
        manifest_path=manifest,
        feature_root=tmp_path / "features",
        output_directory=tmp_path / "results",
        config=ExperimentConfig(model="majority", hidden_dim=8, bootstrap_iterations=20),
        device_name="cpu",
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    protocol_status = json.loads(
        (tmp_path / "results" / "_protocol" / "standard-majority-joint-seed-42.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol_status["status"] == "completed"
    assert protocol_status["result"] == str(result_path)
    assert set(payload["test"]) == {"meld", "emotiontalk"}
    assert payload["test"]["meld"]["bootstrap_unit"] == "context"
    assert (result_path.parent / "best.pt").exists()
    checkpoint = __import__("torch").load(
        result_path.parent / "best.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["metadata"]["model_config"]["use_input_normalization"] is True
    assert "normalizer.audio_mean" in checkpoint["model_state_dict"]
    prediction_path = result_path.parent / "predictions" / "meld.npz"
    assert prediction_path.exists()
    with np.load(prediction_path, allow_pickle=False) as predictions:
        assert predictions["sample_ids"].shape == (2,)
        assert predictions["context_ids"].shape == (2,)
        assert predictions["probabilities"].shape == (2, 7)
        assert predictions["modality_quality"].shape == (2, 3, 4)
        assert predictions["modality_available"].shape == (2, 3)
        assert predictions["gates"].shape == (2, 3)
    robustness_path = evaluate_checkpoint(
        manifest_path=manifest,
        feature_root=tmp_path / "features",
        checkpoint_path=result_path.parent / "best.pt",
        output_path=tmp_path / "missing-vision.json",
        missing_modality="vision",
        bootstrap_iterations=20,
        device_name="cpu",
    )
    robustness = json.loads(robustness_path.read_text(encoding="utf-8"))
    assert robustness["condition"] == "missing_vision"
    with np.load(
        tmp_path / "missing-vision.predictions" / "meld.npz",
        allow_pickle=False,
    ) as predictions:
        assert predictions["modality_quality"].shape == (2, 3, 4)
        assert not predictions["modality_available"][:, 2].any()

    two_missing_path = evaluate_checkpoint(
        manifest_path=manifest,
        feature_root=tmp_path / "features",
        checkpoint_path=result_path.parent / "best.pt",
        output_path=tmp_path / "missing-text-vision.json",
        missing_modality=("vision", "text"),
        bootstrap_iterations=20,
        device_name="cpu",
    )
    two_missing = json.loads(two_missing_path.read_text(encoding="utf-8"))
    assert two_missing["condition"] == "missing_text_vision"

    noisy_path = evaluate_checkpoint(
        manifest_path=manifest,
        feature_root=tmp_path / "features",
        checkpoint_path=result_path.parent / "best.pt",
        output_path=tmp_path / "audio-snr-10db.json",
        condition_name="audio_snr_10db",
        bootstrap_iterations=20,
        device_name="cpu",
    )
    noisy = json.loads(noisy_path.read_text(encoding="utf-8"))
    assert noisy["condition"] == "audio_snr_10db"
    assert noisy["manifest"] == str(manifest)
    assert noisy["feature_root"] == str(tmp_path / "features")

    with pytest.raises(ValueError, match="at least one modality must remain"):
        evaluate_checkpoint(
            manifest_path=manifest,
            feature_root=tmp_path / "features",
            checkpoint_path=result_path.parent / "best.pt",
            output_path=tmp_path / "missing-all.json",
            missing_modality=("text", "audio", "vision"),
            bootstrap_iterations=20,
            device_name="cpu",
        )


def test_single_dataset_scope_does_not_require_other_dataset(tmp_path):
    records = []
    store = FeatureStore(tmp_path / "features")
    for split in ("train", "validation", "test"):
        group = [
            UtteranceRecord(
                dataset="emotiontalk",
                split=split,
                dialogue_id=f"emotiontalk-{split}",
                utterance_id=index,
                text="line",
                emotion="neutral" if index == 0 else "joy",
                language="zh",
                start_seconds=float(index),
                end_seconds=float(index + 1),
            )
            for index in range(2)
        ]
        records.extend(group)
        store.write(
            "emotiontalk",
            split,
            0,
            FeatureShard(
                sample_ids=np.array([record.sample_id for record in group]),
                text=np.ones((2, 4), np.float32),
                audio=np.ones((2, 6), np.float32),
                vision=np.ones((2, 5), np.float32),
                modality_mask=np.ones((2, 3), np.bool_),
            ),
        )

    manifest = write_manifest(records, tmp_path / "manifest.jsonl")
    result_path = run_experiment(
        manifest_path=manifest,
        feature_root=tmp_path / "features",
        output_directory=tmp_path / "results",
        config=ExperimentConfig(
            model="majority",
            hidden_dim=8,
            bootstrap_iterations=20,
            training_scope="emotiontalk",
        ),
        device_name="cpu",
    )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert set(payload["test"]) == {"emotiontalk"}


def test_validation_screen_can_skip_all_test_evaluation(tmp_path):
    records = []
    store = FeatureStore(tmp_path / "features")
    for split in ("train", "validation", "test"):
        group = [
            UtteranceRecord(
                dataset="emotiontalk",
                split=split,
                dialogue_id=f"d-{split}",
                utterance_id=index,
                text="line",
                emotion="neutral" if index == 0 else "joy",
                language="zh",
                start_seconds=float(index),
                end_seconds=float(index + 1),
            )
            for index in range(2)
        ]
        records.extend(group)
        store.write(
            "emotiontalk",
            split,
            0,
            FeatureShard(
                sample_ids=np.array([record.sample_id for record in group]),
                text=np.ones((2, 4), np.float32),
                audio=np.ones((2, 6), np.float32),
                vision=np.ones((2, 5), np.float32),
                modality_mask=np.ones((2, 3), np.bool_),
            ),
        )

    result = run_experiment(
        manifest_path=write_manifest(records, tmp_path / "manifest.jsonl"),
        feature_root=tmp_path / "features",
        output_directory=tmp_path / "results",
        config=ExperimentConfig(
            model="majority",
            hidden_dim=8,
            bootstrap_iterations=20,
            training_scope="emotiontalk",
            evaluate_test=False,
        ),
        device_name="cpu",
    )

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["test"] == {}
    assert payload["evaluation_datasets"] == []
    assert not (result.parent / "predictions").exists()
    validation_prediction = result.parent / "validation_predictions" / "emotiontalk.npz"
    assert validation_prediction.exists()
    with np.load(validation_prediction, allow_pickle=False) as predictions:
        assert predictions["probabilities"].shape == (2, 7)
        assert predictions["languages"].tolist() == ["zh", "zh"]


def test_v4_screen_saves_context_and_prototype_evidence_without_test_access(tmp_path):
    manifest, features = _write_tiny_joint_data(tmp_path)

    result = run_experiment(
        manifest_path=manifest,
        feature_root=features,
        output_directory=tmp_path / "results",
        config=ExperimentConfig(
            model="adaptive_context_prototype",
            seed=42,
            batch_size=2,
            hidden_dim=8,
            max_epochs=1,
            min_epochs=1,
            patience=1,
            evaluate_test=False,
            protocol_stage="v4_screen",
            use_language_embedding=False,
            prototype_loss_weight=0.1,
            prototype_temperature=0.07,
        ),
        device_name="cpu",
    )

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["config"]["prototype_loss_weight"] == 0.1
    assert payload["test"] == {}
    for dataset in ("meld", "emotiontalk"):
        prediction_path = result.parent / "validation_predictions" / f"{dataset}.npz"
        with np.load(prediction_path, allow_pickle=False) as predictions:
            assert predictions["context_lengths"].tolist() == [2, 2]
            assert predictions["context_gates"].shape == (2,)
            assert predictions["prototype_logits"].shape == (2, 7)
            assert np.isfinite(predictions["context_gates"]).all()
            assert np.isfinite(predictions["prototype_logits"]).all()


def test_v4_protocol_stages_protect_official_test_access(tmp_path):
    with pytest.raises(ValueError, match="v4_screen is restricted to seed 42"):
        run_experiment(
            manifest_path=tmp_path / "missing.jsonl",
            feature_root=tmp_path / "missing-features",
            output_directory=tmp_path / "results",
            config=ExperimentConfig(
                seed=123,
                evaluate_test=False,
                protocol_stage="v4_screen",
            ),
            device_name="cpu",
        )
    with pytest.raises(ValueError, match="v4_screen must use --skip-test"):
        run_experiment(
            manifest_path=tmp_path / "missing.jsonl",
            feature_root=tmp_path / "missing-features",
            output_directory=tmp_path / "results",
            config=ExperimentConfig(
                seed=42,
                evaluate_test=True,
                protocol_stage="v4_screen",
            ),
            device_name="cpu",
        )
    with pytest.raises(ValueError, match="v4_formal must use --skip-test"):
        run_experiment(
            manifest_path=tmp_path / "missing.jsonl",
            feature_root=tmp_path / "missing-features",
            output_directory=tmp_path / "results",
            config=ExperimentConfig(
                evaluate_test=True,
                protocol_stage="v4_formal",
            ),
            device_name="cpu",
        )


def test_training_can_append_a_corrupted_feature_view(tmp_path):
    records = []
    clean_store = FeatureStore(tmp_path / "clean")
    for split in ("train", "validation"):
        group = [
            UtteranceRecord(
                dataset="emotiontalk",
                split=split,
                dialogue_id=f"d-{split}",
                utterance_id=index,
                text="line",
                emotion="neutral" if index == 0 else "joy",
                language="zh",
                start_seconds=float(index),
                end_seconds=float(index + 1),
            )
            for index in range(2)
        ]
        records.extend(group)
        clean_store.write(
            "emotiontalk",
            split,
            0,
            FeatureShard(
                sample_ids=np.asarray([record.sample_id for record in group]),
                text=np.ones((2, 4), np.float32),
                audio=np.ones((2, 6), np.float32),
                vision=np.ones((2, 5), np.float32),
                modality_mask=np.ones((2, 3), np.bool_),
            ),
        )
    selected = [record for record in records if str(record.split) == "train"]
    augmentation_manifest = write_manifest(selected, tmp_path / "aug.jsonl")
    augmentation_store = FeatureStore(tmp_path / "aug-features")
    augmentation_store.write(
        "emotiontalk",
        "train",
        0,
        FeatureShard(
            sample_ids=np.asarray([record.sample_id for record in selected]),
            text=np.ones((2, 4), np.float32),
            audio=np.full((2, 6), 2, np.float32),
            vision=np.ones((2, 5), np.float32),
            modality_mask=np.ones((2, 3), np.bool_),
            modality_quality=np.full((2, 3, 4), 0.5, np.float32),
        ),
    )

    result = run_experiment(
        manifest_path=write_manifest(records, tmp_path / "manifest.jsonl"),
        feature_root=tmp_path / "clean",
        output_directory=tmp_path / "results",
        config=ExperimentConfig(
            model="majority",
            hidden_dim=8,
            training_scope="emotiontalk",
            evaluate_test=False,
            augmentation_manifests=(str(augmentation_manifest),),
            augmentation_feature_roots=(str(tmp_path / "aug-features"),),
        ),
        device_name="cpu",
    )

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["training_views"]["clean_examples"] == 1
    assert payload["training_views"]["augmentations"][0]["examples"] == 1
    assert payload["training_views"]["total_examples"] == 2

    paired_result = run_experiment(
        manifest_path=tmp_path / "manifest.jsonl",
        feature_root=tmp_path / "clean",
        output_directory=tmp_path / "paired-results",
        config=ExperimentConfig(
            model="majority",
            hidden_dim=8,
            training_scope="emotiontalk",
            evaluate_test=False,
            augmentation_manifests=(str(augmentation_manifest),),
            augmentation_feature_roots=(str(tmp_path / "aug-features"),),
            augmentation_modalities=("audio",),
            augmentation_severities=(10.0,),
            gate_ranking_weight=0.1,
        ),
        device_name="cpu",
    )
    paired_payload = json.loads(paired_result.read_text(encoding="utf-8"))
    assert paired_payload["training_views"]["total_examples"] == 1
    assert paired_payload["training_views"]["paired_examples"] == 1


def test_experiment_config_records_v3_classification_objective(tmp_path):
    config = ExperimentConfig(
        classification_loss="balanced_softmax",
        focal_gamma=2.0,
    )

    assert config.classification_loss == "balanced_softmax"
    assert config.focal_gamma == 2.0


def test_experiment_config_records_paired_ranking_objective():
    config = ExperimentConfig(
        augmentation_modalities=("audio", "vision"),
        augmentation_severities=(10.0, 0.5),
        corrupted_classification_weight=0.5,
        gate_ranking_weight=0.1,
        gate_ranking_margin=0.1,
    )

    assert config.augmentation_modalities == ("audio", "vision")
    assert config.augmentation_severities == (10.0, 0.5)
    assert config.gate_ranking_weight == 0.1


def test_v3_screen_configuration_forbids_test_access(tmp_path):
    with pytest.raises(ValueError, match="must use --skip-test"):
        run_experiment(
            manifest_path=tmp_path / "not-read.jsonl",
            feature_root=tmp_path / "features",
            output_directory=tmp_path / "results",
            config=ExperimentConfig(
                protocol_stage="v3_screen",
                evaluate_test=True,
            ),
            device_name="cpu",
        )


def test_v3_ranking_rejects_missing_paired_augmentations_before_loading_data(
    tmp_path,
):
    with pytest.raises(ValueError, match="requires paired augmentations"):
        run_experiment(
            manifest_path=tmp_path / "not-read.jsonl",
            feature_root=tmp_path / "features",
            output_directory=tmp_path / "results",
            config=ExperimentConfig(
                gate_ranking_weight=0.1,
                evaluate_test=False,
            ),
            device_name="cpu",
        )


def test_v5_consistency_rejects_missing_paired_augmentations_before_loading_data(
    tmp_path,
):
    with pytest.raises(ValueError, match="ASR consistency requires paired augmentations"):
        run_experiment(
            manifest_path=tmp_path / "not-read.jsonl",
            feature_root=tmp_path / "features",
            output_directory=tmp_path / "results",
            config=ExperimentConfig(
                model="asr_consistent_quality_lagf",
                asr_consistency_weight=0.05,
                protocol_stage="v5_screen",
                evaluate_test=False,
            ),
            device_name="cpu",
        )


def test_v5_tiny_paired_experiment_saves_adapter_and_consistency_evidence(
    tmp_path,
):
    manifest, features = _write_tiny_joint_data(tmp_path)
    training_records = [
        replace(record, text="whisper text", text_source="whisper")
        for record in read_manifest(manifest)
        if str(record.split) == "train"
    ]
    paired_manifest = write_manifest(
        training_records,
        tmp_path / "paired-manifest.jsonl",
    )
    paired_store = FeatureStore(tmp_path / "paired-features")
    for dataset in ("meld", "emotiontalk"):
        group = [record for record in training_records if record.dataset == dataset]
        paired_store.write(
            dataset,
            "train",
            0,
            FeatureShard(
                sample_ids=np.asarray([record.sample_id for record in group]),
                text=-np.ones((len(group), 4), np.float32),
                audio=np.ones((len(group), 6), np.float32),
                vision=np.ones((len(group), 5), np.float32),
                modality_mask=np.ones((len(group), 3), np.bool_),
            ),
        )

    result = run_experiment(
        manifest_path=manifest,
        feature_root=features,
        output_directory=tmp_path / "v5-results",
        config=ExperimentConfig(
            model="asr_consistent_quality_lagf",
            hidden_dim=8,
            batch_size=2,
            max_epochs=1,
            min_epochs=1,
            patience=1,
            modality_dropout=0.0,
            use_language_embedding=False,
            evaluate_test=False,
            protocol_stage="v5_screen",
            augmentation_manifests=(str(paired_manifest),),
            augmentation_feature_roots=(str(paired_store.root),),
            augmentation_modalities=("text",),
            augmentation_severities=(1.0,),
            asr_consistency_weight=0.05,
        ),
        device_name="cpu",
    )

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["config"]["asr_consistency_weight"] == 0.05
    assert payload["history"]["epochs"][0]["asr_consistency_loss"] >= 0.0
    checkpoint = __import__("torch").load(
        result.parent / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert any("text_adapter" in name for name in checkpoint["model_state_dict"])


def test_checkpoint_evaluation_can_target_validation_without_test(tmp_path):
    records = []
    store = FeatureStore(tmp_path / "features")
    for split in ("train", "validation"):
        group = [
            UtteranceRecord(
                dataset="emotiontalk",
                split=split,
                dialogue_id=f"d-{split}",
                utterance_id=index,
                text="line",
                emotion="neutral",
                language="zh",
                start_seconds=float(index),
                end_seconds=float(index + 1),
            )
            for index in range(2)
        ]
        records.extend(group)
        store.write(
            "emotiontalk",
            split,
            0,
            FeatureShard(
                sample_ids=np.asarray([record.sample_id for record in group]),
                text=np.ones((2, 4), np.float32),
                audio=np.ones((2, 6), np.float32),
                vision=np.ones((2, 5), np.float32),
                modality_mask=np.ones((2, 3), np.bool_),
            ),
        )
    manifest = write_manifest(records, tmp_path / "manifest.jsonl")
    trained = run_experiment(
        manifest_path=manifest,
        feature_root=store.root,
        output_directory=tmp_path / "results",
        config=ExperimentConfig(
            model="majority",
            training_scope="emotiontalk",
            evaluate_test=False,
        ),
        device_name="cpu",
    )

    evaluated = evaluate_checkpoint(
        manifest_path=manifest,
        feature_root=store.root,
        checkpoint_path=trained.parent / "best.pt",
        output_path=tmp_path / "validation.json",
        evaluation_role="validation",
        bootstrap_iterations=20,
        device_name="cpu",
    )

    payload = json.loads(evaluated.read_text(encoding="utf-8"))
    assert set(payload["validation"]) == {"emotiontalk"}
    assert "test" not in payload


def test_aggregate_seed_results_supports_single_dataset_runs(tmp_path):
    paths = []
    for index, score in enumerate((0.4, 0.6, 0.5)):
        path = tmp_path / f"seed-{index}.json"
        path.write_text(
            json.dumps(
                {
                    "test": {
                        "meld": {
                            "weighted_f1": score,
                            "macro_f1": score - 0.1,
                            "accuracy": score + 0.1,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)

    output = aggregate_seed_results(paths, tmp_path / "summary.json")
    summary = json.loads(output.read_text(encoding="utf-8"))

    assert set(summary["datasets"]) == {"meld"}
    assert summary["datasets"]["meld"]["weighted_f1"]["mean"] == 0.5


def test_evaluation_windows_do_not_repeat_overlapping_context_samples(tmp_path):
    records = []
    store = FeatureStore(tmp_path / "features")
    split_names = {
        "meld": ("train", "dev", "test"),
        "emotiontalk": ("train", "validation", "test"),
    }
    for dataset, splits in split_names.items():
        for split in splits:
            group = [
                UtteranceRecord(
                    dataset=dataset,
                    split=split,
                    dialogue_id=f"{dataset}-{split}",
                    utterance_id=index,
                    text="line",
                    emotion="neutral" if index % 2 == 0 else "joy",
                    language="en" if dataset == "meld" else "zh",
                    start_seconds=float(index),
                    end_seconds=float(index + 1),
                )
                for index in range(40)
            ]
            records.extend(group)
            store.write(
                dataset,
                split,
                0,
                FeatureShard(
                    sample_ids=np.array([record.sample_id for record in group]),
                    text=np.ones((40, 4), np.float32),
                    audio=np.ones((40, 6), np.float32),
                    vision=np.ones((40, 5), np.float32),
                    modality_mask=np.ones((40, 3), np.bool_),
                ),
            )

    manifest = write_manifest(records, tmp_path / "manifest.jsonl")
    result_path = run_experiment(
        manifest_path=manifest,
        feature_root=tmp_path / "features",
        output_directory=tmp_path / "results",
        config=ExperimentConfig(
            model="majority",
            hidden_dim=8,
            bootstrap_iterations=20,
            training_scope="emotiontalk",
        ),
        device_name="cpu",
    )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert set(payload["test"]) == {"meld", "emotiontalk"}
    assert payload["training_datasets"] == ["emotiontalk"]
    assert payload["selection_datasets"] == ["emotiontalk"]
    assert payload["evaluation_datasets"] == ["meld", "emotiontalk"]
    confusion = payload["test"]["emotiontalk"]["confusion_matrix"]
    assert sum(sum(row) for row in confusion) == 40

    robustness_path = evaluate_checkpoint(
        manifest_path=manifest,
        feature_root=tmp_path / "features",
        checkpoint_path=result_path.parent / "best.pt",
        output_path=tmp_path / "evaluation.json",
        bootstrap_iterations=20,
        device_name="cpu",
    )
    robustness = json.loads(robustness_path.read_text(encoding="utf-8"))
    for dataset in ("meld", "emotiontalk"):
        confusion = robustness["test"][dataset]["confusion_matrix"]
        assert sum(sum(row) for row in confusion) == 40
