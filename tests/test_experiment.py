import json

import numpy as np

from bimer.experiment import ExperimentConfig, evaluate_checkpoint, run_experiment
from bimer.feature_store import FeatureShard, FeatureStore
from bimer.manifest import write_manifest
from bimer.schema import UtteranceRecord


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
    assert set(payload["test"]) == {"meld", "emotiontalk"}
    assert (result_path.parent / "best.pt").exists()
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
