import numpy as np
import torch

from bimer.baselines import EarlyFusionContext, EarlyFusionMLP, UnimodalClassifier
from bimer.training import (
    BalancedDialogueSampler,
    DialogueExample,
    FitConfig,
    collate_dialogues,
    evaluate_batches,
    fit_model,
    train_epoch,
    validation_selection_score,
)


def _example(dataset: str, length: int, language_id: int) -> DialogueExample:
    generator = np.random.default_rng(length + language_id)
    return DialogueExample(
        dataset=dataset,
        sample_ids=tuple(f"{dataset}:{index}" for index in range(length)),
        text=generator.normal(size=(length, 4)).astype(np.float32),
        audio=generator.normal(size=(length, 6)).astype(np.float32),
        vision=generator.normal(size=(length, 5)).astype(np.float32),
        modality_mask=np.ones((length, 3), dtype=np.bool_),
        labels=np.arange(length, dtype=np.int64) % 2,
        language_id=language_id,
    )


def test_collate_dialogues_pads_features_and_attention_mask():
    batch = collate_dialogues([_example("meld", 2, 0), _example("emotiontalk", 3, 1)])
    assert batch.text_features.shape == (2, 3, 4)
    assert batch.audio_features.shape == (2, 3, 6)
    assert batch.attention_mask.tolist() == [[True, True, False], [True, True, True]]
    assert batch.language_ids.tolist() == [0, 1]
    assert batch.labels[0, 2].item() == -100


def test_balanced_sampler_alternates_datasets_deterministically():
    examples = [
        _example("meld", 1, 0),
        _example("meld", 1, 0),
        _example("emotiontalk", 1, 1),
    ]
    first = list(BalancedDialogueSampler(examples, seed=42))
    second = list(BalancedDialogueSampler(examples, seed=42))
    assert first == second
    assert [examples[index].dataset for index in first[:4]] == [
        "meld",
        "emotiontalk",
        "meld",
        "emotiontalk",
    ]


def test_required_baselines_share_the_training_output_contract():
    batch = collate_dialogues([_example("meld", 2, 0)])
    models = [
        UnimodalClassifier("text", input_dim=4, hidden_dim=8, num_classes=2),
        EarlyFusionMLP((4, 6, 5), hidden_dim=8, num_classes=2),
        EarlyFusionContext((4, 6, 5), hidden_dim=4, num_classes=2),
    ]
    for model in models:
        output = model(**batch.model_inputs())
        assert output.logits.shape == (1, 2, 2)
        assert output.gates.shape == (1, 2, 3)


def test_train_epoch_updates_model_parameters():
    batch = collate_dialogues([_example("meld", 3, 0), _example("emotiontalk", 3, 1)])
    model = EarlyFusionMLP((4, 6, 5), hidden_dim=8, num_classes=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    before = next(model.parameters()).detach().clone()
    loss = train_epoch(model, [batch], optimizer, device=torch.device("cpu"))
    after = next(model.parameters()).detach()
    assert np.isfinite(loss)
    assert not torch.equal(before, after)


def test_evaluation_ignores_padding_and_reports_gates():
    batch = collate_dialogues([_example("meld", 2, 0), _example("emotiontalk", 3, 1)])
    model = EarlyFusionMLP((4, 6, 5), hidden_dim=8, num_classes=2)
    report = evaluate_batches(
        model,
        [batch],
        device=torch.device("cpu"),
        label_names=("neutral", "joy"),
    )
    assert len(report.truth) == 5
    assert report.gates.shape == (5, 3)
    assert set(report.metrics) >= {"weighted_f1", "macro_f1", "accuracy"}


def test_validation_score_is_mean_of_bilingual_weighted_f1():
    assert validation_selection_score(
        {"meld": {"weighted_f1": 0.6}, "emotiontalk": {"weighted_f1": 0.8}}
    ) == 0.7


def test_fit_model_writes_best_checkpoint(tmp_path):
    batch = collate_dialogues([_example("meld", 3, 0), _example("emotiontalk", 3, 1)])
    model = EarlyFusionMLP((4, 6, 5), hidden_dim=8, num_classes=2)
    config = FitConfig(max_epochs=2, patience=1, learning_rate=0.01, weight_decay=0.0)
    history = fit_model(
        model,
        train_batches=[batch],
        validation_batches={"meld": [batch], "emotiontalk": [batch]},
        label_names=("neutral", "joy"),
        checkpoint_path=tmp_path / "best.pt",
        config=config,
        device=torch.device("cpu"),
    )
    assert (tmp_path / "best.pt").exists()
    assert history.best_epoch in {1, 2}
    assert len(history.epochs) >= 1
