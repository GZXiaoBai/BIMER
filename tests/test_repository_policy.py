from __future__ import annotations

from pathlib import Path

from bimer.repository_policy import inspect_public_tree


def test_public_tree_rejects_private_paths_large_files_and_secrets(
    tmp_path: Path,
) -> None:
    safe = tmp_path / "src" / "module.py"
    safe.parent.mkdir()
    safe.write_text("value = 42\n", encoding="utf-8")
    private_data = tmp_path / "data" / "processed" / "all.jsonl"
    private_data.parent.mkdir(parents=True)
    private_data.write_text("{}\n", encoding="utf-8")
    secret = tmp_path / "config.py"
    secret.write_text(
        'TOKEN = "hf_abcdefghijklmnopqrstuvwxyz123456"\n',
        encoding="utf-8",
    )
    large = tmp_path / "model.bin"
    large.write_bytes(b"x" * 101)

    violations = inspect_public_tree(
        root=tmp_path,
        tracked_paths=[
            Path("src/module.py"),
            Path("data/processed/all.jsonl"),
            Path("config.py"),
            Path("model.bin"),
        ],
        maximum_bytes=100,
    )

    assert {(item.code, item.path.as_posix()) for item in violations} == {
        ("forbidden-path", "data/processed/all.jsonl"),
        ("secret-pattern", "config.py"),
        ("oversized-file", "model.bin"),
    }


def test_public_tree_allows_external_test_templates(tmp_path: Path) -> None:
    template = tmp_path / "data" / "templates" / "annotations.csv"
    template.parent.mkdir(parents=True)
    template.write_text("video_id,label\n", encoding="utf-8")

    violations = inspect_public_tree(
        root=tmp_path,
        tracked_paths=[Path("data/templates/annotations.csv")],
    )

    assert violations == ()
