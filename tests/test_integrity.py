from __future__ import annotations

from pathlib import Path

from bimer.integrity import (
    build_sha256_manifest,
    verify_sha256_manifest,
    write_sha256_manifest,
)


def test_manifest_uses_sorted_portable_paths_without_duplicates(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "z.txt").write_text("last", encoding="utf-8")
    (tmp_path / "nested" / "a.txt").write_text("first", encoding="utf-8")

    entries = build_sha256_manifest(
        root=tmp_path,
        inputs=[tmp_path / "z.txt", tmp_path / "nested", tmp_path / "nested" / "a.txt"],
    )

    assert [entry.path for entry in entries] == ["nested/a.txt", "z.txt"]
    assert entries[0].sha256 == (
        "a7937b64b8caa58f03721bb6bacf5c78cb235febe0e70b1b84cd99541461a08e"
    )


def test_written_manifest_can_be_verified_and_detects_changes(tmp_path: Path) -> None:
    payload = tmp_path / "result.json"
    payload.write_text('{"score": 0.60148}\n', encoding="utf-8")
    manifest = tmp_path / "evidence.sha256"

    write_sha256_manifest(
        destination=manifest,
        root=tmp_path,
        inputs=[payload],
        header=["BIMER private evidence", "Paths are relative to the project root."],
    )

    assert verify_sha256_manifest(manifest=manifest, root=tmp_path).ok
    text = manifest.read_text(encoding="utf-8")
    assert "# BIMER private evidence" in text
    assert "  result.json" in text

    payload.write_text('{"score": 0.0}\n', encoding="utf-8")
    result = verify_sha256_manifest(manifest=manifest, root=tmp_path)
    assert not result.ok
    assert result.mismatched == ("result.json",)


def test_verification_reports_missing_files(tmp_path: Path) -> None:
    payload = tmp_path / "checkpoint.pt"
    payload.write_bytes(b"checkpoint")
    manifest = tmp_path / "evidence.sha256"
    write_sha256_manifest(
        destination=manifest,
        root=tmp_path,
        inputs=[payload],
    )
    payload.unlink()

    result = verify_sha256_manifest(manifest=manifest, root=tmp_path)

    assert not result.ok
    assert result.missing == ("checkpoint.pt",)
