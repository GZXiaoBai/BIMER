from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_validation_view_script_uses_only_official_validation_splits():
    text = (ROOT / "scripts" / "prepare_v3_validation_views.sh").read_text(
        encoding="utf-8"
    )

    assert '"meld dev" "emotiontalk validation"' in text
    assert "--audio-snr 10" in text
    assert "--frame-drop 0.5" in text
    assert 'SHARD_SIZE="${BIMER_SHARD_SIZE:-16}"' in text
    assert text.count('--shard-size "$SHARD_SIZE"') == 3
    assert "validation-whisper.jsonl" in text
    assert '"test_records_used": False' in text


def test_v3_autodl_runner_has_budgets_archive_hash_and_shutdown_order():
    text = (ROOT / "scripts" / "run_v3_autodl.sh").read_text(encoding="utf-8")

    assert "43200" in text
    assert "28800" in text
    assert "64800" in text
    assert "timeout --signal=TERM" in text
    assert "sha256sum" in text
    assert "AUTODL_AUTO_SHUTDOWN" in text
    assert text.index("sha256sum") < text.index("shutdown -h now")
    assert "DOWNLOAD_READY" in text
    assert "overfit-smoke" in text
    assert "--sample-count 16" in text
    assert 'rm -f \\\n  "$OUTPUT/_status/STAGE_SUCCESS"' in text


def test_local_archive_verification_precedes_cleanup_authorization():
    text = (ROOT / "scripts" / "verify_v3_archive.sh").read_text(encoding="utf-8")

    assert "sha256sum -c" in text
    assert "tar -tzf" in text
    assert "LOCAL_VERIFIED" in text
