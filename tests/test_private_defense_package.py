from __future__ import annotations

import importlib.util
from pathlib import Path


def _package_module():
    script = Path(__file__).parents[1] / "scripts" / "build_private_defense_package.py"
    spec = importlib.util.spec_from_file_location("build_private_defense_package", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_private_package_includes_external_handoff_and_backup_recording():
    module = _package_module()
    paths = {str(path) for path in module.PRIVATE_ASSET_PATHS}

    assert "artifacts/cloud-downloads/v5-screen-20260728/bimer-v5-results.tar.gz" in paths
    assert "artifacts/external/videos" in paths
    assert "artifacts/external/annotation-handoff" in paths
    assert "artifacts/external/external-video-plan.locked.json" in paths
    assert "output/deliverables" in paths


def test_private_package_exposes_post_reboot_acceptance_and_honest_status(tmp_path: Path):
    module = _package_module()
    module.write_launchers(tmp_path)
    module.write_readme(tmp_path)

    assert (tmp_path / "06-重启后最终验收.command").exists()
    assert (tmp_path / "07-播放备用录屏.command").exists()
    assert (tmp_path / "09-检查双人标注.command").exists()
    readme = (tmp_path / "README-答辩使用说明.md").read_text(encoding="utf-8")
    assert "20 段外测素材已锁定" in readme
    assert "备用录屏已完成" in readme
    assert "V5 验证阶段结果原始包" in readme
    assert "V5 未运行三随机种子或官方测试" in readme
    assert "第二名人工标注者" in readme
