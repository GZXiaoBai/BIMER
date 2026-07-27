from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
from pathlib import Path

PRIVATE_ASSET_PATHS = (
    Path(".venv"),
    Path("artifacts/deployment/v2-quality-lagf-seed42"),
    Path("artifacts/models"),
    Path("artifacts/demo/en-noface.mp4"),
    Path("artifacts/demo/zh-face-cao-dewang-voa-50s.mp4"),
    Path("artifacts/analysis/m2-smoke-en.json"),
    Path("artifacts/analysis/v2-formal-ablations"),
    Path("artifacts/analysis/v2-robustness"),
    Path("artifacts/analysis/v4-exploratory"),
    Path("artifacts/acceptance/m2-v2-english-low-memory-20260727"),
    Path("artifacts/acceptance/m2-v2-bilingual-cao-dewang-20260727"),
    Path("artifacts/cloud-downloads/v4-formal-20260727/bimer-v4-results.tar.gz"),
    Path("artifacts/exports/m2-smoke-en"),
    Path("artifacts/external/videos"),
    Path("artifacts/external/annotation-handoff"),
    Path("artifacts/external/external-video-plan.csv"),
    Path("artifacts/external/external-video-plan.locked.json"),
    Path("artifacts/external/SOURCES.md"),
    Path("artifacts/external/SHA256SUMS"),
    Path("output/deliverables"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the private, offline BIMER defense package.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("output/private-defense/BIMER-offline-defense-20260726"),
    )
    parser.add_argument(
        "--copy-assets",
        action="store_true",
        help="Copy large assets instead of using same-volume hard links.",
    )
    return parser.parse_args()


def repository_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"unsafe archive path: {member.name}")
        tar.extractall(destination)


def link_or_copy(source: str, destination: str, *, copy_assets: bool) -> str:
    if copy_assets:
        return shutil.copy2(source, destination)
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def copy_private_path(
    root: Path,
    destination: Path,
    relative: Path,
    *,
    copy_assets: bool,
) -> None:
    source = root / relative
    if not source.exists():
        raise FileNotFoundError(source)
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(
            source,
            target,
            symlinks=True,
            copy_function=lambda src, dst: link_or_copy(
                src,
                dst,
                copy_assets=copy_assets,
            ),
        )
    else:
        link_or_copy(str(source), str(target), copy_assets=copy_assets)


def write_launcher(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def write_launchers(destination: Path) -> None:
    preamble = """#!/bin/zsh
set -euo pipefail
ROOT="${0:A:h}"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
"""
    write_launcher(
        destination / "01-离线自检.command",
        preamble
        + """
exec "$ROOT/.venv/bin/python" -m bimer.cli doctor \
  --deployment "$ROOT/configs/deployment-v2.json" \
  --artifact-root "$ROOT" \
  --offline
""",
    )
    write_launcher(
        destination / "02-启动演示.command",
        preamble
        + """
exec "$ROOT/.venv/bin/python" -m bimer.cli serve \
  --deployment "$ROOT/configs/deployment-v2.json" \
  --artifact-root "$ROOT"
""",
    )
    write_launcher(
        destination / "03-分析英文样例.command",
        preamble
        + """
exec "$ROOT/.venv/bin/python" -m bimer.cli analyze \
  --deployment "$ROOT/configs/deployment-v2.json" \
  --artifact-root "$ROOT" \
  --video "$ROOT/artifacts/demo/en-noface.mp4" \
  --language en \
  --output "$ROOT/artifacts/exports/defense-en"
""",
    )
    write_launcher(
        destination / "04-分析中文样例.command",
        preamble
        + """
exec "$ROOT/.venv/bin/python" -m bimer.cli analyze \
  --deployment "$ROOT/configs/deployment-v2.json" \
  --artifact-root "$ROOT" \
  --video "$ROOT/artifacts/demo/zh-face-cao-dewang-voa-50s.mp4" \
  --language zh \
  --output "$ROOT/artifacts/exports/defense-zh"
""",
    )
    write_launcher(
        destination / "05-校验答辩包.command",
        preamble
        + """
exec shasum -a 256 -c "$ROOT/SHA256SUMS"
""",
    )
    write_launcher(
        destination / "06-重启后最终验收.command",
        preamble
        + """
exec "$ROOT/scripts/run_post_reboot_acceptance.sh"
""",
    )
    write_launcher(
        destination / "07-播放备用录屏.command",
        preamble
        + """
exec open "$ROOT/output/deliverables/BIMER-中文离线演示.mp4"
""",
    )
    write_launcher(
        destination / "08-打开双人标注包.command",
        preamble
        + """
exec open "$ROOT/artifacts/external/annotation-handoff"
""",
    )
    write_launcher(
        destination / "09-检查双人标注.command",
        preamble
        + """
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/check_external_annotations.py" \
  --annotator-one "$ROOT/artifacts/external/annotation-handoff/01-annotator-one.csv" \
  --annotator-two "$ROOT/artifacts/external/annotation-handoff/02-annotator-two.csv" \
  --adjudication "$ROOT/artifacts/external/annotation-handoff/03-adjudication.csv" \
  --report "$ROOT/artifacts/external/annotation-handoff/agreement-report.json"
""",
    )


def write_readme(destination: Path) -> None:
    text = """# BIMER 私有离线答辩包

此目录仅用于本机答辩与私有备份，不得公开上传。它包含：

- 固定的 V2 quality_lagf seed 42 检查点；
- XLM-R、XLS-R、R3D-18、faster-whisper-small 与 YuNet 离线资产；
- 当前 Python 3.11 虚拟环境；
- 已记录来源与哈希的中文制造业人脸访谈、英文无人脸样例和预生成结果；
- V4 探索性结果原始包与双语低内存 M2 实测证据；
- 20 段锁定的中英文外测素材、许可来源、哈希和独立双人标注表；
- 真实最终系统中文备用录屏；
- 经过校验的源码、配置、测试和依赖锁。

## 答辩前流程

1. 双击 `05-校验答辩包.command`，确认全部哈希为 OK。
2. 双击 `01-离线自检.command`，确认 doctor 报告 `ok: true`。
3. 关闭网络后双击 `02-启动演示.command`。
4. 可用 `04-分析中文样例.command` 和 `03-分析英文样例.command` 预演；
   若现场环境异常，展示预生成结果与 PPT。
5. 双击 `07-播放备用录屏.command` 检查离线备用演示。

## 已补齐

- 20 段外测素材已锁定：中英文各 10 段、五类条件各 2 段，许可、时长和
  SHA-256 均已校验。
- 备用录屏已完成，位于 `output/deliverables/BIMER-中文离线演示.mp4`。
- 重启后系统级 swap 验收已自动化，双击 `06-重启后最终验收.command` 即可。

## 仍需两次人工动作

- 重启并在干净登录环境运行 `06-重启后最终验收.command`。当前 BIMER 进程实测
  swap 为 0，但运行前的 macOS 全局 swap 已非零，不能伪造“干净启动”结论。
- 邀请第二名人工标注者独立填写 `02-annotator-two.csv`。第一、第二标注者必须
  禁止互看，完成后才能计算 Cohen's kappa、仲裁并报告外测指标。

EmotionTalk、MELD 媒体、缓存特征和私人视频受许可约束，不包含在公开仓库中。
"""
    (destination / "README-答辩使用说明.md").write_text(text, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(destination: Path) -> Path:
    manifest = destination / "SHA256SUMS"
    files = sorted(path for path in destination.rglob("*") if path.is_file() and path != manifest)
    with manifest.open("w", encoding="utf-8") as handle:
        for path in files:
            relative = path.relative_to(destination)
            handle.write(f"{sha256(path)}  {relative.as_posix()}\n")
    return manifest


def build(destination: Path, *, copy_assets: bool) -> Path:
    root = repository_root()
    destination = (
        (root / destination).resolve() if not destination.is_absolute() else destination.resolve()
    )
    if destination.exists():
        raise FileExistsError(
            f"{destination} already exists; choose a new destination to avoid accidental overwrite",
        )
    destination.mkdir(parents=True)

    with tempfile.TemporaryDirectory(prefix="bimer-defense-") as tmp:
        archive = Path(tmp) / "source.tar"
        subprocess.run(
            ["git", "archive", "--format=tar", f"--output={archive}", "HEAD"],
            cwd=root,
            check=True,
        )
        safe_extract(archive, destination)

    for relative in PRIVATE_ASSET_PATHS:
        copy_private_path(
            root,
            destination,
            relative,
            copy_assets=copy_assets,
        )

    write_launchers(destination)
    write_readme(destination)
    write_manifest(destination)
    return destination


def main() -> int:
    args = parse_args()
    destination = build(args.destination, copy_assets=args.copy_assets)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
