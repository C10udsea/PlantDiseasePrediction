#!/usr/bin/env python3
"""PlantVillage 数据获取(审查修订版):kagglehub 匿名下载 -> 校验 38 类/54,305 张 -> 冻结 manifest。

实际执行环境数据已下载到 ~/plantvillage-data/color;本脚本支持:
  python scripts/download_data.py --verify            # 只校验现有 data
  python scripts/download_data.py --download          # kagglehub 下载并 stage color
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import EXPECTED_CLASSES, IMAGE_EXTS, REPO_ROOT  # noqa: E402

DATA_LINK = REPO_ROOT / "data"


def verify(color_dir: Path) -> dict:
    color_dir = Path(color_dir)
    if not color_dir.exists():
        raise SystemExit(f"data dir not found: {color_dir}")
    classes = sorted(p.name for p in color_dir.iterdir() if p.is_dir())
    if set(classes) != set(EXPECTED_CLASSES):
        missing = set(EXPECTED_CLASSES) - set(classes)
        extra = set(classes) - set(EXPECTED_CLASSES)
        raise SystemExit(f"CLASS MISMATCH missing={sorted(missing)} extra={sorted(extra)}")
    counts = {}
    total = 0
    for cls in classes:
        n = sum(1 for p in (color_dir / cls).iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        counts[cls] = n
        total += n
    print(f"classes={len(classes)} files={total}")
    if not (54000 <= total <= 54500):
        raise SystemExit(f"file count {total} out of expected range")
    min_cls = min(counts, key=counts.get)
    if counts[min_cls] < 150:
        raise SystemExit(f"min class too small: {min_cls}={counts[min_cls]}")
    manifest = {"source": "kagglehub:abdallahalidev/plantvillage-dataset",
                "verified_classes": len(classes), "verified_files": total,
                "per_class": counts}
    out = DATA_LINK / "manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK manifest -> {out}; min class {min_cls}={counts[min_cls]}; max class {max(counts, key=counts.get)}={counts[max(counts, key=counts.get)]}")
    return manifest


def download_and_stage():
    import kagglehub
    path = Path(kagglehub.dataset_download("abdallahalidev/plantvillage-dataset"))
    src = next(path.rglob("color")) if not (path / "color").exists() else path / "color"
    dest = Path.home() / "plantvillage-data" / "color"
    if dest.exists():
        print(f"dest exists, skip copy: {dest}")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest)
    if DATA_LINK.exists() and not DATA_LINK.is_symlink():
        print(f"WARN: {DATA_LINK} exists and is not a symlink")
    else:
        if DATA_LINK.is_symlink():
            DATA_LINK.unlink()
        DATA_LINK.symlink_to(dest.parent)
    verify(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--color-dir", default=str(DATA_LINK / "color"))
    args = ap.parse_args()
    if args.download:
        download_and_stage()
    elif args.verify or True:
        verify(Path(args.color_dir))


if __name__ == "__main__":
    main()
