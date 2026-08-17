#!/usr/bin/env python3
"""Download PlantVillage dataset via kagglehub and verify it."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import EXPECTED_CLASSES, IMAGE_EXTS, REPO_ROOT  # noqa: E402

DATA_LINK = REPO_ROOT / "data"
COLOR_DIR = DATA_LINK / "color"


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
    COLOR_DIR.mkdir(parents=True, exist_ok=True)
    if any(COLOR_DIR.iterdir()):
        print(f"dest already has files, skip copy: {COLOR_DIR}")
    else:
        shutil.copytree(src, COLOR_DIR, dirs_exist_ok=True)
    verify(COLOR_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--color-dir", default=str(COLOR_DIR))
    args = ap.parse_args()
    if args.download:
        download_and_stage()
    else:
        verify(Path(args.color_dir))


if __name__ == "__main__":
    main()
