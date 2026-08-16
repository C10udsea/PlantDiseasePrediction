#!/usr/bin/env python3
"""EDA:38 类分布柱状图 + 每类 1 张样张网格。"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from data import EXPECTED_CLASSES, load_manifest

m = load_manifest()
color_dir = Path(m["color_dir"])
counts = {c: m["per_class"][c] for c in EXPECTED_CLASSES}

# 1) 分布
fig, ax = plt.subplots(figsize=(16, 6))
ax.bar(range(len(counts)), list(counts.values()), color="#4C9F70")
ax.set_xticks(range(len(counts)), list(counts.keys()), rotation=90, fontsize=7)
ax.set_title(f"PlantVillage color: {m['num_images']} images / 38 classes")
ax.set_ylabel("images")
fig.tight_layout()
out1 = REPO / "experiments" / "eda_class_distribution.png"
out1.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out1, dpi=120)
plt.close(fig)
print("saved", out1)

# 2) 样张网格(每类第 1 张)
fig, axes = plt.subplots(8, 5, figsize=(14, 20))
axes = axes.ravel()
for i, c in enumerate(EXPECTED_CLASSES):
    p = next((color_dir / c).iterdir())
    axes[i].imshow(Image.open(p).convert("RGB"))
    axes[i].set_title(f"{c[:28]}\n{counts[c]}", fontsize=6)
    axes[i].axis("off")
for j in range(i + 1, len(axes)):
    axes[j].axis("off")
fig.suptitle("One sample per class", fontsize=14)
fig.tight_layout()
out2 = REPO / "experiments" / "eda_samples.png"
fig.savefig(out2, dpi=100)
plt.close(fig)
print("saved", out2)
