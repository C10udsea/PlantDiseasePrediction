#!/usr/bin/env python3
"""把 experiments/<model>/history.csv 画成 loss/F1 曲线。"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def plot(csv_path: Path):
    df = pd.read_csv(csv_path)
    out = csv_path.with_suffix(".png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if "loss_total" in df.columns:  # distill history schema
        df = df.rename(columns={"loss_total": "train_loss"})
        axes[0].plot(df["epoch"], df["loss_ce"], label="ce")
        axes[0].plot(df["epoch"], df["loss_kd"], label="kd")
        axes[0].plot(df["epoch"], df["loss_feat"], label="feat")
        axes[1].plot(df["epoch"], df["val_f1"], label="val macro-F1")
        axes[1].plot(df["epoch"], df["val_acc"], label="val acc")
    else:
        axes[0].plot(df["epoch"], df["train_loss"], label="train")
        axes[0].plot(df["epoch"], df["val_loss"], label="val")
        axes[1].plot(df["epoch"], df["train_f1"], label="train macro-F1")
        axes[1].plot(df["epoch"], df["val_f1"], label="val macro-F1")
        axes[1].plot(df["epoch"], df["val_acc"], label="val acc")
    axes[0].set_title("loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].set_title("macro-F1 / acc"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="path to history.csv")
    args = ap.parse_args()
    plot(Path(args.csv))
