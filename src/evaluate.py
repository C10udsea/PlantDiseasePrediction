"""统一评估:Acc / Macro-F1(主) / Top-3 / 参数量 / GFLOPs / 延迟 / 体积 -> eval.json。

test 集只在整个项目收尾时使用(--split test);日常用 val。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from data import build_loaders, get_manifest, seed_everything
from models import build_model, count_parameters

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "color"


def accuracy_topk(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1).indices
        correct = (pred == target.view(-1, 1)).any(dim=1).sum().item()
    return correct


def measure_latency(model, device, dummy, warmup=20, reps=100, threads=None):
    model.eval()
    if device.type == "cpu" and threads:
        torch.set_num_threads(threads)
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        times = []
        for _ in range(reps):
            if device.type == "cuda":
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(dummy)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
            else:
                t0 = time.perf_counter()
                model(dummy)
                t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    return float(np.median(times)), float(np.percentile(times, 90))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["resnet18", "mobilenet_v2", "vit_b16", "tinycnn"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--output", default=None)
    ap.add_argument("--skip-latency", action="store_true")
    args = ap.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    get_manifest(DATA_DIR)
    _, val_loader, test_loader, manifest = build_loaders(args.batch_size, args.num_workers)
    loader = val_loader if args.split == "val" else test_loader
    classes = manifest["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    model = build_model(args.model, num_classes=38, pretrained=False)
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(device).eval()

    # 分类指标
    preds, labels, total_loss = [], [], 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += float(torch.nn.functional.cross_entropy(logits, y)) * x.size(0)
            preds.append(torch.argmax(logits, 1).cpu().numpy())
            labels.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    acc = float((preds == labels).mean())
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))
    cm = confusion_matrix(labels, preds, labels=list(range(len(classes))))

    # top3 重新用模型计算(严格)
    top3_hits, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            top3_hits += accuracy_topk(logits, y, 3)
            total += x.size(0)
    top3 = top3_hits / total

    # 效率指标
    params = count_parameters(model)
    dummy = torch.randn(1, 3, 224, 224).to(device)
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        gmacs = float(macs) / 1e9
        gflops = gmacs * 2  # thop 输出 MACs,1 MAC = 2 FLOPs
    except Exception as e:
        print("thop failed:", e)
        gmacs = gflops = None

    latency = {}
    if not args.skip_latency:
        if device.type == "cuda":
            latency["gpu_ms_median"], latency["gpu_ms_p90"] = measure_latency(model, device, dummy)
        cpu_model = build_model(args.model, num_classes=38, pretrained=False)
        cpu_model.load_state_dict(ckpt)
        cpu_model.eval()
        latency["cpu_ms_median"], latency["cpu_ms_p90"] = measure_latency(cpu_model, torch.device("cpu"),
                                                                           dummy.cpu(), threads=4)
        del cpu_model

    size_mb = Path(args.weights).stat().st_size / 1e6

    result = {
        "model": args.model,
        "weights": str(args.weights),
        "split": args.split,
        "n": int(total),
        "loss": total_loss / total,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "top3": top3,
        "params": int(params),
        "gmacs": gmacs,
        "gflops": gflops,
        "latency": latency,
        "size_mb": size_mb,
    }
    out_path = Path(args.output) if args.output else Path(args.weights).parent / f"eval_{args.split}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(Path(out_path).with_suffix(".cm.npy"), cm)
    report = classification_report(labels, preds, target_names=classes, digits=4, zero_division=0)
    Path(out_path).with_name(f"classification_report_{args.split}.txt").write_text(report, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(report)


if __name__ == "__main__":
    main()
