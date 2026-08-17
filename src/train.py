"""Train a model with optional freeze-then-finetune and AMP."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast

from data import build_loaders, get_manifest, seed_everything
from models import build_model, count_parameters, freeze_backbone, unfreeze_all

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "color"


def evaluate(model, loader, criterion, device, max_batches=None):
    model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_labels = [], []
    top3_hits = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            if logits.size(1) >= 3:
                top3 = torch.topk(logits, 3, dim=1).indices
                top3_hits += (top3 == y.view(-1, 1)).any(dim=1).sum().item()
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = float((all_preds == all_labels).mean())
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    return {
        "loss": total_loss / max(total, 1),
        "acc": acc,
        "macro_f1": macro_f1,
        "top3": top3_hits / max(total, 1),
        "n": total,
    }


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, epoch, grad_accum=1):
    model.train()
    total_loss, total = 0.0, 0
    all_preds, all_labels = [], []
    optimizer.zero_grad(set_to_none=True)
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(enabled=(scaler is not None)):
            logits = model(x)
            loss = criterion(logits, y) / grad_accum
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            total_loss += float(loss.item() * grad_accum) * x.size(0)
        preds = torch.argmax(logits.detach(), dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())
        total += x.size(0)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return {
        "loss": total_loss / max(total, 1),
        "acc": float((all_preds == all_labels).mean()),
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18", choices=["resnet18", "mobilenet_v2", "vit_b16", "tinycnn"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--freeze-epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--init-from", default=None, help="load state dict before training")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-n", type=int, default=128)
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 2
        args.freeze_epochs = 1
        args.batch_size = 32
        args.num_workers = 2
        args.no_pretrained = True

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} cuda={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    get_manifest(DATA_DIR)
    train_loader, val_loader, _, _ = build_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed)

    if args.smoke:
        from torch.utils.data import DataLoader, Subset
        n = min(args.smoke_n, len(train_loader.dataset))
        subset = Subset(train_loader.dataset, list(range(n)))
        train_loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = build_model(args.model, num_classes=38, pretrained=not args.no_pretrained)
    if args.init_from:
        model.load_state_dict(torch.load(args.init_from, map_location="cpu"))
        print(f"initialized from {args.init_from}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() if (device.type == "cuda" and not args.no_amp) else None

    out_dir = Path(args.output) if args.output else REPO_ROOT / "experiments" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "vit_b16" and args.batch_size == 64:
        args.batch_size = 16
        args.grad_accum = max(args.grad_accum, 2)

    config = {
        "model": args.model,
        "num_classes": 38,
        "pretrained": not args.no_pretrained,
        "epochs": args.epochs,
        "freeze_epochs": args.freeze_epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "amp": scaler is not None,
        "device": str(device),
        "smoke": args.smoke,
        "manifest": str(REPO_ROOT / "data" / "split_manifest.json"),
        "init_from": args.init_from,
        "train_n": len(train_loader.dataset),
        "val_n": len(val_loader.dataset),
        "params_trainable": count_parameters(model),
    }
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    history_path = out_dir / "history.csv"
    with history_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "phase", "train_loss", "train_acc", "train_f1",
                    "val_loss", "val_acc", "val_f1", "val_top3", "lr", "seconds"])

    best_f1, best_epoch, best_state = -1.0, 0, None
    epoch_num = 0

    def run_phase(optimizer, scheduler, n_epochs, phase):
        nonlocal epoch_num, best_f1, best_epoch, best_state
        for _ in range(n_epochs):
            epoch_num += 1
            t0 = time.time()
            tr = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, epoch_num, args.grad_accum)
            va = evaluate(model, val_loader, criterion, device)
            if scheduler is not None:
                scheduler.step()
            dt = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            with history_path.open("a", newline="") as f:
                csv.writer(f).writerow([epoch_num, phase, tr["loss"], tr["acc"], tr["macro_f1"],
                                        va["loss"], va["acc"], va["macro_f1"], va["top3"], lr, round(dt, 1)])
            print(f"[{phase}] epoch {epoch_num}/{args.epochs} "
                  f"train_loss={tr['loss']:.4f} train_acc={tr['acc']:.4f} train_f1={tr['macro_f1']:.4f} | "
                  f"val_loss={va['loss']:.4f} val_acc={va['acc']:.4f} val_f1={va['macro_f1']:.4f} "
                  f"val_top3={va['top3']:.4f} lr={lr:.2e} {dt:.1f}s", flush=True)
            if va["macro_f1"] > best_f1:
                best_f1, best_epoch = va["macro_f1"], epoch_num
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, out_dir / "best.pth")
                print(f"  -> saved best.pth (val_macro_f1={best_f1:.4f})", flush=True)

    if args.freeze_epochs > 0 and args.model != "tinycnn":
        freeze_backbone(model, args.model)
        head_params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(head_params, lr=args.head_lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.freeze_epochs)
        print(f"phase freeze: {args.freeze_epochs} epochs, head params={count_parameters(model)}")
        run_phase(opt, sched, args.freeze_epochs, "freeze")

    unfreeze_all(model)
    ft_epochs = args.epochs - (args.freeze_epochs if args.model != "tinycnn" else 0)
    if ft_epochs > 0:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ft_epochs)
        print(f"phase finetune: {ft_epochs} epochs, lr={args.lr}")
        run_phase(opt, sched, ft_epochs, "finetune")

    torch.save(best_state, out_dir / "best.pth")
    summary = {"best_val_macro_f1": best_f1, "best_epoch": best_epoch,
               "params_trainable": count_parameters(model)}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE model={args.model} best_val_macro_f1={best_f1:.4f} best_epoch={best_epoch} params={summary['params_trainable']}")


if __name__ == "__main__":
    main()
