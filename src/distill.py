"""知识蒸馏:教师(ViT-B/16 或 MobileNetV2) -> 学生 TinyCNN(<=30K)。

损失: L = alpha*CE(y,s) + (1-alpha)*T^2*KLDiv(log_softmax(s/T), softmax(t/T), batchmean)
            + beta*MSE(proj(student_GAP_feat), teacher_feat)
对齐简历:教师默认 ViT-B/16;软标签 + 特征损失。
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast

from data import build_loaders, get_manifest, seed_everything
from models import TinyCNN, build_model, count_parameters

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "color"


def kd_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.5):
    ce = F.cross_entropy(student_logits, labels)
    kld = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean",
    ) * (T * T) * (1.0 - alpha)
    return alpha * ce, kld


def make_teacher_feature_hook(teacher, teacher_name):
    """返回每次前向更新的特征张量容器。"""
    holder = {"feat": None}

    def vit_hook(module, inp, out):
        holder["feat"] = out[:, 0].detach()  # CLS token

    def cnn_hook(module, inp, out):
        holder["feat"] = out.mean(dim=[2, 3]).detach()  # GAP 特征

    if teacher_name in ("vit_b16", "vit"):
        handle = teacher.encoder.ln.register_forward_hook(vit_hook)
    elif teacher_name in ("mobilenet_v2", "mobilenet"):
        handle = teacher.features.register_forward_hook(cnn_hook)
    else:
        raise ValueError(teacher_name)
    return holder, handle


def evaluate_student(model, loader, device, max_batches=None):
    model.eval()
    all_preds, all_labels, total, total_loss = [], [], 0, 0.0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            all_preds.append(torch.argmax(logits, 1).cpu().numpy())
            all_labels.append(y.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return {
        "loss": total_loss / max(total, 1),
        "acc": float((preds == labels).mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "n": total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="vit_b16", choices=["vit_b16", "mobilenet_v2"])
    ap.add_argument("--teacher-weights", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--no-feature", action="store_true")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.batch_size is None:
        args.batch_size = 32 if args.teacher == "vit_b16" else 64
    if args.smoke:
        args.epochs = 2
        args.batch_size = 32
        args.num_workers = 2

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    get_manifest(DATA_DIR)
    train_loader, val_loader, _, _ = build_loaders(args.batch_size, args.num_workers, args.seed)
    if args.smoke:
        from torch.utils.data import DataLoader, Subset
        n = min(128, len(train_loader.dataset))
        subset = Subset(train_loader.dataset, list(range(n)))
        train_loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # 教师
    teacher = build_model(args.teacher, num_classes=38, pretrained=(args.teacher_weights is None and not args.smoke))
    if args.teacher_weights:
        ckpt = torch.load(args.teacher_weights, map_location="cpu")
        teacher.load_state_dict(ckpt)
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    holder, hook = make_teacher_feature_hook(teacher, args.teacher)

    # 学生
    student = TinyCNN(num_classes=38).to(device)
    proj = nn.Linear(88, holder_feat_dim(teacher, args.teacher)).to(device)

    out_dir = Path(args.output) if args.output else REPO_ROOT / "experiments" / f"distill_tinycnn_{args.teacher}"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update({"device": str(device), "student_params": count_parameters(student),
                   "teacher_params": count_parameters(teacher, trainable_only=False)})
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    # 断言教师冻结
    assert teacher.training is False
    assert all(not p.requires_grad for p in teacher.parameters())

    params = list(student.parameters()) + list(proj.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler() if (device.type == "cuda" and not args.no_amp) else None

    hist_path = out_dir / "history.csv"
    with hist_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "loss_total", "loss_ce", "loss_kd", "loss_feat",
                    "train_acc", "val_loss", "val_acc", "val_f1", "lr", "seconds"])

    best_f1, best_epoch, best_state = -1.0, 0, None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        student.train()
        proj.train()
        total_ce = total_kd = total_feat = total_loss = total = 0.0
        correct = 0
        optimizer.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.no_grad():
                with autocast(enabled=(scaler is not None)):
                    t_logits = teacher(x)
                t_feat = holder["feat"].to(device)
            with autocast(enabled=(scaler is not None)):
                s_feat, s_logits = student.forward_features(x)
                ce, kd = kd_loss(s_logits, t_logits, y, args.T, args.alpha)
                loss = ce + kd
                if not args.no_feature:
                    feat = F.mse_loss(proj(s_feat), t_feat)
                    loss = loss + args.beta * feat
                else:
                    feat = torch.tensor(0.0, device=device)
                loss = loss / args.grad_accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train_loader):
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total_ce += ce.item() * x.size(0)
            total_kd += kd.item() * x.size(0)
            total_feat += feat.item() * x.size(0)
            total_loss += loss.item() * args.grad_accum * x.size(0)
            total += x.size(0)
            correct += (torch.argmax(s_logits.detach(), 1) == y).sum().item()

        scheduler.step()
        va = evaluate_student(student, val_loader, device)
        dt = time.time() - t0
        row = [epoch, total_loss / total, total_ce / total, total_kd / total, total_feat / total,
               correct / total, va["loss"], va["acc"], va["macro_f1"],
               optimizer.param_groups[0]["lr"], round(dt, 1)]
        with hist_path.open("a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"epoch {epoch}/{args.epochs} loss={row[1]:.4f} ce={row[2]:.4f} kd={row[3]:.4f} feat={row[4]:.4f} "
              f"train_acc={row[5]:.4f} | val_acc={va['acc']:.4f} val_f1={va['macro_f1']:.4f} {dt:.1f}s", flush=True)
        if va["macro_f1"] > best_f1:
            best_f1, best_epoch = va["macro_f1"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            torch.save(best_state, out_dir / "best.pth")
            print(f"  -> saved best.pth (val_macro_f1={best_f1:.4f})", flush=True)

    hook.remove()
    torch.save(best_state, out_dir / "best.pth")
    summary = {"teacher": args.teacher, "best_val_macro_f1": best_f1, "best_epoch": best_epoch,
               "student_params": count_parameters(student),
               "feature_loss": not args.no_feature}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE distill teacher={args.teacher} best_val_macro_f1={best_f1:.4f} "
          f"student_params={summary['student_params']}")


def holder_feat_dim(teacher, teacher_name):
    if teacher_name in ("vit_b16", "vit"):
        return teacher.hidden_dim
    return teacher.classifier[1].in_features


if __name__ == "__main__":
    main()
