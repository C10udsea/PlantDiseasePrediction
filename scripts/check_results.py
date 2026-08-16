#!/usr/bin/env python3
"""项目级自动化验收断言:数据/模型/蒸馏/部署结果与阈值核对。

用法: python scripts/check_results.py [--strict]
所有阈值在 README 中同步说明;数字以 experiments/*/eval_*.json 实测为准。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from data import EXPECTED_CLASSES, load_manifest  # noqa: E402

FAILS = []


def check(name, ok, msg=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {msg}")
    if not ok:
        FAILS.append(name)


def load_json(path):
    if not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    # 阶段 0:数据
    m = load_manifest()
    check("38 classes", m["num_images"] == 54305 and len(m["classes"]) == 38)
    check("class set exact", m["classes"] == EXPECTED_CLASSES)
    tr, va, te = m["split"].values()
    check("80/10/10 sizes", (len(tr), len(va), len(te)) == (43444, 5430, 5431))
    check("no split overlap", len(set(tr) & set(va)) == 0 and len(set(tr) & set(te)) == 0 and len(set(va) & set(te)) == 0)
    check("min class >= 150", min(m["per_class"].values()) >= 150)

    # 阶段 1/2:三模型 val 精度(随机划分口径;test 只在收尾用)
    for model, min_acc, min_f1 in [("resnet18", 0.97, 0.97), ("mobilenet_v2", 0.97, 0.97),
                                   ("vit_b16", 0.96, 0.95)]:
        ev = load_json(REPO / "experiments" / model / "eval_val.json")
        if ev is None:
            check(f"{model} eval_val.json exists", False, "(run src/evaluate.py first)")
        else:
            check(f"{model} val_acc >= {min_acc}", ev["accuracy"] >= min_acc, f"acc={ev['accuracy']:.4f}")
            check(f"{model} val_macro_f1 >= {min_f1}", ev["macro_f1"] >= min_f1, f"f1={ev['macro_f1']:.4f}")

    # 阶段 3:学生参数预算与蒸馏收益
    direct = load_json(REPO / "experiments" / "tinycnn" / "train_summary.json")
    if direct:
        check("TinyCNN params <= 30K", direct["params_trainable"] <= 30_000,
              f"params={direct['params_trainable']} (resume 26.3K)")
    kd = load_json(REPO / "experiments" / "distill_tinycnn_vit_b16" / "train_summary.json")
    if direct and kd:
        gain = kd["best_val_macro_f1"] - direct["best_val_macro_f1"]
        check("KD(ViT) beats direct by >= 1pp", gain >= 0.01,
              f"direct_f1={direct['best_val_macro_f1']:.4f} kd_f1={kd['best_val_macro_f1']:.4f} gain={gain:.4f}")
        check("KD student params <= 30K", kd["student_params"] <= 30_000, f"params={kd['student_params']}")
        check("KD feature loss enabled", kd["feature_loss"] is True)

    # 阶段 4:部署(若已生成)
    dep = load_json(REPO / "experiments" / "distill_tinycnn_vit_b16" / "deployment_val.json")
    if dep:
        v = dep["variants"]
        check("ONNX vs torch max diff < 1e-3", dep["max_logits_diff"] < 1e-3,
              f"diff={dep['max_logits_diff']:.6f}")
        if "int8_static" in v and "fp32" in v:
            drop = v["fp32"]["macro_f1"] - v["int8_static"]["macro_f1"]
            check("INT8 static F1 drop <= 1pp", drop <= 0.01, f"drop={drop:.4f}")
            ratio = v["int8_static"]["size_mb"] / max(v["fp32"]["size_mb"], 1e-9)
            check("INT8 static size reduction >= 60%", 1 - ratio >= 0.60, f"ratio={ratio:.3f} (-{(1-ratio)*100:.1f}%)")

    print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"FAILED: {len(FAILS)} -> {FAILS}"))
    if FAILS and args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
