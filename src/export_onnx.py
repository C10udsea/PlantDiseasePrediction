"""Export ONNX models and compare FP32 / dynamic INT8 / static INT8."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from sklearn.metrics import f1_score

from data import build_loaders, get_manifest, seed_everything
from models import build_model

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "color"


def export_fp32(model, path, device):
    model.eval()
    dummy = torch.randn(1, 3, 224, 224, device=device)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["input"], output_names=["logits"],
        opset_version=18, dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
    return dummy


def make_calibration_reader(images: np.ndarray, batch: int = 64):
    class Reader:
        def __init__(self, arr):
            self.arr = arr
            self.i = 0
        def get_next(self):
            if self.i >= len(self.arr):
                return None
            out = {"input": self.arr[self.i:self.i + batch]}
            self.i += batch
            return out
    return Reader(images)


def ort_evaluate(session, loader, n_total=None):
    out_name = session.get_outputs()[0].name
    preds, labels, total = [], [], 0
    for x, y in loader:
        x = x.numpy()
        logits = session.run([out_name], {"input": x})[0]
        preds.append(np.argmax(logits, axis=1))
        labels.append(y.numpy())
        total += len(x)
        if n_total and total >= n_total:
            break
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return {
        "acc": float((preds == labels).mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "n": int(total),
    }


def ort_latency(session, dummy, warmup=20, reps=100, threads=1):
    so = session.get_session_options()
    so.intra_op_num_threads = threads
    out_name = session.get_outputs()[0].name
    for _ in range(warmup):
        session.run([out_name], {"input": dummy})
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        session.run([out_name], {"input": dummy})
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), float(np.percentile(times, 90))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["resnet18", "mobilenet_v2", "vit_b16", "tinycnn"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--calib-samples", type=int, default=256)
    ap.add_argument("--output", default=None)
    ap.add_argument("--skip-static", action="store_true")
    args = ap.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output) if args.output else Path(args.weights).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args.model, num_classes=38, pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    # CPU is used for ONNX consistency because GPU TF32 differs from CPU/ORT by ~1e-2.
    model.eval()
    device = torch.device("cpu")

    fp32_path = out_dir / f"{args.model}_fp32.onnx"
    dummy = export_fp32(model, fp32_path, device)
    dummy_np = dummy.cpu().numpy()

    with torch.no_grad():
        torch_logits = model(dummy).numpy()
    sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    ort_logits = sess_fp32.run(["logits"], {"input": dummy_np})[0]
    max_diff = float(np.abs(torch_logits - ort_logits).max())
    print(f"FP32 torch-vs-ort max|dlogits| = {max_diff:.6f}")
    assert max_diff < 1e-3, "ONNX numerical check failed"

    from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic, quantize_static
    dyn_path = out_dir / f"{args.model}_int8_dynamic.onnx"
    quantize_dynamic(str(fp32_path), str(dyn_path), weight_type=QuantType.QInt8)

    static_path = out_dir / f"{args.model}_int8_static.onnx"
    static_ok = False
    if not args.skip_static:
        get_manifest(DATA_DIR)
        _, val_loader, test_loader, _ = build_loaders(64, num_workers=0)
        loader = val_loader if args.split == "val" else test_loader
        imgs = []
        for i, (x, _y) in enumerate(loader):
            if i >= args.calib_samples:
                break
            imgs.append(x.numpy())
        calib = np.concatenate(imgs, axis=0)
        try:
            quantize_static(
                str(fp32_path), str(static_path), make_calibration_reader(calib),
                weight_type=QuantType.QInt8, quant_format=QuantFormat.QDQ, per_channel=True)
            static_ok = static_path.exists()
        except Exception as e:
            print(f"static quantization failed (continuing): {e}")
            static_ok = False

    results = {"model": args.model, "split": args.split, "max_logits_diff": max_diff, "variants": {}}
    results["variants"]["fp32"] = {"file": str(fp32_path), "size_mb": fp32_path.stat().st_size / 1e6}
    results["variants"]["int8_dynamic"] = {"file": str(dyn_path), "size_mb": dyn_path.stat().st_size / 1e6}
    if static_ok:
        results["variants"]["int8_static"] = {"file": str(static_path), "size_mb": static_path.stat().st_size / 1e6}

    get_manifest(DATA_DIR)
    _, val_loader, test_loader, _ = build_loaders(64, num_workers=8)
    loader = val_loader if args.split == "val" else test_loader
    n_eval = min(len(loader.dataset), 2000) if args.split == "val" else len(loader.dataset)

    for name in list(results["variants"]):
        sess = ort.InferenceSession(results["variants"][name]["file"], providers=["CPUExecutionProvider"])
        met = ort_evaluate(sess, loader, n_total=n_eval)
        lat_med, lat_p90 = ort_latency(sess, dummy_np, threads=1)
        results["variants"][name].update({"acc": met["acc"], "macro_f1": met["macro_f1"],
                                          "n": met["n"], "cpu_ms_median_t1": lat_med,
                                          "cpu_ms_p90_t1": lat_p90})
        print(f"{name}: size={results['variants'][name]['size_mb']:.2f}MB acc={met['acc']:.4f} "
              f"f1={met['macro_f1']:.4f} cpu={lat_med:.2f}ms", flush=True)

    (out_dir / f"deployment_{args.split}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
