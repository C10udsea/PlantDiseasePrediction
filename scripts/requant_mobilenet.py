"""第二步:用 mmap 数据重做 MobileNetV2 静态 INT8 量化(低内存),并写回 deployment_val.json。"""
import json, time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[1]
DATA = Path.home() / "plantvillage-data" / "quant_mv2"
FP32 = REPO / "experiments/mobilenet_v2/mobilenet_v2_fp32.onnx"
OUT = REPO / "experiments/mobilenet_v2/mobilenet_v2_int8_static.onnx"
DEP = REPO / "experiments/mobilenet_v2/deployment_val.json"

class Reader:
    def __init__(self, arr, batch=64):
        self.arr, self.batch, self.i = arr, batch, 0
    def get_next(self):
        if self.i >= len(self.arr):
            return None
        out = {"input": np.asarray(self.arr[self.i:self.i + self.batch])}
        self.i += self.batch
        return out

def evaluate(sess, X, Y):
    out = sess.run(["logits"], {"input": X})[0]
    p = np.argmax(out, 1)
    return float((p == Y).mean()), float(f1_score(Y, p, average="macro", zero_division=0))

def main():
    cal = np.load(DATA / "calib_x.npy", mmap_mode="r")
    ev_x = np.load(DATA / "eval_x.npy", mmap_mode="r")
    ev_y = np.load(DATA / "eval_y.npy", mmap_mode="r")
    sess_fp32 = ort.InferenceSession(str(FP32), providers=["CPUExecutionProvider"])
    fp32_acc, fp32_f1 = evaluate(sess_fp32, ev_x, ev_y)
    print(f"fp32: acc={fp32_acc:.4f} f1={fp32_f1:.4f}", flush=True)

    t0 = time.time()
    quantize_static(str(FP32), str(OUT), Reader(cal, batch=64),
                    weight_type=QuantType.QInt8, quant_format=QuantFormat.QDQ)
    print(f"static quantized in {time.time()-t0:.1f}s", flush=True)

    sess_q = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    q_acc, q_f1 = evaluate(sess_q, ev_x, ev_y)
    print(f"int8_static: acc={q_acc:.4f} f1={q_f1:.4f} acc_drop={fp32_acc-q_acc:.4f} f1_drop={fp32_f1-q_f1:.4f}", flush=True)

    dep = json.loads(DEP.read_text(encoding="utf-8"))
    v = dep["variants"]["int8_static"]
    v.update({"size_mb": OUT.stat().st_size / 1e6, "acc": q_acc, "macro_f1": q_f1,
              "n": int(len(ev_y)), "note": "retuned: 1024 calib / 512 eval, batch64 reader"})
    so = sess_q.get_session_options(); so.intra_op_num_threads = 1
    dummy = np.asarray(ev_x[:1])
    for _ in range(20): sess_q.run(["logits"], {"input": dummy})
    ts = []
    for _ in range(100):
        t = time.perf_counter(); sess_q.run(["logits"], {"input": dummy}); ts.append((time.perf_counter()-t)*1000)
    v["cpu_ms_median_t1"] = float(np.median(ts)); v["cpu_ms_p90_t1"] = float(np.percentile(ts, 90))
    DEP.write_text(json.dumps(dep, ensure_ascii=False, indent=2), encoding="utf-8")
    print("deployment_val.json updated", flush=True)

if __name__ == "__main__":
    main()
