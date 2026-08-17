"""Save calibration and evaluation arrays for static INT8 quantization."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from data import build_loaders, get_manifest, DEFAULT_DATA_DIR as DATA_DIR

OUT = Path(__file__).resolve().parents[1] / "experiments" / "quant_cache" / "quant_mv2"


def main():
    get_manifest(DATA_DIR)
    _, val, _, _ = build_loaders(64, num_workers=0)
    OUT.mkdir(parents=True, exist_ok=True)
    xs, ys = [], []
    n_cal, n_eval, total = 1024, 512, 0
    for x, y in val:
        xs.append(x.numpy().astype("float32"))
        ys.append(y.numpy().astype("int64"))
        total += len(x)
        if total >= n_cal + n_eval:
            break
    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    np.save(OUT / "calib_x.npy", X[:n_cal])
    np.save(OUT / "eval_x.npy", X[n_cal:n_cal + n_eval])
    np.save(OUT / "eval_y.npy", Y[n_cal:n_cal + n_eval])
    print("saved", OUT, "calib", X[:n_cal].shape, "eval", X[n_cal:n_cal + n_eval].shape, flush=True)


if __name__ == "__main__":
    main()
