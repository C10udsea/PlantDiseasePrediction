import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

import numpy as np
import torch
import torch.nn.functional as F

from data import EXPECTED_CLASSES, build_loaders, load_manifest
from distill import kd_loss, make_teacher_feature_hook
from models import TinyCNN, build_model, count_parameters

RUN_SLOW = os.environ.get("RUN_SLOW", "0") == "1"


class TestData(unittest.TestCase):
    def test_manifest_counts_and_no_overlap(self):
        m = load_manifest()
        self.assertEqual(len(m["classes"]), 38)
        self.assertEqual(m["classes"], EXPECTED_CLASSES)
        self.assertEqual(m["num_images"], 54305)
        tr, va, te = m["split"]["train"], m["split"]["val"], m["split"]["test"]
        self.assertEqual((len(tr), len(va), len(te)), (43444, 5430, 5431))
        self.assertEqual(len(set(tr) & set(va)), 0)
        self.assertEqual(len(set(tr) & set(te)), 0)
        self.assertEqual(len(set(va) & set(te)), 0)
        for c in m["classes"]:
            self.assertGreaterEqual(sum(p.startswith(c + "/") for p in tr), 1)
            self.assertGreaterEqual(sum(p.startswith(c + "/") for p in va), 1)
            self.assertGreaterEqual(sum(p.startswith(c + "/") for p in te), 1)

    def test_dataset_shape(self):
        tr, va, te, _ = build_loaders(batch_size=32, num_workers=0)
        x, y = next(iter(tr))
        self.assertEqual(tuple(x.shape), (32, 3, 224, 224))
        self.assertEqual(tuple(y.shape), (32,))


class TestModels(unittest.TestCase):
    def test_param_budget(self):
        tiny = TinyCNN()
        p = count_parameters(tiny)
        self.assertLessEqual(p, 30_000)
        self.assertGreaterEqual(p, 20_000)
        self.assertEqual(torch.Size([2, 38]), tiny(torch.randn(2, 3, 224, 224)).shape)
        r18 = build_model("resnet18", pretrained=False)
        mv2 = build_model("mobilenet_v2", pretrained=False)
        self.assertAlmostEqual(count_parameters(r18) / 1e6, 11.2, delta=0.3)
        self.assertAlmostEqual(count_parameters(mv2) / 1e6, 2.27, delta=0.1)

    def test_teacher_frozen_and_feature_hook(self):
        teacher = build_model("mobilenet_v2", pretrained=False).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        holder, handle = make_teacher_feature_hook(teacher, "mobilenet_v2")
        with torch.no_grad():
            teacher(torch.randn(2, 3, 224, 224))
        handle.remove()
        self.assertFalse(teacher.training)
        self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
        self.assertEqual(tuple(holder["feat"].shape), (2, 1280))
        self.assertTrue(torch.isfinite(holder["feat"]).all())


class TestKDLoss(unittest.TestCase):
    def test_kd_loss_finite_and_shape(self):
        s = torch.randn(4, 38)
        t = torch.randn(4, 38)
        y = torch.randint(0, 38, (4,))
        ce, kd = kd_loss(s, t, y, T=4.0, alpha=0.5)
        self.assertTrue(torch.isfinite(ce) and torch.isfinite(kd))
        self.assertGreater(kd.item(), 0)

    def test_kl_input_order_contract(self):
        s = torch.randn(3, 38, dtype=torch.float64)
        t = torch.randn(3, 38, dtype=torch.float64)
        T = 4.0
        got = F.kl_div(F.log_softmax(s / T, 1), F.softmax(t / T, 1), reduction="batchmean")
        manual = (F.softmax(t / T, 1) * (F.log_softmax(t / T, 1) - F.log_softmax(s / T, 1))).sum(1).mean()
        self.assertLess((got - manual).abs().item(), 1e-6)


class TestONNX(unittest.TestCase):
    def test_onnx_numeric_consistency(self):
        import onnxruntime as ort
        model = TinyCNN().eval()
        dummy = torch.randn(1, 3, 224, 224)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tiny.onnx"
            torch.onnx.export(model, dummy, str(path), input_names=["input"],
                              output_names=["logits"], opset_version=18, dynamo=False)
            with torch.no_grad():
                torch_out = model(dummy).numpy()
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            ort_out = sess.run(["logits"], {"input": dummy.numpy()})[0]
            self.assertLess(np.abs(torch_out - ort_out).max(), 1e-3)


class TestSmokeE2E(unittest.TestCase):
    @unittest.skipUnless(RUN_SLOW, "set RUN_SLOW=1 to run 2-epoch smoke train")
    def test_smoke_train_e2e(self):
        out = subprocess.run(
            [sys.executable, str(SRC / "train.py"), "--model", "resnet18", "--smoke",
             "--output", str(REPO / "experiments" / "smoke_test")],
            cwd=REPO, capture_output=True, text=True, timeout=600)
        self.assertEqual(out.returncode, 0, out.stdout[-2000:] + out.stderr[-2000:])
        self.assertTrue((REPO / "experiments" / "smoke_test" / "best.pth").exists())
        self.assertTrue((REPO / "experiments" / "smoke_test" / "history.csv").exists())


if __name__ == "__main__":
    unittest.main()
