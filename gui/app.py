#!/usr/bin/env python3
"""Tkinter 病害识别 GUI:选图 -> 预处理 -> 推理 -> top-3 置信度。

用法: python gui/app.py [--models resnet18,mobilenet_v2,tinycnn]
推理核心 predict_image() 不依赖 Tk,可无头测试。
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from data import build_transforms  # noqa: E402
from models import build_model  # noqa: E402

MODEL_WEIGHTS = {
    "resnet18": REPO / "experiments" / "resnet18" / "best.pth",
    "mobilenet_v2": REPO / "experiments" / "mobilenet_v2" / "best.pth",
    "tinycnn": REPO / "experiments" / "distill_tinycnn_vit_b16" / "best.pth",
}


def predict_image(model, image_path, transform, device, classes):
    """返回 top-3 [(class, zh, prob), ...],与 GUI 展示逻辑共用。"""
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu()
    top = torch.topk(probs, 3)
    zh_map = json.loads((REPO / "gui" / "labels_zh.json").read_text(encoding="utf-8"))
    return [(classes[int(top.indices[i])], zh_map.get(classes[int(top.indices[i])], ""),
             float(top.values[i])) for i in range(3)]


class App:
    def __init__(self, root):
        self.root = root
        root.title("智慧农业病虫害识别系统 — 38 类叶片病害")
        root.geometry("760x640")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_name = None
        self.transform = build_transforms(train=False)
        self.labels_zh = json.loads((REPO / "gui" / "labels_zh.json").read_text(encoding="utf-8"))
        manifest = json.loads((REPO / "data" / "split_manifest.json").read_text(encoding="utf-8"))
        self.classes = manifest["classes"]

        top = ttk.Frame(root, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="模型:").pack(side="left")
        self.model_var = tk.StringVar(value="resnet18")
        self.model_combo = ttk.Combobox(top, textvariable=self.model_var, state="readonly",
                                        values=list(MODEL_WEIGHTS.keys()))
        self.model_combo.pack(side="left", padx=6)
        ttk.Button(top, text="加载模型", command=self.load_model).pack(side="left")
        ttk.Button(top, text="选择图片", command=self.choose_image).pack(side="left", padx=6)
        self.status = tk.StringVar(value="请先加载模型")
        ttk.Label(top, textvariable=self.status, foreground="#555").pack(side="left")

        mid = ttk.Frame(root, padding=10)
        mid.pack(fill="both", expand=True)
        self.img_label = ttk.Label(mid, text="尚未选择图片", anchor="center")
        self.img_label.pack(side="left", padx=10)
        self.result_text = tk.Text(mid, width=46, height=14, font=("Microsoft YaHei", 11))
        self.result_text.pack(side="left", fill="both", expand=True)

        bottom = ttk.Frame(root, padding=10)
        bottom.pack(fill="x")
        ttk.Button(bottom, text="开始识别", command=self.predict_async).pack(side="left")
        self.bars = []
        for i in range(3):
            row = ttk.Frame(bottom)
            row.pack(fill="x", pady=2)
            ttk.Label(row, width=30, text=f"top{i+1}").pack(side="left")
            bar = ttk.Progressbar(row, maximum=100, length=420)
            bar.pack(side="left", padx=6)
            self.bars.append(bar)

    def load_model(self):
        name = self.model_var.get()
        weights = MODEL_WEIGHTS[name]
        if not weights.exists():
            self.status.set(f"缺少权重: {weights}")
            return
        try:
            self.model = build_model(name, num_classes=38, pretrained=False)
            self.model.load_state_dict(torch.load(weights, map_location="cpu"))
            self.model.to(self.device).eval()
            self.model_name = name
            self.status.set(f"已加载 {name} ({self.device})")
        except Exception as e:
            self.status.set(f"加载失败: {e}")

    def choose_image(self):
        path = filedialog.askopenfilename(filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")])
        if not path:
            return
        self.image_path = Path(path)
        img = Image.open(self.image_path).convert("RGB")
        preview = img.resize((300, 300))
        self._photo = ImageTk.PhotoImage(preview)
        self.img_label.configure(image=self._photo, text="")
        self.status.set(f"已选择: {self.image_path.name}")

    def predict_async(self):
        if self.model is None:
            self.status.set("请先加载模型")
            return
        if not hasattr(self, "image_path"):
            self.status.set("请先选择图片")
            return
        self.status.set("推理中...")
        threading.Thread(target=self._predict, daemon=True).start()

    def _predict(self):
        try:
            preds = predict_image(self.model, self.image_path, self.transform, self.device, self.classes)
            lines = [f"{p*100:5.2f}%  {cls}\n        {zh}" for cls, zh, p in preds]
            for i, (_c, _z, p) in enumerate(preds):
                self.bars[i]["value"] = float(p * 100)
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", "\n".join(lines))
            self.status.set("识别完成")
        except Exception as e:
            self.status.set(f"识别失败: {e}")


def main():
    import tkinter as tk
    from tkinter import filedialog, ttk
    from PIL import ImageTk
    globals().update(tk=tk, ttk=ttk, filedialog=filedialog, ImageTk=ImageTk)

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="resnet18,mobilenet_v2,tinycnn")
    args = ap.parse_args()
    global MODEL_WEIGHTS
    MODEL_WEIGHTS = {k: MODEL_WEIGHTS[k] for k in args.models.split(",")}

    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
