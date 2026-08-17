# Plant Disease Detection - Project Plan

## Goal
Reproduce the resume project: 38-class plant disease recognition with model comparison, knowledge distillation, and a GUI.

## Data
- Source: Kaggle `abdallahalidev/plantvillage-dataset` (color)
- 38 classes, 54,305 images
- Split: stratified 80/10/10, seed 42

## Models
- ResNet-18
- MobileNetV2
- ViT-B/16
- TinyCNN (depthwise-separable, ~27.3K params)

## Pipeline
1. Data pipeline: manifest, augmentation, loaders
2. Train ResNet-18 / MobileNetV2 / ViT-B/16
3. Train TinyCNN directly as baseline
4. Distill TinyCNN from ViT-B/16 (KL + feature loss)
5. Export ONNX and compare FP32 / dynamic INT8 / static INT8
6. Build Tkinter GUI

## Metrics
- Primary: Macro-F1
- Secondary: Accuracy, Top-3, params, FLOPs, latency, model size

## Acceptance
- All checks pass: `python scripts/check_results.py --strict`
- Unit tests pass: `python -m unittest discover -s tests`
