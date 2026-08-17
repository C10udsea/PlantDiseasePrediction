# Folder Structure

All deliverables are under the repository root (`PlantDiseasePrediction`).

## 1. data\ - Dataset
| Content | Path |
|---|---|
| PlantVillage color images, 38 classes / 54,305 images | `data\color\<class>\*.JPG` |
| Stratified 80/10/10 split (seed=42) | `data\split_manifest.json` |
| Dataset verification manifest | `data\manifest.json` |

## 2. src\ - Core Code
| File | Purpose |
|---|---|
| `data.py` | Scan, split, manifest, augmentation, DataLoader |
| `models.py` | ResNet-18, MobileNetV2, ViT-B/16, TinyCNN (27,294) |
| `train.py` | Two-stage transfer training with AMP |
| `distill.py` | ViT/MobileNetV2 teacher -> TinyCNN distillation |
| `evaluate.py` | Accuracy, Macro-F1, Top-3, params, FLOPs, latency, size |
| `export_onnx.py` | ONNX export and INT8 quantization comparison |

## 3. gui\ - GUI
- `app.py`: Tkinter image classification with top-3 confidence
- `labels_en.json`: English class labels

## 4. scripts\ - Utilities
Download/verify, EDA, history plotting, acceptance checks, quantization tuning.

## 5. experiments\ - Training Artifacts
- Per-model: `best.pth`, `config.json`, `history.csv`, `eval_val.json`, `eval_test.json`, `train_summary.json`
- Deployment: `*_fp32.onnx`, `*_int8_dynamic.onnx`, `*_int8_static.onnx`, `deployment_val.json`
- Plots: training curves and EDA images

## 6. tests\ - Automated Tests
8 unit tests: data split, shapes, parameter budget, teacher freezing, KD loss, ONNX numeric, smoke e2e.

## 7. env\ - Environment
`setup.sh`, `requirements.txt`, `requirements.lock`, `env_snapshot.txt`

## 8. Root
- `README.md`: results and reproduction guide
- `PLAN.md`: execution plan
- `LICENSE`, `.gitignore`, `run_*.sh`, `start_gui.bat`
