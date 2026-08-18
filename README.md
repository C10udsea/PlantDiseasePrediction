# PlantDiseasePrediction

面向边缘部署的农作物叶片病害识别系统,支持 38 类叶片健康状态分类。项目包含多模型对比、知识蒸馏、ONNX 静态 INT8 量化与桌面端 GUI,覆盖从训练到部署的完整链路。

---

## Features

- **38 类病害识别**:基于 PlantVillage color 数据集,54,305 张图像。
- **模型对比**:ResNet-18、MobileNetV2、ViT-B/16 统一训练与评估,量化精度/FLOPs/延迟。
- **轻量化学生模型**:TinyCNN(27,294 参数),通过知识蒸馏达到高精度低算力。
- **知识蒸馏**:软标签 KL 蒸馏 + 特征蒸馏,支持 ViT-B/16 与 MobileNetV2 教师。
- **边缘部署模拟**:ONNX 导出,FP32 / 动态 INT8 / 静态 INT8 对比。
- **桌面 GUI**:Tkinter 应用,支持模型切换与 Top-3 置信度展示。

## Highlights

| 模型 | Test Acc | Test Macro-F1 | Params | GFLOPs |
|---|---|---|---|---|
| ResNet-18 | 0.9926 | 0.9911 | 11.20M | 3.647 |
| MobileNetV2 | 0.9910 | 0.9874 | 2.27M | 0.653 |
| ViT-B/16 | 0.9814 | 0.9721 | 85.83M | 22.571 |

MobileNetV2 在保持与 ResNet-18 相近精度的同时,计算量约为其 1/5.6,适合作为部署候选。

| TinyCNN training | Test Acc | Test Macro-F1 |
|---|---|---|
| Baseline | 0.8532 | 0.7780 |
| Distilled from ViT-B/16 | **0.9269** | **0.8922** |
| Distilled from MobileNetV2 | 0.9124 | 0.8756 |

## Architecture

```text
data pipeline -> model zoo -> training -> distillation -> ONNX/INT8 -> GUI
     |              |            |            |              |           |
 manifest      ResNet-18   AMP/CSV    ViT teacher     FP32       Tkinter
 80/10/10     MobileNetV2  best.pth   KL + feature   INT8
 seed=42      ViT-B/16               distillation
              TinyCNN
```

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.x with CUDA (optional, CPU works for inference)
- Tested on: WSL Ubuntu / Python 3.14 / RTX 4060 Laptop 8GB

### Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Verify Data

```bash
python scripts/download_data.py --verify
```

### Run Checks

```bash
python scripts/check_results.py --strict
python -m unittest discover -s tests
```

### Train

```bash
python src/train.py --model resnet18 --epochs 7 --freeze-epochs 1
python src/train.py --model mobilenet_v2 --epochs 7 --freeze-epochs 1
python src/train.py --model vit_b16 --epochs 7 --freeze-epochs 1 --batch-size 16 --grad-accum 2
python src/train.py --model tinycnn --epochs 20
python src/distill.py --teacher vit_b16 --teacher-weights experiments/vit_b16/best.pth --epochs 12
```

### Evaluate & Export

```bash
python src/evaluate.py --model mobilenet_v2 --weights experiments/mobilenet_v2/best.pth --split test
python src/export_onnx.py --model tinycnn --weights experiments/distill_tinycnn_vit_b16/best.pth --split val
```

### GUI

Windows 下双击 `start_gui.bat`,或运行:

```bash
python gui/app.py
```

## Project Structure

```text
data/          Dataset and split manifest
src/           Core code: data, models, train, distill, evaluate, export
gui/           Tkinter GUI and English labels
scripts/       Utilities: download, EDA, plotting, checks, quantization
experiments/   Training artifacts, ONNX models, evaluation results
tests/         Automated tests
env/           Environment setup and dependency lock
```

## Dataset & Acknowledgement

- Dataset: [abdallahalidev/plantvillage-dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) (color, 38 classes, 54,305 images)
- Original work: Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics*. arXiv:1511.08060
- Reference notebook: [Plant Village Disease Classification](https://www.kaggle.com/code/abdallahwagih/plant-village-disease-classification-acc-99-6) (used only as task reference; this repository is an independent PyTorch implementation)

## Limitations

- PlantVillage contains near-duplicate images of the same leaf, so random splits can overestimate generalization. Reported numbers should be interpreted as benchmark results on this dataset.
- Dynamic INT8 quantization only quantizes linear layers; static INT8 with per-channel calibration is the recommended deployment path.
- The project focuses on research/demo deployment; production deployment requires further validation on field data.

## License

MIT License for original code. Dataset and pretrained weights belong to their respective owners.
