# plant-disease — 智慧农业病虫害识别系统(简历项目复现)

> 学习型复现项目,2026-08 启动并完成。目标:对齐简历项目一「基于轻量化神经网络的植物病虫害检测系统」的四点声明:
> 38 类识别、多架构对比、知识蒸馏(TinyCNN ≤30K)、Python GUI。
> **本仓库为 2026-08 对简历项目的事后学习性重建,所有数字以本仓库脚本实测为准。**

---

## 一、结果速览(test,全项目只使用一次)

| 模型 | test Acc | test Macro-F1 | Top-3 | 参数量 | GFLOPs | CPU 单张(ms) | 权重体积 |
|---|---|---|---|---|---|---|---|
| ResNet-18 | **0.9926** | 0.9911 | 0.9996 | 11.20M | 3.647 | 22.3 | 44.85 MB |
| MobileNetV2 | **0.9910** | 0.9874 | 0.9994 | 2.27M | **0.653** | 13.4 | 9.30 MB |
| ViT-B/16 | 0.9814 | 0.9721 | 0.9991 | 85.83M | 22.571 | 184.1 | 343.35 MB |

- MobileNetV2 与 ResNet-18 精度/F1 基本持平(Acc 差 0.16pp),而 **GFLOPs = 0.653 / 3.647 ≈ 1/5.6**,印证简历「FLOPs 约为 ResNet-18 的 1/6」。
- ViT-B/16 精度更低、计算量高 34.5 倍,不是本任务的帕累托最优 → 选择 MobileNetV2 作部署模型是实测结论。

### 知识蒸馏(学生 TinyCNN,实测 **27,294** 参数)

| TinyCNN 训练方式 | test Acc | test Macro-F1 | 较直接训练 F1 提升 |
|---|---|---|---|
| 直接训练(对照组) | 0.8532 | 0.7780 | — |
| 蒸馏自 **ViT-B/16**(软标签 KL + 特征损失) | **0.9269** | **0.8922** | **+11.4pp** |
| 蒸馏自 MobileNetV2(教师消融) | 0.9124 | 0.8756 | +9.8pp |

- 简历写「参数量 26.3K」:本复现架构实测 **27,294(≤30K,26.3K 的 +3.8%)**,同量级,README 以实测为准。
- 教师对齐简历:**ViT-B/16**;蒸馏损失 = `α·CE + (1-α)·T²·KLDiv(batchmean) + β·MSE(proj(GAP), CLS)`,T=4、α=0.5、β=0.05。

### 部署模拟(ONNX + INT8,val 子集)

| 模型 / 变体 | Acc | Macro-F1 | 体积 | CPU 单张(ms) |
|---|---|---|---|---|
| TinyCNN FP32 | 0.9360 | 0.9013 | 0.11 MB | 0.52 |
| TinyCNN 动态INT8 | 0.9277 | 0.8916 | 0.04 MB | 10.13 |
| TinyCNN 静态INT8 | 0.9282 | 0.8914 | 0.04 MB | 0.55 |
| MobileNetV2 FP32 | 0.9941 | 0.9949 | 9.06 MB | 3.37 |
| MobileNetV2 动态INT8 | 0.9629 | 0.9625 | 2.46 MB | 46.79 |
| MobileNetV2 静态INT8 | **0.9941** | **0.9949** | **2.66 MB** | 4.31 |
| ResNet-18 FP32 | 0.9937 | 0.9854 | 44.77 MB | 11.45 |
| ResNet-18 静态INT8 | 0.9937 | 0.9856 | 11.25 MB | 12.03 |

- **静态 INT8(QDQ + per_channel + 校准集)对 Conv 网络才有真收益**:MobileNetV2 体积 -70.7%、精度几乎无损。
- 动态 INT8 只量化 MatMul:在 Conv 主导网络上收益小、本数据上精度损失反而明显,故仅作附加列。
- 延迟为 `intra_op_num_threads=1` 口径;多线程/GPU 部署延迟会更低,以 `evaluate.py`/`export_onnx.py` 输出为准。

---

## 二、与简历四点的对应

| 简历声明 | 本仓库证据 |
|---|---|
| 38 类农作物叶片健康状态检测 | PlantVillage color 版,38 类 / 54,305 张;`data/split_manifest.json` |
| 对比 ResNet-18 / ViT / MobileNetV2,选 MobileNetV2 | `experiments/*/eval_test.json`;结论如上 |
| ViT 教师蒸馏 TinyCNN,软标签 + 特征损失,26.3K | `src/distill.py`;TinyCNN 27,294 参数;F1 0.778→0.892 |
| Python GUI 上传→实时预测 | `gui/app.py`(Tkinter,top-3 + 中文标签,模型切换) |

---

## 三、快速开始(本机实测环境)

```bash
# WSL Ubuntu / Python 3.14.4 / RTX 4060 Laptop 8GB / torch 2.13.0+cu130
python3 -m virtualenv ~/pd-venv
source ~/pd-venv/bin/activate
pip install -r requirements.txt

# 数据(已下载校验:38 类 / 54,305 张;公开数据集可匿名拉取)
python scripts/download_data.py --verify      # 校验现有 data
# 若需重新下载: python scripts/download_data.py --download

# 一键验收(全部为自动断言)
python scripts/check_results.py --strict
python -m unittest discover -s tests

# 完整训练复现顺序
python src/train.py --model resnet18 --epochs 7 --freeze-epochs 1
python src/train.py --model mobilenet_v2 --epochs 7 --freeze-epochs 1
python src/train.py --model vit_b16 --epochs 7 --freeze-epochs 1 --batch-size 16 --grad-accum 2
python src/train.py --model tinycnn --epochs 20                      # 直接训练对照组
python src/distill.py --teacher vit_b16 --teacher-weights experiments/vit_b16/best.pth --epochs 12
python src/evaluate.py --model mobilenet_v2 --weights experiments/mobilenet_v2/best.pth --split test
python src/export_onnx.py --model tinycnn --weights experiments/distill_tinycnn_vit_b16/best.pth --split val
python gui/app.py                                                   # Windows 或带 WSLg/显示的 Linux
```

## 四、目录结构

```
src/data.py           扫描 + 分层 80/10/10(seed=42) + manifest + 增强
src/models.py         ResNet-18 / MobileNetV2 / ViT-B/16 / TinyCNN(27,294)
src/train.py          两阶段迁移训练 + AMP + CSV/checkpoint(通用)
src/distill.py        Hinton KD(batchmean) + 特征蒸馏;教师冻结断言
src/evaluate.py       Acc / Macro-F1 / Top-3 / 参数量 / GFLOPs / 延迟 / 体积
src/export_onnx.py    ONNX 导出 + 静态/动态 INT8 + 数值校验(<1e-3)
gui/app.py            Tkinter GUI(top-3 + 中文映射 + 模型切换)
scripts/              download_data / eda / plot_history / check_results
tests/                8 个自动化测试
```

## 五、数据与参考

- 数据集:Kaggle `abdallahalidev/plantvillage-dataset`(color 版,38 类,54,305 张)
- 原始出处:Hughes & Salathé, *An open access repository of images on plant health to enable the development of mobile disease diagnostics*, arXiv:1511.08060 (2015)
- 镜像源:https://github.com/spmohanty/PlantVillage-Dataset
- 参考 notebook(仅借鉴任务思路,代码为 PyTorch 独立实现):Abdallah Wagih Ibrahim, *Plant Village Disease Classification | Acc: 99.6%*, https://www.kaggle.com/code/abdallahwagih/plant-village-disease-classification-acc-99-6
  - 该 notebook 为 TensorFlow 2.9 + EfficientNetB3 单模型迁移学习,与本仓库方法(轻量化 + 蒸馏)不同,其 99.6% 不可直接横向比较。

## 六、诚实性说明(必读)

1. **PlantVillage 近重复样本**:同一叶片多次拍摄,随机划分会使 train/test 出现近重复样本,精度系统性偏高。本复现沿用随机划分以对齐简历口径;该数字不代表野外泛化能力。
2. **事后重建**:本仓库为 2026-08 的学习性复现,不是 2025 年原始项目记录。
3. **数字以实测为准**:TinyCNN 实测 27,294 参数(简历 26.3K 同量级);MobileNetV2/ResNet-18 FLOPs 实测 1/5.6;所有表格来自 `eval_*.json` 与 `deployment_*.json`。
4. **动态 INT8 的坑**:它不量化 Conv;本仓库主表以静态 INT8 为准,动态量化仅作附加证据。
5. **授权**:数据集与参考 notebook 未标注可再分发许可,故 `data/`、`review/`、权重与 ONNX 均不入库;仓库代码 MIT 许可仅覆盖原创代码。

## 七、License

本仓库原创代码按 MIT 许可提供;数据集、预训练权重与参考 notebook 版权归原权利人。
