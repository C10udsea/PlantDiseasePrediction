# 文件夹分类说明

所有交付材料已收纳在 `D:\Projects\plant-disease\`,分类如下:

## ① data\ — 数据集与划分
| 内容 | 路径 |
|---|---|
| PlantVillage color 原图,38 类 / 54,305 张 / 约 923MB | `data\color\<类别>\*.JPG` |
| 分层 80/10/10 划分清单(seed=42) | `data\split_manifest.json` |
| 数据校验清单(38类、文件数、每类数量) | `data\manifest.json` |

## ② src\ — 核心代码
| 文件 | 职责 |
|---|---|
| `data.py` | 数据扫描、分层划分、manifest、增强、DataLoader |
| `models.py` | ResNet-18 / MobileNetV2 / ViT-B/16 / TinyCNN(27,294) |
| `train.py` | 两阶段迁移训练 + AMP + CSV/checkpoint |
| `distill.py` | ViT→TinyCNN 知识蒸馏(软标签 KL + 特征损失) |
| `evaluate.py` | Acc / Macro-F1 / Top-3 / 参数量 / GFLOPs / 延迟 / 体积 |
| `export_onnx.py` | ONNX 导出 + 静态/动态 INT8 + 数值校验 |

## ③ gui\ — GUI
- `app.py`:Tkinter 选图 → 预处理 → top-3 置信度(中文标签,模型切换)
- `labels_zh.json`:38 类中英文映射

## ④ scripts\ — 工具脚本
下载校验 / EDA / 训练曲线 / 结果断言 / 量化校准与调参

## ⑤ experiments\ — 训练产物
- 每个模型一个子目录:`best.pth`(最终权重)、`config.json`、`history.csv`、`eval_val.json`、`eval_test.json`、`train_summary.json`
- 部署:`*_fp32.onnx`、`*_int8_dynamic.onnx`、`*_int8_static.onnx`、`deployment_val.json`
- 训练曲线 PNG、EDA PNG
- 说明:日志、中间 checkpoint、量化缓存等过程文件已删除;需要可重新生成

## ⑥ tests\ — 自动化测试
8 个 unittest:数据划分/形状/参数预算/教师冻结/KD loss/ONNX 数值/e2e 冒烟

## ⑦ env\ — 环境配置
`setup.sh`、`requirements.txt`、`requirements.lock`、`env_snapshot.txt`

## ⑧ 根目录
- `README.md`:结果汇总与复现说明
- `PLAN.md`:修订后的执行计划
- `LICENSE`、`.gitignore`、一键运行脚本 `run_*.sh`
