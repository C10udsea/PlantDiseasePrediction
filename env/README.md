# 环境说明

- 实测环境快照:`env_snapshot.txt`
  - GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, Driver 595.71
  - torch 2.13.0+cu130 / torchvision 0.28.0+cu130 / onnxruntime 1.28.0
- 复现安装:
  - Linux/WSL: `bash env/setup.sh`(venv 生成在 `~/pd-venv`,venv 本身不随项目移动)
  - Windows: `python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt`
- 完整依赖锁定:`requirements.lock`
- 注:训练数据已在 `data\color`,不需要再从 Kaggle 下载;校验: `python scripts/download_data.py --verify`
