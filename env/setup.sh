#!/bin/bash
# 在 WSL/Linux 下重建训练环境(Windows 请改用原生 Python venv + requirements.txt)
set -e
cd "$(dirname "$0")/.."
python3 -m virtualenv ~/pd-venv
source ~/pd-venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.device_count() else 'cpu')"
