#!/bin/bash
# Recreate the training environment in WSL/Linux.
set -e
cd "$(dirname "$0")/.."
python3 -m virtualenv ~/pd-venv
source ~/pd-venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.device_count() else 'cpu')"
