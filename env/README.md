# Environment

- Snapshot: `env_snapshot.txt`
  - GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, Driver 595.71
  - torch 2.13.0+cu130 / torchvision 0.28.0+cu130 / onnxruntime 1.28.0
- Setup:
  - Linux/WSL: `bash env/setup.sh` (creates venv at `~/pd-venv`)
  - Windows: `python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt`
- Locked dependencies: `requirements.lock`
- Data is already under `data\color`; verify with `python scripts/download_data.py --verify`
