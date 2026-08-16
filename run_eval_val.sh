#!/bin/bash
set -e
cd /mnt/d/Projects/plant-disease
PY=/home/min/pd-venv/bin/python
$PY src/evaluate.py --model resnet18   --weights experiments/resnet18/best.pth   --split val
$PY src/evaluate.py --model mobilenet_v2 --weights experiments/mobilenet_v2/best.pth --split val
$PY src/evaluate.py --model vit_b16    --weights experiments/vit_b16/best.pth    --split val
$PY src/evaluate.py --model tinycnn    --weights experiments/tinycnn/best.pth    --split val
$PY src/evaluate.py --model tinycnn    --weights experiments/distill_tinycnn_vit_b16/best.pth --split val
$PY src/evaluate.py --model tinycnn    --weights experiments/distill_tinycnn_mobilenet_v2/best.pth --split val
echo EVAL_ALL_DONE
