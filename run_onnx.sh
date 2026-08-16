#!/bin/bash
cd /mnt/d/Projects/plant-disease
PY=/home/min/pd-venv/bin/python
$PY src/export_onnx.py --model tinycnn --weights experiments/distill_tinycnn_vit_b16/best.pth --split val --calib-samples 256
$PY src/export_onnx.py --model mobilenet_v2 --weights experiments/mobilenet_v2/best.pth --split val --calib-samples 256
$PY src/export_onnx.py --model resnet18 --weights experiments/resnet18/best.pth --split val --calib-samples 256
echo ONNX_ALL_DONE
