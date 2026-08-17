#!/bin/bash
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY src/evaluate.py --model resnet18   --weights experiments/resnet18/best.pth   --split test
$PY src/evaluate.py --model mobilenet_v2 --weights experiments/mobilenet_v2/best.pth --split test
$PY src/evaluate.py --model vit_b16    --weights experiments/vit_b16/best.pth    --split test
$PY src/evaluate.py --model tinycnn    --weights experiments/tinycnn/best.pth    --split test
$PY src/evaluate.py --model tinycnn    --weights experiments/distill_tinycnn_vit_b16/best.pth --split test
$PY src/evaluate.py --model tinycnn    --weights experiments/distill_tinycnn_mobilenet_v2/best.pth --split test
echo TEST_EVAL_ALL_DONE
