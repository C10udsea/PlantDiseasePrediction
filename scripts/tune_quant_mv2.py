import sys, time
from pathlib import Path
import numpy as np, onnxruntime as ort
from onnxruntime.quantization import QuantType, QuantFormat, quantize_static, CalibrationMethod
from sklearn.metrics import f1_score
REPO=Path('/mnt/d/Projects/plant-disease'); DATA=Path.home()/'plantvillage-data'/'quant_mv2'
FP32=REPO/'experiments/mobilenet_v2/mobilenet_v2_fp32.onnx'
cal=np.load(DATA/'calib_x.npy',mmap_mode='r'); ev_x=np.load(DATA/'eval_x.npy',mmap_mode='r'); ev_y=np.load(DATA/'eval_y.npy',mmap_mode='r')
class R:
    def __init__(self,a,b=64): self.a=a; self.b=b; self.i=0
    def get_next(self):
        if self.i>=len(self.a): return None
        o={'input':np.asarray(self.a[self.i:self.i+self.b])}; self.i+=self.b; return o
def ev(sess):
    out=sess.run(['logits'],{'input':ev_x})[0]; p=np.argmax(out,1)
    return float((p==ev_y).mean()), float(f1_score(ev_y,p,average='macro',zero_division=0))
base=ort.InferenceSession(str(FP32),providers=['CPUExecutionProvider']); print('fp32',ev(base),flush=True)
configs=[
 ('qdq_perch', dict(quant_format=QuantFormat.QDQ, per_channel=True)),
 ('qop_perch', dict(quant_format=QuantFormat.QOperator, per_channel=True)),
 ('qdq_symact', dict(quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8)),
]
for name,kw in configs:
    out=f'/tmp/mv2_{name}.onnx'; t=time.time()
    try:
        quantize_static(str(FP32),out,R(cal),weight_type=QuantType.QInt8,calibrate_method=CalibrationMethod.MinMax,**kw)
        s=ort.InferenceSession(out,providers=['CPUExecutionProvider'])
        print(name,ev(s),f'{time.time()-t:.1f}s',flush=True)
    except Exception as e:
        print(name,'ERR',str(e)[:120],flush=True)
