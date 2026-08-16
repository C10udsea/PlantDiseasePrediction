import json, shutil, time
from pathlib import Path
import numpy as np, onnxruntime as ort
from sklearn.metrics import f1_score
REPO=Path('/mnt/d/Projects/plant-disease'); DATA=Path.home()/'plantvillage-data'/'quant_mv2'
STATIC_SRC=Path('/tmp/mv2_qdq_perch.onnx')
STATIC_DST=REPO/'experiments/mobilenet_v2/mobilenet_v2_int8_static.onnx'
DEP=REPO/'experiments/mobilenet_v2/deployment_val.json'
ev_x=np.load(DATA/'eval_x.npy',mmap_mode='r'); ev_y=np.load(DATA/'eval_y.npy',mmap_mode='r')
shutil.copyfile(STATIC_SRC, STATIC_DST)
dep=json.loads(DEP.read_text(encoding='utf-8'))
def ev(sess):
    out=sess.run(['logits'],{'input':ev_x})[0]; p=np.argmax(out,1)
    return float((p==ev_y).mean()), float(f1_score(ev_y,p,average='macro',zero_division=0))
def lat(sess):
    so=sess.get_session_options(); so.intra_op_num_threads=1
    dummy=np.asarray(ev_x[:1])
    for _ in range(20): sess.run(['logits'],{'input':dummy})
    ts=[]
    for _ in range(100):
        t=time.perf_counter(); sess.run(['logits'],{'input':dummy}); ts.append((time.perf_counter()-t)*1000)
    return float(np.median(ts)), float(np.percentile(ts,90))
for name,v in dep['variants'].items():
    sess=ort.InferenceSession(v['file'],providers=['CPUExecutionProvider'])
    acc,f1=ev(sess); med,p90=lat(sess)
    v.update({'acc':acc,'macro_f1':f1,'n':int(len(ev_y)),'cpu_ms_median_t1':med,'cpu_ms_p90_t1':p90,
              'size_mb':Path(v['file']).stat().st_size/1e6})
    print(name,'acc',acc,'f1',f1,'cpu',round(med,3),'MB',round(v['size_mb'],3))
dep['variants']['int8_static']['note']='per_channel=True QDQ, 1024 calib / 512 eval (retuned)'
dep['note']='all variants evaluated on identical 512-image val subset'
DEP.write_text(json.dumps(dep,ensure_ascii=False,indent=2),encoding='utf-8')
print('updated',DEP)
