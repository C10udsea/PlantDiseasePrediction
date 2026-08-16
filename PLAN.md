# plant-disease — 智慧农业病虫害识别系统(简历项目复现)

> 学习型项目,2026-08 启动。复现简历项目一:**38 类农作物叶片病害识别 + 模型轻量化 + 知识蒸馏 + GUI**。
> 模式与 kb-agent 相同:按本计划自己实现,每阶段先跑通再优化,遇到问题随时讨论。

---

## 一、项目背景

### 简历上的四个要点(复现验收标准)

| # | 简历内容 | 复现对应 |
|---|---|---|
| 1 | 支持 38 类农作物叶片病害状态识别 | PlantVillage 数据集分类任务 |
| 2 | 对比 ResNet-18、ViT、MobileNetV2 等主流架构,最终选定 MobileNetV2(高 F1 且 FLOPs 约为 ResNet-18 的 1/6) | 三模型同流程训练,产出 Acc/Macro-F1/GFLOPs 对比表 |
| 3 | 知识蒸馏:教师模型(ViT)指导 TinyCNN 学生,软标签 + 特征损失,参数量 26.3K | ViT-B/16(教师)→ TinyCNN(≤30K,目标 26.3K±20%),KD + 特征蒸馏,附 MobileNetV2 教师消融 |
| 4 | Python GUI:图片导入 + 实时病害预测 | Tkinter 应用,可切换模型 |

**数据集**:Kaggle `abdallahalidev/plantvillage-dataset`(color 版,~54,000 张,38 类)
**参考 notebook**(仅参考思路,不复刻代码):[Plant Village Disease Classification | Acc: 99.6%](https://www.kaggle.com/code/abdallahwagih/plant-village-disease-classification-acc-99-6)

### 本机环境(已确认,2026-08-16 多 Agent 审查修订)

- 实际执行环境:**WSL Ubuntu + Python 3.14.4 + RTX 4060 Laptop 8GB**(Windows 侧驱动经 WSL 直通,勿在 Linux 内装驱动)
- 数据、venv 放 ext4(`~` 下),仓库留在 `/mnt/d` 并用软链接;`/mnt/d` 是 9p,直接放 5.4 万张小图会拖慢 DataLoader 一个数量级
- 数据集已通过 kagglehub 匿名下载并校验(38 类 / 54,305 张 / 923MB),无需 kaggle CLI 与 token

---

## 二、技术栈

| 组件 | 选型 | 一句话理由 |
|---|---|---|
| 框架 | PyTorch (CUDA) + torchvision | 主流、调试友好、torchvision 自带预训练权重 |
| 数据下载 | kagglehub 匿名下载(已成功)→ kaggle CLI → GitHub 镜像 | 无需 token;下载后校验 38 类/54,305 张并冻结 manifest |
| FLOPs/参数量 | `thop`、`torchinfo` | 一行代码出 GFLOPs,复现简历对比数字 |
| 蒸馏 | **自实现** KD 损失 | 学习目的,不用现成库 |
| 部署模拟 | ONNX + onnxruntime **静态 INT8 量化**(QDQ+校准集,动态量化仅作附加列) | Conv 网络动态量化不压缩 Conv,4×体积下降只写在静态列 |
| GUI | Tkinter | 标准库零依赖,够用 |
| 可视化 | matplotlib | 训练曲线、类别分布 |

安装命令(WSL 版,2026-08-16 修订):

```bash
python3 -m venv ~/pd-venv                  # venv 放 ext4
source ~/pd-venv/bin/activate
pip install torch torchvision              # 优先 PyPI CUDA wheel;若 cp314 不可用则 uv python install 3.13 建 venv
pip install thop torchinfo onnxruntime kagglehub pillow matplotlib pandas tqdm scikit-learn
python -c "import torch; print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

---

## 三、核心概念速查(实现前先理解)

1. **迁移学习**:ImageNet 预训练模型换掉最后分类层(fc → 38 类),先冻结 backbone 只训头(1-2 epoch),再解冻整体微调。本质:复用已学到的通用视觉特征。
2. **深度可分离卷积**(MobileNet 的核心):标准卷积拆成"逐通道卷积 + 逐点卷积",计算量约降 1/8~1/9。这就是 MobileNetV2 GFLOPs ≈ ResNet-18 的 1/6 的来源。
3. **知识蒸馏(Hinton KD + 特征蒸馏)**:响应式损失 `L_KD = α·CE(y, s) + (1-α)·T²·KLDiv(log_softmax(s/T), softmax(t/T), reduction='batchmean')`(学生 log-prob 在前、教师 prob 在后,**batchmean 不能省**,T² 不能忘);特征损失 `L_feat = MSE(proj(学生GAP特征), 教师CLS特征)`,总损失 `L = L_KD + β·L_feat`。起步参数 T=4、α=0.5、β=0.05。
4. **ONNX 量化**:**静态 INT8(QDQ+校准集)才压缩 Conv**,FP32 权重 → INT8 体积约降 4 倍、CPU 延迟下降、精度略降;动态 INT8 只量化 MatMul,对 MobileNetV2/TinyCNN 这类 Conv 主导网络几乎无收益。
5. **诚实性提醒**:PlantVillage 存在同一叶片多次拍摄的近重复样本,随机划分会让精度虚高(99%+ 部分来源于此)。按简历口径(随机划分)复现,但 README 里注明这一局限。

---

## 四、目录结构(随进度搭建)

```
plant-disease/
├── PLAN.md               # 本文件
├── README.md             # 收尾时写:架构图、复现步骤、结果汇总表
├── requirements.txt
├── .gitignore            # data/、*.pth、.venv/、experiments/ 大文件
├── data/                 # 数据集(gitignore)
├── configs/              # 每个实验一份超参(可先硬编码,后期再抽)
├── src/
│   ├── data.py           # 数据集类、分层划分 80/10/10、增强
│   ├── models.py         # ResNet-18 / MobileNetV2 / TinyCNN
│   ├── train.py          # 通用训练脚本 --model resnet|mobilenet,--smoke 冒烟模式
│   ├── distill.py        # 知识蒸馏训练
│   ├── evaluate.py       # 统一评估:精度/Top-3/参数量/GFLOPs/延迟/体积
│   └── export_onnx.py    # ONNX 导出 + 静态/动态 INT8 量化 + 数值校验
├── gui/
│   └── app.py            # Tkinter:选图 → 预测 → top-3 置信度,模型下拉切换
├── scripts/
│   └── download_data.py  # Kaggle 下载 + 解压 + 校验类别数
├── experiments/          # 训练日志、指标 CSV(gitignore 权重)
└── tests/                # 冒烟测试:划分一致性 / 前向 shape / KD loss 有限值
```

---

## 五、实施阶段

### 阶段 0:环境与数据(第 1 周前半,已按审查修订)

- [x] 环境:WSL + RTX 4060 8GB 可见;数据集已用 kagglehub 匿名下载
- [ ] venv 放 `~/pd-venv`(ext4),装 torch/torchvision 等;验证 CUDA 可用
- [ ] `data/` 放 ext4 并软链回仓库;`scripts/download_data.py` 改为 kagglehub 优先 + `--verify`(38 类集合、54,305±100、min≥150)
- [ ] `data.py`:
  - 扫描 `data/color/` 下 38 个类目录 → `sorted(path)` 保证顺序确定
  - 分层 80/10/10(`stratify`、`random_state=42`),**划分结果持久化到 `data/split_manifest.json`**,所有阶段/测试只从 manifest 加载
  - 训练增强:Resize(256)→RandomCrop(224)→RandomHorizontalFlip→ColorJitter(0.2/0.2/0.2/0.1)→ToTensor→Normalize(ImageNet);val/test:Resize(224)→ToTensor→Normalize
  - 全局种子固定(torch/np/random),dataloader `pin_memory=True`
- [ ] EDA 脚本:类别分布柱状图、每类抽 1 张预览图

**自查(自动化)**:batch shape `[64,3,224,224]`;train/val/test≈43,444/5,431/5,430(±1%);三集合路径零重叠;每类在三个集合中均≥1 张。
**主指标纪律**:test 只在整个项目收尾评估一次;所有中间选择只看 val。

### 阶段 1:ResNet-18 基线(第 1 周后半)

- [ ] `models.py`:`build_model('resnet18')` — `torchvision.models.resnet18(weights=IMAGENET1K_V1)`,`fc = nn.Linear(512, 38)`
- [ ] `train.py`:CE + AdamW(lr=3e-4,wd=1e-4)+ CosineAnnealingLR + AMP;冻结训头 1-2 epoch(头部 lr=1e-3、独立 scheduler)→ 解冻微调 6 epoch(独立 scheduler);保存 val 最优 `experiments/resnet18/best.pth`
- [ ] 每 epoch 写 `history.csv`;`evaluate.py` 输出 Acc / **Macro-F1(主)** / Top-3 / 参数量 / **GFLOPs(=2×thop GMacs,注明换算)** / 延迟 / 体积,落 `eval.json`

**自查**:val Acc 硬门槛 ≥97%、目标 ≥99%(目标不阻塞,如实记录);thop 约 1.82 GMacs。
**先冒烟**:`train.py --smoke`(128 张、1 epoch、覆盖 checkpoint save/load 与 CSV)绿了再开全量。

### 阶段 2:MobileNetV2 / ViT-B/16 对比(第 2 周前半)

- [ ] `build_model('mobilenet_v2')` — `classifier[1] = nn.Linear(1280, 38)`;`build_model('vit_b16')` — `heads.head = nn.Linear(768, 38)`(全量微调 batch16+AMP+梯度累积,或先线性探测)
- [ ] 产出三模型对比表(写入 README):

| 模型 | val/test Acc | Macro-F1 | Top-3 | 参数量 | GFLOPs | CPU 延迟 | 体积 |
|---|---|---|---|---|---|---|---|
| ResNet-18 | | | | ~11.7M | ~3.6(1.82 GMacs) | | |
| MobileNetV2 | | | | ~2.2M | ~0.62(0.31 GMacs) | | |
| ViT-B/16 | | | | ~86M | ~17.6 | | |

- [ ] 结论:以 Macro-F1/精度 + 效率为选型依据,MobileNetV2 为部署模型;ViT-B/16 保留为蒸馏教师(对齐简历)

**自查**:thop 比值 MobileNetV2/ResNet-18 ≈1/5.5~1/6,印证简历"约 1/6";全部数字以实测填表。

### 阶段 3:知识蒸馏(第 2-3 周,本项目核心增量)

- [ ] `models.py` 加 TinyCNN:深度可分离 stem + 3-4 个 DW-PW 块 + GAP + fc(审查给出的 26.7K 参考架构),**目标参数量 ≤30K(锚定简历 26.3K,实测如实写),≤200K 仅兜底**;`train.py --model tinycnn` 先**直接训练**对照组
- [ ] `distill.py`:教师 **ViT-B/16**(freeze + eval + no_grad,logits 离线缓存),学生 TinyCNN 从零训练;损失 `L_KD = α·CE + (1-α)·T²·KLDiv(batchmean)` + `β·MSE(proj(学生GAP),教师CLS)`,T=4、α=0.5、β=0.05 起步;同时做 MobileNetV2 教师消融
- [ ] 对照实验表(README 核心表格):

| 学生 TinyCNN(≤30K) | val/test Acc | Macro-F1 | 参数量 | 体积 | CPU 延迟 |
|---|---|---|---|---|---|
| 直接训练 | | | | | |
| 蒸馏自 ViT-B/16(+KL+特征) | | | | | |
| 蒸馏自 MobileNetV2(+KL+特征,消融) | | | | | |

- [ ] 消融:直接 / +KL / +KL+特征;T∈{2,4,8}、α∈{0.3,0.5,0.7} 小网格在 **val** 上选参

**自查(自动化)**:教师 `training==False` 且 `requires_grad` 全 False;KD loss 有限;`reduction='batchmean'` 已写死;蒸馏应比直接训练提升(目标 +1~3pp,但 26K 级模型预期精度 85~95%,不承诺 99%)。
**坑**:`KLDivLoss(input=log_softmax(学生/T), target=softmax(教师/T), reduction='batchmean')`;T² 不能忘。

### 阶段 4:边缘部署模拟(第 3 周前半)

- [ ] `export_onnx.py`:ResNet-18/MobileNetV2/TinyCNN 全部导出 ONNX(opset 17);onnxruntime vs PyTorch 最大 logits 差 <1e-3
- [ ] 量化对比:FP32 / **静态 INT8(QDQ + 100~500 张校准集)** / 动态 INT8(附加列);对比体积、CPU 延迟、精度损失(静态硬门槛 ≤1pp,目标 <0.5pp)
- [ ] 结论表:**部署主角是 TinyCNN→ONNX→INT8**(呼应简历 MCU 潜力),MobileNetV2 为参照;"26.3K"等简历数字以实测为准,如实写

### 阶段 5:GUI 与收尾(第 3 周后半)

- [ ] `gui/app.py`(Tkinter):选图按钮 → 预处理 → 后台线程推理 → 显示类别(`gui/labels_zh.json` 中英映射)+ top-3 置信度水平条;下拉框切换三模型(权重懒加载)
- [ ] `tests/`:至少 8 个用例(划分一致性 / dataset shape / 模型参数预算 / KD loss 有限 / 教师冻结 / ONNX 数值 / INT8 体积与精度 / smoke e2e)
- [ ] README:架构图、快速开始、四要点对应结果表(数字与 `eval.json` 程序化比对)、**诚实性说明**(近重复样本、2026-08 事后重建、参考 notebook 为 EfficientNetB3 与本项目方法不同)
- [ ] GitHub:`.gitignore` 排除 `data/`、`*.pth`、`*.onnx`、`review/`、`kaggle.json`、`__pycache__/`;代码加 LICENSE;README 含 notebook/数据集/原始论文三方署名

**验收**:README 汇总表覆盖简历四点;GUI 实测 5 张图(至少 2 张数据集外照片);`python -m unittest discover -s tests` 全绿。

---

## 六、风险与对策

| 风险 | 对策 |
|---|---|
| Python 3.14 无 cp314 torch wheel | 先 dry-run 验证;不行用 `uv python install 3.13` 建 venv(零风险兜底) |
| 8GB 显存 OOM | batch 降 32/16;AMP;ViT-B/16 用 batch16+梯度累积,必要时梯度 checkpointing |
| `/mnt/d` 9p I/O 瓶颈 | data 与 venv 放 ext4(`~` 下),仓库软链接 |
| Kaggle 下载失败/限速 | kagglehub 匿名(已成功)→ CLI token → GitHub sparse-checkout color |
| TinyCNN 精度上不去(<85%) | 蒸馏/特征损失消融定位;加宽通道但保持 ≤200K 兜底,如实报告;不做 99% 承诺 |
| 简历数字对不上 | 以实测为准,如实呈现,README 显著位置写明与简历口径的差异(ViT 教师已对齐;26.3K 以实测标注) |

## 七、明确不做(本轮)

- 真实 MCU/树莓派烧录(仅 ONNX + 量化模拟)
- 从零训练 backbone(只用预训练迁移)
- 简历项目二(通渠水流/自适应滤波)—— 完成本项目后再复现
