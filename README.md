# Dinomaly 计算机视觉课程复现

《Dinomaly 多类别无监督视觉异常检测方法的复现与分析》的实验代码。
复现对象为 **CVPR 2025 Dinomaly**，不是 Dinomaly2，也不将已有机制作为原创算法。

**当前状态（2026-10-30）：服务器到位前的代码准备；没有真实实验结果。**

公开代码仓库：[hyzf12321/dinomaly-cv-coursework](https://github.com/hyzf12321/dinomaly-cv-coursework)。

首次阅读请看[仓库导览与提交规则](docs/REPOSITORY_GUIDE.md)：包含目录分类、配置字段注解、必要上传范围及 Git 管理流程。代码在阶段检查完成后同步；数据、环境与缓存保持本地，训练权重后续通过 Releases 发布。
6 组本机 CPU 测试已通过，使用随机权重或合成数据。Python 3.8 / Linux CUDA 候选环境尚未验证；默认 ViT-Base、392×392、batch=16 的 GPU 显存和耗时仍待服务器检查。

## 来源

- [官方仓库](https://github.com/guojiajeremy/Dinomaly)，固定提交 `1f252be03a918789b19848f0ca37166e7c28dada`。
- [CVPR 2025 论文](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_Dinomaly_The_Less_Is_More_Philosophy_in_Multi-Class_Unsupervised_Anomaly_CVPR_2025_paper.html)。
- 官方实现的运行相关依赖保留在 `vendor/dinomaly/`；逐文件 SHA256 见 `vendor/SOURCE.json`。这些文件没有修改。
- 新增实现位于 `course/` 和 `run.py`：只增加配置、数据校验、运行记录、断点和统一评测。许可及第三方声明见 [NOTICE](NOTICE.md) 和 [LICENSE](LICENSE)。

## 实验矩阵

所有配置采用一个覆盖 MVTec AD 全部 15 类的统一模型，公开预训练 DINOv2 **ViT-Base/14 + 4 register tokens**，编码器冻结。缩放 448×448、中心裁剪 392×392，batch=16，10,000 个优化器更新步。

| 配置 | 瓶颈 Dropout | 注意力 | 监督 | 损失 |
|---|---:|---|---|---|
| B | 0 | Softmax | 8 层逐层重建 | 普通全局余弦 |
| F | 0.2 | LinearAttention2 | 两组融合重建 | 宽松余弦 |
| A1 | 0 | LinearAttention2 | 两组融合重建 | 宽松余弦 |
| A2 | 0.2 | Softmax | 两组融合重建 | 宽松余弦 |
| A3 | 0.2 | LinearAttention2 | 两组融合重建 | 普通全局余弦 |

B 按论文第 4.4 节的普通重建基线构造，逐层监督由固定源码的 ViTill 单层融合组实现；不是把 F 的三个开关同时关闭后仍保留分组监督。F/A1/A2/A3 的两组均为 `[0,1,2,3]`、`[4,5,6,7]`，目标编码器层为 `[2,3,4,5,6,7,8,9]`，解码器 8 层。宽松损失保留官方梯度钩子实现，`p=min(0.9*step/1000,0.9)`、`factor=0.1`。

先运行五组种子 1，再为 B/F 成对追加种子 2、3，共 9 次。清单见 `configs/run-matrix.csv`。固定终点，不根据测试集选最佳 checkpoint。作者发布的 20,000 步权重不进入自己训练的 10,000 步主对照。

## 环境与目录

服务器建议 Linux x86_64 / RTX 3090 24GB / Python 3.8，torch 1.12.0+cu113 与 torchvision 0.13.0+cu113。以下命令仅为服务器准备说明，尚未在 Linux 执行：

```bash
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install 'pip<25' 'setuptools<70' wheel
python -m pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install --no-build-isolation -r requirements-server-candidate-lock.txt
python -m pip check
python tests/test_workflow.py
```

ptflops 0.7 需要源码构建，所以先安装 torch，再使用 `--no-build-isolation`。pandas 保持 1.3.5，官方 AUPRO 使用 `DataFrame.append`；不要单独升级到 2.x。xFormers 明确禁用。CPU 实测依赖另见 `requirements-cpu-verified-lock.txt`，不能将其当成 GPU 已验证清单。

在本仓库内建立以下本地目录，均已被 Git 忽略；本机仓库本身也位于用户指定的大作业总目录内：

```text
data/mvtec/                 # 数据集
weights/                   # 公开预训练编码器
runs/                      # checkpoint、日志与逐样本预测
results/                   # 后续核验后提交的表格、来源记录和必要图表
```

数据来自 [MVTec AD 官方页面](https://www.mvtec.com/company/research/datasets/mvtec-ad)。保留 `category/train/good/*.png`、`category/test/defect/*.png`、`category/ground_truth/defect/*_mask.png` 原始划分；不重新随机划分或混入异常训练样本。数据使用遵守其官方许可，不上传数据。

编码器来源为官方源码中使用的 [DINOv2 ViT-Base register 预训练文件](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth)。**当前没有下载。** 到服务器后再获取，记录来源和 SHA256；训练入口要求本地文件，严格加载，禁止随机权重冒充预训练编码器。ViT-Small 仅作为计划书的统一降级方案。

## 命令

在仓库根目录运行。以下真实数据命令尚未执行：

```bash
python run.py validate-data --data data/mvtec --output runs/data-validation.json
python run.py train --config configs/F.json --data data/mvtec --encoder weights/dinov2_vitb14_reg4_pretrain.pth --seed 1 --device cuda:0 --output runs/F-s1
python run.py train --config configs/F.json --data data/mvtec --encoder weights/dinov2_vitb14_reg4_pretrain.pth --seed 1 --device cuda:0 --output runs/F-s1 --resume runs/F-s1/last.pt
python run.py evaluate --checkpoint runs/F-s1/inference.pt --data data/mvtec --device cuda:0 --output runs/F-s1/eval
python run.py summarize runs/B-s1/eval/evaluation.json runs/F-s1/eval/evaluation.json --output runs/summary-s1
```

更换 `configs/B.json`、`A1.json`、`A2.json`、`A3.json` 和种子即可运行对应配置；输出目录必须独立。已有非空训练目录必须明确续训；评测和汇总目录不得覆盖。

### 权重与续训

- `last.pt`：每 500 步及终点原子替换保存，包含全模型、优化器、scheduler、随机数状态、epoch 和 batch 位置；避免堆积全部中间模型。崩溃后从最近已保存的更新步恢复。
- `inference.pt`：固定终点自动导出，包含完整编码器和重建网络，另含配置、更新步、来源记录；不依赖再次下载编码器即可评测。
- 每次训练记录 config、源文件哈希、Git 提交、预训练文件 SHA256、训练数据逐文件哈希、环境及逐步损失。`completion.json` 包含终点、会话耗时、显存峰值和最终文件校验值。
- 续训拒绝配置、种子、代码、数据或编码器变化。CPU 合成检查已验证中途恢复与连续训练逐参数一致；CUDA 重现性仍待同硬件实测。
- 数据预处理无随机增强。为可恢复性使用独立的 `seed+epoch` 生成每轮 shuffle，drop_last=True；这与官方共享全局随机数生成器的样本顺序有差别，论文应披露。AdamW、学习率、梯度裁剪和核心方法计算沿用官方。
- PyTorch checkpoint 只加载自己生成或官方可信来源的文件。

### 服务器短训练

到服务器后，从 F.json 复制一份到 `runs/probe-config.json`，将 `steps` 设为 200、`save_every` 设为 50，保留默认主干、分辨率、batch 和 warmup=100。使用新输出目录及 `--purpose probe` 训练，再进行真实数据评测。短训练只用于检查 CUDA、预训练加载、损失、显存、耗时和热力图，不能替代正式实验。主汇总入口拒绝 `purpose=probe`。

不完整断点评测必须显式传 `--allow-incomplete`，其结果同样不能进入主汇总。正式运行不要根据测试结果调终点。

## 固定评测与结果

图像级 AUROC、像素级 AUROC/AP、AUPRO 保留原始 0–1 数值，逐类和 15 类宏平均均输出。汇总对种子计算均值和样本标准差；n=1 的标准差留空，同时输出 B/F 相同种子的差值与尚缺的计划运行。

协议来自固定源码的 `evaluation_batch(..., max_ratio=0.01, resize_mask=256)`：余弦距离图平均，插值到裁剪尺寸时 align_corners=True，再 bilinear/align_corners=False 到 256；掩码按官方 PIL resize/中心裁剪，再 nearest 到 256，使用 >0 二值化；之后施加 5×5、sigma=4 高斯平滑，图像分数取最高 1% 像素均值。**掩码最初的 PIL 插值也被固定，不能临时更换。**

AUPRO 直接复用官方 `compute_pro`：200 阈值、FPR<0.3，并按采样到的最大 FPR 归一化；这不是另行实现的严格连续积分版本。保留这一差异以便追溯；指标未定义时直接报错，不填 0 或猜测值。

评测额外输出：`metrics.csv`、`images.csv`、`evaluation.json`、测试文件哈希，以及每类 `.npz`（原始分数、掩码、标签、相对路径）。这些用于后续缺陷大小、热力图与失败案例分析；分析结论须等待真实实验。每次只在 CPU 累积一个类别的预测，避免将全部测试图驻留 GPU。

## 当前检查与交付边界

```bash
python tests/test_workflow.py
```

检查内容见 [阶段验收](docs/PRE_SERVER_ACCEPTANCE.md)。测试文件的合成指标不进入 results，不提供任何可写入论文的真实结果。

公开仓库默认名称 `dinomaly-cv-coursework`。发布流程及权重 Releases 清单要求见 [GitHub 准备](docs/GITHUB.md)。论文 Word/PDF 在真实实验结束后编写，含个人信息的论文默认不公开。
