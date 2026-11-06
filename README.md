# Dinomaly 课程实验复现

在 MVTec AD 全部 15 类上复现 CVPR 2025 Dinomaly。每次训练使用全类别正常图像，冻结 DINOv2 ViT-B/14 register 编码器，训练 10,000 个优化器更新步。输入先缩放到 448×448，再中心裁剪至 392×392，批大小为 16。

## 实验结果

四项指标均为 15 类宏平均。下表单位为 %，标准差为三个种子的样本标准差。

| 配置 | 图像 AUROC | 像素 AUROC | 像素 AP | AUPRO |
|---|---:|---:|---:|---:|
| B | 98.404 ± 0.019 | 97.189 ± 0.011 | 62.941 ± 0.067 | 92.994 ± 0.032 |
| F | 99.622 ± 0.006 | 98.333 ± 0.010 | 68.860 ± 0.222 | 94.625 ± 0.019 |

完整模型 F 相对基线 B 的同种子平均提升依次为 1.218、1.144、5.919 和 1.631 个百分点。

种子 1 的五组对照如下。A1、A2、A3 分别从 F 移除噪声、换用 Softmax 注意力和换用普通全局余弦损失。A3 的图像 AUROC 略高于 F，其他三项指标较低。

| 配置 | 图像 AUROC | 像素 AUROC | 像素 AP | AUPRO |
|---|---:|---:|---:|---:|
| B | 98.427 | 97.191 | 62.980 | 93.025 |
| F | 99.623 | 98.342 | 68.672 | 94.618 |
| A1 | 99.220 | 97.537 | 62.953 | 93.288 |
| A2 | 99.542 | 98.122 | 68.175 | 94.253 |
| A3 | 99.627 | 98.211 | 68.262 | 94.532 |

[九次运行逐类结果](results/runs.csv)保留 0–1 原始数值。原始掩码在中心裁剪后变空的样本保留原始图像级标签及测试划分。

## 从头复现

实验环境为 Linux x86_64、RTX 3090 24 GB、Python 3.8.20、PyTorch 1.12.0+cu113 和 torchvision 0.13.0+cu113。

```bash
git clone https://github.com/hyzf12321/dinomaly-cv-coursework.git
cd dinomaly-cv-coursework
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install pip==24.3.1 setuptools==75.3.2 wheel==0.45.1 numpy==1.18.4
python -m pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
python tests/test_workflow.py
```

实验使用 [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) 和官方 DINOv2 register 编码器。以下命令下载实际使用的数据镜像版本及编码器，并验证 SHA-256。

```bash
mkdir -p data/mvtec weights
curl -fL 'https://huggingface.co/datasets/micguida1/mvtech_anomaly_detection/resolve/eb692b36147a3b3f12c561b879a062104c1f20b8/mvtec_anomaly_detection.tar.xz' -o data/mvtec.tar.xz
curl -fL 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth' -o weights/dinov2_vitb14_reg4_pretrain.pth
sha256sum -c <<'SHA256'
cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d  data/mvtec.tar.xz
73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71  weights/dinov2_vitb14_reg4_pretrain.pth
SHA256
tar -xJf data/mvtec.tar.xz -C data/mvtec
```

数据根目录下应直接包含 `bottle/`、`cable/` 等类别目录。若解包后多一层目录，把该层下的类别目录放到 `data/mvtec/`。

```bash
bash reproduce.sh
```

脚本先检查数据，再依次训练并评测 B/F 种子 1、A1/A2/A3 种子 1、B/F 种子 2 和 B/F 种子 3。每次使用固定训练终点。完成后比较 `runs/reproduction/summary/runs.csv` 与 [results/runs.csv](results/runs.csv)。`aggregate.csv` 给出均值和样本标准差，`paired.csv` 给出同种子差值。

## 用已训练权重评测

[权重清单](results/weights.json)记录九份终点权重的配置、种子、步数、下载地址与 SHA-256。权重见 [GitHub Release](https://github.com/hyzf12321/dinomaly-cv-coursework/releases/tag/mvtec-ad-nine-runs)。

```bash
mkdir -p weights
curl -fL 'https://github.com/hyzf12321/dinomaly-cv-coursework/releases/download/mvtec-ad-nine-runs/F-s1-inference.pt' -o weights/F-s1-inference.pt
printf '%s  %s\n' 'd9017e9dd98b8f5f1e42a1057f3926a103bdf9624783b89153ed1f225dc917c5' weights/F-s1-inference.pt | sha256sum -c
python run.py evaluate --checkpoint weights/F-s1-inference.pt --data data/mvtec --device cuda:0 --output runs/F-s1-release-eval
```

评测从特征重建误差生成异常图，在 256×256 尺寸计算像素指标，并取异常图最高 1% 像素的均值作为图像分数。正式运行的训练与评测源码来自提交 `11a278013856ce140e8cbdebe20ad275d5100410`，仓库保留这些源码和配置的字节内容，以通过 checkpoint 源码指纹校验。

## 来源与许可

项目自写代码采用 [MIT](LICENSE)。`vendor/` 保留第三方代码的原有许可和版权声明，来源见 [vendor/SOURCE.json](vendor/SOURCE.json)。推理权重包含采用 Apache-2.0 许可的 DINOv2 编码器。

- Guo 等，[Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_Dinomaly_The_Less_Is_More_Philosophy_in_Multi-Class_Unsupervised_Anomaly_CVPR_2025_paper.html)，CVPR 2025。
- [Dinomaly 上游源码](https://github.com/guojiajeremy/Dinomaly/tree/1f252be03a918789b19848f0ca37166e7c28dada)和 [DINOv2 源码及模型许可](https://github.com/facebookresearch/dinov2#license)。
