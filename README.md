# Dinomaly 计算机视觉课程复现

复现与分析 CVPR 2025 Dinomaly，沿用 MVTec AD 全部 15 类的统一模型设置，不将已有机制包装为原创。**已完成服务器基础环境与数据核验；正式实验尚未开始。**

这里的代码是后续实验要使用的程序：按配置训练和保存/恢复模型，用统一协议评测，再汇总真实结果。提前准备代码不代表已经训练；检查用途的少量训练与单类评测单独保存，不作为论文正式结果；正式训练前先交流确认。

**供课程检查：** 仓库公开，老师可直接浏览代码、配置、Git 提交历史和后续真实结果；权重在完成训练并核验后提供 Releases 下载链接。当前入口和设置见下文，结果目录暂不创建。

## 目录

| 位置 | 用途 |
|---|---|
| `run.py` | 数据检查、训练、评测、汇总入口 |
| `configs/` | B、F、A1、A2、A3 五组配置 |
| `course/` | 模型装配、数据读取、训练与断点、固定评测及汇总 |
| `tests/` | CPU 随机权重/合成数据功能检查，不是论文结果 |
| `vendor/` | 实际依赖的上游源码、原许可和 SHA256 来源清单 |
| `requirements.txt` | Linux CUDA 依赖（RTX 3090 基础环境检查已通过） |

说明集中在本文件；真实结果产生后再建立结果目录。数据、环境、缓存和训练输出不进入 Git。

## 实验设置

默认冻结公开预训练 DINOv2 **ViT-Base/14（4 register tokens）**；缩放 448×448，中心裁剪 392×392，batch=16，10,000 个优化器更新步。编码器目标层 2–9、解码器 8 层，分组监督使用 `[0,1,2,3]`、`[4,5,6,7]`。

| 配置 | 瓶颈 Dropout | 注意力 | 监督 | 损失 |
|---|---:|---|---|---|
| B：普通重建基线 | 0 | Softmax | 八层逐层 | 普通全局余弦 |
| F：完整方法 | 0.2 | LinearAttention2 | 两组融合 | 宽松余弦 |
| A1：关闭噪声 | 0 | LinearAttention2 | 两组融合 | 宽松余弦 |
| A2：替换注意力 | 0.2 | Softmax | 两组融合 | 宽松余弦 |
| A3：替换损失 | 0.2 | LinearAttention2 | 两组融合 | 普通全局余弦 |

五组先运行种子 1，B/F 再成对追加种子 2、3，目标九次训练。使用固定终点，不根据测试指标选择模型。作者的 20,000 步权重只能另作参考。

配置字段：`encoder` 为 base/small 主干；`image_size/crop_size` 为缩放/裁剪；`batch_size/steps` 为物理 batch/更新步；`warmup_steps/lr/final_lr` 控制学习率；`weight_decay/grad_clip` 为权重衰减/梯度裁剪；`noise_dropout/attention/supervision/loss` 对应表中机制；`save_every` 为断点间隔；`eval_size/max_ratio` 为评测尺寸/最高分像素比例；`workers` 为加载进程数；`variant` 为实验编号。JSON 不添加注释，避免破坏格式。

## 安装与检查

服务器建议 Linux x86_64、RTX 3090 24GB、Python 3.8、torch 1.12.0+cu113 / torchvision 0.13.0+cu113。服务器已验证 Python 3.8.20、torch 1.12.0+cu113、torchvision 0.13.0+cu113，pip check 和 CUDA 矩阵运算通过。独立环境安装示例：

```bash
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install pip==24.3.1 setuptools==75.3.2 wheel==0.45.1 numpy==1.18.4
python -m pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
python tests/test_workflow.py
```

ptflops 0.7 需要源码构建；先装 torch，再安装其余依赖。pandas 保持 1.3.5，官方 AUPRO 使用 `DataFrame.append`；xFormers 禁用。CPU 实测完整依赖留存在课程项目的环境兼容性检查材料中，不重复放进仓库。

七组 CPU 检查覆盖五种随机 ViT-Small/56×56 路径、与官方评测的数值对照、小型合成网络续训与权重重载、掩码配对、配置保护和十五类合成数据评测/汇总。精简上游副本后必须从干净代码副本重跑这些检查。新增边缘缺陷校验：原始掩码为空时报错；缺陷被既定中心裁剪移除时保留样本并记录路径。

## 数据与运行

服务器到位后再获取 [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) 和[官方 DINOv2 register 编码器](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth)，分别放在 `data/mvtec/`、`weights/`。保留官方 `category/train/good`、`test/defect`、`ground_truth/defect/*_mask.png` 划分，训练数据仅正常样本。遵守原数据及权重许可，不重新上传数据。

以下为正式运行命令（尚未启动正式实验）；B/A1/A2/A3 更换配置名，重复实验更换种子和独立输出目录：

```bash
python run.py validate-data --data data/mvtec --output runs/data-validation.json
python run.py train --config configs/F.json --data data/mvtec --encoder weights/dinov2_vitb14_reg4_pretrain.pth --seed 1 --device cuda:0 --output runs/F-s1
python run.py train --config configs/F.json --data data/mvtec --encoder weights/dinov2_vitb14_reg4_pretrain.pth --seed 1 --device cuda:0 --output runs/F-s1 --resume runs/F-s1/last.pt
python run.py evaluate --checkpoint runs/F-s1/inference.pt --data data/mvtec --device cuda:0 --output runs/F-s1/eval
python run.py summarize runs/B-s1/eval/evaluation.json runs/F-s1/eval/evaluation.json --output runs/summary-s1
```

`validate-data` 检查全部图像、配对掩码和哈希；`defects_removed_by_crop` 记录裁剪后空掩码的异常样本，保留原始图像级标签及划分。该限制须在正式分析中披露。

训练要求本地预训练文件并严格加载。每 500 步滚动原子保存 `last.pt`（模型、优化器、scheduler、随机数和数据位置），终点导出含完整编码器的 `inference.pt`。每次运行记录配置、Git 版本、源码/编码器/数据哈希、环境、损失、会话耗时及显存。续训拒绝配置、种子、代码、数据或编码器不一致；只加载可信 checkpoint。

数据处理无随机增强；使用独立 `seed+epoch` shuffle 和 drop_last，以支持恢复，样本顺序与官方共享全局随机生成器不同，论文须披露。CPU 合成网络续训一致性不等于 CUDA 重现性已验证。宽松损失直接复用官方梯度钩子，`p=min(0.9*step/1000,0.9)`、`factor=0.1`。

检查配置复制到 `runs/`，传 `--purpose probe`。按用户要求缩减为 F 10 步、B 2 步，保留默认输入、batch、主干；仅检查配置的 warmup 改为 1 步，正式配置不变。最短真实评测只检查 bottle 类，不代表全 15 类正式结果。不完整断点评测需 `--allow-incomplete`。这两类结果都不能进入正式主汇总；ViT-Small 降级或缩短预算必须在主对照内统一记录。

## 评测与发布

逐类报告图像 AUROC、像素 AUROC/AP、AUPRO 和十五类宏平均，保存 0–1 原始值。汇总提供跨种子均值、样本标准差（n=1 留空）、B/F 同种子差值和缺失运行清单，拒绝混合数据、编码器、预算、代码或评测协议。

评测沿用固定源码 `evaluation_batch(max_ratio=0.01, resize_mask=256)`：余弦距离图平均，先 bilinear/align_corners=True 到裁剪大小，再 bilinear/False 到 256；掩码使用官方 PIL 插值与中心裁剪，再 nearest 到 256 并 >0 二值化；之后 5×5、sigma=4 高斯平滑，图像分数取最高 1% 像素均值。AUPRO 复用官方 `compute_pro`：200 阈值、FPR<0.3，按采样最大 FPR 归一化；不是另行实现的连续积分版本。指标未定义时报错，不猜测填值。

输出含 `metrics.csv`、`images.csv`、来源 JSON 和逐类 `.npz`（热力图、掩码、标签、路径），供后续缺陷大小、热力图和失败案例分析使用，逐类在 CPU 累积预测。

**Git 管理：** 本地持续提交，完成并检查一个阶段后再同步 GitHub；较大改动用分支，不上传临时调试状态。正式训练记录实际提交和源码指纹，重要阶段建立标签。可用已连接的 GitHub 插件同步；本机 `git push` 另需本机认证，不强制覆盖远端。

**只公开必要产物：** 代码、实际配置、核验后的结果 CSV、必要图表和日志摘要。数据、环境、缓存、凭据、全部运行数组和优化器断点保持本地或单独备份。公开来源记录前检查个人信息和本机路径；论文 Word/PDF 默认不公开。真实实验完成后，推理权重通过 Releases 发布，清单给出配置、种子、更新步、代码版本、SHA256、下载地址和对应评测；发布前从干净目录重载核验。

## 来源、许可与引用

采用宽松的 **Apache-2.0**，与上游保持一致：允许按许可免费使用、复制、修改和再分发，保留原版权、许可证及必要归属说明即可，不额外增加本课程项目的用途限制。许可证已有，不需要等实验开始再配置。公开仓库允许所有人查看、下载、Fork 并修改自己的副本、提交 Pull Request；直接推送到本仓库需要所有者或协作者写权限，这是 GitHub 访问控制，与许可证不同。老师查看评分无需写权限。[许可条款](https://www.apache.org/licenses/LICENSE-2.0.txt) · [GitHub 权限说明](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/permission-levels-for-a-personal-account-repository)。

- [官方仓库](https://github.com/guojiajeremy/Dinomaly)，固定提交 `1f252be03a918789b19848f0ca37166e7c28dada`；[原始训练入口](https://github.com/guojiajeremy/Dinomaly/blob/1f252be03a918789b19848f0ca37166e7c28dada/dinomaly_mvtec_uni.py)。
- Guo, Jia; Lu, Shuai; Zhang, Weihang; Chen, Fang; Li, Huiqi; Liao, Hongen. [Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_Dinomaly_The_Less_Is_More_Philosophy_in_Multi-Class_Unsupervised_Anomaly_CVPR_2025_paper.html). CVPR 2025, 20405–20415。
- `vendor/dinomaly/` 仅保留实际导入的上游源码与原许可证；未修改文件内容。来源和逐文件 SHA256 见 `vendor/SOURCE.json`。保留 DINO/DINOv2、优化器等原有版权与许可声明，根许可证不覆盖第三方自己的声明。
- 新增 `course/`、入口、配置和测试采用 [Apache-2.0](LICENSE)，主要增加运行支持；方法主体、注意力、瓶颈、损失、优化器和 AUPRO 来自上游。本仓库复现 Dinomaly，不是 Dinomaly2，不宣称上述机制原创。
