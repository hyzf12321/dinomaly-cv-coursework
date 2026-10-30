# 来源与许可

本项目为课程复现与实验分析，不宣称 Dinomaly 方法原创。

上游：Jia Guo 等，Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection，CVPR 2025，20405–20415。
官方仓库：https://github.com/guojiajeremy/Dinomaly
固定提交：1f252be03a918789b19848f0ca37166e7c28dada。

`vendor/dinomaly/` 是该提交的运行相关文件子集；原始许可证、README、requirements 及训练入口一并保留。`vendor/SOURCE.json` 记录每个文件的 SHA256，测试核对副本未经修改。

上游仓库根许可证为 Apache-2.0，副本见 `vendor/dinomaly/LICENSE`；本项目新增运行支持代码采用根目录的 Apache-2.0 许可。第三方文件内原有版权、引用和许可证声明均保留，包括 DINO/DINOv2 与优化器来源；根许可证不覆盖第三方自身声明。

新增：`course/`、`run.py`、`configs/`、`tests/`、文档及仓库忽略规则。模型主体、瓶颈、注意力、损失、优化器和 AUPRO 均复用上游。新增评测入口沿用 `utils.evaluation_batch` 的协议，改为逐批移到 CPU 并保存可追溯文件，没有将其表述为新评价方法。

MVTec AD 数据与 DINOv2 预训练权重分别受其原始许可约束，不随本代码仓库重新授权。数据不上传仓库；后续 Release 发布自己训练的模型时应附来源、配置、校验值和适用许可说明。
