# GitHub 与 Releases 准备

公开仓库：[hyzf12321/dinomaly-cv-coursework](https://github.com/hyzf12321/dinomaly-cv-coursework)。
2026-10-30 用户通过网页创建仓库并初始化 README。首次代码上传通过已连接的 GitHub 插件建立文件树、创建提交并更新 main，保留网页的初始提交，不需要用户提供密码或 Token。

本地代码根目录已独立初始化为 Git 仓库，只提交本目录内的实际代码、五组 JSON、运行矩阵、依赖清单、来源哈希及文档。不要把大作业总目录、环境兼容性检查中的 Python 运行时或含个人信息的论文加入仓库。

常规 Git 推送是后续维护的另一种方式，需要本机单独完成 GitHub 认证。插件连接不自动提供命令行凭据。配置远端后，已认证环境可执行：

```bash
git push -u origin main
```

发布核验使用远端文件树 SHA 与本地 Git 树 SHA 对照，覆盖全部路径与文件内容，并确认仓库为 public。源码、配置及说明进入仓库；数据、环境、缓存与权重不进入 Git。插件提交与本地早期提交可能具有不同提交 SHA，文件内容是否一致以树 SHA 为准。

## 训练后发布

真实实验完成并在干净目录重载评测后，再创建正式 Release，不发布随机权重测试文件。B/F/A1/A2/A3 的每个实际完成种子均上传单独命名的 `inference.pt`；优化器状态 `last.pt` 用于两地备份，不作为推理下载的必需文件。

权重清单至少包含：variant、seed、steps、encoder、代码提交、配置文件、训练数据清单哈希、预训练编码器哈希、推理文件名、文件 SHA256、Release 下载 URL 和对应 evaluation.json。清单记录真实完成情况，不提前填充链接或实验值。

发布前检查：Git 跟踪文件里没有 data/weights/runs/.cache/.venv、凭据或论文；Release 文件从公开链接重新下载后 SHA256 相符；用仓库记录的源码和配置重新加载权重，逐类与宏平均和汇总表相符。原文 20,000 步公开权重若使用，只能另标参考。
