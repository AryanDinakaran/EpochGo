<div align="center">

# ⚡ EpochGo

**自主、本地优先的多智能体自动化机器学习 (AutoML) 平台**  
*由 [Epochlypse Research](https://epochlypse.com/) 开发 &bull; 基于 CrewAI、Ollama 以及 Scikit-Learn / XGBoost / LightGBM / CatBoost*

[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![CrewAI](https://img.shields.io/badge/CrewAI-Multi--Agent-FF4B4B)](https://crewai.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?logo=ollama)](https://ollama.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Release](https://img.shields.io/badge/Release-v1.1.0-brightgreen.svg)](https://github.com/AryanDinakaran/EpochGo/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

*零 API 密钥。零云端算力账单。100% 运行在本地设备，彻底保护数据隐私。*

<p align="center">
  <a href="README.md">English</a> | <b>简体中文</b>
</p>

<br/>

<img src="assets/demo.gif" alt="EpochGo 演示动图" width="920"/>

</div>

---

## 🌟 项目概述

**EpochGo** 是由 **Epochlypse Research** 开发的开源、本地优先（Local-First）AutoML 平台。只需 `git clone` 仓库并通过 Docker 运行，即可在浏览器中访问本地控制面板 (`http://localhost:8000`)。你可以自由创建项目、上传原始 CSV 数据集，并交由 **3 个自主协作的 AI 智能体团队** 全权负责数据科学研发全流程：

1. 🧹 **智能体 1: 数据预处理与特征架构师 (PreprocessorAgent)**
   - 自动检测数据模式、缺失值处理、基于 IQR 准则截断异常值、分类变量动态编码、特征缩放以及高维多重共线性处理 (PCA)。
   - 严格将完整流水线保存为可复现的 `preprocessing.py` 脚本及序列化资产 `preprocessor.joblib`。

2. 🧪 **智能体 2: 特征工程智能体 (FeatureEngineerAgent)**
   - 自动提取时间序列特征（周期、月份、星期）、多项式交互项、非线性变换、对数变换及聚合统计特征。
   - 筛选出高信噪比特征子集，并持久化为 `feature_engineering.py`。

3. 🤖 **智能体 3: 竞赛级模型训练智能体 (ModelTrainerAgent)**
   - 针对你的业务目标（如二分类、多分类或连续值回归），同时训练并优化 5 种不同算法：
     - **随机森林 (Random Forest)**
     - **梯度提升树 (Gradient Boosting)**
     - **极度梯度提升 (XGBoost)**
     - **轻量梯度提升 (LightGBM)**
     - **CatBoost**
   - 自动进行 5 折分层交叉验证 (Stratified K-Fold CV) 与超参数网格调优，筛选出综合评估指标最佳的模型。
   - 自动生成符合企业标准的高性能 **FastAPI 独立推理服务 (`serve.py`)**，支持即刻上线部署。

---

## ⚡ v1.1.0 版本核心特性（企业级工业加固）

基于社区高级机器学习专家的同行评审和实际脏数据的压力测试：

* 🛡️ **自动主键与无用 ID 剔除**: 自动识别非预测性主键、UUID 以及行序号 (`uniqueness_ratio > 0.95` 或 ID 语义命名 + 唯一性 > 0.80)，并在流水线拟合前自动剔除，防止梯度提升树拟合并记住随机哈希噪声。
* 🎯 **折内目标编码 (`TargetEncoder(cv=5)`)**: 针对高基数类别特征 ($20 < N_{unique} \le 500$)，自动使用 Scikit-Learn 原生 5 折交叉验证的目标编码器 (`cv=5, smooth='auto'`)，彻底杜绝验证集目标泄露（Target Leakage）。
* 💾 **彻底消除 4 万类别导致的内存溢出 (OOM)**: 不再像朴素 OneHotEncoder 那样生成数万列极度消耗内存的稀疏矩阵，而是将高基数类别压缩为单列连续浮点向量，**内存占用骤降 >99.9%**，确保在 16GB 内存的个人电脑上稳定运行数十万行数据集。
* 🔗 **稳健的特征名动态解析**: 采用原生 `get_feature_names_out()` 支持复杂 `ColumnTransformer`，确保下游推理脚本 (`serve.py`) 能够稳定反序列化并正确响应请求。

---

## 🚀 快速上手（单命令运行）

### 环境要求
1. **Docker 与 Docker Compose**（推荐运行方式）
2. **本地 Ollama 服务**：
   ```bash
   ollama pull qwen2.5-coder:7b
   ```

### 1. 克隆代码仓库
```bash
git clone https://github.com/AryanDinakaran/EpochGo.git
cd EpochGo
```

### 2. 配置环境变量
```bash
cp .env.example .env
```
*(默认配置已预设连接本地 `http://localhost:11434`，零 API Key 需求)*

### 3. 一键启动
```bash
docker compose up --build
```

在浏览器中打开 **`http://localhost:8000`**，即可体验完全属于你自己的本地离线 AutoML 平台！

---

## 💻 本地 Python 原生运行（非 Docker 模式）

如果你偏好直接在宿主机 Python 虚拟环境中运行：

```bash
# 1. 创建并激活虚拟环境
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 安装依赖项
pip install -r requirements.txt

# 3. 确保本地 Ollama 处于启动状态
ollama run qwen2.5-coder:7b

# 4. 启动应用
uvicorn epoch_go.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🛡️ 数据隐私与合规性（零数据外泄）

| 特性 | 云端 SaaS AutoML (DataRobot, AWS) | EpochGo |
| :--- | :--- | :--- |
| **API 成本** | 每月数百至数千美元 | **$0 / 永久免费** |
| **数据流向** | 上传至第三方远程服务器 | **100% 保留在本地内网** |
| **合规性** | 面临数据出境与跨境传输合规风险 | **完全符合当地数据保护法案 (如印度 DPDP、欧盟 GDPR)** |
| **互联网需求** | 必须全程联网 | **支持 100% 物理隔离断网环境 (Air-gapped)** |
| **底层模型掌控** | 依赖专有黑盒模型 | **基于开源 Qwen2.5-Coder，完全透明可审计** |

---

## 📊 v1.0 与 v1.1 高基数编码基准对比

| 指标 | v1.0.0 (朴素 One-Hot 独热编码) | v1.1.0 (折内 TargetEncoder 目标编码) | 性能提升 |
| :--- | :--- | :--- | :--- |
| **列数分配 (5,000 分类类别)** | 5,000 列稀疏矩阵 | **1 列浮点向量** | **维度降低 99.98%** |
| **内存占用 (10 万行数据集)** | 约 4.2 GB | **约 8 MB** | **节省 >99% 内存** |
| **16GB 个人设备崩溃风险** | 极高 (触发 OOM 强制杀进程) | **0% (持续稳定运行)** | **工业级高可靠** |
| **主键过拟合 / 泄露风险** | 存在高风险 | **已彻底消除 (自动检测并剔除)** | **生产级安全泛化** |

---

## 🏛️ 项目维护与开发团队

**[Epochlypse Research](https://epochlypse.com/)**  
- **首席架构师**: Aryan Dinakaran  
- **团队愿景**: 构建本地优先、保护隐私且高度可复现的多智能体系统，服务于高杠杆数据科学与机器学习工程。  
- **联络与交流**:
  - 🌐 官方网站: [epochlypse.com](https://epochlypse.com/)
  - ✉️ 电子邮箱: [aryan@epochlypse.com](mailto:aryan@epochlypse.com)
  - 💼 LinkedIn: [Aryan Dinakaran](https://www.linkedin.com/in/aryan-dinakaran-462277230/)
  - 📸 Instagram: [@aryandinakaran](https://www.instagram.com/aryandinakaran/)
  - 🐙 GitHub: [@AryanDinakaran](https://github.com/AryanDinakaran)

---

## 📄 开源许可证

本项目采用 MIT 许可证。免费用于个人学习、商业落地及学术研究。
