# Lamarckian Knowledge Base - 拉马克式知识库

## 🚀 安装配置

### 环境要求

- Python >= 3.10
- DashScope API Key（用于访问通义千问模型和向量嵌入）

### 安装依赖

```bash
pip install langchain-community
pip install langchain-chroma
pip install dashscope
```

或者使用 conda：

```bash
conda install -c conda-forge langchain-community langchain-chroma
pip install dashscope
```

### 配置 API Key

#### 方法 1：环境变量（推荐）

```bash
export DASHSCOPE_API_KEY='your_api_key_here'
```

```bash
python test_full_workflow.py
```