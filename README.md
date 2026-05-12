# 企业政企知识库 MVP

> ⚙️ code with GPT-5.5

一个最小可跑通的 Python + LangChain + 智谱 AI 知识库原型。

主要完成：
- LangChain 知识库流水线：[src/kb/ingest.py](./src/kb/ingest.py)、[src/kb/chain.py](./src/kb/chain.py)
- 智谱 Embedding + Chat 封装：[src/models/zhipu.py](./src/models/zhipu.py)
- CLI：[src/cli.py](./src/cli.py)
- MCP Server 和客户端演示：[src/mcp_server.py](./src/mcp_server.py)、[src/mcp_client_demo.py](./src/mcp_client_demo.py)
- 样本文档目录：[data/raw](./data/raw)
- 分层评测：[eval](./eval)

## 目录

- `data/raw/` 原始样本文档
- `data/index/` 本地 FAISS 索引
- `src/` 核心代码
- `eval/` 一套可直接扩展的小型评测模板

## 安装

```bash
pip install -r requirements.txt
```

## 环境变量

```bash
set ZHIPU_API_KEY=你的key
set ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4
set ZHIPU_CHAT_MODEL=glm-4-flash
set ZHIPU_EMBED_MODEL=embedding-2
```

## 使用

构建索引:

```bash
python -m src.cli build-index
```

直接问答:

```bash
python -m src.cli ask "企业宽带开通需要多久？"
```

启动 MCP Server:

```bash
python -m src.mcp_server
```

运行 MCP 客户端演示:

```bash
python -m src.mcp_client_demo
```

## 演示问题

- 企业宽带开通周期是多久？
- 光缆套餐包含哪些服务？
- 专线故障怎么报修？

## 学习文档

- [面试学习文档](docs/面试学习文档.md)
- [RAG零基础教程](docs/RAG零基础教程.md)
- [评测模板说明](eval/README.md)

## 评测

检索层评测:

```bash
python -m eval.retrieval_eval
```

端到端 RAG 评测:

```bash
python -m eval.rag_eval
```

固定上下文 LLM 评测:

```bash
python -m eval.llm_eval
```

Agent / MCP Tool 评测:

```bash
python -m eval.agent_eval
```

