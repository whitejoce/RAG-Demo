# 企业政企知识库 MVP

一个最小可跑通的 Python + LangChain + 智谱 AI 知识库原型。

主要完成：
- 新增 LangChain 知识库流水线：[src/kb/ingest.py](D:/Code/test_codex/src/kb/ingest.py)、[src/kb/chain.py](D:/Code/test_codex/src/kb/chain.py)
- 新增智谱 Embedding + Chat 封装：[src/models/zhipu.py](D:/Code/test_codex/src/models/zhipu.py)
- 新增 CLI：[src/cli.py](D:/Code/test_codex/src/cli.py)
- 新增 MCP Server 和客户端演示：[src/mcp_server.py](D:/Code/test_codex/src/mcp_server.py)、[src/mcp_client_demo.py](D:/Code/test_codex/src/mcp_client_demo.py)
- 新增样本文档目录：[data/raw](D:/Code/test_codex/data/raw)
- 更新运行说明：[README.md](D:/Code/test_codex/README.md)


## 目录

- `data/raw/` 原始样本文档
- `data/index/` 本地 FAISS 索引
- `src/` 核心代码

## 安装

```bash
pip install -r requirements.txt
```

## 环境变量

```bash
set ZHIPU_API_KEY=你的key
set ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4
set ZHIPU_CHAT_MODEL=glm-4-flash
set ZHIPU_EMBED_MODEL=embedding-3
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

