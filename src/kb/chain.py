from __future__ import annotations

from collections import OrderedDict
from typing import Any

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from src.models import ZhipuChatModel
from src.settings import load_settings
from .ingest import build_embeddings, build_or_load_vectorstore


SYSTEM_PROMPT = (
    "你是企业政企知识库助手。"
    "只能根据给定资料回答；如果资料不足，直接说明资料中未找到。"
    "回答要简洁、准确，并尽量在结尾标注来源文件名。"
)


def build_llm(settings=None) -> ZhipuChatModel:
    settings = settings or load_settings()
    return ZhipuChatModel(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model_name=settings.chat_model,
    )


def build_retriever(settings=None):
    settings = settings or load_settings()
    embeddings = build_embeddings(settings)
    vectorstore = build_or_load_vectorstore(
        settings.raw_dir,
        settings.index_dir,
        embeddings,
    )
    return vectorstore.as_retriever(search_kwargs={"k": settings.top_k})


def build_qa_chain(retriever=None, llm=None, settings=None):
    settings = settings or load_settings()
    retriever = retriever or build_retriever(settings)
    llm = llm or build_llm(settings)
    document_prompt = PromptTemplate.from_template(
        "来源: {source} 页码: {page}\n内容: {page_content}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "问题：{input}\n\n请仅依据以下资料回答：\n{context}"),
        ]
    )
    combine_chain = create_stuff_documents_chain(
        llm,
        prompt,
        document_prompt=document_prompt,
    )
    return create_retrieval_chain(retriever, combine_chain)


def _unique_sources(documents: list[Document]) -> list[dict[str, Any]]:
    seen: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
    for doc in documents:
        source = str(doc.metadata.get("source", "unknown"))
        page = str(doc.metadata.get("page", ""))
        key = (source, page)
        if key in seen:
            continue
        snippet = doc.page_content.strip().replace("\n", " ")
        seen[key] = {
            "source": source,
            "page": page,
            "snippet": snippet[:200],
        }
    return list(seen.values())


def answer_question(question: str, settings=None) -> dict[str, Any]:
    if not question.strip():
        raise ValueError("question cannot be empty")
    settings = settings or load_settings()
    retriever = build_retriever(settings)
    llm = build_llm(settings)
    chain = build_qa_chain(retriever=retriever, llm=llm, settings=settings)
    result = chain.invoke({"input": question})
    docs = result.get("context", [])
    return {
        "question": question,
        "answer": result["answer"],
        "sources": _unique_sources(docs),
    }

