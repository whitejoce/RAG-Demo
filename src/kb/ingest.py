from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.models import ZhipuEmbeddings


def _normalize_document(doc: Document, source_path: Path) -> Document:
    metadata = dict(doc.metadata or {})
    metadata["source"] = source_path.name
    metadata["source_path"] = str(source_path)
    metadata["page"] = str(metadata.get("page", ""))
    doc.metadata = metadata
    return doc


def load_documents(raw_dir: Path) -> list[Document]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw document directory not found: {raw_dir}")

    docs: list[Document] = []
    for path in sorted(raw_dir.rglob("*")):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif suffix in {".txt", ".md"}:
            loader = TextLoader(str(path), encoding="utf-8")
        else:
            continue
        loaded = loader.load()
        docs.extend(_normalize_document(doc, path) for doc in loaded)

    if not docs:
        raise ValueError(f"No supported documents found in {raw_dir}")
    return docs


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata = dict(chunk.metadata or {})
        chunk.metadata["chunk_id"] = str(index)
    return chunks


def build_embeddings(settings) -> ZhipuEmbeddings:
    return ZhipuEmbeddings(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.embed_model,
        dimensions=settings.embed_dimensions,
    )


def _index_exists(index_dir: Path) -> bool:
    return (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists()


def build_vectorstore(raw_dir: Path, index_dir: Path, embeddings: ZhipuEmbeddings) -> FAISS:
    documents = load_documents(raw_dir)
    chunks = split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    return vectorstore


def load_vectorstore(index_dir: Path, embeddings: ZhipuEmbeddings) -> FAISS:
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_or_load_vectorstore(
    raw_dir: Path,
    index_dir: Path,
    embeddings: ZhipuEmbeddings,
    rebuild: bool = False,
) -> FAISS:
    if not rebuild and _index_exists(index_dir):
        return load_vectorstore(index_dir, embeddings)
    return build_vectorstore(raw_dir, index_dir, embeddings)

