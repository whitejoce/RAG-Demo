from __future__ import annotations

from typing import Any, Iterable

import requests
from pydantic import Field

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _message_role(message: BaseMessage) -> str:
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    return "user"


def _message_content(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


class ZhipuEmbeddings(Embeddings):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://open.bigmodel.cn/api/paas/v4",
        model: str = "embedding-2",
        dimensions: int | None = None,
        timeout: float = 60.0,
        batch_size: int = 16,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.dimensions = dimensions
        self.timeout = timeout
        self.batch_size = batch_size

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            _join_url(self.base_url, "/embeddings"),
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(f"Embedding request failed: {response.text}") from exc
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"Embedding API error: {data['error']}")
        return data

    def _embed(self, texts: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {"model": self.model, "input": texts}
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        data = self._post(payload)
        rows = sorted(data.get("data", []), key=lambda item: item.get("index", 0))
        return [list(map(float, row["embedding"])) for row in rows]

    def _batched(self, texts: Iterable[str]) -> list[list[float]]:
        items = list(texts)
        vectors: list[list[float]] = []
        for start in range(0, len(items), self.batch_size):
            vectors.extend(self._embed(items[start : start + self.batch_size]))
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._batched(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._batched([text])[0]


class ZhipuChatModel(BaseChatModel):
    api_key: str = Field(...)
    base_url: str = Field(default="https://open.bigmodel.cn/api/paas/v4")
    model_name: str = Field(default="glm-4-flash")
    temperature: float = Field(default=0.0)
    max_tokens: int | None = Field(default=None)
    timeout: float = Field(default=60.0)

    @property
    def _llm_type(self) -> str:
        return "zhipu-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "base_url": self.base_url}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": _message_role(message), "content": _message_content(message)}
                for message in messages
            ],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if stop:
            payload["stop"] = stop
        payload.update(kwargs)
        response = requests.post(
            _join_url(self.base_url, "/chat/completions"),
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(f"Chat request failed: {response.text}") from exc
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"Chat API error: {data['error']}")
        content = data["choices"][0]["message"]["content"]
        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation], llm_output=data)

