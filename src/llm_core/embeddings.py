import hashlib
import logging

from langchain_core.embeddings import Embeddings

from app.settings import backend_settings
from domain.storage.json_store import read_json, write_json
from llm_core.client import create_llm_client
from llm_core.settings import llm_settings

logger = logging.getLogger(__name__)


class ProviderEmbeddings(Embeddings):
    """Реализует LangChain Embeddings поверх текущего LLM-провайдера."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Строит embedding-векторы для списка документов.

        Args:
            texts: Тексты документов для векторизации.

        Returns:
            Список embedding-векторов в порядке входных текстов.
        """
        if not texts:
            return []
        return create_llm_client().embed(texts)

    def embed_query(self, text: str) -> list[float]:
        """Строит embedding-вектор поискового запроса.

        Args:
            text: Текст запроса.

        Returns:
            Embedding-вектор запроса.
        """
        return self.embed_documents([text])[0]


class CachingEmbeddings(Embeddings):
    """Оборачивает Embeddings персистентным кешем по хешу контента."""

    def __init__(self, inner: Embeddings) -> None:
        """Создаёт кеширующую обёртку.

        Args:
            inner: Провайдерная реализация Embeddings для промахов кеша.
        """
        self.inner = inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Строит embedding-векторы, переиспользуя персистентный кеш.

        Args:
            texts: Тексты документов для векторизации.

        Returns:
            Список embedding-векторов в порядке входных текстов.
        """
        if not texts:
            return []

        cache: dict[str, list[float]] = read_json(backend_settings.embeddings_cache_file, {})
        keys = [_cache_key(text) for text in texts]
        miss_indexes = [index for index, key in enumerate(keys) if key not in cache]
        if miss_indexes:
            vectors = self.inner.embed_documents([texts[index] for index in miss_indexes])
            for index, vector in zip(miss_indexes, vectors, strict=True):
                cache[keys[index]] = vector
            write_json(backend_settings.embeddings_cache_file, cache)
            logger.debug(
                "Embeddings cache updated",
                extra={"miss_count": len(miss_indexes), "total_count": len(texts)},
            )
        return [cache[key] for key in keys]

    def embed_query(self, text: str) -> list[float]:
        """Строит embedding-вектор запроса, переиспользуя кеш.

        Args:
            text: Текст запроса.

        Returns:
            Embedding-вектор запроса.
        """
        return self.embed_documents([text])[0]


def create_embeddings() -> Embeddings:
    """Создаёт Embeddings текущего провайдера с персистентным кешем.

    Returns:
        Кеширующую обёртку над провайдерными embeddings.
    """
    return CachingEmbeddings(ProviderEmbeddings())


def _cache_key(text: str) -> str:
    """Строит ключ кеша, привязанный к провайдеру и embedding-модели.

    Args:
        text: Исходный текст.

    Returns:
        Ключ вида `provider::model::sha256(text)`.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{llm_settings.normalized_provider}::{llm_settings.normalized_embedding_model}::{digest}"
