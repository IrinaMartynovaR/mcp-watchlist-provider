import logging
from pathlib import Path

from langchain_core.embeddings import Embeddings

from domain.storage.json_store import read_json, update_json
from llm_core.client import LLMClient
from llm_core.parsing import content_cache_key

logger = logging.getLogger(__name__)


class ProviderEmbeddings(Embeddings):
    """Реализует LangChain Embeddings поверх текущего LLM-провайдера."""

    def __init__(self, client: LLMClient) -> None:
        """Создаёт provider-адаптер.

        Args:
            client: Настроенный LLM-клиент с поддержкой embeddings.
        """
        self.client = client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Строит embedding-векторы для списка документов.

        Args:
            texts: Тексты документов для векторизации.

        Returns:
            Список embedding-векторов в порядке входных текстов.
        """
        if not texts:
            return []
        return self.client.embed(texts)

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

    def __init__(self, inner: Embeddings, cache_file: Path, cache_namespace: str) -> None:
        """Создаёт кеширующую обёртку.

        Args:
            inner: Провайдерная реализация Embeddings для промахов кеша.
            cache_file: Путь к персистентному JSON-кешу.
            cache_namespace: Префикс ключей кеша.
        """
        self.inner = inner
        self.cache_file = cache_file
        self.cache_namespace = cache_namespace

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Строит embedding-векторы, переиспользуя персистентный кеш.

        Args:
            texts: Тексты документов для векторизации.

        Returns:
            Список embedding-векторов в порядке входных текстов.
        """
        if not texts:
            return []

        cache: dict[str, list[float]] = read_json(self.cache_file, {})
        keys = [self._cache_key(text) for text in texts]
        miss_indexes = [index for index, key in enumerate(keys) if key not in cache]
        if miss_indexes:
            vectors = self.inner.embed_documents([texts[index] for index in miss_indexes])
            additions: dict[str, list[float]] = {}
            for index, vector in zip(miss_indexes, vectors, strict=True):
                additions[keys[index]] = vector

            def merge(current: dict[str, list[float]]) -> dict[str, list[float]]:
                current.update(additions)
                return current

            cache = update_json(self.cache_file, {}, merge)
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

    def _cache_key(self, text: str) -> str:
        """Строит ключ кеша для одного текста.

        Args:
            text: Исходный текст.

        Returns:
            Ключ вида ``namespace::sha256(text)``.
        """
        return content_cache_key(self.cache_namespace, text)


def create_embeddings(client: LLMClient, cache_file: Path, cache_namespace: str) -> Embeddings:
    """Создаёт Embeddings текущего провайдера с персистентным кешем.

    Args:
        client: Настроенный LLM-клиент.
        cache_file: Путь к JSON-кешу embedding-векторов.
        cache_namespace: Префикс ключей кеша.

    Returns:
        Кеширующую обёртку над провайдерными embeddings.
    """
    return CachingEmbeddings(ProviderEmbeddings(client), cache_file, cache_namespace)
