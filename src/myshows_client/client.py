import logging
from typing import Any

import httpx

from myshows_client.settings import MyShowsSettings

logger = logging.getLogger(__name__)


class MyShowsClient:
    """Реализует минимальный клиент неофициального JSON-RPC API MyShows.me.

    В отличие от recommendation-пайплайна клиент намеренно не fail-soft:
    импорт истории — разовая операция под присмотром пользователя,
    поэтому любые ошибки должны быть видны сразу, а не проглатываться.
    """

    def __init__(self, settings: MyShowsSettings, transport: httpx.BaseTransport | None = None) -> None:
        """Создаёт клиента с настройками доступа к MyShows.

        Args:
            settings: Логин, пароль и таймаут MyShows API.
            transport: Необязательный HTTP transport для тестов.
        """
        self.settings = settings
        self.transport = transport
        self._token: str | None = None

    def _login(self) -> str:
        """Получает bearer-токен по логину и паролю пользователя.

        Токен кешируется на инстансе: повторные вызовы не выполняют новый логин.

        Returns:
            Bearer-токен для последующих JSON-RPC вызовов.

        Raises:
            RuntimeError: Если креды не заданы, запрос завершился ошибкой
                сети или HTTP либо в ответе нет токена.
        """
        if self._token:
            return self._token
        if not self.settings.login or not self.settings.password:
            raise RuntimeError("MYSHOWS_LOGIN and MYSHOWS_PASSWORD are not configured")

        logger.info("MyShows login started", extra={"login": self.settings.login})
        data = self._post_json(
            self.settings.session_url,
            json_body={"login": self.settings.login, "password": self.settings.password},
            error_prefix="MyShows login",
            http_error_hint=". Check MYSHOWS_LOGIN/MYSHOWS_PASSWORD.",
        )

        token = data.get("token") if isinstance(data, dict) else None
        if not isinstance(token, str) or not token:
            raise RuntimeError("MyShows login response contains no token")

        self._token = token
        logger.info("MyShows login succeeded")
        return token

    def _rpc(self, method: str, params: dict[str, Any]) -> Any:
        """Выполняет один JSON-RPC вызов к MyShows.

        Args:
            method: Имя JSON-RPC метода, например `profile.WatchedMovies`.
            params: Параметры вызова.

        Returns:
            Поле `result` первого элемента batch-ответа.

        Raises:
            RuntimeError: Если HTTP-запрос завершился ошибкой сети или статуса,
                ответ не является JSON либо содержит JSON-RPC `error`.
        """
        token = self._login()
        body = [{"jsonrpc": "2.0", "method": method, "params": params, "id": 1}]
        data = self._post_json(
            self.settings.rpc_url,
            json_body=body,
            headers={"authorization2": f"Bearer {token}"},
            error_prefix=f"MyShows RPC {method}",
        )

        entries = data if isinstance(data, list) else [data]
        if not entries or not isinstance(entries[0], dict):
            raise RuntimeError(f"MyShows RPC {method} returned a malformed response")

        entry = entries[0]
        if entry.get("error"):
            raise RuntimeError(f"MyShows RPC {method} returned an error: {entry['error']}")
        return entry.get("result")

    def get_watched_movies(self, page: int = 0) -> list[dict[str, Any]]:
        """Возвращает одну страницу просмотренных фильмов пользователя.

        Args:
            page: Номер страницы, начиная с 0.

        Returns:
            Список фильмов страницы; пустой список означает конец истории.

        Raises:
            RuntimeError: Если вызов API завершился ошибкой.
        """
        result = self._rpc(
            "profile.WatchedMovies",
            {
                "page": page,
                "pageSize": self.settings.movies_page_size,
                "login": "",
                "search": {"sort": "watchedAt_desc"},
            },
        )
        return _as_item_list(result)

    def iter_all_watched_movies(self) -> list[dict[str, Any]]:
        """Возвращает все просмотренные фильмы, листая страницы до первой пустой.

        Returns:
            Полный список фильмов из истории просмотров.

        Raises:
            RuntimeError: Если один из вызовов API завершился ошибкой.
        """
        movies: list[dict[str, Any]] = []
        for page in range(self.settings.max_movie_pages):
            page_items = self.get_watched_movies(page=page)
            if not page_items:
                break
            movies.extend(page_items)
        logger.info("MyShows watched movies fetched", extra={"movie_count": len(movies)})
        return movies

    def get_shows(self) -> list[dict[str, Any]]:
        """Возвращает полный список сериалов пользователя со статусами.

        Returns:
            Список сериалов пользователя.

        Raises:
            RuntimeError: Если вызов API завершился ошибкой.
        """
        result = self._rpc("profile.Shows", {"login": ""})
        shows = _as_item_list(result)
        logger.info("MyShows shows fetched", extra={"show_count": len(shows)})
        return shows

    def _post_json(
        self,
        url: str,
        json_body: Any,
        error_prefix: str,
        headers: dict[str, str] | None = None,
        http_error_hint: str = "",
    ) -> Any:
        """Выполняет POST-запрос и возвращает JSON-ответ.

        Общий помощник login- и RPC-вызовов: оборачивает сетевые,
        HTTP- и JSON-ошибки в `RuntimeError` с читаемым сообщением.

        Args:
            url: Полный URL запроса.
            json_body: JSON-тело запроса.
            error_prefix: Префикс сообщений об ошибках, например `MyShows login`.
            headers: Необязательные HTTP-заголовки.
            http_error_hint: Подсказка, добавляемая к сообщению об HTTP-ошибке.

        Returns:
            Десериализованный JSON-ответ.

        Raises:
            RuntimeError: Если запрос завершился ошибкой сети или HTTP-статуса
                либо ответ не является JSON.
        """
        try:
            with httpx.Client(timeout=self.settings.timeout_seconds, transport=self.transport) as client:
                response = client.post(url, json=json_body, headers=headers)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"{error_prefix} failed with HTTP {exc.response.status_code}{http_error_hint}") from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"{error_prefix} request failed: {exc}") from exc

        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError(f"{error_prefix} returned a non-JSON response") from exc


def _as_item_list(result: Any) -> list[dict[str, Any]]:
    """Приводит результат RPC к списку словарей без жёстких предположений о схеме.

    Реальная форма ответа неофициального API не подтверждена, поэтому
    поддерживаются два варианта: сразу список либо объект-обёртка
    со списком под одним из типовых ключей.

    Args:
        result: Поле `result` JSON-RPC ответа.

    Returns:
        Список словарей-элементов; пустой список для неожиданных форм.
    """
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    if isinstance(result, dict):
        for key in ("items", "movies", "shows", "result"):
            value = result.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []
