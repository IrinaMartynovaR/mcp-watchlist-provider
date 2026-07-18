import json
from typing import Any

import httpx
import pytest

from myshows_client.client import MyShowsClient
from myshows_client.settings import MyShowsSettings

SESSION_URL = str(MyShowsSettings.model_fields["session_url"].default)
RPC_URL = str(MyShowsSettings.model_fields["rpc_url"].default)


def _client(
    login: str = "user",
    password: str = "secret",
    transport: httpx.BaseTransport | None = None,
) -> MyShowsClient:
    return MyShowsClient(
        MyShowsSettings(MYSHOWS_LOGIN=login, MYSHOWS_PASSWORD=password, MYSHOWS_TIMEOUT_SECONDS=1),
        transport=transport,
    )


def _response(url: str, payload: Any, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code=status_code, json=payload, request=httpx.Request("POST", url))


def _capture_posts(
    responses: list[httpx.Response | Exception],
) -> tuple[httpx.MockTransport, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        authorization = request.headers.get("authorization2")
        calls.append(
            {
                "url": str(request.url),
                "json": json.loads(request.content),
                "headers": {"authorization2": authorization} if authorization else None,
            }
        )
        outcome = responses[len(calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return httpx.MockTransport(handler), calls


def _login_response(token: str = "token-1") -> httpx.Response:
    return _response(SESSION_URL, {"token": token})


def _rpc_response(result: Any) -> httpx.Response:
    return _response(RPC_URL, [{"jsonrpc": "2.0", "id": 1, "result": result}])


def test_login_posts_credentials_and_returns_token() -> None:
    transport, calls = _capture_posts([_login_response("abc")])

    assert _client(transport=transport)._login() == "abc"
    assert calls == [{"url": SESSION_URL, "json": {"login": "user", "password": "secret"}, "headers": None}]


def test_login_caches_token() -> None:
    transport, calls = _capture_posts([_login_response("abc")])
    client = _client(transport=transport)

    assert client._login() == "abc"
    assert client._login() == "abc"
    assert len(calls) == 1


def test_login_fast_fails_without_credentials() -> None:
    with pytest.raises(RuntimeError, match="MYSHOWS_LOGIN"):
        _client(login="", password="")._login()


def test_login_raises_on_http_error() -> None:
    transport, _ = _capture_posts([_response(SESSION_URL, {"error": "bad credentials"}, status_code=401)])

    with pytest.raises(RuntimeError, match="HTTP 401"):
        _client(transport=transport)._login()


def test_login_raises_on_network_error() -> None:
    transport, _ = _capture_posts([httpx.ConnectError("boom")])

    with pytest.raises(RuntimeError, match="login request failed"):
        _client(transport=transport)._login()


def test_login_raises_on_missing_token() -> None:
    transport, _ = _capture_posts([_response(SESSION_URL, {"unexpected": "shape"})])

    with pytest.raises(RuntimeError, match="no token"):
        _client(transport=transport)._login()


def test_get_watched_movies_sends_jsonrpc_batch_with_bearer() -> None:
    transport, calls = _capture_posts([_login_response("abc"), _rpc_response([{"title": "Dune"}])])

    assert _client(transport=transport).get_watched_movies(page=2) == [{"title": "Dune"}]
    assert calls[1]["url"] == RPC_URL
    assert calls[1]["headers"] == {"authorization2": "Bearer abc"}
    assert calls[1]["json"] == [
        {
            "jsonrpc": "2.0",
            "method": "profile.WatchedMovies",
            "params": {"page": 2, "pageSize": 20, "login": "", "search": {"sort": "watchedAt_desc"}},
            "id": 1,
        }
    ]


def test_get_shows_sends_profile_shows() -> None:
    transport, calls = _capture_posts([_login_response(), _rpc_response([{"title": "Severance"}])])

    assert _client(transport=transport).get_shows() == [{"title": "Severance"}]
    assert calls[1]["json"] == [{"jsonrpc": "2.0", "method": "profile.Shows", "params": {"login": ""}, "id": 1}]


def test_rpc_unwraps_dict_result_with_items_key() -> None:
    transport, _ = _capture_posts([_login_response(), _rpc_response({"count": 1, "items": [{"title": "Arrival"}]})])

    assert _client(transport=transport).get_watched_movies() == [{"title": "Arrival"}]


def test_rpc_raises_on_http_error() -> None:
    transport, _ = _capture_posts([_login_response(), _response(RPC_URL, {}, status_code=500)])

    with pytest.raises(RuntimeError, match=r"profile\.Shows failed with HTTP 500"):
        _client(transport=transport).get_shows()


def test_rpc_raises_on_jsonrpc_error() -> None:
    error_body = [{"jsonrpc": "2.0", "id": 1, "error": {"code": -32600, "message": "not authorized"}}]
    transport, _ = _capture_posts([_login_response(), _response(RPC_URL, error_body)])

    with pytest.raises(RuntimeError, match="not authorized"):
        _client(transport=transport).get_shows()


def test_iter_all_watched_movies_paginates_until_empty_page() -> None:
    transport, calls = _capture_posts(
        [
            _login_response(),
            _rpc_response([{"title": "A"}, {"title": "B"}]),
            _rpc_response([{"title": "C"}]),
            _rpc_response([]),
        ],
    )

    movies = _client(transport=transport).iter_all_watched_movies()

    assert [movie["title"] for movie in movies] == ["A", "B", "C"]
    requested_pages = [call["json"][0]["params"]["page"] for call in calls[1:]]
    assert requested_pages == [0, 1, 2]
