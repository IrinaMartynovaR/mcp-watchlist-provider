from typing import Any

import httpx
import pytest

from myshows_client.client import RPC_URL, SESSION_URL, MyShowsClient
from myshows_client.settings import MyShowsSettings


def _client(login: str = "user", password: str = "secret") -> MyShowsClient:
    return MyShowsClient(
        MyShowsSettings(MYSHOWS_LOGIN=login, MYSHOWS_PASSWORD=password, MYSHOWS_TIMEOUT_SECONDS=1)
    )


def _response(url: str, payload: Any, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code=status_code, json=payload, request=httpx.Request("POST", url))


def _capture_posts(
    monkeypatch: pytest.MonkeyPatch, responses: list[httpx.Response | Exception]
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_post(self: httpx.Client, url: str, **kwargs: Any) -> httpx.Response:
        calls.append({"url": url, "json": kwargs.get("json"), "headers": kwargs.get("headers")})
        outcome = responses[len(calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    return calls


def _login_response(token: str = "token-1") -> httpx.Response:
    return _response(SESSION_URL, {"token": token})


def _rpc_response(result: Any) -> httpx.Response:
    return _response(RPC_URL, [{"jsonrpc": "2.0", "id": 1, "result": result}])


def test_login_posts_credentials_and_returns_token(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_posts(monkeypatch, [_login_response("abc")])

    assert _client()._login() == "abc"
    assert calls == [{"url": SESSION_URL, "json": {"login": "user", "password": "secret"}, "headers": None}]


def test_login_caches_token(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_posts(monkeypatch, [_login_response("abc")])
    client = _client()

    assert client._login() == "abc"
    assert client._login() == "abc"
    assert len(calls) == 1


def test_login_fast_fails_without_credentials(poison_network: list[str]) -> None:
    with pytest.raises(RuntimeError, match="MYSHOWS_LOGIN"):
        _client(login="", password="")._login()
    assert poison_network == []


def test_login_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _capture_posts(monkeypatch, [_response(SESSION_URL, {"error": "bad credentials"}, status_code=401)])

    with pytest.raises(RuntimeError, match="HTTP 401"):
        _client()._login()


def test_login_raises_on_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _capture_posts(monkeypatch, [httpx.ConnectError("boom")])

    with pytest.raises(RuntimeError, match="login request failed"):
        _client()._login()


def test_login_raises_on_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _capture_posts(monkeypatch, [_response(SESSION_URL, {"unexpected": "shape"})])

    with pytest.raises(RuntimeError, match="no token"):
        _client()._login()


def test_get_watched_movies_sends_jsonrpc_batch_with_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_posts(monkeypatch, [_login_response("abc"), _rpc_response([{"title": "Dune"}])])

    assert _client().get_watched_movies(page=2) == [{"title": "Dune"}]
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


def test_get_shows_sends_profile_shows(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_posts(monkeypatch, [_login_response(), _rpc_response([{"title": "Severance"}])])

    assert _client().get_shows() == [{"title": "Severance"}]
    assert calls[1]["json"] == [{"jsonrpc": "2.0", "method": "profile.Shows", "params": {"login": ""}, "id": 1}]


def test_rpc_unwraps_dict_result_with_items_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _capture_posts(monkeypatch, [_login_response(), _rpc_response({"count": 1, "items": [{"title": "Arrival"}]})])

    assert _client().get_watched_movies() == [{"title": "Arrival"}]


def test_rpc_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _capture_posts(monkeypatch, [_login_response(), _response(RPC_URL, {}, status_code=500)])

    with pytest.raises(RuntimeError, match=r"profile\.Shows failed with HTTP 500"):
        _client().get_shows()


def test_rpc_raises_on_jsonrpc_error(monkeypatch: pytest.MonkeyPatch) -> None:
    error_body = [{"jsonrpc": "2.0", "id": 1, "error": {"code": -32600, "message": "not authorized"}}]
    _capture_posts(monkeypatch, [_login_response(), _response(RPC_URL, error_body)])

    with pytest.raises(RuntimeError, match="not authorized"):
        _client().get_shows()


def test_iter_all_watched_movies_paginates_until_empty_page(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_posts(
        monkeypatch,
        [
            _login_response(),
            _rpc_response([{"title": "A"}, {"title": "B"}]),
            _rpc_response([{"title": "C"}]),
            _rpc_response([]),
        ],
    )

    movies = _client().iter_all_watched_movies()

    assert [movie["title"] for movie in movies] == ["A", "B", "C"]
    requested_pages = [call["json"][0]["params"]["page"] for call in calls[1:]]
    assert requested_pages == [0, 1, 2]
