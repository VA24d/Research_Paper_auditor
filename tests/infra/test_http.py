import json
import random
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from auditor.config import Settings
from auditor.infra.http import (
    HostNotAllowedError,
    HttpClient,
    HttpRequestError,
    HttpResponse,
    HttpStatusError,
    RateLimiter,
    ResponseCache,
)
from auditor.infra.scope import Scope, enter_scope
from auditor.infra.usage import UsageLog

Handler = Callable[[httpx.Request], httpx.Response]


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_client(
    handler: Handler,
    tmp_path: Path,
    fake_time: FakeTime,
    **settings: object,
) -> HttpClient:
    config = Settings.model_validate(
        {"cache_dir": tmp_path / "cache", "http_max_attempts": 3, **settings}
    )
    return HttpClient(
        config,
        transport=httpx.MockTransport(handler),
        sleep=fake_time.sleep,
        clock=fake_time.clock,
        rng=random.Random(0),
    )


@pytest.fixture
def fake_time() -> FakeTime:
    return FakeTime()


async def test_disallowed_host_is_rejected_before_sending(
    tmp_path: Path, fake_time: FakeTime
) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200)

    async with make_client(handler, tmp_path, fake_time) as client:
        with pytest.raises(HostNotAllowedError):
            await client.get("https://example.com/data")
    assert calls == []


async def test_redirect_to_disallowed_host_is_rejected(tmp_path: Path, fake_time: FakeTime) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.crossref.org":
            return httpx.Response(302, headers={"location": "https://evil.example/x"})
        return httpx.Response(200)

    async with make_client(handler, tmp_path, fake_time) as client:
        with pytest.raises(HostNotAllowedError):
            await client.get("https://api.crossref.org/works")


async def test_retries_server_errors_then_succeeds(tmp_path: Path, fake_time: FakeTime) -> None:
    statuses = iter([503, 502, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(next(statuses), json={"ok": True})

    usage = UsageLog()
    async with make_client(handler, tmp_path, fake_time) as client:
        with enter_scope(Scope(run_id="r1", task_id="t1", usage=usage)):
            response = await client.get("https://api.openalex.org/works", use_cache=False)
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    record = usage.http_requests[-1]
    assert record.attempts == 3
    assert record.task_id == "t1"


async def test_retry_after_header_is_respected(tmp_path: Path, fake_time: FakeTime) -> None:
    responses = iter(
        [httpx.Response(429, headers={"retry-after": "7"}), httpx.Response(200, text="done")]
    )

    async with make_client(lambda _: next(responses), tmp_path, fake_time) as client:
        response = await client.get("https://api.github.com/repos/x/y", use_cache=False)
    assert response.text == "done"
    assert 7.0 in fake_time.sleeps


async def test_long_retry_after_is_capped(tmp_path: Path, fake_time: FakeTime) -> None:
    responses = iter([httpx.Response(429, headers={"retry-after": "3600"}), httpx.Response(200)])
    async with make_client(lambda _: next(responses), tmp_path, fake_time) as client:
        await client.get("https://api.github.com/repos/x/y", use_cache=False)
    assert max(fake_time.sleeps) == 60.0


async def test_gives_up_and_returns_last_error_response(
    tmp_path: Path, fake_time: FakeTime
) -> None:
    async with make_client(lambda _: httpx.Response(503), tmp_path, fake_time) as client:
        response = await client.get("https://api.github.com/x", use_cache=False)
    assert response.status_code == 503
    with pytest.raises(HttpStatusError):
        response.raise_for_status()


async def test_client_errors_are_not_retried(tmp_path: Path, fake_time: FakeTime) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404)

    async with make_client(handler, tmp_path, fake_time) as client:
        response = await client.get("https://api.github.com/missing", use_cache=False)
    assert response.status_code == 404
    assert calls == 1


async def test_connection_errors_are_retried_then_raised(
    tmp_path: Path, fake_time: FakeTime
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("refused", request=request)

    usage = UsageLog()
    async with make_client(handler, tmp_path, fake_time) as client:
        with enter_scope(Scope(run_id="r1", usage=usage)), pytest.raises(HttpRequestError):
            await client.get("https://api.github.com/x")
    assert calls == 3
    assert usage.http_requests[-1].error is not None


async def test_successful_get_is_cached(tmp_path: Path, fake_time: FakeTime) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"n": calls}, headers={"content-type": "application/json"})

    usage = UsageLog()
    async with make_client(handler, tmp_path, fake_time) as client:
        with enter_scope(Scope(run_id="r1", usage=usage)):
            first = await client.get("https://api.crossref.org/works", params={"rows": 1})
            second = await client.get("https://api.crossref.org/works", params={"rows": 1})
            other = await client.get("https://api.crossref.org/works", params={"rows": 2})
    assert calls == 2
    assert first.json() == second.json() == {"n": 1}
    assert other.json() == {"n": 2}
    assert not first.from_cache
    assert second.from_cache
    assert second.headers["content-type"] == "application/json"
    assert [r.cache_hit for r in usage.http_requests] == [False, True, False]


async def test_error_responses_are_not_cached(tmp_path: Path, fake_time: FakeTime) -> None:
    statuses = iter([404, 200])

    async with make_client(lambda _: httpx.Response(next(statuses)), tmp_path, fake_time) as client:
        assert (await client.get("https://api.crossref.org/works/x")).status_code == 404
        assert (await client.get("https://api.crossref.org/works/x")).status_code == 200


async def test_cache_can_be_disabled(tmp_path: Path, fake_time: FakeTime) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200)

    async with make_client(handler, tmp_path, fake_time, http_cache_ttl_days=0) as client:
        await client.get("https://api.crossref.org/works")
        await client.get("https://api.crossref.org/works")
    assert calls == 2


def test_cache_entries_expire(tmp_path: Path) -> None:
    now = datetime(2026, 10, 10, tzinfo=UTC)
    cache = ResponseCache(directory=tmp_path, ttl=timedelta(days=1), now=lambda: now)
    response = HttpResponse(url="https://x", status_code=200, headers={}, content=b"body")
    cache.put("k" * 64, response)
    assert cache.get("k" * 64) is not None
    later = ResponseCache(
        directory=tmp_path, ttl=timedelta(days=1), now=lambda: now + timedelta(days=2)
    )
    assert later.get("k" * 64) is None


def test_corrupt_cache_entry_is_a_miss(tmp_path: Path) -> None:
    cache = ResponseCache(directory=tmp_path, ttl=timedelta(days=1))
    response = HttpResponse(url="https://x", status_code=200, headers={}, content=b"body")
    key = "a" * 64
    cache.put(key, response)
    (tmp_path / "aa" / f"{key}.json").write_text("{not json", encoding="utf-8")
    assert cache.get(key) is None


async def test_post_sends_multipart_and_is_not_cached(tmp_path: Path, fake_time: FakeTime) -> None:
    bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        return httpx.Response(200, text="<TEI/>")

    async with make_client(handler, tmp_path, fake_time) as client:
        for _ in range(2):
            await client.post(
                "http://localhost:8070/api/processFulltextDocument",
                data={"teiCoordinates": ["s", "ref"]},
                files={"input": ("paper.pdf", b"%PDF-1.5", "application/pdf")},
            )
    assert len(bodies) == 2
    assert bodies[0].count(b'name="teiCoordinates"') == 2
    assert b"%PDF-1.5" in bodies[0]


async def test_user_agent_includes_contact_email(tmp_path: Path, fake_time: FakeTime) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["user-agent"])
        return httpx.Response(200)

    async with make_client(
        handler, tmp_path, fake_time, contact_email="team@example.org"
    ) as client:
        await client.get("https://api.crossref.org/works", use_cache=False)
    assert seen[0].startswith("research-paper-auditor/")
    assert "mailto:team@example.org" in seen[0]


async def test_arxiv_hosts_share_one_rate_limit(tmp_path: Path, fake_time: FakeTime) -> None:
    async with make_client(lambda _: httpx.Response(200), tmp_path, fake_time) as client:
        await client.get("https://export.arxiv.org/api/query", use_cache=False)
        await client.get("https://arxiv.org/abs/1706.03762", use_cache=False)
        await client.get("https://api.github.com/x", use_cache=False)
    assert fake_time.sleeps == [3.0]


async def test_rate_limiter_spaces_requests() -> None:
    fake = FakeTime()
    limiter = RateLimiter({"slow": 2.0}, clock=fake.clock, sleep=fake.sleep)
    for _ in range(3):
        await limiter.acquire("slow")
    await limiter.acquire("fast")
    assert fake.sleeps == [2.0, 2.0]


def test_response_text_uses_declared_charset() -> None:
    response = HttpResponse(
        url="https://x",
        status_code=200,
        headers={"content-type": "text/plain; charset=latin-1"},
        content="café".encode("latin-1"),
    )
    assert response.text == "café"


def test_response_json() -> None:
    response = HttpResponse(
        url="https://x", status_code=200, headers={}, content=json.dumps([1, 2]).encode()
    )
    assert response.json() == [1, 2]
