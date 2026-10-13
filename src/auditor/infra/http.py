"""The single HTTP client every module uses (decision D8).

It enforces a host allowlist and per-host rate limits on every request it sends,
redirects included, retries with
backoff on 429/5xx and connection errors, an on-disk cache for successful GET requests,
and records each request against the current task.
"""

import asyncio
import hashlib
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import httpx

from auditor import __version__
from auditor.config import Settings
from auditor.infra.files import atomic_write
from auditor.infra.scope import current_scope
from auditor.schemas import HttpRequestRecord

logger = logging.getLogger(__name__)

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

# Hosts that share one rate limit, and the minimum gap between requests to each.
RATE_LIMIT_BUCKETS = {"arxiv.org": "arxiv", "export.arxiv.org": "arxiv"}
DEFAULT_MIN_INTERVALS = {
    "arxiv": 3.0,
    "api.crossref.org": 0.2,
    "api.openalex.org": 0.1,
    "api.semanticscholar.org": 1.0,
}


class HttpError(Exception):
    pass


class HostNotAllowedError(HttpError):
    pass


class HttpRequestError(HttpError):
    """The request could not be completed, even after retries."""


class HttpStatusError(HttpError):
    def __init__(self, response: "HttpResponse") -> None:
        super().__init__(f"HTTP {response.status_code} for {response.url}")
        self.response = response


@dataclass(frozen=True)
class HttpResponse:
    url: str
    status_code: int
    headers: Mapping[str, str]
    content: bytes
    from_cache: bool = False

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    @property
    def text(self) -> str:
        content_type = self.headers.get("content-type", "")
        charset = "utf-8"
        for part in content_type.split(";")[1:]:
            name, _, value = part.strip().partition("=")
            if name.lower() == "charset" and value:
                charset = value.strip('"')
        try:
            return self.content.decode(charset, errors="replace")
        except LookupError:
            return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.content)

    def raise_for_status(self) -> None:
        if not self.ok:
            raise HttpStatusError(self)


# Requests sent for the logical request in progress, counted by the request hook so that
# redirect hops of every attempt are included. Each asyncio task has its own value.
_sent: ContextVar[list[int] | None] = ContextVar("auditor_http_sent", default=None)


class RateLimiter:
    """Keeps a minimum interval between requests that share a bucket."""

    def __init__(
        self,
        min_intervals: Mapping[str, float],
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._intervals = dict(min_intervals)
        self._clock = clock
        self._sleep = sleep
        self._locks: dict[str, asyncio.Lock] = {}
        self._next_allowed: dict[str, float] = {}

    async def acquire(self, bucket: str) -> None:
        interval = self._intervals.get(bucket)
        if not interval:
            return
        lock = self._locks.setdefault(bucket, asyncio.Lock())
        async with lock:
            now = self._clock()
            next_allowed = self._next_allowed.get(bucket, now)
            if next_allowed > now:
                await self._sleep(next_allowed - now)
                # Count from when the request really goes out: a sleep can end late.
                now = max(self._clock(), next_allowed)
            self._next_allowed[bucket] = now + interval


@dataclass
class ResponseCache:
    """Successful GET responses on disk, one body and one metadata file per request."""

    directory: Path
    ttl: timedelta
    now: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))

    @staticmethod
    def key(url: str, headers: Mapping[str, str]) -> str:
        relevant = {k.lower(): v for k, v in headers.items() if k.lower() == "accept"}
        payload = json.dumps({"url": url, "headers": relevant}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _paths(self, key: str) -> tuple[Path, Path]:
        folder = self.directory / key[:2]
        return folder / f"{key}.json", folder / f"{key}.bin"

    def get(self, key: str) -> HttpResponse | None:
        meta_path, body_path = self._paths(key)
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            stored_at = datetime.fromisoformat(meta["stored_at"])
            if self.now() - stored_at > self.ttl:
                return None
            content = body_path.read_bytes()
        except (OSError, ValueError, KeyError):
            return None
        return HttpResponse(
            url=meta["url"],
            status_code=meta["status_code"],
            headers=meta["headers"],
            content=content,
            from_cache=True,
        )

    def put(self, key: str, response: HttpResponse) -> None:
        meta_path, body_path = self._paths(key)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "url": response.url,
            "status_code": response.status_code,
            "headers": {k: v for k, v in response.headers.items() if k == "content-type"},
            "stored_at": self.now().isoformat(),
        }
        # Body first, metadata last: a reader that finds the metadata finds a complete body.
        atomic_write(body_path, response.content)
        atomic_write(meta_path, json.dumps(meta).encode("utf-8"))


def _retry_after_seconds(response: httpx.Response, now: datetime) -> float | None:
    value = response.headers.get("retry-after")
    if not value:
        return None
    if value.strip().isdigit():
        return float(value.strip())
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - now).total_seconds())


class HttpClient:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        cache: ResponseCache | None = None,
        use_default_cache: bool = True,
        min_intervals: Mapping[str, float] = DEFAULT_MIN_INTERVALS,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
        backoff_base_s: float = 1.0,
        max_wait_s: float = 60.0,
    ) -> None:
        # The configured GROBID service is trusted by configuration, wherever it runs.
        grobid_host = httpx.URL(settings.grobid_url).host
        self._allowed_hosts = frozenset(
            host.lower() for host in (*settings.allowed_hosts, grobid_host) if host
        )
        self._max_attempts = settings.http_max_attempts
        if cache is None and use_default_cache and settings.http_cache_ttl_days > 0:
            cache = ResponseCache(
                directory=settings.cache_dir / "http",
                ttl=timedelta(days=settings.http_cache_ttl_days),
            )
        self._cache = cache
        self._limiter = RateLimiter(min_intervals, clock=clock, sleep=sleep)
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._backoff_base_s = backoff_base_s
        self._max_wait_s = max_wait_s
        user_agent = f"research-paper-auditor/{__version__}"
        if settings.contact_email:
            user_agent += f" (mailto:{settings.contact_email})"
        self._client = httpx.AsyncClient(
            transport=transport,
            timeout=settings.http_timeout_s,
            follow_redirects=True,
            headers={"User-Agent": user_agent},
            event_hooks={"request": [self._before_request]},
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    def is_allowed(self, host: str) -> bool:
        return host.lower() in self._allowed_hosts

    async def _before_request(self, request: httpx.Request) -> None:
        """Runs before every request httpx sends, including each redirect hop."""
        if not self.is_allowed(request.url.host):
            raise HostNotAllowedError(f"host {request.url.host!r} is not on the allowlist")
        await self._limiter.acquire(RATE_LIMIT_BUCKETS.get(request.url.host, request.url.host))
        counter = _sent.get()
        if counter is not None:
            counter[0] += 1

    async def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        use_cache: bool = True,
        attempts: int | None = None,
    ) -> HttpResponse:
        return await self._send(
            "GET", url, params=params, headers=headers, use_cache=use_cache, attempts=attempts
        )

    async def post(
        self,
        url: str,
        *,
        data: Mapping[str, Any] | None = None,
        files: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        request_timeout_s: float | None = None,
        attempts: int | None = None,
    ) -> HttpResponse:
        return await self._send(
            "POST",
            url,
            data=data,
            files=files,
            headers=headers,
            request_timeout_s=request_timeout_s,
            use_cache=False,
            attempts=attempts,
        )

    async def _send(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        data: Mapping[str, Any] | None = None,
        files: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        request_timeout_s: float | None = None,
        use_cache: bool,
        attempts: int | None = None,
    ) -> HttpResponse:
        max_attempts = attempts if attempts is not None else self._max_attempts
        request_url = httpx.URL(url)
        if params:
            request_url = request_url.copy_merge_params(params)
        if not self.is_allowed(request_url.host):
            raise HostNotAllowedError(f"host {request_url.host!r} is not on the allowlist")

        started_at = datetime.now(UTC)
        started = time.perf_counter()
        cache_key = None
        if method == "GET" and use_cache and self._cache is not None:
            cache_key = ResponseCache.key(str(request_url), headers or {})
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._record(method, request_url, cached.status_code, 1, True, started_at, started)
                return cached

        sent = [0]
        token = _sent.set(sent)
        try:
            attempt = 0
            while True:
                attempt += 1
                try:
                    raw = await self._client.request(
                        method,
                        request_url,
                        data=data,
                        files=files,
                        headers=headers,
                        timeout=(
                            request_timeout_s
                            if request_timeout_s is not None
                            else httpx.USE_CLIENT_DEFAULT
                        ),
                    )
                except httpx.TransportError as exc:
                    if attempt < max_attempts:
                        await self._sleep(self._backoff(attempt))
                        continue
                    error = f"{type(exc).__name__}: {exc}"
                    self._record(
                        method,
                        request_url,
                        None,
                        attempt,
                        False,
                        started_at,
                        started,
                        error,
                        redirects=sent[0] - attempt,
                    )
                    raise HttpRequestError(f"{method} {request_url} failed: {error}") from exc

                if raw.status_code in RETRY_STATUSES and attempt < max_attempts:
                    wait = _retry_after_seconds(raw, datetime.now(UTC))
                    await self._sleep(
                        min(wait if wait is not None else self._backoff(attempt), self._max_wait_s)
                    )
                    continue

                response = HttpResponse(
                    url=str(raw.url),
                    status_code=raw.status_code,
                    headers={k.lower(): v for k, v in raw.headers.items()},
                    content=raw.content,
                )
                if cache_key is not None and self._cache is not None and raw.status_code == 200:
                    try:
                        self._cache.put(cache_key, response)
                    except OSError:
                        # A cache write can lose a race with another process on Windows;
                        # the response itself is still good.
                        logger.warning("could not cache %s", request_url, exc_info=True)
                self._record(
                    method,
                    request_url,
                    raw.status_code,
                    attempt,
                    False,
                    started_at,
                    started,
                    redirects=sent[0] - attempt,
                )
                return response
        finally:
            _sent.reset(token)

    def _backoff(self, attempt: int) -> float:
        ceiling = min(self._max_wait_s, self._backoff_base_s * 2 ** (attempt - 1))
        return self._rng.uniform(0, ceiling)

    def _record(
        self,
        method: str,
        url: httpx.URL,
        status_code: int | None,
        attempts: int,
        cache_hit: bool,
        started_at: datetime,
        started: float,
        error: str | None = None,
        *,
        redirects: int = 0,
    ) -> None:
        active = current_scope()
        if active is None:
            return
        active.record(
            HttpRequestRecord(
                run_id=active.run_id,
                paper_id=active.paper_id,
                variant=active.variant,
                task_id=active.task_id,
                method=method,
                host=url.host,
                path=url.path,
                status_code=status_code,
                attempts=attempts,
                redirects=redirects,
                cache_hit=cache_hit,
                started_at=started_at,
                duration_s=time.perf_counter() - started,
                error=error,
            )
        )
