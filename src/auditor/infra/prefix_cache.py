"""Explicit context caching of a long first message on Vertex AI Gemini.

Every review-model call about a paper starts with the same message holding the paper
(`planner.theory.paper_context`). Vertex's implicit cache reused only about 9% of that
input on 3.8 Flash, so the message is put in an explicit cache the first time it is sent,
and the calls after it name the cache instead of sending the paper again. Cached input is
billed at a fraction of the input price, plus storage for the cache's lifetime.

A cache is a short-lived resource on the Google Cloud project: each one expires after
`ttl_s` seconds, and the run deletes the ones it made when it ends. When a cache cannot be
made (the text is too short, the model or endpoint does not support it, or the account may
not create caches), the call sends the message as usual.
"""

import asyncio
import hashlib
import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class CacheClient(Protocol):
    async def create(self, model: str, system_instruction: str, ttl_s: int) -> str: ...

    async def delete(self, name: str) -> None: ...


class GenAICacheClient:
    """Caches through the google-genai SDK, on the same project and location as the model
    calls (Application Default Credentials)."""

    def __init__(self, options: dict[str, Any]) -> None:
        from google import genai

        self._client = genai.Client(
            vertexai=bool(options.get("vertexai")),
            project=options.get("project"),
            location=options.get("location"),
        )

    async def create(self, model: str, system_instruction: str, ttl_s: int) -> str:
        from google.genai import types

        cache = await self._client.aio.caches.create(
            model=model,
            config=types.CreateCachedContentConfig(
                system_instruction=system_instruction,
                ttl=f"{ttl_s}s",
                display_name="auditor-paper",
            ),
        )
        if not cache.name:
            raise RuntimeError("the cache has no name")
        return cache.name

    async def delete(self, name: str) -> None:
        await self._client.aio.caches.delete(name=name)


class PrefixCache:
    """The cache made for each (model, first message), made once on first use."""

    def __init__(self, client: CacheClient, *, ttl_s: int, min_chars: int) -> None:
        self.client = client
        self.ttl_s = ttl_s
        self.min_chars = min_chars
        self._names: dict[str, str | None] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self.created: list[str] = []

    @staticmethod
    def _key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}\0{text}".encode()).hexdigest()

    async def name_for(self, model: str, text: str) -> str | None:
        """The cache holding `text` as the system instruction for `model`, or None when
        there is none and the text is to be sent as usual."""
        if len(text) < self.min_chars:
            return None
        key = self._key(model, text)
        if key in self._names:
            return self._names[key]
        async with self._locks.setdefault(key, asyncio.Lock()):
            if key not in self._names:
                try:
                    name: str | None = await self.client.create(model, text, self.ttl_s)
                except Exception as exc:
                    logger.warning("no context cache for %s (%s); sending it whole", model, exc)
                    name = None
                self._names[key] = name
                if name is not None:
                    self.created.append(name)
        return self._names[key]

    def forget(self, model: str, text: str) -> None:
        """Stop using the cache for this text, such as after it expired."""
        self._names[self._key(model, text)] = None

    async def close(self) -> None:
        """Delete the caches this run made; they would expire on their own anyway."""
        for name in self.created:
            try:
                await self.client.delete(name)
            except Exception as exc:
                logger.warning("could not delete context cache %s (%s)", name, exc)
        self.created.clear()
