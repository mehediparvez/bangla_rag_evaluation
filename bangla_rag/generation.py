"""Gemini generator with an on-disk cache, rate limiting and retries.

Every call is cached by (model, temperature, prompt), so an interrupted run
resumes without paying for or waiting on calls it already made.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path


class DailyQuotaExceeded(RuntimeError):
    """Raised when the per-day free-tier quota is used up; re-run later to resume."""


class GeminiGenerator:
    def __init__(
        self,
        model: str = "gemini-2.5-flash-lite",
        rpm: float = 10,
        cache_path: str | Path = "results/gemini_cache.jsonl",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 512,
        max_retries: int = 6,
    ):
        self.model, self.rpm = model, rpm
        self.temperature, self.max_output_tokens = temperature, max_output_tokens
        self.max_retries = max_retries
        self.api_key = api_key
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, str] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec["text"]
        self._client = None
        self._last_call = 0.0
        self.n_api_calls = 0

    def _key(self, prompt: str) -> str:
        raw = f"{self.model}|{self.temperature}|{self.max_output_tokens}|{prompt}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_client(self):
        if self._client is None:
            from google import genai

            key = self.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise RuntimeError("Set the GEMINI_API_KEY environment variable first.")
            self._client = genai.Client(api_key=key)
        return self._client

    def generate(self, prompt: str) -> dict:
        key = self._key(prompt)
        if key in self.cache:
            return {"text": self.cache[key], "cached": True}
        text = self._call(prompt)
        self.cache[key] = text
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "model": self.model, "text": text}, ensure_ascii=False) + "\n")
        return {"text": text, "cached": False}

    def _call(self, prompt: str) -> str:
        from google.genai import types

        cfg = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
        )
        client = self._get_client()
        for attempt in range(self.max_retries):
            wait = self._last_call + 60.0 / self.rpm - time.time()
            if wait > 0:
                time.sleep(wait)
            try:
                resp = client.models.generate_content(model=self.model, contents=prompt, config=cfg)
                self._last_call = time.time()
                self.n_api_calls += 1
                return resp.text or ""
            except Exception as e:  # SDK raises several error types; classify by message
                self._last_call = time.time()
                msg = str(e)
                if "PerDay" in msg or "per day" in msg.lower():
                    raise DailyQuotaExceeded(msg) from e
                transient = any(
                    s in msg for s in ("429", "RESOURCE_EXHAUSTED", "500", "503", "UNAVAILABLE", "DEADLINE")
                )
                if not transient or attempt == self.max_retries - 1:
                    raise
                time.sleep(min(15 * 2**attempt, 120))
        return ""
