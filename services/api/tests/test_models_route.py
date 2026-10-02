from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.routes import models as models_route


class _FakeOllama:
    """Mimics Ollama's /api/ps + /api/generate keep_alive=0 unload, where the
    runner disappears from /api/ps one poll after the unload call."""

    def __init__(self, resident: list[str]) -> None:
        self.resident = list(resident)
        self.pending: list[str] = []
        self.unload_calls: list[str] = []

    async def request(self, method: str, url: str, json: dict[str, Any] | None = None, **_: Any):
        if url.endswith("/api/ps"):
            payload = {"models": [{"name": n} for n in self.resident]}
            self.resident = [n for n in self.resident if n not in self.pending]
            return _response(200, payload)
        if url.endswith("/api/generate"):
            name = json["model"]
            self.unload_calls.append(name)
            tagged = models_route._with_tag(name)
            if tagged not in self.resident:
                return _response(404, {"error": f"model '{name}' not found"})
            self.pending.append(tagged)
            return _response(200, {"model": name, "done_reason": "unload"})
        raise AssertionError(url)


def _response(status: int, payload: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        status_code=status,
        headers={"content-type": "application/json"},
        json=lambda: payload,
        text=json.dumps(payload),
    )


class _FakeRequest:
    def __init__(self, client: _FakeOllama, raw: bytes = b"") -> None:
        self._raw = raw
        self.state = SimpleNamespace(http_client=client)

    async def body(self) -> bytes:
        return self._raw


class UnloadModelsTests(unittest.TestCase):
    def _unload(self, client: _FakeOllama, raw: bytes = b"") -> tuple[int, dict[str, Any]]:
        response = asyncio.run(models_route.unload_models(_FakeRequest(client, raw)))
        return response.status_code, json.loads(response.body)

    def test_unload_all_waits_until_runners_are_gone(self) -> None:
        client = _FakeOllama(["gemma4:12b", "snowflake-arctic-embed:137m"])
        status, body = self._unload(client)
        self.assertEqual(status, 200)
        self.assertEqual(body["unloaded"], ["gemma4:12b", "snowflake-arctic-embed:137m"])
        self.assertEqual(client.resident, [])

    def test_unload_untagged_name_matches_latest(self) -> None:
        client = _FakeOllama(["all-minilm:latest", "gemma4:12b"])
        status, body = self._unload(client, b'{"model": "all-minilm"}')
        self.assertEqual((status, body["unloaded"]), (200, ["all-minilm"]))
        self.assertEqual(client.resident, ["gemma4:12b"])

    def test_unknown_model_is_404(self) -> None:
        status, body = self._unload(_FakeOllama([]), b'{"model": "nope:1b"}')
        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["type"], "not_found_error")

    def test_invalid_bodies_are_400(self) -> None:
        for raw in (b"xx", b'{"model": ""}', b'{"model": 3}'):
            status, _ = self._unload(_FakeOllama([]), raw)
            self.assertEqual(status, 400, raw)

    def test_with_tag(self) -> None:
        self.assertEqual(models_route._with_tag("all-minilm"), "all-minilm:latest")
        self.assertEqual(models_route._with_tag("gemma4:12b"), "gemma4:12b")
        self.assertEqual(models_route._with_tag("hf.co/org/model"), "hf.co/org/model:latest")


if __name__ == "__main__":
    unittest.main()
