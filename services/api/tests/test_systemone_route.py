from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.routes import systemone as systemone_route

_ANSWER = {
    "model": "tev1:0.8b",
    "answers": {"refund": {"type": "noul", "noul": 0.97}},
    "usage": {"input_tokens": 120, "output_tokens": 1},
}


class _FakeClient:
    def __init__(self, status: int, payload: Any, content_type: str = "application/json") -> None:
        self.status = status
        self.payload = payload
        self.content_type = content_type
        self.calls: list[dict[str, Any]] = []

    async def request(self, method: str, url: str, json: Any = None, **_: Any) -> SimpleNamespace:
        self.calls.append({"method": method, "url": url, "json": json})
        raw = json_dumps(self.payload) if not isinstance(self.payload, str) else self.payload
        return SimpleNamespace(
            status_code=self.status,
            headers={"content-type": self.content_type},
            content=raw.encode(),
            text=raw,
            json=lambda: __import__("json").loads(raw),
        )


def json_dumps(value: Any) -> str:
    return json.dumps(value)


class _FakeRequest:
    def __init__(self, client: _FakeClient, raw: bytes) -> None:
        self._raw = raw
        self.state = SimpleNamespace(http_client=client)

    async def json(self) -> Any:
        return json.loads(self._raw)


class SystemOneRouteTests(unittest.TestCase):
    def _call(self, client: _FakeClient, raw: bytes) -> tuple[int, Any]:
        response = asyncio.run(systemone_route.systemone(_FakeRequest(client, raw)))
        return response.status_code, json.loads(response.body)

    def test_forwards_to_bundled_ollama_and_passes_answers_through(self) -> None:
        client = _FakeClient(200, _ANSWER)
        body = {"model": "tev1:0.8b", "state": {"t": "refund please"},
                "questions": {"refund": {"type": "noul", "instructions": "Refund?"}}}
        status, payload = self._call(client, json.dumps(body).encode())
        self.assertEqual(status, 200)
        self.assertEqual(payload, _ANSWER)
        self.assertTrue(client.calls[0]["url"].endswith("/v1/systemone"))
        self.assertEqual(client.calls[0]["json"], body)

    def test_upstream_error_keeps_status_in_openai_shape(self) -> None:
        client = _FakeClient(404, {"error": {"message": 'model "nope" not found', "type": "not_found_error"}})
        status, payload = self._call(client, b'{"model": "nope"}')
        self.assertEqual(status, 404)
        self.assertIn("not found", payload["error"]["message"])

    def test_old_ollama_without_endpoint_explains_version(self) -> None:
        # Ollama < 0.35 answers unknown routes with a plain-text "404 page not found".
        status, payload = self._call(_FakeClient(404, "404 page not found", "text/plain"), b'{"model": "nimble"}')
        self.assertEqual(status, 404)
        self.assertIn("0.35", payload["error"]["message"])

    def test_invalid_bodies_are_400(self) -> None:
        for raw in (b"not json", b"[1, 2]"):
            status, payload = self._call(_FakeClient(200, _ANSWER), raw)
            self.assertEqual(status, 400, raw)
            self.assertEqual(payload["error"]["type"], "invalid_request_error")


if __name__ == "__main__":
    unittest.main()
