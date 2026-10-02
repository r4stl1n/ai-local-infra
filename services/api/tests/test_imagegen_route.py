from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.routes import imagegen as imagegen_route


class _FakeRequest:
    def __init__(self, raw: bytes, content_type: str | None) -> None:
        self._raw = raw
        self.headers = {"content-type": content_type} if content_type else {}
        self.state = SimpleNamespace(http_client=_FakeHttpClient())

    async def body(self) -> bytes:
        return self._raw


class _FakeHttpClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def post(self, url: str, content: bytes, headers: dict[str, str]) -> SimpleNamespace:
        self.calls.append({"url": url, "content": content, "headers": headers})
        return SimpleNamespace(
            status_code=200,
            content=b'{"data": []}',
            headers={"content-type": "application/json"},
        )


class ImagegenProxyTests(unittest.TestCase):
    def _post(self, route, raw: bytes, content_type: str | None) -> tuple[Any, dict[str, Any]]:
        request = _FakeRequest(raw, content_type)
        response = asyncio.run(route(request))
        return response, request.state.http_client.calls[0]

    def test_multipart_edit_is_forwarded_byte_for_byte(self) -> None:
        content_type = "multipart/form-data; boundary=xyz"
        raw = (
            b"--xyz\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\na cat\r\n"
            b"--xyz\r\nContent-Disposition: form-data; name=\"image\"; filename=\"r.png\"\r\n"
            b"Content-Type: image/png\r\n\r\n\x89PNG\x00\xff\r\n--xyz--\r\n"
        )
        response, call = self._post(imagegen_route.edit_image, raw, content_type)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(call["url"].endswith("/v1/images/edits"))
        self.assertEqual(call["content"], raw)
        self.assertEqual(call["headers"]["Content-Type"], content_type)

    def test_json_generation_keeps_content_type(self) -> None:
        raw = b'{"prompt": "a lighthouse", "model": "krea"}'
        _, call = self._post(imagegen_route.create_image, raw, "application/json")
        self.assertTrue(call["url"].endswith("/v1/images/generations"))
        self.assertEqual(call["content"], raw)
        self.assertEqual(call["headers"]["Content-Type"], "application/json")

    def test_bodyless_unload_is_forwarded(self) -> None:
        response, call = self._post(imagegen_route.unload_image_model, b"", None)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(call["content"], b"")

    def test_json_model_extraction(self) -> None:
        self.assertEqual(imagegen_route._json_model(b'{"model": "m"}', "application/json"), "m")
        self.assertIsNone(imagegen_route._json_model(b"not json", "application/json"))
        self.assertIsNone(imagegen_route._json_model(b'{"model": "m"}', "multipart/form-data"))


if __name__ == "__main__":
    unittest.main()
