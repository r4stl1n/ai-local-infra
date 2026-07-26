from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.routes import chat as chat_route


class _FakeRequest:
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body
        self.state = SimpleNamespace(http_client=object())

    async def json(self) -> dict[str, Any]:
        return self._body


class _FakeProxyResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeSettings:
    def __init__(self, llm_provider: str, llm_thinking: bool, llm_num_ctx: int) -> None:
        self.llm_provider = llm_provider
        self.llm_thinking = llm_thinking
        self.llm_num_ctx = llm_num_ctx
        self.llm_model = ""
        self.ollama_url = "http://upstream"
        self.llm_upstream_url = "http://upstream"
        self.llm_upstream_headers: dict[str, str] | None = None


class _FakeHttpClient:
    """Captures the body posted to Ollama's /api/chat and returns a native
    Ollama-style response."""

    def __init__(self) -> None:
        self.captured_url = ""
        self.captured_body: dict[str, Any] = {}

    async def post(self, url: str, json: dict[str, Any]) -> _FakeProxyResponse:
        self.captured_url = url
        self.captured_body = dict(json)
        return _FakeProxyResponse(
            {
                "model": json.get("model", ""),
                "message": {"role": "assistant", "content": "ok"},
                "done": True,
            }
        )


class ChatRouteLocalMergeTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_prefers_request_think_and_num_ctx(self) -> None:
        fake_client = _FakeHttpClient()
        original_settings = chat_route.settings
        try:
            chat_route.settings = _FakeSettings(llm_provider="local", llm_thinking=False, llm_num_ctx=65536)  # type: ignore[assignment]

            request = _FakeRequest(
                {
                    "model": "qwen",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                    "think": True,
                    "options": {"num_ctx": 2048},
                }
            )
            request.state.http_client = fake_client
            response = await chat_route.chat_completions(request)  # type: ignore[arg-type]
        finally:
            chat_route.settings = original_settings  # type: ignore[assignment]

        self.assertEqual("http://upstream/api/chat", fake_client.captured_url)
        self.assertEqual(True, fake_client.captured_body["think"])
        self.assertEqual(2048, fake_client.captured_body["options"]["num_ctx"])
        self.assertEqual(200, response.status_code)
        payload = json.loads(response.body.decode("utf-8"))
        self.assertEqual("ok", payload["choices"][0]["message"]["content"])

    async def test_local_falls_back_to_backend_defaults(self) -> None:
        fake_client = _FakeHttpClient()
        original_settings = chat_route.settings
        try:
            chat_route.settings = _FakeSettings(llm_provider="local", llm_thinking=True, llm_num_ctx=4096)  # type: ignore[assignment]

            request = _FakeRequest(
                {
                    "model": "qwen",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                }
            )
            request.state.http_client = fake_client
            _ = await chat_route.chat_completions(request)  # type: ignore[arg-type]
        finally:
            chat_route.settings = original_settings  # type: ignore[assignment]

        self.assertEqual(True, fake_client.captured_body["think"])
        self.assertEqual(4096, fake_client.captured_body["options"]["num_ctx"])

    async def test_remote_body_is_spec_clean_by_default(self) -> None:
        captured_body: dict[str, Any] = {}

        async def _fake_proxy_request(*args, body: dict[str, Any], **kwargs):
            _ = args, kwargs
            captured_body.update(body)
            return _FakeProxyResponse({"ok": True})

        original_settings = chat_route.settings
        original_proxy_request = chat_route.proxy_request
        try:
            chat_route.settings = _FakeSettings(llm_provider="remote", llm_thinking=False, llm_num_ctx=12288)  # type: ignore[assignment]
            chat_route.proxy_request = _fake_proxy_request  # type: ignore[assignment]

            request = _FakeRequest(
                {
                    "model": "remote-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                    "temperature": 0.2,
                }
            )
            _ = await chat_route.chat_completions(request)  # type: ignore[arg-type]
        finally:
            chat_route.settings = original_settings  # type: ignore[assignment]
            chat_route.proxy_request = original_proxy_request  # type: ignore[assignment]

        # Gateway-internal fields must not leak to an OpenAI-compatible upstream.
        self.assertNotIn("think", captured_body)
        self.assertNotIn("options", captured_body)
        self.assertNotIn("extra_body", captured_body)
        self.assertNotIn("chat_template_kwargs", captured_body)
        self.assertEqual(0.2, captured_body["temperature"])

    async def test_remote_explicit_think_enables_vllm_extension(self) -> None:
        captured_body: dict[str, Any] = {}

        async def _fake_proxy_request(*args, body: dict[str, Any], **kwargs):
            _ = args, kwargs
            captured_body.update(body)
            return _FakeProxyResponse({"ok": True})

        original_settings = chat_route.settings
        original_proxy_request = chat_route.proxy_request
        try:
            chat_route.settings = _FakeSettings(llm_provider="remote", llm_thinking=True, llm_num_ctx=12288)  # type: ignore[assignment]
            chat_route.proxy_request = _fake_proxy_request  # type: ignore[assignment]

            request = _FakeRequest(
                {
                    "model": "remote-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                    "think": False,
                }
            )
            _ = await chat_route.chat_completions(request)  # type: ignore[arg-type]
        finally:
            chat_route.settings = original_settings  # type: ignore[assignment]
            chat_route.proxy_request = original_proxy_request  # type: ignore[assignment]

        self.assertNotIn("think", captured_body)
        self.assertEqual(False, captured_body["chat_template_kwargs"]["enable_thinking"])

    async def test_missing_model_returns_400(self) -> None:
        request = _FakeRequest({"messages": [{"role": "user", "content": "hello"}]})
        response = await chat_route.chat_completions(request)  # type: ignore[arg-type]
        self.assertEqual(400, response.status_code)
        payload = json.loads(response.body.decode("utf-8"))
        self.assertEqual("invalid_request_error", payload["error"]["type"])


class BuildOllamaBodyTests(unittest.TestCase):
    def _settings(self) -> Any:
        return _FakeSettings(llm_provider="local", llm_thinking=False, llm_num_ctx=65536)

    def test_openai_sampling_params_map_to_options(self) -> None:
        from app.ollama import build_ollama_body

        body = build_ollama_body(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": 42,
                "stop": "END",
                "seed": 7,
            },
            self._settings(),  # type: ignore[arg-type]
        )
        options = body["options"]
        self.assertEqual(0.1, options["temperature"])
        self.assertEqual(0.9, options["top_p"])
        self.assertEqual(42, options["num_predict"])
        self.assertEqual(["END"], options["stop"])
        self.assertEqual(7, options["seed"])

    def test_native_options_win_over_mapped_params(self) -> None:
        from app.ollama import build_ollama_body

        body = build_ollama_body(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.1,
                "options": {"temperature": 0.7},
            },
            self._settings(),  # type: ignore[arg-type]
        )
        self.assertEqual(0.7, body["options"]["temperature"])

    def test_content_parts_flattened_with_images(self) -> None:
        from app.ollama import build_ollama_body

        body = build_ollama_body(
            {
                "model": "m",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what is this?"},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
                        ],
                    }
                ],
            },
            self._settings(),  # type: ignore[arg-type]
        )
        msg = body["messages"][0]
        self.assertEqual("what is this?", msg["content"])
        self.assertEqual(["QUJD"], msg["images"])

    def test_finish_reason_length_preserved(self) -> None:
        from app.ollama import _map_finish_reason

        self.assertEqual("length", _map_finish_reason("length"))
        self.assertEqual("tool_calls", _map_finish_reason("tool_calls"))
        self.assertEqual("stop", _map_finish_reason("stop"))


if __name__ == "__main__":
    unittest.main()
