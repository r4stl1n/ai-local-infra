"""End-to-end audio tests against a running stack.

Generates speech via the gateway's TTS endpoint, then feeds that audio to
the Whisper transcription endpoint and checks the text survives the
round-trip.  Uses only the standard library so it runs on the host:

    python3 -m unittest discover -s tests/e2e -v

Configuration via environment (falls back to the repo .env for API_KEY):
    E2E_API_URL   gateway base URL   (default http://localhost:8000)
    API_KEY       gateway bearer token
"""

from __future__ import annotations

import base64
import json
import os
import re
import unittest
import urllib.error
import urllib.request
import uuid
from pathlib import Path

API_URL = os.getenv("E2E_API_URL", "http://localhost:8000").rstrip("/")

TEST_PHRASE = "The quick brown fox jumps over the lazy dog"
# Words that must survive the TTS -> STT round-trip.
EXPECTED_WORDS = ("quick", "brown", "fox", "lazy", "dog")


def _api_key() -> str:
    key = os.getenv("API_KEY", "").strip()
    if key:
        return key
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            match = re.match(r"^API_KEY=(.*)$", line.strip())
            if match:
                return match.group(1).strip().strip('"')
    raise RuntimeError("API_KEY not set and not found in .env")


def _request(
    path: str,
    *,
    data: bytes,
    content_type: str,
    timeout: float = 120.0,
) -> tuple[int, dict[str, str], bytes]:
    req = urllib.request.Request(
        f"{API_URL}{path}",
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": content_type,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, {k.lower(): v for k, v in exc.headers.items()}, exc.read()


def _multipart_file(field: str, filename: str, content_type: str, payload: bytes) -> tuple[bytes, str]:
    boundary = f"e2e-{uuid.uuid4().hex}"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'.encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            payload,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    return body, f"multipart/form-data; boundary={boundary}"


class AudioPipelineTests(unittest.TestCase):
    """TTS and STT endpoint tests; STT consumes the TTS output."""

    _tts_audio: bytes | None = None
    _tts_content_type: str = ""

    @classmethod
    def _generate_speech(cls) -> bytes:
        """Fetch (and cache) a TTS sample so tests also run standalone."""
        if cls._tts_audio is None:
            status, headers, body = _request(
                "/v1/audio/speech",
                data=json.dumps({"input": TEST_PHRASE, "voice": "Bella"}).encode(),
                content_type="application/json",
            )
            if status != 200:
                raise AssertionError(f"TTS request failed: {status} {body[:200]!r}")
            cls._tts_audio = body
            cls._tts_content_type = headers.get("content-type", "")
        return cls._tts_audio

    def test_tts_returns_playable_audio(self) -> None:
        audio = self._generate_speech()
        self.assertGreater(len(audio), 1000, "TTS returned suspiciously little audio data")
        self.assertTrue(
            self._tts_content_type.startswith("audio/"),
            f"Expected audio/* content type, got {self._tts_content_type!r}",
        )
        # MP3 frames start with an 0xFF sync byte (possibly after an ID3 tag);
        # WAV starts with RIFF.  Either counts as playable audio.
        self.assertTrue(
            audio.startswith((b"RIFF", b"ID3")) or audio[0] == 0xFF,
            f"Audio payload does not look like WAV or MP3 (starts with {audio[:8]!r})",
        )

    def test_whisper_transcribes_tts_audio(self) -> None:
        audio = self._generate_speech()
        extension = "wav" if audio.startswith(b"RIFF") else "mp3"
        body, content_type = _multipart_file(
            "file", f"sample.{extension}", f"audio/{extension}", audio
        )

        status, _headers, raw = _request(
            "/v1/audio/transcriptions",
            data=body,
            content_type=content_type,
            timeout=300.0,
        )
        self.assertEqual(200, status, f"STT request failed: {raw[:200]!r}")

        payload = json.loads(raw)
        text = payload.get("text", "")
        self.assertTrue(text.strip(), "Whisper returned an empty transcript")

        normalized = re.sub(r"[^a-z ]", "", text.lower())
        missing = [word for word in EXPECTED_WORDS if word not in normalized]
        self.assertFalse(
            missing,
            f"Transcript lost words {missing}; got: {text!r}",
        )

    def test_imagegen_returns_png(self) -> None:
        status, _headers, raw = _request(
            "/v1/images/generations",
            data=json.dumps(
                {"prompt": "a red fox sitting in snow, photo", "size": "512x512", "n": 1}
            ).encode(),
            content_type="application/json",
            timeout=300.0,
        )
        self.assertEqual(200, status, f"Imagegen request failed: {raw[:200]!r}")

        payload = json.loads(raw)
        data = payload.get("data")
        self.assertIsInstance(data, list)
        self.assertTrue(data, "Imagegen returned no images")

        image = base64.b64decode(data[0]["b64_json"])
        self.assertGreater(len(image), 10_000, "Decoded image is suspiciously small")
        self.assertTrue(
            image.startswith(b"\x89PNG\r\n\x1a\n"),
            f"Payload is not a PNG (starts with {image[:8]!r})",
        )

    def test_whisper_rejects_empty_upload(self) -> None:
        body, content_type = _multipart_file("file", "empty.wav", "audio/wav", b"")
        status, _headers, raw = _request(
            "/v1/audio/transcriptions", data=body, content_type=content_type
        )
        self.assertEqual(400, status, f"Expected 400 for empty upload, got {status}: {raw[:200]!r}")
        payload = json.loads(raw)
        self.assertIn("error", payload)


if __name__ == "__main__":
    unittest.main()
