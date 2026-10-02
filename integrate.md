# Integration Guide — AI Infra API Gateway

Machine-readable integration reference for the OpenAI-compatible gateway.
Written so an LLM (or a human) can integrate against it without reading the
source. Everything below reflects actual gateway behavior, not aspiration.

## TL;DR

- **Base URL**: `http://<host>:8000` (default port `8000`; `HOST` in the examples below is the machine running the stack)
- **Auth**: `Authorization: Bearer <API_KEY>` header on **every** `/v1/*` request. The key is the `API_KEY` value from the server's `.env`.
- **Protocol**: OpenAI REST API shapes. Any OpenAI SDK works by setting `base_url` and `api_key`.
- **No auth needed** for: `GET /health`, `GET /docs`, `GET /openapi.json`.

```python
from openai import OpenAI
client = OpenAI(base_url="http://HOST:8000/v1", api_key="<API_KEY>")
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness probe → `{"status": "ok"}` (no auth) |
| GET | `/v1/models` | List available models |
| POST | `/v1/chat/completions` | Chat (streaming, tools, vision) |
| POST | `/v1/embeddings` | Text embeddings |
| POST | `/v1/audio/speech` | Text-to-speech |
| POST | `/v1/audio/transcriptions` | Speech-to-text (multipart upload) |
| WS | `/v1/audio/transcriptions/stream` | Streaming speech-to-text |
| POST | `/v1/images/generations` | Text-to-image |
| POST | `/v1/images/edits` | Image generation from 1–2 reference images |
| GET | `/v1/models/loaded` | LLM/embedding models currently in VRAM |
| POST | `/v1/models/unload` | Free VRAM held by LLM/embedding models |

---

### GET /v1/models

Returns the models currently available (OpenAI list shape). Model ids are
Ollama tags (e.g. `gemma4:12b`, `snowflake-arctic-embed:137m`) — use these
exact strings as `model` in other requests.

```json
{"object": "list", "data": [{"id": "gemma4:12b", "object": "model", "created": 1784484563, "owned_by": "library"}]}
```

---

### POST /v1/chat/completions

OpenAI chat-completions request/response. `model` and non-empty `messages`
are required (400 otherwise).

**Supported request fields**

| Field | Notes |
|---|---|
| `model` | Required. An id from `/v1/models` |
| `messages` | Required. Roles: `system`, `user`, `assistant`, `tool`. Content may be a string **or** an OpenAI content-part array — `text` parts are concatenated; `image_url` parts must be `data:` URIs (base64) and are passed to the model as images |
| `stream` | `true` → SSE stream (see below) |
| `temperature`, `top_p`, `seed`, `presence_penalty`, `frequency_penalty` | Forwarded to the model |
| `max_tokens` / `max_completion_tokens` | Completion-token cap (produces `finish_reason: "length"` when hit) |
| `stop` | String or array of stop sequences |
| `tools` | OpenAI function-calling tool definitions |
| `response_format` | `{"type": "json_object"}` → constrains output to JSON |
| `think` | **Non-standard extension.** `true`/`false` toggles model reasoning. Default comes from server config (`LLM_THINKING`, default `false`). When off, `<think>...</think>` blocks are stripped from responses, including mid-stream |
| `options` | **Non-standard extension.** Raw Ollama options object (e.g. `{"num_ctx": 8192, "top_k": 40}`); overrides any mapped standard field on conflict |

Context window defaults to the server's `LLM_NUM_CTX` (65536) unless
overridden via `options.num_ctx`.

**Not supported**: `n > 1` (single choice only), `logprobs`,
`stream_options` (usage is always included in the final stream chunk).

**Non-streaming response** (standard OpenAI shape):

```json
{
  "id": "chatcmpl-abc123def456",
  "object": "chat.completion",
  "created": 1784485558,
  "model": "gemma4:12b",
  "choices": [{"index": 0,
               "message": {"role": "assistant", "content": "...", "tool_calls": [/* if any */]},
               "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 19, "completion_tokens": 6, "total_tokens": 25}
}
```

`finish_reason` is one of `stop`, `length` (token cap hit), `tool_calls`.
Tool calls follow the OpenAI shape: `{"id": "call_...", "type": "function",
"index": 0, "function": {"name": "...", "arguments": "<JSON string>"}}`.

**Streaming**: `Content-Type: text/event-stream`; each event is
`data: <chat.completion.chunk JSON>`; deltas carry `role` (first chunk),
`content`, and/or `tool_calls`; the final chunk has an empty delta,
the `finish_reason`, and a `usage` object; the stream ends with
`data: [DONE]`. Errors detected before streaming begins return a normal
HTTP error status with a JSON error body (never a 200 stream).

```bash
curl http://HOST:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "gemma4:12b", "messages": [{"role": "user", "content": "hi"}],
       "stream": true, "max_tokens": 100, "temperature": 0.7}'
```

---

### POST /v1/embeddings

```json
{"model": "snowflake-arctic-embed:137m", "input": "one string or an array of strings"}
```

`model` and `input` are required (400 otherwise). Response is the OpenAI
list shape; vector dimension depends on the model (768 for
`snowflake-arctic-embed:137m`).

```json
{"object": "list",
 "data": [{"object": "embedding", "index": 0, "embedding": [0.01, ...]}],
 "model": "snowflake-arctic-embed:137m",
 "usage": {"prompt_tokens": 6, "total_tokens": 6}}
```

Embeddings always run on the local Ollama regardless of the server's LLM
provider setting.

---

### POST /v1/audio/speech (TTS)

```json
{"input": "Text to speak", "voice": "Bella", "speed": 1.0, "stream": false}
```

| Field | Default | Notes |
|---|---|---|
| `input` | required | Text to synthesize (400 if empty) |
| `voice` | `"Bella"` | Voice name understood by the configured TTS model |
| `speed` | `1.0` | `0 < speed <= 5.0` |
| `stream` | `false` | `false` → complete MP3 (`audio/mpeg`); `true` → chunked WAV stream (`audio/wav`) |
| `model` | ignored | Server uses its configured TTS model |

Response body is raw audio bytes, not JSON. There is no OpenAI
`response_format` field — format is determined by `stream` as above.

---

### POST /v1/audio/transcriptions (STT)

`multipart/form-data` upload, field name **`file`** (audio in any common
format — wav, mp3, m4a, ogg, webm...). Optional **query parameters** (not
form fields — an OpenAI SDK's `language` form field is ignored; append
`?language=en` to the URL instead): `language` (ISO code hint), `model`
(ignored; server uses its configured Whisper model). Empty files are
rejected with 400.

```bash
curl http://HOST:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@recording.mp3;type=audio/mp3" 
```

Response: `{"text": "the transcription"}` (single `text` field; no
segment/timestamp detail).

---

### WS /v1/audio/transcriptions/stream

WebSocket for live transcription. Auth via query parameter (headers are
awkward in browser WebSocket clients):

```
ws://HOST:8000/v1/audio/transcriptions/stream?token=<API_KEY>
```

Send binary audio chunks; receive JSON text frames from the Whisper
service. While a segment is open the service re-decodes the growing buffer
and emits interim `{"type": "partial", "text": "..."}` frames (self-paced,
roughly every WHISPER_PARTIAL_MS of new audio; partials may revise earlier
words). Control messages (client → server, JSON text frames):

| Message | Effect |
|---|---|
| `{"type": "end"}` | Decode the buffered segment, emit the final frame, clear the buffer |
| `{"type": "drop"}` | Discard the buffered segment silently (e.g. to re-send a cleaned-up version of the same audio before ending) |

Final frames carry the decoder's confidence and language signals alongside
the text:

```json
{"type": "final", "text": "...", "language": "fr",
 "language_probability": 0.93, "avg_logprob": -0.31, "no_speech_prob": 0.08}
```

`avg_logprob` and `no_speech_prob` are duration-weighted means over the
decoded segments — high `no_speech_prob` combined with low `avg_logprob` is
Whisper's signature for hallucinated text over speechless audio. `language`
lets callers route (e.g. skip translation for English) without a second
model. The final always supersedes any partials for that segment. Invalid
token → close code `1008`.

---

### POST /v1/audio/separations (music source separation)

`multipart/form-data` upload, field name **`file`** (a music mix in any
common format). Requires the stack to run with `ENABLE_SEPARATION=true`
(off by default). Optional **query parameters**:

| Param | Default | Notes |
|---|---|---|
| `stem` | `vocals` | `vocals`, `drums`, `bass`, `other`, or `no_vocals`/`accompaniment` (sum of everything but vocals) |
| `sample_rate` | model rate (44100) | Resample the returned stem, e.g. `16000` to feed STT directly |
| `mono` | `false` | Downmix the returned stem to one channel |
| `model` | ignored | Server uses its configured Demucs model |

Response body is raw WAV bytes (s16le) of the requested stem. Typical use:
separate a music segment's vocals, then POST them to
`/v1/audio/transcriptions` for lyric-quality STT.

```bash
curl "http://HOST:8000/v1/audio/separations?stem=vocals&sample_rate=16000&mono=true" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@song.wav;type=audio/wav" -o vocals.wav
```

Separation runs on the shared GPU and is serialized server-side; expect
roughly ~0.05x real-time per request (a 10s clip in well under a second)
once the model is warm.

---

### POST /v1/images/generations

```json
{"prompt": "a lighthouse at sunset, photo", "size": "1024x1024", "n": 1}
```

| Field | Default | Notes |
|---|---|---|
| `prompt` | required | 400 if empty. No 77-token limit — Krea 2 encodes prompts with a Qwen3-VL LLM, so long, descriptive prompts work well |
| `negative_prompt` | `""` | Only takes effect with CFG (`krea/Krea-2-Raw`); ignored by the default Turbo model |
| `size` | `"1024x1024"` | `"WxH"`, each dimension clamped to 256–1536 and snapped to a multiple of 16 (so `1536x1024` / `1024x1536` work). `"auto"` = `1024x1024` |
| `n` | `1` | 1–4 images (generated sequentially) |
| `model` | ignored | Server uses its configured diffusion model |
| `response_format` | `b64_json` | Only `b64_json` is supported; there is no `url` mode |

Response: `{"created": <unix>, "data": [{"b64_json": "<base64 PNG>"}]}`.
Decode `b64_json` to get a PNG file.

If no model is loaded (after `unload`), generation returns `503`.

---

### POST /v1/images/edits

Generates an image from a prompt **plus 1–2 reference images**: the model sees
the references as context, so this covers editing ("make the sky purple"),
subject reference ("this dog wearing a space suit"), combining two images, and
— with `reference_mode: "style"` — rendering a new prompt in a reference's style.
It's the OpenAI `images/edits` endpoint, so `client.images.edit(...)` works.

Multipart (OpenAI SDK / curl):

```bash
curl http://HOST:8000/v1/images/edits -H "Authorization: Bearer $API_KEY" \
  -F prompt="the same cat, wearing a tiny wizard hat" \
  -F "image[]=@cat.png" -F size=auto
```

JSON (references as `data:` URIs or bare base64):

```json
{"prompt": "a white yeti reading a book", "images": [{"image_url": "data:image/png;base64,..."}], "reference_mode": "style"}
```

| Field | Default | Notes |
|---|---|---|
| `image` / `image[]` (multipart) or `images` (JSON) | required | 1–2 reference images (PNG/JPEG/WebP, ≤ 20 MB each). JSON entries may be a string or `{"image_url": ...}` / `{"b64_json": ...}`; remote `http(s)` URLs are not fetched (400) |
| `prompt` | required | Describe the result or the change you want |
| `reference_mode` | `"edit"` | Extension. `edit`: references as subject/edit context. `style`: generate the prompt in the references' style (style-reference LoRA) |
| `size` | `"auto"` | `auto` matches the first reference's aspect ratio at ~1 MP; otherwise as in `/generations` |
| `n`, `negative_prompt`, `model`, `response_format` | | As in `/generations` |
| `mask` | — | Not supported (400): Krea 2 conditions on whole images and does not inpaint |

Response: same shape as `/generations`.

---

### Image model management

The image model can be listed, swapped, and unloaded at runtime. Only one
model is resident at a time; loading a new one unloads the current one and
frees its VRAM. Loads are slow (weights + VRAM) and the request blocks until
the new model is ready.

```
GET  /v1/images/models              -> {"current": "<repo>", "data": [{"id": "<repo>", "loaded": bool}, ...]}
POST /v1/images/models/load  {"model": "<repo>"}   -> {"current": "<repo>"}
POST /v1/images/models/unload                      -> {"current": null}
```

`data` lists the image checkpoints present in the local cache. With
`IMAGEGEN_OFFLINE=1`, `load` refuses a model that isn't already cached
(400) rather than trying to download it.

---

### Freeing VRAM

The LLM (Ollama) and the image model share the GPU. To make room for one,
unload the other; both reload paths are explicit and cheap to call.

```
GET  /v1/models/loaded                       -> {"data": [{"id": "gemma4:12b", "size_vram": <bytes>, "expires_at": "..."}]}
POST /v1/models/unload  {"model": "<id>"}    -> {"unloaded": ["<id>"]}
POST /v1/models/unload  (no body)            -> {"unloaded": [<every resident model>]}
POST /v1/images/models/unload                -> {"current": null}
POST /v1/images/models/load {"model": "<repo>"} -> {"current": "<repo>"}
```

- `/v1/models/*` acts on the bundled Ollama, which hosts the chat models
  (local mode) and the embedding model; with no `model`, everything resident
  is unloaded. The call returns once the memory is released. An LLM reloads
  automatically on its next chat/embeddings request (cold-load latency).
  Unknown model → `404`.
- The image model does **not** reload by itself: after
  `/v1/images/models/unload`, image requests return `503` until you call
  `/v1/images/models/load` (see above).

Typical swap: `POST /v1/models/unload` → `POST /v1/images/models/load` →
generate → `POST /v1/images/models/unload` → chat resumes (auto-reload).

---

## Errors

All errors use the OpenAI error object with a meaningful HTTP status:

```json
{"error": {"message": "model 'nope' not found", "type": "not_found_error", "param": null, "code": null}}
```

| Status | `type` | Typical cause |
|---|---|---|
| 400 | `invalid_request_error` | Missing/invalid field, malformed JSON, empty upload |
| 401 | `authentication_error` | Missing or wrong Bearer token (`code: "invalid_api_key"`) |
| 404 | `not_found_error` | Unknown model |
| 502 | `api_error` | A backing service is down or returned garbage |

Streaming requests that fail before generation starts return these same
JSON errors with real status codes — a `200` + `text/event-stream`
response means generation actually began.

## Operational notes for clients

- **Timeouts**: the gateway allows upstream calls up to 900 s. Set client
  timeouts generously for image generation (several seconds per image, more with reference images) and long chat
  completions; first request after a model swap may add cold-load time.
- **Concurrency**: chat runs up to 2 requests in parallel; more are queued
  server-side (up to 32) rather than rejected. Other endpoints serialize on
  GPU work — expect latency, not errors, under load.
- **Model choice**: chat models and the embedding model are discoverable
  via `/v1/models`. TTS voice/model, Whisper model, and the image model are
  server-side configuration; requests can't switch them.
- **Cost/usage**: every response includes real token `usage`; the server
  logs a cost estimate per request, but no billing is enforced.
- **Transport**: HTTP only by default — put a TLS reverse proxy in front
  for untrusted networks, and treat the API key as a secret.
