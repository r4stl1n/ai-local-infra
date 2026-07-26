# AI Infra

Self-hosted model-serving stack for any OpenAI-compatible client. Runs an authenticated AI gateway in front of local GPU/model services so they can be hosted on a dedicated machine:

| Service | Purpose | Port |
|---|---|---|
| **api** | OpenAI-compatible gateway (`/v1/*`) with Bearer auth; the stack's only published port | 8000 |
| **ollama** | LLM inference + embeddings | internal |
| **ollama-init** | One-shot model pull on first start (skips models already present, so startup works offline) | - |
| **whisper** | Speech-to-text (faster-whisper, GPU) | internal |
| **kittentts** | Text-to-speech (KittenTTS, CPU) | internal |
| **imagegen** | Text-to-image (SDXL — Juggernaut-XL by default, GPU) | internal |
| **demucs** | Music source separation (vocals/stems, Demucs, GPU; off by default) | internal |

All traffic goes through **api**, which requires `Authorization: Bearer ${API_KEY}` and exposes `/v1/chat/completions`, `/v1/embeddings`, `/v1/models`, `/v1/audio/transcriptions` (plus a `/v1/audio/transcriptions/stream` WebSocket), `/v1/audio/speech`, `/v1/images/generations`, and `/v1/audio/separations` (stem separation, when enabled), plus `/health`. The model services themselves are not published; add port mappings in `docker-compose.yml` if you need direct access.

## OpenAI compatibility

The gateway aims to be a drop-in `base_url` for OpenAI SDKs:

- **Chat completions** support streaming SSE, tool calls, and vision-style content-part messages (text parts are flattened and base64 `image_url` parts forwarded as images for Ollama).
- **Sampling params** (`temperature`, `top_p`, `max_tokens`/`max_completion_tokens`, `stop`, `seed`, `presence_penalty`, `frequency_penalty`) are translated to Ollama options in local mode; an Ollama-native `options` object passes through and wins on conflict.
- **Errors** use the OpenAI shape (`{"error": {"message", "type", "param", "code"}}`) with real HTTP status codes, including on streaming requests. `finish_reason` distinguishes `stop`, `length`, and `tool_calls`.
- **Thinking models**: the non-standard `think: true|false` request field (default: `LLM_THINKING`) toggles reasoning; when off, `<think>` blocks are stripped from responses, streaming included. In remote mode, passing `think` explicitly also injects the vLLM `chat_template_kwargs.enable_thinking` toggle; otherwise the outgoing body stays strictly OpenAI-spec.
- **Not supported**: `n > 1` and `logprobs`; streaming responses always include `usage` in the final chunk.

With `LLM_PROVIDER=remote`, chat and `/v1/models` proxy to `LLM_URL`, which may be a bare host, a `.../v1` base, or a full `.../v1/chat/completions` URL — the gateway normalizes the path either way. Embeddings always use the bundled Ollama.

## Prerequisites

- Docker with Compose v2 (`docker compose`)
- NVIDIA GPU with drivers and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (ollama, whisper, imagegen)

## Quick Start

```bash
./infra.sh start        # creates .env from .env.example on first run
./infra.sh start-light  # only API, Ollama, TTS (KittenTTS) and image gen (no STT/separation)
./infra.sh pull-models  # (re-)pull the models listed in OLLAMA_PULL_MODELS, updating existing ones
./infra.sh status
```

On first run `infra.sh` creates `.env` with a freshly generated `API_KEY` — read it back with `grep '^API_KEY=' .env` and give it to your clients. If you copy `.env.example` by hand instead, set `API_KEY` yourself (`openssl rand -hex 32`); the gateway refuses to start with an empty key rather than run unauthenticated. Model weights and caches live in `.data/` next to this file.

Once the models are downloaded, the stack starts fully offline: every service loads from the `.data/` caches first and only reaches the network when something is missing.

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `INFRA_BIND_IP` | `0.0.0.0` | Host IP for the published api port (all interfaces by default; set `127.0.0.1` for same-host-only or a LAN IP to pin an interface) |
| `API_KEY` | _(generated)_ | Bearer token required on every `/v1/*` request; `infra.sh` generates one when it creates `.env` |
| `LLM_PROVIDER` | `local` | `local` (bundled Ollama) or `remote` (OpenAI-compatible `LLM_URL` + `LLM_API_TOKEN`) |
| `LLM_THINKING` / `LLM_NUM_CTX` | `false` / `65536` | Default request behavior for local models |
| `BACKEND_UPSTREAM_TIMEOUT_SECONDS` | `900` | Gateway → upstream timeout budget |
| `OLLAMA_PULL_MODELS` | `"gemma4:12b snowflake-arctic-embed:137m"` | Space-separated models pulled by `ollama-init` (keep the quotes — the file is `source`d by `infra.sh`) |
| `ENABLE_STT` / `ENABLE_TTS` / `ENABLE_IMAGEGEN` | `true` | Toggle the optional services (compose profiles) |
| `WHISPER_MODEL` | `large-v3-turbo` | Whisper model size |
| `TTS_MODEL` | `KittenML/kitten-tts-nano-0.8` | KittenTTS model |
| `IMAGEGEN_MODEL` | `RunDiffusion/Juggernaut-XL-v9` | Image generation model (any SDXL-family HF checkpoint) |
| `IMAGEGEN_STEPS` / `IMAGEGEN_GUIDANCE` | `30` / `5.0` | Inference steps and CFG scale, tuned per model — full SDXL checkpoints want ~30 / 5.0; for a faster distillate set `IMAGEGEN_MODEL=stabilityai/sdxl-turbo` with `4` / `0.0` |
| `IMAGEGEN_OFFLINE` | `0` | `1` = never fetch image weights over the network; cached models load, missing ones error fast instead of hanging. Pre-fetch with `./infra.sh pull-models` |

The active image model can also be listed/swapped/unloaded at runtime via
`GET/POST /v1/images/models*` — see [integrate.md](integrate.md).

## Connecting a client

Full endpoint-by-endpoint integration reference (request/response shapes, error format, extensions — written to be pasted into an LLM's context): [integrate.md](integrate.md).

Point any OpenAI-compatible client at the gateway with a Bearer token (replace `HOST` with the address of the machine running the stack):

```bash
curl http://HOST:8000/v1/models \
  -H "Authorization: Bearer $API_KEY"
```

A client running in Docker on the same machine can reach the gateway at `http://host.docker.internal:8000` (with `extra_hosts: host.docker.internal:host-gateway` on Linux).

## Testing

```bash
# Gateway unit tests (run inside the api image; the host needs no deps)
docker run --rm -v $PWD/services/api:/src -w /src -e API_KEY=test \
  $(docker inspect ai-infra-api --format '{{.Image}}') python -m unittest discover -s tests

# End-to-end audio round-trip against the running stack (stdlib only):
# TTS generates a sample, Whisper transcribes it back
python3 -m unittest discover -s tests/e2e -v
```

The e2e tests read `API_KEY` from the environment or `.env`, and target `E2E_API_URL` (default `http://localhost:8000`).

## Security note

The api gateway listens on all interfaces by default and authenticates every `/v1/*` request. `infra.sh` generates a random 32-byte `API_KEY` on first run, so the stack is never exposed with a default credential — keep the host on a trusted network, and firewall port `8000` (or set `INFRA_BIND_IP=127.0.0.1`) if the machine is internet-facing.
