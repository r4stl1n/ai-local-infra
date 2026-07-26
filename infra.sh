#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_FILE="${SCRIPT_DIR}/.env"

is_enabled() {
  local normalized
  normalized=$(echo "${1:-true}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')
  case "${normalized}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

generate_key() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex 32
  else
    # head closing the pipe SIGPIPEs tr, which pipefail would treat as failure.
    ( LC_ALL=C tr -dc 'a-f0-9' </dev/urandom | head -c 64 ) || true
  fi
}

ensure_env() {
  if [ ! -f "${ENV_FILE}" ]; then
    local key
    key=$(generate_key)
    if [ "${#key}" -lt 32 ]; then
      echo "Could not generate an API key (need openssl or /dev/urandom)." >&2
      echo "Copy .env.example to .env by hand and set API_KEY yourself." >&2
      exit 1
    fi
    # Seed .env with a fresh key rather than the placeholder, so a first run is
    # never reachable with a credential that is published in this repo.
    awk -v key="${key}" '
      !seen && /^API_KEY=/ { print "API_KEY=" key; seen = 1; next }
      { print }
    ' "${SCRIPT_DIR}/.env.example" >"${ENV_FILE}"
    chmod 600 "${ENV_FILE}"
    echo "Created ${ENV_FILE} from .env.example with a freshly generated API_KEY."
    echo "Clients need it — read it with: grep '^API_KEY=' ${ENV_FILE}"
  fi
  set -a
  source "${ENV_FILE}"
  set +a

  case "${API_KEY:-}" in
    ""|changeme)
      echo "⚠️  API_KEY in ${ENV_FILE} is empty or still 'changeme'." >&2
      echo "   The gateway would be reachable with a publicly known credential." >&2
      echo "   Set a strong one:  API_KEY=\$(openssl rand -hex 32)" >&2
      ;;
  esac
}

compose() {
  local profiles=()
  is_enabled "${ENABLE_STT:-true}" && profiles+=(--profile stt)
  is_enabled "${ENABLE_TTS:-true}" && profiles+=(--profile tts)
  is_enabled "${ENABLE_IMAGEGEN:-true}" && profiles+=(--profile imagegen)
  is_enabled "${ENABLE_SEPARATION:-false}" && profiles+=(--profile separation)
  HOST_UID=$(id -u) HOST_GID=$(id -g) \
    docker compose --env-file "${ENV_FILE}" "${profiles[@]}" "$@"
}

ensure_data_dirs() {
  mkdir -p "${SCRIPT_DIR}"/.data/{ollama,whisper,kittentts,imagegen,demucs}
}

start_stack() {
  compose up -d
  echo ""
  echo "✅ AI infra is starting"
  local api_host="${INFRA_BIND_IP:-0.0.0.0}"
  if [ "${api_host}" = "0.0.0.0" ]; then
    api_host=$(hostname -I 2>/dev/null | awk '{print $1}')
    api_host="${api_host:-<this-host>}"
  fi
  echo "  API (OpenAI-compatible): http://${api_host}:8000 (bound to ${INFRA_BIND_IP:-0.0.0.0})"
  echo "  Auth: Bearer \${API_KEY} from .env"
  echo "  STT=$(is_enabled "${ENABLE_STT:-true}" && echo on || echo off)" \
       "TTS=$(is_enabled "${ENABLE_TTS:-true}" && echo on || echo off)" \
       "ImageGen=$(is_enabled "${ENABLE_IMAGEGEN:-true}" && echo on || echo off)" \
       "Separation=$(is_enabled "${ENABLE_SEPARATION:-false}" && echo on || echo off)"
}

case "${1:-}" in
  start)
    ensure_env
    ensure_data_dirs
    start_stack
    ;;
  start-light)
    ensure_env
    ensure_data_dirs
    # Minimal stack: API + Ollama (always on) plus TTS (KittenTTS) and image gen.
    # Force STT (whisper) and separation (demucs) off regardless of .env.
    ENABLE_STT=false
    ENABLE_TTS=true
    ENABLE_IMAGEGEN=true
    ENABLE_SEPARATION=false
    start_stack
    ;;
  stop)
    ensure_env
    compose down --remove-orphans
    echo "✅ AI infra stopped"
    ;;
  pull-models)
    ensure_env
    ensure_data_dirs
    OLLAMA_FORCE_PULL=1 compose up --no-deps --exit-code-from ollama-init ollama-init
    # Pre-fetch the image checkpoint into .data/imagegen so the first (or offline)
    # start doesn't stall on a silent multi-GB download.
    if is_enabled "${ENABLE_IMAGEGEN:-true}"; then
      img_model="${IMAGEGEN_MODEL:-RunDiffusion/Juggernaut-XL-v9}"
      echo "Pulling image model: ${img_model}"
      compose run --rm --no-deps \
        -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 -e HF_HOME=/data/imagegen \
        --entrypoint python3 imagegen \
        -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ['IMAGEGEN_MODEL'])"
    fi
    ;;
  rebuild)
    ensure_env
    shift
    compose build --no-cache "$@"
    ;;
  logs)
    ensure_env
    shift || true
    compose logs -f "$@"
    ;;
  status)
    ensure_env
    compose ps
    ;;
  *)
    cat <<USAGE
Usage: ./infra.sh <command>

Commands:
  start        Start the stack (creates .env from .env.example on first run)
  start-light  Start only API, Ollama, TTS (KittenTTS) and image gen
  stop         Stop the stack
  pull-models  Pull the models listed in OLLAMA_PULL_MODELS
  rebuild      Rebuild service image(s), e.g. ./infra.sh rebuild whisper
  logs         Follow logs, e.g. ./infra.sh logs whisper
  status       Show container status
USAGE
    exit 1
    ;;
esac
