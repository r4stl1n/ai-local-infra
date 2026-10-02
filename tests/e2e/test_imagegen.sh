#!/usr/bin/env bash
# End-to-end check of Krea 2 image generation against the running stack.
#
#   tests/e2e/test_imagegen.sh            # generation, references, style, errors
#   tests/e2e/test_imagegen.sh --vram     # also exercise the VRAM unload/load endpoints
#
# Configuration via environment (falls back to the repo .env for API_KEY):
#   E2E_API_URL   gateway base URL   (default http://localhost:8000)
#   API_KEY       gateway bearer token
#   OUT_DIR       where generated PNGs go (default ./imagegen-test-out)
#
# Needs curl and python3 (stdlib only). Open the PNGs in OUT_DIR to judge quality.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_URL="${E2E_API_URL:-http://localhost:8000}"
API_URL="${API_URL%/}"
OUT_DIR="${OUT_DIR:-./imagegen-test-out}"
TEST_VRAM=false
[[ "${1:-}" == "--vram" ]] && TEST_VRAM=true

if [[ -z "${API_KEY:-}" ]]; then
  API_KEY="$(grep -E '^API_KEY=' "${SCRIPT_DIR}/../../.env" 2>/dev/null | head -1 | cut -d= -f2- | tr -d '"'"'"'')"
fi
[[ -n "${API_KEY:-}" ]] || { echo "API_KEY not set and not found in .env" >&2; exit 1; }

mkdir -p "${OUT_DIR}"
PASS=0
FAIL=0

green() { printf '\033[32m%s\033[0m\n' "$*"; }
red() { printf '\033[31m%s\033[0m\n' "$*"; }
ok() { PASS=$((PASS + 1)); green "  PASS  $*"; }
bad() { FAIL=$((FAIL + 1)); red "  FAIL  $*"; }

# api METHOD PATH OUTFILE [curl args...] -> prints the HTTP status
api() {
  local method=$1 path=$2 out=$3
  shift 3
  curl -sS -o "${out}" -w '%{http_code}' --max-time 900 -X "${method}" \
    -H "Authorization: Bearer ${API_KEY}" "${API_URL}${path}" "$@"
}

# save_images RESPONSE_JSON PREFIX -> writes PREFIX_N.png, prints "WxH" of each
save_images() {
  python3 - "$1" "$2" <<'EOF'
import base64, json, struct, sys
data = json.load(open(sys.argv[1]))["data"]
for i, item in enumerate(data):
    png = base64.b64decode(item["b64_json"])
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    path = f"{sys.argv[2]}_{i}.png"
    open(path, "wb").write(png)
    w, h = struct.unpack(">II", png[16:24])
    print(f"{w}x{h} {path}")
EOF
}

# error_message RESPONSE_JSON -> the OpenAI-style error message (or the raw body)
error_message() {
  python3 -c 'import json,sys
try: print(json.load(open(sys.argv[1]))["error"]["message"])
except Exception: print(open(sys.argv[1]).read()[:300])' "$1"
}

# expect_image NAME STATUS RESPONSE PREFIX [EXPECTED_SIZE] — checks a generation result
expect_image() {
  local name=$1 status=$2 resp=$3 prefix=$4 want=${5:-}
  if [[ "${status}" != "200" ]]; then
    bad "${name}: HTTP ${status}: $(error_message "${resp}")"
    return 1
  fi
  local saved
  if ! saved="$(save_images "${resp}" "${prefix}")"; then
    bad "${name}: response did not contain PNG images"
    return 1
  fi
  if [[ -n "${want}" ]] && ! grep -q "^${want} " <<<"${saved}"; then
    bad "${name}: expected ${want}, got: ${saved}"
    return 1
  fi
  ok "${name} (${SECONDS}s): ${saved//$'\n'/, }"
}

# expect_error NAME WANT_STATUS STATUS RESPONSE
expect_error() {
  local name=$1 want=$2 status=$3 resp=$4
  if [[ "${status}" == "${want}" ]]; then
    ok "${name}: ${status} — $(error_message "${resp}")"
  else
    bad "${name}: expected ${want}, got ${status}: $(error_message "${resp}")"
  fi
}

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

echo "Krea 2 image tests against ${API_URL} (images -> ${OUT_DIR})"
echo "Note: the first request after a (re)start includes model warm-up."

echo
echo "== Service"
status=$(api GET /health "${TMP}/health")
[[ "${status}" == "200" ]] && ok "gateway /health" || { bad "gateway /health: HTTP ${status}"; exit 1; }
status=$(api GET /v1/images/models "${TMP}/models")
if [[ "${status}" == "200" ]]; then
  ok "image models: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("current=%s " % d["current"] + ", ".join("%s(downloaded=%s)" % (m["id"], m.get("downloaded")) for m in d["data"]))' "${TMP}/models")"
else
  bad "GET /v1/images/models: HTTP ${status}: $(error_message "${TMP}/models")"
fi

echo
echo "== Text-to-image (/v1/images/generations)"
SECONDS=0
status=$(api POST /v1/images/generations "${TMP}/t2i" -H 'Content-Type: application/json' \
  -d '{"prompt": "a red fox sitting in fresh snow at golden hour, photo, shallow depth of field", "size": "1024x1024"}')
expect_image "1024x1024 text-to-image" "${status}" "${TMP}/t2i" "${OUT_DIR}/01_t2i" 1024x1024
REF="${OUT_DIR}/01_t2i_0.png"

SECONDS=0
status=$(api POST /v1/images/generations "${TMP}/wide" -H 'Content-Type: application/json' \
  -d '{"prompt": "a lighthouse on a cliff above a stormy sea, dramatic clouds, oil painting", "size": "1536x1024"}')
expect_image "1536x1024 (landscape) text-to-image" "${status}" "${TMP}/wide" "${OUT_DIR}/02_wide" 1536x1024
WIDE="${OUT_DIR}/02_wide_0.png"

if [[ ! -f "${REF}" || ! -f "${WIDE}" ]]; then
  red "Text-to-image failed; skipping the reference-image tests (they reuse those images)."
else
  echo
  echo "== Reference images (/v1/images/edits)"
  SECONDS=0
  status=$(api POST /v1/images/edits "${TMP}/edit" \
    -F 'prompt=the same fox, now at night under the northern lights' -F "image[]=@${REF};type=image/png" -F size=1024x1024)
  expect_image "edit mode, multipart, 1 reference" "${status}" "${TMP}/edit" "${OUT_DIR}/03_edit" 1024x1024

  SECONDS=0
  status=$(api POST /v1/images/edits "${TMP}/auto" \
    -F 'prompt=the same lighthouse in bright summer sunshine' -F "image=@${WIDE};type=image/png" -F size=auto)
  expect_image "size=auto follows the landscape reference" "${status}" "${TMP}/auto" "${OUT_DIR}/04_auto" 1248x832

  # JSON body with data: URIs, two references, style mode (style-reference LoRA).
  python3 - "${WIDE}" "${REF}" >"${TMP}/style_req.json" <<'EOF'
import base64, json, sys
uri = lambda p: "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()
print(json.dumps({
    "prompt": "a white yeti with horns reading a book in a cozy library",
    "images": [{"image_url": uri(sys.argv[1])}],
    "reference_mode": "style",
    "size": "1024x1024",
}))
EOF
  SECONDS=0
  status=$(api POST /v1/images/edits "${TMP}/style" -H 'Content-Type: application/json' -d @"${TMP}/style_req.json")
  expect_image "style mode, JSON data URI (yeti in the lighthouse painting's style)" "${status}" "${TMP}/style" "${OUT_DIR}/05_style" 1024x1024

  SECONDS=0
  status=$(api POST /v1/images/edits "${TMP}/two" \
    -F 'prompt=the fox standing in front of the lighthouse' \
    -F "image[]=@${REF};type=image/png" -F "image[]=@${WIDE};type=image/png" -F size=1024x1024)
  expect_image "edit mode, 2 references" "${status}" "${TMP}/two" "${OUT_DIR}/06_two_refs" 1024x1024

  echo
  echo "== Request validation (should be rejected fast)"
  status=$(api POST /v1/images/edits "${TMP}/e1" -F prompt=x \
    -F "image[]=@${REF}" -F "image[]=@${REF}" -F "image[]=@${REF}")
  expect_error "3 reference images" 400 "${status}" "${TMP}/e1"
  status=$(api POST /v1/images/edits "${TMP}/e2" -F prompt=x -F "image=@${REF}" -F "mask=@${REF}")
  expect_error "mask (inpainting unsupported)" 400 "${status}" "${TMP}/e2"
  status=$(api POST /v1/images/edits "${TMP}/e3" -F prompt=x -F "image=@${REF}" -F reference_mode=bogus)
  expect_error "unknown reference_mode" 400 "${status}" "${TMP}/e3"
  status=$(api POST /v1/images/edits "${TMP}/e4" -H 'Content-Type: application/json' \
    -d '{"prompt": "x", "images": ["https://example.com/cat.png"]}')
  expect_error "remote image URL" 400 "${status}" "${TMP}/e4"
fi
status=$(api POST /v1/images/generations "${TMP}/e5" -H 'Content-Type: application/json' -d '{"prompt": " "}')
expect_error "empty prompt" 400 "${status}" "${TMP}/e5"
status=$(api POST /v1/images/models/load "${TMP}/e6" -H 'Content-Type: application/json' -d '{"model": "nope"}')
expect_error "unknown image model (current model stays loaded)" 400 "${status}" "${TMP}/e6"

if ${TEST_VRAM}; then
  echo
  echo "== VRAM endpoints"
  status=$(api GET /v1/models/loaded "${TMP}/llm")
  [[ "${status}" == "200" ]] && ok "LLM models in VRAM: $(cat "${TMP}/llm")" || bad "GET /v1/models/loaded: HTTP ${status}"
  status=$(api POST /v1/models/unload "${TMP}/llm_unload")
  [[ "${status}" == "200" ]] && ok "unload all LLMs: $(cat "${TMP}/llm_unload")" || bad "POST /v1/models/unload: HTTP ${status}: $(error_message "${TMP}/llm_unload")"

  current=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["current"] or "")' "${TMP}/models" 2>/dev/null)
  current="${current:-krea2-turbo}"
  status=$(api POST /v1/images/models/unload "${TMP}/img_unload")
  [[ "${status}" == "200" ]] && ok "unload image model: $(cat "${TMP}/img_unload")" || bad "POST /v1/images/models/unload: HTTP ${status}"
  status=$(api POST /v1/images/generations "${TMP}/unloaded" -H 'Content-Type: application/json' -d '{"prompt": "x"}')
  expect_error "generation while unloaded" 503 "${status}" "${TMP}/unloaded"
  SECONDS=0
  status=$(api POST /v1/images/models/load "${TMP}/img_load" -H 'Content-Type: application/json' -d "{\"model\": \"${current}\"}")
  [[ "${status}" == "200" ]] && ok "reload ${current} (${SECONDS}s): $(cat "${TMP}/img_load")" || bad "reload ${current}: HTTP ${status}: $(error_message "${TMP}/img_load")"
  SECONDS=0
  status=$(api POST /v1/images/generations "${TMP}/after" -H 'Content-Type: application/json' \
    -d '{"prompt": "a small robot watering a plant, studio photo", "size": "1024x1024"}')
  expect_image "generation after reload" "${status}" "${TMP}/after" "${OUT_DIR}/07_after_reload" 1024x1024
  echo "  (LLMs reload automatically on their next chat/embeddings request.)"
fi

echo
if ((FAIL == 0)); then
  green "All ${PASS} checks passed. Images are in ${OUT_DIR}/ — open them to check quality."
else
  red "${FAIL} failed, ${PASS} passed. Server logs: ./infra.sh logs imagegen"
  exit 1
fi
