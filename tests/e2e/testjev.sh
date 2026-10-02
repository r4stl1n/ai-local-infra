#!/usr/bin/env bash
# End-to-end check of Jev-style decision models (POST /v1/systemone) against the
# running stack.
#
#   tests/e2e/testjev.sh                 # tests every model in JEV_MODELS
#   tests/e2e/testjev.sh nimble          # just this model
#   tests/e2e/testjev.sh --unload        # also unload the models afterwards (frees VRAM)
#
# Configuration via environment (falls back to the repo .env for API_KEY):
#   E2E_API_URL   gateway base URL   (default http://localhost:8000)
#   API_KEY       gateway bearer token
#   JEV_MODELS    models to test     (default "tev1 nimble")
#
# Needs curl and python3 (stdlib only). Answer-quality checks are reported as
# WARN, not FAIL: a small model can be wrong without the endpoint being broken.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_URL="${E2E_API_URL:-http://localhost:8000}"
API_URL="${API_URL%/}"
MODELS="${JEV_MODELS:-tev1 nimble}"
UNLOAD=false
args=()
for arg in "$@"; do
  case "${arg}" in
    --unload) UNLOAD=true ;;
    *) args+=("${arg}") ;;
  esac
done
((${#args[@]})) && MODELS="${args[*]}"

if [[ -z "${API_KEY:-}" ]]; then
  API_KEY="$(grep -E '^API_KEY=' "${SCRIPT_DIR}/../../.env" 2>/dev/null | head -1 | cut -d= -f2- | tr -d '"'"'"'')"
fi
[[ -n "${API_KEY:-}" ]] || { echo "API_KEY not set and not found in .env" >&2; exit 1; }

PASS=0
FAIL=0
WARN=0
green() { printf '\033[32m%s\033[0m\n' "$*"; }
red() { printf '\033[31m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
ok() { PASS=$((PASS + 1)); green "  PASS  $*"; }
bad() { FAIL=$((FAIL + 1)); red "  FAIL  $*"; }
warn() { WARN=$((WARN + 1)); yellow "  WARN  $*"; }

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

# api METHOD PATH OUTFILE [curl args...] -> prints the HTTP status
api() {
  local method=$1 path=$2 out=$3
  shift 3
  curl -sS -o "${out}" -w '%{http_code}' --max-time 300 -X "${method}" \
    -H "Authorization: Bearer ${API_KEY}" "${API_URL}${path}" "$@"
}

error_message() {
  python3 -c 'import json,sys
try: print(json.load(open(sys.argv[1]))["error"]["message"])
except Exception: print(open(sys.argv[1]).read()[:300])' "$1"
}

expect_error() {
  local name=$1 want=$2 status=$3 resp=$4
  if [[ "${status}" == "${want}" ]]; then
    ok "${name}: ${status} — $(error_message "${resp}")"
  else
    bad "${name}: expected ${want}, got ${status}: $(error_message "${resp}")"
  fi
}

# request MODEL TICKET -> writes the /v1/systemone body for a support ticket
request() {
  python3 - "$1" "$2" <<'PY'
import json, sys
print(json.dumps({
    "model": sys.argv[1],
    "state": {"ticket": sys.argv[2]},
    "questions": {
        "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
                 "criteria": {"billing": "Payments and refunds", "technical": "Bugs and integrations",
                              "other": "None of the above"}},
        "refund": {"type": "noul", "instructions": "Does the customer explicitly ask for a refund?"},
        "urgency": {"type": "score", "instructions": "How urgent is this ticket?",
                    "criteria": ["Routine", "Soon", "Urgent"]},
    },
}))
PY
}

# check_answers RESPONSE WANT_TEAM WANT_REFUND(yes|no)
# Prints lines "PASS|WARN|FAIL<TAB>message": shape problems FAIL, wrong answers WARN.
check_answers() {
  python3 - "$@" <<'PY'
import json, sys
path, want_team, want_refund = sys.argv[1:4]
out = lambda level, msg: print(f"{level}\t{msg}")
try:
    d = json.load(open(path))
    a = d["answers"]
    team, refund, urgency = a["team"], a["refund"], a["urgency"]
except Exception as e:
    out("FAIL", f"response missing answers: {type(e).__name__}: {e}"); sys.exit()

problems = []
if team.get("type") != "choice" or team.get("choice") not in ("billing", "technical", "other"):
    problems.append(f"team is not a valid choice: {team}")
if abs(sum(team.get("probabilities", {}).values()) - 1) > 0.02:
    problems.append("team probabilities don't sum to 1")
if refund.get("type") != "noul" or not 0 <= refund.get("noul", -1) <= 1:
    problems.append(f"refund is not a 0-1 noul: {refund}")
if urgency.get("type") != "score" or not 0 <= urgency.get("score", -1) <= 2:
    problems.append(f"urgency score not on the 0-2 scale: {urgency}")
if urgency.get("legend") != {"0": "Routine", "1": "Soon", "2": "Urgent"}:
    problems.append(f"urgency legend unexpected: {urgency.get('legend')}")
usage = d.get("usage", {})
summary = (f"team={team.get('choice')} ({team.get('probabilities', {}).get(team.get('choice'), 0):.0%}), "
           f"refund={refund.get('noul', 0):.2f}, urgency={urgency.get('score', 0):.2f}/2 "
           f"[{usage.get('input_tokens')} in / {usage.get('output_tokens')} out tokens]")
if problems:
    out("FAIL", "; ".join(problems)); sys.exit()
out("PASS", f"typed answers OK: {summary}")
if team["choice"] != want_team:
    out("WARN", f"expected team={want_team}, model said {team['choice']}")
if (refund["noul"] > 0.5) != (want_refund == "yes"):
    out("WARN", f"expected refund={want_refund}, model gave {refund['noul']:.2f}")
PY
}

report() {
  local prefix=$1 level msg
  while IFS=$'\t' read -r level msg; do
    case "${level}" in
      PASS) ok "${prefix}: ${msg}" ;;
      WARN) warn "${prefix}: ${msg}" ;;
      *) bad "${prefix}: ${msg}" ;;
    esac
  done
}

echo "Jev decision-model tests against ${API_URL} (models: ${MODELS})"

echo
echo "== Service"
status=$(api GET /health "${TMP}/health")
[[ "${status}" == "200" ]] && ok "gateway /health" || { bad "gateway /health: HTTP ${status}"; exit 1; }

# An Ollama older than 0.35 has no /v1/systemone at all: every request below would
# 404 for that one reason, so check once and stop with the fix.
status=$(api POST /v1/systemone "${TMP}/probe" -H 'Content-Type: application/json' \
  -d '{"model": "no-such-decision-model", "state": {"a": "b"}, "questions": {"q": {"type": "noul", "instructions": "x"}}}')
if grep -q "0.35" "${TMP}/probe" 2>/dev/null; then
  bad "Ollama on the server is older than 0.35 (no /v1/systemone endpoint)"
  echo "        On the server: docker compose pull ollama && ./infra.sh start"
  echo "        then:          docker exec ai-infra-ollama ollama --version   # should be >= 0.35"
  red "Stopping: decision models need a newer Ollama."
  exit 1
fi
ok "Ollama serves /v1/systemone"

for model in ${MODELS}; do
  echo
  echo "== ${model}"
  SECONDS=0
  request "${model}" "I was charged twice. Please refund the extra payment." >"${TMP}/req.json"
  status=$(api POST /v1/systemone "${TMP}/billing" -H 'Content-Type: application/json' -d @"${TMP}/req.json")
  if [[ "${status}" != "200" ]]; then
    bad "billing ticket: HTTP ${status}: $(error_message "${TMP}/billing")"
    if [[ "${status}" == "404" ]] && grep -q "not found" "${TMP}/billing"; then
      echo "        (pull it on the server: ./infra.sh pull-models, or docker exec ai-infra-ollama ollama pull ${model})"
    fi
    continue
  fi
  report "billing ticket (${SECONDS}s, includes load on first call)" < <(check_answers "${TMP}/billing" billing yes)

  SECONDS=0
  request "${model}" "The app crashes every time I open the settings page since the last update. Error code 0x80." >"${TMP}/req.json"
  status=$(api POST /v1/systemone "${TMP}/tech" -H 'Content-Type: application/json' -d @"${TMP}/req.json")
  if [[ "${status}" == "200" ]]; then
    report "technical ticket (${SECONDS}s)" < <(check_answers "${TMP}/tech" technical no)
  else
    bad "technical ticket: HTTP ${status}: $(error_message "${TMP}/tech")"
  fi
done

echo
echo "== Request validation (should be rejected)"
first_model="${MODELS%% *}"
status=$(api POST /v1/systemone "${TMP}/e1" -H 'Content-Type: application/json' \
  -d '{"model": "no-such-decision-model", "state": {"a": "b"}, "questions": {"q": {"type": "noul", "instructions": "x"}}}')
expect_error "unknown model" 404 "${status}" "${TMP}/e1"
status=$(api POST /v1/systemone "${TMP}/e2" -H 'Content-Type: application/json' \
  -d "{\"model\": \"${first_model}\", \"state\": {\"a\": \"b\"}, \"questions\": {\"q\": {\"type\": \"bogus\", \"instructions\": \"x\"}}}")
expect_error "unknown question type" 400 "${status}" "${TMP}/e2"
status=$(api POST /v1/systemone "${TMP}/e3" -H 'Content-Type: application/json' -d 'not json')
expect_error "invalid JSON" 400 "${status}" "${TMP}/e3"
status=$(curl -sS -o "${TMP}/e4" -w '%{http_code}' -X POST "${API_URL}/v1/systemone" -d '{}')
expect_error "missing API key" 401 "${status}" "${TMP}/e4"

echo
echo "== VRAM"
status=$(api GET /v1/models/loaded "${TMP}/loaded")
if [[ "${status}" == "200" ]]; then
  ok "models in VRAM: $(python3 -c 'import json,sys; print(", ".join(m["id"] for m in json.load(open(sys.argv[1]))["data"]) or "none")' "${TMP}/loaded")"
else
  bad "GET /v1/models/loaded: HTTP ${status}"
fi
if ${UNLOAD}; then
  # Only the tested models that are actually resident (Ollama reports "tev1" as "tev1:latest").
  resident=$(python3 - "${TMP}/loaded" ${MODELS} <<'PY'
import json, sys
tag = lambda m: m if ":" in m.rsplit("/", 1)[-1] else m + ":latest"
loaded = {tag(m["id"]) for m in json.load(open(sys.argv[1]))["data"]}
print(" ".join(m for m in sys.argv[2:] if tag(m) in loaded))
PY
)
  [[ -z "${resident}" ]] && echo "  (none of the tested models are loaded; nothing to unload)"
  for model in ${resident}; do
    status=$(api POST /v1/models/unload "${TMP}/unload" -H 'Content-Type: application/json' -d "{\"model\": \"${model}\"}")
    if [[ "${status}" == "200" ]]; then
      ok "unloaded ${model}"
    else
      bad "unload ${model}: HTTP ${status}: $(error_message "${TMP}/unload")"
    fi
  done
fi

echo
summary="${PASS} passed, ${WARN} warnings, ${FAIL} failed"
if ((FAIL == 0)); then
  green "${summary}"
else
  red "${summary}. Logs: ./infra.sh logs ollama api"
  exit 1
fi
