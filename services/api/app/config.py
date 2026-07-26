from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://ollama:11434"))
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "local").strip().lower())
    llm_thinking: bool = field(default_factory=lambda: _env_bool("LLM_THINKING", False))
    llm_num_ctx: int = field(default_factory=lambda: int(os.getenv("LLM_NUM_CTX", "65536")))
    llm_url: str = field(default_factory=lambda: os.getenv("LLM_URL", "").strip())
    llm_api_token: str = field(default_factory=lambda: (os.getenv("LLM_API_TOKEN") or os.getenv("LLM_API_KEY", "")).strip())
    tts_url: str = field(default_factory=lambda: os.getenv("TTS_URL", "http://kittentts:8100"))
    whisper_url: str = field(default_factory=lambda: os.getenv("WHISPER_URL", "http://whisper:8200"))
    imagegen_url: str = field(default_factory=lambda: os.getenv("IMAGEGEN_URL", "http://imagegen:8300"))
    demucs_url: str = field(default_factory=lambda: os.getenv("DEMUCS_URL", "http://demucs:8400"))
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", ""))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    port: int = field(default_factory=lambda: int(os.getenv("BACKEND_PORT", "8000")))
    llm_default_cost_input: float = field(default_factory=lambda: float(os.getenv("LLM_DEFAULT_COST_INPUT", "0.50")))
    llm_default_cost_output: float = field(default_factory=lambda: float(os.getenv("LLM_DEFAULT_COST_OUTPUT", "1.50")))
    backend_upstream_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("BACKEND_UPSTREAM_TIMEOUT_SECONDS", "900"))
    )

    def __post_init__(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "API_KEY environment variable is required but not set. "
                "Run ./infra.sh start to auto-generate one."
            )
        if self.llm_provider not in {"local", "remote"}:
            raise RuntimeError("LLM_PROVIDER must be either 'local' or 'remote'.")
        if self.llm_provider == "remote":
            if not self.llm_url:
                raise RuntimeError("LLM_URL is required when LLM_PROVIDER=remote.")
            if not self.llm_api_token:
                raise RuntimeError("LLM_API_TOKEN is required when LLM_PROVIDER=remote.")
        if self.backend_upstream_timeout_seconds <= 0:
            raise RuntimeError("BACKEND_UPSTREAM_TIMEOUT_SECONDS must be greater than 0.")

    @property
    def ollama_v1_url(self) -> str:
        return f"{self.ollama_url.rstrip('/')}/v1"

    @property
    def llm_upstream_url(self) -> str:
        if self.llm_provider == "remote":
            return self.llm_url.rstrip("/")
        return self.ollama_v1_url

    @property
    def llm_upstream_headers(self) -> dict[str, str] | None:
        if self.llm_provider == "remote":
            return {"Authorization": f"Bearer {self.llm_api_token}"}
        return None


settings = Settings()
