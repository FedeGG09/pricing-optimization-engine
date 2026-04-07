from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import settings


@dataclass
class HFMessageResult:
    text: str
    raw: dict[str, Any] | None = None


class HuggingFaceLLMClient:
    def __init__(self, model_id: str | None = None, token: str | None = None, base_url: str | None = None):
        self.model_id = model_id or settings.hf_model_id
        self.token = token or settings.hf_token
        self.base_url = base_url or settings.hf_base_url
        self.timeout = settings.hf_timeout_seconds

    def _endpoint(self) -> str:
        if self.base_url:
            return self.base_url.rstrip("/")
        return f"https://api-inference.huggingface.co/models/{self.model_id}"

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 500) -> HFMessageResult:
        if not settings.enable_hf_agents:
            return HFMessageResult(text=self._fallback(system_prompt, user_prompt), raw=None)

        prompt = "\n\n".join([
            system_prompt,
            user_prompt,
            "Responde solamente con JSON válido.",
        ])
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.2,
                "top_p": 0.9,
                "return_full_text": False,
            },
        }

        try:
            with httpx.Client(timeout=self.timeout, headers=headers) as client:
                response = client.post(self._endpoint(), json=payload)
                response.raise_for_status()
                data = response.json()
        except Exception:
            return HFMessageResult(text=self._fallback(system_prompt, user_prompt), raw=None)

        text = ""
        raw: dict[str, Any] | None = None

        if isinstance(data, list) and data:
            item = data[0]
            if isinstance(item, dict):
                raw = item
                text = item.get("generated_text", "") or item.get("summary_text", "") or json.dumps(item, ensure_ascii=False)
            else:
                text = str(item)
        elif isinstance(data, dict):
            raw = data
            text = data.get("generated_text", "") or data.get("summary_text", "") or json.dumps(data, ensure_ascii=False)
        else:
            text = str(data)

        return HFMessageResult(text=text, raw=raw)

    def _fallback(self, system_prompt: str, user_prompt: str) -> str:
        return json.dumps(
            {
                "llm_enabled": False,
                "summary": "HF no disponible o deshabilitado. Se devuelve una explicación basada en reglas y métricas locales.",
                "system_prompt_hint": system_prompt[:160],
                "user_prompt_hint": user_prompt[:300],
            },
            ensure_ascii=False,
        )
