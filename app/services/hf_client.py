from __future__ import annotations

import json
import logging
import os
import traceback
from functools import lru_cache
from typing import Any, Optional, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class HFLLMClient:
    """
    Cliente Hugging Face Router usando OpenAI SDK.

    Usa Chat Completions para texto/multimodal.
    Para reasoning avanzado / structured outputs más fuertes,
    conviene migrar luego a Responses API.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        debug: Optional[bool] = None,
    ) -> None:
        self.token = (
            token
            or os.getenv("HF_TOKEN")
            or os.getenv("HF_API_TOKEN")
            or ""
        ).strip()

        self.base_url = (
            base_url
            or os.getenv("HF_API_BASE")
            or os.getenv("HF_LLM_BASE_URL")
            or "https://router.huggingface.co/v1"
        ).strip()

        self.model = (
            model
            or os.getenv("HF_DEFAULT_MODEL")
            or os.getenv("HF_LLM_MODEL")
            or "Qwen/Qwen3.5-9B:together"
        ).strip()

        try:
            self.timeout_seconds = int(
                timeout_seconds
                or os.getenv("HF_TIMEOUT_SECONDS")
                or "60"
            )
        except Exception:
            self.timeout_seconds = 60

        if debug is None:
            env_debug = os.getenv("HF_DEBUG", "false").strip().lower()
            debug = env_debug in {"1", "true", "yes", "y", "on"}
        self.debug = bool(debug)

        self.last_error: Optional[str] = None

        if not self.token or self.token.startswith("hf_TU_TOKEN"):
            logger.warning("HF_TOKEN no configurado. Las llamadas al LLM fallarán hasta definir un token real.")

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.token,
            timeout=self.timeout_seconds,
        )

    def chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        kwargs: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        try:
            if self.debug:
                logger.info(
                    "HF LLM request | model=%s | base_url=%s | temperature=%s | max_tokens=%s",
                    kwargs["model"],
                    self.base_url,
                    temperature,
                    max_tokens,
                )

            resp = self.client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""

            if self.debug:
                logger.info("HF LLM response received | chars=%s", len(content))

            return content

        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.exception(
                "HF LLM error | model=%s | base_url=%s | error=%s",
                kwargs.get("model"),
                self.base_url,
                self.last_error,
            )
            if self.debug:
                logger.error("HF LLM traceback:\n%s", traceback.format_exc())
                logger.error("HF LLM request payload: %s", kwargs)
            raise

    def chat_multimodal(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_url: str,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]

        kwargs: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        try:
            if self.debug:
                logger.info(
                    "HF multimodal request | model=%s | image_url=%s",
                    kwargs["model"],
                    image_url,
                )

            resp = self.client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""

            if self.debug:
                logger.info("HF multimodal response received | chars=%s", len(content))

            return content

        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.exception(
                "HF multimodal error | model=%s | error=%s",
                kwargs.get("model"),
                self.last_error,
            )
            if self.debug:
                logger.error("HF multimodal traceback:\n%s", traceback.format_exc())
                logger.error("HF multimodal request payload: %s", kwargs)
            raise

    def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        raw = self.chat(
            system_prompt=system_prompt + "\nRespondé SOLO con JSON válido, sin markdown ni texto extra.",
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parsed = self._extract_json(raw)

        if self.debug and not parsed:
            logger.warning("HF JSON parse vacío. Respuesta cruda: %s", raw)

        return parsed

    def structured(
        self,
        *,
        schema_model: Type[T],
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> T:
        data = self.chat_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._validate_schema(schema_model, data)

    def probe(self, prompt: str = "Decime hola en una sola frase.") -> str:
        return self.chat(
            system_prompt="Sos un asistente breve y útil.",
            user_prompt=prompt,
            temperature=0.2,
        )

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {}

        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()

        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                return {}

        return {}

    @staticmethod
    def _validate_schema(schema_model: Type[T], data: dict[str, Any]) -> T:
        if hasattr(schema_model, "model_validate"):
            return schema_model.model_validate(data)
        return schema_model.parse_obj(data)


@lru_cache(maxsize=1)
def get_hf_client() -> HFLLMClient:
    return HFLLMClient()#