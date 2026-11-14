"""Claude vision backend (Anthropic API)."""

from __future__ import annotations

import logging

import anthropic

from .base import FatalTranslationError, PageTranslation, TranslationRequest, encode_jpeg
from .prompts import system_prompt

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"


class ClaudeTranslator:
    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "medium", honorifics: str = "keep",
                 max_image_side: int = 2000, api_key: str | None = None):
        # Without an explicit key, reads ANTHROPIC_API_KEY (or an `ant auth login` profile).
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_image_side = max_image_side
        # Identical on every page, so it is cached after the first request.
        self.system = [{"type": "text", "text": system_prompt(honorifics), "cache_control": {"type": "ephemeral"}}]

    def translate_page(self, request: TranslationRequest) -> PageTranslation:
        try:
            return self._translate(request)
        except anthropic.AuthenticationError as exc:
            raise FatalTranslationError("The Anthropic API key was rejected.") from exc
        except anthropic.PermissionDeniedError as exc:
            raise FatalTranslationError(f"This API key may not use {self.model}: {exc}") from exc
        except anthropic.NotFoundError as exc:
            raise FatalTranslationError(f"Model {self.model} was not found: {exc}") from exc

    def _translate(self, request: TranslationRequest) -> PageTranslation:
        response = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=16000,
            system=self.system,
            output_config={"effort": self.effort},
            output_format=PageTranslation,
            # If a safety classifier declines a page (it can happen with violent/mature manga),
            # retry it on a fallback model inside the same call instead of failing the chapter.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                 "data": encode_jpeg(request.marked_image, self.max_image_side)}},
                    {"type": "text", "text": request.user_text},
                ],
            }],
        )
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise RuntimeError(f"Claude declined page {request.page.image}: "
                               f"{getattr(details, 'category', None)} {getattr(details, 'explanation', '')}")
        if response.stop_reason == "max_tokens":
            raise RuntimeError(f"Response for {request.page.image} was cut off (max_tokens)")
        if response.parsed_output is None:
            raise RuntimeError(f"No structured output for {request.page.image}")

        u = response.usage
        log.info("%s: %d in (%d cached) / %d out tokens [%s]", request.page.image, u.input_tokens,
                 u.cache_read_input_tokens or 0, u.output_tokens, response._request_id)
        return response.parsed_output
