"""
LLM Router
----------
Dispatches the two GPT calls (first-pass and vector-fallback) to either
Azure OpenAI or Google Vertex AI Gemini, selected at request time.
Also exposes a health-check helper used by /api/llm/status to decide
which providers are eligible for the UI selector.
"""
import asyncio
from typing import Literal

from core import gpt_caller, gemini_caller
from config import settings

Provider = Literal["azure", "gemini"]


def _normalize(provider: str | None) -> Provider:
    p = (provider or settings.llm_provider or "azure").strip().lower()
    if p in ("gemini", "google", "vertex"):
        return "gemini"
    return "azure"


async def suggest_category(provider: str | None, target: dict, references: list[dict]) -> dict:
    if _normalize(provider) == "gemini":
        return await gemini_caller.suggest_category(target, references)
    return await gpt_caller.suggest_category(target, references)


async def suggest_category_from_candidates(
    provider: str | None,
    target: dict,
    first_reason: str,
    candidates: list[dict],
) -> dict:
    if _normalize(provider) == "gemini":
        return await gemini_caller.suggest_category_from_candidates(
            target, first_reason, candidates
        )
    return await gpt_caller.suggest_category_from_candidates(
        target, first_reason, candidates
    )


async def status() -> dict:
    """Probe both providers in parallel; return per-provider availability so
    the UI can offer only working LLMs."""
    azure_task = asyncio.create_task(gpt_caller.test_connection())
    gemini_task = asyncio.create_task(gemini_caller.test_connection())
    azure_res, gemini_res = await asyncio.gather(
        azure_task, gemini_task, return_exceptions=True
    )

    def _wrap(res):
        if isinstance(res, Exception):
            return {"ok": False, "error": str(res)}
        return res

    return {
        "providers": {
            "azure": {
                "label": "Azure OpenAI",
                "model": settings.azure_openai_deployment,
                **_wrap(azure_res),
            },
            "gemini": {
                "label": "Google Gemini (Vertex AI)",
                "model": settings.gemini_model,
                **_wrap(gemini_res),
            },
        },
        "default": _normalize(None),
    }
