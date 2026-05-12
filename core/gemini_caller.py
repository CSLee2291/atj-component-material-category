"""
Google Vertex AI Gemini caller
------------------------------
Mirror of core.gpt_caller using google-genai with Vertex AI mode.
The Azure path's user-prompt builders and JSON post-processing are reused, but
*system prompts* are loaded from the Gemini-specific prompt files
(data/prompts/gemini/{current_version}.md) so the two providers can diverge
when CE feedback is targeted at one of them.
"""
import json
import os
import logging
from config import settings
from core import prompt_loader
from core.gpt_caller import (
    _build_user_prompt,
    _build_category_select_prompt,
    _clean_gpt_result,
    _valid_categories_block,
)

PROVIDER = "gemini"

logger = logging.getLogger(__name__)

_client = None


def get_system_prompt() -> str:
    """Assemble the Gemini-flavored first-pass system prompt."""
    return prompt_loader.assemble(PROVIDER, "first_pass", _valid_categories_block())


def get_category_select_prompt() -> str:
    """Assemble the Gemini-flavored vector-fallback system prompt."""
    return prompt_loader.assemble(PROVIDER, "fallback", _valid_categories_block())


def _resolve_credentials_path() -> str:
    """Resolve the service-account JSON path. settings.gemini_credentials_path
    may be relative to the app folder; fall back to absolute path probing."""
    p = settings.gemini_credentials_path
    if os.path.isabs(p) and os.path.exists(p):
        return p
    # Relative to current working directory (app dir when run via main.py)
    cwd_path = os.path.abspath(p)
    if os.path.exists(cwd_path):
        return cwd_path
    # Relative to parent of app dir (the workflow root)
    parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", p))
    if os.path.exists(parent_path):
        return parent_path
    return cwd_path  # return best guess; caller will get a clear error


def _get_client():
    """Lazy-init a google-genai Client in Vertex AI mode using the service
    account JSON. Sets GOOGLE_APPLICATION_CREDENTIALS so ADC picks it up."""
    global _client
    if _client is not None:
        return _client

    cred_path = _resolve_credentials_path()
    if not os.path.exists(cred_path):
        raise FileNotFoundError(
            f"Gemini service-account JSON not found at: {cred_path}"
        )
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

    from google import genai  # imported lazily so Azure-only setups don't pay the cost

    _client = genai.Client(
        vertexai=True,
        project=settings.gemini_project_id,
        location=settings.gemini_location,
    )
    return _client


def _build_thinking_config():
    """Per-model thinking strategy.
    - Flash supports thinking_budget=0 (disable thinking → faster, predictable
      output).
    - Pro REQUIRES thinking enabled; passing 0 is rejected. Return None so the
      model uses its dynamic thinking default, and rely on max_output_tokens to
      cover the visible JSON.
    """
    from google.genai import types as genai_types

    model = (settings.gemini_model or "").lower()
    if "pro" in model:
        return None
    try:
        return genai_types.ThinkingConfig(thinking_budget=0)
    except Exception:
        return None  # older SDK / non-thinking model


async def _generate_json(system_prompt: str, user_prompt: str) -> dict:
    """Single Gemini call returning a parsed JSON dict.

    The system prompt embeds the full ~983-pair MATERIAL_CATEGORY whitelist, so
    Flash's default thinking budget would consume the per-call output quota
    before any visible JSON is produced — we disable thinking on Flash. Pro
    models do not allow thinking to be disabled, so we rely on a larger output
    budget instead.
    """
    from google.genai import types as genai_types

    client = _get_client()
    config = genai_types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.1,
        max_output_tokens=4096,
        response_mime_type="application/json",
        thinking_config=_build_thinking_config(),
    )
    response = await client.aio.models.generate_content(
        model=settings.gemini_model,
        contents=user_prompt,
        config=config,
    )
    finish = ""
    try:
        finish = str(response.candidates[0].finish_reason)
    except Exception:
        pass
    text = (response.text or "").strip()
    if not text:
        raise ValueError(f"Empty response from Gemini (finish_reason={finish})")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning(
            "Gemini truncated JSON finish=%s len=%d last=%r",
            finish, len(text), text[-200:],
        )
        raise ValueError(
            f"Gemini incomplete JSON (finish_reason={finish}, len={len(text)})"
        )
    # Some models (notably gemini-3.x preview) wrap the single suggestion in a
    # one-element JSON array. Unwrap so downstream code can call .get().
    if isinstance(parsed, list):
        if not parsed:
            raise ValueError("Gemini returned an empty JSON array")
        if not isinstance(parsed[0], dict):
            raise ValueError(f"Gemini returned non-dict array element: {type(parsed[0]).__name__}")
        return parsed[0]
    return parsed


async def suggest_category(target: dict, references: list[dict]) -> dict:
    """First-pass categorization using fuzzy-match references."""
    try:
        result = await _generate_json(
            get_system_prompt(),
            _build_user_prompt(target, references),
        )
        return _clean_gpt_result(result)
    except Exception as e:
        logger.exception("Gemini suggest_category failed")
        return {
            "ZZMCATG_M": "",
            "ZZMCATG_S": "",
            "MATERIAL_CATEGORY": "",
            "confidence": "error",
            "reason": f"Gemini error: {e}",
        }


async def suggest_category_from_candidates(
    target: dict,
    first_reason: str,
    candidates: list[dict],
) -> dict:
    """Second-pass categorization (vector-DB fallback)."""
    try:
        result = await _generate_json(
            get_category_select_prompt(),
            _build_category_select_prompt(target, first_reason, candidates),
        )
        result = _clean_gpt_result(result)
        result["source"] = "vector_fallback"
        return result
    except Exception as e:
        logger.exception("Gemini suggest_category_from_candidates failed")
        return {
            "ZZMCATG_M": "",
            "ZZMCATG_S": "",
            "MATERIAL_CATEGORY": "",
            "confidence": "error",
            "reason": f"Gemini vector fallback error: {e}",
            "source": "vector_fallback",
        }


async def test_connection() -> dict:
    """Lightweight ping for the LLM connection-test endpoint. Disables thinking
    on Flash so the small max_output_tokens budget actually surfaces text
    instead of being consumed by reasoning tokens; Pro models (which require
    thinking) get a larger output budget instead."""
    try:
        from google.genai import types as genai_types

        client = _get_client()
        thinking_cfg = _build_thinking_config()
        # Pro models keep thinking on, so they need more output tokens to ever
        # produce visible text on the ping.
        budget = 32 if thinking_cfg is not None else 256
        config = genai_types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=budget,
            thinking_config=thinking_cfg,
        )
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents="Reply with the single word: pong",
            config=config,
        )
        text = (response.text or "").strip()
        if not text:
            return {"ok": False, "error": "Empty response from Gemini (model may not be deployed in this region)"}
        return {"ok": True, "model": settings.gemini_model, "sample": text[:60]}
    except Exception as e:
        return {"ok": False, "error": str(e)}
