"""
Azure OpenAI GPT caller
-----------------------
Sends enriched context (target item + top-5 similar references) to GPT
and parses the suggested ZZMCATG_M, ZZMCATG_S, and reasoning.
Uses the openai SDK (AsyncAzureOpenAI).

System prompts (the long instruction blocks with whitelist + CE-validated examples)
are loaded at call time from data/prompts/azure/{current_version}.md via
core.prompt_loader, so the CE Feedback page can deploy new prompt revisions
without a code change.
"""
import json
import os
import re
import logging
from openai import AsyncAzureOpenAI
from config import settings
from core import prompt_loader

PROVIDER = "azure"

logger = logging.getLogger(__name__)


def _clean_category_code(val: str) -> str:
    """
    Strip descriptive names that GPT sometimes appends to category codes.
    E.g. 'DAC (DATA CONVERTER)' → 'DAC', 'DACX (D TO A CONVERTER...)' → 'DACX'
    Also handles 'DAC - DATA CONVERTER' style.
    """
    if not val:
        return val
    val = str(val).strip()
    # Remove anything in parentheses: "DAC (DATA CONVERTER)" → "DAC"
    val = re.sub(r"\s*\(.*?\)\s*$", "", val).strip()
    # Remove trailing " - description": "DAC - DATA CONVERTER" → "DAC"
    val = re.sub(r"\s*-\s+[A-Za-z].*$", "", val).strip()
    return val


# ---------------------------------------------------------------------------
# Valid [MATERIAL_CATEGORY] whitelist
# ---------------------------------------------------------------------------
# GPT must only suggest MATERIAL_CATEGORY values that exist in the PLM database
# (distinct_categories cache). This prevents hallucinated codes like "CLK|CLKX"
# which don't actually exist (correct value is CLK|CLKG).

_VALID_CATEGORIES_SET: set[str] | None = None
_VALID_CATEGORIES_BLOCK: str | None = None


def _load_valid_categories() -> tuple[set[str], str]:
    """
    Load the authoritative list of valid MATERIAL_CATEGORY values from
    distinct_categories.parquet. Returns (set_for_lookup, formatted_block_for_prompt).
    Cached on first call.
    """
    global _VALID_CATEGORIES_SET, _VALID_CATEGORIES_BLOCK
    if _VALID_CATEGORIES_SET is not None and _VALID_CATEGORIES_BLOCK is not None:
        return _VALID_CATEGORIES_SET, _VALID_CATEGORIES_BLOCK

    valid_set: set[str] = set()
    formatted_lines: list[str] = []
    try:
        import pandas as pd
        cache_path = os.path.join(settings.target_cache_dir, "distinct_categories.parquet")
        if os.path.exists(cache_path):
            df = pd.read_parquet(cache_path)
            # Keep only rows with valid M|S pair
            df = df.dropna(subset=["ZZMCATG_M", "ZZMCATG_S"])
            df = df[(df["ZZMCATG_M"].astype(str).str.strip() != "")
                    & (df["ZZMCATG_S"].astype(str).str.strip() != "")]
            df = df.sort_values(["ZZMCATG_M", "ZZMCATG_S"]).reset_index(drop=True)
            for _, r in df.iterrows():
                m = str(r["ZZMCATG_M"]).strip()
                s = str(r["ZZMCATG_S"]).strip()
                mc = f"{m}|{s}"
                valid_set.add(mc)
                m_name = str(r.get("CATE_M_NAME", "") or "").strip()
                s_name = str(r.get("CATE_S_NAME", "") or "").strip()
                formatted_lines.append(f"  {mc}  ({m_name} > {s_name})")
    except Exception as e:
        logger.warning("Could not load valid categories whitelist: %s", e)

    _VALID_CATEGORIES_SET = valid_set
    _VALID_CATEGORIES_BLOCK = "\n".join(formatted_lines) if formatted_lines else "(whitelist unavailable)"
    return _VALID_CATEGORIES_SET, _VALID_CATEGORIES_BLOCK


def _valid_categories_block() -> str:
    _, block = _load_valid_categories()
    return block


def _is_valid_category(material_category: str) -> bool:
    """True if MATERIAL_CATEGORY exists in the distinct_categories whitelist."""
    valid, _ = _load_valid_categories()
    if not valid:
        # Whitelist unavailable — do not block, just accept.
        return True
    return material_category in valid


def _clean_gpt_result(result: dict) -> dict:
    """Post-process GPT JSON to ensure ZZMCATG_M/S are code-only, rebuild MATERIAL_CATEGORY,
    and validate against the distinct_categories whitelist."""
    if "ZZMCATG_M" in result:
        result["ZZMCATG_M"] = _clean_category_code(result["ZZMCATG_M"])
    if "ZZMCATG_S" in result:
        result["ZZMCATG_S"] = _clean_category_code(result["ZZMCATG_S"])
    if "ZZMCATG_M" in result and "ZZMCATG_S" in result:
        result["MATERIAL_CATEGORY"] = f"{result['ZZMCATG_M']}|{result['ZZMCATG_S']}"

    # Whitelist validation: if GPT emitted a code not in [MATERIAL_CATEGORY],
    # downgrade confidence and prepend a warning to reason so CE can review.
    mc = result.get("MATERIAL_CATEGORY", "")
    if mc and not _is_valid_category(mc):
        orig_reason = result.get("reason", "")
        result["confidence"] = "low"
        result["reason"] = (
            f"[INVALID_CATEGORY: '{mc}' is NOT in the [MATERIAL_CATEGORY] whitelist] "
            + orig_reason
        )
        # Flag the invalid output so downstream code (e.g. pipeline fallback) can detect it
        result["invalid_category"] = True
    return result


def get_system_prompt() -> str:
    """Assemble the first-pass system prompt for Azure from the active version
    on disk, with the whitelist block injected."""
    return prompt_loader.assemble(PROVIDER, "first_pass", _valid_categories_block())


def get_category_select_prompt() -> str:
    """Assemble the vector-fallback system prompt for Azure."""
    return prompt_loader.assemble(PROVIDER, "fallback", _valid_categories_block())


# --- Legacy module-level aliases for tests / scripts that import these names ---
# Live calls always go through get_system_prompt() / get_category_select_prompt() so
# the prompt the LLM actually sees stays in sync with whatever the CE Feedback page
# deployed at runtime. The lambdas below render the *currently active* prompt each
# time someone reads them as strings.

class _DynamicPrompt:
    def __init__(self, fn):
        self._fn = fn
    def __str__(self):
        return self._fn()
    def __repr__(self):
        return f"<DynamicPrompt {self._fn.__name__}>"
    def __format__(self, spec):
        return format(str(self), spec)

SYSTEM_PROMPT = _DynamicPrompt(get_system_prompt)
CATEGORY_SELECT_SYSTEM_PROMPT = _DynamicPrompt(get_category_select_prompt)
ITEM_DESC_PREFIX_GUIDE = _DynamicPrompt(
    lambda: prompt_loader.get_sections(PROVIDER).get("PREFIX_GUIDE", "")
)


# Lazy-init Azure OpenAI client (created on first call)
_client: AsyncAzureOpenAI | None = None


def _get_client() -> AsyncAzureOpenAI:
    global _client
    if _client is None:
        _client = AsyncAzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
    return _client


def _build_user_prompt(target: dict, references: list[dict]) -> str:
    ref_lines = []
    for i, r in enumerate(references, 1):
        ref_lines.append(
            f"  {i}. Item: {r.get('Item_Number', '')} | "
            f"MPN: {r.get('MFR_PART_NUMBER', '')} | "
            f"Desc: {r.get('Item_Desc', '')} | "
            f"Middle cat: {r.get('ZZMCATG_M', '')} ({r.get('CATE_M_NAME', '')}) | "
            f"Small cat: {r.get('ZZMCATG_S', '')} ({r.get('CATE_S_NAME', '')}) | "
            f"Similarity: {r.get('similarity_score', 0):.1f}/100"
        )
    refs_block = "\n".join(ref_lines) if ref_lines else "  (no similar items found)"

    return f"""Target component needing MATERIAL_CATEGORY:
- Item_Number:      {target.get('Item_Number', '')}
- Item_Desc:        {target.get('Item_Desc', '')}
- Manufacturer:     {target.get('MANUFACTURE_NAME', '')}
- MFR_PART_NUMBER:  {target.get('MFR_PART_NUMBER', '')}
- LifeCycle_Phase:  {target.get('LifeCycle_Phase', '')}

Top {len(references)} most similar components for reference:
{refs_block}

Based on the above, suggest ZZMCATG_M and ZZMCATG_S for the target component.
Respond with valid JSON only."""


def _build_category_select_prompt(
    target: dict,
    first_reason: str,
    candidates: list[dict],
) -> str:
    cand_lines = []
    for i, c in enumerate(candidates, 1):
        cand_lines.append(
            f"  {i}. ZZMCATG_M: {c.get('ZZMCATG_M', '')} ({c.get('CATE_M_NAME', '')}) | "
            f"ZZMCATG_S: {c.get('ZZMCATG_S', '')} ({c.get('CATE_S_NAME', '')}) | "
            f"MATERIAL_CATEGORY: {c.get('MATERIAL_CATEGORY', '')} | "
            f"Vector similarity: {c.get('similarity', 0):.3f}"
        )
    cands_block = "\n".join(cand_lines)

    return f"""Target component needing MATERIAL_CATEGORY:
- Item_Number:      {target.get('Item_Number', '')}
- Item_Desc:        {target.get('Item_Desc', '')}
- Manufacturer:     {target.get('MANUFACTURE_NAME', '')}
- MFR_PART_NUMBER:  {target.get('MFR_PART_NUMBER', '')}

Initial analysis (from first pass — low confidence):
  {first_reason}

Top {len(candidates)} candidate categories from vector search:
{cands_block}

Based on the component description and the initial analysis, select the BEST matching
MATERIAL_CATEGORY from the candidates above. Respond with valid JSON only."""


async def suggest_category_from_candidates(
    target: dict,
    first_reason: str,
    candidates: list[dict],
) -> dict:
    """
    Second-pass GPT call: given a target item and top-K category candidates
    from the vector DB, select the best match.
    Returns dict with ZZMCATG_M, ZZMCATG_S, MATERIAL_CATEGORY, confidence, reason.
    """
    try:
        client = _get_client()
        response = await client.chat.completions.create(
            model=settings.azure_openai_deployment,
            messages=[
                {"role": "system", "content": get_category_select_prompt()},
                {"role": "user", "content": _build_category_select_prompt(
                    target, first_reason, candidates
                )},
            ],
            temperature=0.1,
            max_completion_tokens=300,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        result = _clean_gpt_result(result)
        # Tag as vector-fallback result
        result["source"] = "vector_fallback"
        return result
    except Exception as e:
        return {
            "ZZMCATG_M": "",
            "ZZMCATG_S": "",
            "MATERIAL_CATEGORY": "",
            "confidence": "error",
            "reason": f"Vector fallback GPT error: {e}",
            "source": "vector_fallback",
        }


async def suggest_category(target: dict, references: list[dict]) -> dict:
    """
    Calls Azure OpenAI and returns parsed suggestion dict:
      {ZZMCATG_M, ZZMCATG_S, MATERIAL_CATEGORY, confidence, reason}
    Falls back to error dict on any exception.
    """
    try:
        client = _get_client()
        response = await client.chat.completions.create(
            model=settings.azure_openai_deployment,
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": _build_user_prompt(target, references)},
            ],
            temperature=0.1,
            max_completion_tokens=300,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        result = _clean_gpt_result(result)
        return result
    except Exception as e:
        return {
            "ZZMCATG_M": "",
            "ZZMCATG_S": "",
            "MATERIAL_CATEGORY": "",
            "confidence": "error",
            "reason": str(e),
        }


async def test_connection() -> dict:
    """Lightweight ping for the LLM connection-test endpoint."""
    try:
        if not settings.azure_openai_api_key or not settings.azure_openai_endpoint:
            return {"ok": False, "error": "Missing Azure OpenAI endpoint or API key in .env"}
        client = _get_client()
        response = await client.chat.completions.create(
            model=settings.azure_openai_deployment,
            messages=[{"role": "user", "content": "ping"}],
            max_completion_tokens=8,
        )
        text = (response.choices[0].message.content or "").strip()
        return {"ok": True, "model": settings.azure_openai_deployment, "sample": text[:60]}
    except Exception as e:
        return {"ok": False, "error": str(e)}
