"""
Azure OpenAI GPT caller
-----------------------
Sends enriched context (target item + top-5 similar references) to GPT
and parses the suggested ZZMCATG_M, ZZMCATG_S, and reasoning.
Uses the openai SDK (AsyncAzureOpenAI).
"""
import json
import re
from openai import AsyncAzureOpenAI
from config import settings


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


def _clean_gpt_result(result: dict) -> dict:
    """Post-process GPT JSON to ensure ZZMCATG_M/S are code-only and rebuild MATERIAL_CATEGORY."""
    if "ZZMCATG_M" in result:
        result["ZZMCATG_M"] = _clean_category_code(result["ZZMCATG_M"])
    if "ZZMCATG_S" in result:
        result["ZZMCATG_S"] = _clean_category_code(result["ZZMCATG_S"])
    if "ZZMCATG_M" in result and "ZZMCATG_S" in result:
        result["MATERIAL_CATEGORY"] = f"{result['ZZMCATG_M']}|{result['ZZMCATG_S']}"
    return result


SYSTEM_PROMPT = """You are a PLM (Product Lifecycle Management) component categorization expert
for Advantech. Your task is to assign the correct MATERIAL_CATEGORY to an electronic component
based on its description, manufacturer part number, and the categories of similar components.

MATERIAL_CATEGORY format is always: ZZMCATG_M|ZZMCATG_S
where ZZMCATG_M is the middle-level category CODE (e.g. "DAC", "FLH", "CLK") and
ZZMCATG_S is the small-level category CODE (e.g. "ADCX", "NORX", "RTCX").
CATE_M_NAME and CATE_S_NAME shown in references are descriptive names — do NOT include them in your output codes.

CRITICAL: ZZMCATG_M and ZZMCATG_S must be SHORT CODES ONLY (typically 2-4 uppercase letters).
Do NOT include descriptive names like "DAC (DATA CONVERTER)" — just return "DAC".

Rules:
- Always respond in valid JSON only, no markdown, no explanation outside the JSON.
- If confidence is low, set confidence to "low" and explain why in reason.
- Suggest only category codes that appear in the reference items provided.
- JSON format: {"ZZMCATG_M": "...", "ZZMCATG_S": "...", "MATERIAL_CATEGORY": "...", "confidence": "high|medium|low", "reason": "..."}
"""

# Lazy-init client (created on first call)
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


CATEGORY_SELECT_SYSTEM_PROMPT = """You are a PLM (Product Lifecycle Management) component categorization expert
for Advantech. You are given a target electronic component and a list of candidate MATERIAL_CATEGORY options
retrieved from a vector similarity search.

Your task is to select the BEST matching MATERIAL_CATEGORY from the candidates for the target component.

CRITICAL: ZZMCATG_M and ZZMCATG_S must be SHORT CODES ONLY (typically 2-4 uppercase letters).
Do NOT include descriptive names in the code fields. Example: return "DAC" not "DAC (DATA CONVERTER)".
The descriptive names (CATE_M_NAME, CATE_S_NAME) go in their own separate fields.

Rules:
- Always respond in valid JSON only, no markdown, no explanation outside the JSON.
- You MUST pick one of the provided candidate categories. Do not invent new codes.
- Consider the component's description and the first GPT analysis reason when choosing.
- JSON format: {"ZZMCATG_M": "...", "ZZMCATG_S": "...", "MATERIAL_CATEGORY": "...", "CATE_M_NAME": "...", "CATE_S_NAME": "...", "confidence": "high|medium|low", "reason": "..."}
"""


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
                {"role": "system", "content": CATEGORY_SELECT_SYSTEM_PROMPT},
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
                {"role": "system", "content": SYSTEM_PROMPT},
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
