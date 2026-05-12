"""
CE Feedback engine
------------------
Backs the /api/feedback/* endpoints. Responsibilities:

  * Parse an uploaded XLSX of CE-validated corrections (mandated columns:
    Material, AI_MATERIAL_CATEGORY, CE_MATERIAL_CATEGORY; optional aliases
    AI_MATERIAL_CATEGORY_v2, Material Number, MPN, Item_Desc, CE_comment).
  * Extract AI≠CE rows and group them by (Item_Desc prefix, AI→CE) so we can
    emit a deterministic appendix to the active prompt's CE_EXAMPLES section.
  * Build a *proposed* next-version prompt file (raw markdown, not yet on disk
    as the active version) so the UI can preview before deploy.
  * Hold proposals in an in-memory store keyed by proposal_id; deploy() writes
    them to disk via prompt_loader and bumps the current.json pointer.
"""
from __future__ import annotations

import io
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Literal

import pandas as pd

from core import prompt_loader

Provider = Literal["azure", "gemini"]

# Mandated column names — the upload form documents these. Aliases below.
COL_ITEM = "Material"
COL_AI = "AI_MATERIAL_CATEGORY"
COL_CE = "CE_MATERIAL_CATEGORY"

# Aliases the parser will accept (case-insensitive, exact match).
COL_ITEM_ALIASES = {"material", "item_number", "material number"}
COL_AI_ALIASES = {"ai_material_category", "ai_material_category_v2"}
COL_CE_ALIASES = {"ce_material_category"}
COL_DESC_ALIASES = {"item_desc", "material number", "description", "desc"}
COL_MPN_ALIASES = {"mpn", "mfr_part_number"}
COL_COMMENT_ALIASES = {"ce_comment", "ce comment", "comment"}

PREFIX_RE = re.compile(r"@[A-Za-z!,]+|@[\u30A0-\u30FF]+")  # English / Japanese katakana


@dataclass
class FeedbackRow:
    item: str
    ai: str
    ce: str
    desc: str = ""
    mpn: str = ""
    comment: str = ""

    @property
    def prefix(self) -> str:
        m = PREFIX_RE.match(self.desc.strip()) if self.desc else None
        return m.group(0) if m else ""


@dataclass
class Proposal:
    proposal_id: str
    provider: Provider
    base_version: str
    next_version: str
    created_at: str
    rows_total: int
    rows_mismatch: int
    rows_match: int
    pattern_groups: list[dict[str, Any]]
    appendix_text: str
    proposed_md: str  # full text of the would-be vN+1.md
    sample_items: list[str]  # item numbers, used by the test step
    test_results: dict[str, Any] | None = None
    deployed_at: str | None = None


class _ProposalStore:
    def __init__(self) -> None:
        self._d: dict[str, Proposal] = {}
        self._lock = Lock()

    def put(self, p: Proposal) -> None:
        with self._lock:
            self._d[p.proposal_id] = p

    def get(self, pid: str) -> Proposal | None:
        with self._lock:
            return self._d.get(pid)

    def list_recent(self, limit: int = 20) -> list[Proposal]:
        with self._lock:
            return sorted(
                self._d.values(), key=lambda p: p.created_at, reverse=True
            )[:limit]


_store = _ProposalStore()


# ---------------------------------------------------------------------------
# XLSX parsing
# ---------------------------------------------------------------------------


def _norm_str(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def _resolve_column(df: pd.DataFrame, aliases: set[str]) -> str | None:
    """Find the first column whose lowercased name appears in `aliases`."""
    for c in df.columns:
        if str(c).strip().lower() in aliases:
            return c
    return None


def parse_feedback_xlsx(content: bytes, sheet: str | int = 0) -> tuple[list[FeedbackRow], dict[str, str]]:
    """Parse uploaded Excel bytes into FeedbackRow list. Raises ValueError if
    mandatory columns are missing. Returns (rows, column_map)."""
    df = pd.read_excel(io.BytesIO(content), sheet_name=sheet)
    col_item = _resolve_column(df, COL_ITEM_ALIASES)
    # Prefer the *latest* AI prediction column when the upload contains both v1
    # and v2: AI_MATERIAL_CATEGORY_v2 takes precedence over AI_MATERIAL_CATEGORY.
    col_ai = _resolve_column(df, {"ai_material_category_v2"}) or _resolve_column(df, COL_AI_ALIASES)
    col_ce = _resolve_column(df, COL_CE_ALIASES)
    col_desc = _resolve_column(df, COL_DESC_ALIASES)
    col_mpn = _resolve_column(df, COL_MPN_ALIASES)
    col_cmt = _resolve_column(df, COL_COMMENT_ALIASES)

    missing = []
    if col_item is None:
        missing.append("Material (item number)")
    if col_ai is None:
        missing.append("AI_MATERIAL_CATEGORY (or AI_MATERIAL_CATEGORY_v2)")
    if col_ce is None:
        missing.append("CE_MATERIAL_CATEGORY")
    if missing:
        raise ValueError(
            "Required columns missing from upload: " + ", ".join(missing)
        )

    rows: list[FeedbackRow] = []
    for _, r in df.iterrows():
        item = _norm_str(r[col_item])
        ai = _norm_str(r[col_ai])
        ce = _norm_str(r[col_ce])
        if not item:
            continue
        rows.append(FeedbackRow(
            item=item, ai=ai, ce=ce,
            desc=_norm_str(r[col_desc]) if col_desc is not None else "",
            mpn=_norm_str(r[col_mpn]) if col_mpn is not None else "",
            comment=_norm_str(r[col_cmt]) if col_cmt is not None else "",
        ))
    column_map = {
        "item": str(col_item),
        "ai": str(col_ai),
        "ce": str(col_ce),
        "desc": str(col_desc) if col_desc is not None else "",
        "mpn": str(col_mpn) if col_mpn is not None else "",
        "comment": str(col_cmt) if col_cmt is not None else "",
    }
    return rows, column_map


# ---------------------------------------------------------------------------
# Pattern extraction → appendix lines
# ---------------------------------------------------------------------------


def _group_mismatches(rows: list[FeedbackRow]) -> list[dict[str, Any]]:
    """Group AI≠CE rows by (prefix, AI→CE) and return summary dicts sorted by count desc."""
    groups: dict[tuple[str, str, str], list[FeedbackRow]] = {}
    for r in rows:
        if not r.ai or not r.ce or r.ai == r.ce:
            continue
        key = (r.prefix, r.ai, r.ce)
        groups.setdefault(key, []).append(r)
    summary: list[dict[str, Any]] = []
    for (prefix, ai, ce), items in groups.items():
        summary.append({
            "prefix": prefix,
            "ai": ai,
            "ce": ce,
            "count": len(items),
            "samples": [
                {"item": r.item, "mpn": r.mpn, "desc": r.desc, "comment": r.comment}
                for r in items[:5]
            ],
            "all_items": [r.item for r in items],
        })
    summary.sort(key=lambda g: (-g["count"], g["prefix"], g["ai"], g["ce"]))
    return summary


def _build_appendix(provider: Provider, groups: list[dict[str, Any]], rows_mismatch: int) -> str:
    """Emit a deterministic appendix block to insert into CE_EXAMPLES."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines: list[str] = []
    lines.append(
        f"\n--- CE-Validated Correction Examples (uploaded {today} — {provider}, "
        f"{rows_mismatch} disagreement{'s' if rows_mismatch != 1 else ''}) ---"
    )
    lines.append(
        "Each line below was a real AI prediction that CE corrected. Use these as ground truth."
    )

    for g in groups:
        head_prefix = g["prefix"] or "(no prefix)"
        lines.append(
            f"\n{head_prefix}  {g['ai']} → {g['ce']}  ({g['count']} item{'s' if g['count'] != 1 else ''}):"
        )
        for s in g["samples"]:
            desc_snip = s["desc"][:80] if s["desc"] else (s["mpn"] or s["item"])
            line = f'  "{desc_snip}" → {g["ce"]}  (NOT {g["ai"]})'
            if s["comment"]:
                line += f"  // CE: {s['comment'][:60]}"
            lines.append(line)
        if len(g["all_items"]) > len(g["samples"]):
            extra = len(g["all_items"]) - len(g["samples"])
            lines.append(f"  ...and {extra} more item{'s' if extra != 1 else ''} with the same correction.")

    return "\n".join(lines)


def _splice_into_section(md_text: str, section: str, appendix: str) -> str:
    """Append `appendix` to the end of section `section` in a vN.md file."""
    pattern = re.compile(
        rf"(^##\s+{re.escape(section)}\s*\n)(.*?)(?=^##\s+\w+|\Z)",
        re.DOTALL | re.MULTILINE,
    )
    m = pattern.search(md_text)
    if not m:
        raise ValueError(f"Could not locate section `## {section}` in prompt file")
    head = m.group(1)
    body = m.group(2)
    body_trimmed = body.rstrip("\n")
    new_body = body_trimmed + "\n\n" + appendix.strip("\n") + "\n\n"
    return md_text[: m.start()] + head + new_body + md_text[m.end():]


# ---------------------------------------------------------------------------
# Public API used by the routes
# ---------------------------------------------------------------------------


def build_proposal(provider: Provider, content: bytes, sheet: str | int = 0) -> Proposal:
    """End-to-end: parse upload → extract patterns → render appendix → render
    proposed next-version markdown. The proposal is held in memory, not yet
    on disk."""
    rows, _col_map = parse_feedback_xlsx(content, sheet=sheet)
    groups = _group_mismatches(rows)
    rows_mismatch = sum(g["count"] for g in groups)
    rows_match = len([r for r in rows if r.ai and r.ce and r.ai == r.ce])

    appendix = _build_appendix(provider, groups, rows_mismatch)
    base_version = prompt_loader.current_version(provider)
    base_md = prompt_loader.read_raw(provider, base_version)
    proposed_md = _splice_into_section(base_md, "CE_EXAMPLES", appendix)
    next_version = prompt_loader.next_version_id(provider)

    sample_items = [r.item for r in rows if r.ai and r.ce and r.ai != r.ce]

    proposal = Proposal(
        proposal_id=uuid.uuid4().hex[:12],
        provider=provider,
        base_version=base_version,
        next_version=next_version,
        created_at=datetime.now(timezone.utc).isoformat(),
        rows_total=len(rows),
        rows_mismatch=rows_mismatch,
        rows_match=rows_match,
        pattern_groups=groups,
        appendix_text=appendix,
        proposed_md=proposed_md,
        sample_items=sample_items,
    )
    _store.put(proposal)
    return proposal


def get_proposal(proposal_id: str) -> Proposal | None:
    return _store.get(proposal_id)


def list_recent_proposals(limit: int = 20) -> list[Proposal]:
    return _store.list_recent(limit)


def write_test_results(proposal_id: str, results: dict[str, Any]) -> None:
    p = _store.get(proposal_id)
    if p is None:
        raise KeyError(proposal_id)
    p.test_results = results


def deploy_proposal(proposal_id: str) -> dict[str, Any]:
    """Persist the proposed markdown as a new versioned file and bump
    current.json so live calls pick it up."""
    p = _store.get(proposal_id)
    if p is None:
        raise KeyError(proposal_id)
    if p.deployed_at:
        return {"already_deployed_at": p.deployed_at, "version": p.next_version}
    path = prompt_loader.write_version(p.provider, p.next_version, p.proposed_md)
    prompt_loader.set_current_version(p.provider, p.next_version)
    p.deployed_at = datetime.now(timezone.utc).isoformat()
    return {
        "provider": p.provider,
        "version": p.next_version,
        "path": str(path),
        "deployed_at": p.deployed_at,
    }


def proposal_to_dict(p: Proposal) -> dict[str, Any]:
    """JSON-safe view used by the routes."""
    return {
        "proposal_id": p.proposal_id,
        "provider": p.provider,
        "base_version": p.base_version,
        "next_version": p.next_version,
        "created_at": p.created_at,
        "rows_total": p.rows_total,
        "rows_mismatch": p.rows_mismatch,
        "rows_match": p.rows_match,
        "pattern_groups": p.pattern_groups,
        "appendix_text": p.appendix_text,
        "sample_items": p.sample_items,
        "test_results": p.test_results,
        "deployed_at": p.deployed_at,
    }
