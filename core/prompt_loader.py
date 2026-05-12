"""
Prompt loader
-------------
Reads provider-specific versioned prompt files from data/prompts/{provider}/vN.md,
parses their `## SECTION_NAME` blocks, injects the runtime-rendered MATERIAL_CATEGORY
whitelist, and assembles the two system prompts the LLM callers need:

  * first_pass — used by suggest_category()
  * fallback   — used by suggest_category_from_candidates()

The active version per provider is stored in data/prompts/current.json. This file
is re-read on every assemble() call so the running app picks up newly deployed
versions without a restart.

Sections expected in each vN.md:
  FIRST_PASS_HEADER, FALLBACK_HEADER, PREFIX_GUIDE, CE_EXAMPLES,
  FIRST_PASS_RULES, FALLBACK_RULES
"""
import json
import os
import re
from pathlib import Path
from threading import Lock
from typing import Literal

Provider = Literal["azure", "gemini"]
Kind = Literal["first_pass", "fallback"]

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "data" / "prompts"
CURRENT_FILE = PROMPTS_DIR / "current.json"
WHITELIST_PLACEHOLDER = "{{WHITELIST_BLOCK}}"

# Cache parsed prompt files keyed by (provider, version) so the disk parse only
# happens once per version, not once per call.
_section_cache: dict[tuple[str, str], dict[str, str]] = {}
_cache_lock = Lock()


def _strip_frontmatter(text: str) -> str:
    """Drop a leading `---\n...\n---\n` block if present."""
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end >= 0:
            return text[end + 5 :]
    return text


def _parse_sections(md_text: str) -> dict[str, str]:
    """Parse a markdown file with `## SECTION_NAME` headers into a dict of
    section_name → body text (with leading/trailing whitespace stripped)."""
    body = _strip_frontmatter(md_text)
    sections: dict[str, str] = {}
    parts = re.split(r"^##\s+([A-Z_][A-Z0-9_]*)\s*\n", body, flags=re.MULTILINE)
    # parts looks like: [preamble, name1, body1, name2, body2, ...]
    for i in range(1, len(parts), 2):
        name = parts[i].strip()
        text = parts[i + 1].strip("\n")
        sections[name] = text
    return sections


def _read_current() -> dict[str, str]:
    """Read data/prompts/current.json and return {provider: version}."""
    with open(CURRENT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_current(mapping: dict[str, str]) -> None:
    """Persist the provider→version pointer."""
    note = "Pointers to the active prompt version per provider. Updated by the CE Feedback page."
    payload = {**{k: v for k, v in mapping.items() if not k.startswith("_")}, "_note": note}
    tmp = CURRENT_FILE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, CURRENT_FILE)


def _provider_dir(provider: Provider) -> Path:
    return PROMPTS_DIR / provider


def _version_path(provider: Provider, version: str) -> Path:
    return _provider_dir(provider) / f"{version}.md"


def _load_sections(provider: Provider, version: str) -> dict[str, str]:
    key = (provider, version)
    with _cache_lock:
        cached = _section_cache.get(key)
    if cached is not None:
        return cached
    path = _version_path(provider, version)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file missing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        sections = _parse_sections(f.read())
    with _cache_lock:
        _section_cache[key] = sections
    return sections


def list_versions(provider: Provider) -> list[str]:
    """Return version ids found on disk for a provider, sorted naturally (v1, v2, ... v10)."""
    pdir = _provider_dir(provider)
    if not pdir.exists():
        return []
    versions = []
    for p in pdir.iterdir():
        if p.suffix == ".md" and p.stem.startswith("v"):
            versions.append(p.stem)

    def sort_key(v: str) -> tuple[int, str]:
        m = re.match(r"v(\d+)$", v)
        return (int(m.group(1)) if m else 9999, v)

    return sorted(versions, key=sort_key)


def current_version(provider: Provider) -> str:
    return _read_current().get(provider, "v1")


def set_current_version(provider: Provider, version: str) -> None:
    if not _version_path(provider, version).exists():
        raise FileNotFoundError(f"Cannot activate {provider} {version}: file does not exist")
    mapping = _read_current()
    mapping[provider] = version
    _write_current(mapping)


def next_version_id(provider: Provider) -> str:
    versions = list_versions(provider)
    nums = [int(re.match(r"v(\d+)$", v).group(1)) for v in versions if re.match(r"v(\d+)$", v)]
    return f"v{(max(nums) + 1) if nums else 1}"


def read_raw(provider: Provider, version: str | None = None) -> str:
    """Return the raw (unassembled, no whitelist injected) markdown file content."""
    v = version or current_version(provider)
    path = _version_path(provider, v)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_version(provider: Provider, version: str, content: str) -> Path:
    """Write a new version file. Refuses to overwrite existing files."""
    path = _version_path(provider, version)
    if path.exists():
        raise FileExistsError(f"Version already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    # invalidate cache for this version (defensive — should not be cached yet)
    with _cache_lock:
        _section_cache.pop((provider, version), None)
    return path


def get_sections(provider: Provider, version: str | None = None) -> dict[str, str]:
    """Return a copy of the parsed sections for previewing in the UI."""
    v = version or current_version(provider)
    return dict(_load_sections(provider, v))


def assemble(
    provider: Provider,
    kind: Kind,
    whitelist_block: str,
    version: str | None = None,
) -> str:
    """Build the full system prompt for the requested call.

    Composition:
      first_pass = FIRST_PASS_HEADER + PREFIX_GUIDE + CE_EXAMPLES + FIRST_PASS_RULES
      fallback   = FALLBACK_HEADER   + PREFIX_GUIDE + CE_EXAMPLES + FALLBACK_RULES
    """
    v = version or current_version(provider)
    sections = _load_sections(provider, v)
    if kind == "first_pass":
        order = ["FIRST_PASS_HEADER", "PREFIX_GUIDE", "CE_EXAMPLES", "FIRST_PASS_RULES"]
    else:
        order = ["FALLBACK_HEADER", "PREFIX_GUIDE", "CE_EXAMPLES", "FALLBACK_RULES"]
    parts = []
    for name in order:
        if name not in sections:
            raise KeyError(f"Prompt {provider} {v} missing required section: {name}")
        parts.append(sections[name])
    assembled = "\n\n".join(parts)
    return assembled.replace(WHITELIST_PLACEHOLDER, whitelist_block)
