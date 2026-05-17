# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# ATJ-Component MATERIAL_CATEGORY Auto-Categorization System

## Project Overview

FastAPI server that auto-suggests `MATERIAL_CATEGORY` for Advantech PLM ATJ-Component parts using fuzzy MPN matching + Azure OpenAI GPT (or Google Gemini) + category vector DB fallback. Produces review-ready Excel for CE engineers before PLM ECO submission. Includes a CE feedback workflow for continuous prompt improvement.

## Quick Start

```bash
cd atj-component-material-category
.venv\Scripts\activate          # Windows venv
python main.py                  # -> http://localhost:8000
```

### Prerequisites

- **`.env`** must have valid Azure OpenAI credentials and Denodo credentials (see `.env` section below)
- **Denodo REST API** (primary data source) must be reachable at `dataplatform.advantech.com.tw:9443` (production)
- **Power BI Desktop** with `2026_plm_alparts.pbix` loaded (fallback only, needed when Denodo is down)
- **Google Vertex AI** service account JSON (optional, only required for Gemini provider)

## Project Structure

```
atj-component-material-category/
├── main.py                     # FastAPI entry point (uvicorn, port 8000)
├── config.py                   # Pydantic settings -- reads .env with override=True
├── requirements.txt            # Python dependencies
├── .env / .env.example         # Environment config
├── HANDOFF.md                  # Operational handoff notes
├── Phase_I_PUR_2024_2025.xlsx  # Phase I item list (7,968 items, sheet "Phase_I_List")
├── run_phase1_all.ps1          # PowerShell script to run full Phase I batch
├── run_medium_rerun.py         # Re-run medium-confidence items in Phase I Excel
├── run_blank_rerun.py          # Re-run blank-confidence items (skips 142* materials)
├── fix_v3_from_logfile.py      # Repair invalid category codes from PLM LogFile.xml
├── api/
│   └── routes.py               # REST endpoints + page routes (732 lines)
├── core/
│   ├── denodo_client.py        # Denodo REST API client (httpx, pagination, auth)
│   ├── data_fetcher.py         # Facade: Denodo-first, PBI fallback
│   ├── pbi_fetcher.py          # DAX queries via pyadomd (fallback data source)
│   ├── fuzzy_matcher.py        # rapidfuzz MPN scoring (composite 3-metric)
│   ├── gpt_caller.py           # Azure OpenAI (2-pass: fuzzy + vector)
│   ├── gemini_caller.py        # Google Vertex AI Gemini (mirror of gpt_caller)
│   ├── llm_router.py           # Provider dispatcher: routes to azure or gemini
│   ├── prompt_loader.py        # Versioned prompt management (load/assemble/deploy)
│   ├── feedback_engine.py      # CE feedback workflow (XLSX → proposal → test → deploy)
│   ├── category_vector_db.py   # Category embeddings via Azure OpenAI text-embedding-3-small
│   ├── kpi_tracker.py          # KPI snapshots: All ATJ + Phase I weekly tracking
│   └── pipeline.py             # Batch + lookup orchestrators (async, 2-pass LLM)
├── export/
│   └── excel_exporter.py       # openpyxl 3-sheet formatted Excel output
├── templates/
│   ├── index.html              # Batch process page (light/dark theme)
│   ├── lookup.html             # Manual lookup page with sorting/filtering
│   ├── kpi.html                # KPI dashboard (All ATJ + Phase I tracking)
│   └── feedback.html           # CE feedback prompt editor UI
├── tests/
│   ├── test_items.json                 # 4 standard test items for algorithm validation
│   ├── test_llm_connections.py         # LLM connection health check script
│   ├── test_gemini_categorize.py       # Single Gemini end-to-end test (JSON + whitelist)
│   ├── run_test_batch.py               # Smoke test script
│   ├── run_ce_verification.py          # Runs CE-validated correction rule suite
│   ├── run_phase1_ce_rerun.py          # Phase I CE correction re-run script
│   └── ce_correction_verification.json # Expected categories for CE correction cases
└── data/
    ├── cache/                  # Parquet caches + vector DB files
    │   ├── atj_targets.parquet
    │   ├── atj_refpool.parquet
    │   ├── distinct_categories.parquet
    │   ├── category_vectors.npz
    │   ├── category_vectors_meta.pkl
    │   ├── kpi_all_atj_state.parquet   # All ATJ current state snapshot
    │   └── kpi_phase1_state.parquet    # Phase I current state snapshot
    ├── kpi/                    # KPI snapshot history (JSON)
    │   ├── snapshots.json      # All ATJ weekly snapshots
    │   └── phase1_snapshots.json # Phase I weekly snapshots
    ├── prompts/                # Versioned LLM prompt files
    │   ├── current.json        # Active version pointer: {"azure": "v2", "gemini": "v1"}
    │   ├── azure/
    │   │   ├── v1.md           # Initial Azure prompt
    │   │   └── v2.md           # Enhanced Azure prompt (current)
    │   └── gemini/
    │       └── v1.md           # Initial Gemini prompt
    ├── batches/                # (reserved for future)
    └── results/                # Excel exports
```

## Architecture: Data Access

### Denodo REST API (primary)

Two Denodo web services provide the data:
- **`iv_plm_allparts_info_latest`** (exposed by `ws_plm_allparts_info_latest_ce_app`) -- item info, lifecycle, material category, category codes
- **`iv_plm_zagile_manufacture`** -- manufacturer names and MPN (part numbers)

`core/denodo_client.py` handles HTTP Basic auth, auto-pagination (`$start_index` + `$count`), and column normalization (`ITEM_NUMBER` -> `Item_Number`).

### Power BI Desktop (fallback)

`core/pbi_fetcher.py` uses pyadomd/ADOMD to query the same data via DAX from `2026_plm_alparts.pbix`. Only used when Denodo is unreachable. Windows-only dependency.

### Facade Pattern

`core/data_fetcher.py` exposes the same public functions. Each tries Denodo first; on failure, falls back to PBI with a warning log. All other modules import from `data_fetcher`, never directly from `pbi_fetcher` or `denodo_client`.

Key data fetcher functions:
- `refresh_target_cache()` -- re-query & save targets to parquet
- `fetch_atj_target_batch()` -- blank MATERIAL_CATEGORY items (paginated)
- `fetch_atj_reference_pool()` -- items with categories for fuzzy matching
- `fetch_distinct_categories()` -- all unique category pairs (for vector DB)
- `fetch_manufacture_for_items()` -- MPN data (tries refpool cache → Denodo → PBI)
- `fetch_all_atj_components()` -- ALL items (filled + blank) for KPI tracking
- `fetch_manufacture_bulk()` -- batch MPN queries via IN-clause (KPI)
- `check_denodo_health()` -- diagnostics endpoint

## Architecture: Multi-Provider LLM

### Provider Abstraction (`core/llm_router.py`)

All LLM calls go through `llm_router.py`, which dispatches to the selected provider at request time:

```
pipeline.py → llm_router.suggest_category()
                 ├─→ gpt_caller.py      (provider="azure")
                 └─→ gemini_caller.py   (provider="gemini")
```

- The `llm_provider` field on batch/lookup requests overrides the default (`LLM_PROVIDER` in `.env`)
- `GET /api/llm/status` does a parallel health check of both providers and reports availability
- Both providers share the same user prompt builders and `_clean_gpt_result()` logic

### Azure OpenAI (`core/gpt_caller.py`)

- Uses `AsyncAzureOpenAI` with `max_completion_tokens` (required for GPT-5.4+)
- Prompts loaded dynamically from `data/prompts/azure/{version}.md`

### Google Vertex AI Gemini (`core/gemini_caller.py`)

- Uses `google-genai` SDK in Vertex AI mode
- Service-account JSON authentication (`GOOGLE_APPLICATION_CREDENTIALS`)
- Thinking strategy: disabled on Flash models (faster), enabled on Pro models
- 4096 max output tokens
- Prompts loaded from `data/prompts/gemini/{version}.md`

## Architecture: 2-Pass LLM Categorization

1. **Pass 1**: Fuzzy match target MPN against ATJ reference pool -> top-5 similar items -> LLM suggests category
2. **Pass 2 (fallback)**: If Pass 1 returns low/error confidence OR zero fuzzy matches -> search category vector DB with `Item_Desc + AI_reason` -> get top-K candidates -> LLM picks best match from candidates

The vector DB contains ~983 distinct `ZZMCATG_M|ZZMCATG_S` pairs embedded with Azure OpenAI `text-embedding-3-small`. Top-10 candidates generally outperform top-5.

## Architecture: Prompt Versioning (`core/prompt_loader.py`)

Prompts are stored as versioned markdown files with named sections. New versions can be deployed without restarting the server.

### Prompt File Structure

Each `data/prompts/{provider}/vN.md` file uses `## SECTION_NAME` delimiters:

- **FIRST_PASS_HEADER** -- instructions for the fuzzy-match pass
- **FALLBACK_HEADER** -- instructions for the vector-fallback pass
- **PREFIX_GUIDE** -- item description prefix interpretation rules
- **CE_EXAMPLES** -- curated disambiguation examples (append-only via feedback engine)
- **FIRST_PASS_RULES** -- categorization rules
- **FALLBACK_RULES** -- fallback selection rules

`data/prompts/current.json` is the active version pointer:
```json
{"azure": "v2", "gemini": "v1"}
```

### Prompt Loader Key Functions

- `assemble(provider, version)` -- compose full system prompt with whitelist injection; cached per (provider, version)
- `list_versions(provider)` -- discover vN.md files on disk
- `current_version(provider)` -- read active version from current.json
- `set_current_version(provider, version)` -- update active version pointer
- `write_version(provider, version_id, content)` -- persist new version file
- `next_version_id(provider)` -- increment version number

## Architecture: CE Feedback Workflow (`core/feedback_engine.py`)

Provides a structured loop for CE engineers to submit corrections and deploy improved prompts.

### Workflow

```
1. CE uploads correction XLSX (Material, AI_MATERIAL_CATEGORY, CE_MATERIAL_CATEGORY)
2. feedback_engine.parse_feedback_xlsx() parses & groups mismatches by pattern
3. build_proposal() renders next-version vN+1.md with new CE_EXAMPLES appendix
4. POST /api/feedback/test/{id} runs proposal against uploaded items + regression set
5. POST /api/feedback/deploy/{id} writes vN+1.md and bumps current.json
```

### Key Functions

- `parse_feedback_xlsx(path)` -- parse uploaded CE corrections
- `_group_mismatches()` -- group by (item prefix, AI→CE) to identify patterns
- `_build_appendix()` -- emit CE-validated correction examples for prompt
- `_splice_into_section()` -- insert appendix into CE_EXAMPLES section
- `build_proposal(provider, feedback_path)` -- end-to-end: parse + extract + render next-version markdown
- `deploy_proposal(proposal_id)` -- write versioned file, bump current.json

## Architecture: LLM Prompt Hardening

Two layers of guardrails sit on top of the 2-pass flow (in both `gpt_caller.py` and `gemini_caller.py`):

- **MATERIAL_CATEGORY whitelist** -- both passes constrain the LLM to return only category codes that exist in the reference pool / vector-DB candidate list. After parsing the LLM response, `_clean_category_code()` strips any descriptive suffix (e.g. `"DAC (DATA CONVERTER)"` -> `"DAC"`) and the result is re-validated against the allowed set. Off-list outputs force a fallback / low confidence.
- **CE-validated correction rules** -- a set of hand-curated disambiguation rules (e.g. CPLD vs. FPGA, DAC vs. ADC, connector vs. cable subfamilies) is injected into the prompt as a prefix guide. The regression harness `tests/run_ce_verification.py` replays the `tests/ce_correction_verification.json` cases against the live server to confirm these corrections still hold after prompt or pipeline edits.

## Key Design Decisions

- **Denodo-first, PBI-fallback** -- Denodo REST API is the primary data source; PBI Desktop is kept as fallback
- **openai SDK** (not raw httpx) -- uses `AsyncAzureOpenAI` with `max_completion_tokens` (GPT-5.4 requirement)
- **dotenv override=True** -- `.env` values take precedence over system environment variables
- **Parquet caching** -- full target list (~80K items), reference pool, and category vectors cached to `data/cache/`
- **2-pass LLM** -- vector DB fallback dramatically improves accuracy for items with weak fuzzy matches
- **LLM output cleanup** -- `_clean_category_code()` strips descriptive names LLMs sometimes append
- **Whitelist enforcement** -- LLM output is hard-checked against the reference/vector candidate set; off-list predictions are rejected
- **Python-side join** for reference pool -- avoids DAX `NATURALLEFTOUTERJOIN` lineage conflicts
- **Provider abstraction at request time** -- `llm_router.py` dispatches per request; Azure and Gemini can be compared side-by-side
- **Dynamic prompt loading** -- new prompt versions deploy without server restart; `current.json` is the live pointer
- **CE feedback loop** -- structured XLSX → proposal → test → deploy workflow; `feedback_engine.py` handles the full round-trip
- **Async concurrency** -- `asyncio.Semaphore(10)` bounds concurrent LLM calls in `pipeline.py`

## API Endpoints

### Pages

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Batch process page |
| `GET`  | `/lookup` | Manual lookup page |
| `GET`  | `/kpi` | KPI dashboard |
| `GET`  | `/feedback` | CE feedback prompt editor |

### Batch & Lookup

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/batch/run` | Start batch job (`offset`, `limit`, `lifecycle_filter`, `llm_provider`, `vector_top_k`) |
| `POST` | `/api/lookup/run` | Start lookup job (user-provided item list, `llm_provider`) |
| `GET`  | `/api/batch/{id}/status` | Poll job status (includes results JSON) |
| `GET`  | `/api/batch/{id}/export` | Download Excel (batch jobs) |
| `POST` | `/api/batch/{id}/export-selected` | Export only selected items to Excel |
| `GET`  | `/api/batches` | List all jobs |

### Cache & Vector DB

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/cache/refresh` | Rebuild Parquet cache |
| `GET`  | `/api/cache/status` | Cache info |
| `POST` | `/api/vector-db/build` | Build/refresh category vector DB |
| `GET`  | `/api/vector-db/status` | Vector DB info |

### KPI

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/kpi/snapshot` | Take All ATJ KPI snapshot |
| `GET`  | `/api/kpi/snapshots` | All ATJ historical snapshots |
| `GET`  | `/api/kpi/latest` | Latest All ATJ snapshot |
| `GET`  | `/api/kpi/details` | Per-item detail (filter, lifecycle) |
| `POST` | `/api/kpi/phase1/snapshot` | Take Phase I KPI snapshot |
| `GET`  | `/api/kpi/phase1/snapshots` | Phase I historical snapshots |
| `GET`  | `/api/kpi/phase1/latest` | Latest Phase I snapshot |
| `GET`  | `/api/kpi/phase1/details` | Phase I per-item detail |
| `GET`  | `/api/kpi/phase1/status` | Phase I item count + Excel path |
| `POST` | `/api/phase1/batch/run` | Run AI categorization on Phase I batch |

### CE Feedback & Prompt Management

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/feedback/upload` | Upload correction XLSX, build proposal |
| `GET`  | `/api/feedback/proposal/{id}` | Retrieve proposal (pattern groups, summary) |
| `GET`  | `/api/feedback/proposal/{id}/markdown` | Full proposed vN+1.md content |
| `POST` | `/api/feedback/test/{id}` | Test proposal against uploaded items + regression set |
| `POST` | `/api/feedback/deploy/{id}` | Activate new prompt version |
| `GET`  | `/api/feedback/proposals` | List recent proposals |
| `GET`  | `/api/prompts/{provider}` | Sectioned prompt preview (current version) |
| `GET`  | `/api/prompts/{provider}/versions` | List available versions |
| `POST` | `/api/prompts/{provider}/rollback` | Activate an older version |

### System

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/llm/status` | Per-provider LLM availability (parallel health check) |
| `GET`  | `/api/denodo/status` | Denodo connectivity check |
| `GET`  | `/api/config` | Current settings |

## Common Tasks

### Check Denodo connectivity
```bash
curl http://localhost:8000/api/denodo/status
```

### Check LLM provider availability
```bash
curl http://localhost:8000/api/llm/status
```

### Refresh cache (uses Denodo, falls back to PBI)
```bash
curl -X POST http://localhost:8000/api/cache/refresh
```

### Build/refresh vector DB
```bash
curl -X POST http://localhost:8000/api/vector-db/build?force=true
```

### Run lookup for specific items (top-10 vector candidates, Azure)
```bash
curl -X POST http://localhost:8000/api/lookup/run \
  -H "Content-Type: application/json" \
  -d '{"item_numbers": ["14TJ2190717-5", "14TJ5620269-7"], "vector_top_k": 10, "llm_provider": "azure"}'
```

### Run lookup with Gemini
```bash
curl -X POST http://localhost:8000/api/lookup/run \
  -H "Content-Type: application/json" \
  -d '{"item_numbers": ["14TJ2190717-5"], "vector_top_k": 10, "llm_provider": "gemini"}'
```

### Run a small test batch
```bash
curl -X POST http://localhost:8000/api/batch/run \
  -H "Content-Type: application/json" \
  -d '{"offset": 0, "limit": 5, "lifecycle_filter": ["Part Number Release"], "vector_top_k": 10}'
```

### Standard test items (4 items for algorithm validation)
```bash
curl -X POST http://localhost:8000/api/lookup/run \
  -H "Content-Type: application/json" \
  -d '{"item_numbers": ["14TJ5620269-7","14TJ2612832-8","14TJ1090124-8","14TJ3256468-7"], "vector_top_k": 10}'
```

### Take KPI snapshot (All ATJ)
```bash
curl -X POST http://localhost:8000/api/kpi/snapshot
```

### Take Phase I KPI snapshot
```bash
curl -X POST http://localhost:8000/api/kpi/phase1/snapshot
```

### Run Phase I AI batch (100 items starting at offset 0)
```bash
curl -X POST http://localhost:8000/api/phase1/batch/run \
  -H "Content-Type: application/json" \
  -d '{"offset": 0, "limit": 100, "vector_top_k": 10}'
```

### Re-run AI on existing Phase I results
Two standalone scripts (require the FastAPI server running on port 8000) drive large re-runs against the Phase I Excel and write results into `*_v2` columns:

- `python run_medium_rerun.py` -- re-runs items currently at `AI_CONFIDENCE == "medium"`, 200-item batches, 30s cooldown
- `python run_blank_rerun.py` -- re-runs items at blank `AI_CONFIDENCE`, skipping Material numbers starting with `142`

Both scripts call `/api/lookup/run`, poll `/api/batch/{id}/status`, and merge results back into `Phase_I_PUR_2024_2025.xlsx`.

### Run full Phase I batch (Windows)
```powershell
.\run_phase1_all.ps1
```

### Fix invalid category codes from PLM import log
```bash
python fix_v3_from_logfile.py  # reads LogFile.xml, outputs corrected Excel
```

### Run CE correction verification suite
```bash
python tests/run_ce_verification.py
```
Replays `tests/ce_correction_verification.json` against the running server; each case asserts an expected category (regression guard for CE-validated correction rules).

### Test LLM connections
```bash
python tests/test_llm_connections.py
```

### Test Gemini categorization end-to-end
```bash
python tests/test_gemini_categorize.py
```

### View/rollback prompt versions
```bash
# List versions
curl http://localhost:8000/api/prompts/azure/versions

# View current prompt sections
curl http://localhost:8000/api/prompts/azure

# Rollback to previous version
curl -X POST http://localhost:8000/api/prompts/azure/rollback \
  -H "Content-Type: application/json" \
  -d '{"version": "v1"}'
```

## `.env` Configuration

```ini
# Denodo REST API (primary data source, production)
DENODO_BASE_URL_ALLPARTS=https://dataplatform.advantech.com.tw:9443/server/dx_ce/ws_plm_allparts_info_latest_ce_app
DENODO_BASE_URL_MANUFACTURE=https://dataplatform.advantech.com.tw:9443/server/dx_ce/iv_plm_zagile_manufacture_ce_app
DENODO_USERNAME=ce_app
DENODO_PASSWORD=<secret>
DENODO_ENABLED=true

# Power BI Desktop (fallback, Windows only)
PBI_CONNECTION_STRING=Data Source=localhost:50614;Application Name=MCP-PBIModeling

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://ce-specbook-openai.openai.azure.com/
AZURE_OPENAI_API_KEY=<secret>
AZURE_OPENAI_DEPLOYMENT=gpt-5.4
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Azure OpenAI Embedding
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_EMBEDDING_API_VERSION=2024-02-01

# Google Vertex AI (optional, for Gemini provider)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GOOGLE_CLOUD_PROJECT=my-gcp-project
GOOGLE_CLOUD_LOCATION=us-central1
GEMINI_MODEL=gemini-2.0-flash

# LLM Provider selection
LLM_PROVIDER=azure           # default provider: azure | gemini

# Fuzzy matching
TOP_K_SIMILAR=5
MIN_SIMILARITY_SCORE=40

# Batch settings
BATCH_SIZE_DEV=100
BATCH_SIZE_PROD=1000
ENV=dev
```

## UI Features

All four pages share a consistent dark/light theme with `data-theme` attribute on `<html>` and CSS custom properties. Theme preference persists via `localStorage`.

### Light/Dark Mode Toggle
- Pill-style toggle button in the header on all pages
- Dark theme (default): dark backgrounds, light text
- Light theme: white backgrounds, dark text
- Chart.js charts re-render with theme-aware colors on toggle

### Navigation
Tab bar on all pages with links to: Batch Process (`/`), Manual Lookup (`/lookup`), KPI Dashboard (`/kpi`), CE Feedback (`/feedback`)

### Manual Lookup Page Enhancements
- **LLM provider selector**: Choose Azure or Gemini per request
- **Column sorting**: Click any column header to sort ascending/descending with visual arrows
- **Filtering**: Dropdown filters for AI Category and Confidence columns with "Showing X / Y" count
- **AI Reason column**: Wider display (min-width: 400px, max-width: 525px) with 180-char truncation

### CE Feedback Page (`/feedback`)
- Upload XLSX with columns: Material, AI_MATERIAL_CATEGORY, CE_MATERIAL_CATEGORY
- Preview pattern groups (item prefix, AI→CE correction, count)
- View proposed vN+1.md markdown (sectioned)
- Test proposal accuracy before deploying
- Deploy button activates new version without restart
- Proposal history list

## KPI Dashboard

### Purpose
Track ATJ component MATERIAL_CATEGORY completion progress. Goal: fill all ~80K ATJ components with correct MATERIAL_CATEGORY in Denodo.

### Two KPI Scopes

**All ATJ Components** (~80K items):
- Queries ALL ATJ items from Denodo (both filled and blank MATERIAL_CATEGORY)
- Weekly snapshots stored in `data/kpi/snapshots.json` (one per ISO week, overwrites same week)
- Parquet state cache: `data/cache/kpi_all_atj_state.parquet`
- Metrics: total, filled, blank, completion %, lifecycle breakdown

**Phase I Items** (7,968 items from `Phase_I_PUR_2024_2025.xlsx`):
- Subset of ATJ items prioritized for Phase I completion
- Separate snapshots in `data/kpi/phase1_snapshots.json`
- Parquet state cache: `data/cache/kpi_phase1_state.parquet`
- Extra metrics: total_excel, found_in_denodo, not_found
- AI batch processing: categorize 100 items at a time, write High/Medium results back to Excel

### KPI Snapshot Schema
```json
{
  "timestamp": "2026-04-15T...",
  "week": "2026-W16",
  "total": 80883,
  "filled": 496,
  "blank": 80387,
  "completion_pct": 0.61,
  "by_lifecycle": {
    "Part Number Release": {"total": 50000, "filled": 300, "blank": 49700, "pct": 0.6}
  }
}
```

### Phase I AI Batch Process
- Reads item list from `Phase_I_PUR_2024_2025.xlsx` (sheet "Phase_I_List", column "Material")
- Runs through the same `run_lookup()` pipeline as manual lookup
- Writes High/Medium confidence results back to Excel with columns: AI_MATERIAL_CATEGORY, AI_CONFIDENCE, AI_REASON, AI_UPDATED_AT
- Batch size: 100 items per run, auto-advances offset

### KPI Module -- `core/kpi_tracker.py`
- Thread-safe JSON read/write with `threading.Lock`
- ISO week deduplication (`%G-W%V` format)
- `take_snapshot()` / `take_phase1_snapshot()`: fetch data, compute metrics, save
- `get_detail_data()` / `get_phase1_detail()`: per-item drill-down with manufacture data
- `write_phase1_results_to_excel()`: openpyxl write-back for AI results

## Known Constraints

- PBI Desktop ADOMD port changes on restart -- update `PBI_CONNECTION_STRING` in `.env`
- Job store is in-memory -- clears on server restart
- Denodo dev server uses HTTPS with self-signed cert (`verify=False` in httpx)
- Reference pool is small (~496 ATJ items) -- vector DB fallback compensates
- CPLD items may get low confidence because PLD category doesn't surface in top-10 vector results
- Denodo `iv_plm_zagile_manufacture` uses `ITEM_NUMBER` (all caps) -- normalized by `denodo_client.py`
- Gemini thinking strategy must be disabled on Flash models (API requirement) -- `gemini_caller.py` handles this automatically
- Prompt assembly is cached per (provider, version) -- call `prompt_loader.invalidate_cache()` after editing raw files if the server is running
- `fix_v3_from_logfile.py` and `run_phase1_all.ps1` are standalone tools, not wired into the web UI
