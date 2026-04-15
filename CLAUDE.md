# ATJ-Component MATERIAL_CATEGORY Auto-Categorization System

## Project Overview

FastAPI server that auto-suggests `MATERIAL_CATEGORY` for Advantech PLM ATJ-Component parts using fuzzy MPN matching + Azure OpenAI GPT + category vector DB fallback. Produces review-ready Excel for CE engineers before PLM ECO submission.

## Quick Start

```bash
cd atj-component-material-category
.venv\Scripts\activate          # Windows venv
python main.py                  # -> http://localhost:8000
```

### Prerequisites

- **`.env`** must have valid Azure OpenAI credentials and Denodo credentials (see `.env` section below)
- **Denodo REST API** (primary data source) must be reachable at `acldtpltfrm-dev:9443`
- **Power BI Desktop** with `2026_plm_alparts.pbix` loaded (fallback only, needed when Denodo is down)

## Project Structure

```
atj-component-material-category/
├── main.py                     # FastAPI entry point (uvicorn, port 8000)
├── config.py                   # Pydantic settings -- reads .env with override=True
├── requirements.txt            # Python dependencies
├── .env / .env.example         # Environment config
├── Phase_I_PUR_2024_2025.xlsx  # Phase I item list (7,968 items, sheet "Phase_I_List")
├── api/
│   └── routes.py               # REST endpoints + page routes
├── core/
│   ├── denodo_client.py        # Denodo REST API client (httpx, pagination, auth)
│   ├── data_fetcher.py         # Facade: Denodo-first, PBI fallback
│   ├── pbi_fetcher.py          # DAX queries via pyadomd (fallback data source)
│   ├── fuzzy_matcher.py        # rapidfuzz MPN scoring (composite 3-metric)
│   ├── gpt_caller.py           # Azure OpenAI via openai SDK (2-pass: fuzzy + vector)
│   ├── category_vector_db.py   # Category embeddings via Azure OpenAI text-embedding-3-small
│   ├── kpi_tracker.py          # KPI snapshots: All ATJ + Phase I weekly tracking
│   └── pipeline.py             # Batch + lookup orchestrators (async, 2-pass GPT)
├── export/
│   └── excel_exporter.py       # openpyxl 3-sheet formatted Excel output
├── templates/
│   ├── index.html              # Batch process page (light/dark theme)
│   ├── lookup.html             # Manual lookup page with sorting/filtering
│   └── kpi.html                # KPI dashboard (All ATJ + Phase I tracking)
├── tests/
│   ├── test_items.json         # 4 standard test items for algorithm validation
│   └── run_test_batch.py       # Smoke test script
└── data/
    ├── cache/                  # Parquet caches + vector DB files
    │   ├── atj_targets.parquet
    │   ├── atj_refpool.parquet
    │   ├── distinct_categories.parquet
    │   ├── category_vectors.npz
    │   └── category_vectors_meta.pkl
    ├── kpi/                    # KPI snapshot history (JSON)
    │   ├── snapshots.json      # All ATJ weekly snapshots
    │   └── phase1_snapshots.json # Phase I weekly snapshots
    ├── batches/                # (reserved for future)
    └── results/                # Excel exports
```

## Architecture: Data Access

### Denodo REST API (primary)

Two Denodo web services provide the data:
- **`iv_allparts_info_for_ce`** -- item info, lifecycle, material category, category codes
- **`iv_plm_zagile_manufacture`** -- manufacturer names and MPN (part numbers)

`core/denodo_client.py` handles HTTP Basic auth, auto-pagination (`$start_index` + `$count`), and column normalization (`ITEM_NUMBER` -> `Item_Number`).

### Power BI Desktop (fallback)

`core/pbi_fetcher.py` uses pyadomd/ADOMD to query the same data via DAX from `2026_plm_alparts.pbix`. Only used when Denodo is unreachable.

### Facade Pattern

`core/data_fetcher.py` exposes the same public functions. Each tries Denodo first; on failure, falls back to PBI with a warning log. All other modules import from `data_fetcher`, never directly from `pbi_fetcher` or `denodo_client`.

Additional functions for KPI:
- `fetch_all_atj_components()` -- queries ALL ATJ items (filled + blank MATERIAL_CATEGORY) for KPI tracking
- `fetch_manufacture_for_items_batched(item_numbers, batch_size=500)` -- batched manufacture queries for large item lists

## Architecture: 2-Pass GPT Categorization

1. **Pass 1**: Fuzzy match target MPN against ATJ reference pool -> top-5 similar items -> GPT suggests category
2. **Pass 2 (fallback)**: If Pass 1 returns low/error confidence OR zero fuzzy matches -> search category vector DB with `Item_Desc + AI_reason` -> get top-K candidates -> GPT picks best match from candidates

The vector DB contains ~983 distinct `ZZMCATG_M|ZZMCATG_S` pairs embedded with Azure OpenAI `text-embedding-3-small`. Top-10 candidates generally outperform top-5.

## Key Design Decisions

- **Denodo-first, PBI-fallback** -- Denodo REST API is the primary data source; PBI Desktop is kept as fallback
- **openai SDK** (not raw httpx) -- uses `AsyncAzureOpenAI` with `max_completion_tokens` (GPT-5.4 requirement)
- **dotenv override=True** -- `.env` values take precedence over system environment variables
- **Parquet caching** -- full target list (~80K items), reference pool, and category vectors cached to `data/cache/`
- **2-pass GPT** -- vector DB fallback dramatically improves accuracy for items with weak fuzzy matches
- **GPT output cleanup** -- `_clean_category_code()` strips descriptive names GPT sometimes appends (e.g. `"DAC (DATA CONVERTER)"` -> `"DAC"`)
- **Python-side join** for reference pool -- avoids DAX `NATURALLEFTOUTERJOIN` lineage conflicts

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Batch process page |
| `GET`  | `/lookup` | Manual lookup page with results preview |
| `GET`  | `/kpi` | KPI dashboard (All ATJ + Phase I) |
| `POST` | `/api/batch/run` | Start batch job (supports `vector_top_k`) |
| `POST` | `/api/lookup/run` | Start lookup job (user-provided item list) |
| `GET`  | `/api/batch/{id}/status` | Poll job status (includes results JSON) |
| `GET`  | `/api/batch/{id}/export` | Download Excel (batch jobs) |
| `POST` | `/api/batch/{id}/export-selected` | Export only selected items to Excel |
| `GET`  | `/api/batches` | List all jobs |
| `POST` | `/api/cache/refresh` | Rebuild Parquet cache |
| `GET`  | `/api/cache/status` | Cache info |
| `POST` | `/api/vector-db/build` | Build/refresh category vector DB |
| `GET`  | `/api/vector-db/status` | Vector DB info |
| `GET`  | `/api/denodo/status` | Denodo connectivity check |
| `GET`  | `/api/config` | Current settings |
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

## Common Tasks

### Check Denodo connectivity
```bash
curl http://localhost:8000/api/denodo/status
```

### Refresh cache (uses Denodo, falls back to PBI)
```bash
curl -X POST http://localhost:8000/api/cache/refresh
```

### Build/refresh vector DB
```bash
curl -X POST http://localhost:8000/api/vector-db/build?force=true
```

### Run lookup for specific items (top-10 vector candidates)
```bash
curl -X POST http://localhost:8000/api/lookup/run \
  -H "Content-Type: application/json" \
  -d '{"item_numbers": ["14TJ2190717-5", "14TJ5620269-7"], "vector_top_k": 10}'
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

## `.env` Configuration

```ini
# Denodo REST API (primary data source)
DENODO_BASE_URL_ALLPARTS=https://acldtpltfrm-dev:9443/server/dx_ce/ws_allparts_info_for_ce_app
DENODO_BASE_URL_MANUFACTURE=https://acldtpltfrm-dev:9443/server/dx_ce/iv_plm_zagile_manufacture_ce_app
DENODO_USERNAME=ce_app
DENODO_PASSWORD=<secret>
DENODO_ENABLED=true

# Power BI Desktop (fallback)
PBI_CONNECTION_STRING=Data Source=localhost:50614;Application Name=MCP-PBIModeling

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://ce-specbook-openai.openai.azure.com/
AZURE_OPENAI_API_KEY=<secret>
AZURE_OPENAI_DEPLOYMENT=gpt-5.4
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Azure OpenAI Embedding
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_EMBEDDING_API_VERSION=2024-02-01

ENV=dev
```

## UI Features

All three pages share a consistent dark/light theme with `data-theme` attribute on `<html>` and CSS custom properties. Theme preference persists via `localStorage`.

### Light/Dark Mode Toggle
- Pill-style toggle button in the header on all pages
- Dark theme (default): dark backgrounds, light text
- Light theme: white backgrounds, dark text
- Chart.js charts re-render with theme-aware colors on toggle

### Manual Lookup Page Enhancements
- **Column sorting**: Click any column header to sort ascending/descending with visual arrows
- **Filtering**: Dropdown filters for AI Category and Confidence columns with "Showing X / Y" count
- **AI Reason column**: Wider display (min-width: 400px, max-width: 525px) with 180-char truncation

### Navigation
Tab bar on all pages with links to: Batch Process (`/`), Manual Lookup (`/lookup`), KPI Dashboard (`/kpi`)

## KPI Dashboard

### Purpose
Track ATJ component MATERIAL_CATEGORY completion progress. Goal: fill all ~80K ATJ components with correct MATERIAL_CATEGORY in Denodo.

### Two KPI Scopes

**All ATJ Components** (~80K items):
- Queries ALL ATJ items from Denodo (both filled and blank MATERIAL_CATEGORY)
- Weekly snapshots stored in `data/kpi/snapshots.json` (one per ISO week, overwrites same week)
- Metrics: total, filled, blank, completion %, lifecycle breakdown

**Phase I Items** (7,968 items from `Phase_I_PUR_2024_2025.xlsx`):
- Subset of ATJ items prioritized for Phase I completion
- Separate snapshots in `data/kpi/phase1_snapshots.json`
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
