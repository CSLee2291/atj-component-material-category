# ATJ-Component MATERIAL_CATEGORY Auto-Categorization System

FastAPI server that auto-suggests `MATERIAL_CATEGORY` for Advantech PLM ATJ-Component parts using fuzzy MPN matching + an LLM (Azure OpenAI **or** Google Vertex AI Gemini) + category vector DB fallback. Includes a KPI dashboard to track completion progress.

## Features

- **Batch Process** -- bulk AI categorization with lifecycle phase filtering
- **Manual Lookup** -- paste item numbers for instant AI suggestions with preview table
- **KPI Dashboard** -- weekly snapshot tracking for All ATJ (~80K items) and Phase I (7,968 items)
- **2-Pass LLM** -- fuzzy match first, vector DB fallback for higher accuracy
- **LLM provider selector** -- choose Azure OpenAI (`gpt-5.4`) or Google Gemini (`gemini-2.5-flash` via Vertex AI) per run; the UI auto-detects which providers are reachable and only enables the working ones
- **Light/Dark Theme** -- switchable UI across all pages
- **Excel Export** -- formatted review sheets for CE engineers

## Architecture

```
[Denodo REST API]  ──primary──►  [data_fetcher.py]  ◄──fallback──  [Power BI Desktop]
                                        │
                                        ▼
                              [pipeline.py]  orchestrates:
                                   │
                         ┌─────────┴──────────┐
                         │                    │
                  [fuzzy_matcher.py]    [llm_router.py]
                  rapidfuzz scoring     dispatch by provider
                         │              ┌────┴────┐
                         │              │         │
                         │      [gpt_caller.py]  [gemini_caller.py]
                         │      Azure OpenAI     Vertex AI Gemini
                         │              │         │
                         │         [category_vector_db.py]
                         │          embeddings fallback
                         └─────────┬──────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │               │
            [excel_exporter.py]  [kpi_tracker.py]  [routes.py]
            Excel output         Weekly snapshots   FastAPI web UI
```

---

## Installation

### Prerequisites

| Requirement | Version | Purpose |
|---|---|---|
| Python | 3.12+ | Runtime |
| Denodo REST API | accessible at `dataplatform.advantech.com.tw:9443` (production) | Primary data source |
| Azure OpenAI | GPT-5.4 + text-embedding-3-small | LLM provider option A + embeddings |
| Google Vertex AI | Gemini 2.5 Flash, service-account JSON | LLM provider option B (optional) |
| Power BI Desktop | (optional) with `2026_plm_alparts.pbix` | Fallback data source |

At least **one** LLM provider (Azure OpenAI **or** Gemini) must be reachable. The UI selector at run time only enables providers that pass the `/api/llm/status` connection probe. Embeddings (used by the category vector DB) still come from Azure OpenAI regardless of which LLM provider is chosen.

### Step 1: Clone and create virtual environment

```bash
git clone <repository-url>
cd atj-component-material-category

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure environment

```bash
# Copy the example config
cp .env.example .env
```

Edit `.env` and fill in your credentials:

| Variable | Required | Description |
|---|---|---|
| `DENODO_USERNAME` | Yes | Denodo REST API username |
| `DENODO_PASSWORD` | Yes | Denodo REST API password |
| `DENODO_ENABLED` | Yes | `true` to use Denodo as primary data source |
| `AZURE_OPENAI_ENDPOINT` | Yes (for embeddings; required if using Azure as LLM) | Azure OpenAI resource URL |
| `AZURE_OPENAI_API_KEY` | Yes (same) | Azure OpenAI API key |
| `AZURE_OPENAI_DEPLOYMENT` | Yes (same) | GPT deployment name (e.g. `gpt-5.4`) |
| `LLM_PROVIDER` | No | Default LLM if request omits one: `azure` (default) or `gemini` |
| `GEMINI_CREDENTIALS_PATH` | Only if using Gemini | Path to Google service-account JSON. Default: `../google_api/indigo-medium-491407-j1-3b43724862db.json` |
| `GEMINI_PROJECT_ID` | Only if using Gemini | GCP project ID, e.g. `indigo-medium-491407-j1` |
| `GEMINI_LOCATION` | No | Vertex AI region (default `us-central1`) |
| `GEMINI_MODEL` | No | Gemini model id (default `gemini-2.5-flash`) |
| `PBI_CONNECTION_STRING` | No | Only needed if using PBI Desktop as fallback |
| `ENV` | No | `dev` (100 items/batch) or `prod` (1000 items/batch) |

The Google service-account JSON shipped with this workflow lives at `../google_api/indigo-medium-491407-j1-3b43724862db.json` (relative to the app folder). Authentication uses Application Default Credentials -- the Gemini caller sets `GOOGLE_APPLICATION_CREDENTIALS` for you on first call.

### Step 4: Start the server

```bash
python main.py
# -> http://localhost:8000
```

### Step 5: Build required caches (first run)

After the server starts, you **must build the caches** before using batch/lookup features. Run these commands in order:

#### 5a. Verify Denodo connectivity

```bash
curl http://localhost:8000/api/denodo/status
```

Expected response: `{"enabled": true, "status": "connected", ...}`

#### 5a-bis. Verify LLM connectivity

```bash
curl http://localhost:8000/api/llm/status
```

Expected response shape:

```json
{
  "providers": {
    "azure":  {"label": "Azure OpenAI", "model": "gpt-5.4",          "ok": true, "sample": "pong"},
    "gemini": {"label": "Google Gemini (Vertex AI)", "model": "gemini-2.5-flash", "ok": true, "sample": "pong"}
  },
  "default": "azure"
}
```

If a provider returns `"ok": false`, its option will appear disabled in the UI selector. The Run button stays disabled until at least one provider is healthy.

#### 5b. Build target cache (Parquet)

This fetches all ATJ items (~80K) from Denodo and saves to `data/cache/`:

```bash
curl -X POST http://localhost:8000/api/cache/refresh
```

This creates:
- `data/cache/atj_targets.parquet` -- all ATJ items needing categorization
- `data/cache/atj_refpool.parquet` -- ATJ reference items with existing categories
- `data/cache/atj_targets_meta.txt` -- cache metadata
- `data/cache/atj_refpool_meta.txt` -- cache metadata

#### 5c. Build category vector DB

This embeds all ~983 distinct category pairs using Azure OpenAI:

```bash
curl -X POST "http://localhost:8000/api/vector-db/build?force=true"
```

This creates:
- `data/cache/category_vectors.npz` -- numpy embedding matrix
- `data/cache/category_vectors_meta.pkl` -- category metadata
- `data/cache/category_vectors_info.txt` -- build info
- `data/cache/distinct_categories.parquet` -- raw category data

#### 5d. Verify caches

```bash
# Check target cache
curl http://localhost:8000/api/cache/status

# Check vector DB
curl http://localhost:8000/api/vector-db/status
```

### Step 6: (Optional) Phase I setup

If you have the Phase I item list (`Phase_I_PUR_2024_2025.xlsx`), place it in the project root directory. This enables:
- Phase I KPI tracking on the dashboard
- Phase I batch AI categorization

The Excel file must have a sheet named `Phase_I_List` with a `Material` column containing item numbers.

---

## Usage

### Web UI

| Page | URL | Description |
|---|---|---|
| Batch Process | `http://localhost:8000/` | Bulk AI categorization |
| Manual Lookup | `http://localhost:8000/lookup` | Paste item numbers for instant review |
| KPI Dashboard | `http://localhost:8000/kpi` | Track completion progress |

### API Quick Reference

All run endpoints accept an optional `llm_provider` field (`"azure"` or `"gemini"`). Omit it to use the server-side default (`LLM_PROVIDER` in `.env`).

```bash
# Check which LLMs are reachable right now
curl http://localhost:8000/api/llm/status

# Run batch using Azure OpenAI (5 items, Part Number Release phase)
curl -X POST http://localhost:8000/api/batch/run \
  -H "Content-Type: application/json" \
  -d '{"offset": 0, "limit": 5, "lifecycle_filter": ["Part Number Release"], "vector_top_k": 10, "llm_provider": "azure"}'

# Lookup specific items via Gemini
curl -X POST http://localhost:8000/api/lookup/run \
  -H "Content-Type: application/json" \
  -d '{"item_numbers": ["14TJ2190717-5", "14TJ5620269-7"], "vector_top_k": 10, "llm_provider": "gemini"}'

# Take KPI snapshot (All ATJ)
curl -X POST http://localhost:8000/api/kpi/snapshot

# Take Phase I KPI snapshot
curl -X POST http://localhost:8000/api/kpi/phase1/snapshot

# Run Phase I AI batch (100 items, Gemini)
curl -X POST http://localhost:8000/api/phase1/batch/run \
  -H "Content-Type: application/json" \
  -d '{"offset": 0, "limit": 100, "vector_top_k": 10, "llm_provider": "gemini"}'
```

### First-Run Checklist

- [ ] `.env` configured with Denodo + at least one LLM provider (Azure OpenAI and/or Gemini)
- [ ] (If using Gemini) service-account JSON present at `GEMINI_CREDENTIALS_PATH`
- [ ] Server starts without errors (`python main.py`)
- [ ] Denodo connectivity OK (`GET /api/denodo/status`)
- [ ] LLM connectivity OK (`GET /api/llm/status` -- at least one provider returns `ok: true`)
- [ ] Target cache built (`POST /api/cache/refresh`)
- [ ] Vector DB built (`POST /api/vector-db/build?force=true`)
- [ ] Test with small batch (`POST /api/batch/run` with `limit: 5`)
- [ ] (Optional) Place `Phase_I_PUR_2024_2025.xlsx` in project root

---

## Project Structure

```
atj-component-material-category/
├── main.py                     # FastAPI entry point (uvicorn, port 8000)
├── config.py                   # Pydantic settings (reads .env with override=True)
├── requirements.txt            # Python dependencies
├── .env.example                # Template -- copy to .env
├── .gitignore                  # Git exclusion rules
├── CLAUDE.md                   # AI assistant context
├── api/
│   └── routes.py               # REST endpoints + HTML page routes
├── core/
│   ├── denodo_client.py        # Denodo REST API client
│   ├── data_fetcher.py         # Facade: Denodo-first, PBI-fallback
│   ├── pbi_fetcher.py          # Power BI DAX queries (fallback)
│   ├── fuzzy_matcher.py        # rapidfuzz MPN scoring
│   ├── gpt_caller.py           # Azure OpenAI GPT (2-pass)
│   ├── gemini_caller.py        # Google Vertex AI Gemini (2-pass, mirrors gpt_caller)
│   ├── llm_router.py           # Dispatches per-request LLM choice + connection probe
│   ├── category_vector_db.py   # Category vector embeddings (Azure embeddings)
│   ├── kpi_tracker.py          # Weekly KPI snapshots
│   └── pipeline.py             # Batch + lookup orchestrators
├── export/
│   └── excel_exporter.py       # openpyxl 3-sheet Excel output
├── templates/
│   ├── index.html              # Batch process page
│   ├── lookup.html             # Manual lookup page
│   └── kpi.html                # KPI dashboard
├── tests/
│   ├── test_items.json             # 4 test items for validation
│   ├── run_test_batch.py           # Smoke test script
│   ├── test_llm_connections.py     # Probe Azure + Gemini, print availability
│   └── test_gemini_categorize.py   # End-to-end Gemini categorization check
└── data/                       # Runtime data (gitignored)
    ├── cache/                  # Parquet + vector DB (rebuild after install)
    ├── kpi/                    # KPI snapshot JSON files
    ├── results/                # Excel exports
    └── batches/                # (reserved)
```

---

## Cache Rebuild Reference

If caches are missing or corrupted, rebuild in this order:

| Step | Command | Creates |
|---|---|---|
| 1 | `POST /api/cache/refresh` | `atj_targets.parquet`, `atj_refpool.parquet` + metadata |
| 2 | `POST /api/vector-db/build?force=true` | `category_vectors.npz`, `.pkl`, `distinct_categories.parquet` |

Both commands take 1-3 minutes depending on network speed to Denodo.

---

## LLM Provider Selection

The Manual Lookup and Batch Process pages each show an **LLM provider** dropdown beneath the Vector DB top-K selector. On page load the UI calls `GET /api/llm/status`, which probes both providers in parallel:

- Healthy providers appear as enabled options (with their model name in parentheses).
- Unhealthy providers appear disabled with a red dot and the underlying error.
- The **Run/Material Categorize** button is disabled until at least one provider is healthy.

The selected provider is sent as `llm_provider` in the request body and threaded through `pipeline.run_batch` / `pipeline.run_lookup` / Phase 1 batch into `core.llm_router`, which dispatches both passes (fuzzy-match + vector fallback) to the chosen provider. Both providers share the same system prompt, MATERIAL_CATEGORY whitelist, and CE-validated correction rules, so output schemas match.

To probe provider health from the command line without hitting the UI:

```bash
.venv\Scripts\python.exe tests\test_llm_connections.py
```

## Known Constraints

- **Job store is in-memory** -- clears on server restart
- **PBI Desktop port is ephemeral** -- update `PBI_CONNECTION_STRING` if PBI restarts
- **Denodo uses self-signed cert** -- `verify=False` in httpx (acceptable for dev)
- **Phase I Excel concurrent access** -- close Excel before running AI batch write-back
- **Reference pool is small** (~496 ATJ items) -- vector DB fallback compensates
- **Gemini 2.5 thinking tokens** -- `gemini_caller.test_connection` sets `thinking_budget=0` so the ping returns visible text; the categorization path leaves thinking enabled (default behavior of `gemini-2.5-flash`)
- **Embeddings stay on Azure** -- vector DB build (`/api/vector-db/build`) calls `text-embedding-3-small` regardless of which LLM provider is chosen for categorization
