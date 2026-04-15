# ATJ-Component MATERIAL_CATEGORY Auto-Categorization System

FastAPI server that auto-suggests `MATERIAL_CATEGORY` for Advantech PLM ATJ-Component parts using fuzzy MPN matching + Azure OpenAI GPT + category vector DB fallback. Includes a KPI dashboard to track completion progress.

## Features

- **Batch Process** -- bulk AI categorization with lifecycle phase filtering
- **Manual Lookup** -- paste item numbers for instant AI suggestions with preview table
- **KPI Dashboard** -- weekly snapshot tracking for All ATJ (~80K items) and Phase I (7,968 items)
- **2-Pass GPT** -- fuzzy match first, vector DB fallback for higher accuracy
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
                  [fuzzy_matcher.py]    [gpt_caller.py]
                  rapidfuzz scoring     Azure OpenAI GPT
                         │                    │
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
| Denodo REST API | accessible at `acldtpltfrm-dev:9443` | Primary data source |
| Azure OpenAI | GPT-5.4 + text-embedding-3-small | AI categorization + embeddings |
| Power BI Desktop | (optional) with `2026_plm_alparts.pbix` | Fallback data source |

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
| `AZURE_OPENAI_ENDPOINT` | Yes | Azure OpenAI resource URL |
| `AZURE_OPENAI_API_KEY` | Yes | Azure OpenAI API key |
| `AZURE_OPENAI_DEPLOYMENT` | Yes | GPT deployment name (e.g. `gpt-5.4`) |
| `DENODO_ENABLED` | Yes | `true` to use Denodo as primary data source |
| `PBI_CONNECTION_STRING` | No | Only needed if using PBI Desktop as fallback |
| `ENV` | No | `dev` (100 items/batch) or `prod` (1000 items/batch) |

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

```bash
# Run batch (5 items, Part Number Release phase, top-10 vector candidates)
curl -X POST http://localhost:8000/api/batch/run \
  -H "Content-Type: application/json" \
  -d '{"offset": 0, "limit": 5, "lifecycle_filter": ["Part Number Release"], "vector_top_k": 10}'

# Lookup specific items
curl -X POST http://localhost:8000/api/lookup/run \
  -H "Content-Type: application/json" \
  -d '{"item_numbers": ["14TJ2190717-5", "14TJ5620269-7"], "vector_top_k": 10}'

# Take KPI snapshot (All ATJ)
curl -X POST http://localhost:8000/api/kpi/snapshot

# Take Phase I KPI snapshot
curl -X POST http://localhost:8000/api/kpi/phase1/snapshot

# Run Phase I AI batch (100 items)
curl -X POST http://localhost:8000/api/phase1/batch/run \
  -H "Content-Type: application/json" \
  -d '{"offset": 0, "limit": 100, "vector_top_k": 10}'
```

### First-Run Checklist

- [ ] `.env` configured with Denodo + Azure OpenAI credentials
- [ ] Server starts without errors (`python main.py`)
- [ ] Denodo connectivity OK (`GET /api/denodo/status`)
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
│   ├── category_vector_db.py   # Category vector embeddings
│   ├── kpi_tracker.py          # Weekly KPI snapshots
│   └── pipeline.py             # Batch + lookup orchestrators
├── export/
│   └── excel_exporter.py       # openpyxl 3-sheet Excel output
├── templates/
│   ├── index.html              # Batch process page
│   ├── lookup.html             # Manual lookup page
│   └── kpi.html                # KPI dashboard
├── tests/
│   ├── test_items.json         # 4 test items for validation
│   └── run_test_batch.py       # Smoke test script
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

## Known Constraints

- **Job store is in-memory** -- clears on server restart
- **PBI Desktop port is ephemeral** -- update `PBI_CONNECTION_STRING` if PBI restarts
- **Denodo uses self-signed cert** -- `verify=False` in httpx (acceptable for dev)
- **Phase I Excel concurrent access** -- close Excel before running AI batch write-back
- **Reference pool is small** (~496 ATJ items) -- vector DB fallback compensates
