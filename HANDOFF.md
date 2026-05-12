# Project Handoff — ATJ-Component MATERIAL_CATEGORY Auto-Categorization

This document covers everything needed to move the project to another developer's machine and keep the weekly KPI workflow running. It is transfer-specific — day-to-day architecture notes live in `CLAUDE.md` and `README.md`.

---

## 1. What to transfer

Copy the entire `atj-component-material-category/` folder. Inside it, the following paths carry state that CANNOT be rebuilt automatically and MUST travel with the code:

| Path | What it contains | Rebuildable? |
|---|---|---|
| `.env` | Denodo + Azure OpenAI credentials (NOT in git) | No — send separately via secure channel |
| `Phase_I_PUR_2024_2025.xlsx` | Phase I source list + AI batch results columns | No — user data, keep in sync |
| `Phase_I_Correct_Material_Category.xlsx` | CE-validated corrections reference | No |
| `data/cache/kpi_all_atj_state.parquet` | Current All-ATJ state (~80K rows) after last snapshot | Yes (click *Take Snapshot*) |
| `data/cache/kpi_phase1_state.parquet` | Current Phase I state (7,968 rows) after last snapshot | Yes (click *Take Phase I Snapshot*) |
| `data/kpi/snapshots.json` | **All-ATJ weekly history** — trend chart depends on it | **No — do not delete** |
| `data/kpi/phase1_snapshots.json` | **Phase I weekly history** — trend chart depends on it | **No — do not delete** |
| `data/cache/atj_targets.parquet` | Target pool (blank-category items) used by batch pipeline | Yes (`POST /api/cache/refresh`) |
| `data/cache/atj_refpool.parquet` | ATJ reference pool for fuzzy match | Yes (`POST /api/cache/refresh?force=true`) |
| `data/cache/distinct_categories.parquet` | Distinct `ZZMCATG_M\|ZZMCATG_S` pairs | Yes |
| `data/cache/category_vectors.npz` + `.pkl` | Embedding vector DB (~983 categories) | Yes (`POST /api/vector-db/build?force=true`) — ~2 min |
| `data/results/` | Historical Excel exports | Keep if CE engineers still reference them |

**Do NOT transfer** `.venv/` or `__pycache__/` — those are environment-specific.

Everything else is in git. If the repository has already been pushed, your colleague can clone it; otherwise zip the folder (minus `.venv` and `__pycache__`).

---

## 2. Prerequisites on the new machine

| Requirement | Notes |
|---|---|
| Python 3.12+ | Install from python.org |
| Access to `dataplatform.advantech.com.tw:9443` | Must be on the Advantech network / VPN for Denodo to be reachable |
| Denodo account (`ce_app` or equivalent) | Shared via `.env`; confirm the account still has read access to the two views |
| Azure OpenAI access | Endpoint + API key from `.env` |
| (Optional) Power BI Desktop with `2026_plm_alparts.pbix` | Only needed if Denodo is down; set `PBI_ENABLED=true` in `.env` to use |

---

## 3. First-time setup

```bash
cd atj-component-material-category

python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # macOS/Linux

pip install -r requirements.txt
```

Then drop the transferred `.env` into the project root (next to `main.py`). It must contain the production Denodo URLs:

```ini
DENODO_BASE_URL_ALLPARTS=https://dataplatform.advantech.com.tw:9443/server/dx_ce/ws_plm_allparts_info_latest_ce_app
DENODO_BASE_URL_MANUFACTURE=https://dataplatform.advantech.com.tw:9443/server/dx_ce/ws_plm_zagile_manufacture_ce_app
DENODO_USERNAME=...
DENODO_PASSWORD=...
DENODO_ENABLED=true
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
...
```

See `.env.example` for the complete template.

---

## 4. Start the app & verify

```bash
.venv\Scripts\activate
python main.py
```

Open http://localhost:8000 and in a second terminal run the three sanity checks:

```bash
# 1. Denodo reachable? expect status=connected
curl http://localhost:8000/api/denodo/status

# 2. Known item returns PWR|REFE (smoke test against the new view)
curl -X POST http://localhost:8000/api/lookup/run \
  -H "Content-Type: application/json" \
  -d '{"item_numbers":["14TJ3092921-1"],"vector_top_k":10}'

# 3. KPI dashboard loads with history intact
# Visit http://localhost:8000/kpi — Weekly Completion Trend chart should show
# the transferred snapshot history.
```

If status comes back `unreachable`, 99% of the time it is VPN / network; Denodo works fine on the dev machine today.

---

## 5. The weekly KPI workflow (what the colleague owns going forward)

Every Monday (or whenever the KPI should advance a week):

1. Open http://localhost:8000/kpi
2. Click **Take Snapshot** in the *All ATJ Components* card.
   - Pulls ~80K allparts rows + bulk manufacture from Denodo
   - Rewrites `data/cache/kpi_all_atj_state.parquet`
   - Appends / overwrites the current ISO-week row in `data/kpi/snapshots.json`
3. Click **Take Phase I Snapshot** in the *Phase I Progress* card.
   - Same, for the 7,968 Phase I items
   - Rewrites `data/cache/kpi_phase1_state.parquet`
   - Updates `data/kpi/phase1_snapshots.json`
4. Scroll to the detail cards and confirm the trend charts extended by one week.

The detail tables (*Detail Data*, *Phase I Details*) read from the parquet state caches — every Refresh and every Item_Number search runs fully from disk, so they are fast and safe to click as often as you want. They only reflect fresh Denodo data after a snapshot.

**Taking a snapshot twice in the same ISO week is safe** — the current week's row is overwritten, never duplicated.

---

## 6. Gotchas worth knowing

- **Denodo `$filter` is case-strict in the new views.** Bare identifiers get lowercased (`Item_Number` → `item_number`, which doesn't exist). Every filter in `core/data_fetcher.py` wraps columns in double quotes (e.g. `"Item_Number" = '...'`). If you add a new filter, do the same.
- **Old service name `iv_plm_zagile_manufacture_ce_app` is gone.** Production uses `ws_plm_zagile_manufacture_ce_app` (note `ws_` prefix). If you see 404s for manufacture queries, check `.env`.
- **Self-signed cert on Denodo.** `httpx` runs with `verify=False`. Do not remove.
- **Power BI Desktop fallback** is disabled by default (`PBI_ENABLED=false`). Turn it on in `.env` only if the colleague has the `.pbix` open locally; the ADOMD port in the connection string changes every time PBI restarts.
- **Jobs are in-memory.** Restarting `main.py` wipes running/queued batch lookups. Re-run them.
- **Phase I AI results** are written into `Phase_I_PUR_2024_2025.xlsx` itself (columns `AI_MATERIAL_CATEGORY`, `AI_CONFIDENCE`, `AI_REASON`, `AI_UPDATED_AT`, plus `_v2` variants from rerun scripts). The Excel file IS the source of truth for AI output — back it up before any mass rerun.
- **The `_v2` column family** is produced by `run_medium_rerun.py` and `run_blank_rerun.py`. Run these only after the FastAPI server is up on port 8000; they drive it via HTTP.

---

## 7. Useful pointers

| Topic | Where |
|---|---|
| Architecture deep-dive | `CLAUDE.md` |
| End-user README | `README.md` |
| API endpoint list | `CLAUDE.md` → *API Endpoints* |
| New view OpenAPI spec | `openapi_ws_plm_allparts_info_latest_ce_app.yaml` |
| Regression suite (CE correction rules) | `tests/run_ce_verification.py` + `tests/ce_correction_verification.json` |
| Smoke test items | `tests/test_items.json` |

---

## 8. First-day checklist for the colleague

- [ ] Clone/receive the project folder
- [ ] Receive `.env` via secure channel
- [ ] VPN / network access to `dataplatform.advantech.com.tw:9443` confirmed
- [ ] `python -m venv .venv` + `pip install -r requirements.txt`
- [ ] `python main.py` starts without error
- [ ] `/api/denodo/status` returns `connected`
- [ ] `/kpi` loads and trend charts show the transferred history
- [ ] Lookup smoke test on `14TJ3092921-1` returns `PWR|REFE`
- [ ] Run `python tests/run_ce_verification.py` and confirm it passes
- [ ] Take one fresh snapshot for each scope (All ATJ, Phase I) to confirm write path works end-to-end
