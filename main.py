"""
ATJ-Component MATERIAL_CATEGORY Update Server
Fuzzy-matches ATJ MPNs, enriches with category context,
calls Azure OpenAI GPT for MATERIAL_CATEGORY suggestions,
and exports review-ready Excel for CE/PLM ECO workflow.
Includes KPI dashboard for tracking material category completion.
"""
import uvicorn
from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="ATJ Category Updater", version="1.0.0")
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
