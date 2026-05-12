"""Probe Azure OpenAI and Vertex AI Gemini and print availability."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.llm_router import status as llm_status


async def main():
    res = await llm_status()
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
