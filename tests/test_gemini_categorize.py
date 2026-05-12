"""Single end-to-end Gemini categorization to validate JSON output and
whitelist enforcement match the Azure path."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import gemini_caller


async def main():
    target = {
        "Item_Number": "14TJ5620269-7",
        "Item_Desc": "@LIN LMV824MTX/NOPB TI quad CMOS op amp",
        "MANUFACTURE_NAME": "TI",
        "MFR_PART_NUMBER": "LMV824MTX/NOPB",
        "LifeCycle_Phase": "Part Number Release",
    }
    references = [
        {
            "Item_Number": "14TJ0000001-0", "MFR_PART_NUMBER": "OPA4197IDR",
            "Item_Desc": "@LIN OPA4197IDR TI quad bipolar op amp",
            "ZZMCATG_M": "DAC", "CATE_M_NAME": "DATA CONVERTER",
            "ZZMCATG_S": "AMPX", "CATE_S_NAME": "AMPLIFIER",
            "similarity_score": 78.0,
        },
    ]
    res = await gemini_caller.suggest_category(target, references)
    print(json.dumps(res, indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
