"""
Azure OpenAI GPT caller
-----------------------
Sends enriched context (target item + top-5 similar references) to GPT
and parses the suggested ZZMCATG_M, ZZMCATG_S, and reasoning.
Uses the openai SDK (AsyncAzureOpenAI).
"""
import json
import re
from openai import AsyncAzureOpenAI
from config import settings


def _clean_category_code(val: str) -> str:
    """
    Strip descriptive names that GPT sometimes appends to category codes.
    E.g. 'DAC (DATA CONVERTER)' → 'DAC', 'DACX (D TO A CONVERTER...)' → 'DACX'
    Also handles 'DAC - DATA CONVERTER' style.
    """
    if not val:
        return val
    val = str(val).strip()
    # Remove anything in parentheses: "DAC (DATA CONVERTER)" → "DAC"
    val = re.sub(r"\s*\(.*?\)\s*$", "", val).strip()
    # Remove trailing " - description": "DAC - DATA CONVERTER" → "DAC"
    val = re.sub(r"\s*-\s+[A-Za-z].*$", "", val).strip()
    return val


def _clean_gpt_result(result: dict) -> dict:
    """Post-process GPT JSON to ensure ZZMCATG_M/S are code-only and rebuild MATERIAL_CATEGORY."""
    if "ZZMCATG_M" in result:
        result["ZZMCATG_M"] = _clean_category_code(result["ZZMCATG_M"])
    if "ZZMCATG_S" in result:
        result["ZZMCATG_S"] = _clean_category_code(result["ZZMCATG_S"])
    if "ZZMCATG_M" in result and "ZZMCATG_S" in result:
        result["MATERIAL_CATEGORY"] = f"{result['ZZMCATG_M']}|{result['ZZMCATG_S']}"
    return result


ITEM_DESC_PREFIX_GUIDE = """
== Advantech Item_Desc Prefix Decoding Guide ==

Item_Desc follows the pattern: @PREFIX MANUFACTURER_PART INFO
The @PREFIX at the start of Item_Desc encodes the component type. Use this to determine the correct category.

--- English Prefixes (30+ types) ---
@R        → Resistor (check suffix: chip, array, network, cement, carbon film, metal film, current sense, potentiometer, trimmer, thermistor NTC/PTC, varistor)
@C        → Capacitor (check suffix: MLCC ceramic, electrolytic aluminum, electrolytic polymer, tantalum, film, super/EDLC)
@CN       → Connector (check suffix: board-to-board, FFC/FPC, pin header, socket, terminal block, D-Sub, USB, RJ45, M.2, PCIe, SATA, SIM, SD, power)
@TR       → Transistor (check suffix: MOSFET, BJT, IGBT, JFET, darlington)
@LIN      → Linear IC — this is a broad category covering many analog IC sub-types. Use the sub-type rules below:
              Sub-type: OpAmp / Operational Amplifier (OPA, LMV, LMC, TLV27x, LM6xxx, AD8xxx, MCP60x) → DAC|AMPX
              Sub-type: Comparator (LMV331, LMV7239, TLV1704, LM339, LM393, MAX9xx) → DAC|AMPX  (Note: Advantech classifies comparators under DAC|AMPX, same as op-amps)
              Sub-type: Analog Switch / Multiplexer (DG94xx, TS5A, ADG7xx, MAX48xx, FSA, SN74CBT) → DAC|MUXX
              Sub-type: LVDS Driver/Receiver (SN65LVDS, DS90LV, MAX9xxx LVDS) → VDO|LVDS
              Sub-type: Voltage Regulator LDO → LDO category
              Sub-type: Voltage Reference → voltage reference category
              Sub-type: Current Sense Amplifier → current sense category
              Sub-type: Instrumentation Amplifier → DAC|AMPX
              Sub-type: Power Management PMIC → PMIC category
@LOG      → Logic IC (check suffix: buffer, gate AND/OR/NAND/NOR/XOR, flip-flop, latch, shift register, counter, level translator, bus transceiver)
@D        → Diode (check suffix: rectifier, Schottky, Zener, TVS, signal, fast recovery, bridge rectifier, LED)
@SA       → Surge Absorber / TVS array / ESD protector
@X        → Relay (electromechanical)
@RELAY    → Relay (same as @X)
@OSC      → Oscillator / Crystal (check suffix: crystal unit, crystal oscillator TCXO/VCXO/OCXO, ceramic resonator, MEMS oscillator, SAW)
@CNV      → DC-DC Converter module / Power module
@L        → Inductor / Coil (check suffix: chip inductor, power inductor, common-mode choke, ferrite bead)
@TF       → Transformer (check suffix: pulse, gate drive, power, LAN/Ethernet, audio, isolation)
@F        → Fuse (check suffix: chip fuse, PTC resettable, glass tube, SMD)
@SW       → Switch (check suffix: tact switch, DIP switch, slide, toggle, rocker, push-button, rotary encoder)
@LED      → LED (check suffix: standard, high-power, SMD, through-hole, IR, UV). Category: LED|LEDS for SMD/surface-mount, LED|LEDD for DIP/through-hole
@PH       → Photo device (check suffix: phototransistor, photodiode, photo interrupter, optocoupler, photo IC)
@PHT      → LED display / LED indicator component — includes LED digit display (7-segment, numeric, alphanumeric), LED dot matrix, LED bar graph, LED light pipe, single LED. Category: LED|LEDS for SMD/surface-mount, LED|LEDD for DIP/through-hole. Example: "@PHT LF-3011MA ﾛ-ﾑ@GRN" = ROHM LF-3011MA SMD 1-digit LED display → LED|LEDS
@PER      → Peripheral IC / Super I/O controller (e.g., SCH3114, SCH3227, W83627, IT8728, F81866) → SIO|SIOX. Note: LPCIO / Super I/O chips are NOT MCU/CPU — they are SIO category.
@IC       → IC general (check suffix: microcontroller MCU, interface UART/SPI/I2C/Ethernet/CAN/RS-232/RS-485, timer, RTC, EEPROM, watchdog, motor driver, audio codec)
@FPGA     → FPGA / CPLD
@CPU,DSP  → Combined CPU/DSP prefix — if the part is a Digital Signal Processor (TMS320, ADSP, SHARC), use DSP|DSPX (NOT CPU|DSPX)
@DSP      → Digital Signal Processor → DSP|DSPX (NOT CPU|DSPX — DSP uses the DSP major category, not CPU)
@ADC      → A/D Converter IC
@DAC      → D/A Converter IC
@CODEC    → Audio/Video Codec
@AMP      → Amplifier module
@SEN      → Sensor (check suffix: temperature, humidity, pressure, accelerometer, gyroscope, gas, current, hall effect)
@MOD      → Module (check suffix: wireless, Bluetooth, Wi-Fi, LTE/5G, GPS, LoRa, Zigbee)
@FAN      → Fan / Cooling device
@BAT      → Battery / Battery holder
@MTR      → Motor / Actuator
@BUZZER   → Buzzer / Speaker
@ANT      → Antenna

--- Japanese Katakana Prefixes (15 types) ---
@コア         → Core / Ferrite core
@ブレーカ     → Breaker / Circuit breaker
@センサ       → Sensor
@ジャック     → Jack (audio, DC power, phone)
@スイッチ     → Switch
@コネクタ     → Connector
@リレー       → Relay
@トランス     → Transformer
@コイル       → Coil / Inductor
@ヒューズ     → Fuse
@バリスタ     → Varistor
@ダイオード   → Diode
@コンデンサ   → Capacitor
@ボリューム   → Volume / Potentiometer
@ファン       → Fan

--- Category-Level Rules ---
IMPORTANT: "DISPLAY MODULE" (ZZMCATG_M) is for semi-products and assembled modules ONLY, NOT for individual components.
For individual LED components (including LED displays, 7-segment, dot matrix, LED indicators, single LEDs):
  - SMD / surface-mount package → LED|LEDS
  - DIP / through-hole package → LED|LEDD
Package clues: "SMD", "SMT", "CHIP", "SOJ", "SOP" = surface-mount → LEDS. "DIP", "THT", "THRU-HOLE", "AR" (axial/radial) = through-hole → LEDD.

--- CE-Validated Correction Examples (use these as ground truth) ---
The following are real items that were incorrectly categorized and corrected by Component Engineering (CE).
Use these examples to learn the correct mapping patterns:

@LIN + OpAmp/Amplifier → DAC|AMPX:
  "@LIN LMC6772AIMM/NOPB TI" (dual CMOS op amp) → DAC|AMPX (NOT IC|BGA IC)
  "@LIN LMV824MTX/NOPB TI" (quad CMOS op amp) → DAC|AMPX (NOT DAC|LEVL)
  "@LIN LMV751M5/NOPB TI" (op amp) → DAC|AMPX (NOT IC|BGA IC)
  "@LIN OPA4197IDR TI" (quad bipolar op amp) → DAC|AMPX (NOT PWR|DETC)
  "@LIN LM6172IMX/NOPB TI" (dual high-speed op amp) → DAC|AMPX (NOT IC|BGA IC)
  "@LIN TLV274IDR TI" (quad CMOS op amp) → DAC|AMPX (NOT LOG|COMR)

@LIN + Comparator → DAC|AMPX:
  "@LIN LMV7239M7/NOPB TI" (comparator) → DAC|AMPX (NOT LOG|COMR)
  "@LIN TLV1704AIPWR TI" (quad comparator) → DAC|AMPX (NOT PWR|DETC)
  "@LIN LMV331M7/NOPB TI" (low-voltage comparator) → DAC|AMPX (NOT PWR|DETC)

@LIN + Analog Switch/MUX → DAC|MUXX:
  "@LIN DG9431EDV-T1-GE3 VISH" (analog switch) → DAC|MUXX (NOT SWX|DETE)
  "@LIN TS5A3159ADBVR TI" (analog switch/MUX) → DAC|MUXX (NOT TYC|MUXX)

@LIN + LVDS → VDO|LVDS:
  "@LIN SN65LVDS387DGGR TI" (LVDS line driver) → VDO|LVDS (NOT DIS|LVDS)

@CPU,DSP + DSP → DSP|DSPX:
  "@CPU,DSP TMS320C6746EZWT4" (TI DSP) → DSP|DSPX (NOT CPU|DSPX)
  "@CPU,DSP TMS320C6657CZH8 T" (TI DSP) → DSP|DSPX (NOT CPU|DSPX)

@PER + Super I/O → SIO|SIOX:
  "(DEL26)@PER SCH3114I-NU MICROCHIP" (LPC Super I/O) → SIO|SIOX (NOT CPU|MCUX)

--- Special Keyword Fallback ---
When the prefix is ambiguous or absent, look for these keywords ANYWHERE in Item_Desc or MFR_PART_NUMBER:
FPGA        → FPGA category
ARM / CORTEX → Microcontroller (MCU)
FLASH       → Flash Memory
RELAY       → Relay
MOSFET      → MOSFET transistor
IGBT        → IGBT transistor
EEPROM      → EEPROM Memory
SRAM / DRAM / SDRAM / DDR → Memory (volatile)
NAND / NOR FLASH → Flash Memory (non-volatile)
LDO         → Voltage Regulator (linear)
BUCK / BOOST → DC-DC Converter (switching)
OPAMP / OP-AMP → Operational Amplifier
COMPARATOR  → Comparator
UART / SPI / I2C / CAN / RS-232 / RS-485 / USB / Ethernet → Interface IC
PWM         → PWM Controller
PLL         → PLL / Clock IC
RTC         → Real-Time Clock
WATCHDOG / WDT → Watchdog Timer
"""

SYSTEM_PROMPT = f"""You are a PLM (Product Lifecycle Management) component categorization expert
for Advantech. Your task is to assign the correct MATERIAL_CATEGORY to an electronic component
based on its description, manufacturer part number, and the categories of similar components.

MATERIAL_CATEGORY format is always: ZZMCATG_M|ZZMCATG_S
where ZZMCATG_M is the middle-level category CODE (e.g. "DAC", "FLH", "CLK") and
ZZMCATG_S is the small-level category CODE (e.g. "ADCX", "NORX", "RTCX").
CATE_M_NAME and CATE_S_NAME shown in references are descriptive names — do NOT include them in your output codes.

CRITICAL: ZZMCATG_M and ZZMCATG_S must be SHORT CODES ONLY (typically 2-4 uppercase letters).
Do NOT include descriptive names like "DAC (DATA CONVERTER)" — just return "DAC".

{ITEM_DESC_PREFIX_GUIDE}

Rules:
- Use the Item_Desc prefix guide above to identify the component type FIRST, then cross-check with reference items.
- If the prefix clearly indicates a component type but references suggest a different category, trust the prefix + MPN analysis over weak similarity matches.
- Always respond in valid JSON only, no markdown, no explanation outside the JSON.
- If confidence is low, set confidence to "low" and explain why in reason.
- Suggest only category codes that appear in the reference items provided.
- JSON format: {{"ZZMCATG_M": "...", "ZZMCATG_S": "...", "MATERIAL_CATEGORY": "...", "confidence": "high|medium|low", "reason": "..."}}
"""

# Lazy-init client (created on first call)
_client: AsyncAzureOpenAI | None = None


def _get_client() -> AsyncAzureOpenAI:
    global _client
    if _client is None:
        _client = AsyncAzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
    return _client


def _build_user_prompt(target: dict, references: list[dict]) -> str:
    ref_lines = []
    for i, r in enumerate(references, 1):
        ref_lines.append(
            f"  {i}. Item: {r.get('Item_Number', '')} | "
            f"MPN: {r.get('MFR_PART_NUMBER', '')} | "
            f"Desc: {r.get('Item_Desc', '')} | "
            f"Middle cat: {r.get('ZZMCATG_M', '')} ({r.get('CATE_M_NAME', '')}) | "
            f"Small cat: {r.get('ZZMCATG_S', '')} ({r.get('CATE_S_NAME', '')}) | "
            f"Similarity: {r.get('similarity_score', 0):.1f}/100"
        )
    refs_block = "\n".join(ref_lines) if ref_lines else "  (no similar items found)"

    return f"""Target component needing MATERIAL_CATEGORY:
- Item_Number:      {target.get('Item_Number', '')}
- Item_Desc:        {target.get('Item_Desc', '')}
- Manufacturer:     {target.get('MANUFACTURE_NAME', '')}
- MFR_PART_NUMBER:  {target.get('MFR_PART_NUMBER', '')}
- LifeCycle_Phase:  {target.get('LifeCycle_Phase', '')}

Top {len(references)} most similar components for reference:
{refs_block}

Based on the above, suggest ZZMCATG_M and ZZMCATG_S for the target component.
Respond with valid JSON only."""


CATEGORY_SELECT_SYSTEM_PROMPT = f"""You are a PLM (Product Lifecycle Management) component categorization expert
for Advantech. You are given a target electronic component and a list of candidate MATERIAL_CATEGORY options
retrieved from a vector similarity search.

Your task is to select the BEST matching MATERIAL_CATEGORY from the candidates for the target component.

CRITICAL: ZZMCATG_M and ZZMCATG_S must be SHORT CODES ONLY (typically 2-4 uppercase letters).
Do NOT include descriptive names in the code fields. Example: return "DAC" not "DAC (DATA CONVERTER)".
The descriptive names (CATE_M_NAME, CATE_S_NAME) go in their own separate fields.

{ITEM_DESC_PREFIX_GUIDE}

Rules:
- Use the Item_Desc prefix guide above to identify the component type FIRST, then select the best matching candidate.
- If the prefix clearly indicates a component type, prefer candidates that match that type even if vector similarity is slightly lower.
- Always respond in valid JSON only, no markdown, no explanation outside the JSON.
- You MUST pick one of the provided candidate categories. Do not invent new codes.
- Consider the component's description and the first GPT analysis reason when choosing.
- JSON format: {{"ZZMCATG_M": "...", "ZZMCATG_S": "...", "MATERIAL_CATEGORY": "...", "CATE_M_NAME": "...", "CATE_S_NAME": "...", "confidence": "high|medium|low", "reason": "..."}}
"""


def _build_category_select_prompt(
    target: dict,
    first_reason: str,
    candidates: list[dict],
) -> str:
    cand_lines = []
    for i, c in enumerate(candidates, 1):
        cand_lines.append(
            f"  {i}. ZZMCATG_M: {c.get('ZZMCATG_M', '')} ({c.get('CATE_M_NAME', '')}) | "
            f"ZZMCATG_S: {c.get('ZZMCATG_S', '')} ({c.get('CATE_S_NAME', '')}) | "
            f"MATERIAL_CATEGORY: {c.get('MATERIAL_CATEGORY', '')} | "
            f"Vector similarity: {c.get('similarity', 0):.3f}"
        )
    cands_block = "\n".join(cand_lines)

    return f"""Target component needing MATERIAL_CATEGORY:
- Item_Number:      {target.get('Item_Number', '')}
- Item_Desc:        {target.get('Item_Desc', '')}
- Manufacturer:     {target.get('MANUFACTURE_NAME', '')}
- MFR_PART_NUMBER:  {target.get('MFR_PART_NUMBER', '')}

Initial analysis (from first pass — low confidence):
  {first_reason}

Top {len(candidates)} candidate categories from vector search:
{cands_block}

Based on the component description and the initial analysis, select the BEST matching
MATERIAL_CATEGORY from the candidates above. Respond with valid JSON only."""


async def suggest_category_from_candidates(
    target: dict,
    first_reason: str,
    candidates: list[dict],
) -> dict:
    """
    Second-pass GPT call: given a target item and top-K category candidates
    from the vector DB, select the best match.
    Returns dict with ZZMCATG_M, ZZMCATG_S, MATERIAL_CATEGORY, confidence, reason.
    """
    try:
        client = _get_client()
        response = await client.chat.completions.create(
            model=settings.azure_openai_deployment,
            messages=[
                {"role": "system", "content": CATEGORY_SELECT_SYSTEM_PROMPT},
                {"role": "user", "content": _build_category_select_prompt(
                    target, first_reason, candidates
                )},
            ],
            temperature=0.1,
            max_completion_tokens=300,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        result = _clean_gpt_result(result)
        # Tag as vector-fallback result
        result["source"] = "vector_fallback"
        return result
    except Exception as e:
        return {
            "ZZMCATG_M": "",
            "ZZMCATG_S": "",
            "MATERIAL_CATEGORY": "",
            "confidence": "error",
            "reason": f"Vector fallback GPT error: {e}",
            "source": "vector_fallback",
        }


async def suggest_category(target: dict, references: list[dict]) -> dict:
    """
    Calls Azure OpenAI and returns parsed suggestion dict:
      {ZZMCATG_M, ZZMCATG_S, MATERIAL_CATEGORY, confidence, reason}
    Falls back to error dict on any exception.
    """
    try:
        client = _get_client()
        response = await client.chat.completions.create(
            model=settings.azure_openai_deployment,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(target, references)},
            ],
            temperature=0.1,
            max_completion_tokens=300,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        result = _clean_gpt_result(result)
        return result
    except Exception as e:
        return {
            "ZZMCATG_M": "",
            "ZZMCATG_S": "",
            "MATERIAL_CATEGORY": "",
            "confidence": "error",
            "reason": str(e),
        }
