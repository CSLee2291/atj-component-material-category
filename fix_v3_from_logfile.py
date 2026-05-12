"""
Fix error PNs using LogFile.xml import validation errors.

Reads:
  - error_list_pn.xlsx   : PNs that had wrong Material Category in AI_MATERIAL_CATEGORY_v2
  - LogFile.xml          : Agile import log with invalid values and valid alternatives
  - Phase_I_PUR_2024_2025.xlsx : Source Excel to update

Writes corrected values into new v3 columns:
  AI_MATERIAL_CATEGORY_v3, AI_CONFIDENCE_v3, AI_REASON_v3, AI_UPDATED_AT_v3

Correction logic:
  - Small category invalid, 1 valid option  -> auto-correct, confidence=corrected
  - Small category invalid, N valid options -> fuzzy-match best, confidence=corrected
  - Big category invalid                   -> fuzzy-match against top-level list,
                                             or promote small-cat if it IS a top-level,
                                             confidence=corrected (needs_review if ambiguous)
"""
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from difflib import get_close_matches, SequenceMatcher

import openpyxl

# ── Paths ───────────────────────────────────────────────────────────────────
LOGFILE_XML   = "../LogFile.xml"
ERROR_LIST    = "error_list_pn.xlsx"
PHASE_I_EXCEL = "Phase_I_PUR_2024_2025.xlsx"
PHASE_I_SHEET = "Phase_I_List"

V3_COLS = [
    "AI_MATERIAL_CATEGORY_v3",
    "AI_CONFIDENCE_v3",
    "AI_REASON_v3",
    "AI_UPDATED_AT_v3",
]


# ── XML parsing ──────────────────────────────────────────────────────────────
_PN_RE = re.compile(r"\*Part Number:\s*([\w\-]+)")


def _parse_error_text(text: str) -> tuple[str, str, list[str]]:
    """
    Returns (invalid_value, cascade_string, valid_values_list).
    Example message:
      "The node value 'MCUX' within the cascade list string 'CPU|MCUX'
       you provided is invalid. The valid values for this cascade list
       node are 'CPUX'"

    Note: valid_values may contain apostrophes (e.g. "ASS'Y FOR ME PARTS"),
    so we capture from "are '" to the END of the text, then rstrip the trailing '.
    """
    m_inv = re.search(r"node value '([^']+)'", text)
    m_cas = re.search(r"cascade list string '([^']+)'", text)
    # Capture everything after "are '" to handle embedded apostrophes
    m_val = re.search(r"valid values.*?are '(.+)", text, re.DOTALL)

    invalid_value  = m_inv.group(1).strip() if m_inv else ""
    cascade_string = m_cas.group(1).strip() if m_cas else ""
    valid_str      = m_val.group(1).strip().rstrip("'") if m_val else ""

    # Valid values are comma-separated
    valid_values = [v.strip() for v in valid_str.split(",") if v.strip()]
    return invalid_value, cascade_string, valid_values


def parse_logfile(xml_path: str) -> dict[str, dict]:
    """
    Returns {pn: {invalid_value, cascade_string, valid_values}} for error elements.
    If a PN appears multiple times, last entry wins (shouldn't normally happen).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    result: dict[str, dict] = {}
    for elem in root.iter("error"):
        context = elem.get("context", "")
        text    = elem.text or ""

        m_pn = _PN_RE.search(context)
        if not m_pn:
            continue
        pn = m_pn.group(1).strip()

        invalid_value, cascade_string, valid_values = _parse_error_text(text)
        result[pn] = {
            "invalid_value":  invalid_value,
            "cascade_string": cascade_string,
            "valid_values":   valid_values,
        }
    return result


# ── Correction logic ─────────────────────────────────────────────────────────
def _fuzzy_best(query: str, candidates: list[str]) -> tuple[str, float]:
    """Return (best_match, score 0-1). Score via SequenceMatcher."""
    best, best_score = "", 0.0
    q_lower = query.lower()
    for c in candidates:
        score = SequenceMatcher(None, q_lower, c.lower()).ratio()
        if score > best_score:
            best, best_score = c, score
    return best, best_score


def determine_correction(
    invalid_value: str,
    cascade_string: str,
    valid_values: list[str],
) -> tuple[str, str, str]:
    """
    Returns (corrected_category, confidence, reason).

    confidence values:
      'corrected'        - single valid option; auto-applied
      'corrected-fuzzy'  - multiple options; best fuzzy match applied
      'needs_review'     - ambiguous or no clear match; best guess applied
    """
    parts = cascade_string.split("|", 1)
    big_cat = parts[0].strip() if len(parts) > 0 else ""
    small_cat = parts[1].strip() if len(parts) > 1 else ""

    # Determine if the big or small category is invalid
    if invalid_value == small_cat:
        # Small category is wrong; big category stays
        if len(valid_values) == 1:
            new_small = valid_values[0]
            corrected = f"{big_cat}|{new_small}"
            reason = (
                f"Small category '{invalid_value}' invalid under '{big_cat}'. "
                f"Only valid option: '{new_small}'."
            )
            return corrected, "corrected", reason

        # Multiple options — fuzzy match
        best, score = _fuzzy_best(invalid_value, valid_values)
        if score < 0.3:
            # No reasonable match; reset small to '-' so CE can pick manually
            corrected = f"{big_cat}|-"
            reason = (
                f"Small category '{invalid_value}' invalid under '{big_cat}'. "
                f"No close fuzzy match (best='{best}', score={score:.2f}); "
                f"reset to '-'. Valid options: {', '.join(valid_values[:8])}{'...' if len(valid_values)>8 else ''}."
            )
            return corrected, "needs_review", reason

        corrected = f"{big_cat}|{best}"
        reason = (
            f"Small category '{invalid_value}' invalid under '{big_cat}'. "
            f"Best fuzzy match from {len(valid_values)} options: '{best}' "
            f"(score={score:.2f}). Valid: {', '.join(valid_values[:8])}{'...' if len(valid_values)>8 else ''}."
        )
        conf = "corrected-fuzzy" if score >= 0.5 else "needs_review"
        return corrected, conf, reason

    elif invalid_value == big_cat:
        # Big/top-level category is wrong; valid_values is the top-level list

        # Special case: if the small_cat itself exists as a top-level, promote it
        if small_cat and small_cat != "-":
            if small_cat in valid_values:
                corrected = f"{small_cat}|-"
                reason = (
                    f"Top-level category '{big_cat}' is invalid. "
                    f"Small category '{small_cat}' is a valid top-level; promoted to '{small_cat}|-'."
                )
                return corrected, "corrected-fuzzy", reason

            # fuzzy match small_cat against valid top-levels
            best_s, score_s = _fuzzy_best(small_cat, valid_values)
            if score_s >= 0.6:
                corrected = f"{best_s}|-"
                reason = (
                    f"Top-level '{big_cat}' invalid. "
                    f"Small '{small_cat}' fuzzy-matched to top-level '{best_s}' "
                    f"(score={score_s:.2f}); used as '{best_s}|-'."
                )
                return corrected, "corrected-fuzzy", reason

        # 1. Prefer prefix matches: valid value that starts WITH big_cat
        #    e.g. "OTHERS SW" -> "OTHERS SW(ADV)", "OTHERS SW(ROYALTY)"
        prefix_matches = [v for v in valid_values if v.startswith(big_cat)]
        if prefix_matches:
            best_b = prefix_matches[0]
            reason = (
                f"Top-level '{big_cat}' invalid. "
                f"Prefix-matched to '{best_b}'; small cat reset to '-'. "
                f"Manual review recommended if multiple options exist."
            )
            return f"{best_b}|-", "corrected-fuzzy", reason

        # 2. Suffix-trim: big_cat starts WITH a valid value (e.g. "I/O BRACKET" -> "I/O")
        suffix_trim = sorted(
            [v for v in valid_values if len(v) >= 3 and big_cat.startswith(v)],
            key=len, reverse=True,  # longest match wins
        )
        if suffix_trim:
            best_b = suffix_trim[0]
            reason = (
                f"Top-level '{big_cat}' invalid. "
                f"Trimmed to valid base '{best_b}'; small cat reset to '-'. "
                f"Manual review recommended."
            )
            return f"{best_b}|-", "corrected-fuzzy", reason

        # 3. Fall back to fuzzy match against valid top-levels
        best_b, score_b = _fuzzy_best(big_cat, valid_values)
        corrected = f"{best_b}|-"
        conf = "corrected-fuzzy" if score_b >= 0.5 else "needs_review"
        reason = (
            f"Top-level '{big_cat}' invalid. "
            f"Fuzzy-matched to '{best_b}' (score={score_b:.2f}) from valid top-levels; "
            f"small cat reset to '-'."
        )
        return corrected, conf, reason

    else:
        # invalid_value matches neither; just report and use best fuzzy guess
        best, score = _fuzzy_best(invalid_value, valid_values)
        reason = (
            f"Cascade '{cascade_string}': value '{invalid_value}' invalid. "
            f"Best match: '{best}' (score={score:.2f}). Manual review recommended."
        )
        corrected = cascade_string.replace(invalid_value, best, 1)
        return corrected, "needs_review", reason


# ── Main ─────────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def main() -> None:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Load error PN list (all rows are PNs, no header)
    log(f"Loading error PN list from {ERROR_LIST}...")
    wb_err = openpyxl.load_workbook(ERROR_LIST)
    ws_err = wb_err.active
    error_pns: set[str] = set()
    for row in ws_err.iter_rows(min_row=1, values_only=True):
        v = row[0]
        if v:
            error_pns.add(str(v).strip())
    log(f"  {len(error_pns)} error PNs loaded.")

    # 2. Parse LogFile.xml
    log(f"Parsing {LOGFILE_XML}...")
    pn_errors = parse_logfile(LOGFILE_XML)
    log(f"  {len(pn_errors)} error entries parsed from XML.")

    # Compute corrections for all error PNs
    corrections: dict[str, tuple[str, str, str]] = {}
    for pn, info in pn_errors.items():
        if pn not in error_pns:
            continue  # skip if not in our target list
        cat, conf, reason = determine_correction(
            info["invalid_value"],
            info["cascade_string"],
            info["valid_values"],
        )
        corrections[pn] = (cat, conf, reason)
        log(f"  {pn}: '{info['cascade_string']}' -> '{cat}' [{conf}]")

    # Warn about error PNs not found in XML
    xml_pns = set(pn_errors.keys())
    missing_in_xml = error_pns - xml_pns
    if missing_in_xml:
        log(f"  WARNING: {len(missing_in_xml)} error PNs not found in XML: {sorted(missing_in_xml)}")

    # 3. Open Phase_I Excel and write v3 columns
    log(f"Opening {PHASE_I_EXCEL} / sheet '{PHASE_I_SHEET}'...")
    wb = openpyxl.load_workbook(PHASE_I_EXCEL)
    ws = wb[PHASE_I_SHEET]

    headers: dict[str, int] = {
        cell.value: cell.column for cell in ws[1] if cell.value
    }

    # Add missing v3 columns
    missing_v3 = [c for c in V3_COLS if c not in headers]
    if missing_v3:
        max_col = ws.max_column
        for i, col_name in enumerate(missing_v3):
            col_idx = max_col + 1 + i
            ws.cell(row=1, column=col_idx, value=col_name)
            headers[col_name] = col_idx
        log(f"  Added new columns: {missing_v3}")
    else:
        log(f"  v3 columns already exist; will overwrite for target rows.")

    col_mat    = headers["Material"]
    col_v3_cat = headers["AI_MATERIAL_CATEGORY_v3"]
    col_v3_con = headers["AI_CONFIDENCE_v3"]
    col_v3_rsn = headers["AI_REASON_v3"]
    col_v3_ts  = headers["AI_UPDATED_AT_v3"]

    written = 0
    skipped = 0
    for ws_row_idx in range(2, ws.max_row + 1):
        mat = ws.cell(row=ws_row_idx, column=col_mat).value
        if not mat or str(mat).strip() not in corrections:
            continue
        pn = str(mat).strip()
        cat, conf, reason = corrections[pn]

        ws.cell(row=ws_row_idx, column=col_v3_cat).value = cat
        ws.cell(row=ws_row_idx, column=col_v3_con).value = conf
        ws.cell(row=ws_row_idx, column=col_v3_rsn).value = reason
        ws.cell(row=ws_row_idx, column=col_v3_ts).value  = now_str
        written += 1

    log(f"Writing complete: {written} rows updated, {skipped} skipped.")

    wb.save(PHASE_I_EXCEL)
    log(f"Saved: {PHASE_I_EXCEL}")

    # 4. Summary
    conf_counts: dict[str, int] = {}
    for _, conf, _ in corrections.values():
        conf_counts[conf] = conf_counts.get(conf, 0) + 1
    log("")
    log("===== CORRECTION SUMMARY =====")
    log(f"Total error PNs in list:   {len(error_pns)}")
    log(f"Matched in LogFile.xml:    {len(corrections)}")
    log(f"Written to Excel (v3):     {written}")
    log("Confidence breakdown:")
    for k, n in sorted(conf_counts.items()):
        log(f"  {k:<20} {n}")


if __name__ == "__main__":
    main()
