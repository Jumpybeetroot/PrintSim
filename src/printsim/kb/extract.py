"""High-signal engineering prefilter, deterministic record extractor, and Klipper CSV parser."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Signal Scoring Patterns
# ---------------------------------------------------------------------------

_SHAPER_TYPE_RE = re.compile(
    r"\b(?:smooth_mzv|smooth_zv|2hump_ei|3hump_ei|mzv|zv|ei|zvd|shaper_calibrate|"
    r"shaper_freq_[xy]|shaper_type_[xy]|input\s*shaper|shaketune|shake&tune)\b",
    re.IGNORECASE,
)
_HZ_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*hz\b", re.IGNORECASE)
_ACCEL_RE = re.compile(
    r"(?:max_accel|accel(?:eration)?|<=|≤)\s*[=:]?\s*~?\s*(\d+(?:\.\d+)?)\s*(k|mm/s(?:\^?2|²)|mm/sec\^?2)?\b"
    r"|\b(\d+(?:\.\d+)?)\s*(k)\s*(?:accel|mm/s2)?\b",
    re.IGNORECASE,
)

_BELT_WIDTH_RE = re.compile(r"\b(6|9|10|12)(?:\s*[-–]\s*(?:9|10|12))?\s*mm\b", re.IGNORECASE)
_BELT_TERM_RE = re.compile(
    r"\b(?:epdm|aramid|kevlar|gates|gt2|gt3|2mgt|2gt|pluck|lbf|awd|2wd|belt\s*tension|belts?)\b",
    re.IGNORECASE,
)

_COMPONENT_NAME_RE = re.compile(
    r"\b(?:stealthburner|xol(?:\s*2)?|a4t|dragon\s*burner|archetype(?:\s*cpap)?|goliath|"
    r"chube(?:\s*conduction)?|rapido(?:\s*2)?|mgn12[hc]?|mgn9[hc]?|gantry|toolhead|"
    r"carriage|extruder|hotend|x-?beam|bed\s*plate|clockwork\s*2|cw2|orbiter|sherpa)\b",
    re.IGNORECASE,
)
_MASS_VALUE_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(g|grams?|kg)\b", re.IGNORECASE)

_MATERIAL_TERM_RE = re.compile(
    r"\b(?:gpa|hdt|creep|ppa-?cf|ppa-?gf|pps-?cf|pps-?gf|pet-?cf|pa12-?cf|pa-?cf|"
    r"5052(?:-h32)?|6061(?:-t6)?|7075(?:-t6)?|acm|dibond|carbon\s*fiber|annealed|"
    r"mic-?6|atp-?5|polycarbonate|modulus)\b",
    re.IGNORECASE,
)

_TROUBLE_SYMPTOM_RE = re.compile(
    r"\b(?:ringing|ghosting|vfa|step\s*skip(?:ping)?|oil-?canning|taco(?:ing)?|"
    r"bimetallic|double-?shear|voron\s*tap|\btap\b|resonance|slipped|slipping|"
    r"drumming|creep|backlash|layer\s*shift|banding)\b",
    re.IGNORECASE,
)
_TROUBLE_FIX_RE = re.compile(
    r"\b(?:fixed|solved|eliminated|resolved|root\s*cause|caused\s*by|replaced\s*with|"
    r"switched\s*to|upgraded\s*to|tightened|bolted|removed\s*foam|verified\s*fix)\b",
    re.IGNORECASE,
)

_SHAPER_ATTACH_NAME_RE = re.compile(
    r"(?:shaper|resonance|axes|adxl|psd|vibrations|belts|input_shaper|calibration_data)",
    re.IGNORECASE,
)
_SHAPER_ATTACH_EXT_RE = re.compile(r"\.(?:png|jpg|jpeg|webp|csv)$", re.IGNORECASE)


def score_engineering_signal(
    text: str,
    attachments: list[dict[str, Any]] | None = None,
) -> tuple[float, list[str]]:
    """Fast regex prefilter scoring 3D printer engineering signal in [0.0, 1.0].

    Returns ``(score, tags)`` where ``tags`` are drawn from:
    ``shaper_data``, ``belt_data``, ``weight_data``, ``material_data``,
    ``troubleshooting``, ``shaper_attachment``.
    """
    body = text or ""
    score = 0.0
    tags: list[str] = []

    has_shaper_kw = bool(_SHAPER_TYPE_RE.search(body))
    hz_matches = _HZ_RE.findall(body)
    has_accel = bool(_ACCEL_RE.search(body))

    if has_shaper_kw and (hz_matches or has_accel):
        score += 0.45
        tags.append("shaper_data")
    elif has_shaper_kw:
        score += 0.20
        tags.append("shaper_data")
    elif len(hz_matches) >= 2 and re.search(r"\b[xy]\s*[:=@]", body, re.IGNORECASE):
        score += 0.30
        tags.append("shaper_data")

    has_belt_width = bool(_BELT_WIDTH_RE.search(body))
    belt_terms = set(m.lower() for m in _BELT_TERM_RE.findall(body))
    if (has_belt_width and belt_terms) or len(belt_terms) >= 2:
        score += 0.25
        tags.append("belt_data")
    elif belt_terms and hz_matches:
        score += 0.20
        tags.append("belt_data")

    # Component weight: check if a component name and a gram/kg value appear within 80 chars
    weight_hit = False
    if _COMPONENT_NAME_RE.search(body) and _MASS_VALUE_RE.search(body):
        for cm in _COMPONENT_NAME_RE.finditer(body):
            window = body[max(0, cm.start() - 40) : min(len(body), cm.end() + 60)]
            if _MASS_VALUE_RE.search(window):
                weight_hit = True
                break
    if weight_hit:
        score += 0.30
        tags.append("weight_data")

    mat_terms = set(m.lower() for m in _MATERIAL_TERM_RE.findall(body))
    if len(mat_terms) >= 2 or (mat_terms and (hz_matches or re.search(r"\b\d+(?:\.\d+)?\s*(?:gpa|°?c|mm|kg|n|lbf)\b", body, re.IGNORECASE))):
        score += 0.25
        tags.append("material_data")

    symptoms = set(m.lower() for m in _TROUBLE_SYMPTOM_RE.findall(body))
    fixes = set(m.lower() for m in _TROUBLE_FIX_RE.findall(body))
    if symptoms and fixes:
        score += 0.30
        tags.append("troubleshooting")
    elif len(symptoms) >= 2:
        score += 0.20
        tags.append("troubleshooting")

    if attachments:
        for att in attachments:
            fname = str(
                att.get("filename")
                or att.get("fileName")
                or att.get("name")
                or att.get("url")
                or ""
            )
            if _SHAPER_ATTACH_EXT_RE.search(fname) and _SHAPER_ATTACH_NAME_RE.search(fname):
                score += 0.35
                tags.append("shaper_attachment")
                break

    return (round(min(1.0, score), 3), tags)


# ---------------------------------------------------------------------------
# Deterministic Structured Record Extraction
# ---------------------------------------------------------------------------

def _parse_accel_number(num_str: str, suffix: str | None) -> float | None:
    try:
        val = float(num_str)
    except ValueError:
        return None
    if suffix and suffix.lower().startswith("k"):
        val *= 1000.0
    if val < 500.0:
        return None
    return round(val, 1)


def _infer_architecture(text: str) -> str:
    low = text.lower()
    if re.search(r"\b(?:delta|kossel|flsun|doron\s*velta)\b", low):
        return "delta"
    if re.search(r"\b(?:bedslinger|ender\s*3|prusa\s*mk[234]|i3|sv06|neptune|switchwire)\b", low):
        return "cartesian_bedslinger"
    return "corexy"


MIN_SHAPER_FREQ_HZ = 15.0
MAX_SHAPER_FREQ_HZ = 250.0

_MALFORMED_BUILD_NAME_RE = re.compile(
    r"^\s*(?:failed\s+to\b|error\b|traceback\b|exception\b|fatal\b|"
    r"crash(?:ed|ing)?\b|warning\b|unable\s+to\b|cannot\b|can'?t\b|"
    r"invalid\b|klippy\.log\b|shaper(?:_[xyz]|s)?\b|"
    r"mcu\s+['\"]?\w+['\"]?\s+shutdown)"
    r"|\b(?:failed\s+to\s+configure|traceback\s*\(most\s+recent\s+call|"
    r"shaper_z\b|syntax\s*error|attribute\s*error|value\s*error|key\s*error|"
    r"internal\s+error)\b",
    re.IGNORECASE,
)


def is_malformed_build_name(name: str) -> bool:
    """Return True if a candidate build name looks like an error log, traceback, or empty noise."""
    cleaned = (name or "").strip()
    if not cleaned or len(cleaned) < 3:
        return True
    if not re.search(r"[a-zA-Z]", cleaned):
        return True
    return bool(_MALFORMED_BUILD_NAME_RE.search(cleaned))


def _is_valid_shaper_hz(freq_hz: float | None) -> bool:
    return freq_hz is not None and MIN_SHAPER_FREQ_HZ <= freq_hz <= MAX_SHAPER_FREQ_HZ


def _infer_printer_model(text: str) -> str:
    patterns = [
        r"\b(Voron\s*Trident(?:\s*\d{3})?)\b",
        r"\b(VT\s*\d{3}(?:\s*AWD)?)\b",
        r"\b(Voron\s*(?:2\.4|V2\.4|V0(?:\.\d)?)(?:\s*\d{3})?)\b",
        r"\b(DoomCube(?:\s*\d{3})?|Doomish)\b",
        r"\b(VzBot(?:\s*\d{3})?(?:\s*AWD)?)\b",
        r"\b(RatRig\s*V-?Core(?:\s*\d(?:\.\d)?)?(?:\s*\d{3})?)\b",
        r"\b(Annex\s*K3|K3Z)\b",
        r"\b(Monolith(?:\s*AWD)?(?:\s*\d{3})?)\b",
        r"\b(HM\d{3})\b",
        r"\b(HevORT)\b",
        r"\b(FLSUN\s*\w+)\b",
        r"\b(Doron\s*Velta)\b",
        r"\b(Ender\s*[357](?:\s*\w+)?)\b",
        r"\b(Prusa\s*MK\d\w*)\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


def _infer_frame_profile(text: str) -> str:
    if re.search(r"4040\s*\+\s*2020|2020\s*\+\s*4040", text, re.IGNORECASE):
        return "4040+2020"
    for prof in ("4040", "3030", "2040", "2020", "1515"):
        if re.search(rf"\b{prof}\b", text):
            return prof
    return ""


def _infer_frame_size_mm(text: str) -> float | None:
    # Look for explicit build sizes like 250, 300, 330, 350, 400, 500
    m = re.search(
        r"\b(?:vt|trident|v2\.4|voron|vzbot|doomcube|hm|stiff|corexy)?\s*(250|300|330|350|400|500)\s*(?:mm|³|\^3|\b)",
        text,
        re.IGNORECASE,
    )
    if m:
        return float(m.group(1))
    return None


def _infer_drive_mode(text: str) -> str:
    has_awd = bool(re.search(r"\bawd\b|4-?motor", text, re.IGNORECASE))
    has_2wd = bool(re.search(r"\b2wd\b|2-?motor", text, re.IGNORECASE))
    if has_awd and has_2wd:
        return "awd"
    if has_awd:
        return "awd"
    if has_2wd:
        return "2wd"
    return ""


def _infer_belt_width_mm(text: str) -> float | None:
    # Prefer belt context, fallback to standalone standard widths
    m = re.search(
        r"\b(6|9|10|12)(?:\s*[-–]\s*(9|10|12))?\s*mm\b(?=[^.\n]{0,40}\b(?:belt|gt2|gt3|2gt|epdm|aramid|kevlar|awd|2wd|pluck)\b)"
        r"|\b(?:belt|gt2|gt3|2gt|awd|2wd)\b[^.\n]{0,30}\b(6|9|10|12)(?:\s*[-–]\s*(9|10|12))?\s*mm\b",
        text,
        re.IGNORECASE,
    )
    if m:
        nums = [float(g) for g in m.groups() if g is not None]
        if nums:
            return round(sum(nums) / len(nums), 1)
    return None


def _infer_belt_core(text: str) -> str:
    low = text.lower()
    if "tpu" in low and ("kevlar" in low or "aramid" in low):
        return "tpu_kevlar"
    if "aramid" in low or "kevlar" in low:
        return "aramid"
    if "steel-core" in low or "steel core" in low:
        return "steel"
    if "epdm" in low or "fiberglass" in low or "gt2" in low or "gt3" in low:
        return "fiberglass"
    return ""


def _infer_belt_pluck_hz(text: str) -> float | None:
    patterns = [
        r"(?:pluck(?:\s*freq(?:uency)?)?|belt\s*tension)\s*(?:@|at|=|:|of)?\s*~?\s*(\d+(?:\.\d+)?)\s*(?:[-–]\s*(\d+(?:\.\d+)?))?\s*hz\b",
        r"@\s*(\d+(?:\.\d+)?)\s*hz\s*pluck\b",
        r"\b(\d{3}(?:\.\d+)?)\s*(?:[-–]\s*(\d{3}(?:\.\d+)?))?\s*hz\s*(?:pluck|belt)\b",
        r"\b(?:6|9|10|12)\s*mm\s*@\s*(\d+(?:\.\d+)?)\s*hz\b",
        r"\b(?:6|9|10|12)(?:\s*[-–]\s*(?:9|10|12))?\s*mm\s*\((\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*hz\)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            v1 = float(m.group(1))
            v2 = float(m.group(2)) if m.lastindex and m.lastindex >= 2 and m.group(2) else None
            val = (v1 + v2) / 2.0 if v2 is not None else v1
            if 70.0 <= val <= 400.0:
                return round(val, 1)
    return None


def _infer_voltage(text: str) -> float | None:
    m = re.search(r"\b(24|36|48|60)\s*v\b", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _infer_toolhead(text: str) -> str:
    for pat, label in [
        (r"\bstealthburner\b", "StealthBurner"),
        (r"\barchetype(?:\s*cpap)?\b", "Archetype"),
        (r"\bdragon\s*burner\b", "Dragon Burner"),
        (r"\bxol(?:\s*2)?\b", "XOL"),
        (r"\ba4t\b", "A4T"),
        (r"\banthead\b", "Anthead"),
        (r"\breaper\b", "Reaper"),
        (r"\bjabberwocky\b", "Jabberwocky"),
    ]:
        if re.search(pat, text, re.IGNORECASE):
            return label
    return ""


def _infer_hotend(text: str) -> str:
    for pat, label in [
        (r"\bchube(?:\s*conduction)?\b", "Chube"),
        (r"\bgoliath(?:\s*air|\s*water)?\b", "Goliath"),
        (r"\brapido(?:\s*2)?(?:\s*uhf)?\b", "Rapido"),
        (r"\bdragon(?:\s*hf|\s*uhf|\s*ace)?\b", "Dragon"),
        (r"\brevo\b", "Revo"),
        (r"\bmosquito\b", "Mosquito"),
    ]:
        if re.search(pat, text, re.IGNORECASE):
            return label
    return ""


def _infer_motors(text: str) -> str:
    found: list[str] = []
    for pat in [
        r"\bLDO\s*42STH\d{2}-\d{4}[A-Z]{0,3}\b",
        r"\bLDO\s*2[58]04[A-Z]{0,3}\b",
        r"\bS45R\b",
        r"\bKraken\b",
        r"\bTMC5160(?:Plus)?\b",
        r"\b5160Plus\b",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            found.append(m.group(0))
    return ", ".join(dict.fromkeys(found))


def _infer_panel_spec(text: str) -> str:
    patterns = [
        r"\b(\d+(?:\.\d+)?\s*(?:[-–]\s*\d+(?:\.\d+)?)?\s*mm\s*(?:Al(?:uminum)?|Aluminium)\s*(?:5052(?:-H32)?|6061)?(?:\s*\+\s*PIR)?(?:\s*panels?)?)\b",
        r"\b(\d+(?:\.\d+)?\s*mm\s*(?:Mild/?304\s*Steel|Steel|ACM|Dibond|Acrylic|Polycarbonate)(?:\s*panels?)?)\b",
        r"\b(ACM\s*Dibond|Al\s*5052(?:-H32)?)\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


def infer_preset_hint(
    architecture: str = "corexy",
    printer_model: str = "",
    frame_profile: str = "",
    drive_mode: str = "",
    belt_width_mm: float | None = None,
    panel_spec: str = "",
) -> str:
    """Suggest the closest built-in PrintSim example YAML preset name."""
    arch = (architecture or "corexy").lower()
    model = (printer_model or "").lower()
    prof = (frame_profile or "").lower()
    drive = (drive_mode or "").lower()
    panel = (panel_spec or "").lower()

    if "flsun" in model or (
        arch == "delta"
        and (
            any(tok in model for tok in ("v400", "super racer", "direct-drive", "direct drive"))
            or bool(re.search(r"\b(?:v400|sr|t1|s1)\b", model))
            or (
                belt_width_mm is not None
                and belt_width_mm >= 9.0
                and not any(tok in model for tok in ("doron", "velta", "kossel"))
            )
        )
    ):
        return "flsun_v400_delta"
    if arch == "delta" or any(tok in model for tok in ("doron", "velta", "kossel")):
        return "delta_racer"
    if arch == "cartesian_bedslinger":
        return "cartesian_bedslinger"
    if "vzbot" in model or "vz330" in model or "vz235" in model:
        return "vzbot_330_awd"
    if "doom" in model or "4040" in prof:
        return "doomcube_4040"
    if "ratrig" in model or "v-core" in model or "vcore" in model:
        return "ratrig_vcore31_300"
    if drive == "awd" or "monolith" in model or (belt_width_mm is not None and belt_width_mm >= 9.0 and "3030" not in prof):
        return "voron_monolith_awd"
    if "3030" in prof:
        return "hyper_corexy_carbon"
    if "2.4" in model or "v2.4" in model:
        return "corexy_voron_style"
    if "trident" in model or re.search(r"\bvt\s*\d", model):
        return "voron_trident_300"
    if "5052" in panel or "steel" in panel:
        return "voron_monolith_awd"
    return "corexy_voron_style"


def _extract_shaper_axis(text: str, axis: str) -> tuple[str, float | None, float | None]:
    """Extract (shaper_type, freq_hz, max_accel) for axis 'x' or 'y'."""
    ax = axis.lower()
    shaper_type = ""
    freq_hz: float | None = None
    max_accel: float | None = None

    # Pattern 1: Klipper SHAPER_CALIBRATE style:
    # "Recommended shaper_type_x = mzv, shaper_freq_x = 64.2 Hz"
    for m1 in re.finditer(
        rf"shaper_type_{ax}\s*[=:]\s*([a-z0-9_]+)\s*[,;]?\s*shaper_freq_{ax}\s*[=:]\s*(\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    ):
        cand_hz = float(m1.group(2))
        if _is_valid_shaper_hz(cand_hz):
            shaper_type = m1.group(1).lower()
            freq_hz = cand_hz
            break

    # Pattern 2: Compact axis prefix:
    # "X: MZV @ 86.8 Hz (<=22000)" / "Y: smooth_zv @ 122 Hz (50k)" / "X: Raw ~136 Hz (46k)"
    if freq_hz is None:
        for m2 in re.finditer(
            rf"(?:^|[\s|,;(\n]){ax}(?:\s*(?:axis|shaper))?\s*[:=-]\s*`?"
            rf"(smooth_mzv|smooth_zv|2hump_ei|3hump_ei|mzv|zvd|zv|ei|raw)?`?\s*"
            rf"(?:@|at)?\s*~?\s*\**(\d+(?:\.\d+)?)\s*hz\**"
            rf"(?:\s*\(\s*(?:<=|≤|~)?\s*(\d+(?:\.\d+)?)\s*(k|mm/s\^?2|mm/s²)?\s*\))?",
            text,
            re.IGNORECASE | re.MULTILINE,
        ):
            cand_hz = float(m2.group(2))
            if not _is_valid_shaper_hz(cand_hz):
                continue
            shaper_type = (m2.group(1) or "").lower()
            freq_hz = cand_hz
            if m2.group(3):
                max_accel = _parse_accel_number(m2.group(3), m2.group(4))
            break

    # Pattern 3: Axis shaper with accel only (no Hz): "X: MZV (~27k)"
    if freq_hz is None and max_accel is None:
        m3 = re.search(
            rf"(?:^|[\s|,;(\n]){ax}(?:\s*(?:axis|shaper))?\s*[:=-]\s*`?"
            rf"(smooth_mzv|smooth_zv|2hump_ei|3hump_ei|mzv|zvd|zv|ei)`?\s*"
            rf"\(\s*(?:<=|≤|~)?\s*(\d+(?:\.\d+)?)\s*(k|mm/s\^?2|mm/s²)?\s*\)",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        if m3:
            shaper_type = m3.group(1).lower()
            max_accel = _parse_accel_number(m3.group(2), m3.group(3))

    # Look for nearby suggested max_accel for this axis if not already captured
    if max_accel is None:
        m_acc = re.search(
            rf"(?:{ax}\s*(?:max_accel|accel)\s*(?:<=|≤|=|:)?\s*~?\s*(\d+(?:\.\d+)?)\s*(k|mm/s(?:\^?2|²)|mm/sec\^?2)?)",
            text,
            re.IGNORECASE,
        )
        if m_acc:
            max_accel = _parse_accel_number(m_acc.group(1), m_acc.group(2))

    # Also handle Klipper single-axis output: "Recommended shaper is mzv @ 64.2 Hz" + "suggested max_accel <= 12100"
    if freq_hz is None:
        axis_hint = re.search(rf"\b(?:axis[_\s]*{ax}|resonances_{ax}|{ax}\s*axis\s*calibration)\b", text, re.IGNORECASE)
        if axis_hint:
            m_rec = re.search(
                r"recommended\s+shaper\s+is\s+(smooth_mzv|smooth_zv|2hump_ei|3hump_ei|mzv|zvd|zv|ei)\s*@\s*(\d+(?:\.\d+)?)\s*hz",
                text,
                re.IGNORECASE,
            )
            if m_rec:
                cand_hz = float(m_rec.group(2))
                if _is_valid_shaper_hz(cand_hz):
                    shaper_type = m_rec.group(1).lower()
                    freq_hz = cand_hz
            m_sug = re.search(
                r"suggested\s+max_accel\s*(?:<=|≤)\s*(\d+(?:\.\d+)?)\s*(k|mm/s(?:\^?2|²)|mm/sec\^?2)?",
                text,
                re.IGNORECASE,
            )
            if m_sug and max_accel is None:
                max_accel = _parse_accel_number(m_sug.group(1), m_sug.group(2))

    if not _is_valid_shaper_hz(freq_hz):
        freq_hz = None

    return (shaper_type, freq_hz, max_accel)


def _classify_weight_category(name: str) -> str:
    low = name.lower()
    if any(k in low for k in ("hotend", "chube", "goliath", "rapido", "dragon", "revo", "mosquito")):
        return "hotend"
    if any(k in low for k in ("extruder", "clockwork", "cw2", "orbiter", "sherpa", "galileo")):
        return "extruder"
    if any(k in low for k in ("rail", "mgn12", "mgn9", "mgn15")):
        return "rail"
    if any(k in low for k in ("carriage", "xy-joint", "xy joint", "shuttle")):
        return "carriage"
    if any(k in low for k in ("gantry", "x-beam", "x beam", "cf tube")):
        return "gantry"
    if any(k in low for k in ("bed", "mic-6", "atp-5", "build plate")):
        return "bed"
    if any(k in low for k in ("frame", "panel", "extrusion")):
        return "frame"
    return "toolhead"


_WEIGHT_EXTRACT_PATTERNS = [
    re.compile(
        r"\b(?P<name>(?:Voron\s+)?(?:StealthBurner|XOL(?:\s*2)?|A4T|Dragon\s*Burner|Archetype(?:\s*CPAP)?|"
        r"Goliath(?:\s*Air)?|Chube(?:\s*Conduction)?|Rapido(?:\s*2)?|MGN12[HC]?|MGN9[HC]?|"
        r"Clockwork\s*2|CW2|Orbiter\s*[\d.]*|Sherpa\s*\w*|"
        r"(?:[XY]-?)?gantry|toolhead|carriage|extruder|hotend|x-?beam|bed\s*plate)"
        r"(?:\s+(?:toolhead|hotend|rail|carriage|gantry|assembly|total))?)"
        r"\s*(?:\([^)]{1,30}\))?\s*"
        r"(?::|=|is|weighs|weighed|at|mass(?:\s*of)?)\s*~?\s*"
        r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>kg|g|grams?)\b",
        re.IGNORECASE,
    ),
]


def extract_structured_records(
    text: str,
    title: str = "",
    meta: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Deterministically extract candidate builds, component weights, material tests, and troubleshooting cases."""
    meta = meta or {}
    combined = f"{title}\n{text}" if title else text

    builds: list[dict[str, Any]] = []
    component_weights: list[dict[str, Any]] = []
    material_tests: list[dict[str, Any]] = []
    troubleshooting_cases: list[dict[str, Any]] = []

    # 1. Build extraction
    x_type, x_hz, x_accel = _extract_shaper_axis(combined, "x")
    y_type, y_hz, y_accel = _extract_shaper_axis(combined, "y")

    if any(v is not None for v in (x_hz, y_hz, x_accel, y_accel)):
        arch = _infer_architecture(combined)
        model = _infer_printer_model(combined)
        frame_prof = _infer_frame_profile(combined)
        frame_size = _infer_frame_size_mm(combined)
        drive_mode = _infer_drive_mode(combined)
        belt_w = _infer_belt_width_mm(combined)
        belt_core = _infer_belt_core(combined)
        belt_pluck = _infer_belt_pluck_hz(combined)
        voltage = _infer_voltage(combined)
        toolhead = _infer_toolhead(combined)
        hotend = _infer_hotend(combined)
        motors = _infer_motors(combined)
        panel_spec = _infer_panel_spec(combined)

        x_mass_m = re.search(r"x\s*(?:moving\s*)?mass\s*[:=]?\s*~?\s*(\d+(?:\.\d+)?)\s*g\b", combined, re.IGNORECASE)
        y_mass_m = re.search(r"y\s*(?:moving\s*)?mass\s*[:=]?\s*~?\s*(\d+(?:\.\d+)?)\s*g\b", combined, re.IGNORECASE)

        author = str(meta.get("author") or "").strip()
        build_name = title.strip() or (f"{author} {model}".strip() if author or model else "Community Build")
        if len(build_name) > 90:
            build_name = build_name[:87] + "..."

        if not is_malformed_build_name(build_name):
            builds.append(
                {
                    "build_name": build_name,
                    "architecture": arch,
                    "printer_model": model,
                    "frame_profile": frame_prof,
                    "frame_size_mm": frame_size,
                    "panel_spec": panel_spec,
                    "drive_mode": drive_mode,
                    "belt_width_mm": belt_w,
                    "belt_core": belt_core,
                    "belt_pluck_hz": belt_pluck,
                    "motors": motors,
                    "voltage_v": voltage,
                    "toolhead": toolhead,
                    "hotend": hotend,
                    "x_shaper_type": x_type,
                    "x_shaper_hz": x_hz,
                    "x_max_accel": x_accel,
                    "y_shaper_type": y_type,
                    "y_shaper_hz": y_hz,
                    "y_max_accel": y_accel,
                    "x_moving_mass_g": float(x_mass_m.group(1)) if x_mass_m else None,
                    "y_moving_mass_g": float(y_mass_m.group(1)) if y_mass_m else None,
                    "preset_hint": infer_preset_hint(
                        architecture=arch,
                        printer_model=model,
                        frame_profile=frame_prof,
                        drive_mode=drive_mode,
                        belt_width_mm=belt_w,
                        panel_spec=panel_spec,
                    ),
                    "notes": text[:280].strip().replace("\n", " "),
                }
            )

    # 2. Component weights extraction
    seen_weights: set[tuple[str, float]] = set()
    for pat in _WEIGHT_EXTRACT_PATTERNS:
        for m in pat.finditer(combined):
            cname = re.sub(r"\s+", " ", m.group("name")).strip()
            val = float(m.group("val"))
            unit = m.group("unit").lower()
            mass_g = round(val * 1000.0 if unit == "kg" else val, 2)
            if mass_g <= 0 or mass_g > 50000:
                continue
            key = (cname.lower(), mass_g)
            if key in seen_weights:
                continue
            seen_weights.add(key)
            # Capture short context snippet
            start = max(0, m.start() - 25)
            end = min(len(combined), m.end() + 45)
            snippet = combined[start:end].strip().replace("\n", " ")
            component_weights.append(
                {
                    "category": _classify_weight_category(cname),
                    "component_name": cname,
                    "mass_g": mass_g,
                    "details": snippet,
                }
            )

    # 3. Material tests extraction (line-by-line scan for quantitative material lines)
    seen_mats: set[str] = set()
    for raw_line in combined.splitlines():
        line = raw_line.strip(" -*•\t")
        if len(line) < 12:
            continue
        mat_m = re.search(
            r"\b(Gates\s+GT[23]\s+[\w\s()+-]{2,25}|Rubber\s*\+\s*Aramid(?:\s*\(Kevlar\))?(?:\s*2GT)?|"
            r"TPU\s*\+\s*Kevlar(?:\s*2GT)?|PPA-?CF|PPA-?GF|PPS-?CF|PET-?CF|PA12-?CF|PA-?CF|"
            r"Al(?:uminum)?\s*5052(?:-H32)?|304\s*Steel|Mild(?:/304)?\s*Steel|ACM(?:\s*Dibond)?|"
            r"Polycarbonate(?:\s*/\s*Acrylic)?)\b",
            line,
            re.IGNORECASE,
        )
        if not mat_m:
            continue
        mod_m = re.search(r"(\d+(?:\.\d+)?)\s*gpa\b", line, re.IGNORECASE)
        hdt_m = re.search(r"(?:hdt|temp(?:s)?|at)\s*(?:of|:|=|@)?\s*~?\s*(\d+(?:\.\d+)?)\s*(?:[-–]\s*\d+)?\s*°?\s*c\b", line, re.IGNORECASE)
        creep_m = re.search(r"(\d+(?:\.\d+)?)\s*(n|lbf)\b", line, re.IGNORECASE)
        shaper_m = re.search(r"\b(\d+(?:\.\d+)?)\s*hz\s*(?:zv|mzv|ei|smooth_zv|smooth_mzv)\b", line, re.IGNORECASE)
        pluck_m = re.search(r"@\s*(\d+(?:\.\d+)?)\s*hz\s*pluck\b", line, re.IGNORECASE)

        if not any((mod_m, hdt_m, shaper_m, pluck_m)):
            continue

        mat_name = re.sub(r"\s+", " ", mat_m.group(1)).strip(" :;-")
        if mat_name.lower() in seen_mats:
            continue
        seen_mats.add(mat_name.lower())

        creep_n: float | None = None
        if creep_m:
            cval = float(creep_m.group(1))
            creep_n = round(cval * 4.44822 if creep_m.group(2).lower() == "lbf" else cval, 2)

        material_tests.append(
            {
                "material_name": mat_name,
                "modulus_gpa": float(mod_m.group(1)) if mod_m else None,
                "hdt_c": float(hdt_m.group(1)) if hdt_m else None,
                "creep_limit_n": creep_n,
                "belt_pluck_hz": float(pluck_m.group(1)) if pluck_m else None,
                "shaper_hz": float(shaper_m.group(1)) if shaper_m else None,
                "test_notes": line[:260],
            }
        )

    # 4. Troubleshooting cases extraction
    sym_m = _TROUBLE_SYMPTOM_RE.search(combined)
    fix_m = _TROUBLE_FIX_RE.search(combined)
    if sym_m and fix_m:
        # Extract sentence containing cause/fix
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", combined) if s.strip()]
        cause_sentence = ""
        fix_sentence = ""
        _action_fix_re = re.compile(
            r"\b(?:fixed|solved|eliminated|resolved|replaced\s*with|switched\s*to|"
            r"upgraded\s*to|tightened|bolted|removed\s*foam|verified\s*fix)\b",
            re.IGNORECASE,
        )
        for sent in sentences:
            if not cause_sentence and re.search(
                r"\b(?:caused\s*by|root\s*cause|due\s*to|because)\b", sent, re.IGNORECASE
            ):
                cause_sentence = sent[:240]
            if not fix_sentence and _action_fix_re.search(sent):
                fix_sentence = sent[:240]
        if not cause_sentence:
            for sent in sentences:
                if re.search(
                    r"\b(?:decouples|creep|loses\s*tension|fatigue)\b", sent, re.IGNORECASE
                ):
                    cause_sentence = sent[:240]
                    break
        if not fix_sentence:
            for sent in sentences:
                if _TROUBLE_FIX_RE.search(sent):
                    fix_sentence = sent[:240]
                    break
        hz_sig = ", ".join(f"{h} Hz" for h in _HZ_RE.findall(combined)[:3])
        troubleshooting_cases.append(
            {
                "printer_class": _infer_printer_model(combined) or _infer_architecture(combined),
                "symptom": sym_m.group(0).strip(),
                "measured_signature": hz_sig,
                "root_cause": cause_sentence or (sentences[0][:220] if sentences else ""),
                "verified_fix": fix_sentence or (sentences[-1][:220] if sentences else ""),
            }
        )

    return {
        "builds": builds,
        "component_weights": component_weights,
        "material_tests": material_tests,
        "troubleshooting_cases": troubleshooting_cases,
    }


# ---------------------------------------------------------------------------
# Klipper / Shake&Tune Shaper CSV Parser
# ---------------------------------------------------------------------------

def parse_klipper_shaper_csv(csv_path: Path) -> dict[str, Any]:
    """Parse Klipper ``resonances_*.csv`` or ``calibration_data_*.csv`` files.

    Detects ``freq`` + ``psd_x``/``psd_y``/``psd_z``/``psd_xyz`` columns, computes the
    dominant peak frequency (Hz), peak PSD magnitude, secondary peaks, and axis (X/Y).
    """
    path = Path(csv_path)
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    header_idx: int | None = None
    header_cols: list[str] = []
    for idx, line in enumerate(raw_lines):
        stripped = line.strip()
        if not stripped:
            continue
        candidate = stripped.lstrip("#").strip()
        parts = [p.strip() for p in candidate.split(",")]
        lower_parts = [p.lower() for p in parts]
        if "freq" in lower_parts and any(c.startswith("psd") for c in lower_parts):
            header_idx = idx
            header_cols = parts
            break

    if header_idx is None:
        raise ValueError(f"Not a recognized Klipper shaper CSV (missing freq/psd columns): {path}")

    lower_cols = [c.lower() for c in header_cols]
    freq_col = lower_cols.index("freq")

    psd_xyz_col = lower_cols.index("psd_xyz") if "psd_xyz" in lower_cols else None
    psd_x_col = lower_cols.index("psd_x") if "psd_x" in lower_cols else None
    psd_y_col = lower_cols.index("psd_y") if "psd_y" in lower_cols else None
    psd_z_col = lower_cols.index("psd_z") if "psd_z" in lower_cols else None

    # Parse any Klipper fitted shaper columns like "mzv(64.2)" or "zv(72.0)"
    fitted_shapers: dict[str, float] = {}
    for col in header_cols:
        sm = re.match(r"^([a-z0-9_]+)\s*\(\s*(\d+(?:\.\d+)?)\s*\)$", col.strip(), re.IGNORECASE)
        if sm:
            fitted_shapers[sm.group(1).lower()] = float(sm.group(2))

    freqs: list[float] = []
    psds: list[float] = []
    psd_x_vals: list[float] = []
    psd_y_vals: list[float] = []

    reader = csv.reader(raw_lines[header_idx + 1 :])
    for row in reader:
        if not row or len(row) <= freq_col:
            continue
        first = row[0].strip()
        if not first or first.startswith("#"):
            continue
        try:
            f = float(row[freq_col].strip())
        except ValueError:
            continue
        if not math.isfinite(f):
            continue

        px = float(row[psd_x_col]) if psd_x_col is not None and len(row) > psd_x_col else 0.0
        py = float(row[psd_y_col]) if psd_y_col is not None and len(row) > psd_y_col else 0.0
        pz = float(row[psd_z_col]) if psd_z_col is not None and len(row) > psd_z_col else 0.0
        if psd_xyz_col is not None and len(row) > psd_xyz_col:
            pxyz = float(row[psd_xyz_col])
        else:
            pxyz = px + py + pz

        freqs.append(f)
        psds.append(pxyz)
        psd_x_vals.append(px)
        psd_y_vals.append(py)

    if not freqs:
        raise ValueError(f"Klipper shaper CSV has no numeric rows: {path}")

    # Determine axis from filename first, else from dominant PSD energy
    fname_low = path.name.lower()
    if re.search(r"(?:^|[_-])x(?:[_.-]|$)", fname_low):
        axis = "X"
    elif re.search(r"(?:^|[_-])y(?:[_.-]|$)", fname_low):
        axis = "Y"
    elif max(psd_x_vals, default=0.0) >= max(psd_y_vals, default=0.0):
        axis = "X"
    else:
        axis = "Y"

    # Primary peak
    peak_idx = max(range(len(psds)), key=lambda i: psds[i])
    peak_freq_hz = round(freqs[peak_idx], 2)
    peak_psd = psds[peak_idx]

    # Local maxima for secondary peaks (>= 15% of primary peak, >= 5 Hz separation)
    local_peaks: list[tuple[float, float]] = []
    for i in range(1, len(psds) - 1):
        if psds[i] > psds[i - 1] and psds[i] >= psds[i + 1]:
            if i != peak_idx and peak_psd > 0 and psds[i] >= 0.15 * peak_psd:
                local_peaks.append((freqs[i], psds[i]))
    local_peaks.sort(key=lambda t: t[1], reverse=True)

    secondary_peaks: list[dict[str, float]] = []
    accepted_freqs = [peak_freq_hz]
    for f_cand, p_cand in local_peaks:
        if all(abs(f_cand - af) >= 5.0 for af in accepted_freqs):
            accepted_freqs.append(f_cand)
            secondary_peaks.append(
                {
                    "freq_hz": round(f_cand, 2),
                    "psd": round(p_cand, 6),
                    "ratio_to_primary": round(p_cand / peak_psd, 3) if peak_psd > 0 else 0.0,
                }
            )
        if len(secondary_peaks) >= 4:
            break

    return {
        "file": str(path),
        "filename": path.name,
        "axis": axis,
        "peak_freq_hz": peak_freq_hz,
        "peak_psd": round(peak_psd, 6),
        "secondary_peaks": secondary_peaks,
        "fitted_shapers": fitted_shapers,
        "min_freq_hz": round(min(freqs), 2),
        "max_freq_hz": round(max(freqs), 2),
        "sample_count": len(freqs),
    }
