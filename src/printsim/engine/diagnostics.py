"""Compact engineering diagnostics and A/B delta comparison for PrintSim printers."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import inspect
from pathlib import Path
import re
from typing import Any, Mapping, Sequence, get_args, get_origin

from pydantic import BaseModel
import yaml

from printsim.engine.evaluator import SimulationReport, evaluate_printer
from printsim.schema import PrinterSpec

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
EXAMPLES_DIR: Path = PROJECT_ROOT / "examples"

_OVERRIDE_KEY_ALIASES: dict[str, str] = {
    "toolhead.hotend_brace": "toolhead.use_hotend_bottom_brace",
}

_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+|\d+[eE][+-]?\d+|\d+\.\d*[eE][+-]?\d+)$")


def _parse_override_scalar(raw_val: str) -> Any:
    """Parse CLI scalar override text into bool, int, float, list/dict, None, or str."""
    s = raw_val.strip()
    if s == "":
        return ""
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "~"):
        return None
    if _INT_RE.fullmatch(s):
        return int(s)
    if _FLOAT_RE.fullmatch(s):
        return float(s)
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            return yaml.safe_load(s)
        except yaml.YAMLError:
            return s
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        try:
            return yaml.safe_load(s)
        except yaml.YAMLError:
            return s[1:-1]
    return s


def parse_overrides_input(
    overrides: Mapping[str, Any] | Sequence[str] | None,
) -> dict[str, Any]:
    """Normalize dict or ['dot.path=value', ...] overrides into a typed dictionary."""
    if overrides is None:
        return {}
    if isinstance(overrides, Mapping):
        raw_items = list(overrides.items())
    elif isinstance(overrides, (list, tuple)):
        raw_items = []
        for entry in overrides:
            if not isinstance(entry, str) or "=" not in entry:
                raise ValueError(
                    f"Invalid override entry {entry!r}: expected 'dot.path=value' string."
                )
            k, v = entry.split("=", 1)
            k_clean = k.strip()
            if not k_clean:
                raise ValueError(f"Empty override key in {entry!r}")
            raw_items.append((k_clean, _parse_override_scalar(v)))
    else:
        raise TypeError(
            f"overrides must be a mapping or sequence of 'key=value' strings, got {type(overrides).__name__}"
        )

    out: dict[str, Any] = {}
    for raw_key, val in raw_items:
        key = str(raw_key).strip()
        if not key:
            raise ValueError(f"Invalid override key: {raw_key!r}")
        if key in _OVERRIDE_KEY_ALIASES:
            alias = _OVERRIDE_KEY_ALIASES[key]
            # Explicit canonical key wins if both are supplied
            if alias not in out:
                key = alias
            else:
                continue
        out[key] = val
    return out


def _unwrap_submodel(annotation: Any) -> type[BaseModel] | None:
    """Return the nested BaseModel class for a field annotation, if any."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    origin = get_origin(annotation)
    if origin is not None:
        for arg in get_args(annotation):
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                return arg
    return None


def _validate_override_path(key: str, value: Any) -> None:
    """Validate dot-path segments against PrinterSpec fields (allowing catalog preset shortcuts)."""
    if key == "kinematics.xy_stepper" and isinstance(value, str):
        return
    if key == "toolhead" and isinstance(value, str):
        return
    if key in ("toolhead.hotend", "toolhead.hotend.preset_id") and isinstance(value, str):
        return

    parts = [p for p in key.split(".") if p]
    if not parts:
        raise ValueError(f"Invalid override key: {key!r}")

    cur_model: type[BaseModel] = PrinterSpec
    for idx, part in enumerate(parts):
        if part not in cur_model.model_fields:
            raise ValueError(
                f"Unknown PrinterSpec field '{'.'.join(parts[: idx + 1])}' in override '{key}'"
            )
        if idx < len(parts) - 1:
            ann = cur_model.model_fields[part].annotation
            next_model = _unwrap_submodel(ann)
            if next_model is not None:
                cur_model = next_model
                continue
            origin = get_origin(ann)
            if ann is dict or origin is dict:
                return
            raise ValueError(
                f"Cannot traverse into non-nested field '{'.'.join(parts[: idx + 1])}' in override '{key}'"
            )


def apply_dot_overrides(
    spec: PrinterSpec,
    overrides: Mapping[str, Any] | Sequence[str] | None,
) -> PrinterSpec:
    """Return a validated copy of ``spec`` with dot-path ``overrides`` applied.

    Supports both ``{"gantry.x_beam_profile": "cf_tube_20x20x2"}`` dicts and CLI
    lists like ``["gantry.x_beam_profile=cf_tube_20x20x2", "toolhead.hotend_brace=false"]``.
    """
    parsed = parse_overrides_input(overrides)
    if not parsed:
        return PrinterSpec.model_validate(spec.model_dump(mode="json"))

    for k, v in parsed.items():
        _validate_override_path(k, v)

    base_data = spec.model_dump(mode="json")
    from printsim.mcp_server import _apply_dot_overrides

    updated = _apply_dot_overrides(base_data, parsed)
    return PrinterSpec.model_validate(updated)


def resolve_spec_input(
    spec_or_path: PrinterSpec | Path | str,
    overrides: Mapping[str, Any] | Sequence[str] | None = None,
) -> tuple[PrinterSpec, str | None]:
    """Load a ``PrinterSpec`` from an object, YAML path, or preset ID, applying optional overrides."""
    source_label: str | None = None
    if isinstance(spec_or_path, PrinterSpec):
        spec = PrinterSpec.model_validate(spec_or_path.model_dump(mode="json"))
    else:
        raw_path = Path(spec_or_path)
        candidates = [
            raw_path,
            PROJECT_ROOT / raw_path,
            EXAMPLES_DIR / f"{spec_or_path}.yaml",
        ]
        resolved: Path | None = None
        for cand in candidates:
            if cand.is_file():
                resolved = cand
                break
        if resolved is None:
            raise FileNotFoundError(
                f"Printer spec YAML not found: {spec_or_path!r} "
                f"(checked cwd, {PROJECT_ROOT}, and {EXAMPLES_DIR})"
            )
        source_label = resolved.stem
        spec = PrinterSpec.from_yaml(resolved)

    if overrides:
        spec = apply_dot_overrides(spec, overrides)
    return spec, source_label


def _run_evaluate_printer(spec: PrinterSpec, *, include_gcode: bool) -> SimulationReport:
    """Invoke ``evaluate_printer`` while staying compatible with optional ``include_gcode`` kwargs."""
    sig = inspect.signature(evaluate_printer)
    if "include_gcode" in sig.parameters:
        return evaluate_printer(spec, include_gcode=include_gcode)  # type: ignore[call-arg]
    return evaluate_printer(spec)


def _shaper_summary(shaper: Any) -> dict[str, Any]:
    if shaper is None:
        return {
            "type": "none",
            "freq_hz": 0.0,
            "max_accel_mm_s2": 0.0,
            "limiting_cause": "none",
        }
    return {
        "type": str(shaper.recommended_shaper),
        "freq_hz": round(float(shaper.effective_freq_hz), 2),
        "max_accel_mm_s2": round(float(shaper.max_accel_mm_s2), 0),
        "limiting_cause": str(shaper.primary_bottleneck),
    }


def _extract_toolhead_breakdown(report: SimulationReport) -> dict[str, Any]:
    geom_dict = report.assembly.to_geometry_dict()
    th_struct = geom_dict.get("toolhead_structure") or {}
    flex = report.static_fea.toolhead_flex_breakdown or {}
    stages = th_struct.get("stage_stiffnesses_n_per_mm") or {}

    eff_mass_g = float(
        flex.get("assembled_mass_g", th_struct.get("assembled_toolhead_mass_g", 0.0))
    )
    cog_mm = list(
        flex.get("effective_cog_offset_mm", th_struct.get("effective_cog_offset_mm", [0.0, 0.0, 0.0]))
    )
    nozzle_drop_mm = float(
        flex.get("nozzle_drop_mm", abs(float(report.spec.toolhead.nozzle_offset_mm[2])))
    )
    return {
        "effective_mass_g": round(eff_mass_g, 1),
        "cog_offset_mm": [round(float(v), 2) for v in cog_mm],
        "nozzle_drop_mm": round(nozzle_drop_mm, 1),
        "effective_tip_stiffness_n_per_mm": round(
            float(flex.get("effective_tip_stiffness_n_per_mm", 0.0)), 2
        ),
        "rail_carriage_stiffness_n_per_mm": round(
            float(
                flex.get(
                    "rail_carriage_stiffness_n_per_mm",
                    stages.get("rail_carriage_n_per_mm", 0.0),
                )
            ),
            2,
        ),
        "carriage_mount_stiffness_n_per_mm": round(
            float(
                flex.get(
                    "carriage_mount_stiffness_n_per_mm",
                    stages.get("carriage_mount_n_per_mm", 0.0),
                )
            ),
            2,
        ),
        "body_stiffness_n_per_mm": round(
            float(flex.get("body_stiffness_n_per_mm", stages.get("body_n_per_mm", 0.0))),
            2,
        ),
        "hotend_stiffness_n_per_mm": round(
            float(
                flex.get(
                    "hotend_stiffness_n_per_mm",
                    th_struct.get("hotend_stiffness_n_per_mm", 0.0),
                )
            ),
            2,
        ),
        "dominant_flex_source": str(flex.get("dominant_flex_source", "unknown")),
        "compliance_share_pct": {
            "rail_and_xbeam": round(float(flex.get("rail_and_xbeam_pct", 0.0)), 2),
            "rail_carriage": round(float(flex.get("rail_carriage_pct", 0.0)), 2),
            "toolhead_body": round(float(flex.get("toolhead_body_pct", 0.0)), 2),
            "hotend_nozzle": round(float(flex.get("hotend_and_nozzle_pct", 0.0)), 2),
        },
    }


def _truncate(text: str, max_len: int) -> str:
    clean = " ".join(str(text).split()).replace("|", "/")
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 1].rstrip() + "…"


@dataclass
class PrinterDiagnosis:
    """Compact multiphysics diagnosis summary for fast CLI and agent consumption."""

    name: str
    architecture: str
    overall_score: float
    total_mass_kg: float
    moving_mass_x_g: float
    moving_mass_y_g: float
    bom_cost_usd: float
    max_volumetric_flow_mm3_s: float
    fundamental_freq_hz: float
    shaper_x: dict[str, Any]
    shaper_y: dict[str, Any]
    recommended_max_accel_mm_s2: float
    static_tip_deflection_um: dict[str, float]
    static_stiffness_n_per_mm: dict[str, float]
    toolhead_breakdown: dict[str, Any]
    top_modes: list[dict[str, Any]] = field(default_factory=list)
    top_adjoint_hotspots: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    virtual_print: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        """Render a compact <= 45-line Markdown engineering summary."""
        sx = self.shaper_x
        sy = self.shaper_y
        defl = self.static_tip_deflection_um
        stiff = self.static_stiffness_n_per_mm
        th = self.toolhead_breakdown
        cog = th.get("cog_offset_mm") or [0.0, 0.0, 0.0]

        lines: list[str] = [
            f"# Diagnosis: {self.name} ({self.architecture})",
            (
                f"Score: **{self.overall_score:.1f}/100** | "
                f"Mass: {self.total_mass_kg:.2f} kg (X {self.moving_mass_x_g:.0f}g / Y {self.moving_mass_y_g:.0f}g) | "
                f"BOM: ${self.bom_cost_usd:.0f} | Flow: {self.max_volumetric_flow_mm3_s:.1f} mm³/s"
            ),
            (
                f"Fundamental: **{self.fundamental_freq_hz:.2f} Hz** | "
                f"Rec Max Accel: **{self.recommended_max_accel_mm_s2:.0f} mm/s²**"
            ),
            (
                f"- Shaper X: {sx['type']} @ {sx['freq_hz']:.1f} Hz ({sx['max_accel_mm_s2']:.0f} mm/s², "
                f"limit: {sx['limiting_cause']}) | "
                f"Y: {sy['type']} @ {sy['freq_hz']:.1f} Hz ({sy['max_accel_mm_s2']:.0f} mm/s², "
                f"limit: {sy['limiting_cause']})"
            ),
            (
                f"- Deflection (µm): X={defl['x']:.1f}, Y={defl['y']:.1f}, Z={defl['z']:.1f}, "
                f"Worst={defl['worst']:.1f} | "
                f"Stiffness (N/mm): X={stiff['x']:.1f}, Y={stiff['y']:.1f}, Z={stiff['z']:.1f}"
            ),
            (
                f"- Toolhead: {th.get('effective_mass_g', 0.0):.0f}g, "
                f"CoG=({cog[0]:+.1f},{cog[1]:+.1f},{cog[2]:+.1f})mm, drop={th.get('nozzle_drop_mm', 0.0):.0f}mm | "
                f"k_tip={th.get('effective_tip_stiffness_n_per_mm', 0.0):.1f}, "
                f"k_rail={th.get('rail_carriage_stiffness_n_per_mm', 0.0):.0f}, "
                f"k_body={th.get('body_stiffness_n_per_mm', 0.0):.0f}, "
                f"k_hot={th.get('hotend_stiffness_n_per_mm', 0.0):.0f} N/mm "
                f"(dom: {th.get('dominant_flex_source', 'n/a')})"
            ),
        ]
        if self.virtual_print is not None:
            vp = self.virtual_print
            lines.append(
                f"- Virtual Print: score={vp.get('print_quality_score', 0.0):.1f}, "
                f"time={vp.get('total_time_s', 0.0):.1f}s, "
                f"ringing={vp.get('ringing_segments', 0)}, underextr={vp.get('underextrusion_segments', 0)}"
            )

        if self.top_modes:
            lines.extend(
                [
                    "",
                    "## Top Modes",
                    "| # | Freq (Hz) | Axis | Description | Root Cause |",
                    "|---:|---:|:---:|---|---|",
                ]
            )
            for m in self.top_modes[:6]:
                lines.append(
                    f"| {m['mode_num']} | {m['freq_hz']:.1f} | {m['dominant_axis']} | "
                    f"{_truncate(m['description'], 36)} | {_truncate(m['root_cause'], 40)} |"
                )

        if self.top_adjoint_hotspots:
            lines.extend(
                [
                    "",
                    "## Adjoint Hotspots",
                    "| Group | Share (%) | Fix Hint |",
                    "|---|---:|---|",
                ]
            )
            for h in self.top_adjoint_hotspots[:5]:
                lines.append(
                    f"| {_truncate(h['group'], 32)} | {h['compliance_share_pct']:.1f}% | "
                    f"{_truncate(h['hint'], 90)} |"
                )

        if self.diagnostics:
            lines.extend(["", "## Diagnostics"])
            max_diag_lines = max(1, 45 - len(lines))
            shown = self.diagnostics[:max_diag_lines]
            if len(self.diagnostics) > max_diag_lines and max_diag_lines >= 2:
                shown = self.diagnostics[: max_diag_lines - 1]
                for d in shown:
                    lines.append(f"- {_truncate(d, 110)}")
                lines.append(f"- ... (+{len(self.diagnostics) - len(shown)} more)")
            else:
                for d in shown:
                    lines.append(f"- {_truncate(d, 110)}")

        return "\n".join(lines[:45])


def extract_diagnosis_from_report(
    report: SimulationReport,
    *,
    include_gcode: bool = False,
) -> PrinterDiagnosis:
    """Build a ``PrinterDiagnosis`` from an existing ``SimulationReport``."""
    spec = report.spec
    sf = report.static_fea
    mf = report.modal_fea
    asm = report.assembly

    top_modes: list[dict[str, Any]] = []
    for m in mf.modes[:6]:
        if m.top_causes:
            c0 = m.top_causes[0]
            root_cause = f"{c0.get('label', m.dominant_problem_area)} ({float(c0.get('share_pct', 0.0)):.0f}%)"
        else:
            root_cause = str(m.dominant_problem_area)
        top_modes.append(
            {
                "mode_num": int(m.mode_number),
                "freq_hz": round(float(m.frequency_hz), 2),
                "dominant_axis": str(m.dominant_axis),
                "description": str(m.mode_label or m.what_is_shaking or f"Mode {m.mode_number}"),
                "root_cause": root_cause,
            }
        )

    hotspots: list[dict[str, Any]] = []
    if report.sensitivity is not None:
        for target in report.sensitivity.top_stiffen_targets[:5]:
            hotspots.append(
                {
                    "group": str(target.get("label") or target.get("id") or "unknown"),
                    "compliance_share_pct": round(float(target.get("compliance_share_pct", 0.0)), 2),
                    "hint": str(target.get("action_hint") or target.get("why_it_hurts") or ""),
                }
            )

    total_mass_kg = (
        float(report.bom_total_mass_kg)
        if report.bom_total_mass_kg is not None
        else float(asm.total_mass_kg)
    )
    bom_cost_usd = (
        float(report.bom_total_cost_usd)
        if report.bom_total_cost_usd is not None
        else float(asm.estimated_material_cost_usd)
    )

    vp_summary: dict[str, Any] | None = None
    if include_gcode and report.virtual_print is not None:
        vp = report.virtual_print
        vp_summary = {
            "print_quality_score": round(float(vp.print_quality_score), 1),
            "total_time_s": round(float(vp.total_time_s), 1),
            "ringing_segments": int(vp.ringing_segments),
            "underextrusion_segments": int(vp.underextrusion_segments),
            "max_ringing_amplitude_um": round(float(vp.max_ringing_amplitude_um), 2),
        }

    return PrinterDiagnosis(
        name=str(spec.name),
        architecture=str(spec.architecture),
        overall_score=round(float(report.scores.composite), 2),
        total_mass_kg=round(total_mass_kg, 3),
        moving_mass_x_g=round(float(asm.x_moving_mass_kg) * 1000.0, 1),
        moving_mass_y_g=round(float(asm.y_moving_mass_kg) * 1000.0, 1),
        bom_cost_usd=round(bom_cost_usd, 2),
        max_volumetric_flow_mm3_s=round(float(report.thermal.max_volumetric_flow_mm3_s), 1),
        fundamental_freq_hz=round(float(mf.fundamental_freq_hz), 2),
        shaper_x=_shaper_summary(mf.shaper_x),
        shaper_y=_shaper_summary(mf.shaper_y),
        recommended_max_accel_mm_s2=round(float(mf.recommended_max_accel_mm_s2), 0),
        static_tip_deflection_um={
            "x": round(float(sf.accel_deflection_x_um), 2),
            "y": round(float(sf.accel_deflection_y_um), 2),
            "z": round(float(sf.gravity_sag_toolhead_um), 2),
            "worst": round(float(sf.combined_max_deflection_um), 2),
        },
        static_stiffness_n_per_mm={
            "x": round(float(sf.toolhead_stiffness_x_n_per_mm), 2),
            "y": round(float(sf.toolhead_stiffness_y_n_per_mm), 2),
            "z": round(float(sf.toolhead_stiffness_z_n_per_mm), 2),
        },
        toolhead_breakdown=_extract_toolhead_breakdown(report),
        top_modes=top_modes,
        top_adjoint_hotspots=hotspots,
        diagnostics=list(report.diagnostics),
        virtual_print=vp_summary,
    )


def diagnose_printer(
    spec_or_path: PrinterSpec | Path | str,
    overrides: Mapping[str, Any] | Sequence[str] | None = None,
    *,
    include_gcode: bool = False,
) -> PrinterDiagnosis:
    """Evaluate a printer specification and return a compact ``PrinterDiagnosis``."""
    spec, _ = resolve_spec_input(spec_or_path, overrides=overrides)
    report = _run_evaluate_printer(spec, include_gcode=include_gcode)
    return extract_diagnosis_from_report(report, include_gcode=include_gcode)


_COMPARE_METRIC_DEFS: tuple[tuple[str, str, str, int], ...] = (
    ("overall_score", "Overall Score", "pts", 2),
    ("fundamental_freq_hz", "Fundamental Freq", "Hz", 2),
    ("shaper_x_hz", "Shaper X Freq", "Hz", 2),
    ("shaper_y_hz", "Shaper Y Freq", "Hz", 2),
    ("rec_max_accel_mm_s2", "Rec Max Accel", "mm/s²", 0),
    ("static_stiffness_x_n_per_mm", "Static Stiffness X", "N/mm", 2),
    ("static_stiffness_y_n_per_mm", "Static Stiffness Y", "N/mm", 2),
    ("moving_mass_x_g", "Moving Mass X", "g", 1),
    ("moving_mass_y_g", "Moving Mass Y", "g", 1),
    ("total_mass_kg", "Total Mass", "kg", 3),
    ("max_volumetric_flow_mm3_s", "Max Vol Flow", "mm³/s", 1),
    ("bom_cost_usd", "BOM Cost", "USD", 2),
)


def _diag_compare_values(diag: PrinterDiagnosis) -> dict[str, float]:
    return {
        "overall_score": float(diag.overall_score),
        "fundamental_freq_hz": float(diag.fundamental_freq_hz),
        "shaper_x_hz": float(diag.shaper_x["freq_hz"]),
        "shaper_y_hz": float(diag.shaper_y["freq_hz"]),
        "rec_max_accel_mm_s2": float(diag.recommended_max_accel_mm_s2),
        "static_stiffness_x_n_per_mm": float(diag.static_stiffness_n_per_mm["x"]),
        "static_stiffness_y_n_per_mm": float(diag.static_stiffness_n_per_mm["y"]),
        "moving_mass_x_g": float(diag.moving_mass_x_g),
        "moving_mass_y_g": float(diag.moving_mass_y_g),
        "total_mass_kg": float(diag.total_mass_kg),
        "max_volumetric_flow_mm3_s": float(diag.max_volumetric_flow_mm3_s),
        "bom_cost_usd": float(diag.bom_cost_usd),
    }


@dataclass
class PrinterComparison:
    """Side-by-side multiphysics comparison and signed deltas between two printer specs."""

    label_a: str
    label_b: str
    base: PrinterDiagnosis
    variant: PrinterDiagnosis
    metrics: dict[str, dict[str, float | None]]
    overrides: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "overrides": copy.deepcopy(self.overrides),
            "metrics": copy.deepcopy(self.metrics),
            "base": self.base.to_dict(),
            "variant": self.variant.to_dict(),
        }

    def to_markdown(self) -> str:
        lines: list[str] = [
            f"# Comparison: {self.label_a} vs {self.label_b}",
        ]
        if self.overrides:
            ov_str = ", ".join(f"`{k}={v}`" for k, v in self.overrides.items())
            lines.append(f"Overrides: {ov_str}")
        lines.extend(
            [
                "",
                f"| Metric | Unit | {self.label_a} | {self.label_b} | Delta | % Delta |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for key, label, unit, decimals in _COMPARE_METRIC_DEFS:
            row = self.metrics[key]
            va = float(row["base"] or 0.0)
            vb = float(row["variant"] or 0.0)
            d = float(row["delta"] or 0.0)
            pct = row["pct_delta"]
            fmt = f".{decimals}f"
            pct_str = f"{pct:+.2f}%" if pct is not None else "n/a"
            lines.append(
                f"| {label} | {unit} | {va:{fmt}} | {vb:{fmt}} | {d:+{fmt}} | {pct_str} |"
            )
        return "\n".join(lines)


def compare_printers(
    base: PrinterSpec | Path | str,
    variant: PrinterSpec | Path | str | None = None,
    *,
    overrides: Mapping[str, Any] | Sequence[str] | None = None,
    label_a: str | None = None,
    label_b: str | None = None,
    include_gcode: bool = False,
) -> PrinterComparison:
    """Compare a baseline printer against an explicit variant or dot-path overrides."""
    parsed_overrides = parse_overrides_input(overrides)
    base_spec, src_a = resolve_spec_input(base)
    if variant is None:
        variant_spec = apply_dot_overrides(base_spec, parsed_overrides)
        src_b = None
    else:
        variant_spec, src_b = resolve_spec_input(variant, overrides=parsed_overrides or None)

    diag_a = diagnose_printer(base_spec, include_gcode=include_gcode)
    diag_b = diagnose_printer(variant_spec, include_gcode=include_gcode)

    resolved_label_a = label_a or src_a or diag_a.name
    if label_b:
        resolved_label_b = label_b
    elif src_b:
        resolved_label_b = src_b
    elif variant is None and parsed_overrides:
        resolved_label_b = f"{resolved_label_a} + overrides"
    else:
        resolved_label_b = diag_b.name if diag_b.name != resolved_label_a else f"{diag_b.name} (variant)"

    vals_a = _diag_compare_values(diag_a)
    vals_b = _diag_compare_values(diag_b)
    metrics: dict[str, dict[str, float | None]] = {}
    for key, _, _, decimals in _COMPARE_METRIC_DEFS:
        va = vals_a[key]
        vb = vals_b[key]
        delta = round(vb - va, max(decimals, 4))
        if abs(va) > 1e-9:
            pct_delta: float | None = round(((vb - va) / abs(va)) * 100.0, 2)
        elif abs(delta) <= 1e-12:
            pct_delta = 0.0
        else:
            pct_delta = None
        metrics[key] = {
            "base": round(va, max(decimals, 4)),
            "variant": round(vb, max(decimals, 4)),
            "delta": delta,
            "pct_delta": pct_delta,
        }

    return PrinterComparison(
        label_a=resolved_label_a,
        label_b=resolved_label_b,
        base=diag_a,
        variant=diag_b,
        metrics=metrics,
        overrides=parsed_overrides,
    )
