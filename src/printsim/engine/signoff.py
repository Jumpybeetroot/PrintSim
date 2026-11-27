"""High-accuracy Sign-Off Verification engine for PrintSim.

Combines four complementary verification layers on top of the core multiphysics solver:
1. Fine-mesh MITC4 shell & bed-plate convergence (3x3/8x8 vs 8x8/16x16).
2. True multi-position 3D FEA workspace sweep (re-assembling and solving K/M at every
   (u, v, z) point across the build envelope rather than using a 1-point heuristic).
3. Confidence-driven parametric uncertainty (central-difference 3D FEA perturbations of the
   lumped stiffness groups and the toolhead moving mass, each scaled to its Model Confidence
   trust score -> 90% parametric bands & ranked uncertainty drivers). Damping, panel fastener
   compliance, and model-form error are not propagated, so the bands are a lower bound on
   total prediction error.
4. Engineering sign-off readiness checks (mesh, workspace uniformity, uncertainty,
   structural safety, and belt/bed margins).

Strictly pure NumPy/SciPy + PrintSim internals so it runs identically in native Python
and browser Pyodide WASM.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

from printsim.engine.evaluator import (
    SimulationReport,
    evaluate_printer,
    verify_printer_deep,
)
from printsim.schema.overrides import (
    build_effective_model_params,
    build_model_confidence_audit,
)
from printsim.schema.printer import PrinterSpec

# Two-sided 90% normal quantile (5th to 95th percentile).
_Z_90: float = 1.6448536269514722

# Readiness gate limits.
# 90% parametric band on the limiting shaper frequency. Safe accel scales with frequency
# squared, so 5% / 10% here equal +/-10% / +/-20% on safe accel (the workspace gate limits).
_BAND_PASS_PCT: float = 5.0
_BAND_FAIL_PCT: float = 10.0
# Belt safety factor = rated working tension / peak tension. Below 1.0 is the same condition
# as the kinematics solver's belt_overtension_risk flag; 1.0-1.25 is a thin margin.
_BELT_SF_PASS: float = 1.25
_BELT_SF_FAIL: float = 1.0
_BED_FLAT_PASS_UM: float = 50.0
_BED_FLAT_FAIL_UM: float = 120.0

# Accepted fine-mesh densities (same bounds as the API request model). The engine clamps to
# these so the WASM, MCP, and SDK entry points cannot request an unbounded solve.
_DEEP_MESH_N_RANGE: tuple[int, int] = (4, 10)
_DEEP_BED_MESH_N_RANGE: tuple[int, int] = (8, 20)

# Architectures whose assembly builder ignores the XY toolhead position (only Z moves the model).
_XY_INSENSITIVE_ARCHITECTURES: frozenset[str] = frozenset({"delta"})
# Relative Z stations for the centre-column sweep. The delta builder clips Z to 0.1-0.8, so its
# top station is reported at the position that is actually solved.
_Z_SWEEP_STATIONS: tuple[float, ...] = (0.1, 0.5, 0.9)
_Z_SWEEP_STATIONS_DELTA: tuple[float, ...] = (0.1, 0.5, 0.8)

# Perturbation descriptors for each ModelOverridesSpec subgroup.
# Each entry maps override_group -> (display_label, [(field_name, min_val, max_val), ...], advice).
_UNCERTAINTY_GROUP_SPECS: dict[str, tuple[str, list[tuple[str, float, float]], str]] = {
    "toolhead_construction": (
        "Toolhead Body Stiffness",
        [
            ("body_stiffness_base_n_per_mm", 25.0, 50000.0),
            ("torsion_mult", 0.1, 10.0),
        ],
        "Switch to a CAD-baked toolhead preset (e.g. Microwaved Bowden / Chube) or measure static nozzle flex.",
    ),
    "gantry_bracket": (
        "XY-Joint & Motor Brackets",
        [("stiffness_n_per_mm", 20.0, 200000.0)],
        "Use a CAD-backed gantry kit (CNC or Tet10 superelement) or run bracket FEA condensation.",
    ),
    "belt": (
        "XY Drive Belt Modulus",
        [
            ("stiffness_ea_n", 1000.0, 2e6),
            ("tooth_shear_k_n_per_m_mm", 5000.0, 1e6),
        ],
        "Stick to calibrated Gates 2GT 6/9/12 mm belts or run an ADXL345 tension-sweep fit.",
    ),
    "belt_routing": (
        "Belt Path & Idler Stiffness",
        [
            ("idler_stiffness_factor", 0.2, 5.0),
        ],
        "Verify idler bolt stiffness and exact free-span length with an ADXL345 calibration CSV.",
    ),
    "corner_joint": (
        "Frame Corner Joints",
        [
            ("k_t_n_per_mm", 100.0, 1e8),
            ("k_r_inplane_nm_rad", 100.0, 1e9),
            ("k_r_torsion_nm_rad", 100.0, 1e9),
        ],
        "Add bolted structural panels or CNC corner gussets to reduce corner-joint sensitivity.",
    ),
    "frame_feet": (
        "Benchtop Feet / Isolators",
        [
            ("k_axial_n_per_mm", 0.01, 1e6),
            ("k_lateral_n_per_mm", 0.01, 1e6),
        ],
        "Bolt frame rigidly or measure bench-rocking frequency with an ADXL345 sweep.",
    ),
    "z_probe": (
        "Z-Probe / Carriage Interface",
        [("compliance_factor", 0.05, 2.0)],
        "Switch from Printed/CNC Tap to a static eddy probe (Beacon/Cartographer) or run before/after ADXL.",
    ),
    "hotend_clamp": (
        "Hotend Clamp & Neck",
        [("clamp_stiffness_base_n_per_mm", 50.0, 50000.0)],
        "Use a CAD-backed 4/6-bolt rigid hotend mount (e.g. Chube / Dragon) instead of a generic clamp.",
    ),
    "rail_quality": (
        "Linear Rail Grade & Preload",
        [("stiffness_mult", 0.1, 5.0)],
        "Specify genuine HIWIN / matched Grade C/H rails and verify carriage roll play.",
    ),
    "rail_preload": (
        "Linear Rail Preload (Q/C)",
        [("preload_fraction", 0.0005, 0.25)],
        "Confirm actual carriage preload class (Z0 vs Z1/ZA) with the rail vendor.",
    ),
    "bed_mount": (
        "Bed Mount Hardware",
        [
            ("k_axial_n_per_mm", 1.0, 500000.0),
            ("k_lateral_n_per_mm", 1.0, 500000.0),
        ],
        "Measure cold-vs-hot BED_MESH_OUTPUT delta to calibrate bed mount compliance.",
    ),
    "z_drive_mechanism": (
        "Z-Drive Mechanism",
        [
            ("k_nut_n_per_um", 0.5, 500.0),
            ("k_thrust_n_per_um", 0.5, 500.0),
            ("k_drive_n_per_um_per_corner", 0.02, 50.0),
        ],
        "Check vertical bed deflection under a known 10 N nozzle-tap load.",
    ),
}


# Perturbations of plain PrinterSpec fields, keyed by Model Confidence entry id, for uncertain
# inputs that have no ModelOverridesSpec group.
# entry_id -> (display_label, [(dotted_spec_path, min_val, max_val), ...], advice)
_UNCERTAINTY_SPEC_FIELD_SPECS: dict[str, tuple[str, list[tuple[str, float, float]], str]] = {
    "toolhead_preset": (
        "Toolhead Moving Mass",
        [("toolhead.total_mass_g", 20.5, 1.0e5)],
        "Weigh the assembled toolhead (hotend, extruder, fans, probe, wiring stub) and enter the scale mass.",
    ),
}

UNCERTAINTY_COVERAGE_NOTE: str = (
    "Parametric band: propagates lumped stiffness and toolhead-mass uncertainty only. Damping, "
    "panel fastener compliance, and model-form error are not included, so treat it as a lower "
    "bound and confirm critical builds with an ADXL345 measurement."
)


def _extract_core_metrics(report: SimulationReport) -> dict[str, float]:
    """Extract key scalar metrics from a SimulationReport."""
    mf = report.modal_fea
    sx = float(mf.shaper_x.effective_freq_hz) if mf.shaper_x else float(mf.fundamental_freq_hz)
    sy = float(mf.shaper_y.effective_freq_hz) if mf.shaper_y else float(mf.fundamental_freq_hz)
    return {
        "shaper_x_hz": sx,
        "shaper_y_hz": sy,
        "min_axis_hz": min(sx, sy),
        "fundamental_freq_hz": float(mf.fundamental_freq_hz),
        "recommended_max_accel_mm_s2": float(mf.recommended_max_accel_mm_s2),
        "combined_max_deflection_um": float(report.static_fea.combined_max_deflection_um),
    }


def _trust_to_sigma_rel(trust_pct: float, overridden: bool = False) -> float:
    """Convert a 0-100% model trust score into a 1-sigma relative parameter uncertainty.

    Calibration anchors:
      - 95% trust -> ~2.3% (1-sigma), 90% CI ~ ±3.7%
      - 85% trust -> ~6.8% (1-sigma), 90% CI ~ ±11%
      - 70% trust -> ~13.5% (1-sigma), 90% CI ~ ±22%
      - 55% trust -> ~20.3% (1-sigma), 90% CI ~ ±33%
      - 40% trust -> ~27.0% (1-sigma), 90% CI ~ ±44%
    User-overridden parameters halve the remaining uncertainty (down to a 3% floor), and an
    override never raises sigma above the un-overridden value.
    """
    t = max(0.0, min(100.0, float(trust_pct)))
    sigma = max(0.02, min(0.32, 0.45 * (1.0 - t / 100.0)))
    if overridden:
        sigma = min(sigma, max(0.03, sigma * 0.5))
    return sigma


def _apply_group_scale(
    spec: PrinterSpec,
    group_id: str,
    eff_values: dict[str, Any],
    field_specs: list[tuple[str, float, float]],
    scale: float,
) -> PrinterSpec | None:
    """Return a deep copy of spec with `group_id` stiffness fields scaled by `scale`.

    Returns None if none of the target fields have a positive finite baseline value.
    """
    updates: dict[str, float] = {}
    for fname, lo, hi in field_specs:
        raw = eff_values.get(fname)
        if not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            continue
        base_val = float(raw)
        if base_val <= 0.0:
            continue
        new_val = max(lo, min(hi, base_val * scale))
        if abs(new_val - base_val) > 1e-9:
            updates[fname] = new_val

    if not updates:
        return None

    cand = spec.model_copy(deep=True)
    sub_obj = getattr(cand.model_overrides, group_id, None)
    if sub_obj is None:
        return None
    for fname, new_val in updates.items():
        setattr(sub_obj, fname, new_val)
    return cand


def _apply_spec_field_scale(
    spec: PrinterSpec,
    field_specs: list[tuple[str, float, float]],
    scale: float,
) -> PrinterSpec | None:
    """Return a deep copy of spec with the dotted-path numeric fields scaled by `scale`.

    Returns None if none of the target fields have a positive finite baseline value.
    """
    cand = spec.model_copy(deep=True)
    changed = False
    for path, lo, hi in field_specs:
        *parents, leaf = path.split(".")
        obj: Any = cand
        for name in parents:
            obj = getattr(obj, name, None)
            if obj is None:
                break
        raw = getattr(obj, leaf, None) if obj is not None else None
        if not isinstance(raw, (int, float)) or not math.isfinite(float(raw)) or float(raw) <= 0.0:
            continue
        new_val = max(lo, min(hi, float(raw) * scale))
        if abs(new_val - float(raw)) > 1e-9:
            setattr(obj, leaf, new_val)
            changed = True
    return cand if changed else None


def _accel_sigma_from_freq(
    nominal_accel_mm_s2: float,
    accel_x_mm_s2: float,
    accel_y_mm_s2: float,
    rel_x: float,
    rel_y: float,
) -> float:
    """1-sigma safe-accel response from signed 1-sigma relative shaper-frequency responses.

    For a fixed input shaper the acceleration that reaches the target smoothing scales with
    the shaper frequency squared, and the recommendation is the lower of the two axes:
    a(+/-) = min(a_x (1 +/- r_x)^2, a_y (1 +/- r_y)^2). Propagating through this smooth law
    avoids differencing the solver's recommendation directly, which is quantized (100 mm/s^2
    steps, 0.25 Hz shaper grid) and jumps when the recommended shaper type flips. When one
    axis limits throughout, the result is exactly 2 * |r| * nominal.
    """
    a_ref = min(accel_x_mm_s2, accel_y_mm_s2)
    if a_ref <= 0.0 or nominal_accel_mm_s2 <= 0.0:
        return 0.0
    a_plus = min(accel_x_mm_s2 * (1.0 + rel_x) ** 2, accel_y_mm_s2 * (1.0 + rel_y) ** 2)
    a_minus = min(accel_x_mm_s2 * (1.0 - rel_x) ** 2, accel_y_mm_s2 * (1.0 - rel_y) ** 2)
    return 0.5 * abs(a_plus - a_minus) / a_ref * nominal_accel_mm_s2


def _run_workspace_fea_sweep(
    spec: PrinterSpec,
    filament_id: str,
    grid_n: int = 3,
    include_z_extremes: bool = True,
    sweep_panel_mesh_n: int = 3,
) -> dict[str, Any]:
    """Run true 3D FEA assembly solves across a grid of real (u, v, z) positions.

    The centre point is always solved and always competes for the worst case, including on
    even grids that have no centre cell. With Z extremes enabled the centre column is swept
    in Z, and so is the softest XY cell when that is not the centre (corner + top/bottom Z). Architectures whose builder ignores XY (delta) solve
    the centre once, reuse it for every grid cell, and report ``xy_position_modeled = False``.
    """
    n = max(2, min(5, int(grid_n)))
    if n == 2:
        coords = [0.15, 0.85]
    else:
        step = 0.80 / float(n - 1)
        coords = [round(0.10 + i * step, 3) for i in range(n)]

    xy_modeled = spec.architecture not in _XY_INSENSITIVE_ARCHITECTURES
    z_stations = _Z_SWEEP_STATIONS if xy_modeled else _Z_SWEEP_STATIONS_DELTA

    t0 = time.perf_counter()
    freq_x_grid: list[list[float]] = []
    freq_y_grid: list[list[float]] = []
    min_axis_grid: list[list[float]] = []
    safe_accel_grid: list[list[float]] = []
    deflection_grid: list[list[float]] = []

    all_points: list[dict[str, Any]] = []
    num_solves = 0

    def _solve(u_rel: float, v_rel: float, z_rel: float) -> dict[str, float]:
        """Solve one real position and register it as a worst-case candidate."""
        nonlocal num_solves
        rep = evaluate_printer(
            spec,
            toolhead_rel_pos=(u_rel, v_rel, z_rel),
            filament_id=filament_id,
            panel_mesh_n=sweep_panel_mesh_n,
            bed_mesh_n=8,
            include_advanced_analysis=False,
        )
        num_solves += 1
        m = _extract_core_metrics(rep)
        all_points.append(
            {
                "u_rel": u_rel,
                "v_rel": v_rel,
                "z_rel": z_rel,
                **{k: round(val, 2) for k, val in m.items()},
            }
        )
        return m

    center_metrics = _solve(0.5, 0.5, 0.5)
    grid_cells: list[tuple[float, float, dict[str, float]]] = []

    def _z_row(z_rel: float, zm: dict[str, float]) -> dict[str, Any]:
        return {
            "z_rel": z_rel,
            "shaper_x_hz": round(zm["shaper_x_hz"], 2),
            "shaper_y_hz": round(zm["shaper_y_hz"], 2),
            "min_axis_hz": round(zm["min_axis_hz"], 2),
            "safe_accel_mm_s2": round(zm["recommended_max_accel_mm_s2"], 0),
            "deflection_um": round(zm["combined_max_deflection_um"], 2),
        }

    # Row = v_rel (Y), Col = u_rel (X) at mid-Z (0.5)
    for v_rel in coords:
        row_fx: list[float] = []
        row_fy: list[float] = []
        row_min: list[float] = []
        row_acc: list[float] = []
        row_def: list[float] = []
        for u_rel in coords:
            is_center = abs(u_rel - 0.5) < 1e-6 and abs(v_rel - 0.5) < 1e-6
            m = center_metrics if (is_center or not xy_modeled) else _solve(u_rel, v_rel, 0.5)
            grid_cells.append((u_rel, v_rel, m))
            row_fx.append(round(m["shaper_x_hz"], 2))
            row_fy.append(round(m["shaper_y_hz"], 2))
            row_min.append(round(m["min_axis_hz"], 2))
            row_acc.append(round(m["recommended_max_accel_mm_s2"], 0))
            row_def.append(round(m["combined_max_deflection_um"], 2))
        freq_x_grid.append(row_fx)
        freq_y_grid.append(row_fy)
        min_axis_grid.append(row_min)
        safe_accel_grid.append(row_acc)
        deflection_grid.append(row_def)

    z_sweep: list[dict[str, Any]] = []
    if include_z_extremes:
        for z_rel in z_stations:
            zm = center_metrics if abs(z_rel - 0.5) < 1e-6 else _solve(0.5, 0.5, z_rel)
            z_sweep.append(_z_row(z_rel, zm))

    # The softest XY cell at the Z extremes: the centre column alone can miss corner + top-Z.
    worst_xy_z_sweep: list[dict[str, Any]] = []
    if include_z_extremes and xy_modeled:
        u_w, v_w, _ = min(grid_cells, key=lambda cell: cell[2]["min_axis_hz"])
        if not (abs(u_w - 0.5) < 1e-6 and abs(v_w - 0.5) < 1e-6):
            for z_rel in z_stations:
                if abs(z_rel - 0.5) < 1e-6:
                    continue
                row = _z_row(z_rel, _solve(u_w, v_w, z_rel))
                worst_xy_z_sweep.append({"u_rel": u_w, "v_rel": v_w, **row})

    worst_freq_pt = min(all_points, key=lambda p: p["min_axis_hz"])
    best_freq_pt = max(all_points, key=lambda p: p["min_axis_hz"])
    worst_accel_pt = min(all_points, key=lambda p: p["recommended_max_accel_mm_s2"])
    worst_def_pt = max(all_points, key=lambda p: p["combined_max_deflection_um"])

    center_min_hz = max(center_metrics["min_axis_hz"], 1e-6)
    center_acc = max(center_metrics["recommended_max_accel_mm_s2"], 1e-6)
    worst_freq_drop_pct = max(
        0.0, 100.0 * (center_min_hz - worst_freq_pt["min_axis_hz"]) / center_min_hz
    )
    worst_accel_drop_pct = max(
        0.0,
        100.0 * (center_acc - worst_accel_pt["recommended_max_accel_mm_s2"]) / center_acc,
    )
    spatial_spread_pct = max(
        0.0,
        100.0 * (best_freq_pt["min_axis_hz"] - worst_freq_pt["min_axis_hz"]) / center_min_hz,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "grid_n": n,
        "coords": coords,
        "xy_position_modeled": xy_modeled,
        "num_fea_solves": num_solves,
        "solve_time_ms": round(elapsed_ms, 1),
        "freq_x_grid_hz": freq_x_grid,
        "freq_y_grid_hz": freq_y_grid,
        "min_axis_grid_hz": min_axis_grid,
        "safe_accel_grid_mm_s2": safe_accel_grid,
        "deflection_grid_um": deflection_grid,
        "z_sweep": z_sweep,
        "worst_xy_z_sweep": worst_xy_z_sweep,
        "center_min_axis_hz": round(center_metrics["min_axis_hz"], 2),
        "center_safe_accel_mm_s2": round(center_metrics["recommended_max_accel_mm_s2"], 0),
        "worst_case_min_axis_hz": round(worst_freq_pt["min_axis_hz"], 2),
        "worst_case_safe_accel_mm_s2": round(worst_accel_pt["recommended_max_accel_mm_s2"], 0),
        "worst_case_deflection_um": round(worst_def_pt["combined_max_deflection_um"], 2),
        "worst_freq_pos": [
            worst_freq_pt["u_rel"],
            worst_freq_pt["v_rel"],
            worst_freq_pt["z_rel"],
        ],
        "worst_accel_pos": [
            worst_accel_pt["u_rel"],
            worst_accel_pt["v_rel"],
            worst_accel_pt["z_rel"],
        ],
        "worst_deflection_pos": [
            worst_def_pt["u_rel"],
            worst_def_pt["v_rel"],
            worst_def_pt["z_rel"],
        ],
        "worst_freq_drop_pct": round(worst_freq_drop_pct, 2),
        "worst_accel_drop_pct": round(worst_accel_drop_pct, 2),
        "spatial_spread_pct": round(spatial_spread_pct, 2),
    }


def _run_uncertainty_analysis(
    spec: PrinterSpec,
    nominal_deep_metrics: dict[str, float],
    filament_id: str,
    max_groups: int | None = None,
) -> dict[str, Any]:
    """Propagate model-confidence uncertainties via central-difference 3D FEA perturbations.

    Perturbs every uncertain lumped-stiffness override group plus the toolhead moving mass.
    ``max_groups`` keeps only the lowest-trust inputs; trust does not rank influence, so
    anything it drops is reported in ``skipped_groups`` and the resulting band is a lower bound.
    The safe-accel band is propagated from the shaper-frequency responses (accel scales with
    frequency squared, see ``_accel_sigma_from_freq``) rather than from the quantized accel.
    """
    t0 = time.perf_counter()
    conf_audit = build_model_confidence_audit(spec)
    eff_params = build_effective_model_params(spec)

    # Map override_group -> lowest-trust entry metadata
    group_entries: dict[str, dict[str, Any]] = {}
    field_entries: dict[str, dict[str, Any]] = {}
    for entry in conf_audit.get("entries", []):
        if entry.get("id") in _UNCERTAINTY_SPEC_FIELD_SPECS:
            field_entries[str(entry["id"])] = entry
        og = entry.get("override_group")
        if not og or og not in _UNCERTAINTY_GROUP_SPECS:
            continue
        prev = group_entries.get(og)
        if prev is None or float(entry.get("trust_pct", 100)) < float(prev.get("trust_pct", 100)):
            group_entries[og] = entry

    # Each candidate: (trust_pct, key, entry, label, advice, perturb(scale) -> spec | None).
    # Exact (>=94% trust) untouched inputs are skipped.
    Candidate = tuple[float, str, dict[str, Any], str, str, Callable[[float], PrinterSpec | None]]
    candidates: list[Candidate] = []
    for og, entry in group_entries.items():
        trust_pct = float(entry.get("trust_pct", 80.0))
        overridden = bool(entry.get("overridden", False))
        if trust_pct >= 94.0 and not overridden:
            continue
        eff_sub = eff_params.get(og, {}).get("effective", {})
        if not isinstance(eff_sub, dict):
            continue
        # Ensure at least one perturbable positive field exists
        label, field_specs, advice = _UNCERTAINTY_GROUP_SPECS[og]
        has_active = any(
            isinstance(eff_sub.get(fn), (int, float)) and float(eff_sub.get(fn)) > 0.0
            for fn, _, _ in field_specs
        )
        if not has_active:
            continue

        def _perturb_group(
            scale: float,
            _og: str = og,
            _eff: dict[str, Any] = eff_sub,
            _fields: list[tuple[str, float, float]] = field_specs,
        ) -> PrinterSpec | None:
            return _apply_group_scale(spec, _og, _eff, _fields, scale)

        candidates.append((trust_pct, og, entry, label, advice, _perturb_group))

    for entry_id, entry in field_entries.items():
        trust_pct = float(entry.get("trust_pct", 80.0))
        if trust_pct >= 94.0:
            continue
        label, field_specs, advice = _UNCERTAINTY_SPEC_FIELD_SPECS[entry_id]

        def _perturb_fields(
            scale: float,
            _fields: list[tuple[str, float, float]] = field_specs,
        ) -> PrinterSpec | None:
            return _apply_spec_field_scale(spec, _fields, scale)

        candidates.append((trust_pct, entry_id, entry, label, advice, _perturb_fields))

    candidates.sort(key=lambda item: item[0])
    selected = candidates if max_groups is None else candidates[: max(1, int(max_groups))]
    skipped_groups = [cand[3] for cand in candidates[len(selected) :]]

    # Accumulators for RSS 1-sigma across selected uncertain parameter groups
    metric_keys = (
        "shaper_x_hz",
        "shaper_y_hz",
        "min_axis_hz",
        "fundamental_freq_hz",
        "recommended_max_accel_mm_s2",
        "combined_max_deflection_um",
    )
    var_accum: dict[str, float] = {k: 0.0 for k in metric_keys}
    drivers_raw: list[dict[str, Any]] = []
    num_solves = 0

    nominal_accel = float(nominal_deep_metrics["recommended_max_accel_mm_s2"])
    accel_x_nom = float(nominal_deep_metrics.get("accel_x_mm_s2") or nominal_accel)
    accel_y_nom = float(nominal_deep_metrics.get("accel_y_mm_s2") or nominal_accel)

    for trust_pct, og, entry, label, advice, perturb in selected:
        overridden = bool(entry.get("overridden", False))
        sigma_rel = _trust_to_sigma_rel(trust_pct, overridden=overridden)

        spec_minus = perturb(1.0 - sigma_rel)
        spec_plus = perturb(1.0 + sigma_rel)
        if spec_minus is None or spec_plus is None:
            continue

        rep_m = evaluate_printer(
            spec_minus,
            filament_id=filament_id,
            panel_mesh_n=2,
            bed_mesh_n=8,
            include_advanced_analysis=False,
        )
        rep_p = evaluate_printer(
            spec_plus,
            filament_id=filament_id,
            panel_mesh_n=2,
            bed_mesh_n=8,
            include_advanced_analysis=False,
        )
        num_solves += 2
        m_minus = _extract_core_metrics(rep_m)
        m_plus = _extract_core_metrics(rep_p)

        sigmas: dict[str, float] = {}
        for k in metric_keys:
            # Central difference: 1-sigma response = 0.5 * |f(+sigma) - f(-sigma)|
            sigmas[k] = 0.5 * abs(m_plus[k] - m_minus[k])

        # Safe accel: propagate the signed relative shaper-frequency responses through a ~ f^2.
        rel_freq: dict[str, float] = {}
        for axis_key in ("shaper_x_hz", "shaper_y_hz"):
            mid = 0.5 * (m_plus[axis_key] + m_minus[axis_key])
            rel_freq[axis_key] = (
                0.5 * (m_plus[axis_key] - m_minus[axis_key]) / mid if mid > 0.0 else 0.0
            )
        sigmas["recommended_max_accel_mm_s2"] = _accel_sigma_from_freq(
            nominal_accel, accel_x_nom, accel_y_nom, rel_freq["shaper_x_hz"], rel_freq["shaper_y_hz"]
        )

        for k in metric_keys:
            var_accum[k] += sigmas[k] * sigmas[k]

        drivers_raw.append(
            {
                "group_id": og,
                "label": label,
                "trust_pct": round(trust_pct, 0),
                "tier": str(entry.get("tier", "estimate")),
                "overridden": overridden,
                "sigma_param_pct": round(sigma_rel * 100.0, 1),
                "delta_freq_hz": round(_Z_90 * sigmas["min_axis_hz"], 2),
                "delta_accel_mm_s2": round(_Z_90 * sigmas["recommended_max_accel_mm_s2"], 0),
                "delta_deflection_um": round(_Z_90 * sigmas["combined_max_deflection_um"], 2),
                "_var_freq": sigmas["min_axis_hz"] ** 2,
                "_var_defl": sigmas["combined_max_deflection_um"] ** 2,
                "how_to_tighten": advice,
            }
        )

    # Rank by the larger of the frequency and flex variance shares, so an input that dominates
    # nozzle-bed flex (e.g. the Z drive) is not buried under ones that only move resonance.
    total_var_freq = sum(d["_var_freq"] for d in drivers_raw)
    total_var_defl = sum(d["_var_defl"] for d in drivers_raw)
    drivers: list[dict[str, Any]] = []
    for d in drivers_raw:
        share = (100.0 * d["_var_freq"] / total_var_freq) if total_var_freq > 1e-12 else 0.0
        share_defl = (100.0 * d["_var_defl"] / total_var_defl) if total_var_defl > 1e-12 else 0.0
        entry_out = {k: v for k, v in d.items() if not k.startswith("_")}
        entry_out["variance_share_pct"] = round(share, 1)
        entry_out["deflection_variance_share_pct"] = round(share_defl, 1)
        drivers.append(entry_out)
    drivers.sort(
        key=lambda d: (
            max(d["variance_share_pct"], d["deflection_variance_share_pct"]),
            d["variance_share_pct"],
        ),
        reverse=True,
    )

    bands: dict[str, dict[str, float]] = {}
    for k in metric_keys:
        nom = float(nominal_deep_metrics[k])
        sigma_tot = math.sqrt(var_accum[k])
        half_90 = _Z_90 * sigma_tot
        lo_90 = max(0.0, nom - half_90)
        hi_90 = nom + half_90
        pct_90 = (100.0 * half_90 / max(abs(nom), 1e-6)) if nom > 0 else 0.0
        digits = 0 if k == "recommended_max_accel_mm_s2" else 2
        bands[k] = {
            "nominal": round(nom, digits),
            "p05": round(lo_90, digits),
            "p95": round(hi_90, digits),
            "ci90_half_width": round(half_90, digits),
            "ci90_pct": round(pct_90, 1),
        }

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "overall_trust_pct": int(conf_audit.get("overall_trust_pct", 0)),
        "num_groups_perturbed": len(drivers),
        "skipped_groups": skipped_groups,
        "coverage_note": UNCERTAINTY_COVERAGE_NOTE,
        "num_fea_solves": num_solves,
        "solve_time_ms": round(elapsed_ms, 1),
        "bands": bands,
        "drivers": drivers,
    }


def _fmt_pos(pos: Any) -> str:
    """Format a relative (u, v, z) workspace position without rounding 0.15 or 0.367 away."""
    return f"(u={float(pos[0]):.2f}, v={float(pos[1]):.2f}, z={float(pos[2]):.2f})"


def _build_readiness_checks(
    deep_verify: dict[str, Any],
    workspace_fea: dict[str, Any],
    uncertainty: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Evaluate 5 engineering sign-off gates and compute overall readiness verdict."""
    conv = deep_verify["convergence"]
    deep_sim = deep_verify["deep_simulation"]
    deep_sum = deep_verify["deep_summary"]

    checks: list[dict[str, Any]] = []

    # 1. Mesh Convergence
    f_d = float(conv["freq_diff_pct"])
    d_d = float(conv["deflection_diff_pct"])
    if f_d <= 5.0 and d_d <= 8.0:
        mesh_status = "pass"
        mesh_detail = f"Fine 8×8/16×16 shell mesh converged (f₁ Δ {f_d:.1f}%, flex Δ {d_d:.1f}%)."
    elif f_d <= 15.0 and d_d <= 15.0:
        mesh_status = "caution"
        mesh_detail = f"Moderate mesh sensitivity (f₁ Δ {f_d:.1f}%, flex Δ {d_d:.1f}%); inspect Fine mesh."
    else:
        mesh_status = "fail"
        mesh_detail = f"High mesh sensitivity (f₁ Δ {f_d:.1f}%, flex Δ {d_d:.1f}%)."
    checks.append(
        {
            "id": "mesh_convergence",
            "title": "Shell & Bed Mesh Convergence",
            "status": mesh_status,
            "value": f"Δf₁ {f_d:.1f}% · Δflex {d_d:.1f}%",
            "detail": mesh_detail,
        }
    )

    # 2. Workspace Uniformity (true 3D FEA across bed): gate on the larger of the resonance
    # drop and the safe-accel drop, each reported at the position where it actually occurs.
    w_drop = float(workspace_fea["worst_freq_drop_pct"])
    a_drop = float(workspace_fea["worst_accel_drop_pct"])
    f_pos = _fmt_pos(workspace_fea["worst_freq_pos"])
    a_pos = _fmt_pos(workspace_fea["worst_accel_pos"])
    w_acc = int(round(float(workspace_fea["worst_case_safe_accel_mm_s2"])))
    gate_drop = max(w_drop, a_drop)
    drops = f"resonance -{w_drop:.1f}% at {f_pos}, safe accel -{a_drop:.1f}% at {a_pos}"
    if gate_drop <= 10.0:
        ws_status = "pass"
        ws_detail = f"Uniform across the workspace ({drops})."
    elif gate_drop <= 20.0:
        ws_status = "caution"
        ws_detail = f"Noticeable corner/Z softening ({drops}); cap safe accel near {w_acc:,} mm/s²."
    else:
        ws_status = "fail"
        ws_detail = f"Severe workspace drop ({drops}); stiffen gantry/frame or tighten belts."
    if not bool(workspace_fea.get("xy_position_modeled", True)):
        ws_detail += " XY position is not modeled for this architecture; Z sweep only."
    checks.append(
        {
            "id": "workspace_uniformity",
            "title": "Full-Workspace 3D FEA Uniformity",
            "status": ws_status,
            "value": f"-{w_drop:.1f}% Hz · -{a_drop:.1f}% accel",
            "detail": ws_detail,
        }
    )

    # 3. Prediction Uncertainty (90% CI)
    min_band = uncertainty["bands"]["min_axis_hz"]
    ci_pct = float(min_band["ci90_pct"])
    top_drv = uncertainty["drivers"][0]["label"] if uncertainty["drivers"] else "None"
    if ci_pct <= _BAND_PASS_PCT:
        unc_status = "pass"
        unc_detail = f"Tight 90% parametric band (±{ci_pct:.1f}% on limiting shaper axis; top driver: {top_drv})."
    elif ci_pct <= _BAND_FAIL_PCT:
        unc_status = "caution"
        unc_detail = f"Moderate prediction spread (±{ci_pct:.1f}%); tighten '{top_drv}' via CAD asset or ADXL fit."
    else:
        unc_status = "fail"
        unc_detail = f"Wide prediction spread (±{ci_pct:.1f}%); '{top_drv}' dominates uncertainty."
    skipped = list(uncertainty.get("skipped_groups") or [])
    if skipped:
        if unc_status == "pass":
            unc_status = "caution"
        unc_detail += (
            f" Band is a lower bound: {len(skipped)} uncertain group(s) were not perturbed"
            f" ({', '.join(skipped)})."
        )
    checks.append(
        {
            "id": "prediction_uncertainty",
            "title": "Parametric Uncertainty (90% Band)",
            "status": unc_status,
            "value": f"±{ci_pct:.1f}% ({uncertainty['overall_trust_pct']}% trust)",
            "detail": unc_detail,
        }
    )

    # 4. Structural Safety & Clearance
    coll = deep_sim.get("collision", {})
    has_coll = bool(coll.get("has_collision", False))
    sf = float(deep_sim.get("static_fea", {}).get("min_safety_factor", 10.0))
    if not has_coll and sf >= 2.0:
        str_status = "pass"
        str_detail = f"No clearance collisions; minimum static safety factor {sf:.1f}×."
    elif not has_coll and sf >= 1.25:
        str_status = "caution"
        str_detail = f"Low static safety margin ({sf:.2f}×); inspect peak stress members."
    else:
        str_status = "fail"
        str_detail = (
            "Frame/toolhead clearance collision detected."
            if has_coll
            else f"Insufficient structural safety factor ({sf:.2f}×)."
        )
    checks.append(
        {
            "id": "structural_safety",
            "title": "Structural Safety & Clearance",
            "status": str_status,
            "value": "Collision" if has_coll else f"SF {sf:.1f}×",
            "detail": str_detail,
        }
    )

    # 5. Bed Flatness & Mechanical Margins
    bed_flat = float(deep_sum.get("bed_flatness_error_um", 0.0))
    kin = deep_sim.get("kinematics", {})
    peak_t = float(kin.get("belt_peak_tension_n", 0.0))
    max_t = float(kin.get("belt_max_working_tension_n", 0.0))
    belt_sf = (max_t / peak_t) if (peak_t > 1e-6 and max_t > 0.0) else 3.0
    # Truncate (never round up) so a value just under a limit cannot display as meeting it.
    sf_txt = f"{math.floor(belt_sf * 100.0 + 1e-9) / 100.0:.2f}"
    overtension = bool(kin.get("belt_overtension_risk", False))
    creep_risk = bool(kin.get("bracket_creep_risk", False))
    if (
        bed_flat <= _BED_FLAT_PASS_UM
        and belt_sf >= _BELT_SF_PASS
        and not overtension
        and not creep_risk
    ):
        mech_status = "pass"
        mech_detail = f"Fine 16×16 bed flatness {bed_flat:.1f} µm; belt tension safety factor {sf_txt}×."
    elif bed_flat <= _BED_FLAT_FAIL_UM and belt_sf >= _BELT_SF_FAIL and not overtension:
        mech_status = "caution"
        mech_detail = (
            f"Bed flatness {bed_flat:.1f} µm · belt SF {sf_txt}×"
            + (" · bracket creep risk" if creep_risk else "")
            + " — review mount or bracket material."
        )
    else:
        mech_status = "fail"
        reasons: list[str] = []
        if bed_flat > _BED_FLAT_FAIL_UM:
            reasons.append(f"bed flatness {bed_flat:.1f} µm exceeds {_BED_FLAT_FAIL_UM:.0f} µm")
        if belt_sf < _BELT_SF_FAIL:
            reasons.append(
                f"belt safety factor {sf_txt}× is below {_BELT_SF_FAIL:.2f}×"
                f" ({peak_t:.1f} N peak vs {max_t:.1f} N working limit)"
            )
        elif overtension:
            reasons.append("belt overtension risk is flagged")
        mech_detail = "Out of sign-off bounds: " + "; ".join(reasons) + "."
    checks.append(
        {
            "id": "mechanical_margins",
            "title": "Bed Flatness & Belt Margin",
            "status": mech_status,
            "value": f"{bed_flat:.1f} µm · Belt {sf_txt}×",
            "detail": mech_detail,
        }
    )

    statuses = {c["status"] for c in checks}
    if "fail" in statuses:
        verdict = "fail"
    elif "caution" in statuses:
        verdict = "caution"
    else:
        verdict = "pass"
    return verdict, checks


def run_signoff_verification(
    spec: PrinterSpec,
    gcode_text: str | None = None,
    filament_id: str = "pla",
    deep_mesh_n: int = 8,
    deep_bed_mesh_n: int = 16,
    workspace_grid_n: int = 3,
    include_z_extremes: bool = True,
    max_uncertainty_groups: int | None = None,
) -> dict[str, Any]:
    """Run Sign-Off Verification: fine-mesh convergence, workspace 3D FEA, 90% parametric bands."""
    t_total = time.perf_counter()
    deep_mesh_n = max(_DEEP_MESH_N_RANGE[0], min(_DEEP_MESH_N_RANGE[1], int(deep_mesh_n)))
    deep_bed_mesh_n = max(
        _DEEP_BED_MESH_N_RANGE[0], min(_DEEP_BED_MESH_N_RANGE[1], int(deep_bed_mesh_n))
    )

    # Stage 1: Fine-mesh MITC4 panel + bed convergence
    deep_verify = verify_printer_deep(
        spec,
        gcode_text=gcode_text,
        filament_id=filament_id,
        deep_mesh_n=deep_mesh_n,
        deep_bed_mesh_n=deep_bed_mesh_n,
    )

    # Extract nominal high-accuracy metrics from the fine-mesh solve
    deep_sim = deep_verify["deep_simulation"]
    mf = deep_sim.get("modal_fea", {})
    sx = float((mf.get("shaper_x") or {}).get("effective_freq_hz") or mf.get("fundamental_freq_hz", 0.0))
    sy = float((mf.get("shaper_y") or {}).get("effective_freq_hz") or mf.get("fundamental_freq_hz", 0.0))
    nominal_deep_metrics = {
        "shaper_x_hz": sx,
        "shaper_y_hz": sy,
        "min_axis_hz": min(sx, sy),
        "fundamental_freq_hz": float(mf.get("fundamental_freq_hz", 0.0)),
        "recommended_max_accel_mm_s2": float(mf.get("recommended_max_accel_mm_s2", 0.0)),
        "combined_max_deflection_um": float(
            deep_sim.get("static_fea", {}).get("combined_max_deflection_um", 0.0)
        ),
        # Per-axis shaper accel limits feed the f^2 safe-accel uncertainty propagation.
        "accel_x_mm_s2": float((mf.get("shaper_x") or {}).get("max_accel_mm_s2") or 0.0),
        "accel_y_mm_s2": float((mf.get("shaper_y") or {}).get("max_accel_mm_s2") or 0.0),
    }

    # Stage 2: True multi-position 3D FEA workspace sweep
    workspace_fea = _run_workspace_fea_sweep(
        spec,
        filament_id=filament_id,
        grid_n=workspace_grid_n,
        include_z_extremes=include_z_extremes,
    )

    # Stage 3: Confidence-driven 90% CI uncertainty propagation & ranked drivers
    uncertainty = _run_uncertainty_analysis(
        spec,
        nominal_deep_metrics=nominal_deep_metrics,
        filament_id=filament_id,
        max_groups=max_uncertainty_groups,
    )

    # Stage 4: Readiness checks & overall verdict
    verdict, checks = _build_readiness_checks(deep_verify, workspace_fea, uncertainty)

    total_ms = (time.perf_counter() - t_total) * 1000.0
    total_solves = (
        2
        + int(workspace_fea["num_fea_solves"])
        + int(uncertainty["num_fea_solves"])
    )

    return {
        "verdict": verdict,
        "checks": checks,
        "total_solve_time_ms": round(total_ms, 1),
        "total_fea_solves": total_solves,
        # Fine-mesh convergence (compatible with DeepVerifyResponse fields)
        "fast_mesh_n": deep_verify["fast_mesh_n"],
        "deep_mesh_n": deep_verify["deep_mesh_n"],
        "fast_bed_mesh_n": deep_verify["fast_bed_mesh_n"],
        "deep_bed_mesh_n": deep_verify["deep_bed_mesh_n"],
        "fast_summary": deep_verify["fast_summary"],
        "deep_summary": deep_verify["deep_summary"],
        "convergence": deep_verify["convergence"],
        # True multi-position 3D FEA workspace sweep
        "workspace_fea": workspace_fea,
        # Confidence-driven 90% CI & ranked uncertainty drivers
        "uncertainty": uncertainty,
        # Full fine-mesh SimulationReport dict for 3D viewport inspection
        "deep_simulation": deep_sim,
    }
