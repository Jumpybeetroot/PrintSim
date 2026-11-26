"""Multi-objective Pareto front optimizer (pymoo NSGA-II + pure-NumPy fallback).

Explores the Speed vs. Stiffness vs. Build Cost trade-off frontier for a 3D printer
design while preserving the user's architecture, build volume, and outer frame dimensions.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from printsim.engine.evaluator import SimulationReport, evaluate_printer
from printsim.geometry.bom import PrinterBOM, generate_bom
from printsim.kinematics import analyze_kinematics
from printsim.schema import (
    HOTEND_CATALOG,
    STEPPER_CATALOG,
    TOOLHEAD_PRESETS,
    PrinterSpec,
    StepperSpec,
)

# ---------------------------------------------------------------------------
# Discrete Engineering Design Space (12 hardware upgrade genes)
# ---------------------------------------------------------------------------

FRAME_PROFILES: tuple[str, ...] = (
    "extrusion_2020",
    "extrusion_3030",
    "extrusion_4040",
)

BRACING_OPTIONS: tuple[str, ...] = (
    "none",
    "corner_gussets",
    "rear_cross",
    "full_truss",
)

# (id, label, enclosed_panels, panel_mounting, panel_material, panel_thickness_mm)
PANEL_CONFIGS: tuple[tuple[str, str, bool, str, str, float], ...] = (
    ("open", "open (no panels)", False, "none", "acrylic_pmma", 3.0),
    ("clip_3mm_acrylic", "3mm clip acrylic", True, "clip_enclosure", "acrylic_pmma", 3.0),
    ("bolted_4mm_acm", "4mm bolted ACM", True, "bolted_structural", "acm_dibond", 4.0),
    ("bolted_4mm_al5052", "4mm bolted Al 5052", True, "bolted_structural", "aluminum_5052_sheet", 4.0),
    ("bolted_3mm_cf", "3mm bolted Carbon Fiber", True, "bolted_structural", "carbon_fiber_plate", 3.0),
)

# (id, label, profile, material)
X_BEAM_OPTIONS: tuple[tuple[str, str, str, str], ...] = (
    ("al_2020", "Aluminum 2020 extrusion", "extrusion_2020", "aluminum_6063_t5"),
    ("cf_tube_20x20x2", "Carbon Fiber 20x20x2mm tube", "cf_tube_20x20x2", "carbon_fiber_tube"),
    ("cf_20x20x2_high_tg", "Carbon Fiber 20x20x2mm (High-Tg 125C)", "cf_tube_20x20x2", "carbon_fiber_high_tg"),
    ("cf_carbonara_torsional_45", "Carbonara 20x20mm (±45°/0° High-Tg)", "cf_tube_20x20_carbonara", "carbon_fiber_torsional_45"),
    ("cf_30x20x2_torsional_45", "Carbon Fiber 30x20x2mm Wide-Y (±45°/0°)", "cf_tube_30x20x2", "carbon_fiber_torsional_45"),
    ("cf_carbonara_m40j_hm", "Carbonara 20x20mm (M40J High-Modulus)", "cf_tube_20x20_carbonara", "carbon_fiber_m40j_hm"),
)

DRIVE_MODES: tuple[str, ...] = ("2wd", "awd")

# (id, label, belt_type, belt_core, pretension_n)
BELT_OPTIONS: tuple[tuple[str, str, str, str, float], ...] = (
    ("gt2_6mm_fg_28n", "GT2 6mm fiberglass @ 28N", "gt2_6mm", "fiberglass", 28.0),
    ("gt2_9mm_fg_48n", "GT2 9mm fiberglass @ 48N", "gt2_9mm", "fiberglass", 48.0),
    ("gt2_9mm_fg_62n", "GT2 9mm fiberglass @ 62N", "gt2_9mm", "fiberglass", 62.0),
    ("gt2_12mm_fg_80n", "GT2 12mm fiberglass @ 80N", "gt2_12mm", "fiberglass", 80.0),
)

# (stepper_preset_id, supply_voltage_v, run_current_a)
STEPPER_SETUPS: tuple[tuple[str, float, float], ...] = (
    ("ldo_2504ac", 24.0, 1.2),
    ("ldo_2504ac", 48.0, 2.0),
    ("ldo_2804ah", 48.0, 2.2),
    ("excit3d_max", 48.0, 3.5),
    ("ldo_60_3004", 48.0, 2.4),
    ("ldo_09deg_2504", 48.0, 1.8),
)

TOOLHEAD_OPTIONS: tuple[str, ...] = (
    "stealthburner",
    "xol_compact",
    "archetype_cpap",
)

# Realistic catalog hotend tiers (mass, stiffness, length, melt zone, power, & cost coupled)
HOTEND_PRESET_OPTIONS: tuple[str, ...] = (
    "phaetus_dragon_hf",
    "phaetus_rapido_2_hf",
    "phaetus_rapido_2_uhf",
)

# (melt_zone_length_mm, heater_power_w) kept for backward-compatible inspection
HOTEND_TIERS: tuple[tuple[float, float], ...] = tuple(
    (
        float(HOTEND_CATALOG[pid]["melt_zone_length_mm"]),
        float(HOTEND_CATALOG[pid]["heater_power_w"]),
    )
    for pid in HOTEND_PRESET_OPTIONS
)

_PARETO_REF_ACCEL_MM_S2 = 10000.0

TOOLHEAD_MOUNT_OPTIONS: tuple[str, ...] = (
    "abs_printed",
    "ppa_cf_printed",
    "aluminum_6061_t6",
)

TOOLHEAD_MOUNT_STIFFNESS_N_PER_MM: dict[str, float] = {
    "abs_printed": 380.0,
    "pc_pbt_gf_printed": 580.0,
    "pet_cf_printed": 650.0,
    "pa12_cf_printed": 800.0,
    "pps_cf_printed": 1050.0,
    "ppa_cf_printed": 1150.0,
    "slm_aluminum": 1740.0,
    "aluminum_6061_t6": 1850.0,
    "aluminum_7075_t6": 1950.0,
    "slm_ti6al4v": 2150.0,
    "slm_maraging_steel": 2300.0,
    "slm_316l_steel": 2320.0,
    "slm_17_4ph_steel": 2350.0,
    "sheet_metal_steel": 2400.0,
}

BED_MOUNT_OPTIONS: tuple[str, ...] = (
    "springs_4pt",
    "silicone_4pt",
    "floating_4pt",
    "kinematic_3pt",
    "rigid_4pt",
)

BRACKET_OPTIONS: tuple[str, ...] = (
    "abs_printed",
    "ppa_cf_printed",
    "aluminum_6061_t6",
)

_N_VARS = 12
_XL = np.zeros(_N_VARS, dtype=int)
_XU = np.array(
    [
        len(FRAME_PROFILES) - 1,
        len(BRACING_OPTIONS) - 1,
        len(PANEL_CONFIGS) - 1,
        len(X_BEAM_OPTIONS) - 1,
        len(DRIVE_MODES) - 1,
        len(BELT_OPTIONS) - 1,
        len(STEPPER_SETUPS) - 1,
        len(TOOLHEAD_OPTIONS) - 1,
        len(HOTEND_TIERS) - 1,
        len(TOOLHEAD_MOUNT_OPTIONS) - 1,
        len(BED_MOUNT_OPTIONS) - 1,
        len(BRACKET_OPTIONS) - 1,
    ],
    dtype=int,
)

MAX_PARETO_POINTS = 16


# ---------------------------------------------------------------------------
# Backend availability probe
# ---------------------------------------------------------------------------


def is_pymoo_available() -> bool:
    """Return True if pymoo and its NSGA-II integer operators are importable."""
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2  # noqa: F401
        from pymoo.core.problem import ElementwiseProblem  # noqa: F401
        from pymoo.operators.crossover.sbx import SBX  # noqa: F401
        from pymoo.operators.mutation.pm import PM  # noqa: F401
        from pymoo.operators.repair.rounding import RoundingRepair  # noqa: F401
        from pymoo.operators.sampling.rnd import IntegerRandomSampling  # noqa: F401
        from pymoo.optimize import minimize  # noqa: F401

        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public Result Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ParetoPoint:
    """A single Pareto-optimal (or baseline) printer hardware configuration."""

    id: str
    label: str
    changes: list[str]
    total_cost_usd: float
    safe_max_accel_mm_s2: float
    nozzle_deflection_um: float
    fundamental_freq_hz: float
    max_flow_mm3_s: float
    total_mass_kg: float
    composite_score: float
    spec: PrinterSpec
    primary_problem_subsystem: str | None = None

    def to_dict(self, include_spec: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "changes": list(self.changes),
            "total_cost_usd": round(float(self.total_cost_usd), 2),
            "safe_max_accel_mm_s2": round(float(self.safe_max_accel_mm_s2), 1),
            "nozzle_deflection_um": round(float(self.nozzle_deflection_um), 2),
            "fundamental_freq_hz": round(float(self.fundamental_freq_hz), 1),
            "max_flow_mm3_s": round(float(self.max_flow_mm3_s), 1),
            "total_mass_kg": round(float(self.total_mass_kg), 3),
            "composite_score": round(float(self.composite_score), 1),
        }
        if self.primary_problem_subsystem is not None:
            payload["primary_problem_subsystem"] = self.primary_problem_subsystem
        if include_spec:
            payload["spec"] = self.spec.model_dump(mode="json")
        return payload


@dataclass
class ParetoResult:
    """Complete multi-objective optimization result with baseline and non-dominated frontier."""

    backend: str
    evaluations: int
    population_size: int
    generations: int
    baseline: ParetoPoint
    pareto_front: list[ParetoPoint]

    def to_dict(self, include_specs: bool = True) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "evaluations": int(self.evaluations),
            "population_size": int(self.population_size),
            "generations": int(self.generations),
            "baseline": self.baseline.to_dict(include_spec=include_specs),
            "pareto_front": [pt.to_dict(include_spec=include_specs) for pt in self.pareto_front],
        }


# ---------------------------------------------------------------------------
# Design Vector Encoding, Decoding & Evaluation Helpers
# ---------------------------------------------------------------------------


def _normalize_gene_vector(x: np.ndarray | list[int] | tuple[int, ...], is_corexy: bool) -> tuple[int, ...]:
    arr = np.rint(np.asarray(x, dtype=float)).astype(int)
    arr = np.clip(arr, _XL, _XU)
    if not is_corexy:
        arr[4] = 0
    return tuple(int(v) for v in arr)


def _encode_structural_polymer_or_cnc(mat_id: str, options: tuple[str, ...]) -> int:
    """Map a mount/bracket material ID onto a discrete (abs, ppa_cf, aluminum_6061_t6) gene."""
    if mat_id in options:
        return options.index(mat_id)
    if mat_id == "aluminum_7075_t6":
        return 2
    if mat_id in ("pet_cf_printed", "pa12_cf_printed", "pps_cf_printed"):
        return 1
    return 0


def _encode_baseline_vector(spec: PrinterSpec) -> tuple[int, ...]:
    """Map a baseline PrinterSpec to its closest discrete 12-gene vector for population seeding."""
    # x[0] frame profile
    x0 = FRAME_PROFILES.index(spec.frame.profile) if spec.frame.profile in FRAME_PROFILES else 0
    # x[1] bracing
    x1 = BRACING_OPTIONS.index(spec.frame.bracing) if spec.frame.bracing in BRACING_OPTIONS else 1
    # x[2] panels
    if not spec.frame.enclosed_panels or spec.frame.panel_mounting == "none":
        x2 = 0
    elif spec.frame.panel_mounting == "clip_enclosure":
        x2 = 1
    elif spec.frame.panel_material == "aluminum_5052_sheet":
        x2 = 3
    elif spec.frame.panel_material == "carbon_fiber_plate":
        x2 = 4
    else:
        x2 = 2
    # x[3] x-beam
    x3 = 0
    for idx_xb, (_, _, xb_prof, xb_mat) in enumerate(X_BEAM_OPTIONS):
        if spec.gantry.x_beam_profile == xb_prof and spec.gantry.x_beam_material == xb_mat:
            x3 = idx_xb
            break
    else:
        if "cf_" in spec.gantry.x_beam_profile or "carbon" in spec.gantry.x_beam_material:
            if spec.gantry.x_beam_material == "carbon_fiber_m40j_hm":
                x3 = 5
            elif spec.gantry.x_beam_profile == "cf_tube_30x20x2":
                x3 = 4
            elif spec.gantry.x_beam_material == "carbon_fiber_torsional_45":
                x3 = 3
            elif spec.gantry.x_beam_material == "carbon_fiber_high_tg":
                x3 = 2
            else:
                x3 = 1
    # x[4] drive mode (CoreXY only)
    x4 = 1 if (spec.architecture == "corexy" and spec.gantry.drive_mode == "awd") else 0
    # x[5] belt
    belt = spec.gantry.belt
    if belt.type == "gt2_12mm":
        x5 = 3
    elif belt.type == "gt2_9mm" and belt.pretension_n >= 55.0:
        x5 = 2
    elif belt.type == "gt2_9mm":
        x5 = 1
    else:
        x5 = 0
    # x[6] stepper
    st = spec.kinematics.xy_stepper
    x6 = 0
    if st.supply_voltage_v >= 36.0:
        if st.steps_per_rev >= 400 or st.preset_id == "ldo_09deg_2504":
            x6 = 5
        elif st.holding_torque_nm >= 0.72 or st.preset_id in ("ldo_60_3004", "kraken_motor_v2", "ldo_60_2004ah"):
            x6 = 4
        elif st.preset_id == "excit3d_max" or (st.rated_current_a >= 3.5 and st.inductance_mh <= 0.8):
            x6 = 3
        elif st.inductance_mh <= 0.8 or st.preset_id == "ldo_2804ah":
            x6 = 2
        else:
            x6 = 1
    # x[7] toolhead
    if spec.toolhead.preset_id in TOOLHEAD_OPTIONS:
        x7 = TOOLHEAD_OPTIONS.index(spec.toolhead.preset_id)
    elif spec.toolhead.total_mass_g <= 255.0:
        x7 = 2
    elif spec.toolhead.total_mass_g <= 325.0:
        x7 = 1
    else:
        x7 = 0
    # x[8] hotend
    mz = spec.toolhead.hotend.melt_zone_length_mm
    pw = spec.toolhead.hotend.heater_power_w
    if mz >= 24.0 or pw >= 90.0:
        x8 = 2
    elif mz >= 19.0 or pw >= 70.0:
        x8 = 1
    else:
        x8 = 0
    # x[9] toolhead mount material
    x9 = _encode_structural_polymer_or_cnc(spec.toolhead.mount_material, TOOLHEAD_MOUNT_OPTIONS)
    # x[10] bed mount type
    x10 = (
        BED_MOUNT_OPTIONS.index(spec.bed.mount_type)
        if spec.bed.mount_type in BED_MOUNT_OPTIONS
        else BED_MOUNT_OPTIONS.index("kinematic_3pt")
    )
    # x[11] gantry bracket material
    x11 = _encode_structural_polymer_or_cnc(spec.gantry.bracket_material, BRACKET_OPTIONS)

    return (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11)


def _build_seed_vectors(baseline_spec: PrinterSpec, population_size: int, seed: int) -> np.ndarray:
    """Build initial population seeded with baseline + smart engineering anchor archetypes."""
    is_corexy = baseline_spec.architecture == "corexy"
    rng = np.random.default_rng(seed)

    base_vec = _encode_baseline_vector(baseline_spec)
    anchors = [
        base_vec,
        # Ultra-budget open frame
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        # Budget lightweight open + compact toolhead + PPA-CF mounts
        (0, 1, 0, 0, 0, 0, 0, 2, 0, 1, 2, 1),
        # Balanced structural ACM + CF X-beam + 9mm FG (48N) + 48V + Archetype + Kinematic 3-pt
        (0, 2, 2, 1, 0, 1, 1, 2, 1, 1, 3, 1),
        # Mid-high 3030 + rear cross + bolted ACM + high-Tg CF + 9mm FG (62N) + low-L 48V + CNC toolhead mount
        (1, 2, 2, 2, 0, 2, 2, 2, 2, 2, 3, 1),
        # Fast AWD (or fast 2WD for non-CoreXY): 3030 + full truss + ACM + Carbonara ±45/0 + AWD + 9mm FG (62N) + 0.9deg 48V
        (1, 3, 2, 3, 1 if is_corexy else 0, 2, 4, 2, 2, 2, 3, 2),
        # High-stiffness AWD Al5052: 3030 + full truss + Al5052 + Wide-Y 30x20 CF + AWD + 12mm FG (80N) + Kraken 48V
        (1, 3, 3, 4, 1 if is_corexy else 0, 3, 3, 2, 2, 2, 3, 2),
        # Max-stiffness 4040 fortress + Al5052 + Wide-Y 30x20 CF + AWD + 12mm FG (80N) + 0.9deg 48V + rigid bed mount
        (2, 3, 3, 4, 1 if is_corexy else 0, 3, 4, 2, 2, 2, 4, 2),
        # Max-stiffness 4040 + bolted CF + Carbonara M40J + AWD + 12mm FG (80N) + Kraken 48V + kinematic 3-pt
        (2, 3, 4, 5, 1 if is_corexy else 0, 3, 3, 2, 2, 2, 3, 2),
    ]

    unique_seeds: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    # Keep at least 2 random slots when population_size >= 6
    max_anchors = max(1, population_size - 2) if population_size >= 6 else population_size
    for vec in anchors:
        norm = _normalize_gene_vector(vec, is_corexy)
        if norm not in seen:
            seen.add(norm)
            unique_seeds.append(norm)
            if len(unique_seeds) >= max_anchors:
                break

    # Fill remainder with uniform random integer samples
    attempts = 0
    while len(unique_seeds) < population_size and attempts < population_size * 20:
        attempts += 1
        rand_vec = tuple(int(rng.integers(_XL[i], _XU[i] + 1)) for i in range(_N_VARS))
        norm = _normalize_gene_vector(rand_vec, is_corexy)
        if norm not in seen:
            seen.add(norm)
            unique_seeds.append(norm)

    while len(unique_seeds) < population_size:
        rand_vec = tuple(int(rng.integers(_XL[i], _XU[i] + 1)) for i in range(_N_VARS))
        unique_seeds.append(_normalize_gene_vector(rand_vec, is_corexy))

    return np.asarray(unique_seeds, dtype=int)


def _decode_candidate(
    baseline_spec: PrinterSpec,
    x: tuple[int, ...],
) -> tuple[PrinterSpec, list[str]]:
    """Apply 12 discrete engineering choices onto a deep copy of baseline_spec and record diffs.

    When a gene matches the baseline's encoded index ``base_vec[i]``, baseline's exact original
    fields for that gene are preserved (exact as-baseline fidelity).
    """
    spec = copy.deepcopy(baseline_spec)
    base_vec = _encode_baseline_vector(baseline_spec)
    changes: list[str] = []

    # x[0]: Frame profile
    if x[0] != base_vec[0]:
        frame_prof = FRAME_PROFILES[x[0]]
        spec.frame.profile = frame_prof
        if frame_prof != baseline_spec.frame.profile:
            short = frame_prof.replace("extrusion_", "")
            changes.append(f"Frame: {short} extrusion")

    # x[1]: Bracing
    if x[1] != base_vec[1]:
        bracing = BRACING_OPTIONS[x[1]]
        spec.frame.bracing = bracing  # type: ignore[assignment]
        if bracing != baseline_spec.frame.bracing:
            changes.append(f"Bracing: {bracing}")

    # x[2]: Panel config
    if x[2] != base_vec[2]:
        panel_id, panel_label, enclosed, mounting, p_mat, p_thick = PANEL_CONFIGS[x[2]]
        spec.frame.enclosed_panels = enclosed
        spec.frame.panel_mounting = mounting  # type: ignore[assignment]
        spec.frame.panel_material = p_mat
        spec.frame.panel_thickness_mm = p_thick
        bolt_gantry = (mounting == "bolted_structural") and (not spec.is_flying_gantry)
        spec.frame.bolt_panels_to_gantry = bolt_gantry

        base_panel_sig = (
            baseline_spec.frame.enclosed_panels,
            baseline_spec.frame.panel_mounting,
            baseline_spec.frame.panel_material if baseline_spec.frame.enclosed_panels else "",
            round(baseline_spec.frame.panel_thickness_mm, 1) if baseline_spec.frame.enclosed_panels else 0.0,
            baseline_spec.panels_bolted_to_gantry_effective,
        )
        cand_panel_sig = (
            spec.frame.enclosed_panels,
            spec.frame.panel_mounting,
            spec.frame.panel_material if spec.frame.enclosed_panels else "",
            round(spec.frame.panel_thickness_mm, 1) if spec.frame.enclosed_panels else 0.0,
            spec.panels_bolted_to_gantry_effective,
        )
        if cand_panel_sig != base_panel_sig:
            suffix = " (+gantry bolts)" if spec.panels_bolted_to_gantry_effective else ""
            changes.append(f"Panels: {panel_label}{suffix}")

    # x[3]: X-beam
    if x[3] != base_vec[3]:
        _, x_beam_label, xb_prof, xb_mat = X_BEAM_OPTIONS[x[3]]
        spec.gantry.x_beam_profile = xb_prof
        spec.gantry.x_beam_material = xb_mat
        if (xb_prof, xb_mat) != (
            baseline_spec.gantry.x_beam_profile,
            baseline_spec.gantry.x_beam_material,
        ):
            changes.append(f"X-beam: {x_beam_label}")

    # x[4]: Drive mode (CoreXY only)
    if spec.architecture == "corexy" and x[4] != base_vec[4]:
        drive_mode = DRIVE_MODES[x[4]]
        spec.gantry.drive_mode = drive_mode  # type: ignore[assignment]
        if drive_mode != baseline_spec.gantry.drive_mode:
            changes.append("Drive: 4-motor AWD" if drive_mode == "awd" else "Drive: 2-motor 2WD")

    # x[5]: Belt setup (+ automatic double-shear pulley support when tension > 45N or AWD)
    if x[5] != base_vec[5]:
        _, belt_label, b_type, b_core, b_pret = BELT_OPTIONS[x[5]]
        spec.gantry.belt.type = b_type  # type: ignore[assignment]
        spec.gantry.belt.core = b_core  # type: ignore[assignment]
        spec.gantry.belt.pretension_n = b_pret
        use_double_shear = (b_pret > 45.0) or (
            spec.architecture == "corexy" and spec.gantry.drive_mode == "awd"
        )
        spec.gantry.double_shear_pulleys = use_double_shear

        base_belt = baseline_spec.gantry.belt
        if (
            b_type != base_belt.type
            or b_core != base_belt.core
            or abs(b_pret - base_belt.pretension_n) > 1.0
            or use_double_shear != baseline_spec.gantry.double_shear_pulleys
        ):
            ds_suffix = " (+double-shear)" if use_double_shear else ""
            changes.append(f"Belt: {belt_label}{ds_suffix}")
    elif (
        spec.architecture == "corexy"
        and spec.gantry.drive_mode == "awd"
        and not spec.gantry.double_shear_pulleys
    ):
        # Upgrading drive_mode to AWD automatically pairs with double-shear pulley support
        spec.gantry.double_shear_pulleys = True

    # x[6]: Stepper + voltage/current
    if x[6] != base_vec[6]:
        st_id, st_v, st_i = STEPPER_SETUPS[x[6]]
        st_data = {
            **STEPPER_CATALOG[st_id],
            "supply_voltage_v": st_v,
            "run_current_a": st_i,
        }
        spec.kinematics.xy_stepper = StepperSpec(**st_data)
        base_st = baseline_spec.kinematics.xy_stepper
        if (
            st_id != base_st.preset_id
            or abs(st_v - base_st.supply_voltage_v) > 1.0
            or abs(st_i - base_st.run_current_a) > 0.05
        ):
            changes.append(f"XY Motors: {st_id} @ {st_v:.0f}V/{st_i:.1f}A")

    # x[7]: Toolhead preset (copy full preset fields including construction, probe, envelope & mount)
    if x[7] != base_vec[7]:
        th_id = TOOLHEAD_OPTIONS[x[7]]
        th_preset = TOOLHEAD_PRESETS[th_id]
        spec.toolhead.preset_id = th_preset["preset_id"]
        spec.toolhead.name = th_preset["name"]
        spec.toolhead.total_mass_g = float(th_preset["total_mass_g"])
        spec.toolhead.cog_offset_mm = tuple(float(v) for v in th_preset["cog_offset_mm"])  # type: ignore[assignment]
        spec.toolhead.nozzle_offset_mm = tuple(float(v) for v in th_preset["nozzle_offset_mm"])  # type: ignore[assignment]
        spec.toolhead.offset_datum_rail_mount = th_preset.get("offset_datum_rail_mount", "front")  # type: ignore[assignment]
        if "body_construction" in th_preset:
            spec.toolhead.body_construction = th_preset["body_construction"]  # type: ignore[assignment]
        if "z_probe_type" in th_preset:
            spec.toolhead.z_probe_type = th_preset["z_probe_type"]  # type: ignore[assignment]
        if "envelope_mm" in th_preset:
            spec.toolhead.envelope_mm = tuple(float(v) for v in th_preset["envelope_mm"])  # type: ignore[assignment]
        if "mount_material" in th_preset:
            spec.toolhead.mount_material = th_preset["mount_material"]  # type: ignore[assignment]
        if "internal_stiffness_n_per_mm" in th_preset:
            spec.toolhead.internal_stiffness_n_per_mm = float(th_preset["internal_stiffness_n_per_mm"])

        if (
            th_id != baseline_spec.toolhead.preset_id
            or abs(spec.toolhead.total_mass_g - baseline_spec.toolhead.total_mass_g) > 1.0
        ):
            changes.append(f"Toolhead: {th_id} ({spec.toolhead.total_mass_g:.0f}g)")

    # x[8]: Hotend catalog preset swap (couples melt zone, heater power, mass, stiffness, & cost)
    if x[8] != base_vec[8]:
        he_id = HOTEND_PRESET_OPTIONS[x[8]]
        spec.toolhead.hotend.preset_id = he_id
        if he_id != baseline_spec.toolhead.hotend.preset_id:
            he_meta = HOTEND_CATALOG[he_id]
            changes.append(
                f"Hotend: {he_meta['short_name']} ({he_meta['melt_zone_length_mm']:.0f}mm / {he_meta['heater_power_w']:.0f}W)"
            )

    # x[9]: Toolhead mount material & internal carriage-to-nozzle stiffness
    if x[9] != base_vec[9]:
        th_mount = TOOLHEAD_MOUNT_OPTIONS[x[9]]
        spec.toolhead.mount_material = th_mount  # type: ignore[assignment]
        spec.toolhead.internal_stiffness_n_per_mm = TOOLHEAD_MOUNT_STIFFNESS_N_PER_MM[th_mount]
        if th_mount != baseline_spec.toolhead.mount_material:
            changes.append(f"toolhead mount -> {th_mount}")

    # x[10]: Bed mount type
    if x[10] != base_vec[10]:
        bed_mount = BED_MOUNT_OPTIONS[x[10]]
        spec.bed.mount_type = bed_mount  # type: ignore[assignment]
        if bed_mount != baseline_spec.bed.mount_type:
            changes.append(f"bed mount -> {bed_mount}")

    # x[11]: Gantry bracket material
    if x[11] != base_vec[11]:
        bracket_mat = BRACKET_OPTIONS[x[11]]
        spec.gantry.bracket_material = bracket_mat  # type: ignore[assignment]
        if bracket_mat != baseline_spec.gantry.bracket_material:
            changes.append(f"brackets -> {bracket_mat}")

    if not changes:
        changes.append("Baseline (unchanged)")

    return spec, changes


@dataclass
class _EvaluatedCandidate:
    point: ParetoPoint
    f: tuple[float, float, float]  # (cost, -safe_accel, deflection)
    g: tuple[float, float, float, float, float]  # (collision, overtension, shaft_bending, step_skip, belt_slip)
    feasible: bool


def _evaluate_spec_candidate(
    spec: PrinterSpec,
    changes: list[str],
    filament_id: str,
    point_id: str,
    label: str = "",
) -> _EvaluatedCandidate:
    report: SimulationReport = evaluate_printer(
        spec,
        filament_id=filament_id,
        panel_mesh_n=2,
        include_advanced_analysis=False,
    )
    if report.bom_total_cost_usd is not None and report.bom_total_mass_kg is not None:
        cost = float(report.bom_total_cost_usd)
        bom_mass_kg = float(report.bom_total_mass_kg)
    else:
        bom: PrinterBOM = generate_bom(report.assembly)
        cost = float(bom.total_cost_usd)
        bom_mass_kg = float(bom.total_mass_kg)

    # Consistent 3-way 1.0x achievable acceleration ceiling: min(shaper, motor, belt)
    safe_accel = min(
        float(report.modal_fea.recommended_max_accel_mm_s2),
        float(report.kinematics.motor_max_accel_mm_s2),
        float(report.kinematics.belt_max_accel_mm_s2),
    )

    # Evaluate kinematic constraints at the advertised safe_accel (not baseline max_accel_mm_s2).
    # Use a tiny 1e-9 interior tolerance so candidates exactly on the motor/belt boundary do not
    # trip due to 1-ULP floating-point rounding.
    eval_accel = max(safe_accel * (1.0 - 1e-9), 1.0)
    spec_at_safe = spec.model_copy(
        update={
            "kinematics": spec.kinematics.model_copy(
                update={"max_accel_mm_s2": eval_accel}
            )
        }
    )
    kin_at_safe = analyze_kinematics(replace(report.assembly, spec=spec_at_safe))

    # Inertial-only dynamic nozzle deflection normalized to fixed 10,000 mm/s^2 reference accel:
    # excludes static gravity sag (C-01) and decouples from candidate safe_accel (A-10).
    raw_inertial_defl = max(
        float(report.static_fea.accel_deflection_x_um),
        float(report.static_fea.accel_deflection_y_um),
    )
    spec_accel = max(float(spec.kinematics.max_accel_mm_s2), 1.0)
    defl = raw_inertial_defl * (_PARETO_REF_ACCEL_MM_S2 / spec_accel)

    g1 = 1.0 if report.collision.has_collision else -1.0
    g2 = 1.0 if kin_at_safe.belt_overtension_risk else -1.0
    g3 = 1.0 if kin_at_safe.shaft_bending_risk else -1.0
    g4 = 1.0 if kin_at_safe.step_skip_risk else -1.0
    g5 = 1.0 if kin_at_safe.belt_slip_risk else -1.0
    feasible = (g1 <= 0.0) and (g2 <= 0.0) and (g3 <= 0.0) and (g4 <= 0.0) and (g5 <= 0.0)

    primary_subsystem: str | None = None
    if report.sensitivity is not None:
        primary_subsystem = report.sensitivity.primary_problem_subsystem
    elif report.modal_fea.modes:
        worst_mode = next(
            (
                m
                for m in report.modal_fea.modes
                if m.mode_number == report.modal_fea.worst_print_mode_number
            ),
            report.modal_fea.modes[0],
        )
        primary_subsystem = worst_mode.dominant_problem_area

    point = ParetoPoint(
        id=point_id,
        label=label,
        changes=changes,
        total_cost_usd=cost,
        safe_max_accel_mm_s2=safe_accel,
        nozzle_deflection_um=defl,
        fundamental_freq_hz=float(report.modal_fea.fundamental_freq_hz),
        max_flow_mm3_s=float(report.thermal.max_volumetric_flow_mm3_s),
        total_mass_kg=bom_mass_kg,
        composite_score=float(report.scores.composite),
        spec=spec,
        primary_problem_subsystem=primary_subsystem,
    )
    return _EvaluatedCandidate(
        point=point,
        f=(cost, -safe_accel, defl),
        g=(g1, g2, g3, g4, g5),
        feasible=feasible,
    )


# ---------------------------------------------------------------------------
# Non-Dominated Sorting, Spacing & Friendly Labeling
# ---------------------------------------------------------------------------


def _dominates(fa: tuple[float, float, float], fb: tuple[float, float, float]) -> bool:
    """Return True if objective tuple fa strictly dominates fb (all minimized)."""
    return (
        fa[0] <= fb[0]
        and fa[1] <= fb[1]
        and fa[2] <= fb[2]
        and (fa[0] < fb[0] or fa[1] < fb[1] or fa[2] < fb[2])
    )


def _extract_pareto_front(
    candidates: list[_EvaluatedCandidate],
    baseline_spec: PrinterSpec,
    max_points: int = MAX_PARETO_POINTS,
) -> list[ParetoPoint]:
    """Extract unique feasible non-dominated points, space to <= max_points, sort by cost, and label."""
    feasible = [c for c in candidates if c.feasible]
    pool = feasible if feasible else list(candidates)
    if not pool:
        return []

    # 1. Deduplicate identical objective tuples (rounded display precision), keeping highest composite_score
    by_obj: dict[tuple[float, float, float], _EvaluatedCandidate] = {}
    for cand in pool:
        key = (
            round(cand.point.total_cost_usd, 2),
            round(cand.point.safe_max_accel_mm_s2, 1),
            round(cand.point.nozzle_deflection_um, 2),
        )
        prev = by_obj.get(key)
        if prev is None or (
            cand.point.composite_score,
            cand.point.max_flow_mm3_s,
            cand.point.fundamental_freq_hz,
        ) > (
            prev.point.composite_score,
            prev.point.max_flow_mm3_s,
            prev.point.fundamental_freq_hz,
        ):
            by_obj[key] = cand

    deduped = list(by_obj.values())

    # 2. Strict Pareto non-dominated filtering (check both raw and rounded values for airtight invariance)
    nondom: list[_EvaluatedCandidate] = []
    for i, ci in enumerate(deduped):
        dominated = False
        fi_raw = ci.f
        fi_rnd = (
            round(ci.point.total_cost_usd, 2),
            -round(ci.point.safe_max_accel_mm_s2, 1),
            round(ci.point.nozzle_deflection_um, 2),
        )
        for j, cj in enumerate(deduped):
            if i == j:
                continue
            fj_raw = cj.f
            fj_rnd = (
                round(cj.point.total_cost_usd, 2),
                -round(cj.point.safe_max_accel_mm_s2, 1),
                round(cj.point.nozzle_deflection_um, 2),
            )
            if _dominates(fj_raw, fi_raw) or _dominates(fj_rnd, fi_rnd):
                dominated = True
                break
        if not dominated:
            nondom.append(ci)

    # Sort ascending by total_cost_usd (tie-break by accel desc, deflection asc)
    nondom.sort(
        key=lambda c: (
            c.point.total_cost_usd,
            -c.point.safe_max_accel_mm_s2,
            c.point.nozzle_deflection_um,
        )
    )

    # 3. Compute normalized [0, 1]^3 objective coordinates relative to utopia (0, 0, 0)
    def _normalized_coords(items: list[_EvaluatedCandidate]) -> np.ndarray:
        costs = np.array([c.point.total_cost_usd for c in items], dtype=float)
        accels = np.array([c.point.safe_max_accel_mm_s2 for c in items], dtype=float)
        defls = np.array([c.point.nozzle_deflection_um for c in items], dtype=float)

        c_span = max(float(costs.max() - costs.min()), 1e-9)
        a_span = max(float(accels.max() - accels.min()), 1e-9)
        d_span = max(float(defls.max() - defls.min()), 1e-9)

        c_norm = (costs - costs.min()) / c_span
        a_norm = (accels.max() - accels) / a_span
        d_norm = (defls - defls.min()) / d_span
        return np.column_stack([c_norm, a_norm, d_norm])

    # 4. Cap to max_points using farthest-point spacing while preserving extremes, knee & top score
    if len(nondom) > max_points:
        coords = _normalized_coords(nondom)
        utopia_dists = np.linalg.norm(coords, axis=1)

        must_keep: set[int] = {
            int(np.argmin([c.point.total_cost_usd for c in nondom])),
            int(np.argmax([c.point.safe_max_accel_mm_s2 for c in nondom])),
            int(np.argmin([c.point.nozzle_deflection_um for c in nondom])),
            int(np.argmin(utopia_dists)),
            int(np.argmax([c.point.composite_score for c in nondom])),
        }
        selected = sorted(must_keep)
        remaining = [idx for idx in range(len(nondom)) if idx not in must_keep]

        while len(selected) < max_points and remaining:
            sel_coords = coords[selected]
            best_idx = -1
            best_min_dist = -1.0
            for idx in remaining:
                dists = np.linalg.norm(sel_coords - coords[idx], axis=1)
                min_d = float(dists.min())
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_idx = idx
            selected.append(best_idx)
            remaining.remove(best_idx)

        nondom = [nondom[idx] for idx in sorted(selected)]
        nondom.sort(
            key=lambda c: (
                c.point.total_cost_usd,
                -c.point.safe_max_accel_mm_s2,
                c.point.nozzle_deflection_um,
            )
        )

    # 5. Assign friendly standout labels
    n = len(nondom)
    coords = _normalized_coords(nondom)
    utopia_dists = np.linalg.norm(coords, axis=1)

    idx_budget = int(np.argmin([c.point.total_cost_usd for c in nondom]))
    idx_fastest = int(np.argmax([c.point.safe_max_accel_mm_s2 for c in nondom]))
    idx_stiffest = int(np.argmin([c.point.nozzle_deflection_um for c in nondom]))
    idx_top_score = int(np.argmax([c.point.composite_score for c in nondom]))

    labels: dict[int, str] = {}
    labels[idx_budget] = "Best Budget"
    if idx_fastest not in labels:
        labels[idx_fastest] = "Fastest Safe Accel"
    if idx_stiffest not in labels:
        labels[idx_stiffest] = "Stiffest Nozzle"

    # Pick knee closest to utopia (preferring unlabeled interior point if extremes collided)
    knee_order = [int(i) for i in np.argsort(utopia_dists)]
    knee_idx = next((i for i in knee_order if i not in labels), knee_order[0])
    if knee_idx not in labels:
        labels[knee_idx] = "Sweet Spot (Knee)"

    if idx_top_score not in labels:
        labels[idx_top_score] = "Highest Overall Score"

    front: list[ParetoPoint] = []
    for i, cand in enumerate(nondom):
        label = labels.get(i, f"Pareto #{i + 1}")
        pt_spec = copy.deepcopy(cand.point.spec)
        if cand.point.changes != ["Baseline (unchanged)"]:
            pt_spec.name = f"{baseline_spec.name} — {label}"
        front.append(
            ParetoPoint(
                id=f"pareto_{i + 1:02d}",
                label=label,
                changes=list(cand.point.changes),
                total_cost_usd=cand.point.total_cost_usd,
                safe_max_accel_mm_s2=cand.point.safe_max_accel_mm_s2,
                nozzle_deflection_um=cand.point.nozzle_deflection_um,
                fundamental_freq_hz=cand.point.fundamental_freq_hz,
                max_flow_mm3_s=cand.point.max_flow_mm3_s,
                total_mass_kg=cand.point.total_mass_kg,
                composite_score=cand.point.composite_score,
                spec=pt_spec,
                primary_problem_subsystem=cand.point.primary_problem_subsystem,
            )
        )
    return front


# ---------------------------------------------------------------------------
# Pure-NumPy NSGA-II Fallback (used only if pymoo is unavailable)
# ---------------------------------------------------------------------------


def _fast_nondominated_sort_indices(
    objs: list[tuple[float, float, float]],
    feasible_flags: list[bool],
) -> list[list[int]]:
    """Deb's constrained non-dominated sorting (feasible solutions always beat infeasible)."""
    n = len(objs)

    def _constrained_dominates(i: int, j: int) -> bool:
        if feasible_flags[i] and not feasible_flags[j]:
            return True
        if not feasible_flags[i] and feasible_flags[j]:
            return False
        return _dominates(objs[i], objs[j])

    dom_sets: list[list[int]] = [[] for _ in range(n)]
    dom_counts = [0] * n
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _constrained_dominates(p, q):
                dom_sets[p].append(q)
            elif _constrained_dominates(q, p):
                dom_counts[p] += 1
        if dom_counts[p] == 0:
            fronts[0].append(p)

    idx = 0
    while idx < len(fronts) and fronts[idx]:
        next_front: list[int] = []
        for p in fronts[idx]:
            for q in dom_sets[p]:
                dom_counts[q] -= 1
                if dom_counts[q] == 0:
                    next_front.append(q)
        idx += 1
        if next_front:
            fronts.append(next_front)

    return [f for f in fronts if f]


def _crowding_distance(
    front: list[int],
    objs: list[tuple[float, float, float]],
) -> dict[int, float]:
    dist = {i: 0.0 for i in front}
    if len(front) <= 2:
        for i in front:
            dist[i] = float("inf")
        return dist
    for m in range(3):
        ordered = sorted(front, key=lambda idx: objs[idx][m])
        dist[ordered[0]] = float("inf")
        dist[ordered[-1]] = float("inf")
        span = objs[ordered[-1]][m] - objs[ordered[0]][m]
        if span <= 1e-12:
            continue
        for k in range(1, len(ordered) - 1):
            dist[ordered[k]] += (objs[ordered[k + 1]][m] - objs[ordered[k - 1]][m]) / span
    return dist


def _run_numpy_nsga2(
    baseline_spec: PrinterSpec,
    eval_fn,
    population_size: int,
    generations: int,
    seed: int,
) -> None:
    """Clean pure-NumPy NSGA-II fallback operating on the discrete 12-gene vector."""
    is_corexy = baseline_spec.architecture == "corexy"
    rng = np.random.default_rng(seed)

    pop = [tuple(int(v) for v in row) for row in _build_seed_vectors(baseline_spec, population_size, seed)]
    for x in pop:
        eval_fn(x)

    for _ in range(max(generations - 1, 0)):
        # Create offspring via uniform crossover + integer mutation
        offspring: list[tuple[int, ...]] = []
        for _k in range(population_size):
            p1 = pop[int(rng.integers(0, len(pop)))]
            p2 = pop[int(rng.integers(0, len(pop)))]
            child = list(p1)
            for g in range(_N_VARS):
                if rng.random() < 0.5:
                    child[g] = p2[g]
                if rng.random() < 0.22:
                    child[g] = int(rng.integers(_XL[g], _XU[g] + 1))
            norm = _normalize_gene_vector(child, is_corexy)
            eval_fn(norm)
            offspring.append(norm)

        # Combine & deduplicate genes
        combined = list(dict.fromkeys(pop + offspring))
        cands = [eval_fn(x) for x in combined]
        objs = [c.f for c in cands]
        feas = [c.feasible for c in cands]

        fronts = _fast_nondominated_sort_indices(objs, feas)
        next_pop: list[tuple[int, ...]] = []
        for front in fronts:
            if len(next_pop) + len(front) <= population_size:
                next_pop.extend(combined[i] for i in front)
            else:
                cd = _crowding_distance(front, objs)
                ranked = sorted(front, key=lambda i: cd[i], reverse=True)
                slots = population_size - len(next_pop)
                next_pop.extend(combined[i] for i in ranked[:slots])
                break
        pop = next_pop


# ---------------------------------------------------------------------------
# Public Entry Point
# ---------------------------------------------------------------------------


def optimize_pareto_front(
    baseline_spec: PrinterSpec,
    population_size: int = 16,
    generations: int = 6,
    seed: int = 42,
    filament_id: str = "pla",
) -> ParetoResult:
    """Explore the Speed vs. Stiffness vs. Build Cost Pareto frontier using NSGA-II.

    Parameters
    ----------
    baseline_spec:
        Starting 3D printer specification. Architecture, build volume, and outer
        frame dimensions are preserved while 12 hardware upgrade genes are optimized.
    population_size:
        NSGA-II population size (minimum 4, default 16).
    generations:
        Number of evolutionary generations (minimum 1, default 6).
    seed:
        Random seed for deterministic, reproducible optimization.
    filament_id:
        Filament material ID passed to ``evaluate_printer`` (default ``"pla"``).
    """
    pop_size = max(4, int(population_size))
    n_gen = max(1, int(generations))
    is_corexy = baseline_spec.architecture == "corexy"

    # Evaluate exact unmodified baseline first for faithful side-by-side comparison
    baseline_cand = _evaluate_spec_candidate(
        spec=copy.deepcopy(baseline_spec),
        changes=["Baseline (unchanged)"],
        filament_id=filament_id,
        point_id="baseline",
        label="Baseline",
    )

    # Memoized discrete gene evaluation cache
    cache: dict[tuple[int, ...], _EvaluatedCandidate] = {}

    def _eval_gene(x_raw: np.ndarray | list[int] | tuple[int, ...]) -> _EvaluatedCandidate:
        key = _normalize_gene_vector(x_raw, is_corexy)
        hit = cache.get(key)
        if hit is not None:
            return hit
        cand_spec, changes = _decode_candidate(baseline_spec, key)
        cand = _evaluate_spec_candidate(
            spec=cand_spec,
            changes=changes,
            filament_id=filament_id,
            point_id=f"cand_{len(cache) + 1:03d}",
        )
        cache[key] = cand
        return cand

    if is_pymoo_available():
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.core.sampling import Sampling
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.repair.rounding import RoundingRepair
        from pymoo.operators.sampling.rnd import IntegerRandomSampling
        from pymoo.optimize import minimize

        class _SeededIntegerSampling(Sampling):
            def _do(self, problem, n_samples, **kwargs):
                X_seed = _build_seed_vectors(baseline_spec, n_samples, seed)
                if len(X_seed) < n_samples:
                    extra = IntegerRandomSampling()._do(problem, n_samples - len(X_seed), **kwargs)
                    X_seed = np.vstack([X_seed, extra])
                return X_seed[:n_samples]

        class _PrinterParetoProblem(ElementwiseProblem):
            def __init__(self) -> None:
                super().__init__(
                    n_var=_N_VARS,
                    n_obj=3,
                    n_ieq_constr=5,
                    xl=_XL,
                    xu=_XU,
                    vtype=int,
                )

            def _evaluate(self, x, out, *args, **kwargs):
                cand = _eval_gene(x)
                out["F"] = list(cand.f)
                out["G"] = list(cand.g)

        problem = _PrinterParetoProblem()
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=_SeededIntegerSampling(),
            crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
            mutation=PM(prob=0.25, eta=20, vtype=float, repair=RoundingRepair()),
            eliminate_duplicates=True,
        )
        minimize(
            problem,
            algorithm,
            termination=("n_gen", n_gen),
            seed=seed,
            verbose=False,
        )
        backend = "pymoo (NSGA-II)"
    else:
        _run_numpy_nsga2(
            baseline_spec=baseline_spec,
            eval_fn=_eval_gene,
            population_size=pop_size,
            generations=n_gen,
            seed=seed,
        )
        backend = "numpy (NSGA-II fallback)"

    # Combine all unique evaluated candidates + exact baseline for non-dominated extraction
    all_candidates = [baseline_cand, *cache.values()]
    pareto_front = _extract_pareto_front(
        candidates=all_candidates,
        baseline_spec=baseline_spec,
        max_points=MAX_PARETO_POINTS,
    )

    return ParetoResult(
        backend=backend,
        evaluations=len(cache) + 1,
        population_size=pop_size,
        generations=n_gen,
        baseline=baseline_cand.point,
        pareto_front=pareto_front,
    )
