"""Headless AI / evolutionary design exploration loop for optimizing 3D printer builds."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Literal

from printsim.engine.evaluator import SimulationReport, evaluate_printer
from printsim.schema import HOTEND_CATALOG, TOOLHEAD_PRESETS, PrinterSpec

Objective = Literal["balanced", "speed", "rigidity", "budget"]

TOOLHEAD_MOUNT_STIFFNESS_N_PER_MM: dict[str, float] = {
    "abs_printed": 380.0,
    "pet_cf_printed": 650.0,
    "pa12_cf_printed": 800.0,
    "ppa_cf_printed": 1150.0,
    "aluminum_6061_t6": 1850.0,
}

_PANEL_CONFIG_OPTIONS: tuple[tuple[str, str, float], ...] = (
    ("none", "acrylic_pmma", 3.0),
    ("clip_enclosure", "acrylic_pmma", 3.0),
    ("bolted_structural", "acm_dibond", 4.0),
    ("bolted_structural", "aluminum_5052_sheet", 4.0),
)


@dataclass
class OptimizationCandidate:
    iteration: int
    objective_value: float
    composite_score: float
    changes: list[str]
    spec: PrinterSpec
    summary: dict


@dataclass
class OptimizationResult:
    objective: Objective
    iterations: int
    baseline_objective: float
    best_objective: float
    improvement_delta: float
    best_spec: PrinterSpec
    best_report: SimulationReport
    history: list[OptimizationCandidate] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "objective": self.objective,
            "iterations": self.iterations,
            "baseline_objective": round(self.baseline_objective, 2),
            "best_objective": round(self.best_objective, 2),
            "improvement_delta": round(self.improvement_delta, 2),
            "best_spec": self.best_spec.model_dump(mode="json"),
            "best_report": self.best_report.to_dict(include_toolpath_segments=False),
            "top_candidates": [
                {
                    "iteration": c.iteration,
                    "objective_value": round(c.objective_value, 2),
                    "composite_score": round(c.composite_score, 1),
                    "changes": c.changes,
                    "summary": c.summary,
                }
                for c in sorted(self.history, key=lambda x: x.objective_value, reverse=True)[:5]
            ],
        }


def _score_objective(report: SimulationReport, objective: Objective) -> float:
    if objective == "speed":
        # Prioritize high print speed, high recommended acceleration, high melt flow, and quality
        time_bonus = min(40.0, 40.0 * (120.0 / max(report.virtual_print.total_time_s, 30.0)))
        score = (
            0.35 * report.scores.dynamics
            + 0.25 * report.scores.thermal
            + 0.25 * report.scores.print_quality
            + 0.15 * min(100.0, time_bonus * 2.5)
        )
    elif objective == "rigidity":
        score = (
            0.65 * report.scores.structural
            + 0.20 * report.scores.dynamics
            + 0.15 * report.scores.print_quality
        )
    elif objective == "budget":
        score = (
            0.45 * report.scores.efficiency
            + 0.30 * report.scores.print_quality
            + 0.25 * report.scores.structural
        )
    else:
        # report.scores.composite already applies the 30-point collision penalty once
        return report.scores.composite

    if report.collision.has_collision:
        # Apply the same single 30-point collision penalty as composite (without double-counting)
        return max(0.0, score - 30.0)
    return score


def _mutate_spec(parent: PrinterSpec, rng: random.Random) -> tuple[PrinterSpec, list[str]]:
    child = copy.deepcopy(parent)
    changes: list[str] = []

    n_mutations = rng.randint(1, 3)
    mutation_pool = [
        "frame_profile",
        "bracing",
        "panel_config",
        "x_beam",
        "belt",
        "hotend_preset",
        "toolhead_preset",
        "toolhead_mount",
        "bed_mount",
        "gantry_brackets",
        "extrusion_filling",
    ]
    if child.architecture == "corexy":
        mutation_pool.extend(["drive_mode", "bed_z_drive"])
    chosen = rng.sample(mutation_pool, n_mutations)

    for m in chosen:
        if m == "frame_profile":
            opts = ["extrusion_2020", "extrusion_2040", "extrusion_3030", "extrusion_4040", "al_tube_30x30x3"]
            alts = [x for x in opts if x != child.frame.profile] or opts
            val = rng.choice(alts)
            child.frame.profile = val
            changes.append(f"frame.profile={val}")
        elif m == "bracing":
            opts = ["none", "corner_gussets", "rear_cross", "full_truss"]
            alts = [x for x in opts if x != child.frame.bracing] or opts
            val = rng.choice(alts)
            child.frame.bracing = val  # type: ignore[assignment]
            changes.append(f"frame.bracing={val}")
        elif m == "panel_config":
            cur_panel = (
                child.frame.panel_mounting if child.frame.enclosed_panels else "none",
                child.frame.panel_material,
                round(float(child.frame.panel_thickness_mm), 1),
            )
            alts = [opt for opt in _PANEL_CONFIG_OPTIONS if opt != cur_panel] or list(_PANEL_CONFIG_OPTIONS)
            mounting, p_mat, p_thick = rng.choice(alts)
            child.frame.enclosed_panels = mounting != "none"
            child.frame.panel_mounting = mounting  # type: ignore[assignment]
            child.frame.panel_material = p_mat
            child.frame.panel_thickness_mm = p_thick
            child.frame.bolt_panels_to_gantry = (
                mounting == "bolted_structural"
            ) and (not child.is_flying_gantry)
            changes.append(
                "frame.panels=none"
                if mounting == "none"
                else f"frame.panels={mounting}/{p_mat}/{p_thick:.0f}mm"
            )
        elif m == "x_beam":
            opts = [
                ("extrusion_2020", "aluminum_6063_t5"),
                ("cf_tube_20x20x1_5", "carbon_fiber_tube"),
                ("cf_tube_20x20x2", "carbon_fiber_tube"),
                ("cf_tube_20x20x2", "carbon_fiber_high_tg"),
                ("cf_tube_20x20_carbonara", "carbon_fiber_torsional_45"),
                ("cf_tube_30x20x2", "carbon_fiber_torsional_45"),
                ("cf_tube_20x20_carbonara", "carbon_fiber_m40j_hm"),
                ("cf_tube_25x25x2", "carbon_fiber_tube"),
                ("al_tube_20x20x2", "aluminum_6061_t6"),
            ]
            cur_xb = (child.gantry.x_beam_profile, child.gantry.x_beam_material)
            alts = [x for x in opts if x != cur_xb] or opts
            prof, mat = rng.choice(alts)
            child.gantry.x_beam_profile = prof
            child.gantry.x_beam_material = mat
            changes.append(f"gantry.x_beam={prof}({mat})")
        elif m == "belt":
            btype = rng.choice(["gt2_6mm", "gt2_9mm", "gt2_12mm"])
            core = "fiberglass"
            pret_opts = (
                [25.0, 35.0, 45.0]
                if btype == "gt2_6mm"
                else ([35.0, 48.0, 62.0] if btype == "gt2_9mm" else [45.0, 65.0, 80.0])
            )
            pret = rng.choice(pret_opts)
            child.gantry.belt.type = btype  # type: ignore[assignment]
            child.gantry.belt.core = core  # type: ignore[assignment]
            child.gantry.belt.pretension_n = pret
            if pret > 45.0:
                child.gantry.double_shear_pulleys = True
            changes.append(f"belt={btype}/{core}@{pret:.0f}N")
        elif m == "hotend_preset":
            hotend_ids = list(HOTEND_CATALOG.keys())
            cur_he = child.toolhead.hotend.preset_id
            alts = [hid for hid in hotend_ids if hid != cur_he] or hotend_ids
            pid = rng.choice(alts)
            child.toolhead.hotend.preset_id = pid  # auto-populates full catalog physics/mass/length/stiffness
            changes.append(f"hotend.preset={pid}")
        elif m == "toolhead_preset":
            th_ids = list(TOOLHEAD_PRESETS.keys())
            cur_th = child.toolhead.preset_id
            alts = [tid for tid in th_ids if tid != cur_th] or th_ids
            tid = rng.choice(alts)
            preset = TOOLHEAD_PRESETS[tid]
            child.toolhead.preset_id = preset["preset_id"]
            child.toolhead.name = preset["name"]
            child.toolhead.total_mass_g = float(preset["total_mass_g"])
            child.toolhead.cog_offset_mm = tuple(float(v) for v in preset["cog_offset_mm"])  # type: ignore[assignment]
            child.toolhead.nozzle_offset_mm = tuple(float(v) for v in preset["nozzle_offset_mm"])  # type: ignore[assignment]
            if "body_construction" in preset:
                child.toolhead.body_construction = preset["body_construction"]  # type: ignore[assignment]
            if "z_probe_type" in preset:
                child.toolhead.z_probe_type = preset["z_probe_type"]  # type: ignore[assignment]
            if "envelope_mm" in preset:
                child.toolhead.envelope_mm = tuple(float(v) for v in preset["envelope_mm"])  # type: ignore[assignment]
            if "mount_material" in preset:
                child.toolhead.mount_material = preset["mount_material"]  # type: ignore[assignment]
            child.toolhead.internal_stiffness_n_per_mm = TOOLHEAD_MOUNT_STIFFNESS_N_PER_MM.get(
                child.toolhead.mount_material,
                float(preset.get("internal_stiffness_n_per_mm", child.toolhead.internal_stiffness_n_per_mm)),
            )
            changes.append(f"toolhead.preset={tid}({child.toolhead.total_mass_g:.0f}g)")
        elif m == "toolhead_mount":
            mount_mats = list(TOOLHEAD_MOUNT_STIFFNESS_N_PER_MM.keys())
            alts = [x for x in mount_mats if x != child.toolhead.mount_material] or mount_mats
            mat = rng.choice(alts)
            k_n_mm = TOOLHEAD_MOUNT_STIFFNESS_N_PER_MM[mat]
            child.toolhead.mount_material = mat  # type: ignore[assignment]
            child.toolhead.internal_stiffness_n_per_mm = k_n_mm
            changes.append(f"toolhead.mount={mat}({k_n_mm:.0f}N/mm)")
        elif m == "bed_mount":
            bed_mounts = ["kinematic_3pt", "floating_4pt", "silicone_4pt", "rigid_4pt"]
            alts = [x for x in bed_mounts if x != child.bed.mount_type] or bed_mounts
            val = rng.choice(alts)
            child.bed.mount_type = val  # type: ignore[assignment]
            changes.append(f"bed.mount_type={val}")
        elif m == "gantry_brackets":
            bkt_mats = ["abs_printed", "ppa_cf_printed", "aluminum_6061_t6"]
            alts = [x for x in bkt_mats if x != child.gantry.bracket_material] or bkt_mats
            val = rng.choice(alts)
            child.gantry.bracket_material = val  # type: ignore[assignment]
            changes.append(f"gantry.bracket_material={val}")
        elif m == "extrusion_filling":
            fillings = ["none", "silica_sand", "epoxy_granite"]
            alts = [x for x in fillings if x != child.frame.extrusion_filling] or fillings
            val = rng.choice(alts)
            child.frame.extrusion_filling = val  # type: ignore[assignment]
            changes.append(f"frame.extrusion_filling={val}")
        elif m == "drive_mode":
            val = "awd" if child.gantry.drive_mode != "awd" else "2wd"
            child.gantry.drive_mode = val  # type: ignore[assignment]
            if val == "awd":
                child.gantry.double_shear_pulleys = True
            changes.append(f"gantry.drive_mode={val}")
        elif m == "bed_z_drive":
            z_opts = ["leadscrew_triple", "belted_quad_z"]
            alts = [x for x in z_opts if x != child.bed.z_drive] or z_opts
            val = rng.choice(alts)
            child.bed.z_drive = val  # type: ignore[assignment]
            if child.frame.enclosed_panels and child.frame.panel_mounting == "bolted_structural":
                child.frame.bolt_panels_to_gantry = not child.is_flying_gantry
            changes.append(f"bed.z_drive={val}")

    return child, changes


def optimize_design(
    base_spec: PrinterSpec,
    iterations: int = 20,
    objective: Objective = "balanced",
    seed: int = 42,
) -> OptimizationResult:
    """Run an evolutionary hill-climbing search to improve a 3D printer design."""
    rng = random.Random(seed)
    base_report = evaluate_printer(base_spec, panel_mesh_n=2, include_advanced_analysis=False)
    base_obj = _score_objective(base_report, objective)

    best_spec = copy.deepcopy(base_spec)
    best_report = base_report
    best_obj = base_obj

    history: list[OptimizationCandidate] = [
        OptimizationCandidate(
            iteration=0,
            objective_value=base_obj,
            composite_score=base_report.scores.composite,
            changes=["baseline"],
            spec=copy.deepcopy(base_spec),
            summary={
                "fundamental_hz": round(base_report.modal_fea.fundamental_freq_hz, 1),
                "rec_accel": round(base_report.modal_fea.recommended_max_accel_mm_s2, 0),
                "max_flow": round(base_report.thermal.max_volumetric_flow_mm3_s, 1),
                "print_time_s": round(base_report.virtual_print.total_time_s, 1),
                "cost_usd": round(
                    base_report.bom_total_cost_usd
                    if base_report.bom_total_cost_usd is not None
                    else base_report.assembly.estimated_material_cost_usd,
                    0,
                ),
            },
        )
    ]

    for it in range(1, iterations + 1):
        # 75% exploit best design, 25% explore from baseline
        parent = best_spec if rng.random() < 0.75 else base_spec
        cand_spec, changes = _mutate_spec(parent, rng)
        cand_spec.name = f"{base_spec.name} (Opt #{it})"
        rep = evaluate_printer(cand_spec, panel_mesh_n=2, include_advanced_analysis=False)
        val = _score_objective(rep, objective)

        history.append(
            OptimizationCandidate(
                iteration=it,
                objective_value=val,
                composite_score=rep.scores.composite,
                changes=changes,
                spec=cand_spec,
                summary={
                    "fundamental_hz": round(rep.modal_fea.fundamental_freq_hz, 1),
                    "rec_accel": round(rep.modal_fea.recommended_max_accel_mm_s2, 0),
                    "max_flow": round(rep.thermal.max_volumetric_flow_mm3_s, 1),
                    "print_time_s": round(rep.virtual_print.total_time_s, 1),
                    "cost_usd": round(
                        rep.bom_total_cost_usd
                        if rep.bom_total_cost_usd is not None
                        else rep.assembly.estimated_material_cost_usd,
                        0,
                    ),
                },
            )
        )
        if val > best_obj:
            best_obj = val
            best_spec = cand_spec
            best_report = rep

    best_report = evaluate_printer(best_spec)
    return OptimizationResult(
        objective=objective,
        iterations=iterations,
        baseline_objective=base_obj,
        best_objective=best_obj,
        improvement_delta=best_obj - base_obj,
        best_spec=best_spec,
        best_report=best_report,
        history=history,
    )
