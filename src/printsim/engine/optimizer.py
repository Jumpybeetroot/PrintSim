"""Headless AI / evolutionary design exploration loop for optimizing 3D printer builds."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Literal

from printsim.engine.evaluator import SimulationReport, evaluate_printer
from printsim.schema import PrinterSpec

Objective = Literal["balanced", "speed", "rigidity", "budget"]


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
    if report.collision.has_collision:
        return report.scores.composite - 30.0
    if objective == "speed":
        # Prioritize high print speed, high recommended acceleration, high melt flow, and quality
        time_bonus = max(0.0, 40.0 - report.virtual_print.total_time_s * 0.25)
        return (
            0.35 * report.scores.dynamics
            + 0.25 * report.scores.thermal
            + 0.25 * report.scores.print_quality
            + 0.15 * min(100.0, time_bonus * 2.5)
        )
    if objective == "rigidity":
        return 0.65 * report.scores.structural + 0.20 * report.scores.dynamics + 0.15 * report.scores.print_quality
    if objective == "budget":
        return 0.45 * report.scores.efficiency + 0.30 * report.scores.print_quality + 0.25 * report.scores.structural
    return report.scores.composite


def _mutate_spec(parent: PrinterSpec, rng: random.Random) -> tuple[PrinterSpec, list[str]]:
    child = copy.deepcopy(parent)
    changes: list[str] = []

    n_mutations = rng.randint(1, 3)
    mutation_pool = [
        "frame_profile",
        "bracing",
        "enclosed_panels",
        "x_beam",
        "belt",
        "hotend_melt",
        "hotend_heater",
        "heatbreak",
        "toolhead_mass",
        "accel",
    ]
    chosen = rng.sample(mutation_pool, n_mutations)

    for m in chosen:
        if m == "frame_profile":
            val = rng.choice(["extrusion_2020", "extrusion_2040", "extrusion_3030", "extrusion_4040", "al_tube_30x30x3"])
            if val != child.frame.profile:
                child.frame.profile = val
                changes.append(f"frame.profile={val}")
        elif m == "bracing":
            val = rng.choice(["none", "corner_gussets", "rear_cross", "full_truss"])
            if val != child.frame.bracing:
                child.frame.bracing = val  # type: ignore[assignment]
                changes.append(f"frame.bracing={val}")
        elif m == "enclosed_panels":
            child.frame.enclosed_panels = not child.frame.enclosed_panels
            changes.append(f"frame.enclosed_panels={child.frame.enclosed_panels}")
        elif m == "x_beam":
            prof, mat = rng.choice(
                [
                    ("extrusion_2020", "aluminum_6063_t5"),
                    ("cf_tube_20x20x1_5", "carbon_fiber_tube"),
                    ("cf_tube_20x20x2", "carbon_fiber_tube"),
                    ("cf_tube_25x25x2", "carbon_fiber_tube"),
                    ("al_tube_20x20x2", "aluminum_6061_t6"),
                ]
            )
            child.gantry.x_beam_profile = prof
            child.gantry.x_beam_material = mat
            changes.append(f"gantry.x_beam={prof}({mat})")
        elif m == "belt":
            btype = rng.choice(["gt2_6mm", "gt2_9mm", "gt2_12mm"])
            core = rng.choice(["fiberglass", "aramid", "steel"])
            pret = rng.choice([25.0, 35.0, 45.0])
            child.gantry.belt.type = btype  # type: ignore[assignment]
            child.gantry.belt.core = core  # type: ignore[assignment]
            child.gantry.belt.pretension_n = pret
            changes.append(f"belt={btype}/{core}@{pret:.0f}N")
        elif m == "hotend_melt":
            val = rng.choice([12.0, 16.0, 20.0, 24.0])
            child.toolhead.hotend.melt_zone_length_mm = val
            child.toolhead.hotend.block_material = rng.choice(["aluminum_6061_t6", "copper_c110"])
            changes.append(f"hotend.melt={val}mm/{child.toolhead.hotend.block_material}")
        elif m == "hotend_heater":
            val = rng.choice([40.0, 60.0, 80.0, 100.0])
            child.toolhead.hotend.heater_power_w = val
            changes.append(f"hotend.heater={val:.0f}W")
        elif m == "heatbreak":
            mat = rng.choice(["stainless_304", "titanium_grade5"])
            child.toolhead.hotend.heatbreak_material = mat
            changes.append(f"hotend.heatbreak={mat}")
        elif m == "toolhead_mass":
            val = rng.choice([220.0, 280.0, 340.0, 420.0])
            child.toolhead.total_mass_g = val
            changes.append(f"toolhead.mass={val:.0f}g")
        elif m == "accel":
            val = rng.choice([4000.0, 7000.0, 10000.0, 14000.0, 18000.0])
            child.kinematics.max_accel_mm_s2 = val
            changes.append(f"kinematics.max_accel={val:.0f}")

    return child, changes


def optimize_design(
    base_spec: PrinterSpec,
    iterations: int = 20,
    objective: Objective = "balanced",
    seed: int = 42,
) -> OptimizationResult:
    """Run an evolutionary hill-climbing search to improve a 3D printer design."""
    rng = random.Random(seed)
    base_report = evaluate_printer(base_spec, include_advanced_analysis=False)
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
                "cost_usd": round(base_report.assembly.estimated_material_cost_usd, 0),
            },
        )
    ]

    for it in range(1, iterations + 1):
        # 75% exploit best design, 25% explore from baseline
        parent = best_spec if rng.random() < 0.75 else base_spec
        cand_spec, changes = _mutate_spec(parent, rng)
        cand_spec.name = f"{base_spec.name} (Opt #{it})"
        rep = evaluate_printer(cand_spec, include_advanced_analysis=False)
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
                    "cost_usd": round(rep.assembly.estimated_material_cost_usd, 0),
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
