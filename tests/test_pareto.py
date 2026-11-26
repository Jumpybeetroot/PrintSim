"""Tests for the pymoo (NSGA-II) multi-objective Pareto design optimizer."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from printsim.engine import (
    ParetoPoint,
    ParetoResult,
    is_pymoo_available,
    optimize_pareto_front,
)
from printsim.engine import pareto as pareto_mod
from printsim.schema import PrinterSpec

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _assert_pareto_invariants(front: list[ParetoPoint]) -> None:
    """Assert ascending cost sorting and strict non-dominance on raw & rounded objectives."""
    assert len(front) >= 1
    assert len(front) <= pareto_mod.MAX_PARETO_POINTS

    # Sorted by ascending total_cost_usd
    costs = [p.total_cost_usd for p in front]
    assert costs == sorted(costs), "Pareto front must be sorted by ascending total_cost_usd"

    # Distinct IDs and distinct objective tuples
    assert len({p.id for p in front}) == len(front)

    # Strict non-dominance across (cost, -safe_accel, nozzle_deflection)
    for i, a in enumerate(front):
        for j, b in enumerate(front):
            if i == j:
                continue
            a_dom_b = (
                a.total_cost_usd <= b.total_cost_usd
                and a.safe_max_accel_mm_s2 >= b.safe_max_accel_mm_s2
                and a.nozzle_deflection_um <= b.nozzle_deflection_um
                and (
                    a.total_cost_usd < b.total_cost_usd
                    or a.safe_max_accel_mm_s2 > b.safe_max_accel_mm_s2
                    or a.nozzle_deflection_um < b.nozzle_deflection_um
                )
            )
            assert not a_dom_b, (
                f"Pareto point {a.id} ({a.total_cost_usd}, {a.safe_max_accel_mm_s2}, {a.nozzle_deflection_um}) "
                f"strictly dominates {b.id} ({b.total_cost_usd}, {b.safe_max_accel_mm_s2}, {b.nozzle_deflection_um})"
            )


def test_pymoo_available():
    if not is_pymoo_available():
        pytest.skip("Optional pymoo extra not installed (`uv sync --extra opt`)")
    assert is_pymoo_available() is True


def test_voron_trident_pareto_front():
    import os

    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")

    t0 = time.perf_counter()
    result = optimize_pareto_front(spec, population_size=14, generations=5, seed=42)
    elapsed_s = time.perf_counter() - t0

    assert isinstance(result, ParetoResult)
    expected_backend = "pymoo (NSGA-II)" if is_pymoo_available() else "numpy (NSGA-II fallback)"
    assert result.backend == expected_backend
    budget_s = 30.0 if os.environ.get("CI") else 14.0
    assert elapsed_s < budget_s, f"Pareto optimization took too long: {elapsed_s:.2f}s"
    assert result.evaluations >= 14
    assert len(result.pareto_front) >= 4

    _assert_pareto_invariants(result.pareto_front)

    # Frontier must strictly beat Stock Voron Trident on both speed and nozzle stiffness
    max_accel = max(p.safe_max_accel_mm_s2 for p in result.pareto_front)
    min_defl = min(p.nozzle_deflection_um for p in result.pareto_front)
    assert max_accel > result.baseline.safe_max_accel_mm_s2 * 2.0
    assert min_defl < result.baseline.nozzle_deflection_um * 0.5

    # Standout labels must include Sweet Spot (Knee) or Fastest Safe Accel, plus Best Budget
    labels = {p.label for p in result.pareto_front}
    assert "Best Budget" in labels
    assert ("Sweet Spot (Knee)" in labels) or ("Fastest Safe Accel" in labels)

    # Verify architectural & hardware engineering invariants on every Pareto spec
    assert all(opt[3] == "fiberglass" for opt in pareto_mod.BELT_OPTIONS)
    for pt in result.pareto_front:
        assert pt.spec.architecture == spec.architecture
        assert pt.spec.kinematics.build_volume_mm == spec.kinematics.build_volume_mm
        assert pt.spec.frame.width_mm == spec.frame.width_mm
        assert pt.spec.frame.depth_mm == spec.frame.depth_mm
        assert pt.spec.frame.height_mm == spec.frame.height_mm
        assert pt.spec.gantry.belt.core == "fiberglass"
        if pt.spec.gantry.belt.pretension_n > 45.0 or pt.spec.gantry.drive_mode == "awd":
            assert pt.spec.gantry.double_shear_pulleys is True
        if (
            pt.spec.frame.enclosed_panels
            and pt.spec.frame.panel_mounting == "bolted_structural"
            and not pt.spec.is_flying_gantry
        ):
            assert pt.spec.frame.bolt_panels_to_gantry is True

    # Verify clean JSON serialization & valid embedded PrinterSpec dicts
    payload = result.to_dict(include_specs=True)
    serialized = json.dumps(payload)
    parsed = json.loads(serialized)
    assert parsed["backend"] == expected_backend
    assert len(parsed["pareto_front"]) == len(result.pareto_front)
    for item in parsed["pareto_front"]:
        assert "spec" in item
        validated = PrinterSpec.model_validate(item["spec"])
        assert validated.architecture == "corexy"

    compact = result.to_dict(include_specs=False)
    assert "spec" not in compact["baseline"]
    assert all("spec" not in item for item in compact["pareto_front"])


def test_flying_gantry_and_non_corexy_architectures():
    # Flying gantry CoreXY must NOT enable bolt_panels_to_gantry
    flying = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    assert flying.is_flying_gantry is True
    res_flying = optimize_pareto_front(flying, population_size=10, generations=3, seed=7)
    _assert_pareto_invariants(res_flying.pareto_front)
    for pt in res_flying.pareto_front:
        assert pt.spec.panels_bolted_to_gantry_effective is False
        if pt.spec.frame.panel_mounting == "bolted_structural" and pt.changes != ["Baseline (unchanged)"]:
            assert pt.spec.frame.bolt_panels_to_gantry is False

    # Cartesian bedslinger preserves architecture and keeps drive_mode == "2wd"
    cart = PrinterSpec.from_yaml(EXAMPLES_DIR / "cartesian_bedslinger.yaml")
    res_cart = optimize_pareto_front(cart, population_size=10, generations=3, seed=7)
    _assert_pareto_invariants(res_cart.pareto_front)
    assert len(res_cart.pareto_front) >= 2
    for pt in res_cart.pareto_front:
        assert pt.spec.architecture == "cartesian_bedslinger"
        assert pt.spec.gantry.drive_mode == "2wd"


def test_numpy_fallback_path(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pareto_mod, "is_pymoo_available", lambda: False)
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")

    result = optimize_pareto_front(spec, population_size=10, generations=4, seed=42)
    assert result.backend == "numpy (NSGA-II fallback)"
    assert len(result.pareto_front) >= 3
    _assert_pareto_invariants(result.pareto_front)


def test_deterministic_reproducibility():
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    r1 = optimize_pareto_front(spec, population_size=10, generations=3, seed=123)
    r2 = optimize_pareto_front(spec, population_size=10, generations=3, seed=123)

    assert len(r1.pareto_front) == len(r2.pareto_front)
    for p1, p2 in zip(r1.pareto_front, r2.pareto_front, strict=True):
        assert p1.label == p2.label
        assert p1.total_cost_usd == pytest.approx(p2.total_cost_usd)
        assert p1.safe_max_accel_mm_s2 == pytest.approx(p2.safe_max_accel_mm_s2)
        assert p1.nozzle_deflection_um == pytest.approx(p2.nozzle_deflection_um)
