"""Shared ``GlobalSystem`` assembly must be a pure performance optimization.

``evaluate_printer`` assembles K/M and the per-element blocks once and hands them to the static,
modal, and sensitivity solvers. These tests pin that the shared path is numerically identical to
each solver assembling on its own, that shared data is immutable, and that the hoisted
micro-kernels match their reference formulas bit-for-bit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from printsim.fea import (
    assemble_global_matrices,
    assemble_global_system,
    solve_modal_fea,
    solve_static_fea,
)
from printsim.fea.klipper_shaper import (
    get_shaper_pulses,
    get_shaper_smoothing,
    prepare_shaper_smoothing,
)
from printsim.fea.sensitivity import compute_structural_sensitivities
from printsim.fea.shell_mitc4 import cross3
from printsim.geometry import build_assembly
from printsim.schema import PrinterSpec

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

# One paneled CoreXY (sparse eigen path, shells + fastener springs + superelements),
# one bedslinger and one delta (small dense paths).
_PRESETS = ("voron_trident_300", "cartesian_bedslinger", "delta_racer")


@pytest.mark.parametrize("preset", _PRESETS)
def test_shared_system_matches_independent_solves_exactly(preset: str) -> None:
    spec = PrinterSpec.from_yaml(EXAMPLES / f"{preset}.yaml")
    asm = build_assembly(spec)

    static_ref = solve_static_fea(asm)
    modal_ref = solve_modal_fea(asm)
    sens_ref = compute_structural_sensitivities(asm, static_ref, modal_ref)

    system = assemble_global_system(asm)
    static_sh = solve_static_fea(asm, system=system)
    modal_sh = solve_modal_fea(asm, system=system)
    sens_sh = compute_structural_sensitivities(asm, static_sh, modal_sh, system=system)

    assert static_sh.to_dict() == static_ref.to_dict()
    assert modal_sh.to_dict() == modal_ref.to_dict()
    assert sens_sh.to_dict() == sens_ref.to_dict()


def test_global_system_matches_legacy_tuple_and_is_read_only() -> None:
    spec = PrinterSpec.from_yaml(EXAMPLES / "voron_trident_300.yaml")
    asm = build_assembly(spec)

    K, M, F_grav, node_idx, free_dofs = assemble_global_matrices(asm)
    system = assemble_global_system(asm)

    assert np.array_equal(system.K, K)
    assert np.array_equal(system.M, M)
    assert np.array_equal(system.F_grav, F_grav)
    assert system.node_idx == node_idx
    assert system.free_dofs == free_dofs
    assert len(system.member_blocks) == len(asm.members)
    assert len(system.shell_blocks) == len(asm.shell_elements)
    assert len(system.spring_blocks) == len(asm.fastener_springs)

    # Legacy API still hands out private, writable arrays
    assert K.flags.writeable and M.flags.writeable

    # Shared arrays and element blocks reject in-place edits so one solver cannot corrupt another
    with pytest.raises(ValueError):
        system.K[0, 0] = 1.0
    with pytest.raises(ValueError):
        system.member_blocks[0][1][0, 0] = 1.0
    with pytest.raises(ValueError):
        system.reduced_stiffness()[0, 0] = 1.0

    # Element blocks re-assemble to the global stiffness (scatter-add consistency)
    k_check = np.zeros_like(K)
    for dofs, ke, *_ in (
        *system.member_blocks,
        *system.shell_blocks,
        *system.spring_blocks,
        *system.superelement_blocks,
    ):
        k_check[np.ix_(dofs, dofs)] += ke
    assert np.allclose(k_check, K, rtol=1e-12, atol=1e-9)

    # Derived quantities are memoized, not recomputed
    assert system.reduced_stiffness() is system.reduced_stiffness()
    assert system.reduced_mass() is system.reduced_mass()


def test_cross3_is_bit_identical_to_numpy_cross() -> None:
    rng = np.random.default_rng(7)
    for _ in range(200):
        a = rng.standard_normal(3) * 10.0 ** rng.integers(-6, 6)
        b = rng.standard_normal(3) * 10.0 ** rng.integers(-6, 6)
        assert np.array_equal(cross3(a, b), np.cross(a, b))
    ez = np.array([0.0, 0.0, 1.0])
    ex = np.array([1.0, 0.0, 0.0])
    assert np.array_equal(cross3(ez, ex), np.array([0.0, 1.0, 0.0]))


@pytest.mark.parametrize("name", ["zv", "mzv", "ei", "2hump_ei", "3hump_ei", "zvdd"])
def test_prepared_shaper_smoothing_matches_reference_expression(name: str) -> None:
    shaper = get_shaper_pulses(name, 48.5, damping_ratio=0.1)
    amps = np.asarray(shaper[0], dtype=float)
    times = np.asarray(shaper[1], dtype=float)
    prepared = prepare_shaper_smoothing(shaper)

    for accel in (100.0, 3517.25, 5000.0, 20000.0, 1.0e6):
        for scv in (5.0, 8.0):
            # Klipper reference expression evaluated in full each call
            inv_d = 1.0 / float(np.sum(amps))
            ts = float(np.sum(amps * times)) * inv_d
            mask = times >= ts
            dt_pos = times[mask] - ts
            offset_90 = (
                np.sqrt(2.0) * inv_d * float(np.sum(amps[mask] * (scv + 0.5 * accel * dt_pos) * dt_pos))
            )
            offset_180 = 0.5 * accel * inv_d * float(np.sum(amps * ((times - ts) ** 2)))
            expected = max(float(offset_90), offset_180)
            assert prepared(accel, scv) == pytest.approx(expected, rel=1e-15, abs=0.0)
            assert get_shaper_smoothing(shaper, accel=accel, scv=scv) == prepared(accel, scv)
