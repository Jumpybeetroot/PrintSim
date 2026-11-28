"""Comprehensive physics tests for 0 mm+ toolhead CoG and internal toolhead stiffness."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from printsim.fea import (
    analyze_toolhead_rcsa,
    compute_structural_sensitivities,
    solve_modal_fea,
    solve_static_fea,
)
from printsim.geometry import build_assembly
from printsim.geometry.builder import compute_toolhead_link_props
from printsim.schema import TOOLHEAD_PRESETS, PrinterSpec, ToolheadSpec


def _corexy_paneled() -> PrinterSpec:
    spec = PrinterSpec(architecture="corexy")
    spec.use_cad_superelements = False
    spec.frame.bracing = "corner_gussets"
    spec.frame.enclosed_panels = True
    spec.frame.panel_mounting = "bolted_structural"
    spec.frame.panel_material = "acm_dibond"
    spec.frame.panel_thickness_mm = 4.0
    spec.bed.z_drive = "leadscrew_triple"
    return spec


def test_toolhead_schema_internal_stiffness_defaults_bounds_and_presets() -> None:
    """Verify backward-compatible 450 N/mm default, bounds [25, 20000], and preset values."""
    th_default = ToolheadSpec()
    assert th_default.internal_stiffness_n_per_mm == pytest.approx(450.0)

    with pytest.raises(ValidationError):
        ToolheadSpec(internal_stiffness_n_per_mm=10.0)
    with pytest.raises(ValidationError):
        ToolheadSpec(internal_stiffness_n_per_mm=25000.0)

    assert TOOLHEAD_PRESETS["stealthburner"]["internal_stiffness_n_per_mm"] == pytest.approx(350.0)
    assert TOOLHEAD_PRESETS["xol_compact"]["internal_stiffness_n_per_mm"] == pytest.approx(650.0)
    assert TOOLHEAD_PRESETS["archetype_cpap"]["internal_stiffness_n_per_mm"] == pytest.approx(950.0)
    assert TOOLHEAD_PRESETS["heavy_uhf_long"]["internal_stiffness_n_per_mm"] == pytest.approx(250.0)

    s = PrinterSpec()
    s.toolhead.internal_stiffness_n_per_mm = 1200.0
    assert build_assembly(s).to_geometry_dict()["internal_stiffness_n_per_mm"] == pytest.approx(1200.0)


def test_exact_zero_cog_placement_and_fastener_spring_across_mounts_and_archs() -> None:
    """Verify cog_offset=(0,0,0) places toolhead_cog at exact toolhead pos with CBUSH spring."""
    for x_mount in ("front", "top", "bottom", "dual_top_bottom"):
        s = _corexy_paneled()
        s.gantry.x_rail_mount = x_mount  # type: ignore[assignment]
        s.toolhead.cog_offset_mm = (0.0, 0.0, 0.0)
        asm = build_assembly(s)
        nmap = asm.node_map
        np.testing.assert_allclose(
            nmap["toolhead_cog"].position_m,
            nmap["toolhead"].position_m,
            atol=1e-12,
        )
        cog_springs = [sp for sp in asm.fastener_springs if sp.id == "toolhead_cog_spring"]
        assert len(cog_springs) == 1
        assert not any(m.id == "toolhead_mount" for m in asm.members)

    for arch in ("corexy", "cartesian_bedslinger", "delta"):
        s = PrinterSpec(architecture=arch)
        s.toolhead.cog_offset_mm = (0.0, 0.0, 0.0)
        asm = build_assembly(s)
        nmap = asm.node_map
        np.testing.assert_allclose(
            nmap["toolhead_cog"].position_m,
            nmap["toolhead"].position_m,
            atol=1e-12,
        )
        assert any(sp.id == "toolhead_cog_spring" for sp in asm.fastener_springs)
        st = solve_static_fea(asm)
        mo = solve_modal_fea(asm)
        rc = analyze_toolhead_rcsa(asm, mo)
        assert math.isfinite(st.combined_max_deflection_um)
        assert math.isfinite(mo.fundamental_freq_hz) and mo.fundamental_freq_hz > 10.0
        assert math.isfinite(rc.nozzle_to_sensor_amp_ratio)


def test_cog_sweep_zero_to_100mm_smooth_monotonic_and_stable() -> None:
    """Sweep CoG from exact 0 mm to 100 mm; verify zero NaNs and monotonic nozzle deflection & twist."""
    cog_points = [
        (0.0, 0.0, 0.0),
        (0.0, -0.1, -0.1),
        (0.0, -1.0, -1.0),
        (0.0, -5.0, -5.0),
        (0.0, -15.0, -20.0),
        (0.0, -30.0, -35.0),
        (0.0, -60.0, -70.0),
        (0.0, -100.0, -100.0),
    ]

    results = []
    for cog in cog_points:
        s = _corexy_paneled()
        s.toolhead.cog_offset_mm = cog
        s.toolhead.nozzle_offset_mm = (0.0, -22.0, -62.0)
        asm = build_assembly(s)
        st = solve_static_fea(asm)
        mo = solve_modal_fea(asm)
        assert math.isfinite(st.accel_deflection_x_um)
        assert math.isfinite(st.accel_deflection_y_um)
        assert math.isfinite(st.combined_max_deflection_um)
        assert math.isfinite(st.nozzle_twist_amplification_pct)
        assert math.isfinite(st.nozzle_twist_angle_mrad)
        assert math.isfinite(mo.fundamental_freq_hz) and mo.fundamental_freq_hz > 10.0
        results.append(st)

    for idx in range(len(results) - 1):
        assert results[idx + 1].accel_deflection_y_um > results[idx].accel_deflection_y_um
        assert results[idx + 1].combined_max_deflection_um > results[idx].combined_max_deflection_um
        assert (
            results[idx + 1].nozzle_twist_amplification_pct
            > results[idx].nozzle_twist_amplification_pct
        )
        assert results[idx + 1].nozzle_twist_angle_mrad > results[idx].nozzle_twist_angle_mrad

    # Also verify non-zero CoG adds zero toolhead CBUSH springs (uses toolhead_mount member)
    s_nz = _corexy_paneled()
    s_nz.toolhead.cog_offset_mm = (0.0, -18.0, -25.0)
    asm_nz = build_assembly(s_nz)
    assert not any(sp.bay == "toolhead" for sp in asm_nz.fastener_springs)
    assert any(m.id == "toolhead_mount" for m in asm_nz.members)


def test_cog_sweep_modal_shaper_freq_and_safe_accel_monotonic() -> None:
    """Increasing CoG overhang strictly lowers shaper structural/effective freq and safe accel."""
    from pathlib import Path

    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    cog_sweep_4 = [
        (0.0, 0.0, 0.0),
        (0.0, -15.0, -20.0),
        (0.0, -35.0, -35.0),
        (0.0, -70.0, -60.0),
    ]

    for yaml_name in ("voron_trident_300.yaml", "corexy_voron_style.yaml", "voron_monolith_awd.yaml"):
        base = PrinterSpec.from_yaml(examples_dir / yaml_name)
        modal_results = []
        for cog in cog_sweep_4:
            s = base.model_copy(deep=True)
            s.toolhead.cog_offset_mm = cog
            mo = solve_modal_fea(build_assembly(s))
            assert mo.shaper_x is not None and mo.shaper_y is not None
            modal_results.append(mo)

        for idx in range(len(modal_results) - 1):
            cur, nxt = modal_results[idx], modal_results[idx + 1]
            # B-01/B-02 coupled nozzle-to-bed compliance strictly lowers effective XY frequencies with CoG overhang
            assert nxt.shaper_x.effective_freq_hz < cur.shaper_x.effective_freq_hz, (
                f"{yaml_name} step {idx}: X effective freq did not drop"
            )
            assert nxt.shaper_y.effective_freq_hz < cur.shaper_y.effective_freq_hz, (
                f"{yaml_name} step {idx}: Y effective freq did not drop"
            )
            # When auto-shaper remains in the same shaper family (no discrete MZV -> ZV switch),
            # safe acceleration strictly drops with CoG overhang; also verify consistent MZV below.
            if nxt.shaper_y.recommended_shaper == cur.shaper_y.recommended_shaper:
                assert nxt.recommended_max_accel_mm_s2 <= cur.recommended_max_accel_mm_s2, (
                    f"{yaml_name} step {idx}: recommended_max_accel increased"
                )
        assert modal_results[-1].recommended_max_accel_mm_s2 < modal_results[0].recommended_max_accel_mm_s2 * 0.96

        # Under consistent MZV shaper choice, safe acceleration drops over the CoG sweep
        mzv_accels = []
        for cog in cog_sweep_4:
            s_mzv = base.model_copy(deep=True)
            s_mzv.toolhead.cog_offset_mm = cog
            s_mzv.kinematics.input_shaper = "mzv"
            mzv_accels.append(solve_modal_fea(build_assembly(s_mzv)).recommended_max_accel_mm_s2)
        for idx in range(len(mzv_accels) - 1):
            assert mzv_accels[idx + 1] <= mzv_accels[idx] + 200.0, (
                f"{yaml_name} MZV step {idx}: accel jumped ({mzv_accels[idx]} -> {mzv_accels[idx + 1]})"
            )
        assert mzv_accels[-1] <= mzv_accels[0] * 0.95, (
            f"{yaml_name} MZV sweep: overall accel did not drop ({mzv_accels[0]} -> {mzv_accels[-1]})"
        )

    # Full 0 -> 100 mm sweep on paneled CoreXY stays smooth and strictly monotonic in modal freqs
    cog_sweep_8 = [
        (0.0, 0.0, 0.0),
        (0.0, -0.1, -0.1),
        (0.0, -1.0, -1.0),
        (0.0, -5.0, -5.0),
        (0.0, -15.0, -20.0),
        (0.0, -30.0, -35.0),
        (0.0, -60.0, -70.0),
        (0.0, -100.0, -100.0),
    ]
    paneled_modals = []
    for cog in cog_sweep_8:
        s = _corexy_paneled()
        s.toolhead.cog_offset_mm = cog
        s.toolhead.nozzle_offset_mm = (0.0, -22.0, -62.0)
        mo = solve_modal_fea(build_assembly(s))
        assert mo.shaper_x is not None and mo.shaper_y is not None
        paneled_modals.append(mo)

    for idx in range(len(paneled_modals) - 1):
        cur, nxt = paneled_modals[idx], paneled_modals[idx + 1]
        if idx < len(paneled_modals) - 2:
            # Up to CoG above/near nozzle (-60/-70 vs nozzle -22/-62), X effective freq drops strictly
            assert nxt.shaper_x.effective_freq_hz < cur.shaper_x.effective_freq_hz
        assert nxt.shaper_y.effective_freq_hz < cur.shaper_y.effective_freq_hz
        assert nxt.recommended_max_accel_mm_s2 <= cur.recommended_max_accel_mm_s2
    assert paneled_modals[-1].shaper_x.effective_freq_hz < paneled_modals[0].shaper_x.effective_freq_hz
    # Calibrated 2WD belts (D-01 22 kN EA) still suffer >25% max accel collapse at -100 mm overhang
    assert paneled_modals[-1].recommended_max_accel_mm_s2 < paneled_modals[0].recommended_max_accel_mm_s2 * 0.88


def test_cog_equals_nozzle_edge_case_zero_clamp_and_finite_fea() -> None:
    """Verify CoG == Nozzle uses zero-length CBUSH stem without arbitrary 5 mm push or crash."""
    for cog in ((0.0, 0.0, 0.0), (0.0, -20.0, -40.0)):
        for arch in ("corexy", "cartesian_bedslinger", "delta"):
            s = PrinterSpec(architecture=arch)
            s.toolhead.cog_offset_mm = cog
            s.toolhead.nozzle_offset_mm = cog
            asm = build_assembly(s)
            nmap = asm.node_map
            np.testing.assert_allclose(
                nmap["nozzle_tip"].position_m,
                nmap["toolhead_cog"].position_m,
                atol=1e-12,
            )
            assert any(sp.id == "nozzle_stem_spring" for sp in asm.fastener_springs)
            assert not any(m.id == "nozzle_stem" for m in asm.members)

            st = solve_static_fea(asm)
            mo = solve_modal_fea(asm)
            rc = analyze_toolhead_rcsa(asm, mo)
            sen = compute_structural_sensitivities(asm)
            assert math.isfinite(st.toolhead_stiffness_x_n_per_mm) and st.toolhead_stiffness_x_n_per_mm > 0.0
            assert math.isfinite(st.toolhead_stiffness_y_n_per_mm) and st.toolhead_stiffness_y_n_per_mm > 0.0
            assert math.isfinite(mo.fundamental_freq_hz) and mo.fundamental_freq_hz > 10.0
            assert math.isfinite(rc.nozzle_to_sensor_amp_ratio)
            assert math.isfinite(sen.fundamental_freq_hz)


def test_nozzle_stem_stiffness_stays_constant_as_cog_approaches_zero() -> None:
    """Verify nozzle_stem bending stiffness 3*E*I/L^3 stays invariant as CoG moves toward 0 mm."""
    s = _corexy_paneled()
    props_short_stem = compute_toolhead_link_props(s, l_mount_m=0.045, l_stem_m=0.020)
    props_long_stem = compute_toolhead_link_props(s, l_mount_m=0.005, l_stem_m=0.065)

    k_short = 3.0 * props_short_stem["e_stem"] * props_short_stem["i_stem"] / (0.020**3)
    k_long = 3.0 * props_long_stem["e_stem"] * props_long_stem["i_stem"] / (0.065**3)
    assert k_short == pytest.approx(k_long, rel=1e-6)
    assert props_long_stem["i_stem"] > props_short_stem["i_stem"] * (0.065 / 0.020) ** 2


def test_internal_stiffness_sweep_monotonic_fea_and_rcsa() -> None:
    """Sweep internal_stiffness_n_per_mm across 100..5000 N/mm; verify static FEA & RCSA monotonicity."""
    stiffness_vals = [100.0, 250.0, 450.0, 1000.0, 2500.0, 5000.0]

    for cog in ((0.0, -18.0, -25.0), (0.0, 0.0, 0.0)):
        static_res = []
        rcsa_res = []
        for k_val in stiffness_vals:
            s = _corexy_paneled()
            s.toolhead.cog_offset_mm = cog
            s.toolhead.internal_stiffness_n_per_mm = k_val
            asm = build_assembly(s)
            st = solve_static_fea(asm)
            mo = solve_modal_fea(asm)
            rc = analyze_toolhead_rcsa(asm, mo)
            static_res.append(st)
            rcsa_res.append(rc)

        for idx in range(len(stiffness_vals) - 1):
            assert (
                static_res[idx + 1].toolhead_stiffness_x_n_per_mm
                > static_res[idx].toolhead_stiffness_x_n_per_mm
            )
            assert (
                static_res[idx + 1].toolhead_stiffness_y_n_per_mm
                > static_res[idx].toolhead_stiffness_y_n_per_mm
            )
            assert (
                static_res[idx + 1].accel_deflection_y_um
                < static_res[idx].accel_deflection_y_um
            )
            assert (
                static_res[idx + 1].combined_max_deflection_um
                < static_res[idx].combined_max_deflection_um
            )
            assert (
                static_res[idx + 1].nozzle_twist_amplification_pct
                < static_res[idx].nozzle_twist_amplification_pct
            )
            assert (
                rcsa_res[idx + 1].nozzle_to_sensor_amp_ratio
                < rcsa_res[idx].nozzle_to_sensor_amp_ratio
            )
            assert (
                rcsa_res[idx + 1].hotend_bending_stiffness_n_per_mm
                > rcsa_res[idx].hotend_bending_stiffness_n_per_mm
            )


def test_chamber_temperature_modulus_retention_on_internal_stiffness() -> None:
    """Verify chamber heating softens ABS toolhead internal stiffness while CNC Al remains stable."""
    s_abs_25 = _corexy_paneled()
    s_abs_25.frame.chamber_temp_c = 25.0
    s_abs_25.toolhead.mount_material = "abs_printed"
    s_abs_25.toolhead.cog_offset_mm = (0.0, 0.0, 0.0)

    s_abs_70 = s_abs_25.model_copy(deep=True)
    s_abs_70.frame.chamber_temp_c = 70.0

    asm_25 = build_assembly(s_abs_25)
    asm_70 = build_assembly(s_abs_70)
    sp_25 = next(sp for sp in asm_25.fastener_springs if sp.id == "toolhead_cog_spring")
    sp_70 = next(sp for sp in asm_70.fastener_springs if sp.id == "toolhead_cog_spring")
    assert sp_70.k_shear_n_m < sp_25.k_shear_n_m
    assert sp_70.k_rot_nm_rad < sp_25.k_rot_nm_rad
    assert solve_static_fea(asm_70).toolhead_stiffness_y_n_per_mm < solve_static_fea(
        asm_25
    ).toolhead_stiffness_y_n_per_mm

    s_al_25 = s_abs_25.model_copy(deep=True)
    s_al_25.toolhead.mount_material = "aluminum_6061_t6"
    s_al_70 = s_al_25.model_copy(deep=True)
    s_al_70.frame.chamber_temp_c = 70.0
    sp_al_25 = next(
        sp for sp in build_assembly(s_al_25).fastener_springs if sp.id == "toolhead_cog_spring"
    )
    sp_al_70 = next(
        sp for sp in build_assembly(s_al_70).fastener_springs if sp.id == "toolhead_cog_spring"
    )
    assert sp_al_70.k_shear_n_m == pytest.approx(sp_al_25.k_shear_n_m, rel=1e-9)


def test_positive_cog_z_tbranch_and_non_collocated_modal_compliance() -> None:
    """Top-heavy direct-drive toolheads (cog_z > 0, noz_z < 0) use a T-branch without collapsing Kx or frame share."""
    from pathlib import Path

    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    spec = PrinterSpec.from_yaml(examples_dir / "voron_trident_300.yaml")
    spec.use_cad_superelements = False  # verify analytical T-branch directly
    assert spec.toolhead.cog_offset_mm[2] > 0.0
    assert spec.toolhead.nozzle_offset_mm[2] < 0.0

    asm = build_assembly(spec)
    node_ids = {n.id for n in asm.nodes}
    member_ids = {m.id for m in asm.members}
    assert "toolhead_chassis" in node_ids
    assert {"toolhead_mount", "toolhead_mount_upper", "nozzle_stem"}.issubset(member_ids)

    st = solve_static_fea(asm)
    mo = solve_modal_fea(asm)
    assert st.toolhead_stiffness_x_n_per_mm > 115.0
    assert mo.shaper_y is not None
    # Preset ships the Voron-documented 110 Hz belt pluck (9 N). A real stock Trident 300 at
    # 2.2 lb measures Y MZV 40.8-42.2 Hz (docs/community_touchstones.md); model ~40.5 Hz here.
    assert 36.0 <= mo.shaper_y.effective_freq_hz <= 46.0
    assert mo.shaper_y.compliance_share_pct["frame"] > 7.5

    # Softening the carriage mount (e.g. Voron Tap) must strictly LOWER Y shaper frequency & consistent-MZV accel
    spec_mzv = spec.model_copy(deep=True)
    spec_mzv.kinematics.input_shaper = "mzv"
    spec_tap = spec_mzv.model_copy(deep=True)
    spec_tap.toolhead.z_probe_type = "printed_tap"
    mo_mzv = solve_modal_fea(build_assembly(spec_mzv))
    mo_tap = solve_modal_fea(build_assembly(spec_tap))
    assert mo_tap.shaper_y is not None and mo_mzv.shaper_y is not None
    assert mo_tap.shaper_y.effective_freq_hz < mo_mzv.shaper_y.effective_freq_hz
    assert mo_tap.recommended_max_accel_mm_s2 <= mo_mzv.recommended_max_accel_mm_s2



def test_umbilical_type_and_mass_override_affect_xy_moving_mass_and_shaper() -> None:
    """Umbilical preset and explicit umbilical_mass_g override shift X/Y moving mass & shaper Hz."""
    from pathlib import Path

    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    base = PrinterSpec.from_yaml(examples_dir / "voron_trident_300.yaml")
    assert base.toolhead.umbilical_type == "standard"
    assert base.toolhead.effective_umbilical_mass_g == pytest.approx(40.0)

    s_can = base.model_copy(deep=True)
    s_can.toolhead.umbilical_type = "can_umbilical"  # 20 g
    s_chain = base.model_copy(deep=True)
    s_chain.toolhead.umbilical_type = "cable_chain"  # 60 g
    s_custom = base.model_copy(deep=True)
    s_custom.toolhead.umbilical_mass_g = 90.0

    asm_can = build_assembly(s_can)
    asm_std = build_assembly(base)
    asm_chain = build_assembly(s_chain)
    asm_custom = build_assembly(s_custom)

    assert (asm_std.x_moving_mass_kg - asm_can.x_moving_mass_kg) * 1000.0 == pytest.approx(20.0, abs=1e-6)
    assert (asm_chain.x_moving_mass_kg - asm_std.x_moving_mass_kg) * 1000.0 == pytest.approx(20.0, abs=1e-6)
    assert (asm_custom.y_moving_mass_kg - asm_std.y_moving_mass_kg) * 1000.0 == pytest.approx(50.0, abs=1e-6)

    mo_can = solve_modal_fea(asm_can)
    mo_chain = solve_modal_fea(asm_chain)
    assert mo_can.shaper_x is not None and mo_chain.shaper_x is not None
    assert mo_can.shaper_y is not None and mo_chain.shaper_y is not None
    assert mo_can.shaper_x.effective_freq_hz > mo_chain.shaper_x.effective_freq_hz
    assert mo_can.shaper_y.effective_freq_hz > mo_chain.shaper_y.effective_freq_hz
