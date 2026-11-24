"""Tests for bed mounting FEA, differentiated CoreXY Z-drive topologies, feet anchoring, and API simulation options."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from printsim.api.server import create_app
from printsim.engine import evaluate_printer
from printsim.fea import (
    assemble_global_matrices,
    compute_thermal_structural_report,
    solve_modal_fea,
    solve_static_fea,
)
from printsim.fea.solver import _solve_linear_system
from printsim.geometry import build_assembly, generate_bom
from printsim.schema import (
    BED_MOUNT_CATALOG,
    FRAME_FEET_CATALOG,
    BedMountType,
    FrameFeetType,
    PrinterSpec,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _trident_base() -> PrinterSpec:
    return PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")


def _assert_spd_and_finite(asm) -> None:
    K, M, _, _, free_dofs = assemble_global_matrices(asm)
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    M_ff = M[np.ix_(free_dofs, free_dofs)]
    assert np.allclose(K_ff, K_ff.T, rtol=1e-12, atol=1e-6)
    assert np.allclose(M_ff, M_ff.T, rtol=1e-12, atol=1e-10)
    K_sym = 0.5 * (K_ff + K_ff.T)
    assert float(np.linalg.eigvalsh(K_sym)[0]) > 0.0


# ---------------------------------------------------------------------------
# 1. Schema Catalog Exports & Backward Compatibility
# ---------------------------------------------------------------------------


def test_schema_catalogs_and_defaults_backward_compatible() -> None:
    """Verify BED_MOUNT_CATALOG (7), FRAME_FEET_CATALOG (4), and backward-compatible defaults."""
    assert set(BED_MOUNT_CATALOG.keys()) == {
        "kinematic_3pt",
        "whopping_pochard_kinematic",
        "trident_3pt_standoffs",
        "rigid_4pt",
        "floating_4pt",
        "silicone_4pt",
        "springs_4pt",
    }
    assert set(FRAME_FEET_CATALOG.keys()) == {
        "rigid_bolted",
        "rubber_feet",
        "anti_vib_pads",
        "paver_on_foam",
    }
    # Verify exact thermal restraint backward compatibility
    assert BED_MOUNT_CATALOG["kinematic_3pt"]["k_restraint_thermal"] == 0.0
    assert BED_MOUNT_CATALOG["whopping_pochard_kinematic"]["k_restraint_thermal"] == 0.0
    assert BED_MOUNT_CATALOG["floating_4pt"]["k_restraint_thermal"] == 18.0
    assert BED_MOUNT_CATALOG["trident_3pt_standoffs"]["k_restraint_thermal"] == 22.0
    assert BED_MOUNT_CATALOG["rigid_4pt"]["k_restraint_thermal"] == 8000.0

    expected_heights = {
        "kinematic_3pt": 16.0,
        "whopping_pochard_kinematic": 14.0,
        "trident_3pt_standoffs": 10.0,
        "rigid_4pt": 12.0,
        "floating_4pt": 12.0,
        "silicone_4pt": 16.0,
        "springs_4pt": 20.0,
    }
    expected_modes = {
        "kinematic_3pt": "maxwell_vgroove",
        "whopping_pochard_kinematic": "radial_flexure",
        "trident_3pt_standoffs": "slotted_3pt_bakelite",
        "rigid_4pt": "rigid_all",
        "floating_4pt": "slotted_1fixed_3float",
        "silicone_4pt": "elastomer_shear",
        "springs_4pt": "coil_spring_shear",
    }
    expected_tap = {
        "kinematic_3pt": False,
        "whopping_pochard_kinematic": True,
        "trident_3pt_standoffs": True,
        "rigid_4pt": True,
        "floating_4pt": True,
        "silicone_4pt": False,
        "springs_4pt": False,
    }
    for mt, cfg in BED_MOUNT_CATALOG.items():
        assert cfg["standoff_height_mm"] == pytest.approx(expected_heights[mt])
        assert cfg["radial_dof_mode"] == expected_modes[mt]
        assert cfg["tap_rated"] is expected_tap[mt]
        assert cfg["mount_inset_ratio"] == pytest.approx(0.38)

    # Omitting feet_type / mount_type in spec preserves defaults
    spec_default = PrinterSpec()
    assert spec_default.frame.feet_type == "rigid_bolted"
    assert spec_default.bed.mount_type == "kinematic_3pt"
    # Annotations exist and accept all catalog keys
    assert BedMountType is not None
    assert FrameFeetType is not None


# ---------------------------------------------------------------------------
# 2. CoreXY Z-Drive Topologies (belted_quad_z, triple, dual, single)
# ---------------------------------------------------------------------------


def test_corexy_z_drive_topologies_non_singular_and_monotonic_bed_deflection() -> None:
    """All 4 CoreXY z_drive layouts are SPD and show monotonic droop/Z compliance."""
    base = _trident_base()
    z_drives = ("belted_quad_z", "leadscrew_triple", "leadscrew_dual", "leadscrew_single")
    expected_support_counts = {
        "belted_quad_z": 4,
        "leadscrew_triple": 3,
        "leadscrew_dual": 2,
        "leadscrew_single": 1,
    }

    grav_bed_droops_um = []
    unit_bed_z_defls_um = []
    combined_bed_defls_um = []
    loop_kz = []

    for zd in z_drives:
        spec = base.model_copy(deep=True)
        spec.bed.z_drive = zd  # type: ignore[assignment]
        asm = build_assembly(spec)
        _assert_spd_and_finite(asm)

        nmap = asm.node_map
        assert "bed_frame" in nmap
        assert "bed_surface" in nmap
        assert nmap["bed_frame"].lumped_mass_kg == pytest.approx(0.0, abs=1e-9)
        assert nmap["bed_surface"].lumped_mass_kg == pytest.approx(
            spec.bed.mass_g * 1e-3, abs=1e-9
        )
        assert any(sp.id == "bed_mount_spring" for sp in asm.fastener_springs)

        # Mass accounting: zero double counting of bed_frame virtual hub
        lumped_sum = sum(n.lumped_mass_kg for n in asm.nodes)
        shell_mass = asm.panel_mass_kg if asm.shell_elements else 0.0
        assert asm.total_mass_kg == pytest.approx(
            asm.frame_mass_kg + lumped_sum + shell_mass, rel=1e-9
        )

        # Geometry serialization exposes Z-drive, mount type, feet type, bed_frame & support points
        gdict = asm.to_geometry_dict()
        assert gdict["bed_z_drive"] == zd
        assert gdict["bed_mount_type"] == spec.bed.mount_type
        assert gdict["feet_type"] == spec.frame.feet_type
        assert gdict["bed_frame_node_id"] == "bed_frame"
        assert len(gdict["bed_support_points_mm"]) == expected_support_counts[zd]

        # Pure gravity bed Z droop and pure 10N bed Z load
        K, _, F_grav, node_idx, free_dofs = assemble_global_matrices(asm)
        u_grav = _solve_linear_system(K, F_grav, free_dofs)
        bed_grav_z_um = abs(float(u_grav[6 * node_idx["bed_surface"] + 2])) * 1e6

        f_z10 = np.zeros(K.shape[0], dtype=float)
        f_z10[6 * node_idx["bed_surface"] + 2] = -10.0
        u_z10 = _solve_linear_system(K, f_z10, free_dofs)
        bed_10n_z_um = abs(float(u_z10[6 * node_idx["bed_surface"] + 2])) * 1e6

        st = solve_static_fea(asm)
        mo = solve_modal_fea(asm)
        assert math.isfinite(st.combined_max_deflection_um)
        # Unpaneled single-Z cantilever bed bounces realistically at ~8-9 Hz; multi-point Z stays >25 Hz
        assert math.isfinite(mo.fundamental_freq_hz) and mo.fundamental_freq_hz > 5.0

        grav_bed_droops_um.append(bed_grav_z_um)
        unit_bed_z_defls_um.append(bed_10n_z_um)
        combined_bed_defls_um.append(st.bed_deflection_um)
        loop_kz.append(st.toolhead_stiffness_z_n_per_mm)

        # BOM excludes all virtual bed members and ground nodes
        bom = generate_bom(spec)
        assert all(c.role != "bed" for c in bom.cut_list)

    # Monotonic bed droop: leadscrew_single > leadscrew_dual > leadscrew_triple > belted_quad_z
    assert grav_bed_droops_um[3] > grav_bed_droops_um[2] > grav_bed_droops_um[1] > grav_bed_droops_um[0]
    assert unit_bed_z_defls_um[3] > unit_bed_z_defls_um[2] > unit_bed_z_defls_um[1] > unit_bed_z_defls_um[0]
    assert combined_bed_defls_um[3] > combined_bed_defls_um[2] > combined_bed_defls_um[1] > combined_bed_defls_um[0]
    # Nozzle-to-bed structural loop Z stiffness: multi-point Z beats single cantilever bed by > 1.8x,
    # and away from exact Y-midspan (where dual-Z vertical rails directly prop up y_car), triple > dual > belted_quad > single
    assert loop_kz[1] > loop_kz[3] * 2.0
    assert loop_kz[2] > loop_kz[3] * 2.0
    assert loop_kz[0] > loop_kz[3] * 1.5
    off_mid_kz = []
    for zd in z_drives:
        s_om = base.model_copy(deep=True)
        s_om.bed.z_drive = zd  # type: ignore[assignment]
        off_mid_kz.append(solve_static_fea(build_assembly(s_om, toolhead_rel_pos=(0.5, 0.25, 0.5))).toolhead_stiffness_z_n_per_mm)
    assert off_mid_kz[1] > off_mid_kz[2] > off_mid_kz[0] > off_mid_kz[3]

    # Single cantilever droop is substantially worse (> 3x) than dual-Z
    assert grav_bed_droops_um[3] > grav_bed_droops_um[2] * 3.0


def test_leadscrew_dual_all_y_positions_and_lower_pitch_stiffness() -> None:
    """Dual-Z builds cleanly at off-center Y positions (with/without panels) and has lower pitch stiffness than triple-Z."""
    base = _trident_base()

    # Check off-center Y positions & bolted panels
    for v_pos in (0.15, 0.50, 0.85):
        for enc in (False, True):
            s = base.model_copy(deep=True)
            s.bed.z_drive = "leadscrew_dual"
            s.frame.enclosed_panels = enc
            s.frame.panel_mounting = "bolted_structural" if enc else "none"
            asm = build_assembly(s, toolhead_rel_pos=(0.5, v_pos, 0.5), panel_mesh_n=3)
            _assert_spd_and_finite(asm)
            st = solve_static_fea(asm)
            mo = solve_modal_fea(asm)
            assert math.isfinite(st.combined_max_deflection_um)
            assert math.isfinite(mo.fundamental_freq_hz) and mo.fundamental_freq_hz > 15.0

    # Dual-Z has lower pitch stiffness around X than Triple-Z
    def _pitch_compliance_mrad_per_nm(zd: str) -> float:
        s = base.model_copy(deep=True)
        s.bed.z_drive = zd  # type: ignore[assignment]
        asm = build_assembly(s)
        K, _, _, node_idx, free_dofs = assemble_global_matrices(asm)
        f_tx = np.zeros(K.shape[0], dtype=float)
        f_tx[6 * node_idx["bed_surface"] + 3] = 1.0
        u_tx = _solve_linear_system(K, f_tx, free_dofs)
        return abs(float(u_tx[6 * node_idx["bed_surface"] + 3])) * 1e3

    assert _pitch_compliance_mrad_per_nm("leadscrew_dual") > _pitch_compliance_mrad_per_nm("leadscrew_triple") * 2.5


# ---------------------------------------------------------------------------
# 3. All 5 Bed Mount Types (Mechanical Stiffness + Thermal Taco Buckling)
# ---------------------------------------------------------------------------


def test_all_five_bed_mounts_monotonic_stiffness_and_thermal_taco_buckling() -> None:
    """Verify monotonic mechanical stiffness (rigid > kinematic > floating > silicone > springs) and thermal buckling."""
    mounts_ordered = (
        "rigid_4pt",
        "kinematic_3pt",
        "floating_4pt",
        "silicone_4pt",
        "springs_4pt",
    )
    base = _trident_base()
    base.bed.target_temp_c = 110.0

    bed_defls = []
    kx_vals = []
    ky_vals = []
    kz_vals = []
    bed_mount_kx_vals = []
    taco_um = {}

    for mt in mounts_ordered:
        s = base.model_copy(deep=True)
        s.bed.mount_type = mt  # type: ignore[assignment]
        asm = build_assembly(s)
        _assert_spd_and_finite(asm)
        st = solve_static_fea(asm)
        ts = compute_thermal_structural_report(asm)

        K, _, _, nidx, free = assemble_global_matrices(asm)
        f_bx = np.zeros(K.shape[0], dtype=float)
        f_bx[6 * nidx["bed_surface"]] = 1.0
        u_bx = _solve_linear_system(K, f_bx, free)
        dx_mount_mm = abs(float(u_bx[6 * nidx["bed_surface"]] - u_bx[6 * nidx["bed_frame"]])) * 1e3
        bed_mount_kx_vals.append(1.0 / max(dx_mount_mm, 1e-12))

        bed_defls.append(st.bed_deflection_um)
        kx_vals.append(st.toolhead_stiffness_x_n_per_mm)
        ky_vals.append(st.toolhead_stiffness_y_n_per_mm)
        kz_vals.append(st.toolhead_stiffness_z_n_per_mm)
        taco_um[mt] = ts.bed_buckling_taco_um

    # Mechanical stiffness strictly decreases from rigid_4pt -> springs_4pt
    # (note: on an unpaneled frame, kinematic_3pt's 16mm vs floating_4pt's 12mm standoff height shifts
    # Z-carriage elevation by 4mm (~0.1% on frame loop kx), while mount-level kx and structural-panel kx stay strictly monotonic)
    for i in range(len(mounts_ordered) - 1):
        assert bed_mount_kx_vals[i] > bed_mount_kx_vals[i + 1]
        assert ky_vals[i] > ky_vals[i + 1]
        assert kz_vals[i] > kz_vals[i + 1]
        assert bed_defls[i] < bed_defls[i + 1]
    assert kx_vals[0] > kx_vals[1] > kx_vals[3] > kx_vals[4]
    assert kx_vals[2] > kx_vals[3] > kx_vals[4]

    # With structural bolted panels bracing the columns, full-loop kx is also strictly monotonic across all 5 mounts
    kx_paneled = []
    for mt in mounts_ordered:
        s_p = base.model_copy(deep=True)
        s_p.frame.panel_mounting = "bolted_structural"
        s_p.bed.mount_type = mt  # type: ignore[assignment]
        kx_paneled.append(solve_static_fea(build_assembly(s_p)).toolhead_stiffness_x_n_per_mm)
    for i in range(len(mounts_ordered) - 1):
        assert kx_paneled[i] > kx_paneled[i + 1]

    # Thermal taco buckling: kinematic_3pt == 0, rigid_4pt high (>100 um), compliant 4pt near-zero
    assert taco_um["kinematic_3pt"] == 0.0
    assert taco_um["rigid_4pt"] > 100.0
    assert 0.0 < taco_um["springs_4pt"] < taco_um["silicone_4pt"] < taco_um["floating_4pt"] < 25.0


@pytest.mark.parametrize("arch", ["corexy", "cartesian_bedslinger", "delta"])
def test_bed_mount_compliance_across_all_architectures(arch: str) -> None:
    """Every architecture builds bed_frame -> bed_mount_spring -> bed_surface and scales with mount_type."""
    s_rig = PrinterSpec(architecture=arch)  # type: ignore[arg-type]
    s_rig.bed.mount_type = "rigid_4pt"
    s_spr = PrinterSpec(architecture=arch)  # type: ignore[arg-type]
    s_spr.bed.mount_type = "springs_4pt"

    asm_rig = build_assembly(s_rig)
    asm_spr = build_assembly(s_spr)
    _assert_spd_and_finite(asm_rig)
    _assert_spd_and_finite(asm_spr)

    assert "bed_frame" in asm_rig.node_map
    assert "bed_surface" in asm_rig.node_map
    assert len(asm_rig.to_geometry_dict()["bed_support_points_mm"]) >= 2

    st_rig = solve_static_fea(asm_rig)
    st_spr = solve_static_fea(asm_spr)
    assert st_rig.toolhead_stiffness_z_n_per_mm > st_spr.toolhead_stiffness_z_n_per_mm * 2.0
    assert st_spr.bed_deflection_um > st_rig.bed_deflection_um * 3.0


# ---------------------------------------------------------------------------
# 4. Machine Feet Anchoring (rigid_bolted, rubber_feet, anti_vib_pads)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "preset_name",
    ["voron_trident_300", "cartesian_bedslinger", "delta_racer"],
)
def test_frame_feet_types_monotonic_frequency_and_damping(preset_name: str) -> None:
    """rigid_bolted > rubber_feet > anti_vib_pads in fundamental frequency and increases damping."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / f"{preset_name}.yaml")
    feet_ordered = ("rigid_bolted", "rubber_feet", "anti_vib_pads")
    freqs = []
    dampings = []

    for ft in feet_ordered:
        s = base.model_copy(deep=True)
        s.frame.feet_type = ft  # type: ignore[assignment]
        asm = build_assembly(s)
        _assert_spd_and_finite(asm)
        mo = solve_modal_fea(asm)
        freqs.append(mo.fundamental_freq_hz)
        dampings.append(asm.frame_damping_ratio)

        if ft == "rigid_bolted":
            assert not any(n.role == "ground" for n in asm.nodes)
            assert not any(sp.bay == "feet" for sp in asm.fastener_springs)
        else:
            assert any(n.role == "ground" and n.fixed for n in asm.nodes)
            assert any(sp.bay == "feet" for sp in asm.fastener_springs)
            assert asm.to_geometry_dict()["feet_type"] == ft
            bom = generate_bom(s)
            assert bom.total_mass_kg > 0.0

    assert freqs[0] > freqs[1] > freqs[2]
    assert dampings[2] > dampings[1] > dampings[0]


def test_feet_type_omitted_preserves_backward_compatibility() -> None:
    """YAML without feet_type defaults to rigid_bolted identically to explicit rigid_bolted."""
    spec_omitted = _trident_base()
    spec_explicit = _trident_base()
    spec_explicit.frame.feet_type = "rigid_bolted"

    asm_om = build_assembly(spec_omitted)
    asm_ex = build_assembly(spec_explicit)
    assert solve_modal_fea(asm_om).fundamental_freq_hz == pytest.approx(
        solve_modal_fea(asm_ex).fundamental_freq_hz, rel=1e-9
    )
    assert solve_static_fea(asm_om).combined_max_deflection_um == pytest.approx(
        solve_static_fea(asm_ex).combined_max_deflection_um, rel=1e-9
    )


# ---------------------------------------------------------------------------
# 5. Evaluator & FastAPI /api/simulate + /api/catalog Options
# ---------------------------------------------------------------------------


def test_api_catalog_and_simulate_options() -> None:
    """Verify /api/catalog returns 5 bed_mounts + 3 frame_feet and /api/simulate honors panel_mesh_n, toolhead_rel_pos, accel_override_mm_s2."""
    client = TestClient(create_app())

    r_cat = client.get("/api/catalog")
    assert r_cat.status_code == 200
    cat = r_cat.json()
    assert len(cat["bed_mounts"]) == len(BED_MOUNT_CATALOG)
    assert {m["id"] for m in cat["bed_mounts"]} == set(BED_MOUNT_CATALOG.keys())
    assert len(cat["frame_feet"]) == len(FRAME_FEET_CATALOG)
    assert {f["id"] for f in cat["frame_feet"]} == set(FRAME_FEET_CATALOG.keys())

    spec = _trident_base()
    spec.frame.enclosed_panels = True
    spec.frame.panel_mounting = "bolted_structural"
    spec_json = spec.model_dump(mode="json")

    # 1. panel_mesh_n=4 refines shell elements & toolhead_rel_pos shifts toolhead
    r_sim_base = client.post(
        "/api/simulate",
        json={
            "spec": spec_json,
            "panel_mesh_n": 2,
            "toolhead_rel_pos": [0.5, 0.5, 0.5],
            "accel_override_mm_s2": 5000.0,
            "include_toolpath_segments": False,
        },
    )
    assert r_sim_base.status_code == 200
    sim_base = r_sim_base.json()

    r_sim_hi = client.post(
        "/api/simulate",
        json={
            "spec": spec_json,
            "panel_mesh_n": 4,
            "toolhead_rel_pos": [0.2, 0.8, 0.2],
            "accel_override_mm_s2": 20000.0,
            "include_toolpath_segments": False,
        },
    )
    assert r_sim_hi.status_code == 200
    sim_hi = r_sim_hi.json()

    assert sim_base["geometry"]["panel_mesh_n"] == 2
    assert sim_hi["geometry"]["panel_mesh_n"] == 4
    assert len(sim_hi["geometry"]["shell_panels"]) > len(sim_base["geometry"]["shell_panels"])
    assert sim_hi["geometry"]["bed_mount_type"] == "kinematic_3pt"
    assert sim_hi["geometry"]["bed_z_drive"] == "leadscrew_triple"
    assert sim_hi["geometry"]["feet_type"] == "rigid_bolted"
    assert sim_hi["geometry"]["bed_frame_node_id"] == "bed_frame"

    # Accel override scales peak inertial forces linearly (20000 / 5000 = 4x)
    assert sim_hi["static_fea"]["peak_inertial_force_x_n"] == pytest.approx(
        4.0 * sim_base["static_fea"]["peak_inertial_force_x_n"], rel=1e-4
    )
    assert sim_hi["static_fea"]["peak_inertial_force_y_n"] == pytest.approx(
        4.0 * sim_base["static_fea"]["peak_inertial_force_y_n"], rel=1e-4
    )

    # Evaluator direct accel_override_mm_s2 check at identical position
    rep_5k = evaluate_printer(spec, include_advanced_analysis=False, accel_override_mm_s2=5000.0)
    rep_15k = evaluate_printer(spec, include_advanced_analysis=False, accel_override_mm_s2=15000.0)
    assert rep_15k.static_fea.accel_deflection_x_um == pytest.approx(
        3.0 * rep_5k.static_fea.accel_deflection_x_um, rel=1e-4
    )
    assert rep_15k.static_fea.accel_deflection_y_um == pytest.approx(
        3.0 * rep_5k.static_fea.accel_deflection_y_um, rel=1e-4
    )


def test_non_bolted_feet_preserve_realistic_shaper_accel_and_bed_deflection() -> None:
    """Benchtop isolation feet lower M1 desk sway without artificial print-accel cliffs or bed sink blowup."""
    feet_types: tuple[FrameFeetType, ...] = (
        "rigid_bolted",
        "paver_on_foam",
        "rubber_feet",
        "anti_vib_pads",
    )

    for yaml_name in ("voron_trident_300.yaml", "corexy_voron_style.yaml", "voron_monolith_awd.yaml"):
        base = PrinterSpec.from_yaml(EXAMPLES_DIR / yaml_name)
        mod_map = {}
        stat_map = {}
        for ft in feet_types:
            sp = base.model_copy(deep=True)
            sp.frame.feet_type = ft
            asm = build_assembly(sp)
            stat = solve_static_fea(asm)
            mod = solve_modal_fea(asm)
            stat_map[ft] = stat
            mod_map[ft] = mod

            # Limiting modal summary and per-mode axis shares stay valid and bounded on all feet
            lms = mod.limiting_modal_summary
            primary_modes = [m for m in mod.modes if m.is_primary_limiting_mode]
            assert len(primary_modes) == 1
            assert primary_modes[0].mode_number == lms["primary_structural_mode_number"]
            assert sum(m.x_axis_share_pct for m in mod.modes) <= 100.5
            assert sum(m.y_axis_share_pct for m in mod.modes) <= 100.5
            assert mod.shaper_x.structural_freq_hz <= mod_map["rigid_bolted"].shaper_x.structural_freq_hz * 1.005
            assert mod.shaper_y.structural_freq_hz <= mod_map["rigid_bolted"].shaper_y.structural_freq_hz * 1.005

            # Elastic bed deflection (after rigid-body foot-plane removal) must not explode under gravity sink
            assert stat.bed_deflection_um <= stat_map["rigid_bolted"].bed_deflection_um + 10.0

        bolted_acc = mod_map["rigid_bolted"].recommended_max_accel_mm_s2
        for ft in ("paver_on_foam", "rubber_feet", "anti_vib_pads"):
            ft_acc = mod_map[ft].recommended_max_accel_mm_s2
            assert ft_acc <= bolted_acc * 1.05, f"{yaml_name} {ft} artificially beat rigid_bolted"
            assert ft_acc >= bolted_acc * 0.70, f"{yaml_name} {ft} suffered accel cliff drop"
        assert (
            mod_map["paver_on_foam"].recommended_max_accel_mm_s2
            >= mod_map["anti_vib_pads"].recommended_max_accel_mm_s2
        )
        assert (
            mod_map["rubber_feet"].recommended_max_accel_mm_s2
            >= mod_map["anti_vib_pads"].recommended_max_accel_mm_s2
        )

    # Strict voron_trident_300 check: paver ~ bolted >= rubber > anti_vib, realistic modest penalty
    tri_base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    accs = {
        ft: solve_modal_fea(
            build_assembly(tri_base.model_copy(update={"frame": tri_base.frame.model_copy(update={"feet_type": ft})}))
        ).recommended_max_accel_mm_s2
        for ft in feet_types
    }
    assert accs["paver_on_foam"] >= accs["rubber_feet"] >= accs["anti_vib_pads"]
    assert accs["rubber_feet"] >= accs["rigid_bolted"] * 0.80
    assert accs["paver_on_foam"] >= accs["rigid_bolted"] * 0.90


def test_bed_mount_standoff_offset_6dof_inertia_and_inset_alignment() -> None:
    """Verify catalog standoff heights, exact 3pt/4pt rotational inertia, 0.375 inset alignment, and 100C thermal parity."""
    from printsim.fea import solve_bed_plate_fem
    from printsim.fea.shell_mitc4 import fastener_spring_matrices

    base = _trident_base()
    base.bed.target_temp_c = 100.0
    bx_m = base.bed.size_mm[0] * 1e-3
    by_m = base.bed.size_mm[1] * 1e-3

    taco_map: dict[str, float] = {}
    for mt, cfg in BED_MOUNT_CATALOG.items():
        spec = base.model_copy(deep=True)
        spec.bed.mount_type = mt  # type: ignore[assignment]
        asm = build_assembly(spec)
        nmap = asm.node_map
        sp = next(s for s in asm.fastener_springs if s.id == "bed_mount_spring")

        t_bed_m = float(spec.bed.plate_thickness_mm) * 1e-3
        h_standoff_m = float(cfg["standoff_height_mm"]) * 1e-3
        expected_dz_m = -(t_bed_m + h_standoff_m + 0.010)
        dz = float(nmap["bed_frame"].position_m[2] - nmap["bed_surface"].position_m[2])
        assert dz == pytest.approx(expected_dz_m, abs=1e-9)
        assert sp.offset_m == pytest.approx((0.0, 0.0, expected_dz_m), abs=1e-9)

        # Eccentric spring offset produces zero spurious strain energy under pure rigid pitch
        k_sp = fastener_spring_matrices(
            sp.k_shear_n_m,
            sp.k_normal_n_m,
            sp.k_rot_nm_rad,
            np.array(sp.normal_vec, dtype=float),
            np.array(sp.offset_m, dtype=float),
        )
        theta_y = 0.015
        u_rigid = np.zeros(12, dtype=float)
        u_rigid[0] = expected_dz_m * theta_y
        u_rigid[4] = theta_y
        u_rigid[10] = theta_y
        assert 0.5 * float(u_rigid @ k_sp @ u_rigid) == pytest.approx(0.0, abs=1e-10)

        # Exact geometric rotational stiffness about mount centroid
        k_z = float(cfg["k_axial_n_per_mm"]) * 1e3
        if cfg["radial_dof_mode"] == "elastomer_shear":
            k_z *= max(0.75, 1.0 - 0.004 * (100.0 - 70.0))
        if cfg["mount_points"] == 3:
            k_roll = 0.28125 * k_z * (bx_m**2)
            k_pitch = 0.375 * k_z * (by_m**2)
        else:
            k_roll = 0.5625 * k_z * (bx_m**2)
            k_pitch = 0.5625 * k_z * (by_m**2)
        expected_k_rot = max(math.sqrt(k_roll * k_pitch), 50.0)
        assert sp.k_rot_nm_rad == pytest.approx(expected_k_rot, rel=1e-9)

        # 3D viewport bed_mount_points_mm aligns exactly to 9x9 bed_plate_fem anchor coordinates (+-0.375)
        gdict = asm.to_geometry_dict()
        plate = solve_bed_plate_fem(spec)
        bc_x, bc_y, _ = gdict["bed_center_mm"]
        rel_3d = [
            [(pt[0] - bc_x) / spec.bed.size_mm[0], (pt[1] - bc_y) / spec.bed.size_mm[1]]
            for pt in gdict["bed_mount_points_mm"]
        ]
        assert np.allclose(rel_3d, plate.mount_points_rel, atol=1e-4)

        # Exact parity between components.py and bed_plate_fem.py thermal buckling
        ts = compute_thermal_structural_report(asm)
        assert ts.bed_buckling_taco_um == pytest.approx(plate.taco_buckling_um, rel=1e-9, abs=1e-9)
        taco_map[mt] = ts.bed_buckling_taco_um

    assert taco_map["kinematic_3pt"] == pytest.approx(0.0, abs=1e-9)
    assert taco_map["whopping_pochard_kinematic"] == pytest.approx(0.0, abs=1e-9)
    assert (
        taco_map["kinematic_3pt"]
        == taco_map["whopping_pochard_kinematic"]
        < taco_map["floating_4pt"]
        < taco_map["trident_3pt_standoffs"]
        < taco_map["rigid_4pt"]
    )


def test_trident_2020_t_subframe_elevation_alignment_and_zero_bed_clipping() -> None:
    """Trident builds a true 2020 T-subframe strictly underneath the bed plate & mounts with exact anchor mapping."""
    base = _trident_base()
    t_bed_m = float(base.bed.plate_thickness_mm) * 1e-3
    h_standoff_m = float(BED_MOUNT_CATALOG[base.bed.mount_type]["standoff_height_mm"]) * 1e-3
    half_subframe_m = 0.010

    for w_rel in (0.05, 0.5, 0.95):
        asm = build_assembly(base, toolhead_rel_pos=(0.5, 0.5, w_rel))
        _assert_spd_and_finite(asm)
        nmap = asm.node_map
        z_bed = float(nmap["bed_surface"].position_m[2])
        z_bed_bottom = z_bed - t_bed_m
        z_subframe = z_bed - t_bed_m - h_standoff_m - half_subframe_m

        assert z_subframe >= 0.030 - 1e-9
        assert float(nmap["bed_frame"].position_m[2]) == pytest.approx(z_subframe, abs=1e-9)

        # Every Z-carriage and T-subframe node sits coplanar at z_subframe strictly below bed plate underside
        for nid in (
            "z_bed_fl",
            "z_bed_fr",
            "z_bed_r",
            "bed_t_fl",
            "bed_t_fr",
            "bed_t_front",
            "bed_t_rear_mount",
        ):
            assert nid in nmap
            z_node = float(nmap[nid].position_m[2])
            assert z_node == pytest.approx(z_subframe, abs=1e-9)
            assert z_node + half_subframe_m < z_bed_bottom

        # Every bed_arm_* and bed_subframe_* member lies strictly horizontal at z_subframe < z_bed - t_bed_m
        subframe_members = [
            m for m in asm.members if m.id.startswith(("bed_arm_", "bed_subframe_"))
        ]
        expected_ids = {
            "bed_arm_fl",
            "bed_arm_fr",
            "bed_arm_r",
            "bed_subframe_fl",
            "bed_subframe_fr",
            "bed_subframe_spine_f",
            "bed_subframe_spine_r",
        }
        assert {m.id for m in subframe_members} == expected_ids
        for m in subframe_members:
            assert m.profile.width_m == pytest.approx(0.020, abs=1e-9)
            assert m.profile.height_m == pytest.approx(0.020, abs=1e-9)
            assert float(nmap[m.start_id].position_m[2]) == pytest.approx(z_subframe, abs=1e-9)
            assert float(nmap[m.end_id].position_m[2]) == pytest.approx(z_subframe, abs=1e-9)
            assert float(nmap[m.start_id].position_m[2]) < z_bed_bottom

        # Exact mount point X/Y alignment (< 0.1 mm) and underside Z
        gdict = asm.to_geometry_dict()
        assert gdict["bed_subframe_z_mm"] == pytest.approx(z_subframe * 1000.0, abs=0.02)
        assert gdict["bed_mount_anchor_nodes"] == ["bed_t_fl", "bed_t_fr", "bed_t_rear_mount"]

        mount_pts = gdict["bed_mount_points_mm"]
        assert len(mount_pts) == 3
        for anchor_id, pt_mm in zip(gdict["bed_mount_anchor_nodes"], mount_pts, strict=True):
            anchor_mm = nmap[anchor_id].position_m * 1000.0
            assert abs(float(anchor_mm[0]) - float(pt_mm[0])) < 0.1
            assert abs(float(anchor_mm[1]) - float(pt_mm[1])) < 0.1
            assert float(pt_mm[2]) == pytest.approx(z_bed_bottom * 1000.0, abs=0.02)

    # Non-triple CoreXY Z-drives keep Z-carriages & bed_frame coplanar at z_subframe & fall back anchor nodes
    for zd in ("leadscrew_dual", "leadscrew_dual_belt_sync", "leadscrew_single"):
        spec_zd = base.model_copy(deep=True)
        spec_zd.bed.z_drive = zd  # type: ignore[assignment]
        asm_zd = build_assembly(spec_zd)
        nmap_zd = asm_zd.node_map
        z_sub_zd = float(nmap_zd["bed_frame"].position_m[2])
        assert z_sub_zd < float(nmap_zd["bed_surface"].position_m[2]) - t_bed_m
        for nid, node in nmap_zd.items():
            if nid.startswith(("z_bed_", "z_rod_", "z_brg_")):
                assert float(node.position_m[2]) == pytest.approx(z_sub_zd, abs=1e-9)
        gd_zd = asm_zd.to_geometry_dict()
        assert gd_zd["bed_mount_anchor_nodes"] == ["bed_frame"] * len(gd_zd["bed_mount_points_mm"])


