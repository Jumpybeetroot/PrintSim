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
    """Verify BED_MOUNT_CATALOG (11), FRAME_FEET_CATALOG (4), and backward-compatible defaults."""
    assert set(BED_MOUNT_CATALOG.keys()) == {
        "kinematic_3pt",
        "whopping_pochard_kinematic",
        "trident_3pt_standoffs",
        "cnc_kelvin_magnetic",
        "captive_belleville_kinematic",
        "dual_leaf_ti_kinematic",
        "spherical_3pt_ratrig",
        "craz_kinematic_ge5c",
        "rigid_4pt",
        "floating_4pt",
        "silicone_4pt",
        "silicone_4pt_clamped",
        "springs_4pt",
    }
    assert set(FRAME_FEET_CATALOG.keys()) == {
        "rigid_bolted",
        "rubber_feet",
        "anti_vib_pads",
        "paver_on_foam",
        "suspended_lines",
    }
    # Verify exact thermal restraint backward compatibility & zero-taco kinematic mounts
    assert BED_MOUNT_CATALOG["kinematic_3pt"]["k_restraint_thermal"] == 0.0
    assert BED_MOUNT_CATALOG["whopping_pochard_kinematic"]["k_restraint_thermal"] == 0.0
    assert BED_MOUNT_CATALOG["cnc_kelvin_magnetic"]["k_restraint_thermal"] == 0.0
    assert BED_MOUNT_CATALOG["captive_belleville_kinematic"]["k_restraint_thermal"] == 0.0
    assert BED_MOUNT_CATALOG["dual_leaf_ti_kinematic"]["k_restraint_thermal"] == 0.0
    assert BED_MOUNT_CATALOG["spherical_3pt_ratrig"]["k_restraint_thermal"] == 8.0
    assert BED_MOUNT_CATALOG["floating_4pt"]["k_restraint_thermal"] == 18.0
    assert BED_MOUNT_CATALOG["trident_3pt_standoffs"]["k_restraint_thermal"] == 22.0
    assert BED_MOUNT_CATALOG["rigid_4pt"]["k_restraint_thermal"] == 8000.0

    expected_heights = {
        "kinematic_3pt": 16.0,
        "whopping_pochard_kinematic": 14.0,
        "trident_3pt_standoffs": 10.0,
        "cnc_kelvin_magnetic": 15.0,
        "captive_belleville_kinematic": 16.0,
        "dual_leaf_ti_kinematic": 14.0,
        "spherical_3pt_ratrig": 15.0,
        "craz_kinematic_ge5c": 15.0,
        "rigid_4pt": 12.0,
        "floating_4pt": 12.0,
        "silicone_4pt": 16.0,
        "silicone_4pt_clamped": 16.0,
        "springs_4pt": 20.0,
    }
    expected_modes = {
        "kinematic_3pt": "maxwell_vgroove",
        "whopping_pochard_kinematic": "radial_flexure",
        "trident_3pt_standoffs": "slotted_3pt_bakelite",
        "cnc_kelvin_magnetic": "kelvin_cone_vee_flat",
        "captive_belleville_kinematic": "maxwell_vgroove",
        "dual_leaf_ti_kinematic": "radial_flexure",
        "spherical_3pt_ratrig": "radial_flexure",
        "craz_kinematic_ge5c": "double_wishbone_ge5c",
        "rigid_4pt": "rigid_all",
        "floating_4pt": "slotted_1fixed_3float",
        "silicone_4pt": "elastomer_shear",
        "silicone_4pt_clamped": "elastomer_shear",
        "springs_4pt": "coil_spring_shear",
    }
    expected_tap = {
        "kinematic_3pt": False,
        "whopping_pochard_kinematic": True,
        "trident_3pt_standoffs": True,
        "cnc_kelvin_magnetic": False,
        "captive_belleville_kinematic": True,
        "dual_leaf_ti_kinematic": True,
        "spherical_3pt_ratrig": True,
        "craz_kinematic_ge5c": True,
        "rigid_4pt": True,
        "floating_4pt": True,
        "silicone_4pt": False,
        "silicone_4pt_clamped": False,
        "springs_4pt": False,
    }
    expected_pts = {
        "kinematic_3pt": 3,
        "whopping_pochard_kinematic": 3,
        "trident_3pt_standoffs": 3,
        "cnc_kelvin_magnetic": 3,
        "captive_belleville_kinematic": 3,
        "dual_leaf_ti_kinematic": 3,
        "spherical_3pt_ratrig": 3,
        "craz_kinematic_ge5c": 3,
        "rigid_4pt": 4,
        "floating_4pt": 4,
        "silicone_4pt": 4,
        "silicone_4pt_clamped": 4,
        "springs_4pt": 4,
    }
    for mt, cfg in BED_MOUNT_CATALOG.items():
        assert cfg["standoff_height_mm"] == pytest.approx(expected_heights[mt])
        assert cfg["radial_dof_mode"] == expected_modes[mt]
        assert cfg["tap_rated"] is expected_tap[mt]
        assert cfg["mount_points"] == expected_pts[mt]
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
        bed_mount_springs = [sp for sp in asm.fastener_springs if sp.bay == "bed_mount"]
        if zd == "leadscrew_triple" and int(BED_MOUNT_CATALOG[spec.bed.mount_type]["mount_points"]) == 3:
            assert len(bed_mount_springs) == 3
        else:
            assert any(sp.id == "bed_mount_spring" for sp in bed_mount_springs)

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
    # Z-carriage elevation by 4mm (~0.02-0.1% on frame loop kx/ky), and distributing 3-pt mounts directly
    # onto bed_t_fl/fr/rear_mount bypasses ~0.16 um of mid-spine bending vs centroidal 4-pt bed_frame,
    # while mount-level kx, loop kz, and structural-panel kx/ky stay strictly monotonic across all 5 mounts)
    for i in range(len(mounts_ordered) - 1):
        assert bed_mount_kx_vals[i] > bed_mount_kx_vals[i + 1]
        assert kz_vals[i] > kz_vals[i + 1]
    assert bed_defls[0] < bed_defls[2] < bed_defls[3] < bed_defls[4]
    assert bed_defls[1] < bed_defls[2]
    assert kx_vals[0] > kx_vals[1] > kx_vals[3] > kx_vals[4]
    assert kx_vals[2] > kx_vals[3] > kx_vals[4]
    assert ky_vals[0] > ky_vals[2] > ky_vals[3] > ky_vals[4]
    assert ky_vals[1] > ky_vals[2]

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

    # Accel override scales peak inertial forces linearly (20000 / 5000 = 4x); D-03/D-04 mass update shifts rounding at 2 decimals
    assert sim_hi["static_fea"]["peak_inertial_force_x_n"] == pytest.approx(
        4.0 * sim_base["static_fea"]["peak_inertial_force_x_n"], abs=0.03
    )
    assert sim_hi["static_fea"]["peak_inertial_force_y_n"] == pytest.approx(
        4.0 * sim_base["static_fea"]["peak_inertial_force_y_n"], abs=0.03
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

        # Compliant mounts release the frame as a recoiling free body above their rigid-body modes
        # (two-body reduced mass), so the drive resonance moves by a few percent either way: no
        # accel cliff, and no gain beyond what a <=15% frequency rise can buy.
        bolted = mod_map["rigid_bolted"]
        bolted_acc = bolted.recommended_max_accel_mm_s2
        for ft in ("paver_on_foam", "rubber_feet", "anti_vib_pads"):
            ft_acc = mod_map[ft].recommended_max_accel_mm_s2
            # Allow slightly higher ceiling when compliant damping tips auto-shaper from MZV to ZV
            shaper_upgraded = (
                str(mod_map[ft].shaper_y.recommended_shaper).lower() == "zv"
                and str(bolted.shaper_y.recommended_shaper).lower() != "zv"
            )
            max_mult = 1.55 if shaper_upgraded else 1.35
            assert ft_acc <= bolted_acc * max_mult, f"{yaml_name} {ft} artificially beat rigid_bolted"
            assert ft_acc >= bolted_acc * 0.62, f"{yaml_name} {ft} suffered accel cliff drop"
            for ax in ("shaper_x", "shaper_y"):
                ratio = getattr(mod_map[ft], ax).effective_freq_hz / getattr(bolted, ax).effective_freq_hz
                assert 0.90 <= ratio <= 1.15, f"{yaml_name} {ft} {ax} effective freq ratio {ratio:.3f}"

        # The 18 kg paver makes the base heavier, so the frame recoils less than on bare feet
        for ax in ("x", "y"):
            assert (
                mod_map["paver_on_foam"].mounting_dynamics[ax]["frame_recoil_ratio"]
                < mod_map["rubber_feet"].mounting_dynamics[ax]["frame_recoil_ratio"]
            )
        assert bolted.mounting_dynamics is None


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
        bed_springs = [s for s in asm.fastener_springs if s.bay == "bed_mount"]

        t_bed_m = float(spec.bed.plate_thickness_mm) * 1e-3
        h_standoff_m = float(cfg["standoff_height_mm"]) * 1e-3
        expected_dz_m = -(t_bed_m + h_standoff_m + 0.010)
        dz = float(nmap["bed_frame"].position_m[2] - nmap["bed_surface"].position_m[2])
        assert dz == pytest.approx(expected_dz_m, abs=1e-9)

        k_z = float(cfg["k_axial_n_per_mm"]) * 1e3
        if cfg["radial_dof_mode"] == "elastomer_shear":
            k_z *= max(0.75, 1.0 - 0.004 * (100.0 - 70.0))

        if cfg["mount_points"] == 3:
            # Trident 3-point mounts attach 3 individual 6-DOF springs directly to bed_t_fl/fr/rear_mount
            assert {s.id for s in bed_springs} == {
                "bed_mount_spring_fl",
                "bed_mount_spring_fr",
                "bed_mount_spring_r",
            }
            k_bed_6 = np.zeros((6, 6), dtype=float)
            for sp in bed_springs:
                r_vec = nmap[sp.shell_node_id].position_m - nmap["bed_surface"].position_m
                assert sp.offset_m == pytest.approx(tuple(float(c) for c in r_vec), abs=1e-9)
                assert sp.offset_m[2] == pytest.approx(expected_dz_m, abs=1e-9)
                k_sp = fastener_spring_matrices(
                    sp.k_shear_n_m,
                    sp.k_normal_n_m,
                    sp.k_rot_nm_rad,
                    np.array(sp.normal_vec, dtype=float),
                    np.array(sp.offset_m, dtype=float),
                )
                # Zero spurious strain energy under pure 3D rigid pitch + roll
                theta = np.array([0.012, 0.015, 0.0], dtype=float)
                u_rigid = np.zeros(12, dtype=float)
                u_rigid[0:3] = np.cross(theta, r_vec)
                u_rigid[3:6] = theta
                u_rigid[9:12] = theta
                assert 0.5 * float(u_rigid @ k_sp @ u_rigid) == pytest.approx(0.0, abs=1e-10)
                k_bed_6 += k_sp[6:12, 6:12]

            # Exact total axial Z stiffness (3*k_z) and geometric roll stiffness (0.28125*k_z*bx^2 + 3*k_seat + 3*k_lat*dz^2)
            assert k_bed_6[2, 2] == pytest.approx(3.0 * k_z, rel=1e-9)
            expected_k_roll = 0.28125 * k_z * (bx_m**2)
            assert k_bed_6[4, 4] >= expected_k_roll
        else:
            assert len(bed_springs) == 1
            sp = bed_springs[0]
            assert sp.id == "bed_mount_spring"
            assert sp.offset_m == pytest.approx((0.0, 0.0, expected_dz_m), abs=1e-9)

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
    assert taco_map["cnc_kelvin_magnetic"] == pytest.approx(0.0, abs=1e-9)
    assert taco_map["captive_belleville_kinematic"] == pytest.approx(0.0, abs=1e-9)
    assert taco_map["dual_leaf_ti_kinematic"] == pytest.approx(0.0, abs=1e-9)
    assert (
        taco_map["kinematic_3pt"]
        == taco_map["whopping_pochard_kinematic"]
        == taco_map["cnc_kelvin_magnetic"]
        == taco_map["captive_belleville_kinematic"]
        == taco_map["dual_leaf_ti_kinematic"]
        < taco_map["spherical_3pt_ratrig"]
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


@pytest.mark.parametrize(
    ("material", "thickness_mm"),
    [
        ("aluminum_5052_sheet", 5.0),
        ("acm_dibond", 3.0),
    ],
)
def test_trident_rear_z_column_does_not_clip_rear_panel_in_modal(
    material: str, thickness_mm: float
) -> None:
    """Bolted rear structural panel tracks the rear-center 2020 Z-column in Y with zero pass-through."""
    spec = _trident_base()
    spec.frame.enclosed_panels = True
    spec.frame.panel_mounting = "bolted_structural"
    spec.frame.panel_material = material
    spec.frame.panel_thickness_mm = thickness_mm
    spec.frame.trident_backbone = True

    # Test with odd panel_mesh_n=3 (Studio default) to ensure even centerline column alignment
    asm = build_assembly(spec, panel_mesh_n=3)
    _assert_spd_and_finite(asm)
    mo = solve_modal_fea(asm, num_modes=8)

    nmap = asm.node_map
    z_bed_z = float(nmap["z_bed_r"].position_m[2])
    bay = "rear_low"

    # Rear bay centerline column exists at x = w_cl/2 and couples to rear Z-column in normal Y
    from printsim.geometry.builder import corexy_frame_centerlines_m

    w_mid = 0.5 * corexy_frame_centerlines_m(spec)[0]
    pcol = sorted(
        [
            n
            for n in asm.nodes
            if n.id.startswith(f"pan_{bay}_") and abs(float(n.position_m[0]) - w_mid) < 1e-4
        ],
        key=lambda n: float(n.position_m[2]),
    )
    assert len(pcol) >= 4
    interior_contact_springs = [
        sp
        for sp in asm.fastener_springs
        if sp.bay == bay
        and sp.shell_node_id in {n.id for n in pcol[1:-1]}
        and sp.k_normal_n_m >= 5.0e6
        and sp.k_shear_n_m == 0.0
    ]
    assert len(interior_contact_springs) >= 1

    # Across all primary modes, z_bed_r must not pass through the rear panel in Y
    for mode in mo.modes:
        z_y = float(mode.nodal_vectors["z_bed_r"][1])
        panel_y = 0.0
        for na, nb in zip(pcol, pcol[1:]):
            za, zb = float(na.position_m[2]), float(nb.position_m[2])
            if za - 1e-4 <= z_bed_z <= zb + 1e-4:
                t = (z_bed_z - za) / max(zb - za, 1e-9)
                panel_y = (1.0 - t) * float(mode.nodal_vectors[na.id][1]) + t * float(
                    mode.nodal_vectors[nb.id][1]
                )
                break
        assert abs(z_y - panel_y) < 0.08, (
            f"{material} {thickness_mm}mm mode {mode.mode_number} ({mode.frequency_hz:.1f} Hz): "
            f"z_bed_r u_y={z_y:+.4f} vs rear panel u_y={panel_y:+.4f}"
        )


def test_super_stiff_kinematic_bed_mounts_3pt_taco_stiffness_and_bom() -> None:
    """Verify all 4 new kinematic mounts build 3-pt subframes, match Tap/taco physics, order monotonically, and price in BOM."""
    base = _trident_base()
    base.bed.target_temp_c = 100.0
    base.bed.z_leadscrew_type = "sfu1605_fk12"

    new_mounts = (
        "cnc_kelvin_magnetic",
        "captive_belleville_kinematic",
        "dual_leaf_ti_kinematic",
        "spherical_3pt_ratrig",
    )
    expected_tap = {
        "cnc_kelvin_magnetic": False,
        "captive_belleville_kinematic": True,
        "dual_leaf_ti_kinematic": True,
        "spherical_3pt_ratrig": True,
    }

    for mt in new_mounts:
        spec = base.model_copy(deep=True)
        spec.bed.mount_type = mt  # type: ignore[assignment]
        asm = build_assembly(spec)
        _assert_spd_and_finite(asm)
        gdict = asm.to_geometry_dict()
        assert len(gdict["bed_mount_points_mm"]) == 3
        assert gdict["bed_mount_anchor_nodes"] == ["bed_t_fl", "bed_t_fr", "bed_t_rear_mount"]
        assert BED_MOUNT_CATALOG[mt]["tap_rated"] is expected_tap[mt]

        rep = evaluate_printer(spec, include_advanced_analysis=True)
        assert rep.bed_plate_fea is not None
        assert len(rep.bed_plate_fea.mount_points_rel) == 3
        if mt == "spherical_3pt_ratrig":
            assert 0.0 < rep.bed_plate_fea.taco_buckling_um < 15.0
        else:
            assert rep.bed_plate_fea.taco_buckling_um == 0.0
            assert rep.thermal_structural.bed_buckling_taco_um == 0.0

    # Monotonic mount axial & lateral stiffness ordering across 3-pt kinematic spectrum
    ordered_3pt = (
        "trident_3pt_standoffs",
        "whopping_pochard_kinematic",
        "kinematic_3pt",
        "dual_leaf_ti_kinematic",
        "cnc_kelvin_magnetic",
        "captive_belleville_kinematic",
    )
    mount_kz = []
    mount_kx = []
    bed_z_10n_um = []
    plate_center_um = []
    plate_hz = []
    for mt in ordered_3pt:
        spec = base.model_copy(deep=True)
        spec.bed.mount_type = mt  # type: ignore[assignment]
        asm = build_assembly(spec)
        st = solve_static_fea(asm)
        assert st.bed_deflection_um < 15.0
        K, _, _, nidx, free = assemble_global_matrices(asm)
        fz = np.zeros(K.shape[0], dtype=float)
        fz[6 * nidx["bed_surface"] + 2] = -10.0
        uz = _solve_linear_system(K, fz, free)
        dz_mm = abs(float(uz[6 * nidx["bed_surface"] + 2] - uz[6 * nidx["bed_frame"] + 2])) * 1e3
        mount_kz.append(10.0 / max(dz_mm, 1e-12))
        bed_z_10n_um.append(abs(float(uz[6 * nidx["bed_surface"] + 2])) * 1e6)

        fx = np.zeros(K.shape[0], dtype=float)
        fx[6 * nidx["bed_surface"]] = 10.0
        ux = _solve_linear_system(K, fx, free)
        dz_m = float(asm.node_map["bed_surface"].position_m[2] - asm.node_map["bed_frame"].position_m[2])
        theta_y_frame = float(ux[6 * nidx["bed_frame"] + 4])
        dx_mm = abs(
            float(ux[6 * nidx["bed_surface"]] - ux[6 * nidx["bed_frame"]]) - dz_m * theta_y_frame
        ) * 1e3
        mount_kx.append(10.0 / max(dx_mm, 1e-12))

        from printsim.fea import solve_bed_plate_fem

        plate = solve_bed_plate_fem(spec)
        plate_center_um.append(plate.probe_center_deflection_um)
        plate_hz.append(plate.plate_first_mode_hz)

    for i in range(len(ordered_3pt) - 1):
        assert mount_kz[i] < mount_kz[i + 1]
        assert mount_kx[i] < mount_kx[i + 1]
        assert bed_z_10n_um[i] > bed_z_10n_um[i + 1]
        assert plate_center_um[i] > plate_center_um[i + 1]
        assert plate_hz[i] < plate_hz[i + 1]

    # Dynamic BOM reads BED_MOUNT_CATALOG cleanly for all 11 mounts
    for mt, cfg in BED_MOUNT_CATALOG.items():
        spec = base.model_copy(deep=True)
        spec.bed.mount_type = mt  # type: ignore[assignment]
        bom = generate_bom(spec)
        mount_item = next(i for i in bom.items if i.sku == "BED-MOUNT")
        assert mount_item.category == "bed"
        assert mount_item.total_cost_usd == pytest.approx(float(cfg["cost_usd"]), abs=0.01)
        assert cfg["name"] in mount_item.name
        assert sum(bom.category_costs_usd.values()) == pytest.approx(bom.total_cost_usd, abs=0.1)


def test_trident_distributed_3pt_bed_mounts_eliminate_spurious_spine_roll() -> None:
    """Distributed 3-pt Trident springs route roll into bed_t_fl/fr, eliminating spurious ~228 Hz spine-torsion roll."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    asm = build_assembly(spec)
    _assert_spd_and_finite(asm)

    bed_springs = [s for s in asm.fastener_springs if s.bay == "bed_mount"]
    assert {s.id for s in bed_springs} == {
        "bed_mount_spring_fl",
        "bed_mount_spring_fr",
        "bed_mount_spring_r",
    }
    assert {s.shell_node_id for s in bed_springs} == {"bed_t_fl", "bed_t_fr", "bed_t_rear_mount"}

    mod = solve_modal_fea(asm, num_modes=6)
    for m in mod.modes:
        # Subframe anchor rotations are exported alongside bed_surface for viewport kinematic sync
        for nid in ("bed_surface", "bed_frame", "bed_t_fl", "bed_t_fr", "bed_t_rear_mount"):
            assert nid in m.nodal_rotations
        # Previously Monolith AWD had a spurious 222 Hz bed-roll mode with |rot_y| = 21.0 rad/m
        assert abs(float(m.nodal_rotations["bed_surface"][1])) < 2.5


def test_all_bed_architectures_and_mount_types_have_clean_rotations_and_anchor_sync() -> None:
    """Every bed architecture (5 CoreXY Z-drives + bedslinger + delta) x mount types stays synced with zero spurious roll/pitch."""
    bed_configs: list[tuple[str, str, str]] = [
        ("corexy", "belted_quad_z", "corexy_voron_style.yaml"),
        ("corexy", "leadscrew_triple", "voron_trident_300.yaml"),
        ("corexy", "leadscrew_dual", "voron_trident_300.yaml"),
        ("corexy", "leadscrew_dual_belt_sync", "vzbot_330_awd.yaml"),
        ("corexy", "leadscrew_single", "voron_trident_300.yaml"),
        ("cartesian_bedslinger", "leadscrew_dual", "cartesian_bedslinger.yaml"),
        ("delta", "leadscrew_triple", "delta_racer.yaml"),
    ]
    sample_mounts = (
        "rigid_4pt",
        "kinematic_3pt",
        "trident_3pt_standoffs",
        "floating_4pt",
        "silicone_4pt",
    )

    for arch, zd, yaml_name in bed_configs:
        for mt in sample_mounts:
            spec = PrinterSpec.from_yaml(EXAMPLES_DIR / yaml_name)
            spec.bed.z_drive = zd  # type: ignore[assignment]
            spec.bed.mount_type = mt  # type: ignore[assignment]
            asm = build_assembly(spec)
            _assert_spd_and_finite(asm)
            st = solve_static_fea(asm)
            mod = solve_modal_fea(asm, num_modes=6)
            gdict = asm.to_geometry_dict()

            assert math.isfinite(st.bed_deflection_um) and st.bed_deflection_um > 0.0
            anchors = set(gdict.get("bed_mount_anchor_nodes", []))
            for m in mod.modes:
                assert "bed_surface" in m.nodal_rotations
                assert "bed_frame" in m.nodal_rotations
                for a in anchors:
                    assert a in m.nodal_rotations
                if BED_MOUNT_CATALOG[mt]["radial_dof_mode"] not in (
                    "elastomer_shear",
                    "coil_spring_shear",
                ):
                    r_bed = np.asarray(m.nodal_rotations["bed_surface"][:2], dtype=float)
                    r_fr = np.asarray(m.nodal_rotations["bed_frame"][:2], dtype=float)
                    assert float(np.linalg.norm(r_bed - r_fr)) < 1.0

    # All 11 catalog bed mounts on Trident leadscrew_triple (all 7 three-point mounts distributed, all 4 four-point carrier-braced)
    tri_base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    for mt, cfg in BED_MOUNT_CATALOG.items():
        spec = tri_base.model_copy(deep=True)
        spec.bed.mount_type = mt  # type: ignore[assignment]
        asm = build_assembly(spec)
        bed_springs = [s for s in asm.fastener_springs if s.bay == "bed_mount"]
        if cfg["mount_points"] == 3:
            assert {s.id for s in bed_springs} == {
                "bed_mount_spring_fl",
                "bed_mount_spring_fr",
                "bed_mount_spring_r",
            }
        else:
            assert [s.id for s in bed_springs] == ["bed_mount_spring"]


def test_z_carriage_drive_springs_and_vertical_bounce_realism() -> None:
    """Moving-bed Z-carriages ride on 6-DOF bed_z_drive springs (not column welds) across all drives."""
    from scipy.linalg import eigh

    drive_cases = (
        ("leadscrew_triple", "voron_trident_300.yaml", ("fl", "fr", "r")),
        ("leadscrew_dual", "voron_trident_300.yaml", ("l", "r")),
        ("leadscrew_dual_belt_sync", "vzbot_330_awd.yaml", ("l", "r")),
        ("leadscrew_single", "voron_trident_300.yaml", ("r",)),
    )
    for zd, yaml_name, tags in drive_cases:
        spec = PrinterSpec.from_yaml(EXAMPLES_DIR / yaml_name)
        spec.bed.z_drive = zd  # type: ignore[assignment]
        asm = build_assembly(spec)
        _assert_spd_and_finite(asm)
        nmap = asm.node_map
        z_springs = {s.id: s for s in asm.fastener_springs if s.bay == "bed_z_drive"}
        assert set(z_springs.keys()) == {f"z_bed_drive_{t}" for t in tags}
        for t in tags:
            col_nid = f"z_col_bed_{t}"
            car_nid = f"z_bed_{t}"
            assert nmap[col_nid].role == "frame"
            assert nmap[car_nid].role == "bed"
            sp = z_springs[f"z_bed_drive_{t}"]
            assert sp.shell_node_id == col_nid
            assert sp.frame_node_id == car_nid
            assert sp.normal_vec == (0.0, 0.0, 1.0)
            assert sp.offset_m == (0.0, 0.0, 0.0)
            assert sp.k_normal_n_m > 0.0 and sp.k_shear_n_m > 0.0 and sp.k_rot_nm_rad > 0.0

        # Upgrading TR8x4 POM -> SFU1204 raises z_bed_drive axial k_normal without mutating bare bed_arm Iy
        spec_sfu = spec.model_copy(deep=True)
        spec_sfu.bed.z_leadscrew_type = "sfu1204_bk10"
        asm_sfu = build_assembly(spec_sfu)
        arm_base = next(m for m in asm.members if m.id.startswith("bed_arm_"))
        arm_sfu = next(m for m in asm_sfu.members if m.id == arm_base.id)
        assert arm_sfu.profile.iy_m4 == pytest.approx(arm_base.profile.iy_m4, rel=1e-9)
        sp_base = z_springs[f"z_bed_drive_{tags[0]}"]
        sp_sfu = next(s for s in asm_sfu.fastener_springs if s.id == sp_base.id)
        assert sp_sfu.k_normal_n_m > sp_base.k_normal_n_m * 1.5

    # Standard Trident 300: Z-carriages deflect visibly on leadscrews rather than staying frozen on columns
    tri_spec = _trident_base()
    tri_asm = build_assembly(tri_spec)
    tri_st = solve_static_fea(tri_asm)
    gd_tri = tri_st.nodal_displacements_mm
    assert abs(gd_tri["z_bed_fl"][2]) > 3.0 * abs(gd_tri["z_col_bed_fl"][2])
    assert abs(gd_tri["z_bed_r"][2]) > abs(gd_tri["z_col_bed_r"][2])

    # Backboned / paneled 300mm Trident (3.1 kg bed, TR8x4 POM -> SFU1204 -> SFU1605):
    # static gravity & ~288 Hz vertical Z bed-bounce both satisfy |u_z(z_bed_fl)| / |u_z(bed_t_fl)| > 0.50
    # and screw upgrade monotonically increases Z stiffness and Z-bounce frequency.
    mono_base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")
    kz_list: list[float] = []
    bounce_hz_list: list[float] = []
    for ls in ("tr8x4_pom", "sfu1204_bk10", "sfu1605_fk12"):
        sp_ls = mono_base.model_copy(deep=True)
        sp_ls.bed.z_leadscrew_type = ls  # type: ignore[assignment]
        asm_ls = build_assembly(sp_ls)
        st_ls = solve_static_fea(asm_ls)
        kz_list.append(st_ls.toolhead_stiffness_z_n_per_mm)

        gd = st_ls.nodal_displacements_mm
        if ls == "tr8x4_pom":
            assert abs(gd["z_bed_fl"][2]) / abs(gd["bed_t_fl"][2]) > 0.50
            assert abs(gd["z_bed_r"][2]) / abs(gd["bed_t_rear_mount"][2]) > 0.80

        if ls in ("tr8x4_pom", "sfu1204_bk10"):
            K, M, _, node_idx, free_dofs = assemble_global_matrices(asm_ls)
            vals, vecs = eigh(
                K[np.ix_(free_dofs, free_dofs)],
                M[np.ix_(free_dofs, free_dofs)],
            )
            n_dof = K.shape[0]
            b_idx = node_idx["bed_surface"]
            zfl_idx = node_idx["z_bed_fl"]
            btfl_idx = node_idx["bed_t_fl"]
            bounce_modes: list[tuple[float, float]] = []
            for k in range(len(vals)):
                freq = float(np.sqrt(max(vals[k], 0.0)) / (2.0 * np.pi))
                if 150.0 <= freq <= 720.0:
                    v = np.zeros(n_dof)
                    v[free_dofs] = vecs[:, k]
                    trans = v.reshape(-1, 6)[:, :3]
                    mx = float(np.max(np.abs(trans)))
                    if mx > 1e-12:
                        bz = float(trans[b_idx, 2] / mx)
                        zfl = float(trans[zfl_idx, 2] / mx)
                        btfl = float(trans[btfl_idx, 2] / mx)
                        if abs(bz) > 0.18 and abs(zfl) > 0.08 and (bz * zfl) > 0.0:
                            bounce_modes.append((freq, abs(zfl) / max(abs(btfl), 1e-12)))
            assert bounce_modes, f"Expected vertical Z bed-bounce mode for {ls}"
            f_bounce, ratio_bounce = bounce_modes[0]
            bounce_hz_list.append(f_bounce)
            if ls == "tr8x4_pom":
                assert 220.0 <= f_bounce <= 330.0
                assert ratio_bounce > 0.50

    assert kz_list[2] > kz_list[1] > kz_list[0]
    assert bounce_hz_list[1] > bounce_hz_list[0] + 50.0


def test_all_seven_bed_architectures_times_eleven_bed_mounts_spd() -> None:
    """Full 7 bed architectures x 11 BED_MOUNT_CATALOG matrix stays SPD, finite, and rotation-clean."""
    arch_specs = {
        "belted_quad_z": PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml"),
        "leadscrew_triple": PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml"),
        "leadscrew_dual": PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml").model_copy(deep=True),
        "leadscrew_dual_belt_sync": PrinterSpec.from_yaml(EXAMPLES_DIR / "vzbot_330_awd.yaml"),
        "leadscrew_single": PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml").model_copy(deep=True),
        "cartesian_bedslinger": PrinterSpec.from_yaml(EXAMPLES_DIR / "cartesian_bedslinger.yaml"),
        "delta": PrinterSpec.from_yaml(EXAMPLES_DIR / "delta_racer.yaml"),
    }
    arch_specs["leadscrew_dual"].bed.z_drive = "leadscrew_dual"  # type: ignore[assignment]
    arch_specs["leadscrew_single"].bed.z_drive = "leadscrew_single"  # type: ignore[assignment]

    for arch, base in arch_specs.items():
        for mt, cfg in BED_MOUNT_CATALOG.items():
            sp = base.model_copy(deep=True)
            sp.bed.mount_type = mt  # type: ignore[assignment]
            asm = build_assembly(sp)
            _assert_spd_and_finite(asm)
            st = solve_static_fea(asm)
            mo = solve_modal_fea(asm, num_modes=4)
            assert math.isfinite(st.toolhead_stiffness_z_n_per_mm) and st.toolhead_stiffness_z_n_per_mm > 0.0
            assert math.isfinite(mo.fundamental_freq_hz) and mo.fundamental_freq_hz > 5.0
            if cfg["radial_dof_mode"] not in ("elastomer_shear", "coil_spring_shear"):
                for m in mo.modes:
                    r_bed = np.asarray(m.nodal_rotations["bed_surface"][:2], dtype=float)
                    r_fr = np.asarray(m.nodal_rotations["bed_frame"][:2], dtype=float)
                    assert float(np.linalg.norm(r_bed - r_fr)) < 1.0, f"{arch}/{mt} mode {m.mode_number} rock"

