"""Calibration verification tests for CAL-08 (bed mounts), CAL-10 (Z-drive CMS), and CAL-14 (clip-in panels)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from printsim.fea.bed_plate_fem import (
    CAL_BED_MOUNT_K_THERM_N_PER_M,
    _compute_carrier_z_stiffness,
    compute_bed_bimetallic_pei_bow_um,
    compute_nozzle_tap_z_deflection_um,
    solve_bed_plate_fem,
)
from printsim.fea.modal import solve_modal_fea
from printsim.fea.shell_mitc4 import (
    clip_gasket_stiffness_per_m,
    mitc4_panel_bay_racking_stiffness,
)
from printsim.fea.solver import solve_static_fea
from printsim.geometry._builder_bed import (
    _compute_corexy_z_drive_corner_stiffnesses,
    resolve_cad_z_drive_mount_stiffness_n_m,
)
from printsim.geometry.builder import build_assembly
from printsim.schema import PrinterSpec, get_material

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


# ---------------------------------------------------------------------------
# CAL-08: Bed Mount Hardware — Thermal Restraint & Bimetallic PEI Bed Warp
# ---------------------------------------------------------------------------


def test_cal08_bimetallic_pei_bow_physics_and_scaling() -> None:
    """Volkersen/Timoshenko magnetic PEI stack bow is 0 at 25 C and scales with t^-2 and L^2."""
    spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    spec.bed.target_temp_c = 25.0
    assert compute_bed_bimetallic_pei_bow_um(spec) == 0.0

    spec.bed.target_temp_c = 110.0
    bow_8mm = compute_bed_bimetallic_pei_bow_um(spec)
    assert 8.0 <= bow_8mm <= 15.0

    # Thinner 4 mm plate has ~8x lower D and ~1.6x smaller lever arm -> > 2.5x larger bow
    spec_4mm = spec.model_copy(deep=True)
    spec_4mm.bed.plate_thickness_mm = 4.0
    bow_4mm = compute_bed_bimetallic_pei_bow_um(spec_4mm)
    assert bow_4mm > bow_8mm * 2.5

    # Larger bed span increases bimetallic bow monotonically (L^2 * gamma_mag(L))
    bows_by_span = []
    for size_mm in (200.0, 300.0, 350.0):
        s_sz = spec.model_copy(deep=True)
        s_sz.bed.size_mm = [size_mm, size_mm]
        bows_by_span.append(compute_bed_bimetallic_pei_bow_um(s_sz))
    assert bows_by_span[0] < bows_by_span[1] < bows_by_span[2]


def test_cal08_klipper_bed_mesh_cold_vs_110c_deltas() -> None:
    """Calibrated thermal warp matches real Klipper BED_MESH_OUTPUT cold-vs-110 C benchmarks."""
    trident = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    trident.bed.target_temp_c = 110.0
    trident.frame.chamber_temp_c = 55.0

    # All 6 true kinematic mounts produce identical low thermal warp (18-35 um, zero taco)
    kin_types = (
        "kinematic_3pt",
        "whopping_pochard_kinematic",
        "cnc_kelvin_magnetic",
        "captive_belleville_kinematic",
        "dual_leaf_ti_kinematic",
        "craz_kinematic_ge5c",
    )
    kin_ptvs = []
    for mt in kin_types:
        s = trident.model_copy(deep=True)
        s.bed.mount_type = mt  # type: ignore[assignment]
        res = solve_bed_plate_fem(s)
        assert res.taco_buckling_um == 0.0
        assert 18.0 <= res.thermal_peak_to_valley_um <= 35.0
        assert res.first_layer_risk == "low"
        kin_ptvs.append(res.thermal_peak_to_valley_um)
    for ptv in kin_ptvs[1:]:
        assert ptv == pytest.approx(kin_ptvs[0], rel=1e-9)

    # Stock Voron Trident Bakelite 3-pt standoffs: 50-95 um @ 110 C / 55 C chamber
    s_tri_bak = trident.model_copy(deep=True)
    s_tri_bak.bed.mount_type = "trident_3pt_standoffs"
    r_tri_bak = solve_bed_plate_fem(s_tri_bak)
    assert 50.0 <= r_tri_bak.thermal_peak_to_valley_um <= 95.0

    # Voron 2.4r2 "1 tight, 3 loose" floating 4-pt standoffs: 40-75 um @ 110 C / 55 C
    v24_flt = PrinterSpec.from_yaml(EXAMPLES_DIR / "ldo_voron_24_revd_300.yaml")
    v24_flt.bed.mount_type = "floating_4pt"
    v24_flt.bed.target_temp_c = 110.0
    v24_flt.frame.chamber_temp_c = 55.0
    r_v24_flt = solve_bed_plate_fem(v24_flt)
    assert 40.0 <= r_v24_flt.thermal_peak_to_valley_um <= 75.0

    # Voron 2.4r2 over-tightened 4-pt rigid standoffs ("taco bed"): 180-320 um @ 110 C / 55 C
    v24_rig = v24_flt.model_copy(deep=True)
    v24_rig.bed.mount_type = "rigid_4pt"
    r_v24_rig = solve_bed_plate_fem(v24_rig)
    assert 180.0 <= r_v24_rig.thermal_peak_to_valley_um <= 320.0
    assert r_v24_rig.taco_buckling_um > 100.0
    assert r_v24_rig.first_layer_risk == "critical"

    # VzBot 330 torqued M3x50 silicone spacers: 35-65 um @ 110 C / 55 C
    vz = PrinterSpec.from_yaml(EXAMPLES_DIR / "vzbot_330_awd.yaml")
    vz.bed.target_temp_c = 110.0
    vz.frame.chamber_temp_c = 55.0
    r_vz = solve_bed_plate_fem(vz)
    assert 35.0 <= r_vz.thermal_peak_to_valley_um <= 65.0

    # Strict monotonic ordering on identical 300x300x8 mm plate @ 110 C / 55 C
    ordered_mounts = (
        "kinematic_3pt",
        "silicone_4pt_clamped",
        "spherical_3pt_ratrig",
        "floating_4pt",
        "trident_3pt_standoffs",
        "rigid_4pt",
    )
    ptvs = []
    for mt in ordered_mounts:
        s = trident.model_copy(deep=True)
        s.bed.mount_type = mt  # type: ignore[assignment]
        ptvs.append(solve_bed_plate_fem(s).thermal_peak_to_valley_um)
    for i in range(len(ptvs) - 1):
        assert ptvs[i] < ptvs[i + 1], f"Expected {ordered_mounts[i]} < {ordered_mounts[i + 1]}"

    # User model_overrides.bed_mount.k_restraint_thermal_n_per_mm takes precedence over calibration table
    s_ov = s_tri_bak.model_copy(deep=True)
    s_ov.model_overrides.bed_mount.k_restraint_thermal_n_per_mm = 0.0
    r_ov = solve_bed_plate_fem(s_ov)
    assert r_ov.thermal_peak_to_valley_um == pytest.approx(kin_ptvs[0], rel=1e-6)
    assert CAL_BED_MOUNT_K_THERM_N_PER_M["trident_3pt_standoffs"] > CAL_BED_MOUNT_K_THERM_N_PER_M["floating_4pt"]


# ---------------------------------------------------------------------------
# CAL-10: Z-Drive Mechanisms — Baked CMS Wiring, Belt Elasticity & Nozzle Tap
# ---------------------------------------------------------------------------


def test_cal10_baked_z_drive_cms_wired_into_voron24_and_trident() -> None:
    """Baked voron_24_z_drive_* and voron_trident_z_mount_* CMS wire directly into Z vertical stiffness."""
    # 1. Voron 2.4r2: baked voron_24_z_drive_left/right at all 4 flying-gantry corners
    v24 = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    v24_cms = resolve_cad_z_drive_mount_stiffness_n_m(v24)
    assert set(v24_cms.keys()) == {"fl", "fr", "rl", "rr"}
    assert v24_cms["fl"] == pytest.approx(v24_cms["rl"], rel=1e-9)
    assert v24_cms["fr"] == pytest.approx(v24_cms["rr"], rel=1e-9)
    assert v24_cms["fl"] == pytest.approx(v24_cms["fr"], rel=0.02)
    assert 8.0e6 <= v24_cms["fl"] <= 16.0e6  # ABS Z-drive pod vertical stiffness ~12.6 MN/m

    # Upgrading bracket_material ABS -> CNC 6061 stiffens the Z-drive pod and z_joint_fl normal stiffness
    v24_cnc = v24.model_copy(deep=True)
    v24_cnc.gantry.bracket_material = "aluminum_6061_t6"
    v24_nocms = v24.model_copy(deep=True)
    v24_nocms.use_cad_superelements = False

    asm_abs = build_assembly(v24)
    asm_cnc = build_assembly(v24_cnc)
    asm_nocms = build_assembly(v24_nocms)
    kz_abs = next(sp.k_normal_n_m for sp in asm_abs.fastener_springs if sp.id == "z_joint_fl")
    kz_cnc = next(sp.k_normal_n_m for sp in asm_cnc.fastener_springs if sp.id == "z_joint_fl")
    kz_nocms = next(sp.k_normal_n_m for sp in asm_nocms.fastener_springs if sp.id == "z_joint_fl")
    assert kz_abs < kz_cnc < kz_nocms

    # 2. Voron Trident: baked voron_trident_z_mount_{left,right,rear} wired per corner
    tri = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    tri_cms = resolve_cad_z_drive_mount_stiffness_n_m(tri)
    assert set(tri_cms.keys()) == {"fl", "fr", "r"}
    assert tri_cms["fl"] == pytest.approx(tri_cms["fr"], rel=1e-9)
    assert tri_cms["r"] > tri_cms["fl"] * 1.4  # rear mount is stiffer than front corner mounts

    asm_tri = build_assembly(tri)
    z_sp = {sp.id: sp.k_normal_n_m for sp in asm_tri.fastener_springs if sp.bay == "bed_z_drive"}
    assert set(z_sp.keys()) == {"z_bed_drive_fl", "z_bed_drive_fr", "z_bed_drive_r"}
    assert z_sp["z_bed_drive_fl"] == pytest.approx(z_sp["z_bed_drive_fr"], rel=1e-9)
    assert z_sp["z_bed_drive_r"] > z_sp["z_bed_drive_fl"] * 1.15

    # Trident Backbone & CNC 6061 motor mounts substantially stiffen the Z vertical chain
    tri_bb = tri.model_copy(deep=True)
    tri_bb.frame.trident_backbone = True
    tri_cnc = tri.model_copy(deep=True)
    tri_cnc.gantry.bracket_material = "aluminum_6061_t6"
    k_car_abs = _compute_carrier_z_stiffness(tri)
    k_car_bb = _compute_carrier_z_stiffness(tri_bb)
    k_car_cnc = _compute_carrier_z_stiffness(tri_cnc)
    assert k_car_abs < k_car_bb
    assert k_car_abs < k_car_cnc * 0.65


def test_cal10_z_belt_height_scaling_pom_softening_and_nozzle_tap() -> None:
    """Verify two-span Z-belt height scaling, POM thermal softening, and 6 N nozzle-tap deflections."""
    # 1. Z-belt two-span parallel stiffness scales inversely with frame height (250 > 300 > 350 mm)
    v24 = PrinterSpec.from_yaml(EXAMPLES_DIR / "corexy_voron_style.yaml")
    k_by_h = []
    for h_mm in (430.0, 480.0, 560.0):
        s_h = v24.model_copy(deep=True)
        s_h.frame.height_mm = h_mm
        ck = _compute_corexy_z_drive_corner_stiffnesses(
            s_h, z_gantry=0.5 * h_mm * 1e-3, z_bed=0.08, is_flying=True, trident_backbone_active=False
        )
        k_by_h.append(ck["fl"])
    assert k_by_h[0] > k_by_h[1] > k_by_h[2]

    # 2. POM nut softens in a 60 C chamber while steel SFU1204 ballscrew stays invariant
    tri_pom = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    tri_pom.use_cad_superelements = False  # isolate nut thermal effect from ABS mount softening
    tri_pom.frame.chamber_temp_c = 25.0
    k_pom_25 = _compute_carrier_z_stiffness(tri_pom)
    tri_pom_60 = tri_pom.model_copy(deep=True)
    tri_pom_60.frame.chamber_temp_c = 60.0
    k_pom_60 = _compute_carrier_z_stiffness(tri_pom_60)
    assert k_pom_60 < k_pom_25 * 0.94

    tri_sfu = tri_pom.model_copy(deep=True)
    tri_sfu.bed.z_leadscrew_type = "sfu1204_bk10"
    k_sfu_25 = _compute_carrier_z_stiffness(tri_sfu)
    tri_sfu_60 = tri_sfu.model_copy(deep=True)
    tri_sfu_60.frame.chamber_temp_c = 60.0
    k_sfu_60 = _compute_carrier_z_stiffness(tri_sfu_60)
    assert k_sfu_60 == pytest.approx(k_sfu_25, rel=1e-9)

    # 3. Voron Tap 6.0 N (~610 gf) nozzle-tap vertical deflection benchmarks
    tri_stock = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    tap_tri = compute_nozzle_tap_z_deflection_um(tri_stock, tap_force_n=6.0)
    assert 2.0 <= tap_tri["total_center_tap_deflection_um"] <= 6.0
    assert tap_tri["active_tap_backlash_um"] == 0.0  # gravity-seated POM threads under downward tap
    assert tap_tri["reversal_backlash_um"] == pytest.approx(28.0)

    tri_up = tri_stock.model_copy(deep=True)
    tri_up.gantry.bracket_material = "aluminum_6061_t6"
    tri_up.bed.z_leadscrew_type = "sfu1204_bk10"
    tap_tri_up = compute_nozzle_tap_z_deflection_um(tri_up, tap_force_n=6.0)
    assert tap_tri_up["total_center_tap_deflection_um"] < tap_tri["total_center_tap_deflection_um"]
    assert tap_tri_up["reversal_backlash_um"] == pytest.approx(8.0)

    # Flying-gantry Voron 2.4r2 adds 4-corner Z-belt reaction compliance (~7 um @ 6 N)
    v24_ldo = PrinterSpec.from_yaml(EXAMPLES_DIR / "ldo_voron_24_revd_300.yaml")
    tap_v24_6 = compute_nozzle_tap_z_deflection_um(v24_ldo, tap_force_n=6.0)
    assert 5.0 <= tap_v24_6["total_center_tap_deflection_um"] <= 14.0
    assert tap_v24_6["gantry_z_drive_elastic_um"] > 4.0
    assert tap_v24_6["total_center_tap_deflection_um"] > tap_tri["total_center_tap_deflection_um"] * 2.0

    v24_9 = v24_ldo.model_copy(deep=True)
    v24_9.bed.z_leadscrew_type = "gt2_9mm_z_belt"
    tap_v24_9 = compute_nozzle_tap_z_deflection_um(v24_9, tap_force_n=6.0)
    assert tap_v24_9["gantry_z_drive_elastic_um"] < tap_v24_6["gantry_z_drive_elastic_um"] * 0.75


# ---------------------------------------------------------------------------
# CAL-14: Clip-In Enclosure Panels — EPDM Foam Gasket & ADXL Panels-On vs Off
# ---------------------------------------------------------------------------


def test_cal14_epdm_foam_gasket_physics_and_mitc4_bay_racking() -> None:
    """ASTM D1056 EPDM foam + twist-clip model matches analytical PMMA drum & mild bay racking."""
    g_1515 = clip_gasket_stiffness_per_m(frame_width_m=0.015, chamber_temp_c=25.0)
    g_2020 = clip_gasket_stiffness_per_m(frame_width_m=0.020, chamber_temp_c=25.0)
    g_3030 = clip_gasket_stiffness_per_m(frame_width_m=0.030, chamber_temp_c=25.0)
    g_hot = clip_gasket_stiffness_per_m(frame_width_m=0.020, chamber_temp_c=60.0)

    # 10 mm foam tape caps contact width on 2020/3030; 1515 uses narrower 7.5 mm strip
    assert g_1515["k_shear_per_m"] < g_2020["k_shear_per_m"]
    assert g_2020["k_shear_per_m"] == pytest.approx(g_3030["k_shear_per_m"], rel=1e-9)
    assert g_hot["k_shear_per_m"] < g_2020["k_shear_per_m"]

    # Single 460x480x3 mm cast PMMA bay on EPDM foam tape: drum mode matches Navier SS (~21.8 Hz)
    pmma = get_material("acrylic_pmma")
    bw, bh, t = 0.46, 0.48, 0.003
    d_plate = pmma.youngs_modulus_pa * (t**3) / (12.0 * (1.0 - pmma.poisson_ratio**2))
    areal = pmma.density_kg_m3 * t
    f11_navier = 0.5 * math.pi * math.sqrt(d_plate / areal) * (1.0 / (bw**2) + 1.0 / (bh**2))

    k_rack_clip, f11_clip = mitc4_panel_bay_racking_stiffness(
        bay_w_m=bw,
        bay_h_m=bh,
        e_pa=pmma.youngs_modulus_pa,
        nu=pmma.poisson_ratio,
        thickness_m=t,
        areal_mass_kg_m2=areal,
        k_shear_per_m=g_2020["k_shear_per_m"],
        k_normal_per_m=g_2020["k_normal_per_m"],
        k_rot_per_m=g_2020["k_rot_per_m"],
    )
    assert 50_000.0 <= k_rack_clip <= 120_000.0
    assert abs(f11_clip - f11_navier) / f11_navier < 0.10

    # Bolted structural M5 Huth boundary is >10x stiffer in racking than clip-in EPDM foam tape
    k_rack_bolted, _ = mitc4_panel_bay_racking_stiffness(
        bay_w_m=bw,
        bay_h_m=bh,
        e_pa=pmma.youngs_modulus_pa,
        nu=pmma.poisson_ratio,
        thickness_m=t,
        areal_mass_kg_m2=areal,
        k_shear_per_m=2.5e7,
        k_normal_per_m=5.0e7,
        k_rot_per_m=25.0,
    )
    assert k_rack_bolted > k_rack_clip * 10.0


@pytest.mark.parametrize(
    "preset_name",
    ["ldo_voron_24_revd_300.yaml", "voron_trident_300.yaml"],
)
def test_cal14_voron_clip_panels_on_vs_off_adxl_validation(preset_name: str) -> None:
    """Stock Voron 2.4r2 / Trident clip-in panels give mild ADXL shift (+0.1..+2.0 Hz) and no accel cliff."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / preset_name)

    def _run(enc: bool, mnt: str, mat: str = "acrylic_pmma"):
        s = base.model_copy(deep=True)
        s.frame.enclosed_panels = enc
        s.frame.panel_mounting = mnt  # type: ignore[assignment]
        s.frame.panel_material = mat
        s.frame.panel_thickness_mm = 3.0
        asm = build_assembly(s)
        return solve_static_fea(asm), solve_modal_fea(asm)

    st_off, mo_off = _run(False, "none")
    st_clip, mo_clip = _run(True, "clip_enclosure", "acrylic_pmma")
    st_bolt_pmma, _ = _run(True, "bolted_structural", "acrylic_pmma")
    st_bolt_acm, _ = _run(True, "bolted_structural", "acm_dibond")

    dx_hz = mo_clip.shaper_x.effective_freq_hz - mo_off.shaper_x.effective_freq_hz
    dy_hz = mo_clip.shaper_y.effective_freq_hz - mo_off.shaper_y.effective_freq_hz
    assert 0.0 <= dx_hz <= 2.0
    assert 0.0 <= dy_hz <= 2.5

    # Mild static stiffening (+0.3% to +8%) and zero spurious Y shaper accel drop
    kx_gain_pct = 100.0 * (st_clip.toolhead_stiffness_x_n_per_mm / st_off.toolhead_stiffness_x_n_per_mm - 1.0)
    ky_gain_pct = 100.0 * (st_clip.toolhead_stiffness_y_n_per_mm / st_off.toolhead_stiffness_y_n_per_mm - 1.0)
    assert 0.3 <= kx_gain_pct <= 8.0
    assert 0.3 <= ky_gain_pct <= 8.0
    assert mo_clip.shaper_y.max_accel_mm_s2 >= mo_off.shaper_y.max_accel_mm_s2 * 0.95
    assert 15.0 <= mo_clip.panel_drum_freq_hz <= 26.0

    # Strict stiffness hierarchy: Open < Clip PMMA < Bolted PMMA < Bolted ACM
    assert (
        st_off.toolhead_stiffness_x_n_per_mm
        < st_clip.toolhead_stiffness_x_n_per_mm
        < st_bolt_pmma.toolhead_stiffness_x_n_per_mm
        < st_bolt_acm.toolhead_stiffness_x_n_per_mm
    )
    assert (
        st_off.toolhead_stiffness_y_n_per_mm
        < st_clip.toolhead_stiffness_y_n_per_mm
        < st_bolt_pmma.toolhead_stiffness_y_n_per_mm
        < st_bolt_acm.toolhead_stiffness_y_n_per_mm
    )
