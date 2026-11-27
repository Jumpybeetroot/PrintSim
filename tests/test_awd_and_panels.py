"""Physics verification tests for CoreXY AWD, high-tension belts, and structural shear panels."""

from __future__ import annotations

import math

import pytest

from printsim.engine import evaluate_printer
from printsim.fea import solve_modal_fea, solve_static_fea
from printsim.geometry import build_assembly, generate_bom
from printsim.geometry.builder import panel_mechanical_properties
from printsim.kinematics import analyze_kinematics
from printsim.schema import PrinterSpec, get_material


def _panel_corexy_spec(
    enclosed: bool,
    mounting: str = "bolted_structural",
    material: str = "acm_dibond",
    thickness_mm: float = 3.0,
) -> PrinterSpec:
    spec = PrinterSpec(architecture="corexy")
    spec.frame.bracing = "none"
    spec.frame.enclosed_panels = enclosed
    spec.frame.panel_mounting = mounting  # type: ignore[assignment]
    spec.frame.panel_material = material
    spec.frame.panel_thickness_mm = thickness_mm
    return spec


def test_equivalent_shear_panel_formula_and_geometry_dict() -> None:
    """Verify CoreXY MITC4 shell grids, Huth CBUSH springs, zero legacy X-struts, and metadata."""
    spec = _panel_corexy_spec(True, "bolted_structural", "aluminum_5052_sheet", 4.0)
    spec.gantry.drive_mode = "awd"
    spec.gantry.double_shear_pulleys = True

    asm = build_assembly(spec)
    pmat = get_material("aluminum_5052_sheet")

    assert not any(m.id.startswith("panel_shear_") for m in asm.members)
    assert len(asm.shell_elements) == 16  # 4 bays * 2x2 quads
    panel_springs = [s for s in asm.fastener_springs if s.bay in ("left", "right", "rear", "top")]
    assert len(panel_springs) == 32  # 4 bays * 8 perimeter CBUSH fasteners
    rear_shell = next(s for s in asm.shell_elements if s.bay == "rear")
    assert rear_shell.material.id == "aluminum_5052_sheet"
    assert rear_shell.thickness_m == pytest.approx(0.004, rel=1e-6)

    from printsim.geometry.builder import corexy_frame_centerlines_m

    b, d, h = corexy_frame_centerlines_m(spec)
    t = spec.frame.panel_thickness_mm * 1e-3
    props = panel_mechanical_properties(pmat, t, b, h, "bolted_structural")

    expected_panel_mass = (b * h + 2.0 * d * h + b * d) * props["areal_mass_kg_m2"]
    gdict = asm.to_geometry_dict()
    assert gdict["drive_mode"] == "awd"
    assert gdict["double_shear_pulleys"] is True
    assert gdict["panel_mesh_n"] == 2
    assert len(gdict["shell_panels"]) == 16
    expected_panels = {
        "enabled": True,
        "top_panel": True,
        "mounting": "bolted_structural",
        "material_id": "aluminum_5052_sheet",
        "material_name": pmat.name,
        "side_material_id": "aluminum_5052_sheet",
        "side_material_name": pmat.name,
        "thickness_mm": 4.0,
        "color_hex": pmat.color_hex,
        "bolted_to_gantry": False,
        "is_flying_gantry": True,
        "effective_shear_gt_kn_mm": round(props["gt_eff_n_m"] * 1e-6, 3),
        "buckling_factor": round(props["psi_buckle"], 3),
        "areal_mass_kg_m2": round(props["areal_mass_kg_m2"], 2),
        "total_panel_mass_kg": round(expected_panel_mass, 3),
    }
    for k, v in expected_panels.items():
        assert gdict["panels"][k] == v
    assert set(gdict["panels"]["faces"].keys()) == {"left", "right", "rear", "top"}


def test_structural_panel_monotonicity_frequency_stiffness_and_mass() -> None:
    """Verify monotonic stiffness open < clip < bolted acrylic < bolted 3mm Dibond < 6mm Dibond < 4mm Al5052."""
    configs = [
        ("open", _panel_corexy_spec(False, "none", "acm_dibond", 3.0)),
        ("clip_3mm_acrylic", _panel_corexy_spec(True, "clip_enclosure", "acrylic_pmma", 3.0)),
        ("bolt_3mm_acrylic", _panel_corexy_spec(True, "bolted_structural", "acrylic_pmma", 3.0)),
        ("bolt_3mm_dibond", _panel_corexy_spec(True, "bolted_structural", "acm_dibond", 3.0)),
        ("bolt_6mm_dibond", _panel_corexy_spec(True, "bolted_structural", "acm_dibond", 6.0)),
        ("bolt_4mm_al5052", _panel_corexy_spec(True, "bolted_structural", "aluminum_5052_sheet", 4.0)),
    ]

    results = []
    for label, spec in configs:
        asm = build_assembly(spec)
        st = solve_static_fea(asm)
        mo = solve_modal_fea(asm)
        results.append(
            (
                label,
                mo.fundamental_freq_hz,
                st.toolhead_stiffness_x_n_per_mm,
                st.toolhead_stiffness_y_n_per_mm,
                st.combined_max_deflection_um,
                asm.total_mass_kg,
                st.accel_deflection_x_um,
                st.accel_deflection_y_um,
            )
        )

    for i in range(len(results) - 1):
        curr_label, _, kx_c, ky_c, _, _, _, _ = results[i]
        next_label, _, kx_n, ky_n, _, _, _, _ = results[i + 1]
        assert kx_n > kx_c, f"Expected kx({next_label})={kx_n:.2f} > kx({curr_label})={kx_c:.2f}"
        assert ky_n > ky_c, f"Expected ky({next_label})={ky_n:.2f} > ky({curr_label})={ky_c:.2f}"

    # Clip-in acrylic adds only realistic mild stiffening (< 25% deflection drop vs open)
    _, f_open, kx_open, _, defl_open, _, ax_open, ay_open = results[0]
    _, f_clip, kx_clip, _, defl_clip, _, _, _ = results[1]
    _, f_bolt_acr, _, _, defl_bolt_acr, _, ax_bolt_acr, ay_bolt_acr = results[2]
    _, f_bolt_3acm, _, _, _, m_3, _, _ = results[3]
    _, f_bolt_6acm, _, _, _, m_6, _, _ = results[4]
    _, f_bolt_4al, _, _, _, _, _, _ = results[5]

    assert kx_clip < kx_open * 1.25
    assert (defl_open - defl_clip) / defl_open < 0.25
    assert f_clip < f_open * 1.15  # no artificial 2x jump from foam-tape/twist-clip enclosure

    # Bolted structural panels outperform open & clip-in frames; solid Al beats ACM beats acrylic
    # (note: flying-gantry Z-belt gravity sag ~66 um is constant across panel configs, while dynamic XY accel drops >40-60%,
    #  and once bolted ACM stiffens the outer frame, f1 reaches the ~82 Hz flying-gantry ceiling)
    assert f_bolt_acr > f_open * 1.45
    assert ax_bolt_acr < ax_open * 0.68
    assert ay_bolt_acr < ay_open * 0.65
    assert defl_bolt_acr < defl_open * 0.94
    assert f_bolt_3acm > f_bolt_acr > f_clip
    assert f_bolt_6acm > f_bolt_acr and f_bolt_6acm > f_open * 2.0
    assert f_bolt_4al > f_bolt_3acm and f_bolt_4al > f_bolt_6acm and f_bolt_4al > f_open * 2.15

    # Thickness increase (3mm -> 6mm Dibond) adds 3mm LDPE core mass across the 4 centerline bays
    expected_core_delta = (2.0 * 0.44 * 0.48 + 0.44 * 0.48 + 0.44 * 0.44) * (0.003 * 920.0)
    assert (m_6 - m_3) == pytest.approx(expected_core_delta, rel=1e-3)
    assert m_6 > m_3 + 2.2

    # Even in clip_enclosure mode, thickness (3mm -> 6mm) and heavier material (PC -> Al 5052) increase mass
    clip_pc_3 = build_assembly(_panel_corexy_spec(True, "clip_enclosure", "polycarbonate", 3.0))
    clip_pc_6 = build_assembly(_panel_corexy_spec(True, "clip_enclosure", "polycarbonate", 6.0))
    clip_al_3 = build_assembly(_panel_corexy_spec(True, "clip_enclosure", "aluminum_5052_sheet", 3.0))
    assert clip_pc_6.total_mass_kg > clip_pc_3.total_mass_kg + 2.0
    assert clip_al_3.total_mass_kg > clip_pc_3.total_mass_kg + 2.5

    # panel_mounting == "none" adds no shear struts even if enclosed_panels=True
    none_mount = build_assembly(_panel_corexy_spec(True, "none", "aluminum_5052_sheet", 4.0))
    open_asm = build_assembly(configs[0][1])
    assert not any(m.id.startswith("panel_shear_") for m in none_mount.members)
    assert none_mount.to_geometry_dict()["panels"]["enabled"] is False
    assert solve_modal_fea(none_mount).fundamental_freq_hz == pytest.approx(
        solve_modal_fea(open_asm).fundamental_freq_hz, rel=1e-6
    )


def test_delta_outer_bay_shear_panels_and_bom_exclusion() -> None:
    """Delta enclosed outer bays add shear stiffness and exclude virtual struts from BOM."""
    spec_open = PrinterSpec(architecture="delta")
    spec_open.kinematics.build_volume_mm = (200.0, 200.0, 250.0)
    spec_open.kinematics.delta_arm_length_mm = 300.0
    spec_open.kinematics.delta_radius_mm = 155.0
    spec_open.frame.bracing = "none"
    spec_open.frame.enclosed_panels = False

    spec_clip = spec_open.model_copy(deep=True)
    spec_clip.frame.enclosed_panels = True
    spec_clip.frame.panel_mounting = "clip_enclosure"
    spec_clip.frame.panel_material = "acrylic_pmma"

    spec_bolt = spec_clip.model_copy(deep=True)
    spec_bolt.frame.panel_mounting = "bolted_structural"
    spec_bolt.frame.panel_material = "acm_dibond"

    asm_open = build_assembly(spec_open)
    asm_clip = build_assembly(spec_clip)
    asm_bolt = build_assembly(spec_bolt)

    assert len([m for m in asm_bolt.members if m.id.startswith("panel_shear_")]) == 6
    assert solve_modal_fea(asm_bolt).fundamental_freq_hz > solve_modal_fea(asm_open).fundamental_freq_hz
    assert solve_modal_fea(asm_bolt).fundamental_freq_hz > solve_modal_fea(asm_clip).fundamental_freq_hz
    assert (
        solve_static_fea(asm_bolt).toolhead_stiffness_x_n_per_mm
        > solve_static_fea(asm_clip).toolhead_stiffness_x_n_per_mm
        > solve_static_fea(asm_open).toolhead_stiffness_x_n_per_mm
    )

    bom = generate_bom(spec_bolt)
    assert all(c.role not in ("brace", "toolhead") for c in bom.cut_list)


def test_belt_induced_frame_preload_and_panel_resistance() -> None:
    """High belt tension deforms unbraced 2020 frames; gravity sag stays invariant; panels/3030 resist."""
    spec_25 = _panel_corexy_spec(False, "none")
    spec_25.gantry.belt.pretension_n = 25.0

    spec_90 = spec_25.model_copy(deep=True)
    spec_90.gantry.belt.pretension_n = 90.0

    spec_90_bolted = spec_90.model_copy(deep=True)
    spec_90_bolted.frame.enclosed_panels = True
    spec_90_bolted.frame.panel_mounting = "bolted_structural"
    spec_90_bolted.frame.panel_material = "acm_dibond"

    spec_90_3030 = spec_90.model_copy(deep=True)
    spec_90_3030.frame.profile = "extrusion_3030"

    fea_25 = solve_static_fea(build_assembly(spec_25))
    fea_90 = solve_static_fea(build_assembly(spec_90))
    fea_90_bolted = solve_static_fea(build_assembly(spec_90_bolted))
    fea_90_3030 = solve_static_fea(build_assembly(spec_90_3030))

    assert fea_90.gravity_sag_toolhead_um == pytest.approx(fea_25.gravity_sag_toolhead_um, rel=1e-6)
    # Self-equilibrating belt preload (sum F = 0) raises gantry axial compression & stress without fake net lateral push
    rear_bar_25 = next(m for m in fea_25.member_stresses if m.member_id == "gantry_rear_bar")
    rear_bar_90 = next(m for m in fea_90.member_stresses if m.member_id == "gantry_rear_bar")
    assert abs(rear_bar_90.axial_force_n) > abs(rear_bar_25.axial_force_n) * 2.5
    assert fea_90.max_von_mises_mpa > fea_25.max_von_mises_mpa
    assert fea_90_bolted.accel_deflection_x_um < fea_90.accel_deflection_x_um * 0.64
    assert fea_90_bolted.accel_deflection_y_um < fea_90.accel_deflection_y_um * 0.61
    assert fea_90_bolted.combined_max_deflection_um < fea_90.combined_max_deflection_um * 0.93
    assert fea_90_3030.accel_deflection_x_um < fea_90.accel_deflection_x_um * 0.58
    # Castigliano w_end=0.49 Z-rail stiffening + 3D CAD ABS XY-joint floor keeps 3030 Y accel deflection >30% stiffer than 2020
    assert fea_90_3030.accel_deflection_y_um < fea_90.accel_deflection_y_um * 0.70
    assert fea_90_3030.combined_max_deflection_um < fea_90.combined_max_deflection_um
    assert fea_90_3030.max_von_mises_mpa < fea_90.max_von_mises_mpa


def test_corexy_awd_vs_2wd_at_high_accel() -> None:
    """Compare identical CoreXY at 20,000 mm/s^2 in 2WD vs AWD."""
    spec_2wd = _panel_corexy_spec(True, "bolted_structural", "acm_dibond", 4.0)
    spec_2wd.kinematics.max_accel_mm_s2 = 20000.0
    spec_2wd.gantry.belt.type = "gt2_9mm"
    spec_2wd.gantry.belt.core = "aramid"
    spec_2wd.gantry.belt.pretension_n = 55.0
    spec_2wd.gantry.double_shear_pulleys = True
    spec_2wd.gantry.drive_mode = "2wd"

    spec_awd = spec_2wd.model_copy(deep=True)
    spec_awd.gantry.drive_mode = "awd"

    asm_2wd = build_assembly(spec_2wd)
    asm_awd = build_assembly(spec_awd)

    # Front corners carry front XY motor + mount mass in AWD (either directly on g_* or cantilevered on pulley_*)
    nmap_awd = asm_awd.node_map
    expected_front_kg = spec_awd.kinematics.xy_stepper.motor_mass_g * 1e-3 + 0.03
    for tag in ("fl", "fr"):
        nid = f"g_{tag}"
        pid = f"pulley_{tag}"
        assert nmap_awd[nid].role == "mount"
        corner_mass = nmap_awd[nid].lumped_mass_kg + (
            nmap_awd[pid].lumped_mass_kg if pid in nmap_awd else 0.0
        )
        assert corner_mass == pytest.approx(expected_front_kg, rel=1e-6)
    assert asm_awd.total_mass_kg > asm_2wd.total_mass_kg + 0.6

    kin_2wd = analyze_kinematics(asm_2wd)
    kin_awd = analyze_kinematics(asm_awd)
    mod_2wd = solve_modal_fea(asm_2wd)
    mod_awd = solve_modal_fea(asm_awd)

    assert kin_2wd.xy_motor_count == 2
    assert kin_awd.xy_motor_count == 4
    assert kin_awd.drive_mode == "awd"
    assert kin_awd.frame_belt_preload_n == pytest.approx(2.0 * kin_2wd.frame_belt_preload_n, rel=1e-6)
    assert kin_awd.torque_safety_margin > kin_2wd.torque_safety_margin * 1.45

    # Dynamic belt tension swing (peak - pretension): in the perimeter CoreXY routing both layouts
    # have equal belt branches at bed centre (50 / 50 split), so the per-belt swing is about the same.
    swing_2wd = kin_2wd.belt_peak_tension_n - spec_2wd.gantry.belt.pretension_n
    swing_awd = kin_awd.belt_peak_tension_n - spec_awd.gantry.belt.pretension_n
    assert swing_awd == pytest.approx(swing_2wd, rel=0.25)

    assert mod_awd.shaper_x is not None and mod_2wd.shaper_x is not None
    assert mod_awd.shaper_y is not None and mod_2wd.shaper_y is not None
    # By spring superposition, splitting the 2WD 3L:L loop into L:L AWD boosts center cord stiffness
    # by ~1.5x (<= sqrt(1.5) ~ 1.22x cord frequency gain before tooth mesh & idler series compliance).
    assert mod_awd.shaper_x.belt_freq_hz > mod_2wd.shaper_x.belt_freq_hz * 1.15
    assert mod_awd.shaper_y.belt_freq_hz > mod_2wd.shaper_y.belt_freq_hz * 1.15
    # On 2020 + 4mm ACM with calibrated 9mm aramid belts, AWD raises max accel +35% over 2WD (20,500 vs 15,100)
    assert mod_awd.recommended_max_accel_mm_s2 > mod_2wd.recommended_max_accel_mm_s2 * 1.30


def test_high_tension_double_shear_and_pluck_frequency() -> None:
    """Verify overtension/shaft-bending flags, diagnostics, dynamics score, and 150mm pluck freq."""
    spec_risky = _panel_corexy_spec(True, "bolted_structural", "acm_dibond", 4.0)
    spec_risky.gantry.belt.type = "gt2_6mm"
    spec_risky.gantry.belt.core = "fiberglass"
    spec_risky.gantry.belt.pretension_n = 70.0
    spec_risky.gantry.double_shear_pulleys = False

    rep_risky = evaluate_printer(spec_risky)
    assert rep_risky.kinematics.shaft_bending_risk is True
    assert rep_risky.kinematics.belt_overtension_risk is True
    joined_diag = " ".join(rep_risky.diagnostics).lower()
    assert "overtension" in joined_diag
    assert "double-shear" in joined_diag

    # Enabling double-shear clears shaft bending risk and raises dynamics score
    spec_ds = spec_risky.model_copy(deep=True)
    spec_ds.gantry.double_shear_pulleys = True
    rep_ds = evaluate_printer(spec_ds)
    assert rep_ds.kinematics.shaft_bending_risk is False
    assert rep_ds.kinematics.belt_overtension_risk is True
    assert rep_ds.scores.dynamics > rep_risky.scores.dynamics

    # Upgrading to 9mm Aramid + double-shear clears both flags and boosts dynamics further
    spec_safe = spec_ds.model_copy(deep=True)
    spec_safe.gantry.belt.type = "gt2_9mm"
    spec_safe.gantry.belt.core = "aramid"
    rep_safe = evaluate_printer(spec_safe)
    assert rep_safe.kinematics.shaft_bending_risk is False
    assert rep_safe.kinematics.belt_overtension_risk is False
    assert rep_safe.scores.dynamics > rep_ds.scores.dynamics

    # Exact transverse belt pluck frequency check: f = (1 / 2L) * sqrt(T / mu) within < 0.1%
    mu = spec_safe.gantry.belt.linear_mass_kg_m
    expected_pluck = (1.0 / (2.0 * 0.150)) * math.sqrt(70.0 / mu)
    assert rep_safe.kinematics.belt_pluck_freq_150mm_hz == pytest.approx(expected_pluck, rel=1e-3)
    assert rep_safe.kinematics.to_dict()["belt_pluck_freq_150mm_hz"] == pytest.approx(expected_pluck, rel=1e-3)

    # Belt pretension stiffness coupling increases modal belt_freq_hz from 15 N -> 30 N -> 45 N
    mod_15 = solve_modal_fea(build_assembly(_with_pretension(spec_safe, 15.0)))
    mod_30 = solve_modal_fea(build_assembly(_with_pretension(spec_safe, 30.0)))
    mod_45 = solve_modal_fea(build_assembly(_with_pretension(spec_safe, 45.0)))
    assert mod_15.shaper_x is not None and mod_30.shaper_x is not None and mod_45.shaper_x is not None
    assert mod_15.shaper_x.belt_freq_hz < mod_30.shaper_x.belt_freq_hz < mod_45.shaper_x.belt_freq_hz


def _with_pretension(spec: PrinterSpec, pretension_n: float) -> PrinterSpec:
    out = spec.model_copy(deep=True)
    out.gantry.belt.pretension_n = pretension_n
    return out


def test_dibond_tds_sandwich_mechanics_and_timoshenko_buckling() -> None:
    """Verify 3/4/6 mm ACM Dibond areal mass & flexural rigidity match 3A Composites TDS."""
    dibond = get_material("acm_dibond")
    tds_targets = [
        # (thickness_mm, tds_areal_mass_kg_m2, tds_flexural_d_nm)
        (3.0, 3.80, 86.5),
        (4.0, 4.75, 162.0),
        (6.0, 6.60, 384.0),
    ]
    for t_mm, target_mass, target_d in tds_targets:
        props = panel_mechanical_properties(dibond, t_mm * 1e-3, 0.46, 0.50, "bolted_structural")
        assert props["areal_mass_kg_m2"] == pytest.approx(target_mass, rel=0.02)
        # Timoshenko core-shear knockdown keeps effective D slightly below pure bending TDS D
        assert 0.80 * target_d <= props["flexural_d_nm"] <= 1.05 * target_d
        assert 0.22 <= props["psi_buckle"] <= 1.0
        # Series stiffness identity holds identically with Huth bolted joint stiffness
        expected_k_eff = (props["k_sheet_n_m"] * props["k_mount_n_m"]) / (
            props["k_sheet_n_m"] + props["k_mount_n_m"]
        )
        assert props["k_panel_eff_n_m"] == pytest.approx(expected_k_eff, rel=1e-9)
        expected_eta = props["k_panel_eff_n_m"] / props["k_sheet_n_m"]
        assert props["eta_mount"] == pytest.approx(expected_eta, rel=1e-9)
        props_gantry = panel_mechanical_properties(
            dibond, t_mm * 1e-3, 0.46, 0.50, "bolted_structural", bolted_to_gantry=True
        )
        assert props_gantry["k_mount_n_m"] == pytest.approx(props["k_mount_n_m"] * 1.35, rel=1e-6)
        assert props_gantry["gt_eff_n_m"] > props["gt_eff_n_m"]

    # Timoshenko buckling factor & Huth joint stiffness: solid Al 5052 far stiffer than ACM & acrylic
    acrylic = get_material("acrylic_pmma")
    al5052 = get_material("aluminum_5052_sheet")
    p_2mm_acrylic = panel_mechanical_properties(acrylic, 0.002, 0.46, 0.50, "bolted_structural")
    p_3mm_acrylic = panel_mechanical_properties(acrylic, 0.003, 0.46, 0.50, "bolted_structural")
    p_15mm_al = panel_mechanical_properties(al5052, 0.0015, 0.46, 0.50, "bolted_structural")
    p_4mm_al = panel_mechanical_properties(al5052, 0.004, 0.46, 0.50, "bolted_structural")
    p_5mm_al = panel_mechanical_properties(al5052, 0.005, 0.46, 0.50, "bolted_structural")
    p_3mm_acm = panel_mechanical_properties(dibond, 0.003, 0.46, 0.50, "bolted_structural")
    p_3mm_acrylic_gantry = panel_mechanical_properties(acrylic, 0.003, 0.46, 0.445, "bolted_structural")

    assert 0.22 < p_2mm_acrylic["psi_buckle"] < p_3mm_acrylic["psi_buckle"] < p_15mm_al["psi_buckle"] < p_4mm_al["psi_buckle"] < p_5mm_al["psi_buckle"] <= 1.0
    assert p_5mm_al["k_panel_eff_n_m"] > p_4mm_al["k_panel_eff_n_m"] > p_15mm_al["k_panel_eff_n_m"] > p_3mm_acm["k_panel_eff_n_m"] > p_3mm_acrylic["k_panel_eff_n_m"]
    assert p_4mm_al["k_mount_n_m"] > p_3mm_acm["k_mount_n_m"] * 5.0
    assert p_3mm_acrylic_gantry["psi_buckle"] > p_3mm_acrylic["psi_buckle"]
    clip_props = panel_mechanical_properties(acrylic, 0.003, 0.46, 0.50, "clip_enclosure")
    assert clip_props["k_mount_n_m"] == pytest.approx(220_000.0, rel=1e-9)
    assert clip_props["k_panel_eff_n_m"] < 220_000.0
    assert panel_mechanical_properties(acrylic, 0.003, 0.46, 0.50, "none")["gt_eff_n_m"] == 0.0


def test_thick_aluminum_beats_thin_aluminum_and_acm_without_front_gantry_bar() -> None:
    """Standard Voron Trident / Monolith has no front gantry bar, and 4-5mm Al beats 1.5-2mm Al & ACM."""
    base = PrinterSpec.from_yaml("examples/voron_monolith_awd.yaml")

    def _run(mat: str, thk_mm: float):
        s = base.model_copy(deep=True)
        s.frame.enclosed_panels = True
        s.frame.panel_mounting = "bolted_structural"
        s.frame.panel_material = mat
        s.frame.panel_thickness_mm = thk_mm
        asm = build_assembly(s)
        return asm, solve_static_fea(asm), solve_modal_fea(asm)

    asm_acm3, st_acm3, mo_acm3 = _run("acm_dibond", 3.0)
    _, st_acm4, mo_acm4 = _run("acm_dibond", 4.0)
    _, st_al15, mo_al15 = _run("aluminum_5052_sheet", 1.5)
    _, st_al20, mo_al20 = _run("aluminum_5052_sheet", 2.0)
    _, st_al40, mo_al40 = _run("aluminum_5052_sheet", 4.0)
    _, st_al50, mo_al50 = _run("aluminum_5052_sheet", 5.0)

    # Standard Voron Trident / Monolith CoreXY has an open front door at gantry height (no gantry_front_bar)
    assert all(m.id != "gantry_front_bar" for m in asm_acm3.members)

    # 4mm & 5mm solid Al 5052 strictly beat 3mm & 4mm ACM on frequency, safe accel, and stiffness
    assert mo_al40.fundamental_freq_hz > mo_acm3.fundamental_freq_hz + 35.0
    assert mo_al50.fundamental_freq_hz > mo_acm4.fundamental_freq_hz + 40.0
    # 4mm solid Al reaches the 9mm aramid AWD belt ceiling (D-01 datasheet EA), while structural
    # frequency and nozzle stiffness strictly beat 3mm ACM
    assert mo_al40.recommended_max_accel_mm_s2 >= mo_acm3.recommended_max_accel_mm_s2
    assert mo_al40.shaper_x.structural_freq_hz > mo_acm3.shaper_x.structural_freq_hz * 1.10
    assert st_al40.toolhead_stiffness_x_n_per_mm > st_acm3.toolhead_stiffness_x_n_per_mm
    assert st_al50.accel_deflection_x_um < st_acm3.accel_deflection_x_um
    assert st_al50.accel_deflection_y_um < st_acm3.accel_deflection_y_um

    # 4mm & 5mm solid Al strictly beat flimsy 1.5mm & 2mm Al sheets (buckling + composite flange action;
    # 4mm Al panel drum crosses near the ~128 Hz Archetype CAD toolhead mode causing minor Mode-1 veering,
    # while structural X/Y frequencies and static stiffness stay strictly monotonic 1.5 < 2.0 < 4.0 < 5.0 mm)
    assert mo_al40.fundamental_freq_hz > mo_al15.fundamental_freq_hz
    assert mo_al50.fundamental_freq_hz > mo_al20.fundamental_freq_hz > mo_al15.fundamental_freq_hz
    assert mo_al50.shaper_x.structural_freq_hz > mo_al40.shaper_x.structural_freq_hz > mo_al20.shaper_x.structural_freq_hz > mo_al15.shaper_x.structural_freq_hz
    assert mo_al40.shaper_y.structural_freq_hz > mo_al20.shaper_y.structural_freq_hz > mo_al15.shaper_y.structural_freq_hz
    assert st_al50.toolhead_stiffness_x_n_per_mm > st_al40.toolhead_stiffness_x_n_per_mm > st_al20.toolhead_stiffness_x_n_per_mm > st_al15.toolhead_stiffness_x_n_per_mm

    # Even without gantry_front_bar, true CoreXY belt routing keeps front corner flex tiny (< 8 um w/ CAL-09 Tet10 joints)
    gfl_um = (sum(v * v for v in st_al40.nodal_displacements_mm["g_fl"]) ** 0.5) * 1000.0
    assert gfl_um < 8.0


@pytest.mark.parametrize("z_drive,bolt_to_gantry", [
    ("leadscrew_triple", True),
    ("leadscrew_triple", False),
    ("belted_quad_z", True),
])
@pytest.mark.parametrize("material,thickness_mm", [
    ("acm_dibond", 3.0),
    ("acm_dibond", 6.0),
    ("aluminum_5052_sheet", 4.0),
])
def test_zero_phantom_strut_mass_on_corexy(
    z_drive: str,
    bolt_to_gantry: bool,
    material: str,
    thickness_mm: float,
) -> None:
    """Equivalent X-struts add zero phantom mass even on skinny 460x55mm upper gantry strips."""
    spec_open = _panel_corexy_spec(False, "none", material, thickness_mm)
    spec_open.bed.z_drive = z_drive  # type: ignore[assignment]
    spec_open.frame.bolt_panels_to_gantry = bolt_to_gantry

    spec_enc = spec_open.model_copy(deep=True)
    spec_enc.frame.enclosed_panels = True
    spec_enc.frame.panel_mounting = "bolted_structural"

    asm_open = build_assembly(spec_open)
    asm_enc = build_assembly(spec_enc)

    from printsim.geometry.builder import corexy_frame_centerlines_m

    w, d, h = corexy_frame_centerlines_m(spec_enc)
    pmat = get_material(material)
    areal_mass = panel_mechanical_properties(pmat, thickness_mm * 1e-3, w, h, "bolted_structural")[
        "areal_mass_kg_m2"
    ]
    expected_bay_mass_kg = (w * h + 2.0 * d * h + w * d) * areal_mass

    actual_delta_kg = asm_enc.total_mass_kg - asm_open.total_mass_kg
    assert actual_delta_kg == pytest.approx(expected_bay_mass_kg, rel=5e-3)
    assert asm_enc.panel_mass_kg == pytest.approx(expected_bay_mass_kg, rel=1e-6)
    assert abs(asm_enc.frame_mass_kg - asm_open.frame_mass_kg) < 1e-4


def test_fixed_gantry_vs_flying_gantry_panel_bolting() -> None:
    """Fixed-gantry Trident bolts panels directly to gantry beams; flying-gantry cannot."""
    spec_trident_bolt = _panel_corexy_spec(True, "bolted_structural", "acm_dibond", 4.0)
    spec_trident_bolt.bed.z_drive = "leadscrew_triple"
    spec_trident_bolt.frame.bolt_panels_to_gantry = True

    spec_trident_nobolt = spec_trident_bolt.model_copy(deep=True)
    spec_trident_nobolt.frame.bolt_panels_to_gantry = False

    spec_flying = spec_trident_bolt.model_copy(deep=True)
    spec_flying.bed.z_drive = "belted_quad_z"

    assert spec_trident_bolt.is_flying_gantry is False
    assert spec_trident_bolt.panels_bolted_to_gantry_effective is True
    assert spec_trident_nobolt.panels_bolted_to_gantry_effective is False
    assert spec_flying.is_flying_gantry is True
    assert spec_flying.panels_bolted_to_gantry_effective is False

    asm_bolt = build_assembly(spec_trident_bolt)
    asm_nobolt = build_assembly(spec_trident_nobolt)
    asm_flying = build_assembly(spec_flying)

    # Split gantry panel bays + Y-rail CBUSH fasteners exist ONLY when bolted_to_gantry is effective
    bolt_bays = {s.bay for s in asm_bolt.shell_elements}
    nobolt_bays = {s.bay for s in asm_nobolt.shell_elements}
    flying_bays = {s.bay for s in asm_flying.shell_elements}

    assert {"rear_low", "rear_top", "left_low", "left_top", "right_low", "right_top", "top"} == bolt_bays
    assert {"rear", "left", "right", "top"} == nobolt_bays
    assert {"rear", "left", "right", "top"} == flying_bays
    assert any(sp.id.startswith("cbush_") and sp.frame_node_id in ("y_rail_car_l", "y_rail_car_r") for sp in asm_bolt.fastener_springs)
    assert not any(sp.id.startswith("cbush_") and sp.frame_node_id in ("y_rail_car_l", "y_rail_car_r", "y_car_l", "y_car_r") for sp in asm_nobolt.fastener_springs)
    assert not any(sp.id.startswith("cbush_") and sp.frame_node_id in ("y_rail_car_l", "y_rail_car_r", "y_car_l", "y_car_r") for sp in asm_flying.fastener_springs)

    st_bolt = solve_static_fea(asm_bolt)
    st_nobolt = solve_static_fea(asm_nobolt)
    st_flying = solve_static_fea(asm_flying)

    mo_bolt = solve_modal_fea(asm_bolt)
    mo_nobolt = solve_modal_fea(asm_nobolt)
    mo_flying = solve_modal_fea(asm_flying)

    # Bolting to fixed gantry beams increases X stiffness, reduces deflection, and raises panel drum frequency;
    # for stiff solid metal panels (4mm Al 5052) it also raises f1 & shaper_x (on 4mm ACM, continuous bolting
    # couples the soft LDPE core's 75.5 Hz drum mode into the gantry)
    assert st_bolt.toolhead_stiffness_x_n_per_mm > st_nobolt.toolhead_stiffness_x_n_per_mm
    assert st_bolt.toolhead_stiffness_y_n_per_mm >= st_nobolt.toolhead_stiffness_y_n_per_mm * 0.995
    assert st_bolt.accel_deflection_x_um < st_nobolt.accel_deflection_x_um
    assert st_bolt.accel_deflection_y_um < st_nobolt.accel_deflection_y_um
    assert st_bolt.bed_deflection_um < st_nobolt.bed_deflection_um
    assert mo_bolt.panel_drum_freq_hz > mo_nobolt.panel_drum_freq_hz

    spec_al_bolt = spec_trident_bolt.model_copy(deep=True)
    spec_al_bolt.frame.panel_material = "aluminum_5052_sheet"
    spec_al_nobolt = spec_al_bolt.model_copy(deep=True)
    spec_al_nobolt.frame.bolt_panels_to_gantry = False
    mo_al_bolt = solve_modal_fea(build_assembly(spec_al_bolt))
    mo_al_nobolt = solve_modal_fea(build_assembly(spec_al_nobolt))
    assert mo_al_bolt.shaper_x is not None and mo_al_nobolt.shaper_x is not None
    assert mo_al_bolt.shaper_x.structural_freq_hz > mo_al_nobolt.shaper_x.structural_freq_hz + 5.0

    assert st_bolt.toolhead_stiffness_x_n_per_mm > st_flying.toolhead_stiffness_x_n_per_mm
    # Flying-gantry anchors its bed directly to fixed base feet, whereas Trident's moving Z-bed adds ~0.2 um/N loop compliance
    assert st_bolt.toolhead_stiffness_y_n_per_mm >= st_flying.toolhead_stiffness_y_n_per_mm * 0.96
    assert st_bolt.combined_max_deflection_um < st_flying.combined_max_deflection_um
    assert mo_al_bolt.fundamental_freq_hz > mo_flying.fundamental_freq_hz

    # Unhealthy flying gantry (structural < 80) emits helpful gantry-bolting note
    spec_weak_fly = _panel_corexy_spec(True, "bolted_structural", "acrylic_pmma", 2.0)
    spec_weak_fly.bed.z_drive = "belted_quad_z"
    spec_weak_fly.frame.bolt_panels_to_gantry = True
    spec_weak_fly.kinematics.max_accel_mm_s2 = 22000.0
    rep_weak_fly = evaluate_printer(spec_weak_fly)
    assert rep_weak_fly.scores.structural < 80.0
    assert any("flying-gantry" in d.lower() for d in rep_weak_fly.diagnostics)
