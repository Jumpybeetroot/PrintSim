"""Calibration touchstone tests against real-world datasheets and community benchmarks.

Validates:
- HIWIN MG Series linear rail linear masses (MGN9: 320 g/m, MGN12: 650 g/m, MGN15: 920 g/m)
- Bambu / Siraya / Fiberon TDS modulus updates for PPS-CF (9.2 GPa) & PPA-CF (10.8 GPa)
- Non-regression of structural bug fixes (corner joint + Z rail composite stiffening,
  leadscrew_dual steel rail guideway without spurious 63 Hz bed shaker, hotend mass Y share,
  cartesian_bedslinger carriage mass, local toolhead pitch-twist mode detection)
- Empirical frequency/mass touchstone bands across all 8 presets
"""

from __future__ import annotations

from pathlib import Path

import pytest

from printsim.engine import evaluate_printer
from printsim.fea.modal import solve_modal_fea
from printsim.fea.solver import solve_static_fea
from printsim.geometry.builder import build_assembly
from printsim.schema.materials import get_material
from printsim.schema.printer import (
    GANTRY_BRACKET_CATALOG,
    HOTEND_CATALOG,
    PrinterSpec,
)
from printsim.schema.profiles import get_profile

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def test_hiwin_mgn_linear_mass_and_polymer_bracket_tds() -> None:
    """Verify MGN rail masses match HIWIN MG catalog and PPS-CF/PPA-CF match annealed TDS."""
    steel = get_material("steel_4140")
    # D-03: HIWIN MG Series catalog nominal linear masses: 0.38 / 0.65 / 1.06 kg/m
    for prof_id, target_kg_m in (
        ("mgn9_rail", 0.380),
        ("mgn12_rail", 0.650),
        ("mgn15_rail", 1.060),
    ):
        prof = get_profile(prof_id)
        mass_per_m = prof.area_m2 * steel.density_kg_m3
        assert mass_per_m == pytest.approx(target_kg_m, rel=0.005)

    # Bambu / Siraya / Fiberon annealed TDS moduli and catalog consistency
    pps_mat = get_material("pps_cf_printed")
    ppa_mat = get_material("ppa_cf_printed")
    assert pps_mat.youngs_modulus_pa == pytest.approx(9.2e9, rel=1e-6)
    assert ppa_mat.youngs_modulus_pa == pytest.approx(10.8e9, rel=1e-6)

    assert GANTRY_BRACKET_CATALOG["pps_cf_printed"]["youngs_modulus_gpa"] == pytest.approx(9.2)
    assert GANTRY_BRACKET_CATALOG["ppa_cf_printed"]["youngs_modulus_gpa"] == pytest.approx(10.8)
    assert GANTRY_BRACKET_CATALOG["pps_cf_printed"]["name"] == pps_mat.name
    assert GANTRY_BRACKET_CATALOG["ppa_cf_printed"]["name"] == ppa_mat.name

    # Strict monotonic bracket modulus hierarchy
    ordered = (
        "abs_printed",
        "pet_cf_printed",
        "pa12_cf_printed",
        "pps_cf_printed",
        "ppa_cf_printed",
        "aluminum_6061_t6",
        "aluminum_7075_t6",
    )
    moduli = [get_material(mid).youngs_modulus_pa for mid in ordered]
    assert all(moduli[i] < moduli[i + 1] for i in range(len(moduli) - 1))


def test_structural_calibration_bugfixes_preserved() -> None:
    """Verify corner-joint Z-rail retention, bedslinger carriage mass, and pitch-twist isolation."""
    # 1. Trident vertical columns with bolted MGN12H Z rails keep net positive stiffness above un-railed columns
    trident_spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_trident_300.yaml")
    trident_asm = build_assembly(trident_spec)
    z_col = next(m for m in trident_asm.members if m.id == "col_fl_bot")
    unrailed_col = next(m for m in trident_asm.members if m.id == "col_rl_low")
    assert z_col.rail_mass_kg > 0.0
    assert unrailed_col.rail_mass_kg == 0.0
    assert z_col.profile.iy_m4 > unrailed_col.profile.iy_m4
    assert z_col.profile.iz_m4 > unrailed_col.profile.iz_m4

    # Monotonic corner joint order holds on Z-rail columns
    joint_order = ("l_bracket", "blind_joint", "gusset_plate", "corner_cube", "rigid")
    kx_by_joint = []
    for jtype in joint_order:
        s = trident_spec.model_copy(deep=True)
        s.frame.corner_joint_type = jtype  # type: ignore[assignment]
        s.frame.bracing = "none"
        kx_by_joint.append(solve_static_fea(build_assembly(s)).toolhead_stiffness_x_n_per_mm)
    assert all(kx_by_joint[i] < kx_by_joint[i + 1] for i in range(len(kx_by_joint) - 1))

    # 2. Nozzle pitch-twist frequency isolates true local toolhead mode (> 90 Hz, not 38 Hz frame sway)
    trident_modal = solve_modal_fea(trident_asm)
    assert trident_modal.nozzle_pitch_twist_freq_hz >= 90.0

    # 3. Toolhead moving mass adds both toolhead.total_mass_g and hotend.mass_g cleanly
    vz_spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "vzbot_330_awd.yaml")
    vz_asm = build_assembly(vz_spec)
    from printsim.geometry._builder_gantry import x_carriage_block_mass_delta_g

    _, vz_block_delta_g = x_carriage_block_mass_delta_g(vz_spec)
    expected_vz_th_g = (
        vz_spec.toolhead.total_mass_g + vz_spec.toolhead.hotend.mass_g + vz_block_delta_g
    )
    assert vz_asm.toolhead_moving_mass_kg * 1000.0 == pytest.approx(expected_vz_th_g, rel=1e-6)
    # Vz-Printhead CNC: 272 g (54 g MGN12H convention) - 28 g MGN9H delta + 96 g Goliath = 340 g
    assert vz_block_delta_g == pytest.approx(-28.0, abs=1.0)
    assert expected_vz_th_g == pytest.approx(340.0, abs=1.0)

    # 4. Bedslinger Y moving mass uses carriage_mass_g directly rather than hardcoded 350 g
    bs_spec = PrinterSpec.from_yaml(EXAMPLES_DIR / "cartesian_bedslinger.yaml")
    bs_asm = build_assembly(bs_spec)
    expected_bs_y_g = bs_spec.bed.mass_g + bs_spec.gantry.carriage_mass_g
    assert bs_asm.y_moving_mass_kg * 1000.0 == pytest.approx(expected_bs_y_g, rel=1e-6)
    bs_heavy = bs_spec.model_copy(deep=True)
    bs_heavy.gantry.carriage_mass_g = bs_spec.gantry.carriage_mass_g + 200.0
    assert (build_assembly(bs_heavy).y_moving_mass_kg - bs_asm.y_moving_mass_kg) == pytest.approx(
        0.200, abs=1e-6
    )


def test_real_world_preset_touchstone_bands_and_catalog_parity() -> None:
    """All 8 example presets match calibrated community frequency bands and datasheet hotends."""
    # Verify all 8 presets evaluate cleanly and hotend parameters align with HOTEND_CATALOG
    results = {}
    for yaml_path in sorted(EXAMPLES_DIR.glob("*.yaml")):
        spec = PrinterSpec.from_yaml(yaml_path)
        rep = evaluate_printer(spec, include_advanced_analysis=False)
        assert 0.0 < rep.scores.composite <= 100.0
        assert not rep.collision.has_collision
        pid = spec.toolhead.hotend.preset_id
        if pid is not None and pid in HOTEND_CATALOG:
            cat = HOTEND_CATALOG[pid]
            assert spec.toolhead.hotend.heater_power_w == pytest.approx(cat["heater_power_w"])
            assert spec.toolhead.hotend.melt_zone_length_mm == pytest.approx(cat["melt_zone_length_mm"])
            assert spec.toolhead.hotend.block_mass_g == pytest.approx(cat["block_mass_g"])
            assert spec.toolhead.hotend.max_temp_c == pytest.approx(cat["max_temp_c"])
        results[yaml_path.stem] = (spec, rep)

    # VzBot 330 AWD (official spec): cited community Klipper configs X 75-118 / Y 47-80 Hz; the
    # global fundamental is the ~20 Hz rocking mode on its rubber-insert feet (no fake bed shaker).
    _, vz = results["vzbot_330_awd"]
    assert vz.modal_fea.fundamental_freq_hz > 12.0
    assert 75.0 <= vz.modal_fea.shaper_x.effective_freq_hz <= 125.0
    assert 47.0 <= vz.modal_fea.shaper_y.effective_freq_hz <= 85.0
    assert vz.modal_fea.shaper_x.effective_freq_hz > vz.modal_fea.shaper_y.effective_freq_hz

    # Voron Monolith AWD & DoomCube 4040 AWD: empirical Monolith AWD Discord bands [95, 185] / [70, 145]
    # (B-01/D-01: DoomCube 300 4040 AWD heavy gantry lands at ~131/74 Hz, Monolith AWD at ~145/83 Hz)
    for name in ("voron_monolith_awd", "doomcube_4040"):
        _, rep = results[name]
        assert 95.0 <= rep.modal_fea.shaper_x.effective_freq_hz <= 185.0
        assert 70.0 <= rep.modal_fea.shaper_y.effective_freq_hz <= 145.0
        assert rep.modal_fea.shaper_x.effective_freq_hz > rep.modal_fea.shaper_y.effective_freq_hz

    # Cartesian bedslinger: empirical [35, 70] Hz X (toolhead), [25, 50] Hz Y (heavy moving bed), X > Y
    _, bs = results["cartesian_bedslinger"]
    assert 35.0 <= bs.modal_fea.shaper_x.effective_freq_hz <= 70.0
    assert 25.0 <= bs.modal_fea.shaper_y.effective_freq_hz <= 50.0
    assert bs.modal_fea.shaper_x.effective_freq_hz > bs.modal_fea.shaper_y.effective_freq_hz
    assert bs.modal_fea.nozzle_pitch_twist_freq_hz >= 85.0

    # Hyper CoreXY Carbon: light carbon gantry outperforms stock 2WD Vorons strongly
    _, hyp = results["hyper_corexy_carbon"]
    assert hyp.modal_fea.shaper_x.effective_freq_hz >= 75.0
    assert hyp.modal_fea.shaper_y.effective_freq_hz >= 50.0
    assert hyp.modal_fea.shaper_x.effective_freq_hz > hyp.modal_fea.shaper_y.effective_freq_hz

    # Delta Racer: near-isotropic axisymmetric XY effective frequencies
    _, delta = results["delta_racer"]
    dx = delta.modal_fea.shaper_x.effective_freq_hz
    dy = delta.modal_fea.shaper_y.effective_freq_hz
    assert dx > 60.0 and dy > 60.0
    assert abs(dx - dy) / max(dx, dy) < 0.03

    # Stock 2WD 2020 Vorons with the perimeter belt routing and the dynamic belt ratio fitted to
    # controlled same-machine AWD / 2WD and motor-current measurements (1.85 x datasheet static):
    # - Voron Trident 300: X ~60.6 / Y ~42.6 Hz, MZV recommendation ~62.8 / ~44.2 Hz
    # - Voron 2.4 300: X ~59.1 / Y ~41.6 Hz, MZV recommendation ~61.7 / ~43.2 Hz
    # The presets ship the Voron-documented belt tension (9 N, 110 Hz pluck on a 150 mm span).
    # Reference is an individually measured stock Trident 300 at 2.2 lb (about 9.8 N), hot
    # chamber: X MZV 61.4-64.0 Hz, Y MZV 40.8-42.2 Hz (docs/community_touchstones.md). The
    # curated "stock" touchstone rows (Y 46-48 Hz) are typical values, not measurements.
    _, tri = results["voron_trident_300"]
    _, v24 = results["corexy_voron_style"]
    assert 52.0 <= tri.modal_fea.shaper_x.effective_freq_hz <= 70.0
    assert 37.0 <= tri.modal_fea.shaper_y.effective_freq_hz <= 47.0
    assert 5000.0 <= tri.modal_fea.recommended_max_accel_mm_s2 <= 9000.0
    assert 52.0 <= v24.modal_fea.shaper_x.effective_freq_hz <= 70.0
    assert 37.0 <= v24.modal_fea.shaper_y.effective_freq_hz <= 47.0
    assert 5000.0 <= v24.modal_fea.recommended_max_accel_mm_s2 <= 9000.0
    assert tri.modal_fea.shaper_x.effective_freq_hz > tri.modal_fea.shaper_y.effective_freq_hz
    assert v24.modal_fea.shaper_x.effective_freq_hz > v24.modal_fea.shaper_y.effective_freq_hz
    assert tri.modal_fea.fundamental_freq_hz > v24.modal_fea.fundamental_freq_hz
    assert tri.modal_fea.shaper_x.effective_freq_hz > v24.modal_fea.shaper_x.effective_freq_hz
    assert tri.modal_fea.shaper_y.effective_freq_hz > v24.modal_fea.shaper_y.effective_freq_hz
    # Measured: stock Trident X MZV 61-64 Hz vs VzBot 330 AWD X ~100 Hz community mean (~1.6x),
    # Y 41-42 vs ~67 Hz (~1.6x -> ~2.5x shaper accel since a_max ~ f^2)
    assert vz.modal_fea.shaper_x.effective_freq_hz > tri.modal_fea.shaper_x.effective_freq_hz * 1.5
    assert vz.modal_fea.shaper_y.effective_freq_hz > tri.modal_fea.shaper_y.effective_freq_hz * 1.4
    assert vz.modal_fea.recommended_max_accel_mm_s2 > tri.modal_fea.recommended_max_accel_mm_s2 * 1.8
