"""CAL-15: Analytical & bench-benchmark tests for Cross-WLF melt rheology & hotend push-force curves."""

from __future__ import annotations

import math

import pytest

from printsim.schema import PrinterSpec
from printsim.schema.printdb_bridge import get_record_specs, is_printdb_available
from printsim.thermal.filaments import FILAMENT_CATALOG, get_filament
from printsim.thermal.hotend import (
    compute_push_force_curve,
    coupled_melt_pressure_drop_pa,
    cross_wlf_nozzle_pressure_drop_pa,
    cross_wlf_viscosity_pa_s,
    cross_wlf_wall_shear_stress_pa,
    cross_wlf_zero_shear_viscosity_pa_s,
    graetz_characteristic_flow_m3_s,
    nozzle_total_pressure_drop_pa,
    simulate_hotend_thermal,
)


# ---------------------------------------------------------------------------
# 1. Analytical Cross-WLF & Weissenberg-Rabinowitsch Verification
# ---------------------------------------------------------------------------


def test_cross_wlf_matches_ingeo_4032d_published_rheometry() -> None:
    """Exact reproduction of Di Lorenzo et al. (2025) Cross-WLF fit for Ingeo 4032D PLA."""
    # PrintDB materials/natureworks-ingeo-4032d verified Cross-WLF parameters
    d1, d2_k, a1, a3_k, tau_s, n = 3505.0, 433.0, 12.3, 175.0, 410000.0, 0.17

    assert cross_wlf_zero_shear_viscosity_pa_s(160.0, d1, d2_k, a1, a3_k) == pytest.approx(3468.0, rel=0.01)
    assert cross_wlf_zero_shear_viscosity_pa_s(180.0, d1, d2_k, a1, a3_k) == pytest.approx(984.0, rel=0.01)
    assert cross_wlf_zero_shear_viscosity_pa_s(200.0, d1, d2_k, a1, a3_k) == pytest.approx(353.0, rel=0.01)
    assert cross_wlf_zero_shear_viscosity_pa_s(210.0, d1, d2_k, a1, a3_k) == pytest.approx(226.0, rel=0.01)

    eta_100 = cross_wlf_viscosity_pa_s(100.0, 210.0, d1, d2_k, a1, tau_s, n, a3_k)
    eta_1000 = cross_wlf_viscosity_pa_s(1000.0, 210.0, d1, d2_k, a1, tau_s, n, a3_k)
    assert eta_100 == pytest.approx(208.0, rel=0.01)
    assert eta_1000 == pytest.approx(141.0, rel=0.01)


def test_cross_wlf_newtonian_and_power_law_asymptotic_limits() -> None:
    """Cross-WLF pipe & cone solver recovers exact Hagen-Poiseuille at low shear and Bellini at high shear."""
    fil = get_filament("pla")
    eta0 = cross_wlf_zero_shear_viscosity_pa_s(
        fil.print_temp_c, fil.wlf_d1_pa_s, fil.wlf_d2_k, fil.wlf_a1, fil.wlf_a3_k
    )
    assert eta0 == pytest.approx(850.0, rel=0.01)

    # 1) Ultra-low shear rate -> exact Newtonian Hagen-Poiseuille: dP = 8 * eta0 * L * Q / (pi * R^4)
    q_tiny = 1e-16  # m^3/s -> gamma_a ~ 1.6e-5 s^-1 in 0.4 mm nozzle (deep inside Newtonian plateau)
    r_pipe = 0.0002
    l_pipe = 0.005
    tau_w, gamma_w, eta_w = cross_wlf_wall_shear_stress_pa(
        q_tiny, r_pipe, eta0, fil.cross_tau_star_pa, fil.cross_n
    )
    dp_cross = 2.0 * l_pipe * tau_w / r_pipe
    dp_hp = (8.0 * eta0 * l_pipe * q_tiny) / (math.pi * (r_pipe**4))
    assert eta_w == pytest.approx(eta0, rel=1e-3)
    assert dp_cross == pytest.approx(dp_hp, rel=2e-3)

    # 2) Ultra-high shear rate -> exact power-law asymptote K_eff = eta0^n * (tau*)^(1-n)
    k_eff = (eta0**fil.cross_n) * (fil.cross_tau_star_pa ** (1.0 - fil.cross_n))
    assert k_eff == pytest.approx(fil.power_law_k_pa_sn, rel=0.02)

    q_high = 5e-5  # very high flow so eta0 * gamma / tau* >> 1 everywhere including cone
    res_cross = cross_wlf_nozzle_pressure_drop_pa(
        q_m3_s=q_high,
        nozzle_diameter_m=0.0004,
        melt_zone_length_m=0.0,
        chamber_diameter_m=0.002,
        eta0_film_pa_s=eta0,
        eta0_exit_pa_s=eta0,
        tau_star_pa=fil.cross_tau_star_pa,
        n=fil.cross_n,
        orifice_length_m=0.001,
        bagley_factor=3.0,
    )
    dp_bellini = nozzle_total_pressure_drop_pa(
        q_m3_s=q_high,
        nozzle_diameter_m=0.0004,
        melt_zone_length_m=0.0,
        chamber_diameter_m=0.002,
        k_pa_sn=k_eff,
        n=fil.cross_n,
        orifice_length_m=0.001,
        bagley_factor=3.0,
    )
    assert res_cross["dp_total_pa"] == pytest.approx(dp_bellini, rel=0.015)

    # 3) Pressure sensitivity D3 > 0 raises viscosity monotonically with pressure
    eta_p0 = cross_wlf_zero_shear_viscosity_pa_s(215.0, fil.wlf_d1_pa_s, fil.wlf_d2_k, fil.wlf_a1, fil.wlf_a3_k, 0.0, 2e-7)
    eta_p20 = cross_wlf_zero_shear_viscosity_pa_s(215.0, fil.wlf_d1_pa_s, fil.wlf_d2_k, fil.wlf_a1, fil.wlf_a3_k, 20e6, 2e-7)
    assert eta_p20 > eta_p0


def test_cross_wlf_rabinowitsch_accuracy_vs_exact_radial_pipe_integral() -> None:
    """Verify local-slope Weissenberg-Rabinowitsch wall stress matches exact radial pipe integral <= 3%."""
    eta0 = 850.0
    tau_star = 85000.0
    n = 0.36
    radius_m = 0.0002

    def invert_gamma(tau_val: float) -> float:
        """Solve eta0 * g / (1 + (eta0 * g / tau_star)^(1-n)) = tau_val by bisection."""
        if tau_val <= 0.0:
            return 0.0
        lo, hi = 0.0, max(tau_val / eta0, 1.0)
        while (eta0 * hi) / (1.0 + (eta0 * hi / tau_star) ** (1.0 - n)) < tau_val:
            hi *= 2.0
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            t_mid = (eta0 * mid) / (1.0 + (eta0 * mid / tau_star) ** (1.0 - n))
            if t_mid < tau_val:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # Sweep across Newtonian, transition knee, and power-law flow rates
    for q_mm3_s in (0.05, 0.5, 2.0, 10.0, 40.0):
        q = q_mm3_s * 1e-9
        tau_w, _, _ = cross_wlf_wall_shear_stress_pa(q, radius_m, eta0, tau_star, n)
        # Exact volumetric flow from radial integral: Q = (pi * R^3 / tau_w^3) * int_0^{tau_w} tau^2 * gamma(tau) dtau
        n_steps = 200
        dtau = tau_w / n_steps
        integral = 0.0
        for k in range(n_steps):
            t_mid = (k + 0.5) * dtau
            integral += (t_mid**2) * invert_gamma(t_mid) * dtau
        q_exact = (math.pi * (radius_m**3) / (tau_w**3)) * integral
        assert q_exact == pytest.approx(q, rel=0.03)


# ---------------------------------------------------------------------------
# 2. Serdeczny et al. & CNC Kitchen Push-Force Curve F(Q) Calibration
# ---------------------------------------------------------------------------


def test_serdeczny_v6_pla_push_force_curve_and_temperature_sensitivity() -> None:
    """E3D V6 0.4 mm PLA @ 215 C matches Serdeczny stable regime (2-15 N) and steep hockey-stick knee."""
    spec = PrinterSpec()
    spec.toolhead.hotend.preset_id = "e3d_v6_groove"
    spec.toolhead.hotend.nozzle_diameter_mm = 0.4

    flows = [2.0, 5.0, 8.0, 10.0, 12.0, 14.0, 16.0, 17.5, 19.0]
    curve_215 = compute_push_force_curve(spec, filament_id="pla", target_temp_c=215.0, flow_points_mm3_s=flows)
    f_by_q = {pt["flow_mm3_s"]: pt["push_force_n"] for pt in curve_215}
    t_by_q = {pt["flow_mm3_s"]: pt["melt_exit_temp_c"] for pt in curve_215}

    # Stable low-to-moderate extrusion regime (Serdeczny et al. 2019/2020: ~2-15 N)
    assert 1.5 <= f_by_q[2.0] <= 5.0
    assert 3.0 <= f_by_q[5.0] <= 9.0
    assert 6.0 <= f_by_q[10.0] <= 15.0
    assert 8.0 <= f_by_q[12.0] <= 18.0

    # Steep non-linear hockey-stick rise approaching melt-zone ceiling (~16-18 mm^3/s)
    assert f_by_q[17.5] > f_by_q[12.0] * 3.0
    assert f_by_q[19.0] > 80.0

    # Strict monotonicity: F(Q) increases and bulk melt exit temp T_exit(Q) decreases with Q
    for i in range(len(curve_215) - 1):
        assert curve_215[i + 1]["push_force_n"] > curve_215[i]["push_force_n"]
        assert curve_215[i + 1]["melt_exit_temp_c"] <= curve_215[i]["melt_exit_temp_c"]

    # Higher nozzle temperature lowers push force at fixed flow (200 -> 215 -> 230 C)
    curve_200 = compute_push_force_curve(spec, filament_id="pla", target_temp_c=200.0, flow_points_mm3_s=[10.0])
    curve_230 = compute_push_force_curve(spec, filament_id="pla", target_temp_c=230.0, flow_points_mm3_s=[10.0])
    assert curve_200[0]["push_force_n"] > f_by_q[10.0] > curve_230[0]["push_force_n"]
    assert t_by_q[2.0] > 213.0
    assert t_by_q[16.0] < 180.0


def test_nozzle_diameter_and_cht_split_core_scaling() -> None:
    """Larger orifices (0.4 -> 0.6 -> 0.8 mm) and CHT split-core nozzles cut push force and raise max flow."""
    # 1. Nozzle diameter relief at fixed flow and on max volumetric flow
    spec = PrinterSpec()
    spec.toolhead.hotend.preset_id = "phaetus_rapido_2_hf"
    f_at_25: dict[float, float] = {}
    q_max_by_d: dict[float, float] = {}
    for d_mm in (0.4, 0.6, 0.8):
        spec.toolhead.hotend.nozzle_diameter_mm = d_mm
        rep = simulate_hotend_thermal(spec, filament_id="pla")
        q_max_by_d[d_mm] = rep.max_volumetric_flow_mm3_s
        f_at_25[d_mm] = compute_push_force_curve(spec, filament_id="pla", flow_points_mm3_s=[25.0])[0]["push_force_n"]

    assert f_at_25[0.8] < f_at_25[0.6] < f_at_25[0.4]
    assert q_max_by_d[0.4] < q_max_by_d[0.6] < q_max_by_d[0.8]

    # 2. CHT nozzle boost matches Bondtech / CNC Kitchen / E3D Revo HF scaling:
    #    Largest relative gain on short melt zones (V6 +45-65%, Revo +35-55%), tapering on 50-52 mm hotends.
    cht_gain_bounds = {
        "e3d_v6_groove": (1.42, 1.68),
        "e3d_revo_voron": (1.35, 1.58),
        "phaetus_dragon_hf": (1.30, 1.50),
        "phaetus_rapido_2_hf": (1.22, 1.42),
        "mellow_goliath_air": (1.06, 1.22),
        "chube_conduction": (1.06, 1.20),
    }
    for hid, (lo_gain, hi_gain) in cht_gain_bounds.items():
        s_h = PrinterSpec()
        s_h.toolhead.hotend.preset_id = hid
        s_h.toolhead.hotend.nozzle_diameter_mm = 0.4
        r_std = simulate_hotend_thermal(s_h, filament_id="pla", nozzle_type="standard")
        r_cht = simulate_hotend_thermal(s_h, filament_id="pla", nozzle_type="cht")
        gain = r_cht.max_volumetric_flow_mm3_s / r_std.max_volumetric_flow_mm3_s
        assert lo_gain <= gain <= hi_gain, f"{hid}: CHT gain {gain:.2f}x outside [{lo_gain}, {hi_gain}]"
        assert r_cht.nozzle_type == "cht"

    # CHT significantly reduces push force at 15 mm^3/s on E3D V6 PLA 215 C
    s_v6 = PrinterSpec()
    s_v6.toolhead.hotend.preset_id = "e3d_v6_groove"
    f_v6_std = compute_push_force_curve(s_v6, filament_id="pla", nozzle_type="standard", flow_points_mm3_s=[15.0])[0]["push_force_n"]
    f_v6_cht = compute_push_force_curve(s_v6, filament_id="pla", nozzle_type="cht", flow_points_mm3_s=[15.0])[0]["push_force_n"]
    assert f_v6_cht < f_v6_std * 0.65


# ---------------------------------------------------------------------------
# 3. Full Multi-Hotend x Multi-Polymer Volumetric Flow Bench Calibration Matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hotend_id", "filament_id", "nozzle_mm", "nozzle_type", "q_min", "q_max"),
    [
        # PLA @ 215 C (CNC Kitchen, Bondtech CHT, E3D Revo HF, Phaetus, VzBot Goliath, Luke's Lab Chube)
        ("e3d_v6_groove", "pla", 0.4, "standard", 14.0, 19.0),
        ("e3d_v6_groove", "pla", 0.4, "cht", 22.0, 30.0),
        ("e3d_revo_voron", "pla", 0.4, "standard", 17.0, 23.0),
        ("e3d_revo_voron", "pla", 0.4, "cht", 25.0, 34.0),
        ("phaetus_dragon_hf", "pla", 0.4, "standard", 21.0, 27.5),
        ("phaetus_rapido_2_hf", "pla", 0.4, "standard", 29.0, 37.0),
        ("phaetus_rapido_2_uhf", "pla", 0.4, "standard", 44.0, 56.0),
        ("chube_compact", "pla", 0.4, "standard", 42.0, 54.0),
        ("chube_air", "pla", 0.4, "standard", 54.0, 68.0),
        ("mellow_goliath_air", "pla", 0.4, "standard", 62.0, 78.0),
        ("chube_conduction", "pla", 0.4, "standard", 65.0, 82.0),
        ("chube_conduction", "pla", 0.6, "cht", 76.0, 96.0),
        # PETG @ 245 C
        ("e3d_v6_groove", "petg", 0.4, "standard", 12.0, 17.5),
        ("phaetus_dragon_hf", "petg", 0.4, "standard", 18.5, 25.0),
        ("phaetus_rapido_2_hf", "petg", 0.4, "standard", 25.5, 34.0),
        ("chube_conduction", "petg", 0.4, "standard", 56.0, 72.0),
        # ABS @ 255 C & ASA @ 260 C
        ("phaetus_dragon_hf", "abs", 0.4, "standard", 21.5, 28.5),
        ("phaetus_rapido_2_hf", "abs", 0.4, "standard", 29.5, 38.5),
        ("mellow_goliath_air", "abs", 0.4, "standard", 62.0, 80.0),
        ("chube_conduction", "abs", 0.4, "standard", 64.0, 84.0),
        ("phaetus_rapido_2_hf", "asa", 0.4, "standard", 28.0, 37.0),
        # PA12-CF15 @ 285 C (0.4 mm & 0.6 mm — verifies no inverted melt-zone length penalty!)
        ("e3d_v6_groove", "pa12_cf", 0.4, "standard", 9.5, 14.0),
        ("phaetus_dragon_hf", "pa12_cf", 0.4, "standard", 14.0, 19.5),
        ("phaetus_rapido_2_hf", "pa12_cf", 0.4, "standard", 19.0, 26.0),
        ("mellow_goliath_air", "pa12_cf", 0.4, "standard", 32.0, 44.0),
        ("chube_conduction", "pa12_cf", 0.4, "standard", 33.0, 46.0),
        ("chube_conduction", "pa12_cf", 0.6, "standard", 40.0, 56.0),
    ],
)
def test_hotend_volumetric_flow_benchmark_matrix(
    hotend_id: str,
    filament_id: str,
    nozzle_mm: float,
    nozzle_type: str,
    q_min: float,
    q_max: float,
) -> None:
    """Verify calibrated coupled max volumetric flow across all target hotends, nozzles & polymers."""
    spec = PrinterSpec()
    spec.toolhead.hotend.preset_id = hotend_id
    spec.toolhead.hotend.nozzle_diameter_mm = nozzle_mm
    rep = simulate_hotend_thermal(spec, filament_id=filament_id, nozzle_type=nozzle_type)
    assert q_min <= rep.max_volumetric_flow_mm3_s <= q_max, (
        f"{hotend_id} ({filament_id}, {nozzle_mm}mm {nozzle_type}): "
        f"{rep.max_volumetric_flow_mm3_s:.2f} mm^3/s outside [{q_min}, {q_max}]"
    )
    assert rep.required_push_force_n > 0.0
    assert rep.apparent_viscosity_pa_s > 0.0
    assert rep.wall_shear_rate_s1 > 0.0
    assert len(rep.push_force_curve) == 8


def test_longer_melt_zone_monotonically_improves_flow_for_all_polymers_including_pa12_cf_and_tpu() -> None:
    """Regression guard: Cross-WLF barrel plateau ensures Chube/Goliath always beat V6/Revo/Dragon."""
    ordered_hotends = [
        "e3d_v6_groove",
        "e3d_revo_voron",
        "phaetus_dragon_hf",
        "phaetus_rapido_2_hf",
        "chube_air",
        "chube_conduction",
    ]
    for fid in ("pla", "petg", "abs", "asa", "pa12_cf", "tpu_95a"):
        flows = []
        for hid in ordered_hotends:
            spec = PrinterSpec()
            spec.toolhead.hotend.preset_id = hid
            flows.append(simulate_hotend_thermal(spec, filament_id=fid).max_volumetric_flow_mm3_s)
        for i in range(len(flows) - 1):
            assert flows[i + 1] > flows[i], (
                f"{fid}: {ordered_hotends[i + 1]} ({flows[i + 1]:.1f}) <= "
                f"{ordered_hotends[i]} ({flows[i]:.1f})"
            )


def test_extruder_push_force_limit_and_report_serialization() -> None:
    """Weak extruder triggers extruder_pressure limit; report dict exposes full CAL-15 rheology fields."""
    spec_weak = PrinterSpec()
    spec_weak.toolhead.hotend.preset_id = "chube_conduction"
    spec_weak.toolhead.extruder.max_push_force_n = 18.0

    spec_strong = spec_weak.model_copy(deep=True)
    spec_strong.toolhead.extruder.max_push_force_n = 90.0

    r_weak = simulate_hotend_thermal(spec_weak, filament_id="pla")
    r_strong = simulate_hotend_thermal(spec_strong, filament_id="pla")
    assert r_weak.flow_limiting_factor == "extruder_pressure"
    assert r_weak.max_volumetric_flow_mm3_s < r_strong.max_volumetric_flow_mm3_s
    assert r_weak.required_push_force_n == pytest.approx(18.0, abs=0.25)

    payload = r_strong.to_dict()
    for key in (
        "coupled_flow_limit_mm3_s",
        "melt_exit_temp_c",
        "apparent_viscosity_pa_s",
        "wall_shear_rate_s1",
        "nozzle_type",
        "push_force_curve",
    ):
        assert key in payload
    assert len(payload["push_force_curve"]) == 8


def test_printdb_calibrated_cross_wlf_records_synchronized() -> None:
    """When PrintDB is available, verify CAL-15 calibrated Cross-WLF records match FILAMENT_CATALOG."""
    if not is_printdb_available():
        pytest.skip("PrintDB not available in this environment")

    mapping = {
        "pla": "materials/fff-pla-cross-wlf-calibrated",
        "petg": "materials/fff-petg-cross-wlf-calibrated",
        "abs": "materials/fff-abs-cross-wlf-calibrated",
        "asa": "materials/fff-asa-cross-wlf-calibrated",
        "pa12_cf": "materials/fff-pa12-cf15-cross-wlf-calibrated",
        "tpu_95a": "materials/fff-tpu-95a-cross-wlf-calibrated",
    }
    for fid, rid in mapping.items():
        specs = get_record_specs(rid, si=False)
        assert specs is not None, f"Missing PrintDB calibrated record: {rid}"
        cw = specs.get("cross_wlf")
        assert isinstance(cw, dict), f"Missing cross_wlf block in {rid}"
        fil = FILAMENT_CATALOG[fid]
        assert cw["d1_pa_s"] == pytest.approx(fil.wlf_d1_pa_s, rel=1e-3)
        assert cw["d2_k"] == pytest.approx(fil.wlf_d2_k, rel=1e-4)
        assert cw["a1_ratio"] == pytest.approx(fil.wlf_a1, rel=1e-4)
        assert cw["a3_k"] == pytest.approx(fil.wlf_a3_k, rel=1e-4)
        assert cw["tau_star_pa"] == pytest.approx(fil.cross_tau_star_pa, rel=1e-4)
        assert cw["n_ratio"] == pytest.approx(fil.cross_n, rel=1e-4)
