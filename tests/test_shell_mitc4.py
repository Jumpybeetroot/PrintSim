"""Analytical continuum benchmarks and multiphysics integration tests for MITC4 shell & Huth CBUSH FEA."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.linalg import eigh, solve
from typer.testing import CliRunner

from printsim.api.server import create_app
from printsim.cli import app as cli_app
from printsim.engine import evaluate_printer
from printsim.engine.evaluator import verify_printer_deep
from printsim.fea import solve_modal_fea, solve_static_fea
from printsim.fea.shell_mitc4 import (
    fastener_spring_matrices,
    mitc4_shell_matrices,
    mitc4_shell_von_mises_mpa,
)
from printsim.geometry import build_assembly, generate_bom
from printsim.mcp_server import dispatch_tool
from printsim.schema import PrinterSpec

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _assemble_3d_plate(
    a: float,
    b: float,
    nx_elem: int,
    ny_elem: int,
    e_pa: float,
    nu: float,
    thickness_m: float,
    rho_kg_m3: float,
    pressure_pa: float = 0.0,
    orient: str = "xy",
    d_bend_nm: float | None = None,
    g_transverse_h_n_m: float | None = None,
    e_mem_h_n_m: float | None = None,
):
    """Build a 3D simply-supported rectangular plate mesh using mitc4_shell_matrices."""
    nx_n = nx_elem + 1
    ny_n = ny_elem + 1
    n_nodes = nx_n * ny_n
    n_dof = 6 * n_nodes

    coords_3d = np.zeros((n_nodes, 3), dtype=float)
    for j in range(ny_n):
        for i in range(nx_n):
            u = i * a / nx_elem
            v = j * b / ny_elem
            if orient == "xy":
                coords_3d[j * nx_n + i] = [u, v, 0.0]
            elif orient == "yz":
                coords_3d[j * nx_n + i] = [0.0, u, v]
            else:
                # Arbitrary 3D tilted plane
                coords_3d[j * nx_n + i] = [u, 0.6 * v, 0.8 * v]

    K = np.zeros((n_dof, n_dof), dtype=float)
    M = np.zeros((n_dof, n_dof), dtype=float)
    F = np.zeros(n_dof, dtype=float)
    areal_mass = rho_kg_m3 * thickness_m

    # Local normal unit vector for pressure load
    if orient == "xy":
        n_vec = np.array([0.0, 0.0, 1.0])
    elif orient == "yz":
        n_vec = np.array([1.0, 0.0, 0.0])
    else:
        n_vec = np.array([0.0, -0.8, 0.6])

    elems: list[tuple[int, int, int, int]] = []
    elem_area = (a / nx_elem) * (b / ny_elem)
    for ey in range(ny_elem):
        for ex in range(nx_elem):
            n0 = ey * nx_n + ex
            n1 = ey * nx_n + ex + 1
            n2 = (ey + 1) * nx_n + ex + 1
            n3 = (ey + 1) * nx_n + ex
            elems.append((n0, n1, n2, n3))
            c_e = coords_3d[[n0, n1, n2, n3]]
            ke, me, _ = mitc4_shell_matrices(
                c_e,
                e_pa=e_pa,
                nu=nu,
                thickness_m=thickness_m,
                areal_mass_kg_m2=areal_mass,
                d_bend_nm=d_bend_nm,
                g_transverse_h_n_m=g_transverse_h_n_m,
                e_mem_h_n_m=e_mem_h_n_m,
            )
            dofs = [6 * nid + d for nid in (n0, n1, n2, n3) for d in range(6)]
            K[np.ix_(dofs, dofs)] += ke
            M[np.ix_(dofs, dofs)] += me
            if pressure_pa != 0.0:
                fn = 0.25 * pressure_pa * elem_area * n_vec
                for nid in (n0, n1, n2, n3):
                    F[6 * nid : 6 * nid + 3] += fn

    # Hard simply-supported (SS2) on XY/YZ or soft simply-supported (SS1) on tilted
    fixed: set[int] = set()
    for j in range(ny_n):
        for i in range(nx_n):
            nid = j * nx_n + i
            if i == 0 or i == nx_elem or j == 0 or j == ny_elem:
                # Fix all 3 translations on perimeter (w=0 out-of-plane + prevent in-plane rigid modes)
                fixed.update([6 * nid + 0, 6 * nid + 1, 6 * nid + 2])
                if orient == "xy":
                    if i == 0 or i == nx_elem:
                        fixed.add(6 * nid + 3)
                    if j == 0 or j == ny_elem:
                        fixed.add(6 * nid + 4)

    free = [d for d in range(n_dof) if d not in fixed]
    return coords_3d, elems, K, M, F, free, n_vec


# ---------------------------------------------------------------------------
# 1. Closed-Form Analytical MITC4 Benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slenderness", [20.0, 200.0, 300.0])
def test_benchmark1_navier_simply_supported_plate_deflection_and_stress(slenderness: float) -> None:
    """8x8 MITC4 matches Navier plate deflection (< 2%) & center stress at L/t = 20, 200, 300 (zero shear locking)."""
    a = 0.46
    t = a / slenderness
    e_pa = 70.3e9
    nu = 0.33
    rho = 2680.0
    q = 200.0 * (20.0 / slenderness) ** 2  # scale pressure so deflection stays in small-strain regime

    coords_3d, elems, K, _, F, free, _ = _assemble_3d_plate(
        a=a, b=a, nx_elem=8, ny_elem=8, e_pa=e_pa, nu=nu, thickness_m=t, rho_kg_m3=rho, pressure_pa=q
    )
    assert np.allclose(K, K.T, atol=1e-9)

    u = np.zeros(K.shape[0], dtype=float)
    u[free] = solve(K[np.ix_(free, free)], F[free], assume_a="pos")

    w_center = abs(float(u[6 * (4 * 9 + 4) + 2]))
    d_rigidity = e_pa * (t**3) / (12.0 * (1.0 - nu**2))
    w_navier = 0.00406235 * q * (a**4) / d_rigidity
    rel_err = abs(w_center - w_navier) / w_navier
    assert rel_err < 0.02, f"L/t={slenderness}: w_fea={w_center:.6e}, w_navier={w_navier:.6e}, err={rel_err:.4f}"

    # Verify center bending von Mises stress against exact Navier series at plate center
    m_center_exact = 0.0
    for m in range(1, 22, 2):
        for n in range(1, 22, 2):
            sign = ((-1) ** ((m - 1) // 2)) * ((-1) ** ((n - 1) // 2))
            coeff = 16.0 * q * (a**2) / ((math.pi**4) * m * n * ((m * m + n * n) ** 2))
            m_center_exact += sign * coeff * (m * m + nu * n * n)
    vm_exact_mpa = (6.0 * abs(m_center_exact) / (t**2)) * 1e-6

    # Center element (3, 3) & (4, 4) sit adjacent to (a/2, a/2)
    center_elem = elems[3 * 8 + 3]
    u_elem = np.concatenate([u[6 * nid : 6 * nid + 6] for nid in center_elem])
    vm_elem_mpa = mitc4_shell_von_mises_mpa(coords_3d[list(center_elem)], u_elem, e_pa, nu, t)
    assert vm_elem_mpa == pytest.approx(vm_exact_mpa, rel=0.05)


@pytest.mark.parametrize("orient,slenderness", [("xy", 20.0), ("xy", 200.0), ("yz", 200.0), ("tilted", 200.0)])
def test_benchmark2_navier_plate_fundamental_frequency_and_3d_frame_invariance(
    orient: str, slenderness: float
) -> None:
    """8x8 MITC4 matches exact analytical plate drum frequency f11 (< 2%) across 3D orientations."""
    a = 0.46
    b = 0.50
    t = a / slenderness
    e_pa = 70.3e9
    nu = 0.33
    rho = 2680.0

    _, _, K, M, _, free, _ = _assemble_3d_plate(
        a=a, b=b, nx_elem=8, ny_elem=8, e_pa=e_pa, nu=nu, thickness_m=t, rho_kg_m3=rho, orient=orient
    )
    assert np.allclose(M, M.T, atol=1e-12)
    # Exact total mass conservation check: M translational sum == 4*a*b*rho*t? Wait: 3 axes sum = 3 * m_tot
    m_tot_exact = a * b * rho * t
    assert float(np.sum(np.diag(M)[0::6])) == pytest.approx(m_tot_exact, rel=1e-9)

    eigvals, _ = eigh(K[np.ix_(free, free)], M[np.ix_(free, free)], subset_by_index=[0, 0])
    f11_fea = math.sqrt(max(float(eigvals[0]), 0.0)) / (2.0 * math.pi)
    d_rigidity = e_pa * (t**3) / (12.0 * (1.0 - nu**2))
    f11_exact = (math.pi / 2.0) * math.sqrt(d_rigidity / (rho * t)) * (1.0 / (a**2) + 1.0 / (b**2))
    assert abs(f11_fea - f11_exact) / f11_exact < 0.02


def test_benchmark3_pure_shear_membrane_and_cbush_rigid_body_invariance() -> None:
    """In-plane pure shear patch test (< 1e-6) + rigid-body zero modes on MITC4 & eccentric CBUSH."""
    w = 0.46
    h = 0.50
    t = 0.004
    e_pa = 70.3e9
    nu = 0.33
    g_pa = e_pa / (2.0 * (1.0 + nu))
    rho = 2680.0

    # Vertical quad in YZ plane
    coords = np.array(
        [[0.0, 0.0, 0.0], [0.0, w, 0.0], [0.0, w, h], [0.0, 0.0, h]],
        dtype=float,
    )
    K, M, F_g = mitc4_shell_matrices(coords, e_pa, nu, t, rho * t)
    assert np.allclose(K, K.T, atol=1e-9)
    assert np.allclose(M, M.T, atol=1e-12)
    assert float(np.sum(F_g[2::6])) == pytest.approx(-w * h * rho * t * 9.80665, rel=1e-9)

    # Exactly 6 zero rigid-body modes for unconstrained MITC4 element
    evals = np.linalg.eigvalsh(K)
    assert np.all(evals >= -1e-6)
    assert np.sum(evals < 1e-5) == 6

    # Pure simple shear test: fix bottom nodes (0, 1); constrain X and Z on top nodes (2, 3), pull +Y
    force_y = 1000.0
    fixed = [
        0, 1, 2, 3, 4, 5,
        6, 7, 8, 9, 10, 11,
        12 + 0, 12 + 2, 12 + 3, 12 + 4, 12 + 5,
        18 + 0, 18 + 2, 18 + 3, 18 + 4, 18 + 5,
    ]
    free = [d for d in range(24) if d not in fixed]
    F = np.zeros(24, dtype=float)
    F[12 + 1] = 0.5 * force_y
    F[18 + 1] = 0.5 * force_y
    u = np.zeros(24, dtype=float)
    u[free] = solve(K[np.ix_(free, free)], F[free], assume_a="pos")

    delta_exact = force_y * h / (g_pa * t * w)
    assert u[12 + 1] == pytest.approx(delta_exact, rel=1e-6)
    assert u[18 + 1] == pytest.approx(delta_exact, rel=1e-6)

    # Eccentric CBUSH fastener spring: symmetric PSD with exact 6 rigid-body zero modes
    k_cbush = fastener_spring_matrices(
        k_shear_n_m=2.5e7,
        k_normal_n_m=5.0e7,
        k_rot_nm_rad=15.0,
        normal_vec=np.array([-1.0, 0.0, 0.0]),
        offset_vec=np.array([-0.012, 0.0, 0.0]),
    )
    assert np.allclose(k_cbush, k_cbush.T, atol=1e-9)
    cb_evals = np.linalg.eigvalsh(k_cbush)
    assert np.all(cb_evals >= -1e-6)
    assert np.sum(cb_evals < 1e-5) == 6


def test_acm_sandwich_core_shear_capture() -> None:
    """Custom d_bend_nm and g_transverse_h_n_m capture LDPE core shear compliance on short spans."""
    a = 0.25
    t = 0.004
    t_skin = 0.0003
    t_core = t - 2.0 * t_skin
    e_al = 70.0e9
    nu_al = 0.33
    d_bend = e_al * (t_skin * (t - t_skin) ** 2) / (2.0 * (1.0 - nu_al**2))
    e_mem_h = e_al * (2.0 * t_skin) + 0.48e9 * t_core
    g_soft_core = 0.17e9 * t_core
    g_stiff_core = 26.3e9 * t_core

    _, _, k_soft, _, f_soft, free_s, _ = _assemble_3d_plate(
        a, a, 6, 6, 22.0e9, nu_al, t, 1350.0, pressure_pa=200.0,
        d_bend_nm=d_bend, g_transverse_h_n_m=g_soft_core, e_mem_h_n_m=e_mem_h,
    )
    _, _, k_stiff, _, f_stiff, free_r, _ = _assemble_3d_plate(
        a, a, 6, 6, 22.0e9, nu_al, t, 1350.0, pressure_pa=200.0,
        d_bend_nm=d_bend, g_transverse_h_n_m=g_stiff_core, e_mem_h_n_m=e_mem_h,
    )
    u_soft_full = np.zeros(k_soft.shape[0], dtype=float)
    u_stiff_full = np.zeros(k_stiff.shape[0], dtype=float)
    u_soft_full[free_s] = solve(k_soft[np.ix_(free_s, free_s)], f_soft[free_s], assume_a="pos")
    u_stiff_full[free_r] = solve(k_stiff[np.ix_(free_r, free_r)], f_stiff[free_r], assume_a="pos")
    w_soft = float(np.max(np.abs(u_soft_full[2::6])))
    w_stiff = float(np.max(np.abs(u_stiff_full[2::6])))
    assert w_soft > w_stiff * 1.03


# ---------------------------------------------------------------------------
# 2. Monolith AWD Panel Sweep & Oil-Canning Diagnostics
# ---------------------------------------------------------------------------


def test_benchmark4_monolith_awd_panel_sweep_and_drum_frequency_scaling() -> None:
    """4-5mm Al > 1.5-2mm Al > 3mm ACM > Clip Acrylic > Open; Al drum freq climbs linearly; <65 Hz triggers diagnostic."""
    base = PrinterSpec.from_yaml(EXAMPLES_DIR / "voron_monolith_awd.yaml")

    def _eval(enclosed: bool, mounting: str, mat: str, thk_mm: float):
        s = base.model_copy(deep=True)
        s.frame.enclosed_panels = enclosed
        s.frame.panel_mounting = mounting  # type: ignore[assignment]
        s.frame.panel_material = mat
        s.frame.panel_thickness_mm = thk_mm
        return evaluate_printer(s)

    r_open = _eval(False, "none", "acm_dibond", 3.0)
    r_clip = _eval(True, "clip_enclosure", "acrylic_pmma", 3.0)
    r_acm3 = _eval(True, "bolted_structural", "acm_dibond", 3.0)
    r_al15 = _eval(True, "bolted_structural", "aluminum_5052_sheet", 1.5)
    r_al20 = _eval(True, "bolted_structural", "aluminum_5052_sheet", 2.0)
    r_al30 = _eval(True, "bolted_structural", "aluminum_5052_sheet", 3.0)
    r_al40 = _eval(True, "bolted_structural", "aluminum_5052_sheet", 4.0)
    r_al50 = _eval(True, "bolted_structural", "aluminum_5052_sheet", 5.0)
    r_acr3 = _eval(True, "bolted_structural", "acrylic_pmma", 3.0)

    # Structural stiffness and fundamental frequency hierarchy
    assert (
        r_al50.static_fea.toolhead_stiffness_x_n_per_mm
        > r_al40.static_fea.toolhead_stiffness_x_n_per_mm
        > r_al20.static_fea.toolhead_stiffness_x_n_per_mm
        > r_al15.static_fea.toolhead_stiffness_x_n_per_mm
        > r_acm3.static_fea.toolhead_stiffness_x_n_per_mm
        > r_clip.static_fea.toolhead_stiffness_x_n_per_mm
        > r_open.static_fea.toolhead_stiffness_x_n_per_mm
    )
    assert (
        r_al50.modal_fea.fundamental_freq_hz
        > r_al40.modal_fea.fundamental_freq_hz
        > r_al20.modal_fea.fundamental_freq_hz
        > r_al15.modal_fea.fundamental_freq_hz
        > r_acm3.modal_fea.fundamental_freq_hz
        > r_open.modal_fea.fundamental_freq_hz
    )

    # Aluminum panel drum frequency climbs linearly with thickness (1.5 < 3.0 < 5.0 mm)
    assert r_open.modal_fea.panel_drum_freq_hz == 0.0
    assert (
        r_al15.modal_fea.panel_drum_freq_hz
        < r_al20.modal_fea.panel_drum_freq_hz
        < r_al30.modal_fea.panel_drum_freq_hz
        < r_al40.modal_fea.panel_drum_freq_hz
        < r_al50.modal_fea.panel_drum_freq_hz
    )
    ratio_5_to_15 = r_al50.modal_fea.panel_drum_freq_hz / r_al15.modal_fea.panel_drum_freq_hz
    assert 2.6 < ratio_5_to_15 < 3.6  # analytical f11 ~ t -> 5/1.5 = 3.33x

    # Bolted 1.5mm Al & 3mm Acrylic (< 65 Hz) trigger oil-canning diagnostic; 4-5mm Al (> 65 Hz) do not
    assert 0.0 < r_al15.modal_fea.panel_drum_freq_hz < 65.0
    assert 0.0 < r_acr3.modal_fea.panel_drum_freq_hz < 65.0
    assert r_al40.modal_fea.panel_drum_freq_hz > 65.0
    assert r_al50.modal_fea.panel_drum_freq_hz > 65.0
    assert any("oil-canning" in d.lower() for d in r_al15.diagnostics)
    assert any("oil-canning" in d.lower() for d in r_acr3.diagnostics)
    assert not any("oil-canning" in d.lower() for d in r_al40.diagnostics)
    assert not any("oil-canning" in d.lower() for d in r_al50.diagnostics)

    # Floppy 3mm acrylic drum mode (~20 Hz) must NOT hijack structural fundamental_freq_hz (~55+ Hz)
    assert r_acr3.modal_fea.panel_drum_freq_hz < 30.0
    assert r_acr3.modal_fea.fundamental_freq_hz > 50.0


# ---------------------------------------------------------------------------
# 3. Deep Offline 6x6 Verification, BOM Merging, API, CLI & MCP
# ---------------------------------------------------------------------------


def test_benchmark5_deep_verify_convergence_bom_api_cli_and_mcp() -> None:
    """Verify fast 2x2 vs deep 6x6 convergence, clean BOM bar merging, FastAPI, CLI, and MCP."""
    monolith_path = EXAMPLES_DIR / "voron_monolith_awd.yaml"
    monolith_yaml = monolith_path.read_text(encoding="utf-8")
    spec = PrinterSpec.from_yaml(monolith_path)

    # BOM cut-list must stay identical even when frame/rails are subdivided for 6x6 deep shell bolts
    bom_fast = generate_bom(build_assembly(spec, panel_mesh_n=2))
    bom_deep = generate_bom(build_assembly(spec, panel_mesh_n=6))
    cuts_fast = [(c.profile_id, c.material_id, c.role, c.length_mm, c.quantity) for c in bom_fast.cut_list]
    cuts_deep = [(c.profile_id, c.material_id, c.role, c.length_mm, c.quantity) for c in bom_deep.cut_list]
    assert cuts_deep == cuts_fast

    # Direct Python evaluator deep verify
    deep_res = verify_printer_deep(spec)
    assert deep_res["fast_mesh_n"] == 2
    assert deep_res["deep_mesh_n"] == 6
    assert deep_res["deep_summary"]["num_shell_elements"] > deep_res["fast_summary"]["num_shell_elements"]
    assert deep_res["deep_summary"]["panel_max_von_mises_mpa"] > 0.0
    assert deep_res["deep_summary"]["panel_drum_freq_hz"] > 65.0
    assert deep_res["convergence"]["converged"] is True
    assert deep_res["convergence"]["freq_diff_pct"] < 15.0
    assert deep_res["convergence"]["deflection_diff_pct"] < 15.0
    sim = deep_res["deep_simulation"]
    assert len(sim["static_fea"]["shell_panel_stresses"]) == deep_res["deep_summary"]["num_shell_elements"]
    assert sim["modal_fea"]["panel_drum_freq_hz"] == deep_res["deep_summary"]["panel_drum_freq_hz"]
    assert sim["geometry"]["panel_mesh_n"] == 6
    assert len(sim["geometry"]["shell_panels"]) == deep_res["deep_summary"]["num_shell_elements"]

    # FastAPI /api/health (v1.1.0) & POST /api/verify/deep
    client = TestClient(create_app())
    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json()["version"] == "1.1.0"

    r_verify = client.post("/api/verify/deep", json={"yaml_text": monolith_yaml})
    assert r_verify.status_code == 200
    api_data = r_verify.json()
    assert api_data["deep_mesh_n"] == 6
    assert api_data["convergence"]["converged"] is True

    # CLI `printsim verify <yaml> --json`
    runner = CliRunner()
    cli_res = runner.invoke(cli_app, ["verify", str(monolith_path), "--json"])
    assert cli_res.exit_code == 0, cli_res.output
    assert '"deep_mesh_n": 6' in cli_res.output

    # MCP `run_deep_frame_verify`
    mcp_res = dispatch_tool("run_deep_frame_verify", {"yaml_text": monolith_yaml})
    assert mcp_res["deep_mesh_n"] == 6
    assert mcp_res["convergence"]["converged"] is True
    assert mcp_res["deep_summary"]["panel_max_von_mises_mpa"] > 0.0


def test_benchmark6_aluminum_thickness_monotonic_no_shaper_cliffs() -> None:
    """Al 5052 thickness sweep (1.5 -> 6.0 mm) climbs monotonically with zero mode-switch or shaper cliffs."""
    thicknesses = (1.5, 2.0, 3.0, 4.0, 5.0, 6.0)

    for yaml_name, mesh_ns in (("voron_monolith_awd.yaml", (2, 6)), ("voron_trident_300.yaml", (2,))):
        base = PrinterSpec.from_yaml(EXAMPLES_DIR / yaml_name)
        for mesh_n in mesh_ns:
            prev_kx = 0.0
            prev_fxs = 0.0
            prev_ax = 0.0
            prev_drum = 0.0
            for thk in thicknesses:
                s = base.model_copy(deep=True)
                s.frame.enclosed_panels = True
                s.frame.panel_mounting = "bolted_structural"
                s.frame.panel_material = "aluminum_5052_sheet"
                s.frame.panel_thickness_mm = thk
                asm = build_assembly(s, panel_mesh_n=mesh_n)
                st = solve_static_fea(asm)
                mo = solve_modal_fea(asm)
                assert mo.shaper_x is not None and mo.shaper_y is not None

                kx = st.toolhead_stiffness_x_n_per_mm
                fxs = mo.shaper_x.structural_freq_hz
                ax = mo.shaper_x.max_accel_mm_s2
                drum = mo.panel_drum_freq_hz

                assert kx > prev_kx, f"{yaml_name} n={mesh_n} Al {thk}mm Kx {kx} <= {prev_kx}"
                assert fxs > prev_fxs, f"{yaml_name} n={mesh_n} Al {thk}mm fXs {fxs} <= {prev_fxs}"
                assert ax >= prev_ax, f"{yaml_name} n={mesh_n} Al {thk}mm AX {ax} < {prev_ax}"
                assert drum > prev_drum, f"{yaml_name} n={mesh_n} Al {thk}mm drum {drum} <= {prev_drum}"
                # Panel drum mode at 5-6mm must never hijack f1 downward
                assert mo.fundamental_freq_hz > 100.0

                prev_kx, prev_fxs, prev_ax, prev_drum = kx, fxs, ax, drum
