"""Rigid-body invariance of 6-DOF bearing springs, including springs between separated nodes.

A two-node spring must store no energy when both nodes move together as one rigid body. For a
6-DOF contact stiffness acting at a point ``p_c`` the relative motion is
``(u0 + th0 x (p_c - p0)) - (u1 + th1 x (p_c - p1))``; using one shared arm for both nodes is
only valid when ``p0 == p1`` (inner-mounted Y rails put the carriage node on the raceway, about
15 mm from the rail node).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from printsim.fea.shell_mitc4 import fastener_spring_matrices
from printsim.geometry.builder import build_assembly
from printsim.schema import PrinterSpec

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
_COREXY_EXAMPLES = sorted(
    p.name for p in EXAMPLES_DIR.glob("*.yaml") if "architecture: corexy" in p.read_text()
)
_K6 = (2.0e8, 1.0e9, 3.0e8, 2.0e4, 8.0e3, 1.5e4)
_UNUSED = dict(k_shear_n_m=1.0, k_normal_n_m=1.0, k_rot_nm_rad=1.0, normal_vec=np.array([0.0, 1.0, 0.0]))


def _rigid_modes(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Six rigid-body motions (3 translations, 3 rotations about the origin) of two nodes."""
    modes = np.zeros((12, 6))
    for i in range(3):
        modes[i, i] = 1.0
        modes[6 + i, i] = 1.0
        w = np.zeros(3)
        w[i] = 1.0
        modes[0:3, 3 + i] = np.cross(w, p0)
        modes[3:6, 3 + i] = w
        modes[6:9, 3 + i] = np.cross(w, p1)
        modes[9:12, 3 + i] = w
    return modes


def _spring_matrix(sp) -> np.ndarray:
    return fastener_spring_matrices(
        sp.k_shear_n_m,
        sp.k_normal_n_m,
        sp.k_rot_nm_rad,
        np.asarray(sp.normal_vec, dtype=float),
        offset_vec=np.asarray(sp.offset_m, dtype=float),
        k_diag_6dof=sp.k_diag_6dof,
        offset_vec_frame=None if sp.offset_frame_m is None else np.asarray(sp.offset_frame_m, dtype=float),
    )


def test_examples_cover_inner_and_coincident_mounts():
    mounts = {PrinterSpec.from_yaml(EXAMPLES_DIR / n).gantry.y_rail_mount for n in _COREXY_EXAMPLES}
    assert {"inner", "top", "bottom"} <= mounts


@pytest.mark.parametrize("example", _COREXY_EXAMPLES)
def test_six_dof_springs_store_no_energy_under_rigid_body_motion(example: str):
    asm = build_assembly(PrinterSpec.from_yaml(EXAMPLES_DIR / example))
    nm = asm.node_map
    six_dof = [sp for sp in asm.fastener_springs if sp.k_diag_6dof is not None]
    assert any(sp.id.startswith("y_car_bearing_") for sp in six_dof)
    for sp in six_dof:
        p0 = np.asarray(nm[sp.shell_node_id].position_m, dtype=float)
        p1 = np.asarray(nm[sp.frame_node_id].position_m, dtype=float)
        k12 = _spring_matrix(sp)
        residual = np.abs(k12 @ _rigid_modes(p0, p1)).max() / np.abs(k12).max()
        assert residual < 1e-9, f"{example}:{sp.id} resists rigid-body motion (residual {residual:.2e})"


@pytest.mark.parametrize("example", [n for n in _COREXY_EXAMPLES if "vzbot" in n or "monolith_awd" in n])
def test_inner_rail_bearing_contact_sits_at_the_carriage_node(example: str):
    asm = build_assembly(PrinterSpec.from_yaml(EXAMPLES_DIR / example))
    nm = asm.node_map
    for side in ("l", "r"):
        sp = next(s for s in asm.fastener_springs if s.id == f"y_car_bearing_{side}")
        p_rail = np.asarray(nm[sp.shell_node_id].position_m, dtype=float)
        p_car = np.asarray(nm[sp.frame_node_id].position_m, dtype=float)
        assert np.linalg.norm(p_car - p_rail) > 0.010  # nodes are apart
        # Both rigid arms end at the same point: the raceway, where the carriage node sits
        assert p_rail + np.asarray(sp.offset_m) == pytest.approx(p_car, abs=1e-12)
        assert p_car + np.asarray(sp.offset_frame_m) == pytest.approx(p_car, abs=1e-12)
        # With the rail node held, the carriage node sees the bare diagonal contact stiffness
        k12 = _spring_matrix(sp)
        assert k12[6:12, 6:12] == pytest.approx(np.diag(sp.k_diag_6dof), rel=1e-12, abs=1e-6)


def test_separated_node_element_closed_form():
    a = 0.0148  # carriage node (frame) sits +a in X from the rail node (shell); contact at the carriage node
    p0, p1 = np.zeros(3), np.array([a, 0.0, 0.0])
    k12 = fastener_spring_matrices(
        **_UNUSED,
        offset_vec=p1 - p0,
        k_diag_6dof=_K6,
        offset_vec_frame=np.zeros(3),
    )
    assert np.allclose(k12, k12.T)
    assert np.abs(k12 @ _rigid_modes(p0, p1)).max() / np.abs(k12).max() < 1e-12
    # Six rigid-body modes and six elastic modes
    eig = np.linalg.eigvalsh(k12)
    assert int(np.sum(np.abs(eig) < 1e-6 * eig.max())) == 6

    # Carriage held; rail node twists by theta about Y. The rail-side contact point (at +a in X)
    # moves theta x a = (0, 0, -theta * a), so the spring pushes the carriage with k_z * (-theta * a)
    # in Z and with k_ry * theta as a moment about Y.
    theta = 1.0e-4
    q = np.zeros(12)
    q[4] = theta
    f = k12 @ q
    f_car = -f[6:12]  # force the spring applies to the held carriage node
    assert f_car[2] == pytest.approx(_K6[2] * (-theta * a), rel=1e-12)
    assert f_car[4] == pytest.approx(_K6[4] * theta, rel=1e-12)
    assert f_car[[0, 1, 3, 5]] == pytest.approx(np.zeros(4), abs=1e-9)
    # Reaction on the rail node balances force and moment about the rail node
    assert f[0:3] == pytest.approx(f_car[0:3], rel=1e-12, abs=1e-9)
    moment_about_rail = f_car[3:6] + np.cross(p1 - p0, f_car[0:3])
    assert f[3:6] == pytest.approx(moment_about_rail, rel=1e-12, abs=1e-9)


def test_one_shared_arm_on_separated_nodes_is_not_rigid_body_invariant():
    """Documents why the per-node arms exist: the shared-arm form grounds rotations when nodes are apart."""
    a = 0.0148
    p0, p1 = np.zeros(3), np.array([a, 0.0, 0.0])
    shared = fastener_spring_matrices(**_UNUSED, offset_vec=p0 - p1, k_diag_6dof=_K6)
    modes = _rigid_modes(p0, p1)
    spurious = [float(modes[:, 3 + i] @ shared @ modes[:, 3 + i]) for i in range(3)]
    assert spurious[0] == pytest.approx(0.0, abs=1e-6)
    assert spurious[1] == pytest.approx(_K6[2] * a**2, rel=1e-9)  # k_z * a^2 about Y
    assert spurious[2] == pytest.approx(_K6[1] * a**2, rel=1e-9)  # k_y * a^2 about Z


def test_coincident_node_path_is_unchanged_and_matches_equal_per_node_arms():
    off = np.array([0.0, 0.0, 0.0075])
    legacy = fastener_spring_matrices(**_UNUSED, offset_vec=off, k_diag_6dof=_K6)
    # Reference built exactly as before this change: K = T^T K6 T in all four blocks
    t_r = np.eye(6)
    t_r[0:3, 3:6] = -np.array([[0.0, -off[2], off[1]], [off[2], 0.0, -off[0]], [-off[1], off[0], 0.0]])
    k6 = t_r.T @ np.diag(_K6) @ t_r
    ref = np.zeros((12, 12))
    ref[:6, :6] = k6
    ref[6:, 6:] = k6
    ref[:6, 6:] = -k6
    ref[6:, :6] = -k6
    ref = 0.5 * (ref + ref.T)
    assert np.array_equal(legacy, ref)  # bit-identical for coincident nodes
    per_node = fastener_spring_matrices(**_UNUSED, offset_vec=off, k_diag_6dof=_K6, offset_vec_frame=off)
    assert per_node == pytest.approx(legacy, rel=1e-12, abs=1e-6)
