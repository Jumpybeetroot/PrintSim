"""2D MITC4 heated bed plate FEM, mount compliance, thermal taco warp, and modal solver."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh, solve

from printsim.fea.shell_mitc4 import mitc4_shell_matrices
from printsim.schema import BED_MOUNT_CATALOG, PrinterSpec, get_material

_CARRIER_Z_STIFFNESS_N_M: dict[str, float] = {
    "belted_quad_z": 2.0e7,
    "leadscrew_triple": 1.6e6,
    "leadscrew_dual": 8.0e5,
    "leadscrew_single": 1.6e5,
}
_PROBE_FORCE_N = 2.0
_ROT_REG_NM_RAD = 1e-2


@dataclass
class BedPlateFEAResult:
    """Results from 2D MITC4 heated bed plate static, thermal-taco, and modal FEA."""

    grid_nx: int
    grid_ny: int
    grid_x_rel: list[float]
    grid_y_rel: list[float]
    mount_points_rel: list[list[float]]
    z_gravity_um: list[float]
    z_thermal_um: list[float]
    z_total_um: list[float]
    mode1_shape_z: list[float]
    gravity_sag_um: float
    thermal_peak_to_valley_um: float
    taco_buckling_um: float
    total_flatness_error_um: float
    probe_center_deflection_um: float
    probe_corner_deflection_um: float
    plate_first_mode_hz: float
    plate_first_mode_type: str
    plate_modes_hz: list[float]
    mesh_bed_leveling_required: bool
    first_layer_risk: str
    summary: str

    def to_dict(self) -> dict:
        return {
            "grid_nx": int(self.grid_nx),
            "grid_ny": int(self.grid_ny),
            "grid_x_rel": [round(float(v), 4) for v in self.grid_x_rel],
            "grid_y_rel": [round(float(v), 4) for v in self.grid_y_rel],
            "mount_points_rel": [
                [round(float(pt[0]), 4), round(float(pt[1]), 4)] for pt in self.mount_points_rel
            ],
            "z_gravity_um": [round(float(v), 2) for v in self.z_gravity_um],
            "z_thermal_um": [round(float(v), 2) for v in self.z_thermal_um],
            "z_total_um": [round(float(v), 2) for v in self.z_total_um],
            "mode1_shape_z": [round(float(v), 4) for v in self.mode1_shape_z],
            "gravity_sag_um": round(float(self.gravity_sag_um), 2),
            "thermal_peak_to_valley_um": round(float(self.thermal_peak_to_valley_um), 2),
            "taco_buckling_um": round(float(self.taco_buckling_um), 2),
            "total_flatness_error_um": round(float(self.total_flatness_error_um), 2),
            "probe_center_deflection_um": round(float(self.probe_center_deflection_um), 2),
            "probe_corner_deflection_um": round(float(self.probe_corner_deflection_um), 2),
            "plate_first_mode_hz": round(float(self.plate_first_mode_hz), 1),
            "plate_first_mode_type": self.plate_first_mode_type,
            "plate_modes_hz": [round(float(f), 1) for f in self.plate_modes_hz[:3]],
            "mesh_bed_leveling_required": bool(self.mesh_bed_leveling_required),
            "first_layer_risk": self.first_layer_risk,
            "summary": self.summary,
        }


def _resolve_mount_layout(
    spec: PrinterSpec,
    grid_x_rel: np.ndarray,
    grid_y_rel: np.ndarray,
) -> tuple[list[int], list[list[float]], float]:
    """Return snapped anchor node indices, relative coords, and Z-carrier stiffness (N/m)."""
    arch = spec.architecture
    z_drive = spec.bed.z_drive
    mount_type = spec.bed.mount_type

    if arch == "corexy" and z_drive == "leadscrew_single":
        target_pts = [(-0.25, 0.5), (0.0, 0.5), (0.25, 0.5)]
    elif arch == "corexy" and z_drive == "leadscrew_dual":
        target_pts = [(-0.5, 0.0), (0.5, 0.0)]
    elif mount_type in ("kinematic_3pt", "whopping_pochard_kinematic") or arch == "delta":
        target_pts = [(-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)]
    else:
        target_pts = [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)]

    nx = len(grid_x_rel)
    anchor_ids: list[int] = []
    mount_points_rel: list[list[float]] = []
    for tx, ty in target_pts:
        ix = int(np.argmin(np.abs(grid_x_rel - tx)))
        iy = int(np.argmin(np.abs(grid_y_rel - ty)))
        nid = iy * nx + ix
        if nid not in anchor_ids:
            anchor_ids.append(nid)
            mount_points_rel.append([float(grid_x_rel[ix]), float(grid_y_rel[iy])])

    if arch == "delta":
        k_z_carrier = 2.0e7
    elif arch == "cartesian_bedslinger":
        k_z_carrier = 2.5e6
    else:
        k_z_carrier = _CARRIER_Z_STIFFNESS_N_M.get(z_drive, 1.6e6)

    return anchor_ids, mount_points_rel, k_z_carrier


def _apply_mount_springs(
    k_mat: np.ndarray,
    spec: PrinterSpec,
    anchor_ids: list[int],
    k_z_carrier: float,
    thickness_m: float,
) -> None:
    """Add grounded translational/rotational mount & Z-carrier springs to global K (in-place)."""
    cfg = BED_MOUNT_CATALOG.get(spec.bed.mount_type, BED_MOUNT_CATALOG["kinematic_3pt"])
    n_pts = float(cfg["mount_points"])
    k_z_mount = n_pts * float(cfg["k_axial_n_per_mm"]) * 1e3
    k_xy_mount = n_pts * float(cfg["k_lateral_n_per_mm"]) * 1e3
    k_z_eff_tot = (k_z_mount * k_z_carrier) / max(k_z_mount + k_z_carrier, 1e-9)

    n_anc = max(len(anchor_ids), 1)
    k_z_each = k_z_eff_tot / n_anc
    k_xy_each = k_xy_mount / n_anc

    k_pitch_each = 0.0
    if spec.architecture == "corexy" and spec.bed.z_drive == "leadscrew_single":
        k_pitch_each = (1200.0 * math.sqrt(max(thickness_m, 1e-4) / 0.008)) / n_anc
    elif spec.architecture == "corexy" and spec.bed.z_drive == "leadscrew_dual":
        k_pitch_each = 2200.0 / n_anc

    for nid in anchor_ids:
        base = 6 * nid
        k_mat[base + 0, base + 0] += k_xy_each
        k_mat[base + 1, base + 1] += k_xy_each
        k_mat[base + 2, base + 2] += k_z_each
        k_mat[base + 3, base + 3] += k_pitch_each + _ROT_REG_NM_RAD
        k_mat[base + 4, base + 4] += _ROT_REG_NM_RAD
        k_mat[base + 5, base + 5] += _ROT_REG_NM_RAD


def _compute_thermal_warp_field_um(
    spec: PrinterSpec,
    lx_m: float,
    ly_m: float,
    thickness_m: float,
    grid_x_rel: np.ndarray,
    grid_y_rel: np.ndarray,
    anchor_ids: list[int],
) -> tuple[np.ndarray, float, float]:
    """Compute 2D gradient dome + rigid-mount taco buckling field in micrometers.

    Matches the thermal bow and post-buckling elastica physics in
    ``printsim.fea.components.compute_thermal_structural_report``.
    """
    bed_mat = get_material(spec.bed.plate_material)
    frame_mat = get_material(spec.frame.material)
    bed_temp = float(spec.bed.target_temp_c)
    chamber_c = float(spec.frame.chamber_temp_c)

    l_bed = max(lx_m, ly_m)
    q_top = 12.0 * max(0.0, bed_temp - chamber_c)
    bed_grad_bow_um = (
        (l_bed**2)
        * bed_mat.thermal_expansion_1_k
        * q_top
        / (8.0 * max(bed_mat.thermal_conductivity_w_mk, 0.1))
        * 1e6
    )

    l_mount = 0.85 * l_bed
    dl_thermal = (
        bed_mat.thermal_expansion_1_k * max(0.0, bed_temp - 25.0)
        - frame_mat.thermal_expansion_1_k * max(0.0, chamber_c - 25.0)
    ) * l_mount
    dl_excess = max(0.0, dl_thermal - 0.00015)

    mount_cfg = BED_MOUNT_CATALOG.get(spec.bed.mount_type, BED_MOUNT_CATALOG["kinematic_3pt"])
    k_restraint = float(mount_cfg["k_restraint_thermal"])
    if k_restraint <= 0.0 or dl_excess <= 0.0:
        taco_buckling_um = 0.0
    else:
        d_plate = (
            bed_mat.youngs_modulus_pa
            * (thickness_m**3)
            / (12.0 * (1.0 - bed_mat.poisson_ratio**2))
        )
        k_plate_taco = 48.0 * d_plate * l_bed / max(l_mount**3, 1e-9)
        w_elastica = (2.0 / math.pi) * math.sqrt(l_mount * dl_excess)
        arch_fraction = k_restraint / (k_restraint + k_plate_taco)
        taco_buckling_um = w_elastica * math.sqrt(arch_fraction) * 1e6

    rx_nodes = np.array([grid_x_rel[i] for _ in grid_y_rel for i in range(len(grid_x_rel))], dtype=float)
    ry_nodes = np.array([ry for ry in grid_y_rel for _ in grid_x_rel], dtype=float)

    w_grad = bed_grad_bow_um * (1.0 - 2.0 * (rx_nodes**2 + ry_nodes**2))
    w_taco = taco_buckling_um * (
        np.cos(math.pi * rx_nodes) * np.cos(math.pi * ry_nodes)
        - 0.35 * (rx_nodes**2 - ry_nodes**2)
    )
    z_thermal = w_grad + w_taco
    z_thermal -= float(np.mean(z_thermal[anchor_ids]))
    thermal_ptv_um = float(np.max(z_thermal) - np.min(z_thermal))
    return z_thermal, thermal_ptv_um, float(taco_buckling_um)


def _solve_plate_statics(
    k_mat: np.ndarray,
    f_grav: np.ndarray,
    nx: int,
    ny: int,
    grid_x_rel: np.ndarray,
    grid_y_rel: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Solve gravity sag and 2.0 N center & worst-corner probe deflections."""
    n_dof = k_mat.shape[0]
    center_ix = int(np.argmin(np.abs(grid_x_rel)))
    center_iy = int(np.argmin(np.abs(grid_y_rel)))
    center_idx = center_iy * nx + center_ix

    # Check all 4 corners (front corners ordered first in case of exact tie)
    corner_ids = [
        nx - 1,  # (+0.5, -0.5) front-right
        0,  # (-0.5, -0.5) front-left
        ny * nx - 1,  # (+0.5, +0.5) rear-right
        (ny - 1) * nx,  # (-0.5, +0.5) rear-left
    ]

    rhs = np.zeros((n_dof, 2 + len(corner_ids)), dtype=float)
    rhs[:, 0] = f_grav
    rhs[6 * center_idx + 2, 1] = -_PROBE_FORCE_N
    for col, cid in enumerate(corner_ids, start=2):
        rhs[6 * cid + 2, col] = -_PROBE_FORCE_N

    try:
        cf = cho_factor(k_mat, lower=True, check_finite=False)
        sol = cho_solve(cf, rhs, check_finite=False)
    except np.linalg.LinAlgError:
        sol = solve(k_mat, rhs, assume_a="sym", check_finite=False)

    z_gravity_um = sol[2::6, 0] * 1e6
    probe_center_um = abs(float(sol[6 * center_idx + 2, 1])) * 1e6
    probe_corner_um = max(
        abs(float(sol[6 * cid + 2, col])) * 1e6 for col, cid in enumerate(corner_ids, start=2)
    )
    return z_gravity_um, probe_center_um, probe_corner_um


def _solve_plate_modes(
    k_mat: np.ndarray,
    m_mat: np.ndarray,
    spec: PrinterSpec,
    anchor_ids: list[int],
) -> tuple[list[float], np.ndarray, str]:
    """Extract first 3 out-of-plane plate eigenfrequencies, Mode 1 Z-shape, and classification."""
    n_dof = k_mat.shape[0]
    n_req = min(10, n_dof)
    m_diag = np.diag(m_mat)
    if float(np.min(m_diag)) < 1e-14:
        m_mat = m_mat.copy()
        np.fill_diagonal(m_mat, np.maximum(m_diag, 1e-14))

    evals, evecs = eigh(k_mat, m_mat, subset_by_index=[0, n_req - 1], check_finite=False)

    z_modes: list[tuple[float, np.ndarray]] = []
    fallback_modes: list[tuple[float, np.ndarray]] = []
    for idx in range(evals.size):
        lam = float(evals[idx])
        if lam <= (2.0 * math.pi * 1.0) ** 2:
            continue
        freq_hz = math.sqrt(lam) / (2.0 * math.pi)
        phi = evecs[:, idx]
        ux = phi[0::6]
        uy = phi[1::6]
        uz = phi[2::6]
        z_energy = float(np.dot(uz, uz))
        trans_energy = float(np.dot(ux, ux) + np.dot(uy, uy) + z_energy)
        fallback_modes.append((freq_hz, uz))
        if z_energy > 0.25 * max(trans_energy, 1e-30):
            z_modes.append((freq_hz, uz))

    selected = z_modes if len(z_modes) >= 3 else (z_modes + [m for m in fallback_modes if m not in z_modes])
    if not selected:
        selected = [(1.0, np.zeros(n_dof // 6, dtype=float))]

    plate_modes_hz = [float(f) for f, _ in selected[:3]]
    phi1_z = selected[0][1].copy()
    max_abs = float(np.max(np.abs(phi1_z)))
    if max_abs > 1e-12:
        mode1_shape_z = phi1_z / max_abs
        peak_i = int(np.argmax(np.abs(mode1_shape_z)))
        if mode1_shape_z[peak_i] < 0.0:
            mode1_shape_z = -mode1_shape_z
    else:
        mode1_shape_z = np.zeros_like(phi1_z)

    if spec.architecture == "corexy" and spec.bed.z_drive == "leadscrew_single":
        mode_type = "cantilever_flap"
    elif (
        spec.bed.mount_type in ("springs_4pt", "silicone_4pt")
        or float(np.min(np.abs(mode1_shape_z[anchor_ids]))) > 0.35
    ):
        mode_type = "mount_bounce"
    else:
        mode_type = "elastic_drum"

    return plate_modes_hz, mode1_shape_z, mode_type


def solve_bed_plate_fem(spec: PrinterSpec, mesh_n: int = 4) -> BedPlateFEAResult:
    """Solve 2D MITC4 heated bed plate static sag, thermal taco warp, probe flex, and modes."""
    n_elem = int(mesh_n)
    if n_elem < 2:
        raise ValueError(f"mesh_n must be >= 2, got {mesh_n}")

    nx = ny = n_elem + 1
    lx_m = max(float(spec.bed.size_mm[0]) * 1e-3, 0.05)
    ly_m = max(float(spec.bed.size_mm[1]) * 1e-3, 0.05)
    thickness_m = max(float(spec.bed.plate_thickness_mm) * 1e-3, 1e-4)
    bed_mat = get_material(spec.bed.plate_material)
    areal_mass = (float(spec.bed.mass_g) * 1e-3) / max(lx_m * ly_m, 1e-6)

    grid_x_rel = np.linspace(-0.5, 0.5, nx, dtype=float)
    grid_y_rel = np.linspace(-0.5, 0.5, ny, dtype=float)
    coords_3d = np.array(
        [[gx * lx_m, gy * ly_m, 0.0] for gy in grid_y_rel for gx in grid_x_rel],
        dtype=float,
    )

    n_nodes = nx * ny
    n_dof = 6 * n_nodes
    k_glob = np.zeros((n_dof, n_dof), dtype=float)
    m_glob = np.zeros((n_dof, n_dof), dtype=float)
    f_grav = np.zeros(n_dof, dtype=float)

    # All elements on the uniform Cartesian plate mesh are translated copies of quad (0, 0).
    ref_ids = (0, 1, nx + 1, nx)
    ke_ref, me_ref, fg_ref = mitc4_shell_matrices(
        coords_3d[list(ref_ids)],
        e_pa=bed_mat.youngs_modulus_pa,
        nu=bed_mat.poisson_ratio,
        thickness_m=thickness_m,
        areal_mass_kg_m2=areal_mass,
    )

    for ey in range(n_elem):
        row0 = ey * nx
        row1 = (ey + 1) * nx
        for ex in range(n_elem):
            n0 = row0 + ex
            n1 = n0 + 1
            n2 = row1 + ex + 1
            n3 = row1 + ex
            dofs = [6 * nid + d for nid in (n0, n1, n2, n3) for d in range(6)]
            k_glob[np.ix_(dofs, dofs)] += ke_ref
            m_glob[np.ix_(dofs, dofs)] += me_ref
            f_grav[dofs] += fg_ref

    anchor_ids, mount_points_rel, k_z_carrier = _resolve_mount_layout(
        spec, grid_x_rel, grid_y_rel
    )
    _apply_mount_springs(k_glob, spec, anchor_ids, k_z_carrier, thickness_m)
    k_glob = 0.5 * (k_glob + k_glob.T)
    m_glob = 0.5 * (m_glob + m_glob.T)

    z_gravity_um, probe_center_um, probe_corner_um = _solve_plate_statics(
        k_glob, f_grav, nx, ny, grid_x_rel, grid_y_rel
    )
    gravity_sag_um = float(np.max(np.abs(z_gravity_um)))

    z_thermal_um, thermal_ptv_um, taco_buckling_um = _compute_thermal_warp_field_um(
        spec, lx_m, ly_m, thickness_m, grid_x_rel, grid_y_rel, anchor_ids
    )

    mean_grav_at_anchors = float(np.mean(z_gravity_um[anchor_ids]))
    z_total_um = (z_gravity_um - mean_grav_at_anchors) + z_thermal_um
    total_flatness_error_um = float(np.max(z_total_um) - np.min(z_total_um))
    mesh_req = total_flatness_error_um > 50.0

    if total_flatness_error_um >= 180.0:
        risk = "critical"
    elif total_flatness_error_um >= 100.0:
        risk = "high"
    elif total_flatness_error_um >= 45.0:
        risk = "moderate"
    else:
        risk = "low"

    plate_modes_hz, mode1_shape_z, mode_type = _solve_plate_modes(
        k_glob, m_glob, spec, anchor_ids
    )
    first_mode_hz = plate_modes_hz[0]

    summary = (
        f"{spec.bed.size_mm[0]:.0f}x{spec.bed.size_mm[1]:.0f}x{spec.bed.plate_thickness_mm:.1f}mm "
        f"{spec.bed.plate_material} ({spec.bed.mount_type} / {spec.bed.z_drive}): "
        f"total flatness error {total_flatness_error_um:.1f} um ({risk} first-layer risk, "
        f"taco {taco_buckling_um:.1f} um, grav sag {gravity_sag_um:.1f} um), "
        f"2N probe {probe_center_um:.1f} um center / {probe_corner_um:.1f} um corner, "
        f"f1 = {first_mode_hz:.1f} Hz ({mode_type})."
    )

    return BedPlateFEAResult(
        grid_nx=nx,
        grid_ny=ny,
        grid_x_rel=grid_x_rel.tolist(),
        grid_y_rel=grid_y_rel.tolist(),
        mount_points_rel=mount_points_rel,
        z_gravity_um=z_gravity_um.tolist(),
        z_thermal_um=z_thermal_um.tolist(),
        z_total_um=z_total_um.tolist(),
        mode1_shape_z=mode1_shape_z.tolist(),
        gravity_sag_um=gravity_sag_um,
        thermal_peak_to_valley_um=thermal_ptv_um,
        taco_buckling_um=taco_buckling_um,
        total_flatness_error_um=total_flatness_error_um,
        probe_center_deflection_um=probe_center_um,
        probe_corner_deflection_um=probe_corner_um,
        plate_first_mode_hz=first_mode_hz,
        plate_first_mode_type=mode_type,
        plate_modes_hz=plate_modes_hz,
        mesh_bed_leveling_required=mesh_req,
        first_layer_risk=risk,
        summary=summary,
    )
