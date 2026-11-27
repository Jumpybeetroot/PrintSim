"""2D MITC4 heated bed plate FEM, mount compliance, thermal taco warp, and modal solver."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh, solve

from printsim.fea.components import compute_bed_thermal_bow_and_taco_um
from printsim.fea.shell_mitc4 import mitc4_shell_matrices
from printsim.schema import (
    PrinterSpec,
    get_material,
    resolve_bed_mount,
    resolve_z_drive_mechanism,
)

_PROBE_FORCE_N = 2.0
_ROT_REG_NM_RAD = 1e-2
_RIGID_FLANGE_CLAMP_NM_RAD = 40.0
_MOUNT_INSET_REL = 0.375
_CORNER_Z_DRIVE_COUNTS: dict[str, int] = {
    "belted_quad_z": 4,
    "leadscrew_triple": 3,
    "leadscrew_dual": 2,
    "leadscrew_dual_belt_sync": 2,
    "leadscrew_single": 1,
}

# CAL-08: Klipper BED_MESH_OUTPUT cold (25 C) vs 110 C calibrated thermal restraint stiffnesses (N/m)
# accounting for bolted-standoff frictional lock, slotted-tab sliding friction, and M3 through-bolt
# clearance take-up. Explicit user overrides in `spec.model_overrides.bed_mount` take precedence.
CAL_BED_MOUNT_K_THERM_N_PER_M: dict[str, float] = {
    "kinematic_3pt": 0.0,
    "whopping_pochard_kinematic": 0.0,
    "cnc_kelvin_magnetic": 0.0,
    "captive_belleville_kinematic": 0.0,
    "dual_leaf_ti_kinematic": 0.0,
    "craz_kinematic_ge5c": 0.0,
    "springs_4pt": 9_000.0,
    "silicone_4pt": 32_000.0,
    "silicone_4pt_clamped": 850_000.0,
    "spherical_3pt_ratrig": 1_500_000.0,
    "floating_4pt": 3_200_000.0,
    "trident_3pt_standoffs": 5_200_000.0,
    "rigid_4pt": 80_000_000.0,
}


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
    mount_settlement_um: float
    rigid_tilt_um: float
    plate_flatness_um: float
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
            "mount_settlement_um": round(float(self.mount_settlement_um), 2),
            "rigid_tilt_um": round(float(self.rigid_tilt_um), 2),
            "plate_flatness_um": round(float(self.plate_flatness_um), 2),
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


def _compute_carrier_z_stiffness(spec: PrinterSpec) -> float:
    """Compute effective Z-carrier vertical stiffness (N/m) including baked Z-mount CMS."""
    arch = spec.architecture
    z_drive = spec.bed.z_drive
    if arch == "delta":
        return 2.0e7
    if arch == "cartesian_bedslinger":
        return 2.5e6

    z_mech = resolve_z_drive_mechanism(spec)
    if z_drive == "belted_quad_z" and str(z_mech["kind"]) == "belt":
        # Flying-gantry CoreXY bed bolts directly to 4x 2020 base extrusions (belts lift gantry).
        return 2.0e7

    from printsim.geometry._builder_bed import _compute_corexy_z_drive_corner_stiffnesses

    h_m = max(float(spec.frame.height_mm) * 1e-3, 0.25)
    bz_m = max(float(spec.kinematics.build_volume_mm[2]) * 1e-3, 0.10)
    top_clear_m = min(
        float(spec.frame.gantry_top_clearance_mm) * 1e-3,
        max(h_m - 0.16, 0.025),
    )
    z_gantry = h_m - top_clear_m
    z_min = min(0.08, 0.5 * z_gantry)
    z_max = max(z_min + 0.01, z_gantry - 0.045)
    z_stroke = min(bz_m, max(z_gantry - 0.14, 0.02))
    z_bed = float(np.clip(z_gantry - 0.045 - 0.5 * z_stroke, z_min, z_max))
    trident_bb = bool(spec.frame.trident_backbone and z_drive == "leadscrew_triple")

    corner_k = _compute_corexy_z_drive_corner_stiffnesses(
        spec,
        z_gantry=z_gantry,
        z_bed=z_bed,
        is_flying=bool(spec.is_flying_gantry),
        trident_backbone_active=trident_bb,
    )
    if corner_k:
        return float(sum(corner_k.values()))
    n_corners = _CORNER_Z_DRIVE_COUNTS.get(z_drive, 3)
    return float(n_corners) * 1.8e5


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
    elif (
        int(resolve_bed_mount(spec)["mount_points"]) == 3
        or arch == "delta"
    ):
        target_pts = [
            (-_MOUNT_INSET_REL, -_MOUNT_INSET_REL),
            (_MOUNT_INSET_REL, -_MOUNT_INSET_REL),
            (0.0, _MOUNT_INSET_REL),
        ]
    else:
        target_pts = [
            (-_MOUNT_INSET_REL, -_MOUNT_INSET_REL),
            (_MOUNT_INSET_REL, -_MOUNT_INSET_REL),
            (-_MOUNT_INSET_REL, _MOUNT_INSET_REL),
            (_MOUNT_INSET_REL, _MOUNT_INSET_REL),
        ]

    nx = len(grid_x_rel)
    anchor_ids: list[int] = []
    mount_points_rel: list[list[float]] = []
    for tx, ty in target_pts:
        # Interior tie-breaker keeps coarse grids symmetric around 0 while exact 9x9 hits +-0.375 directly.
        ix = int(np.argmin(np.abs(grid_x_rel - tx) + 1e-12 * np.abs(grid_x_rel)))
        iy = int(np.argmin(np.abs(grid_y_rel - ty) + 1e-12 * np.abs(grid_y_rel)))
        nid = iy * nx + ix
        if nid not in anchor_ids:
            anchor_ids.append(nid)
            mount_points_rel.append([float(grid_x_rel[ix]), float(grid_y_rel[iy])])

    k_z_carrier = _compute_carrier_z_stiffness(spec)
    return anchor_ids, mount_points_rel, k_z_carrier


def _anchor_inplane_stiffness_2x2(
    idx: int,
    n_anc: int,
    rx: float,
    ry: float,
    lx_m: float,
    ly_m: float,
    k_xy_each: float,
    radial_mode: str,
    is_bracket_line: bool,
    rear_center_idx: int | None = None,
    first_front_idx: int | None = None,
) -> np.ndarray:
    """Build exact 2x2 rotated in-plane anchor stiffness K_xy = k_r r r^T + k_t t t^T (N/m)."""
    px = rx * lx_m
    py = ry * ly_m
    rnorm = math.hypot(px, py)
    directional_modes = (
        "maxwell_vgroove",
        "kelvin_cone_vee_flat",
        "radial_flexure",
        "slotted_1fixed_3float",
        "slotted_3pt_bakelite",
    )
    if is_bracket_line or rnorm <= 1e-9 or radial_mode not in directional_modes:
        return np.diag([k_xy_each, k_xy_each])

    r_hat = np.array([px / rnorm, py / rnorm], dtype=float)
    t_hat = np.array([-r_hat[1], r_hat[0]], dtype=float)
    if rear_center_idx is not None:
        is_rear_center = idx == rear_center_idx
    elif n_anc == 3:
        is_rear_center = (idx == n_anc - 1) or (ry > 0.0 and abs(rx) < 0.25)
    else:
        is_rear_center = idx == n_anc - 1
    vee_idx = first_front_idx if first_front_idx is not None else 0

    if radial_mode == "maxwell_vgroove":
        k_r, k_t = 500.0, k_xy_each
    elif radial_mode == "kelvin_cone_vee_flat":
        # Exact 6-DOF Kelvin constraint: rear-center cone pinned in X/Y, one front anchor V-groove, other flat.
        if is_rear_center:
            k_r, k_t = k_xy_each, k_xy_each
        elif idx == vee_idx:
            k_r, k_t = 500.0, k_xy_each
        else:
            k_r, k_t = 500.0, 500.0
    elif radial_mode == "radial_flexure":
        k_r, k_t = 5000.0, k_xy_each
    elif radial_mode == "slotted_1fixed_3float":
        k_r, k_t = (k_xy_each, k_xy_each) if idx == 0 else (0.08 * k_xy_each, k_xy_each)
    else:  # slotted_3pt_bakelite: rear-center anchor pinned, front two slotted
        k_r, k_t = (k_xy_each, k_xy_each) if is_rear_center else (0.15 * k_xy_each, k_xy_each)

    return k_r * np.outer(r_hat, r_hat) + k_t * np.outer(t_hat, t_hat)


def _apply_mount_springs(
    k_mat: np.ndarray,
    spec: PrinterSpec,
    anchor_ids: list[int],
    k_z_carrier: float,
    thickness_m: float,
    mount_points_rel: list[list[float]] | None = None,
    lx_m: float | None = None,
    ly_m: float | None = None,
) -> None:
    """Add grounded directional/rotational mount & Z-carrier springs to global K (in-place)."""
    cfg = resolve_bed_mount(spec)
    radial_mode = str(cfg.get("radial_dof_mode", "maxwell_vgroove"))
    temp_soft = (
        max(0.75, 1.0 - 0.004 * max(0.0, float(spec.bed.target_temp_c) - 70.0))
        if radial_mode == "elastomer_shear"
        else 1.0
    )

    n_pts = float(cfg["mount_points"])
    k_z_mount = n_pts * float(cfg["k_axial_n_per_mm"]) * 1e3 * temp_soft
    k_xy_mount = n_pts * float(cfg["k_lateral_n_per_mm"]) * 1e3 * temp_soft
    k_z_eff_tot = (k_z_mount * k_z_carrier) / max(k_z_mount + k_z_carrier, 1e-9)

    n_anc = max(len(anchor_ids), 1)
    k_z_each = k_z_eff_tot / n_anc
    k_xy_each = k_xy_mount / n_anc

    k_pitch_each = 0.0
    if spec.architecture == "corexy" and spec.bed.z_drive == "leadscrew_single":
        k_pitch_each = (1200.0 * math.sqrt(max(thickness_m, 1e-4) / 0.008)) / n_anc
    elif spec.architecture == "corexy" and spec.bed.z_drive == "leadscrew_dual":
        k_pitch_each = 2200.0 / n_anc

    bed_lx_m = lx_m if lx_m is not None else max(float(spec.bed.size_mm[0]) * 1e-3, 0.05)
    bed_ly_m = ly_m if ly_m is not None else max(float(spec.bed.size_mm[1]) * 1e-3, 0.05)
    if mount_points_rel is None:
        n_nodes = k_mat.shape[0] // 6
        nx = max(int(round(math.sqrt(n_nodes))), 2)
        gx = np.linspace(-0.5, 0.5, nx, dtype=float)
        gy = np.linspace(-0.5, 0.5, nx, dtype=float)
        rel_pts = [[float(gx[nid % nx]), float(gy[nid // nx])] for nid in anchor_ids]
    else:
        rel_pts = mount_points_rel

    is_bracket_line = spec.architecture == "corexy" and spec.bed.z_drive in (
        "leadscrew_single",
        "leadscrew_dual",
    )
    k_clamp_local = _RIGID_FLANGE_CLAMP_NM_RAD if radial_mode == "rigid_all" else 0.0

    rear_center_idx: int | None = None
    first_front_idx: int | None = None
    if n_anc == 3 and len(rel_pts) == 3:
        rear_center_idx = min(
            range(len(rel_pts)),
            key=lambda k: (rel_pts[k][0] ** 2) + ((rel_pts[k][1] - _MOUNT_INSET_REL) ** 2),
        )
        front_indices = [k for k in range(len(rel_pts)) if k != rear_center_idx]
        first_front_idx = front_indices[0] if front_indices else 0

    for idx, (nid, (rx, ry)) in enumerate(zip(anchor_ids, rel_pts)):
        base = 6 * nid
        k_xy_2x2 = _anchor_inplane_stiffness_2x2(
            idx=idx,
            n_anc=n_anc,
            rx=rx,
            ry=ry,
            lx_m=bed_lx_m,
            ly_m=bed_ly_m,
            k_xy_each=k_xy_each,
            radial_mode=radial_mode,
            is_bracket_line=is_bracket_line,
            rear_center_idx=rear_center_idx,
            first_front_idx=first_front_idx,
        )
        k_mat[base : base + 2, base : base + 2] += k_xy_2x2
        k_mat[base + 2, base + 2] += k_z_each
        k_mat[base + 3, base + 3] += k_pitch_each + k_clamp_local + _ROT_REG_NM_RAD
        k_mat[base + 4, base + 4] += k_clamp_local + _ROT_REG_NM_RAD
        k_mat[base + 5, base + 5] += (0.5 * k_clamp_local if k_clamp_local > 0.0 else 0.0) + _ROT_REG_NM_RAD


def compute_bed_bimetallic_pei_bow_um(spec: PrinterSpec) -> float:
    """Compute Timoshenko + Volkersen shear-lag bimetallic bow (um) of bed plate + magnetic PEI stack.

    Models a 0.40 mm spring-steel PEI flex sheet (alpha = 11.5 ppm/K, E = 200 GPa) coupled
    through a 2.0 mm ferrite-loaded CPE magnetic rubber sheet (G_mag(25 C) = 0.45 MPa,
    thermal softening above 25 C) to the heated tooling plate. Returns the center-to-edge
    parabolic bow amplitude in micrometers (zero when T_bed <= 25 C; mount-independent).
    """
    dt_bed = max(0.0, float(spec.bed.target_temp_c) - 25.0)
    if dt_bed <= 0.0:
        return 0.0
    bed_mat = get_material(spec.bed.plate_material)
    alpha_steel = 11.5e-6
    delta_alpha = float(bed_mat.thermal_expansion_1_k) - alpha_steel
    if abs(delta_alpha) <= 1e-12:
        return 0.0

    lx_m = max(float(spec.bed.size_mm[0]) * 1e-3, 0.05)
    ly_m = max(float(spec.bed.size_mm[1]) * 1e-3, 0.05)
    l_bed = max(lx_m, ly_m)
    t_a = max(float(spec.bed.plate_thickness_mm) * 1e-3, 1e-4)
    e_a = float(bed_mat.youngs_modulus_pa)
    nu_a = float(bed_mat.poisson_ratio)

    t_s = 0.00035  # 0.35 mm spring steel PEI flex sheet
    e_s = 200.0e9
    h_mag = 0.0020  # 2.0 mm ferrite-loaded CPE magnetic rubber sheet + 3M 468MP adhesive
    g_mag = 0.36e6 / (1.0 + 0.012 * dt_bed)

    inv_ea = 1.0 / (e_s * t_s) + 1.0 / (e_a * t_a)
    beta = math.sqrt((g_mag / h_mag) * inv_ea)
    beta_half = beta * (0.5 * l_bed)
    gamma_mag = (
        1.0 - math.tanh(beta_half) / beta_half
        if beta_half > 1e-4
        else (beta_half**2) / 3.0
    )

    f_s = gamma_mag * (delta_alpha * dt_bed) / inv_ea
    c_arm = 0.5 * t_a + h_mag + 0.5 * t_s
    m_b = f_s * c_arm
    d_plate = e_a * (t_a**3) / (12.0 * (1.0 - nu_a**2))
    kappa_b = m_b / (d_plate * (1.0 + nu_a))
    return float(kappa_b * (l_bed**2) / 8.0 * 1e6)


def _resolve_calibrated_k_therm_n_per_m(spec: PrinterSpec) -> float:
    """Resolve calibrated bed-mount thermal restraint stiffness (N/m), honoring explicit overrides."""
    ov = getattr(spec.model_overrides, "bed_mount", None)
    if ov is not None and getattr(ov, "k_restraint_thermal_n_per_mm", None) is not None:
        return max(0.0, float(ov.k_restraint_thermal_n_per_mm) * 1e3)
    if spec.bed.mount_type in CAL_BED_MOUNT_K_THERM_N_PER_M:
        return float(CAL_BED_MOUNT_K_THERM_N_PER_M[spec.bed.mount_type])
    return max(0.0, float(resolve_bed_mount(spec)["k_restraint_thermal_n_per_m"]))


def _build_eccentric_thermal_restraint_rhs(
    spec: PrinterSpec,
    n_dof: int,
    lx_m: float,
    ly_m: float,
    thickness_m: float,
    anchor_ids: list[int],
    mount_points_rel: list[list[float]],
) -> np.ndarray:
    """Build nodal bending-moment RHS (N*m) from eccentric radial mount thermal restraint.

    When a non-kinematic bed mount resists radial thermal expansion (``Delta L > clearance``),
    the mount shear force ``F_r`` (bounded by physical M3/M4 bolt-preload friction slip) acts
    at the plate bottom face eccentricity ``e_z = t_bed / 2``, inducing radial bending moments
    at the MITC4 anchors aligned with the plate thermal curvature mode.
    """
    rhs = np.zeros(n_dof, dtype=float)
    k_therm = _resolve_calibrated_k_therm_n_per_m(spec)
    if k_therm <= 0.0 or not anchor_ids:
        return rhs

    bed_temp = float(spec.bed.target_temp_c)
    chamber_c = float(spec.frame.chamber_temp_c)
    bed_mat = get_material(spec.bed.plate_material)
    frame_mat = get_material(spec.frame.material)
    mount_cfg = resolve_bed_mount(spec)
    l_bed = max(lx_m, ly_m)
    mount_inset_ratio = float(mount_cfg.get("mount_inset_ratio", 0.38))
    l_mount = 2.0 * mount_inset_ratio * l_bed
    dl_thermal = (
        bed_mat.thermal_expansion_1_k * max(0.0, bed_temp - 25.0)
        - frame_mat.thermal_expansion_1_k * max(0.0, chamber_c - 25.0)
    ) * l_mount
    dl_excess = max(0.0, dl_thermal - 0.00015)
    if dl_excess <= 0.0:
        return rhs

    k_plate_axial = bed_mat.youngs_modulus_pa * l_bed * thickness_m / max(l_mount, 1e-6)
    k_ser = (k_therm * k_plate_axial) / max(k_therm + k_plate_axial, 1e-9)
    n_anc = max(len(anchor_ids), 1)
    # Physical bolt-preload friction slip limit per anchor (post-buckling lives in w_taco):
    # slotted 4-pt tabs slip earlier than phenolic Bakelite standoffs or hard-bolted M4 rigid_4pt.
    radial_mode = str(mount_cfg.get("radial_dof_mode", ""))
    if radial_mode == "rigid_all":
        f_slip_per_anchor = 420.0
    elif radial_mode == "slotted_3pt_bakelite":
        f_slip_per_anchor = 340.0
    elif radial_mode == "slotted_1fixed_3float":
        f_slip_per_anchor = 185.0
    else:
        f_slip_per_anchor = 240.0
    f_r_per_anchor = min((k_ser * dl_excess) / n_anc, f_slip_per_anchor)

    # Bottom-face shear bearing eccentricity e_z = t_bed / 2 (plus slight standoff shoulder clamping)
    h_standoff_m = float(mount_cfg.get("standoff_height_mm", 12.0)) * 1e-3
    e_z = 0.5 * thickness_m + 0.25 * h_standoff_m
    m_per_anchor = f_r_per_anchor * e_z

    for nid, (rx, ry) in zip(anchor_ids, mount_points_rel):
        px = float(rx) * lx_m
        py = float(ry) * ly_m
        rnorm = math.hypot(px, py)
        if rnorm <= 1e-9:
            continue
        r_x = px / rnorm
        r_y = py / rnorm
        # MITC4 right-hand-rule convention: theta_x = +dw/dy, theta_y = -dw/dx.
        # Concave-up radial edge curl (dw/dx ~ +r_x, dw/dy ~ +r_y) aligns with w_grad dome.
        base = 6 * nid
        rhs[base + 3] += m_per_anchor * r_y
        rhs[base + 4] -= m_per_anchor * r_x
    return rhs


def _compute_thermal_warp_field_um(
    spec: PrinterSpec,
    lx_m: float,
    ly_m: float,
    thickness_m: float,
    grid_x_rel: np.ndarray,
    grid_y_rel: np.ndarray,
    anchor_ids: list[int],
    z_ecc_restraint_um: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    """Compute 2D gradient + bimetallic PEI dome + eccentric restraint + taco buckling field (um).

    Matches ``taco_buckling_um`` identically to
    ``printsim.fea.components.compute_bed_thermal_bow_and_taco_um`` while adding
    Volkersen/Timoshenko magnetic PEI bimetallic curvature and MITC4 eccentric
    thermal-restraint bending calibrated to Klipper ``BED_MESH_OUTPUT`` cold-vs-110 C deltas.
    """
    del thickness_m
    l_bed = max(lx_m, ly_m)
    bed_grad_bow_um, taco_buckling_um = compute_bed_thermal_bow_and_taco_um(spec)
    pei_bimetal_bow_um = compute_bed_bimetallic_pei_bow_um(spec)
    total_dome_bow_um = bed_grad_bow_um + pei_bimetal_bow_um

    rx_nodes = np.array([grid_x_rel[i] for _ in grid_y_rel for i in range(len(grid_x_rel))], dtype=float)
    ry_nodes = np.array([ry for ry in grid_y_rel for _ in grid_x_rel], dtype=float)

    # Bottom-heated plate + lower-CTE top PEI spring-steel sheet both expand bottom > top,
    # curving concave-up (edges/corners curl upward relative to center).
    w_grad = 4.0 * total_dome_bow_um * (
        ((lx_m / l_bed) * rx_nodes) ** 2 + ((ly_m / l_bed) * ry_nodes) ** 2
    )
    # Post-buckling elastica mode snaps in the direction of the pre-existing concave-up thermal
    # dome & eccentric anchor moments (center dips relative to edges + saddle asymmetry).
    w_taco = -taco_buckling_um * (
        np.cos(math.pi * rx_nodes) * np.cos(math.pi * ry_nodes)
        - 0.35 * (rx_nodes**2 - ry_nodes**2)
    )
    w_ecc = (
        np.asarray(z_ecc_restraint_um, dtype=float)
        if z_ecc_restraint_um is not None
        else np.zeros_like(w_grad)
    )
    z_thermal = w_grad + w_ecc + w_taco
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
    f_thermal_ecc: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Solve gravity sag, 2.0 N center & worst-corner probe deflections, and eccentric thermal warp."""
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

    has_ecc = f_thermal_ecc is not None and bool(np.any(f_thermal_ecc != 0.0))
    n_cols = 2 + len(corner_ids) + (1 if has_ecc else 0)
    rhs = np.zeros((n_dof, n_cols), dtype=float)
    rhs[:, 0] = f_grav
    rhs[6 * center_idx + 2, 1] = -_PROBE_FORCE_N
    for col, cid in enumerate(corner_ids, start=2):
        rhs[6 * cid + 2, col] = -_PROBE_FORCE_N
    if has_ecc and f_thermal_ecc is not None:
        rhs[:, -1] = f_thermal_ecc

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
    z_ecc_um = (sol[2::6, -1] * 1e6) if has_ecc else np.zeros(nx * ny, dtype=float)
    return z_gravity_um, probe_center_um, probe_corner_um, z_ecc_um


def _solve_plate_modes(
    k_mat: np.ndarray,
    m_mat: np.ndarray,
    spec: PrinterSpec,
    anchor_ids: list[int],
) -> tuple[list[float], np.ndarray, str]:
    """Extract first 3 out-of-plane plate eigenfrequencies, Mode 1 Z-shape, and classification."""
    n_dof = k_mat.shape[0]
    n_req = min(12, n_dof)
    m_diag = np.diag(m_mat)
    if float(np.min(m_diag)) < 1e-14:
        m_mat = m_mat.copy()
        np.fill_diagonal(m_mat, np.maximum(m_diag, 1e-14))

    evals, evecs = eigh(k_mat, m_mat, subset_by_index=[0, n_req - 1], check_finite=False)

    z_modes: list[tuple[float, np.ndarray]] = []
    z_indices: set[int] = set()
    fallback_modes: list[tuple[int, float, np.ndarray]] = []
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
        fallback_modes.append((idx, freq_hz, uz))
        if z_energy > 0.25 * max(trans_energy, 1e-30):
            z_modes.append((freq_hz, uz))
            z_indices.add(idx)

    selected = (
        z_modes
        if len(z_modes) >= 3
        else (z_modes + [(f, uz) for idx, f, uz in fallback_modes if idx not in z_indices])
    )
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


def solve_bed_plate_fem(spec: PrinterSpec, mesh_n: int = 8) -> BedPlateFEAResult:
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
    _apply_mount_springs(
        k_glob,
        spec,
        anchor_ids,
        k_z_carrier,
        thickness_m,
        mount_points_rel=mount_points_rel,
        lx_m=lx_m,
        ly_m=ly_m,
    )
    k_glob = 0.5 * (k_glob + k_glob.T)
    m_glob = 0.5 * (m_glob + m_glob.T)

    f_thermal_ecc = _build_eccentric_thermal_restraint_rhs(
        spec, n_dof, lx_m, ly_m, thickness_m, anchor_ids, mount_points_rel
    )
    z_gravity_um, probe_center_um, probe_corner_um, z_ecc_um = _solve_plate_statics(
        k_glob, f_grav, nx, ny, grid_x_rel, grid_y_rel, f_thermal_ecc=f_thermal_ecc
    )
    gravity_sag_um = float(np.max(np.abs(z_gravity_um)))

    z_thermal_um, thermal_ptv_um, taco_buckling_um = _compute_thermal_warp_field_um(
        spec,
        lx_m,
        ly_m,
        thickness_m,
        grid_x_rel,
        grid_y_rel,
        anchor_ids,
        z_ecc_restraint_um=z_ecc_um,
    )

    mean_grav_at_anchors = float(np.mean(z_gravity_um[anchor_ids]))
    mount_settlement_um = abs(mean_grav_at_anchors)
    z_total_um = (z_gravity_um - mean_grav_at_anchors) + z_thermal_um
    total_flatness_error_um = float(np.max(z_total_um) - np.min(z_total_um))

    # Decompose static deformation via least-squares best-fit rigid plane:
    # - mount_settlement_um: uniform vertical settlement at mounts (removed by Z-homing)
    # - rigid_tilt_um: peak-to-valley of best-fit rigid plane (removed by tramming / QGL / Z-tilt)
    # - plate_flatness_um: peak-to-valley non-planar residual requiring mesh bed leveling
    rx_all = np.array([gx for _ in grid_y_rel for gx in grid_x_rel], dtype=float)
    ry_all = np.array([gy for gy in grid_y_rel for _ in grid_x_rel], dtype=float)
    plane_basis = np.column_stack([np.ones_like(rx_all), rx_all, ry_all])
    plane_coeffs, _, _, _ = np.linalg.lstsq(plane_basis, z_total_um, rcond=None)
    z_tilt_only_um = plane_coeffs[1] * rx_all + plane_coeffs[2] * ry_all
    rigid_tilt_um = float(np.max(z_tilt_only_um) - np.min(z_tilt_only_um))
    z_flatness_residual_um = z_total_um - (plane_basis @ plane_coeffs)
    plate_flatness_um = float(
        np.max(z_flatness_residual_um) - np.min(z_flatness_residual_um)
    )

    mesh_req = plate_flatness_um > 50.0

    if plate_flatness_um >= 180.0:
        risk = "critical"
    elif plate_flatness_um >= 100.0:
        risk = "high"
    elif plate_flatness_um >= 45.0:
        risk = "moderate"
    else:
        risk = "low"

    plate_modes_hz, mode1_shape_z, mode_type = _solve_plate_modes(
        k_glob, m_glob, spec, anchor_ids
    )
    first_mode_hz = plate_modes_hz[0]

    bed_limit_note = ""
    if plate_flatness_um >= 50.0:
        if taco_buckling_um >= 30.0:
            bed_limit_note = " Main limit: over-constrained thermal expansion causing bed taco buckling."
        else:
            bed_limit_note = " Main limit: bed plate flexural rigidity against non-planar gravity sag and thermal bow."
    elif rigid_tilt_um >= 120.0:
        bed_limit_note = " Main limit: cantilevered single-Z bed support compliance causing front-edge tilt/droop."
    elif first_mode_hz < 60.0:
        bed_limit_note = (
            " Main limit: cantilevered Z support compliance."
            if mode_type == "cantilever_flap"
            else (
                " Main limit: compliant bed-mount vertical spring stiffness."
                if mode_type == "mount_bounce"
                else " Main limit: bed plate flexural stiffness relative to bed mass and support span."
            )
        )

    summary = (
        f"{spec.bed.size_mm[0]:.0f}x{spec.bed.size_mm[1]:.0f}x{spec.bed.plate_thickness_mm:.1f}mm "
        f"{spec.bed.plate_material} ({spec.bed.mount_type} / {spec.bed.z_drive}): "
        f"plate flatness {plate_flatness_um:.1f} um ({risk} first-layer risk; "
        f"rigid tilt {rigid_tilt_um:.1f} um, mount settlement {mount_settlement_um:.1f} um, "
        f"taco {taco_buckling_um:.1f} um), "
        f"2N probe {probe_center_um:.1f} um center / {probe_corner_um:.1f} um corner, "
        f"f1 = {first_mode_hz:.1f} Hz ({mode_type}).{bed_limit_note}"
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
        mount_settlement_um=mount_settlement_um,
        rigid_tilt_um=rigid_tilt_um,
        plate_flatness_um=plate_flatness_um,
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


def compute_nozzle_tap_z_deflection_um(
    spec: PrinterSpec,
    tap_force_n: float = 6.0,
) -> dict[str, float]:
    """Compute nozzle-tap (Voron Tap / load-cell) vertical deflection & Z-backlash budget (um).

    Evaluates elastic vertical deflection under a downward nozzle tap load ``tap_force_n``
    (default 6.0 N ~ 610 gf, standard Voron Tap 500-800 gf actuation band), combining:
    - Bed plate + bed mount + moving-bed Z-drive elastic deflection (from MITC4 plate FEM).
    - On flying-gantry CoreXY (``belted_quad_z``), adds the upward reaction compliance of the
      4 corner Z-belts + baked ``voron_24_z_drive_*`` CAD pods lifting the gantry.
    - Checks gravity preload against leadscrew/nut or Z-belt backlash deadband: under downward
      tap on a gravity-loaded moving bed, POM nut threads remain seated (``active_tap_backlash_um = 0``)
      while reversal / upward Z-hop exposes ``reversal_backlash_um``.
    """
    f_tap = max(float(tap_force_n), 0.1)
    res = solve_bed_plate_fem(spec)
    scale = f_tap / _PROBE_FORCE_N
    bed_center_um = res.probe_center_deflection_um * scale
    bed_corner_um = res.probe_corner_deflection_um * scale

    z_mech = resolve_z_drive_mechanism(spec)
    reversal_backlash_um = float(z_mech.get("backlash_um", 0.0))

    gantry_z_belt_um = 0.0
    if spec.is_flying_gantry:
        from printsim.geometry._builder_bed import _compute_corexy_z_drive_corner_stiffnesses

        h_m = max(float(spec.frame.height_mm) * 1e-3, 0.25)
        z_g = min(0.14, 0.5 * h_m)
        corner_k = _compute_corexy_z_drive_corner_stiffnesses(
            spec,
            z_gantry=z_g,
            z_bed=0.08,
            is_flying=True,
            trident_backbone_active=False,
        )
        k_z_gantry_tot = max(sum(corner_k.values()), 1.0)
        gantry_z_belt_um = (f_tap / k_z_gantry_tot) * 1e6
        # Flying gantry weighs ~2.0-2.8 kg (20-27 N >> 6 N tap), keeping Z-belts in tension
        gravity_preload_n = 22.0
    else:
        gravity_preload_n = max(float(spec.bed.mass_g) * 1e-3 * 9.80665, 0.0)

    # Downward nozzle tap acts WITH gravity on moving beds (threads stay seated) and against
    # gravity on flying gantries (partially unloads belts, zero slack while f_tap < gravity_preload).
    active_tap_backlash_um = (
        0.0
        if (not spec.is_flying_gantry and gravity_preload_n >= 5.0)
        or (spec.is_flying_gantry and f_tap < 0.85 * gravity_preload_n)
        else reversal_backlash_um
    )
    total_center_um = bed_center_um + gantry_z_belt_um + active_tap_backlash_um
    total_corner_um = bed_corner_um + gantry_z_belt_um + active_tap_backlash_um
    return {
        "tap_force_n": float(f_tap),
        "bed_center_elastic_um": float(bed_center_um),
        "bed_corner_elastic_um": float(bed_corner_um),
        "gantry_z_drive_elastic_um": float(gantry_z_belt_um),
        "active_tap_backlash_um": float(active_tap_backlash_um),
        "reversal_backlash_um": float(reversal_backlash_um),
        "total_center_tap_deflection_um": float(total_center_um),
        "total_corner_tap_deflection_um": float(total_corner_um),
        "gravity_preload_n": float(gravity_preload_n),
    }
