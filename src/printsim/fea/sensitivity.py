"""Exact Analytical Adjoint Static & Fox-Kapoor Modal Sensitivity Engine.

Computes closed-form analytical derivatives (zero finite-difference loops) across all
beam members, MITC4 shell panels, CBUSH fastener springs, and lumped masses:
- Static adjoint compliance sensitivities via a single back-substitution of ``K_ff``
  for nozzle-to-bed relative displacement ``u_rel`` in X and Y.
- Fox-Kapoor (1968) modal frequency sensitivities ``df_k / d(ln E_e)`` and
  ``df_k / d(ln m_e)`` with Dailey (1989) repeated-root subspace protection.
- Uniform thickening ROI metrics ``hz_per_100g``, ``hz_per_usd``, ``compliance_share_pct``,
  normalized ``heat_score`` in ``[-1, +1]``, bay-grouped panel sensitivities, and top
  stiffening / weight-reduction hotspots.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, eigsh, splu

from printsim.fea.beam3d import (
    GlobalSystem,
    assemble_global_system,
    node_rotational_inertia_xyz,
)
from printsim.fea.modal import ModalResult
from printsim.fea.solver import (
    StaticFEAResult,
    build_inertial_load_vectors,
    classify_part_group,
    shared_factorized_solver,
)
from printsim.geometry.builder import AssemblyModel, StructuralMember
from printsim.schema import PrinterSpec, get_material, resolve_extrusion_filling


@dataclass
class SensitivityReport:
    """Analytical structural sensitivity and ROI breakdown for an AssemblyModel."""

    fundamental_freq_hz: float
    primary_x_freq_hz: float
    primary_y_freq_hz: float
    static_rel_disp_x_um: float
    static_rel_disp_y_um: float
    member_sensitivities: dict[str, dict[str, Any]] = field(default_factory=dict)
    panel_bay_sensitivities: dict[str, dict[str, Any]] = field(default_factory=dict)
    lumped_mass_sensitivities: dict[str, dict[str, Any]] = field(default_factory=dict)
    top_stiffen_targets: list[dict[str, Any]] = field(default_factory=list)
    top_lighten_targets: list[dict[str, Any]] = field(default_factory=list)
    subsystem_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)
    primary_problem_subsystem: str = "frame"
    summary_diagnosis: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "fundamental_freq_hz": round(float(self.fundamental_freq_hz), 2),
            "primary_x_freq_hz": round(float(self.primary_x_freq_hz), 2),
            "primary_y_freq_hz": round(float(self.primary_y_freq_hz), 2),
            "static_rel_disp_x_um": round(float(self.static_rel_disp_x_um), 3),
            "static_rel_disp_y_um": round(float(self.static_rel_disp_y_um), 3),
            "member_sensitivities": self.member_sensitivities,
            "panel_bay_sensitivities": self.panel_bay_sensitivities,
            "lumped_mass_sensitivities": self.lumped_mass_sensitivities,
            "top_stiffen_targets": self.top_stiffen_targets,
            "top_lighten_targets": self.top_lighten_targets,
            "subsystem_breakdown": self.subsystem_breakdown,
            "primary_problem_subsystem": self.primary_problem_subsystem,
            "summary_diagnosis": self.summary_diagnosis,
        }


def _classify_sensitivity_group(
    member_id: str, role: str = "", bay: str = ""
) -> tuple[str, str]:
    """Local grouping helper extending ``classify_part_group`` with feet and Z gantry joints."""
    mid = member_id.strip()
    if mid.startswith("feet_") or role == "foot" or bay == "feet":
        return "frame_feet", "Machine Feet & Benchtop Pads"
    if bay == "z_joint" or mid.startswith("z_joint"):
        return "z_gantry_joints", "Z-Axis Gantry Joints"
    if bay in ("bed_z_drive", "z_guide") or mid.startswith(("z_bed_drive_", "z_rod_brg_")):
        return "bed_support", "Bed Support & Z Arms"
    if bay == "top_brace" or mid.startswith(("bay_top_brace", "top_brace")):
        return "top_brace", "Top Open-Center Brace"
    if bay == "front_brace" or mid.startswith(("bay_front_brace", "front_brace")):
        return "front_brace", "Front Open-Door Brace"
    return classify_part_group(member_id, role)


def _axis_comp_share(du: float, u_rel: float) -> float:
    """Signed static adjoint compliance share max(0, -du / u_rel), guarded for tiny u_rel."""
    if abs(u_rel) < 1e-12:
        return 0.0
    return max(0.0, -du / u_rel)


def _solve_eigen_modes(
    K_ff: np.ndarray,
    M_ff: np.ndarray,
    target_scan: int,
    system: GlobalSystem | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve generalized eigenproblem K_ff phi = lam M_ff phi with M-orthonormalization.

    With a shared ``system``, eigenpairs already published by ``solve_modal_fea`` for the same
    solver path and mode count (same ``K_ff``/``M_ff``, same ARPACK start vector, same
    normalization) are reused verbatim; otherwise the solve runs here unchanged.
    """
    n_free = K_ff.shape[0]
    k_scan = min(target_scan, n_free)
    want_sparse = n_free > 500 and 1 <= k_scan < n_free - 1
    if system is not None:
        shared = system.peek(("eigen", "sparse" if want_sparse else "dense", int(k_scan)))
        if shared is not None:
            return shared[0].copy(), shared[1].copy()
    used_sparse = False
    if want_sparse:
        try:
            k_csc = system.reduced_stiffness_csc() if system is not None else csc_matrix(K_ff)
            m_csc = csc_matrix(M_ff)
            lu = system.reduced_stiffness_lu() if system is not None else splu(k_csc)
            op = LinearOperator(K_ff.shape, matvec=lu.solve, dtype=float)
            v0 = np.random.default_rng(0).standard_normal(n_free)
            eigvals, eigvecs = eigsh(
                k_csc,
                k=k_scan,
                M=m_csc,
                sigma=0.0,
                which="LM",
                OPinv=op,
                v0=v0,
            )
            order = np.argsort(eigvals)
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            used_sparse = True
        except RuntimeError:
            used_sparse = False

    if not used_sparse:
        try:
            # Invert to M_ff phi = mu K_ff phi so eigh Cholesky-factorizes well-conditioned SPD K_ff
            mu, eigvecs = eigh(
                M_ff,
                K_ff,
                subset_by_index=[n_free - k_scan, n_free - 1],
            )
            mu = mu[::-1]
            eigvecs = eigvecs[:, ::-1].copy()
            eigvals = 1.0 / np.maximum(mu, 1e-30)
        except np.linalg.LinAlgError:
            # Clean fallback if K_ff is singular/semidefinite (e.g. unconstrained mechanism)
            eigvals, eigvecs = eigh(K_ff, M_ff, subset_by_index=[0, k_scan - 1])

    eigvals = np.maximum(eigvals, 0.0)
    for j in range(eigvecs.shape[1]):
        v = eigvecs[:, j]
        m_gen = float(v @ M_ff @ v)
        if m_gen > 1e-24:
            v = v / math.sqrt(m_gen)
        max_abs = float(np.max(np.abs(v)))
        if max_abs > 0.0:
            lead_idx = int(np.flatnonzero(np.abs(v) >= 0.99 * max_abs)[0])
            if v[lead_idx] < 0.0:
                v = -v
        eigvecs[:, j] = v
    return eigvals, eigvecs


def _dailey_subspace_vector(
    eigvals: np.ndarray,
    full_vecs: np.ndarray,
    target_idx: int,
    exc_vec: np.ndarray,
    rel_tol: float = 1e-4,
) -> np.ndarray:
    """Dailey (1989) repeated-root guard returning an M-normalized subspace mode vector."""
    f_all = np.sqrt(np.maximum(eigvals, 0.0)) / (2.0 * math.pi)
    f_k = float(f_all[target_idx])
    cluster = [
        j
        for j in range(len(f_all))
        if abs(float(f_all[j]) - f_k) / max(f_k, 1.0) < rel_tol
    ]
    if len(cluster) <= 1:
        return full_vecs[:, target_idx]

    Phi_R = full_vecs[:, cluster]
    w_proj = Phi_R.T @ exc_vec
    norm_w = float(np.linalg.norm(w_proj))
    if norm_w > 1e-12:
        w_proj = w_proj / norm_w
        return Phi_R @ w_proj
    return full_vecs[:, target_idx]


def _member_mass_and_cost(
    member: StructuralMember,
    length_m: float,
    assembly: AssemblyModel,
) -> tuple[float, float, float, float]:
    """Return ``(exact_mass_kg, ref_mass_kg, cost_usd, cost_per_kg_usd)`` for a member."""
    m_exact = member.profile.area_m2 * length_m * member.material.density_kg_m3
    spec = assembly.spec
    if member.material.density_kg_m3 >= 1.0:
        rail_mat = get_material(spec.gantry.rail_material)
        base_mk = max(
            m_exact - member.fill_mass_kg - member.rail_mass_kg - member.backer_mass_kg,
            0.0,
        )
        cost = base_mk * member.material.cost_per_kg_usd
        if member.rail_mass_kg > 0.0:
            cost += member.rail_mass_kg * rail_mat.cost_per_kg_usd
        if member.backer_mass_kg > 0.0 and member.backer_material_id:
            cost += member.backer_mass_kg * get_material(member.backer_material_id).cost_per_kg_usd
        if member.fill_mass_kg > 0.0:
            fill_spec = resolve_extrusion_filling(spec)
            cost += member.fill_mass_kg * float(fill_spec["cost_per_kg_usd"])
        eff_cost_kg = cost / max(m_exact, 1e-6) if cost > 0.0 else member.material.cost_per_kg_usd
        return m_exact, max(m_exact, 1e-4), cost, max(eff_cost_kg, 0.5)

    # Virtual massless link (toolhead_mount, nozzle_stem, bed_arm_*, panel_shear_*)
    if member.id == "toolhead_mount":
        ref_mat = get_material(spec.toolhead.mount_material)
        m_ref = 0.06
    elif member.id == "nozzle_stem":
        ref_mat = get_material("aluminum_6061_t6")
        m_ref = 0.025
    elif member.role == "bed":
        ref_mat = get_material(spec.bed.plate_material)
        m_ref = max(spec.bed.mass_g * 1e-3 / 3.0, 0.25)
    else:
        ref_mat = get_material(spec.frame.panel_material)
        m_ref = 0.25
    cost_kg = max(ref_mat.cost_per_kg_usd, 1.0)
    return m_exact, m_ref, m_ref * cost_kg, cost_kg


def _stiffen_hint(group_id: str, role: str, spec: PrinterSpec | None = None) -> str:
    if spec is None:
        if group_id == "frame_feet":
            return "Main limit: benchtop foot anchoring stiffness (compliant isolation pads allow base rocking/bounce)."
        if role == "toolhead" or group_id == "toolhead_mount":
            return "Main limit: toolhead mount stiffness, carriage pitch compliance, and CoG/nozzle lever arm."
        if role == "rail" or group_id == "y_axis_rail_beams":
            return "Main limit: Y-rail beam & carriage lateral/torsional stiffness along the gantry load path."
        if group_id == "trident_backbone":
            return "Main limit: lower-frame mild-steel backbone plate shear/bending stiffness and perimeter M5 bolting."
        if group_id == "top_brace":
            return "Main limit: top open-center brace shear/bending stiffness and perimeter M5 bolting to top rails."
        if group_id == "front_brace":
            return "Main limit: front open-door brace shear/bending stiffness and perimeter M5 bolting to front columns/rails."
        if role == "panel" or group_id in ("side_panels", "rear_panel", "top_panel", "enclosure_panels"):
            return "Main limit: outer frame shear-panel in-plane stiffness and panel-to-frame coupling."
        if role == "brace" or group_id == "frame_bracing":
            return "Main limit: outer frame diagonal racking and triangulation stiffness."
        if role == "gantry" or group_id == "x_gantry_beam":
            return "Main limit: X-gantry beam & X-rail bending/torsional stiffness under toolhead acceleration."
        return "Main limit: outer frame bending and racking stiffness along the structural load path."

    if group_id in (
        "front_frame_columns",
        "rear_frame_columns",
        "frame_columns",
        "frame_perimeter_beams",
    ):
        if spec.architecture == "corexy":
            if not spec.frame.enclosed_panels or spec.frame.panel_mounting != "bolted_structural":
                return (
                    "Main limit: outer frame shear & racking stiffness "
                    "(non-structural or open panels leave frame columns uncoupled in shear)."
                )
            if (
                spec.frame.panel_material not in ("aluminum_5052_sheet", "carbon_fiber_plate")
                or spec.frame.panel_thickness_mm < 4.0
            ):
                return (
                    "Main limit: outer frame shear-panel in-plane stiffness, fastener shear stiffness, "
                    "and column bending rigidity."
                )
            return "Main limit: outer frame column bending/torsional stiffness and corner-joint rigidity."
        if spec.architecture == "cartesian_bedslinger":
            if spec.frame.bracing != "full_truss":
                return "Main limit: portal frame fore-aft/lateral bending stiffness and diagonal racking rigidity."
            return "Main limit: portal frame column bending/torsional stiffness and base-joint rigidity."
        return "Main limit: vertical tower bending/torsional stiffness and perimeter triangulation rigidity."

    if group_id == "z_gantry_uprights":
        if spec.frame.bracing != "full_truss":
            return "Main limit: Z-upright portal bending stiffness and fore-aft diagonal racking rigidity."
        return "Main limit: Z-upright extrusion bending/torsional stiffness and corner-joint rigidity."

    if group_id == "z_gantry_joints":
        return "Main limit: flying-gantry Z-belt/carriage joint compliance coupling the gantry to the 4 corner columns."

    if group_id == "delta_towers":
        if spec.frame.bracing != "full_truss":
            return "Main limit: delta tower bending stiffness and perimeter triangulation rigidity."
        return "Main limit: delta tower extrusion bending and torsional stiffness."

    if group_id == "delta_arms":
        return "Main limit: diagonal delta arm axial/buckling stiffness and effector inertia."

    if group_id == "trident_backbone":
        return (
            "Main limit: 3 mm mild-steel lower-frame backbone plate shear/flexural continuity "
            "and M5 perimeter bolting into the Z-motor mounts."
        )

    if group_id == "top_brace":
        return (
            "Main limit: 5 mm Al 5052 top open-center halo ring shear/flexural stiffness "
            "and M5 perimeter bolting into the top frame rails."
        )

    if group_id == "front_brace":
        return (
            "Main limit: 5 mm Al 5052 front open-door portal brace shear/flexural stiffness "
            "and M5 bolting into the front frame columns and rails."
        )

    if group_id in ("side_panels", "rear_panel", "top_panel", "enclosure_panels") or role == "panel":
        if not spec.frame.enclosed_panels or spec.frame.panel_mounting != "bolted_structural":
            return "Main limit: outer frame shear-panel coupling (panels are unbolted and carry no frame shear load)."
        if (
            spec.frame.panel_material not in ("aluminum_5052_sheet", "carbon_fiber_plate")
            or spec.frame.panel_thickness_mm < 4.0
        ):
            return "Main limit: outer frame shear-panel in-plane stiffness and panel-to-frame fastener shear stiffness."
        return "Main limit: shear-panel membrane/flexural stiffness and bolted gantry-frame joint compliance."

    if group_id == "y_axis_rail_beams" or role == "rail":
        if spec.architecture == "corexy" and not spec.panels_bolted_to_gantry_effective and not spec.is_flying_gantry:
            return "Main limit: Y-rail beam lateral/torsional stiffness (Y-rails are not shear-coupled to side panels)."
        return "Main limit: Y-rail beam & carriage lateral/torsional stiffness and XY-joint bracket compliance."

    if group_id == "x_gantry_beam" or role == "gantry":
        return "Main limit: X-gantry beam & X-rail bending/torsional stiffness under toolhead acceleration."

    if group_id == "toolhead_mount" or role == "toolhead":
        if spec.architecture == "delta":
            return (
                f"Main limit: effector mount compliance ('{spec.toolhead.mount_material}'), "
                "nozzle/CoG lever arm overhang, and effector mass."
            )
        return (
            f"Main limit: toolhead mount stiffness ('{spec.toolhead.mount_material}'), "
            "rail carriage pitch compliance, and CoG/nozzle lever arm."
        )

    if group_id == "bed_support" or role == "bed":
        if spec.architecture == "cartesian_bedslinger":
            return "Main limit: moving Y-bed carriage stiffness, Y-axis drive belt compliance, and bed inertia."
        return "Main limit: bed support & Z-mount axial/cantilever stiffness and bed plate flexural rigidity."

    if group_id == "frame_feet":
        return "Main limit: benchtop foot anchoring stiffness (compliant isolation pads allow base rocking/bounce)."

    if group_id == "frame_bracing" or role == "brace":
        return "Main limit: outer frame diagonal racking and triangulation stiffness."

    return "Main limit: outer frame bending and racking stiffness along the primary structural load path."


def _stiffen_why(group_id: str, comp_pct: float, est_hz_gain: float) -> str:
    if group_id == "frame_feet":
        if est_hz_gain >= 0.4:
            return (
                f"Soft benchtop feet allow the printer base to rock, carrying {comp_pct:.0f}% of "
                f"flex (+{est_hz_gain:.1f} Hz potential)"
            )
        return f"Soft benchtop feet allow base rocking/bounce ({comp_pct:.0f}% of structural loop flex)"
    if comp_pct >= 1.5 and est_hz_gain >= 0.4:
        if group_id in ("front_frame_columns", "rear_frame_columns", "frame_columns"):
            return f"Carries {comp_pct:.0f}% of nozzle-to-bed flex & controls frame sway (+{est_hz_gain:.1f} Hz potential)"
        if group_id == "z_gantry_uprights":
            return f"Drives portal bending in Y and carries {comp_pct:.0f}% of toolhead flex (+{est_hz_gain:.1f} Hz potential)"
        if group_id == "delta_arms":
            return f"Carries {comp_pct:.0f}% of effector-to-frame flex & drives primary resonance (+{est_hz_gain:.1f} Hz potential)"
        if group_id == "y_axis_rail_beams":
            return f"Carries {comp_pct:.0f}% of nozzle-to-bed flex along Y & supports gantry inertia (+{est_hz_gain:.1f} Hz potential)"
        if group_id == "x_gantry_beam":
            return f"Carries {comp_pct:.0f}% of nozzle-to-bed flex under X/Y acceleration (+{est_hz_gain:.1f} Hz potential)"
        if group_id == "toolhead_mount":
            return (
                f"Carries {comp_pct:.0f}% of nozzle-to-bed flex via carriage pitch & nozzle overhang "
                f"(+{est_hz_gain:.1f} Hz potential)"
            )
        if group_id == "trident_backbone":
            return f"Acts as a 3 mm mild-steel lower-frame diaphragm carrying {comp_pct:.0f}% of loop flex (+{est_hz_gain:.1f} Hz potential)"
        if group_id == "top_brace":
            return f"Acts as a 5 mm Al top open-center brace carrying {comp_pct:.0f}% of loop flex (+{est_hz_gain:.1f} Hz potential)"
        if group_id == "front_brace":
            return f"Acts as a 5 mm Al front open-door brace carrying {comp_pct:.0f}% of loop flex (+{est_hz_gain:.1f} Hz potential)"
        if group_id in ("side_panels", "rear_panel", "top_panel", "enclosure_panels"):
            return f"Acts as a structural shear diaphragm carrying {comp_pct:.0f}% of loop flex (+{est_hz_gain:.1f} Hz potential)"
        return f"Carries {comp_pct:.0f}% of nozzle-to-bed flex & boosts structural resonance (+{est_hz_gain:.1f} Hz potential)"
    if comp_pct >= 1.5:
        if group_id == "toolhead_mount":
            return f"Carries {comp_pct:.0f}% of nozzle-to-bed flex via carriage pitch & nozzle overhang"
        return f"Carries {comp_pct:.0f}% of static nozzle-to-bed flex under peak acceleration"
    if est_hz_gain >= 0.3:
        return f"Strongly participates in the fundamental vibration mode (+{est_hz_gain:.1f} Hz potential for +50% stiffness)"
    return "Contributes structural stiffness along the nozzle-to-bed load path"


def _lighten_hint(group_id: str, role: str, spec: PrinterSpec | None = None) -> str:
    if role in ("cog", "toolhead", "nozzle") or group_id == "toolhead_assembly":
        return "Main limit: moving mass on toolhead assembly (accelerates on every X and Y move)."
    if group_id == "x_gantry_beam" or (role == "gantry" and "x_gantry" in group_id):
        return "Main limit: moving mass along the X-gantry beam (accelerates on every Y move)."
    if group_id in ("y_carriages", "z_carriages", "delta_carriages"):
        return "Main limit: moving carriage mass at gantry ends (loads rails, brackets, and drive belts)."
    if group_id == "moving_y_bed" or role == "bed":
        return "Main limit: moving Y-axis heated bed inertia (loads Y-drive belt and bed carriage)."
    if role == "rail" or group_id == "y_axis_rail_beams":
        return "Main limit: linear rail and backing mass along the gantry/frame load path."
    return "Main limit: moving or upper-frame inertia participating in the fundamental vibration mode."


def _lighten_why(group_id: str, hz_gain_100g: float, is_moving: bool) -> str:
    if group_id == "toolhead_assembly":
        return (
            f"Toolhead accelerates on every X and Y move — cutting 100 g adds +{hz_gain_100g:.2f} Hz "
            "and trims inertial flex without changing the frame"
        )
    if group_id == "x_gantry_beam":
        return (
            f"Entire X-gantry beam accelerates on every Y move — cutting 100 g adds +{hz_gain_100g:.2f} Hz "
            "while lowering Y-motor torque load"
        )
    if group_id in ("y_carriages", "z_carriages", "delta_carriages"):
        return (
            f"Carriage mass rides on the gantry ends — cutting 100 g adds +{hz_gain_100g:.2f} Hz "
            "to gantry and portal resonance"
        )
    if group_id == "moving_y_bed":
        return (
            f"Heavy bed accelerates on every Y move — cutting 100 g adds +{hz_gain_100g:.2f} Hz "
            "and reduces Y-belt stretch"
        )
    if is_moving:
        return f"Moving mass accelerates on every print move — cutting 100 g raises resonance by +{hz_gain_100g:.2f} Hz"
    return f"Upper-frame mass lowers structural frequency — trimming 100 g adds +{hz_gain_100g:.2f} Hz"


def compute_structural_sensitivities(
    assembly: AssemblyModel,
    static_res: StaticFEAResult | None = None,
    modal_res: ModalResult | None = None,
    system: GlobalSystem | None = None,
) -> SensitivityReport:
    """Compute exact analytical static adjoint and Fox-Kapoor modal sensitivities.

    Pass the ``system`` already used for the static and modal solves to reuse its global
    matrices, element blocks, ``K_ff`` factorization, and (when the solver path matches) eigenpairs.
    """
    _ = static_res  # static fields are solved exactly with one factorization of K_ff
    if system is None:
        system = assemble_global_system(assembly)
    K, M, _, node_idx, free_dofs = system.as_tuple()
    n_dof = K.shape[0]
    n_free = len(free_dofs)

    if n_free == 0:
        return SensitivityReport(
            fundamental_freq_hz=0.0,
            primary_x_freq_hz=0.0,
            primary_y_freq_hz=0.0,
            static_rel_disp_x_um=0.0,
            static_rel_disp_y_um=0.0,
        )

    K_ff = system.reduced_stiffness()
    M_ff = system.reduced_mass()

    # 1. Static primary solves (X & Y peak acceleration loads) + single-backsub adjoints
    th_idx = node_idx[assembly.toolhead_node_id]
    cog_id = (
        assembly.toolhead_cog_node_id
        if assembly.toolhead_cog_node_id in node_idx
        else assembly.toolhead_node_id
    )
    noz_id = (
        assembly.nozzle_node_id
        if assembly.nozzle_node_id in node_idx
        else assembly.toolhead_node_id
    )
    cog_idx = node_idx[cog_id]
    noz_idx = node_idx[noz_id]
    bed_idx = node_idx.get(assembly.bed_node_id) if assembly.bed_node_id else None
    has_bed = bed_idx is not None

    solve_rhs = shared_factorized_solver(system)
    accel_m_s2 = assembly.spec.kinematics.max_accel_mm_s2 * 1e-3

    F_x, F_y, _, _ = build_inertial_load_vectors(
        assembly, node_idx, n_dof, accel_m_s2
    )

    # Translational nozzle-to-bed adjoint probes (keeps frame_feet rocking compliance visible)
    c_x = np.zeros(n_dof, dtype=float)
    c_x[6 * noz_idx + 0] = 1.0
    if has_bed and bed_idx is not None:
        c_x[6 * bed_idx + 0] = -1.0

    c_y = np.zeros(n_dof, dtype=float)
    c_y[6 * noz_idx + 1] = 1.0
    if has_bed and bed_idx is not None:
        c_y[6 * bed_idx + 1] = -1.0

    u_x = solve_rhs(F_x)
    u_y = solve_rhs(F_y)
    lam_x = solve_rhs(c_x)
    lam_y = solve_rhs(c_y)

    u_rel_x = float(c_x @ u_x)
    u_rel_y = float(c_y @ u_y)

    # 2. Modal eigensolve and primary X / Y / fundamental structural mode identification
    panel_indices = [i for i, n in enumerate(assembly.nodes) if n.role == "panel"]
    non_panel_indices = [i for i, n in enumerate(assembly.nodes) if n.role != "panel"]
    gantry_indices = [
        i
        for i, n in enumerate(assembly.nodes)
        if n.role in ("gantry", "toolhead", "nozzle", "cog")
    ]
    has_panels = len(panel_indices) > 0 and len(non_panel_indices) > 0
    target_scan = 56 if has_panels else 24

    eigvals, eigvecs = _solve_eigen_modes(K_ff, M_ff, target_scan=target_scan, system=system)
    n_eig = len(eigvals)
    full_vecs = np.zeros((n_dof, n_eig), dtype=float)
    full_vecs[free_dofs, :] = eigvecs

    m_cog_lumped = max(float(assembly.nodes[cog_idx].lumped_mass_kg), 0.0)
    m_noz_lumped = max(float(assembly.nodes[noz_idx].lumped_mass_kg), 0.0) if noz_idx != cog_idx else 0.0
    if noz_idx != cog_idx and (m_cog_lumped + m_noz_lumped) > 1e-9:
        noz_inertia_frac = m_noz_lumped / (m_cog_lumped + m_noz_lumped)
        cog_inertia_frac = 1.0 - noz_inertia_frac
    else:
        noz_inertia_frac = 0.0
        cog_inertia_frac = 1.0

    fx_exc = np.zeros(n_dof, dtype=float)
    fx_exc[6 * cog_idx + 0] = cog_inertia_frac
    if noz_inertia_frac > 0.0:
        fx_exc[6 * noz_idx + 0] += noz_inertia_frac
    fy_exc = np.zeros(n_dof, dtype=float)
    if assembly.spec.architecture == "cartesian_bedslinger":
        if has_bed and bed_idx is not None:
            fy_exc[6 * bed_idx + 1] = -1.0
        else:
            fy_exc[6 * cog_idx + 1] = cog_inertia_frac
            if noz_inertia_frac > 0.0:
                fy_exc[6 * noz_idx + 1] += noz_inertia_frac
    elif assembly.spec.architecture == "delta":
        fy_exc[6 * cog_idx + 1] = cog_inertia_frac
        if noz_inertia_frac > 0.0:
            fy_exc[6 * noz_idx + 1] += noz_inertia_frac
    else:
        th_share = min(
            1.0,
            max(0.0, assembly.toolhead_moving_mass_kg / max(assembly.y_moving_mass_kg, 1e-6)),
        )
        other_gantry = [
            gid
            for gid in assembly.gantry_node_ids
            if gid != assembly.toolhead_node_id and gid in node_idx
        ]
        if other_gantry:
            fy_exc[6 * cog_idx + 1] = th_share * cog_inertia_frac
            if noz_inertia_frac > 0.0:
                fy_exc[6 * noz_idx + 1] += th_share * noz_inertia_frac
            rem = (1.0 - th_share) / len(other_gantry)
            for gid in other_gantry:
                fy_exc[6 * node_idx[gid] + 1] += rem
        else:
            fy_exc[6 * cog_idx + 1] = cog_inertia_frac
            if noz_inertia_frac > 0.0:
                fy_exc[6 * noz_idx + 1] += noz_inertia_frac

    free_panel_mask = np.zeros(n_free, dtype=bool)
    if has_panels:
        panel_dof_set = {6 * pi + d for pi in panel_indices for d in range(6)}
        free_panel_mask = np.array([dof in panel_dof_set for dof in free_dofs], dtype=bool)

    struct_candidates: list[dict[str, Any]] = []
    for idx in range(n_eig):
        lam = max(float(eigvals[idx]), 0.0)
        freq_hz = math.sqrt(lam) / (2.0 * math.pi)
        phi_f = eigvecs[:, idx]
        full_vec = full_vecs[:, idx]

        trans_raw = np.array([full_vec[6 * i : 6 * i + 3] for i in range(len(assembly.nodes))])
        max_amp = float(np.max(np.linalg.norm(trans_raw, axis=1)))
        trans = trans_raw / max_amp if max_amp > 1e-12 else trans_raw.copy()

        noz_rel_all = (
            np.abs(trans[noz_idx] - trans[bed_idx]) if bed_idx is not None else np.abs(trans[noz_idx])
        )
        noz_norm_all = float(np.linalg.norm(noz_rel_all))

        is_isolated_panel_drum = False
        if has_panels:
            np_max = float(np.max(np.linalg.norm(trans[non_panel_indices], axis=1)))
            gantry_max = (
                float(np.max(np.linalg.norm(trans[gantry_indices], axis=1)))
                if gantry_indices
                else np_max
            )
            phi_p = np.where(free_panel_mask, phi_f, 0.0)
            ke_panel = float(phi_p @ M_ff @ phi_p)
            is_isolated_panel_drum = (
                (ke_panel > 0.55 and noz_norm_all < 0.12)
                or (ke_panel > 0.58 and noz_norm_all < 0.25)
                or (ke_panel > 0.68 and gantry_max < 0.35 and noz_norm_all < 0.28)
                or (np_max < 0.25 and gantry_max < 0.20)
            )
            tr_struct = trans_raw[non_panel_indices]
            max_struct = max(float(np.max(np.linalg.norm(tr_struct, axis=1))), 1e-12)
            noz_rel = (
                np.abs(trans_raw[noz_idx] - trans_raw[bed_idx]) / max_struct
                if bed_idx is not None
                else np.abs(trans_raw[noz_idx]) / max_struct
            )
            bed_z_amp = abs(float(trans_raw[bed_idx][2] / max_struct)) if bed_idx is not None else 0.0
            axis_energy = np.sum(tr_struct**2, axis=0)
        else:
            noz_rel = noz_rel_all
            bed_z_amp = abs(float(trans[bed_idx][2])) if bed_idx is not None else 0.0
            axis_energy = np.sum(trans**2, axis=0)

        dom_axis = ["X", "Y", "Z"][int(np.argmax(axis_energy))]
        is_bed_z_mode = (
            bed_idx is not None
            and bed_z_amp > 0.60
            and float(noz_rel[0]) < 0.20
            and float(noz_rel[1]) < 0.20
        )

        rec_x = 0.0
        rec_y = 0.0
        if lam > 1.0:
            rel_x_raw = (
                float(full_vec[6 * noz_idx + 0] - full_vec[6 * bed_idx + 0])
                if bed_idx is not None
                else float(full_vec[6 * noz_idx + 0])
            )
            rel_y_raw = (
                float(full_vec[6 * noz_idx + 1] - full_vec[6 * bed_idx + 1])
                if bed_idx is not None
                else float(full_vec[6 * noz_idx + 1])
            )
            rec_x = abs(rel_x_raw * float(full_vec @ fx_exc)) / lam
            rec_y = abs(rel_y_raw * float(full_vec @ fy_exc)) / lam

        struct_candidates.append(
            {
                "idx": idx,
                "freq_hz": freq_hz,
                "lam": lam,
                "dom_axis": dom_axis,
                "noz_rel": noz_rel,
                "is_isolated_panel_drum": is_isolated_panel_drum,
                "is_bed_z_mode": is_bed_z_mode,
                "rec_x": rec_x,
                "rec_y": rec_y,
            }
        )

    rx_max = max((c["rec_x"] for c in struct_candidates), default=0.0)
    ry_max = max((c["rec_y"] for c in struct_candidates), default=0.0)
    filtered = [
        c
        for c in struct_candidates
        if not c["is_isolated_panel_drum"]
        and not (
            c["dom_axis"] == "Z"
            and (
                bool(c["is_bed_z_mode"])
                or float(max(c["noz_rel"][0], c["noz_rel"][1])) < 0.15
                or (
                    rx_max > 1e-18
                    and ry_max > 1e-18
                    and c["rec_x"] < 0.03 * rx_max
                    and c["rec_y"] < 0.03 * ry_max
                )
            )
        )
    ]
    if not filtered:
        filtered = struct_candidates

    # Match modal_res.fundamental_freq_hz if supplied, else lowest filtered structural mode
    if modal_res is not None and modal_res.fundamental_freq_hz > 0.0:
        c_fund = min(filtered, key=lambda c: abs(c["freq_hz"] - modal_res.fundamental_freq_hz))
    else:
        c_fund = filtered[0]

    # Pick primary X and primary Y structural modes (max compliance-weighted residue)
    c_x_mode = max(
        filtered,
        key=lambda c: c["rec_x"] / max(c["freq_hz"] ** 2, 1.0),
    )
    c_y_mode = max(
        filtered,
        key=lambda c: c["rec_y"] / max(c["freq_hz"] ** 2, 1.0),
    )

    idx_fund = int(c_fund["idx"])
    idx_x = int(c_x_mode["idx"])
    idx_y = int(c_y_mode["idx"])

    f_fund = max(float(c_fund["freq_hz"]), 1e-6)
    f_x = max(float(c_x_mode["freq_hz"]), 1e-6)
    f_y = max(float(c_y_mode["freq_hz"]), 1e-6)
    eig_fund = float(c_fund["lam"])
    eig_x = float(c_x_mode["lam"])
    eig_y = float(c_y_mode["lam"])

    phi_fund = _dailey_subspace_vector(eigvals, full_vecs, idx_fund, fx_exc + fy_exc)
    phi_x = _dailey_subspace_vector(eigvals, full_vecs, idx_x, fx_exc)
    phi_y = _dailey_subspace_vector(eigvals, full_vecs, idx_y, fy_exc)

    inv_8pi2_fund = 1.0 / (8.0 * (math.pi**2) * f_fund)
    inv_8pi2_x = 1.0 / (8.0 * (math.pi**2) * f_x)
    inv_8pi2_y = 1.0 / (8.0 * (math.pi**2) * f_y)

    # 3. Element-level exact Fox-Kapoor & Adjoint evaluation
    nmap = assembly.node_map
    raw_members: list[dict[str, Any]] = []

    for m_i, mem in enumerate(assembly.members):
        # Element blocks were computed once during global assembly (identical values)
        dofs, Ke, Me, length = system.member_blocks[m_i]

        # Static adjoint derivatives
        dux = -float(lam_x[dofs] @ Ke @ u_x[dofs])
        duy = -float(lam_y[dofs] @ Ke @ u_y[dofs])
        comp_raw = _axis_comp_share(dux, u_rel_x) + _axis_comp_share(duy, u_rel_y)

        # Modal Fox-Kapoor derivatives
        pf = phi_fund[dofs]
        px = phi_x[dofs]
        py = phi_y[dofs]

        stiff_f = float(pf @ Ke @ pf) * inv_8pi2_fund
        mass_f = -eig_fund * float(pf @ Me @ pf) * inv_8pi2_fund
        stiff_x = float(px @ Ke @ px) * inv_8pi2_x
        mass_x = -eig_x * float(px @ Me @ px) * inv_8pi2_x
        stiff_y = float(py @ Ke @ py) * inv_8pi2_y
        mass_y = -eig_y * float(py @ Me @ py) * inv_8pi2_y

        m_exact, m_ref, cost_usd, cost_per_kg = _member_mass_and_cost(mem, length, assembly)
        net_f = stiff_f + mass_f
        df_dm = net_f / max(m_ref, 1e-6)

        raw_members.append(
            {
                "member_id": mem.id,
                "role": mem.role,
                "mass_kg": m_exact,
                "ref_mass_kg": m_ref,
                "cost_usd": cost_usd,
                "cost_per_kg": cost_per_kg,
                "stiffness_sens_hz": stiff_f,
                "mass_sens_hz": mass_f,
                "stiffness_sens_x_hz": stiff_x,
                "mass_sens_x_hz": mass_x,
                "stiffness_sens_y_hz": stiff_y,
                "mass_sens_y_hz": mass_y,
                "du_rel_x_dlnE_m": dux,
                "du_rel_y_dlnE_m": duy,
                "du_rel_dlnE_m": dux + duy,
                "hz_per_100g": 0.1 * df_dm,
                "hz_per_usd": df_dm / max(cost_per_kg, 0.5),
                "_comp_raw": comp_raw,
                "_net_sens": net_f,
            }
        )

    # Shell panel elements grouped by canonical bay ('left', 'right', 'rear', 'top')
    bay_acc: dict[str, dict[str, float]] = {
        bay: {
            "mass_kg": 0.0,
            "cost_usd": 0.0,
            "cost_per_kg": 0.0,
            "stiffness_sens_hz": 0.0,
            "mass_sens_hz": 0.0,
            "stiffness_sens_x_hz": 0.0,
            "mass_sens_x_hz": 0.0,
            "stiffness_sens_y_hz": 0.0,
            "mass_sens_y_hz": 0.0,
            "du_rel_x_dlnE_m": 0.0,
            "du_rel_y_dlnE_m": 0.0,
            "du_rel_dlnE_m": 0.0,
            "_comp_raw": 0.0,
            "n_elements": 0,
        }
        for bay in ("left", "right", "rear", "top")
    }

    for sh_i, sh in enumerate(assembly.shell_elements):
        bay_key = (
            sh.bay
            if sh.bay in ("top_brace", "front_brace", "backbone")
            else sh.bay.split("_")[0]
        )
        if bay_key not in bay_acc:
            bay_acc[bay_key] = {k: 0.0 for k in bay_acc["left"]}
        dofs, k_sh, m_sh = system.shell_blocks[sh_i]
        dux = -float(lam_x[dofs] @ k_sh @ u_x[dofs])
        duy = -float(lam_y[dofs] @ k_sh @ u_y[dofs])
        comp_raw = _axis_comp_share(dux, u_rel_x) + _axis_comp_share(duy, u_rel_y)

        pf = phi_fund[dofs]
        px = phi_x[dofs]
        py = phi_y[dofs]
        b = bay_acc[bay_key]
        m_sh_kg = float(np.sum(np.diag(m_sh)[0::6]))
        b["mass_kg"] += m_sh_kg
        b["cost_usd"] += m_sh_kg * sh.material.cost_per_kg_usd
        b["cost_per_kg"] = max(sh.material.cost_per_kg_usd, 0.5)
        b["stiffness_sens_hz"] += float(pf @ k_sh @ pf) * inv_8pi2_fund
        b["mass_sens_hz"] += -eig_fund * float(pf @ m_sh @ pf) * inv_8pi2_fund
        b["stiffness_sens_x_hz"] += float(px @ k_sh @ px) * inv_8pi2_x
        b["mass_sens_x_hz"] += -eig_x * float(px @ m_sh @ px) * inv_8pi2_x
        b["stiffness_sens_y_hz"] += float(py @ k_sh @ py) * inv_8pi2_y
        b["mass_sens_y_hz"] += -eig_y * float(py @ m_sh @ py) * inv_8pi2_y
        b["du_rel_x_dlnE_m"] += dux
        b["du_rel_y_dlnE_m"] += duy
        b["du_rel_dlnE_m"] += dux + duy
        b["_comp_raw"] += comp_raw
        b["n_elements"] += 1

    # Fastener & joint springs: fold panel bays into bay_acc, and non-panel springs into raw_members
    for sp_i, sp in enumerate(assembly.fastener_springs):
        bay_raw = sp.bay
        bay_prefix = (
            bay_raw
            if bay_raw in ("top_brace", "front_brace", "backbone")
            else bay_raw.split("_")[0]
        )
        dofs, k_sp = system.spring_blocks[sp_i]
        dux = -float(lam_x[dofs] @ k_sp @ u_x[dofs])
        duy = -float(lam_y[dofs] @ k_sp @ u_y[dofs])
        comp_raw = _axis_comp_share(dux, u_rel_x) + _axis_comp_share(duy, u_rel_y)
        pf = phi_fund[dofs]
        px = phi_x[dofs]
        py = phi_y[dofs]
        stiff_f = float(pf @ k_sp @ pf) * inv_8pi2_fund
        stiff_x = float(px @ k_sp @ px) * inv_8pi2_x
        stiff_y = float(py @ k_sp @ py) * inv_8pi2_y

        if bay_raw not in ("bed_mount", "bed_z_drive", "z_guide", "toolhead", "feet", "z_joint") and bay_prefix in bay_acc:
            b = bay_acc[bay_prefix]
            b["du_rel_x_dlnE_m"] += dux
            b["du_rel_y_dlnE_m"] += duy
            b["du_rel_dlnE_m"] += dux + duy
            b["_comp_raw"] += comp_raw
            b["stiffness_sens_hz"] += stiff_f
            b["stiffness_sens_x_hz"] += stiff_x
            b["stiffness_sens_y_hz"] += stiff_y
            continue

        if bay_raw == "bed_mount":
            sp_role, sp_cost, sp_ref_mass = "bed", 20.0, 0.25
        elif bay_raw in ("bed_z_drive", "z_guide"):
            sp_role, sp_cost, sp_ref_mass = "bed", 25.0, 0.20
        elif bay_raw == "toolhead":
            sp_role, sp_cost, sp_ref_mass = "toolhead", 25.0, 0.06
        elif bay_raw == "z_joint":
            sp_role, sp_cost, sp_ref_mass = "gantry", 25.0, 0.15
        else:
            sp_role, sp_cost, sp_ref_mass = "frame", 15.0, 0.10

        raw_members.append(
            {
                "member_id": sp.id,
                "role": sp_role,
                "_bay": bay_raw,
                "mass_kg": 0.0,
                "ref_mass_kg": sp_ref_mass,
                "cost_usd": sp_cost,
                "cost_per_kg": sp_cost / sp_ref_mass,
                "stiffness_sens_hz": stiff_f,
                "mass_sens_hz": 0.0,
                "stiffness_sens_x_hz": stiff_x,
                "mass_sens_x_hz": 0.0,
                "stiffness_sens_y_hz": stiff_y,
                "mass_sens_y_hz": 0.0,
                "du_rel_x_dlnE_m": dux,
                "du_rel_y_dlnE_m": duy,
                "du_rel_dlnE_m": dux + duy,
                "hz_per_100g": 0.0,
                "hz_per_usd": stiff_f / sp_cost,
                "_comp_raw": comp_raw,
                "_net_sens": stiff_f,
            }
        )

    # 2-port Craig-Bampton CAD superelements: exact static adjoint & Fox-Kapoor derivatives
    internal_dof_cursor = 6 * len(assembly.nodes)
    for se_i, se in enumerate(getattr(assembly, "superelements", ())):
        dofs_se, Ke_se, Me_se = system.superelement_blocks[se_i]
        n_m_se = max(0, int(getattr(se, "n_internal_modes", max(0, Ke_se.shape[0] - 12))))
        internal_dof_cursor += n_m_se

        dux = -float(lam_x[dofs_se] @ Ke_se @ u_x[dofs_se])
        duy = -float(lam_y[dofs_se] @ Ke_se @ u_y[dofs_se])
        comp_raw = _axis_comp_share(dux, u_rel_x) + _axis_comp_share(duy, u_rel_y)

        pf = phi_fund[dofs_se]
        px = phi_x[dofs_se]
        py = phi_y[dofs_se]

        stiff_f = float(pf @ Ke_se @ pf) * inv_8pi2_fund
        stiff_x = float(px @ Ke_se @ px) * inv_8pi2_x
        stiff_y = float(py @ Ke_se @ py) * inv_8pi2_y
        if Me_se is not None:
            mass_f = -eig_fund * float(pf @ Me_se @ pf) * inv_8pi2_fund
            mass_x = -eig_x * float(px @ Me_se @ px) * inv_8pi2_x
            mass_y = -eig_y * float(py @ Me_se @ py) * inv_8pi2_y
        else:
            mass_f = 0.0
            mass_x = 0.0
            mass_y = 0.0

        se_role = getattr(se, "role", "toolhead") or "toolhead"
        if se.id == "toolhead_mount":
            ref_mat = get_material(assembly.spec.toolhead.effective_body_material)
            m_ref = max(float(getattr(se, "mass_kg", 0.0)), 0.06)
        elif se.id == "nozzle_stem":
            ref_mat = get_material("aluminum_6061_t6")
            m_ref = max(float(getattr(se, "mass_kg", 0.0)), 0.025)
        else:
            ref_mat = get_material(assembly.spec.toolhead.mount_material)
            m_ref = max(float(getattr(se, "mass_kg", 0.0)), 0.05)

        m_exact = float(getattr(se, "mass_kg", 0.0)) if Me_se is not None else 0.0
        cost_per_kg = max(ref_mat.cost_per_kg_usd, 1.0)
        cost_usd = m_ref * cost_per_kg
        net_f = stiff_f + mass_f
        df_dm = net_f / max(m_ref, 1e-6)

        raw_members.append(
            {
                "member_id": se.id,
                "role": se_role,
                "mass_kg": m_exact,
                "ref_mass_kg": m_ref,
                "cost_usd": cost_usd,
                "cost_per_kg": cost_per_kg,
                "stiffness_sens_hz": stiff_f,
                "mass_sens_hz": mass_f,
                "stiffness_sens_x_hz": stiff_x,
                "mass_sens_x_hz": mass_x,
                "stiffness_sens_y_hz": stiff_y,
                "mass_sens_y_hz": mass_y,
                "du_rel_x_dlnE_m": dux,
                "du_rel_y_dlnE_m": duy,
                "du_rel_dlnE_m": dux + duy,
                "hz_per_100g": 0.1 * df_dm,
                "hz_per_usd": df_dm / max(cost_per_kg, 0.5),
                "_comp_raw": comp_raw,
                "_net_sens": net_f,
                "_part_group_id": getattr(se, "part_group_id", "") or "toolhead_mount",
                "_part_group_label": getattr(se, "part_group_label", "")
                or "Toolhead & Nozzle Mount",
            }
        )

    # Axis compliance weights for multi-mode print-impact scoring (softer axis weighs more)
    w_x_raw = 1.0 / max(f_x**2, 100.0)
    w_y_raw = 1.0 / max(f_y**2, 100.0)
    w_x = w_x_raw / (w_x_raw + w_y_raw)
    w_y = w_y_raw / (w_x_raw + w_y_raw)

    # Normalize compliance_share_pct across all structural elements (members + bays) to sum to 100.0%
    tot_comp_raw = sum(m["_comp_raw"] for m in raw_members) + sum(
        b["_comp_raw"] for b in bay_acc.values()
    )
    comp_scale = 100.0 / max(tot_comp_raw, 1e-18)

    # Precompute XY-weighted metrics for members and bays
    for m in raw_members:
        m["_comp_pct"] = m["_comp_raw"] * comp_scale
        m["_xy_stiff_hz"] = w_x * max(float(m["stiffness_sens_x_hz"]), 0.0) + w_y * max(
            float(m["stiffness_sens_y_hz"]), 0.0
        )
        m["_xy_mass_hz"] = w_x * float(m["mass_sens_x_hz"]) + w_y * float(m["mass_sens_y_hz"])
        m["_xy_net_hz"] = (
            w_x * (float(m["stiffness_sens_x_hz"]) + float(m["mass_sens_x_hz"]))
            + w_y * (float(m["stiffness_sens_y_hz"]) + float(m["mass_sens_y_hz"]))
        )

    for b in bay_acc.values():
        b["_comp_pct"] = b["_comp_raw"] * comp_scale
        b["_xy_stiff_hz"] = w_x * max(float(b["stiffness_sens_x_hz"]), 0.0) + w_y * max(
            float(b["stiffness_sens_y_hz"]), 0.0
        )
        b["_xy_mass_hz"] = w_x * float(b["mass_sens_x_hz"]) + w_y * float(b["mass_sens_y_hz"])
        b["_xy_net_hz"] = (
            w_x * (float(b["stiffness_sens_x_hz"]) + float(b["mass_sens_x_hz"]))
            + w_y * (float(b["stiffness_sens_y_hz"]) + float(b["mass_sens_y_hz"]))
        )

    # Lumped nodal mass sensitivities computed before mass normalization so toolhead_cog sets the scale
    lumped_mass_sensitivities: dict[str, dict[str, Any]] = {}
    for i, node in enumerate(assembly.nodes):
        if node.lumped_mass_kg <= 0.0 or node.fixed:
            continue
        base = 6 * i
        ixx, iyy, izz = node_rotational_inertia_xyz(node, assembly)
        m_diag_nd = np.array(
            [node.lumped_mass_kg, node.lumped_mass_kg, node.lumped_mass_kg, ixx, iyy, izz],
            dtype=float,
        )
        pf = phi_fund[base : base + 6]
        px = phi_x[base : base + 6]
        py = phi_y[base : base + 6]
        ms_f = -eig_fund * float(np.sum(m_diag_nd * (pf**2))) * inv_8pi2_fund
        ms_x = -eig_x * float(np.sum(m_diag_nd * (px**2))) * inv_8pi2_x
        ms_y = -eig_y * float(np.sum(m_diag_nd * (py**2))) * inv_8pi2_y
        xy_mass_nd = w_x * ms_x + w_y * ms_y
        lumped_mass_sensitivities[node.id] = {
            "node_id": node.id,
            "role": node.role,
            "mass_kg": round(node.lumped_mass_kg, 5),
            "mass_sens_hz": float(ms_f),
            "mass_sens_x_hz": float(ms_x),
            "mass_sens_y_hz": float(ms_y),
            "hz_per_100g": round(0.1 * ms_f / max(node.lumped_mass_kg, 1e-5), 4),
            "_xy_mass_hz": xy_mass_nd,
        }

    active_bays = [b for b in bay_acc.values() if b["n_elements"] > 0]
    max_comp_pct = max(
        [m["_comp_pct"] for m in raw_members] + [b["_comp_pct"] for b in active_bays],
        default=1e-9,
    )
    max_comp_pct = max(max_comp_pct, 1e-9)

    max_xy_stiff = max(
        [m["_xy_stiff_hz"] for m in raw_members] + [b["_xy_stiff_hz"] for b in active_bays],
        default=1e-9,
    )
    max_xy_stiff = max(max_xy_stiff, 1e-9)

    max_mass_pen = max(
        [-min(0.0, m["_xy_mass_hz"]) for m in raw_members]
        + [-min(0.0, b["_xy_mass_hz"]) for b in active_bays]
        + [-min(0.0, ld["_xy_mass_hz"]) for ld in lumped_mass_sensitivities.values()],
        default=1e-9,
    )
    max_mass_pen = max(max_mass_pen, 1e-9)

    all_xy_net_struct = [m["_xy_net_hz"] for m in raw_members] + [b["_xy_net_hz"] for b in active_bays]
    max_pos_xy_net = max([v for v in all_xy_net_struct if v > 0.0], default=1e-9)
    max_neg_xy_net = max([-v for v in all_xy_net_struct if v < 0.0], default=1e-9)

    def _raw_pos_score(comp_pct: float, xy_stiff: float) -> float:
        h_comp = min(1.0, max(0.0, comp_pct / max_comp_pct))
        return 0.65 * h_comp + 0.35 * max(0.0, xy_stiff / max_xy_stiff)

    max_pos_score = max(
        [_raw_pos_score(m["_comp_pct"], m["_xy_stiff_hz"]) for m in raw_members]
        + [_raw_pos_score(b["_comp_pct"], b["_xy_stiff_hz"]) for b in active_bays],
        default=1e-9,
    )
    max_pos_score = max(max_pos_score, 1e-9)

    def _compute_lens_scores(
        comp_pct: float,
        xy_stiff: float,
        xy_mass: float,
        xy_net: float,
        is_moving_elem: bool,
    ) -> tuple[float, float, float, float]:
        h_comp = min(1.0, max(0.0, comp_pct / max_comp_pct))
        if xy_net > 0.0:
            h_modal = min(1.0, xy_net / max(max_pos_xy_net, 1e-12))
        elif xy_net < 0.0:
            h_modal = max(-1.0, xy_net / max(max_neg_xy_net, 1e-12))
        else:
            h_modal = 0.0

        h_mass = 0.0 if xy_mass >= 0.0 else max(-1.0, xy_mass / max_mass_pen)

        pos_score = _raw_pos_score(comp_pct, xy_stiff)
        neg_score = abs(h_mass) if ((is_moving_elem or xy_net < 0.0) and h_comp < 0.30) else 0.0
        if pos_score >= 0.45 * neg_score and pos_score > 0.02:
            h_impact = min(1.0, pos_score / max_pos_score)
        elif neg_score > 0.02:
            h_impact = -min(1.0, neg_score)
        else:
            h_impact = 0.0
        return (
            round(h_impact, 4) + 0.0,
            round(h_comp, 4) + 0.0,
            round(h_modal, 4) + 0.0,
            round(h_mass, 4) + 0.0,
        )

    member_sensitivities: dict[str, dict[str, Any]] = {}
    for m in raw_members:
        comp_pct = m["_comp_pct"]
        is_moving_mem = m["role"] in ("gantry", "toolhead", "bed") or m["member_id"].startswith(
            ("x_gantry", "x_beam", "arm_", "bed_")
        )
        hs, hs_comp, hs_modal, hs_mass = _compute_lens_scores(
            comp_pct=comp_pct,
            xy_stiff=m["_xy_stiff_hz"],
            xy_mass=m["_xy_mass_hz"],
            xy_net=m["_xy_net_hz"],
            is_moving_elem=is_moving_mem,
        )
        member_sensitivities[m["member_id"]] = {
            "member_id": m["member_id"],
            "role": m["role"],
            "mass_kg": round(m["mass_kg"], 5),
            "cost_usd": round(m["cost_usd"], 3),
            "hz_per_100g": round(m["hz_per_100g"], 4),
            "hz_per_usd": round(m["hz_per_usd"], 4),
            "stiffness_sens_hz": float(m["stiffness_sens_hz"]),
            "mass_sens_hz": float(m["mass_sens_hz"]),
            "stiffness_sens_x_hz": float(m["stiffness_sens_x_hz"]),
            "mass_sens_x_hz": float(m["mass_sens_x_hz"]),
            "stiffness_sens_y_hz": float(m["stiffness_sens_y_hz"]),
            "mass_sens_y_hz": float(m["mass_sens_y_hz"]),
            "du_rel_x_dlnE_m": float(m["du_rel_x_dlnE_m"]),
            "du_rel_y_dlnE_m": float(m["du_rel_y_dlnE_m"]),
            "du_rel_dlnE_m": float(m["du_rel_dlnE_m"]),
            "compliance_share_pct": round(comp_pct, 3),
            "heat_score": hs,
            "heat_score_compliance": hs_comp,
            "heat_score_modal": hs_modal,
            "heat_score_mass": hs_mass,
        }

    panel_bay_sensitivities: dict[str, dict[str, Any]] = {}
    for bay_key, b in bay_acc.items():
        comp_pct_b = b["_comp_pct"]
        if b["n_elements"] > 0 and b["mass_kg"] > 1e-6:
            net_b = b["stiffness_sens_hz"] + b["mass_sens_hz"]
            df_dm_b = net_b / b["mass_kg"]
            hz_100g_b = 0.1 * df_dm_b
            hz_usd_b = df_dm_b / max(b["cost_per_kg"], 0.5)
            hs_b, hs_comp_b, hs_modal_b, hs_mass_b = _compute_lens_scores(
                comp_pct=comp_pct_b,
                xy_stiff=b["_xy_stiff_hz"],
                xy_mass=b["_xy_mass_hz"],
                xy_net=b["_xy_net_hz"],
                is_moving_elem=False,
            )
        else:
            hz_100g_b = 0.0
            hz_usd_b = 0.0
            hs_b, hs_comp_b, hs_modal_b, hs_mass_b = 0.0, 0.0, 0.0, 0.0
        panel_bay_sensitivities[bay_key] = {
            "bay": bay_key,
            "n_elements": int(b["n_elements"]),
            "mass_kg": round(b["mass_kg"], 4),
            "cost_usd": round(b["cost_usd"], 3),
            "hz_per_100g": round(hz_100g_b, 4),
            "hz_per_usd": round(hz_usd_b, 4),
            "stiffness_sens_hz": float(b["stiffness_sens_hz"]),
            "mass_sens_hz": float(b["mass_sens_hz"]),
            "stiffness_sens_x_hz": float(b["stiffness_sens_x_hz"]),
            "mass_sens_x_hz": float(b["mass_sens_x_hz"]),
            "stiffness_sens_y_hz": float(b["stiffness_sens_y_hz"]),
            "mass_sens_y_hz": float(b["mass_sens_y_hz"]),
            "du_rel_x_dlnE_m": float(b["du_rel_x_dlnE_m"]),
            "du_rel_y_dlnE_m": float(b["du_rel_y_dlnE_m"]),
            "du_rel_dlnE_m": float(b["du_rel_dlnE_m"]),
            "compliance_share_pct": round(comp_pct_b, 3),
            "heat_score": hs_b,
            "heat_score_compliance": hs_comp_b,
            "heat_score_modal": hs_modal_b,
            "heat_score_mass": hs_mass_b,
        }

    # Finalize 4-lens heat scores on lumped_mass_sensitivities
    for ld in lumped_mass_sensitivities.values():
        xy_m = float(ld.pop("_xy_mass_hz"))
        hm = 0.0 if xy_m >= 0.0 else max(-1.0, xy_m / max_mass_pen)
        hm_r = round(hm, 4) + 0.0
        ld["heat_score"] = hm_r
        ld["heat_score_compliance"] = 0.0
        ld["heat_score_modal"] = hm_r
        ld["heat_score_mass"] = hm_r
        ld["_xy_mass_hz"] = xy_m

    # 4b. Subsystem breakdown ('toolhead', 'gantry', 'frame', 'bed')
    _SUBSYSTEM_LABELS: dict[str, str] = {
        "toolhead": "Toolhead & Mount",
        "gantry": "X/Y Gantry & Rails",
        "frame": "Outer Frame & Panels",
        "bed": "Bed & Z-Mount",
    }

    def _role_to_subsystem(role: str) -> str:
        if role in ("toolhead", "cog", "nozzle"):
            return "toolhead"
        if role in ("gantry", "rail"):
            return "gantry"
        if role == "bed":
            return "bed"
        return "frame"

    sub_acc: dict[str, dict[str, float]] = {
        sid: {"comp_pct": 0.0, "stiff_xy": 0.0, "mass_xy": 0.0}
        for sid in ("toolhead", "gantry", "frame", "bed")
    }
    for m in raw_members:
        sid = _role_to_subsystem(m["role"])
        sub_acc[sid]["comp_pct"] += float(m["_comp_pct"])
        sub_acc[sid]["stiff_xy"] += float(m["_xy_stiff_hz"])
        sub_acc[sid]["mass_xy"] += float(m["_xy_mass_hz"])
    for b in active_bays:
        sub_acc["frame"]["comp_pct"] += float(b["_comp_pct"])
        sub_acc["frame"]["stiff_xy"] += float(b["_xy_stiff_hz"])
        sub_acc["frame"]["mass_xy"] += float(b["_xy_mass_hz"])
    for ld in lumped_mass_sensitivities.values():
        sid = _role_to_subsystem(ld["role"])
        sub_acc[sid]["mass_xy"] += float(ld.pop("_xy_mass_hz"))

    tot_sub_comp = sum(v["comp_pct"] for v in sub_acc.values())
    if tot_sub_comp <= 1e-9:
        sub_acc["frame"]["comp_pct"] = 100.0

    primary_problem_subsystem = max(
        ("toolhead", "gantry", "frame", "bed"),
        key=lambda sid: (sub_acc[sid]["comp_pct"], sub_acc[sid]["stiff_xy"]),
    )
    rounded_sub_comp = {sid: round(sub_acc[sid]["comp_pct"], 1) for sid in sub_acc}
    diff_sub_comp = round(100.0 - sum(rounded_sub_comp.values()), 1)
    rounded_sub_comp[primary_problem_subsystem] = round(
        rounded_sub_comp[primary_problem_subsystem] + diff_sub_comp, 1
    )

    def _sub_severity(pct: float) -> str:
        if pct >= 35.0:
            return "critical"
        if pct >= 20.0:
            return "high"
        if pct >= 8.0:
            return "moderate"
        return "low"

    subsystem_breakdown: dict[str, dict[str, Any]] = {
        sid: {
            "id": sid,
            "label": _SUBSYSTEM_LABELS[sid],
            "compliance_share_pct": rounded_sub_comp[sid],
            "modal_stiff_hz": round(sub_acc[sid]["stiff_xy"], 2),
            "modal_mass_hz": round(sub_acc[sid]["mass_xy"], 2),
            "severity": _sub_severity(rounded_sub_comp[sid]),
        }
        for sid in ("toolhead", "gantry", "frame", "bed")
    }

    # 5. Top 5 stiffening ROI hotspots & Top 3 weight-reduction hotspots (grouped by physical part)
    spec = assembly.spec
    group_acc: dict[str, dict[str, Any]] = {}
    for m in raw_members:
        if m.get("_part_group_id"):
            gid = str(m["_part_group_id"])
            label = str(m.get("_part_group_label") or gid)
        else:
            gid, label = _classify_sensitivity_group(m["member_id"], m["role"], m.get("_bay", ""))
        g = group_acc.setdefault(
            gid,
            {
                "id": gid,
                "label": label,
                "role": m["role"],
                "stiffness_sens_hz": 0.0,
                "mass_sens_hz": 0.0,
                "stiffness_sens_x_hz": 0.0,
                "mass_sens_x_hz": 0.0,
                "stiffness_sens_y_hz": 0.0,
                "mass_sens_y_hz": 0.0,
                "compliance_share_pct": 0.0,
                "cost_usd": 0.0,
                "mass_kg": 0.0,
            },
        )
        g["stiffness_sens_hz"] += float(m["stiffness_sens_hz"])
        g["mass_sens_hz"] += float(m["mass_sens_hz"])
        g["stiffness_sens_x_hz"] += float(m["stiffness_sens_x_hz"])
        g["mass_sens_x_hz"] += float(m["mass_sens_x_hz"])
        g["stiffness_sens_y_hz"] += float(m["stiffness_sens_y_hz"])
        g["mass_sens_y_hz"] += float(m["mass_sens_y_hz"])
        g["compliance_share_pct"] += float(m["_comp_pct"])
        g["cost_usd"] += float(m["cost_usd"])
        g["mass_kg"] += float(m["mass_kg"])

    for bay_key, b in bay_acc.items():
        if b["n_elements"] <= 0:
            continue
        gid, label = _classify_sensitivity_group(f"bay_{bay_key}", "panel", bay=bay_key)
        g = group_acc.setdefault(
            gid,
            {
                "id": gid,
                "label": label,
                "role": "panel",
                "stiffness_sens_hz": 0.0,
                "mass_sens_hz": 0.0,
                "stiffness_sens_x_hz": 0.0,
                "mass_sens_x_hz": 0.0,
                "stiffness_sens_y_hz": 0.0,
                "mass_sens_y_hz": 0.0,
                "compliance_share_pct": 0.0,
                "cost_usd": 0.0,
                "mass_kg": 0.0,
            },
        )
        g["stiffness_sens_hz"] += float(b["stiffness_sens_hz"])
        g["mass_sens_hz"] += float(b["mass_sens_hz"])
        g["stiffness_sens_x_hz"] += float(b["stiffness_sens_x_hz"])
        g["mass_sens_x_hz"] += float(b["mass_sens_x_hz"])
        g["stiffness_sens_y_hz"] += float(b["stiffness_sens_y_hz"])
        g["mass_sens_y_hz"] += float(b["mass_sens_y_hz"])
        g["compliance_share_pct"] += float(b["_comp_pct"])
        g["cost_usd"] += float(b["cost_usd"])
        g["mass_kg"] += float(b["mass_kg"])

    stiffen_pool: list[dict[str, Any]] = []
    for gid, g in group_acc.items():
        stiff_fund = float(g["stiffness_sens_hz"])
        mass_fund = float(g["mass_sens_hz"])
        stiff_x = float(g["stiffness_sens_x_hz"])
        stiff_y = float(g["stiffness_sens_y_hz"])
        mass_x = float(g["mass_sens_x_hz"])
        mass_y = float(g["mass_sens_y_hz"])
        net_fund = stiff_fund + mass_fund
        xy_net = w_x * (stiff_x + mass_x) + w_y * (stiff_y + mass_y)
        f_min_xy = min(f_x, f_y)
        stiff_xy_blend = w_x * max(stiff_x, 0.0) + w_y * max(stiff_y, 0.0)
        stiff_xy_scaled = (
            max(
                stiff_x * (f_min_xy / max(f_x, 1.0)),
                stiff_y * (f_min_xy / max(f_y, 1.0)),
            )
            * 0.75
        )
        eff_stiff_hz, f_ref_hz = max(
            (
                (stiff_fund, f_fund),
                (stiff_xy_blend, w_x * f_x + w_y * f_y),
                (stiff_xy_scaled, f_min_xy),
            ),
            key=lambda pair: pair[0],
        )
        eff_net_hz = max(net_fund, xy_net)
        comp_pct = float(g["compliance_share_pct"])
        cost_usd = max(float(g["cost_usd"]), 1.0)
        if not ((eff_stiff_hz > 1e-4 and (eff_net_hz > 0.0 or comp_pct > 0.5)) or comp_pct > 1.0):
            continue
        # Frozen-mode Rayleigh quotient gain for +50% stiffness:
        # Since df/d(ln E) = (f / 2) * s_e, 0.5 * s_e = eff_stiff_hz / f_ref_hz,
        # giving exact df = f * (sqrt(1 + 0.5 * s_e) - 1) = f * (sqrt(1 + eff_stiff_hz / f) - 1).
        f_ref_safe = max(float(f_ref_hz), 1e-6)
        rayleigh_gain_hz = f_ref_safe * (
            math.sqrt(1.0 + max(eff_stiff_hz, 0.0) / f_ref_safe) - 1.0
        )
        est_hz_gain = round(rayleigh_gain_hz, 1)
        hz_per_usd = eff_net_hz / cost_usd
        impact_score = comp_pct * 0.65 + est_hz_gain * (1.2 + comp_pct / 20.0)
        stiffen_pool.append(
            {
                "id": gid,
                "role": g["role"],
                "label": g["label"],
                "hz_per_usd": round(hz_per_usd, 4),
                "compliance_share_pct": round(comp_pct, 2),
                "est_hz_gain": est_hz_gain,
                "why_it_hurts": _stiffen_why(gid, comp_pct, est_hz_gain),
                "action_hint": _stiffen_hint(gid, g["role"], spec),
                "_score": impact_score,
            }
        )
    stiffen_pool.sort(
        key=lambda x: (x["_score"], x["est_hz_gain"], x["compliance_share_pct"]),
        reverse=True,
    )
    top_stiffen_targets = [
        {k: v for k, v in item.items() if k != "_score"} for item in stiffen_pool[:5]
    ]

    lighten_groups: dict[str, dict[str, Any]] = {}

    def _add_lighten_group(
        gid: str,
        label: str,
        role: str,
        mass_sens_hz: float,
        net_hz: float,
        mass_kg: float,
        is_moving: bool,
    ) -> None:
        g = lighten_groups.setdefault(
            gid,
            {
                "id": gid,
                "label": label,
                "role": role,
                "mass_sens_hz": 0.0,
                "net_hz": 0.0,
                "mass_kg": 0.0,
                "is_moving": is_moving,
            },
        )
        g["mass_sens_hz"] += mass_sens_hz
        g["net_hz"] += net_hz
        g["mass_kg"] += mass_kg
        g["is_moving"] = g["is_moving"] or is_moving

    for gid, g in group_acc.items():
        net_hz = float(g["stiffness_sens_hz"]) + float(g["mass_sens_hz"])
        if g["mass_sens_hz"] < -1e-4 and net_hz < 0.0 and g["mass_kg"] > 1e-5:
            _add_lighten_group(
                gid=gid,
                label=g["label"],
                role=g["role"],
                mass_sens_hz=float(g["mass_sens_hz"]),
                net_hz=net_hz,
                mass_kg=float(g["mass_kg"]),
                is_moving=(gid in ("x_gantry_beam", "delta_arms")),
            )

    # Moving lumped-mass bodies grouped by physical assembly (toolhead, carriages, moving bed)
    th_group_ids = [
        nid
        for nid in (
            assembly.toolhead_node_id,
            assembly.toolhead_cog_node_id,
            assembly.nozzle_node_id,
        )
        if nid in lumped_mass_sensitivities
    ]
    th_unique = list(dict.fromkeys(th_group_ids))
    if th_unique:
        th_ms = sum(lumped_mass_sensitivities[nid]["mass_sens_hz"] for nid in th_unique)
        th_kg = sum(lumped_mass_sensitivities[nid]["mass_kg"] for nid in th_unique)
        if th_ms < -1e-4 and th_kg > 1e-5:
            _add_lighten_group(
                gid="toolhead_assembly",
                label="Toolhead Assembly",
                role="toolhead",
                mass_sens_hz=th_ms,
                net_hz=th_ms,
                mass_kg=th_kg,
                is_moving=True,
            )

    for nid, ld in lumped_mass_sensitivities.items():
        if nid in th_unique:
            continue
        if ld["role"] == "bed" and spec.architecture != "cartesian_bedslinger":
            continue
        if ld["role"] not in ("gantry", "bed") or ld["mass_sens_hz"] >= -1e-4:
            continue
        if nid.startswith("y_car_"):
            lgid, llabel, lrole = "y_carriages", "Y-Axis Gantry Carriages", "gantry"
        elif nid in ("z_l", "z_r"):
            lgid, llabel, lrole = "z_carriages", "Z-Axis Carriages & X-Motor", "gantry"
        elif nid.startswith("c_"):
            lgid, llabel, lrole = "delta_carriages", "Delta Tower Carriages", "gantry"
        elif ld["role"] == "bed":
            lgid, llabel, lrole = "moving_y_bed", "Moving Heated Bed", "bed"
        else:
            lgid, llabel = _classify_sensitivity_group(nid, ld["role"])
            lrole = ld["role"]
        _add_lighten_group(
            gid=lgid,
            label=llabel,
            role=lrole,
            mass_sens_hz=float(ld["mass_sens_hz"]),
            net_hz=float(ld["mass_sens_hz"]),
            mass_kg=float(ld["mass_kg"]),
            is_moving=True,
        )

    lighten_pool: list[dict[str, Any]] = []
    for gid, lg in lighten_groups.items():
        if lg["mass_sens_hz"] >= -1e-4 or lg["mass_kg"] <= 1e-5:
            continue
        hz_per_100g = 0.1 * lg["net_hz"] / lg["mass_kg"]
        hz_gain_100g = round(abs(hz_per_100g), 2)
        if hz_gain_100g <= 0.0:
            continue
        lighten_pool.append(
            {
                "id": gid,
                "role": lg["role"],
                "label": lg["label"],
                "mass_sens_hz": round(lg["mass_sens_hz"], 4),
                "hz_per_100g": round(hz_per_100g, 4),
                "hz_gain_per_100g_saved": hz_gain_100g,
                "why_it_hurts": _lighten_why(gid, hz_gain_100g, bool(lg["is_moving"])),
                "action_hint": _lighten_hint(gid, lg["role"], spec),
                "_is_moving": bool(lg["is_moving"]),
            }
        )

    lighten_pool.sort(
        key=lambda x: (not x["_is_moving"], -x["hz_gain_per_100g_saved"], x["mass_sens_hz"])
    )
    top_lighten_targets = [
        {k: v for k, v in item.items() if k != "_is_moving"} for item in lighten_pool[:3]
    ]

    summary_parts: list[str] = []
    th_sub_pct = subsystem_breakdown["toolhead"]["compliance_share_pct"]
    ga_sub_pct = subsystem_breakdown["gantry"]["compliance_share_pct"]
    fr_sub_pct = subsystem_breakdown["frame"]["compliance_share_pct"]
    be_sub_pct = subsystem_breakdown["bed"]["compliance_share_pct"]
    if primary_problem_subsystem == "toolhead" or th_sub_pct > fr_sub_pct:
        summary_parts.append(
            f"Toolhead & Mount drives {th_sub_pct:.1f}% of nozzle-to-bed flex (vs {fr_sub_pct:.1f}% outer frame)."
        )
    elif primary_problem_subsystem == "gantry" or ga_sub_pct > fr_sub_pct:
        summary_parts.append(
            f"X/Y Gantry & Rails drives {ga_sub_pct:.1f}% of nozzle-to-bed flex (vs {fr_sub_pct:.1f}% outer frame)."
        )
    elif primary_problem_subsystem == "bed":
        summary_parts.append(
            f"Bed & Z-Mount drives {be_sub_pct:.1f}% of nozzle-to-bed flex (vs {fr_sub_pct:.1f}% outer frame)."
        )
    else:
        summary_parts.append(
            f"Outer Frame & Panels drives {fr_sub_pct:.1f}% of nozzle-to-bed flex."
        )

    if top_stiffen_targets:
        s0 = top_stiffen_targets[0]
        summary_parts.append(
            f"Primary stiffness bottleneck: {s0['label']} — {s0['action_hint'].rstrip('.')}."
        )
    if top_lighten_targets:
        l0 = top_lighten_targets[0]
        gain = float(l0["hz_gain_per_100g_saved"])
        gain_str = f"{gain:.1f}" if gain >= 0.1 else f"{gain:.2f}"
        summary_parts.append(
            f"Primary inertia sensitivity: {l0['label']} (+{gain_str} Hz per 100g reduction)."
        )
    summary_diagnosis = " ".join(summary_parts)

    return SensitivityReport(
        fundamental_freq_hz=f_fund,
        primary_x_freq_hz=f_x,
        primary_y_freq_hz=f_y,
        static_rel_disp_x_um=u_rel_x * 1e6,
        static_rel_disp_y_um=u_rel_y * 1e6,
        member_sensitivities=member_sensitivities,
        panel_bay_sensitivities=panel_bay_sensitivities,
        lumped_mass_sensitivities=lumped_mass_sensitivities,
        top_stiffen_targets=top_stiffen_targets,
        top_lighten_targets=top_lighten_targets,
        subsystem_breakdown=subsystem_breakdown,
        primary_problem_subsystem=primary_problem_subsystem,
        summary_diagnosis=summary_diagnosis,
    )
