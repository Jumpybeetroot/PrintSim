"""Frame envelope clearance and G-code toolpath collision checks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from printsim.geometry.builder import AssemblyModel
from printsim.schema import get_profile


@dataclass
class CollisionReport:
    has_collision: bool
    clearance_x_mm: float
    clearance_y_mm: float
    clearance_z_mm: float
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "has_collision": self.has_collision,
            "clearance_x_mm": round(self.clearance_x_mm, 1),
            "clearance_y_mm": round(self.clearance_y_mm, 1),
            "clearance_z_mm": round(self.clearance_z_mm, 1),
            "issues": self.issues,
        }


def check_frame_clearance(assembly: AssemblyModel) -> CollisionReport:
    """Verify toolhead & bed stay clear of frame extrusions across the full build volume."""
    spec = assembly.spec
    fw, fd, fh = spec.frame.width_mm, spec.frame.depth_mm, spec.frame.height_mm
    bx, by, bz = spec.kinematics.build_volume_mm
    ex, ey, ez = spec.toolhead.envelope_mm
    prof = get_profile(spec.frame.profile)
    beam_w = prof.width_m * 1000.0

    issues: list[str] = []

    # Usable interior span after deducting frame beam thickness and toolhead half-widths on both sides
    clear_x = (fw - 2.0 * beam_w) - (bx + ex)
    clear_y = (fd - 2.0 * beam_w) - (by + ey)
    max_z_car = bz
    if spec.architecture == "delta":
        from printsim.kinematics.solvers import inverse_kinematics

        r_bed = 0.5 * min(bx, by)
        cx_bed, cy_bed = 0.5 * bx, 0.5 * by
        r_test = min(r_bed, spec.kinematics.delta_radius_mm)
        max_z_car = bz + spec.kinematics.delta_arm_length_mm
        z_candidates: list[float] = []
        for ang_deg in (90.0, 210.0, 330.0):
            ang = math.radians(ang_deg)
            px = cx_bed + r_test * math.cos(ang)
            py = cy_bed + r_test * math.sin(ang)
            try:
                z_candidates.extend(inverse_kinematics(spec, (px, py, bz)))
            except ValueError:
                pass
        if z_candidates:
            max_z_car = max(z_candidates)
        # Delta tower stack: bed height above base + top carriage/pulley/frame clearances
        carriage_clearance = beam_w + 18.0
        effector_top_clearance = beam_w + 25.0
        clear_z = fh - (max_z_car + carriage_clearance + effector_top_clearance)
    elif (
        spec.architecture == "corexy"
        and bool(spec.frame.gantry_on_top_frame)
        and not spec.is_flying_gantry
    ):
        # Gantry plane IS the top ring (VzBot): parts above the X beam live in the top hat, so the
        # Z budget is the explicit stack bottom ring + bed carriage + build Z + nozzle drop + top-ring
        # half height (VzBot 330 CAD: 20 + 80 + 400 + 54.2 + 10 = 564.2 <= 570 mm, 7.8 mm spare).
        from printsim.schema.overrides import resolve_bed_mount

        nozzle_drop = abs(float(spec.toolhead.nozzle_offset_mm[2]))
        if (
            spec.gantry.effective_x_rail_mount == "top"
            and spec.toolhead.offset_datum_rail_mount != "top"
        ):
            nozzle_drop += 6.0  # builder top-mount datum shift (_attach_toolhead_cog_and_nozzle)
        bed_cfg = resolve_bed_mount(spec)
        # PEI + plate + mount standoff + bed sub-frame extrusion + ~35 mm Z-carriage / bearing-mount drop
        bed_stack = (
            1.0
            + float(spec.bed.plate_thickness_mm)
            + float(bed_cfg.get("standoff_height_mm", 14.0))
            + prof.height_m * 1000.0
            + 35.0
        )
        overhead_z = prof.height_m * 1000.0 * 1.5 + bed_stack + nozzle_drop
        clear_z = fh - (bz + overhead_z)
    else:
        # Z clearance: bed + top frame stack (~85 mm) + build Z + upper toolhead half-height (0.5 * ez)
        overhead_z = 85.0 + 0.5 * ez
        clear_z = fh - (bz + overhead_z)

    if spec.architecture == "cartesian_bedslinger":
        # Bedslinger bed sweeps front-to-back beyond frame depth; frame width still constrains X
        clear_y = fd - (by + 40.0)

    if clear_x < 0:
        issues.append(
            f"X interference: toolhead collides with frame uprights by {abs(clear_x):.1f} mm "
            f"(frame width {fw:.0f}mm too narrow for build X {bx:.0f}mm + toolhead {ex:.0f}mm)."
        )
    if clear_y < 0:
        issues.append(
            f"Y interference: toolhead/gantry collides with front/rear frame by {abs(clear_y):.1f} mm."
        )
    if clear_z < 0:
        if spec.architecture == "delta":
            issues.append(
                f"Z interference: delta frame height {fh:.0f}mm is {abs(clear_z):.1f} mm too short "
                f"for max carriage height {max_z_car:.1f}mm at Z={bz:.0f}mm."
            )
        else:
            issues.append(
                f"Z interference: insufficient frame height by {abs(clear_z):.1f} mm for Z={bz:.0f}mm build height."
            )

    # Check bed fits build volume
    bed_x, bed_y = spec.bed.size_mm
    if bed_x < bx - 1.0 or bed_y < by - 1.0:
        issues.append(
            f"Bed plate ({bed_x:.0f}x{bed_y:.0f}mm) is smaller than build volume ({bx:.0f}x{by:.0f}mm)."
        )

    return CollisionReport(
        has_collision=len(issues) > 0,
        clearance_x_mm=clear_x,
        clearance_y_mm=clear_y,
        clearance_z_mm=clear_z,
        issues=issues,
    )


def check_trajectory_bounds(
    assembly: AssemblyModel, points_mm: np.ndarray
) -> list[str]:
    """Check an Nx3 array of toolpath coordinates (mm) against build volume bounds."""
    if points_mm.size == 0:
        return []
    pts = np.atleast_2d(points_mm)
    bx, by, bz = assembly.spec.kinematics.build_volume_mm
    issues: list[str] = []
    min_pt = np.min(pts, axis=0)
    max_pt = np.max(pts, axis=0)
    if assembly.spec.architecture == "delta":
        r_bed = 0.5 * min(bx, by)
        radial_dist = np.hypot(pts[:, 0] - 0.5 * bx, pts[:, 1] - 0.5 * by)
        max_r = float(np.max(radial_dist))
        if max_r > r_bed + 0.5:
            issues.append(
                f"Toolpath radial reach {max_r:.1f} mm exceeds delta circular bed radius {r_bed:.1f} mm"
            )
    else:
        if min_pt[0] < -0.5 or max_pt[0] > bx + 0.5:
            issues.append(f"Toolpath X range [{min_pt[0]:.1f}, {max_pt[0]:.1f}] exceeds [0, {bx:.0f}] mm")
        if min_pt[1] < -0.5 or max_pt[1] > by + 0.5:
            issues.append(f"Toolpath Y range [{min_pt[1]:.1f}, {max_pt[1]:.1f}] exceeds [0, {by:.0f}] mm")
    if min_pt[2] < -0.5 or max_pt[2] > bz + 0.5:
        issues.append(f"Toolpath Z range [{min_pt[2]:.1f}, {max_pt[2]:.1f}] exceeds [0, {bz:.0f}] mm")
    return issues
