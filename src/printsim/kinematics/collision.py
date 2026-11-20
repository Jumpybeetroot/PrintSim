"""Frame envelope clearance and G-code toolpath collision checks."""

from __future__ import annotations

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
    # Z clearance: bed assembly (~75mm) + build Z + gantry/toolhead overhead (~80mm)
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
    bx, by, bz = assembly.spec.kinematics.build_volume_mm
    issues: list[str] = []
    min_pt = np.min(points_mm, axis=0)
    max_pt = np.max(points_mm, axis=0)
    if min_pt[0] < -0.5 or max_pt[0] > bx + 0.5:
        issues.append(f"Toolpath X range [{min_pt[0]:.1f}, {max_pt[0]:.1f}] exceeds [0, {bx:.0f}] mm")
    if min_pt[1] < -0.5 or max_pt[1] > by + 0.5:
        issues.append(f"Toolpath Y range [{min_pt[1]:.1f}, {max_pt[1]:.1f}] exceeds [0, {by:.0f}] mm")
    if min_pt[2] < -0.5 or max_pt[2] > bz + 0.5:
        issues.append(f"Toolpath Z range [{min_pt[2]:.1f}, {max_pt[2]:.1f}] exceeds [0, {bz:.0f}] mm")
    return issues
