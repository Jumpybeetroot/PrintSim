"""G-code parser (Marlin/Klipper/OrcaSlicer/PrusaSlicer) & procedural benchmark generator."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class GCodeMove:
    index: int
    start_mm: tuple[float, float, float]
    end_mm: tuple[float, float, float]
    extrude_mm: float
    feedrate_mm_s: float
    is_extrusion: bool
    layer_index: int
    accel_override_mm_s2: float | None = None

    @property
    def length_mm(self) -> float:
        dx = self.end_mm[0] - self.start_mm[0]
        dy = self.end_mm[1] - self.start_mm[1]
        dz = self.end_mm[2] - self.start_mm[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)


def parse_gcode(gcode_text: str) -> list[GCodeMove]:
    """Parse standard G-code text into a sequence of linear 3D moves."""
    moves: list[GCodeMove] = []
    pos = np.array([0.0, 0.0, 0.0], dtype=float)
    e_pos = 0.0
    feedrate = 60.0  # mm/s default
    abs_xyz = True
    abs_e = True
    accel_override: float | None = None
    last_ext_z = -1.0
    layer_idx = 0

    for raw_line in gcode_text.splitlines():
        line = raw_line.split(";", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        cmd = tokens[0].upper()

        params: dict[str, float] = {}
        for tok in tokens[1:]:
            if len(tok) >= 2 and tok[0].isalpha():
                try:
                    params[tok[0].upper()] = float(tok[1:])
                except ValueError:
                    continue

        if cmd == "G90":
            abs_xyz = True
        elif cmd == "G91":
            abs_xyz = False
        elif cmd == "M82":
            abs_e = True
        elif cmd == "M83":
            abs_e = False
        elif cmd == "G28":
            pos[:] = 0.0
        elif cmd == "G92":
            if "X" in params:
                pos[0] = params["X"]
            if "Y" in params:
                pos[1] = params["Y"]
            if "Z" in params:
                pos[2] = params["Z"]
            if "E" in params:
                e_pos = params["E"]
        elif cmd == "M204":
            if "P" in params:
                accel_override = params["P"]
            elif "S" in params:
                accel_override = params["S"]
        elif cmd in ("G0", "G1"):
            if "F" in params and params["F"] > 0:
                feedrate = params["F"] / 60.0  # mm/min -> mm/s

            new_pos = pos.copy()
            for ax_idx, ax_key in enumerate(("X", "Y", "Z")):
                if ax_key in params:
                    new_pos[ax_idx] = params[ax_key] if abs_xyz else pos[ax_idx] + params[ax_key]

            de = 0.0
            if "E" in params:
                if abs_e:
                    de = params["E"] - e_pos
                    e_pos = params["E"]
                else:
                    de = params["E"]
                    e_pos += de

            is_ext = de > 1e-6 and float(np.linalg.norm(new_pos - pos)) > 1e-5
            if is_ext and new_pos[2] > last_ext_z + 0.02:
                if last_ext_z >= 0:
                    layer_idx += 1
                last_ext_z = float(new_pos[2])

            if float(np.linalg.norm(new_pos - pos)) > 1e-5:
                moves.append(
                    GCodeMove(
                        index=len(moves),
                        start_mm=(float(pos[0]), float(pos[1]), float(pos[2])),
                        end_mm=(float(new_pos[0]), float(new_pos[1]), float(new_pos[2])),
                        extrude_mm=float(max(de, 0.0)),
                        feedrate_mm_s=float(feedrate),
                        is_extrusion=is_ext,
                        layer_index=layer_idx,
                        accel_override_mm_s2=accel_override,
                    )
                )
            pos = new_pos

    return moves


def generate_benchmark_gcode(
    build_center_xy: tuple[float, float] = (150.0, 150.0),
    size_mm: float = 40.0,
    height_mm: float = 20.0,
    layer_height_mm: float = 0.25,
    print_speed_mm_s: float = 240.0,
    infill_speed_mm_s: float = 340.0,
    travel_speed_mm_s: float = 450.0,
) -> str:
    """Generate a synthetic calibration cube + sharp-corner ringing & flow benchmark G-code."""
    cx, cy = build_center_xy
    half = size_mm * 0.5
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half

    # Filament E per mm of 0.42mm width x layer_height extrusion for 1.75mm filament
    fil_area = math.pi * (1.75 * 0.5) ** 2
    e_per_mm = (0.42 * layer_height_mm) / fil_area

    lines = [
        "; PrintSim Standard Calibration & High-Speed Torture Cube",
        "G90 ; absolute positioning",
        "M83 ; relative extrusion",
        "G28 ; home",
        f"G0 X{x0:.2f} Y{y0:.2f} Z{layer_height_mm:.2f} F{travel_speed_mm_s * 60:.0f}",
    ]

    n_layers = max(1, int(round(height_mm / layer_height_mm)))
    f_perim = print_speed_mm_s * 60.0
    f_infill = infill_speed_mm_s * 60.0
    f_travel = travel_speed_mm_s * 60.0

    for layer in range(n_layers):
        z = (layer + 1) * layer_height_mm
        lines.append(f"; LAYER:{layer}")
        lines.append(f"G0 X{x0:.2f} Y{y0:.2f} Z{z:.2f} F{f_travel:.0f}")

        # Outer perimeter (tests sharp 90-degree corners & ringing)
        perim_pts = [(x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        px, py = x0, y0
        for nx, ny in perim_pts:
            dist = math.hypot(nx - px, ny - py)
            lines.append(f"G1 X{nx:.2f} Y{ny:.2f} E{dist * e_per_mm:.4f} F{f_perim:.0f}")
            px, py = nx, ny

        # Inner high-speed diagonal/zigzag infill (tests volumetric flow & high accel)
        inset = 2.0
        ix0, ix1 = x0 + inset, x1 - inset
        iy0, iy1 = y0 + inset, y1 - inset
        step = 6.0
        lines.append(f"G0 X{ix0:.2f} Y{iy0:.2f} F{f_travel:.0f}")
        px, py = ix0, iy0
        y_cur = iy0
        direction = 1
        while y_cur <= iy1 + 1e-3:
            target_x = ix1 if direction == 1 else ix0
            dist = abs(target_x - px)
            if dist > 1e-3:
                lines.append(f"G1 X{target_x:.2f} Y{y_cur:.2f} E{dist * e_per_mm:.4f} F{f_infill:.0f}")
                px = target_x
            next_y = y_cur + step
            if next_y <= iy1 + 1e-3:
                lines.append(f"G1 X{px:.2f} Y{next_y:.2f} E{step * e_per_mm:.4f} F{f_infill:.0f}")
                py = next_y
            y_cur = next_y
            direction *= -1

    return "\n".join(lines) + "\n"
