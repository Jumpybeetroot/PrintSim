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
    print_accel_override: float | None = None
    travel_accel_override: float | None = None
    last_ext_z = -1.0
    layer_idx = 0

    for raw_line in gcode_text.splitlines():
        line = raw_line.split(";", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        cmd = tokens[0].upper()
        if cmd in ("G00", "G01", "G02", "G03"):
            cmd = f"G{int(cmd[1:])}"

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
            if "S" in params:
                print_accel_override = params["S"]
                travel_accel_override = params["S"]
            if "P" in params:
                print_accel_override = params["P"]
            if "T" in params:
                travel_accel_override = params["T"]
        elif cmd == "SET_VELOCITY_LIMIT":
            for tok in tokens[1:]:
                if "=" in tok:
                    key, val_str = tok.split("=", 1)
                    if key.upper() == "ACCEL":
                        try:
                            accel_val = float(val_str)
                            if accel_val > 0.0:
                                print_accel_override = accel_val
                                travel_accel_override = accel_val
                        except ValueError:
                            continue
        elif cmd in ("G0", "G1", "G2", "G3"):
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

            def _emit_segment(p0: np.ndarray, p1: np.ndarray, seg_de: float) -> None:
                nonlocal last_ext_z, layer_idx
                seg_len = float(np.linalg.norm(p1 - p0))
                has_xyz = seg_len > 1e-5
                has_e = abs(seg_de) > 1e-6
                if not has_xyz and not has_e:
                    return
                is_ext = has_xyz and seg_de > 1e-6
                if is_ext and p1[2] > last_ext_z + 0.02:
                    if last_ext_z >= 0:
                        layer_idx += 1
                    last_ext_z = float(p1[2])
                seg_accel = (
                    print_accel_override
                    if is_ext
                    else (
                        travel_accel_override
                        if travel_accel_override is not None
                        else print_accel_override
                    )
                )
                moves.append(
                    GCodeMove(
                        index=len(moves),
                        start_mm=(float(p0[0]), float(p0[1]), float(p0[2])),
                        end_mm=(float(p1[0]), float(p1[1]), float(p1[2])),
                        extrude_mm=float(seg_de),
                        feedrate_mm_s=float(feedrate),
                        is_extrusion=is_ext,
                        layer_index=layer_idx,
                        accel_override_mm_s2=seg_accel,
                    )
                )

            if cmd in ("G0", "G1"):
                _emit_segment(pos, new_pos, de)
            else:
                has_center = False
                cx = float(pos[0])
                cy = float(pos[1])
                if "I" in params or "J" in params:
                    cx = float(pos[0]) + params.get("I", 0.0)
                    cy = float(pos[1]) + params.get("J", 0.0)
                    has_center = True
                elif "R" in params:
                    dx = float(new_pos[0] - pos[0])
                    dy = float(new_pos[1] - pos[1])
                    d = math.hypot(dx, dy)
                    if d > 1e-9:
                        r_val = params["R"]
                        r_abs = abs(r_val)
                        h = math.sqrt(max(0.0, r_abs * r_abs - 0.25 * d * d))
                        mx = float(pos[0]) + 0.5 * dx
                        my = float(pos[1]) + 0.5 * dy
                        is_ccw = cmd == "G3"
                        sign = 1.0 if (is_ccw == (r_val > 0.0)) else -1.0
                        cx = mx + sign * h * (-dy / d)
                        cy = my + sign * h * (dx / d)
                        has_center = True

                if has_center:
                    r_start = math.hypot(float(pos[0]) - cx, float(pos[1]) - cy)
                    r_end = math.hypot(float(new_pos[0]) - cx, float(new_pos[1]) - cy)
                    if r_start <= 1e-6:
                        _emit_segment(pos, new_pos, de)
                    else:
                        theta0 = math.atan2(float(pos[1]) - cy, float(pos[0]) - cx)
                        theta1 = math.atan2(float(new_pos[1]) - cy, float(new_pos[0]) - cx)
                        xy_dist = math.hypot(
                            float(new_pos[0] - pos[0]), float(new_pos[1] - pos[1])
                        )
                        if cmd == "G3":
                            if xy_dist <= 1e-5:
                                d_theta = 2.0 * math.pi
                            else:
                                d_theta = (theta1 - theta0) % (2.0 * math.pi)
                                if d_theta <= 1e-9:
                                    d_theta = 2.0 * math.pi
                        else:
                            if xy_dist <= 1e-5:
                                d_theta = -2.0 * math.pi
                            else:
                                d_theta = -((theta0 - theta1) % (2.0 * math.pi))
                                if d_theta >= -1e-9:
                                    d_theta = -2.0 * math.pi

                        r_avg = 0.5 * (r_start + r_end)
                        arc_len = r_avg * abs(d_theta)
                        n_seg = max(
                            1,
                            math.ceil(arc_len / 1.0),
                            math.ceil(abs(d_theta) / math.radians(5.0)),
                        )
                        pts = [pos.copy()]
                        for k in range(1, n_seg):
                            t = k / n_seg
                            th = theta0 + t * d_theta
                            rk = r_start + t * (r_end - r_start)
                            zk = float(pos[2]) + t * float(new_pos[2] - pos[2])
                            pts.append(
                                np.array(
                                    [cx + rk * math.cos(th), cy + rk * math.sin(th), zk],
                                    dtype=float,
                                )
                            )
                        pts.append(new_pos.copy())

                        seg_lens = [
                            float(np.linalg.norm(pts[k + 1] - pts[k])) for k in range(n_seg)
                        ]
                        total_seg_len = sum(seg_lens)
                        for k in range(n_seg):
                            seg_de = (
                                de * (seg_lens[k] / total_seg_len)
                                if total_seg_len > 1e-12
                                else (de if k == 0 else 0.0)
                            )
                            _emit_segment(pts[k], pts[k + 1], seg_de)
                else:
                    _emit_segment(pos, new_pos, de)

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
