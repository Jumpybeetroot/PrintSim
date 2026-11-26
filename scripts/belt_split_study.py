"""Belt-versus-mount stiffness split study (zero-LLM, reproducible).

Community shaper data pins the total stiffness of a belt drive, not how much of it is belt
cord and how much is mounts and brackets. This script does two things with the forward model:

1. ``fit``        sweep belt stiffness and mount stiffness against measured community machines
                  and print the error of the predicted shaper frequency for each combination.
2. ``signatures`` for a stock Trident 300, print what each same-machine experiment (front-to-back
                  position sweep, 6 -> 9 mm belt swap, tension change, 2WD -> AWD) would show under
                  different splits that all reproduce the same centre-of-bed frequencies.

Usage:
    uv run python scripts/belt_split_study.py fit
    uv run python scripts/belt_split_study.py signatures

Measured values are from docs/monolith_discord_benchmarks.md and docs/community_touchstones.md
(recommended shaper type and frequency per axis). Voron entries assume the documented 110 Hz
pluck on a 150 mm span (9.15 N on 6 mm 2GT).
"""

from __future__ import annotations

import itertools
import math
import sys

from printsim.fea import solve_modal_fea
from printsim.geometry import build_assembly
from printsim.schema import PrinterSpec, resolve_gantry_bracket

EXAMPLES = "examples/"


def _load(name: str, **mods: object) -> PrinterSpec:
    spec = PrinterSpec.from_yaml(EXAMPLES + name)
    for dotted, value in mods.items():
        obj = spec
        parts = dotted.split("__")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    return PrinterSpec.model_validate(spec.model_dump())


def _solve(spec: PrinterSpec, belt_scale: float, mount_scale: float, pos=(0.5, 0.5, 0.5)):
    s = spec.model_copy(deep=True)
    s.model_overrides.belt.stiffness_ea_n = belt_scale * s.gantry.belt.stiffness_ea_n
    base_k = resolve_gantry_bracket(s)["stiffness_n_m"] * 1e-3
    s.model_overrides.gantry_bracket.stiffness_n_per_mm = min(200000.0, max(20.0, mount_scale * base_k))
    return solve_modal_fea(build_assembly(s, toolhead_rel_pos=pos), include_adxl_diagnostic=False)


# (label, spec, {axis: (recommended shaper type, measured recommended frequency in Hz)})
def _cases():
    return [
        (
            "M0 AWD 6mm 44.5N",
            _load("monolith_zero_m0.yaml", kinematics__xy_stepper__run_current_a=2.5),
            {"X": ("MZV", 185.2), "Y": ("MZV", 132.0)},
        ),
        (
            "M0 2WD 6mm 44.5N",
            _load("monolith_zero_m0.yaml", gantry__drive_mode="2wd", kinematics__xy_stepper__run_current_a=2.5),
            {"X": ("MZV", 142.2), "Y": ("MZV", 104.0)},
        ),
        ("CW AWD 10mm 89N", _load("cw_fixed_gantry_300.yaml"), {"X": ("MZV", 175.0), "Y": ("ZV", 115.0)}),
        (
            "Trident300 SB 9.15N",
            _load("voron_trident_300.yaml", gantry__belt__pretension_n=9.15),
            {"X": ("MZV", 54.2), "Y": ("MZV", 47.1)},
        ),
        (
            "V2.4 300 SB 9.15N",
            _load("corexy_voron_style.yaml", gantry__belt__pretension_n=9.15),
            {"X": ("MZV", 58.4), "Y": ("MZV", 48.2)},
        ),
    ]


def fit() -> None:
    cases = _cases()
    belt_scales = [0.45, 0.55, 0.7, 0.85, 1.0, 1.2]
    mount_scales = [0.5, 1.0, 2.0, 4.0, 1000.0]
    print("Error (%) of predicted vs measured recommended shaper frequency, X / Y per machine")
    print(f"{'belt x':>7} {'mount x':>8} | " + " | ".join(f"{c[0]:>20s}" for c in cases) + " |  rms %")
    ranked = []
    for bs, ms in itertools.product(belt_scales, mount_scales):
        errs, cells = [], []
        for _, spec, meas in cases:
            modal = _solve(spec, bs, ms)
            pair = []
            for ax, graph in (("X", modal.shaper_graph_x), ("Y", modal.shaper_graph_y)):
                cand = {c["name"]: c["freq_hz"] for c in graph.to_dict()["candidates"]}
                pair.append(100.0 * (cand[meas[ax][0]] / meas[ax][1] - 1.0))
            errs += pair
            cells.append(f"{pair[0]:+8.1f} /{pair[1]:+8.1f}  ")
        rms = math.sqrt(sum(e * e for e in errs) / len(errs))
        ranked.append((rms, bs, ms))
        print(f"{bs:7.2f} {ms:8.1f} | " + " | ".join(cells) + f" | {rms:5.1f}")
    print("Lowest rms:", [(round(r, 1), b, m) for r, b, m in sorted(ranked)[:5]])


def signatures() -> None:
    base = _load("voron_trident_300.yaml")

    def run(bs: float, ms: float, pos=(0.5, 0.5, 0.5), **mods: object):
        spec = _load("voron_trident_300.yaml", **mods) if mods else base
        modal = _solve(spec, bs, ms, pos)
        return modal.shaper_x.effective_freq_hz, modal.shaper_y.effective_freq_hz

    def fit_mount(bs: float, target_y: float) -> float:
        lo, hi = 0.3, 2000.0
        if run(bs, hi)[1] < target_y:
            return hi
        for _ in range(26):
            mid = math.sqrt(lo * hi)
            if run(bs, mid)[1] < target_y:
                lo = mid
            else:
                hi = mid
        return math.sqrt(lo * hi)

    x0, y0 = run(1.0, 1.0)
    print(f"Stock Trident 300 at the preset tension: X {x0:.1f} Hz, Y {y0:.1f} Hz")
    print("Mount stiffness is refitted for each belt stiffness so the centre Y frequency is unchanged.")
    print(
        f"{'belt x':>7} {'mount x':>8} | {'centre X / Y':>13} | {'X front -> back':>20} | "
        f"{'6 -> 9 mm (X, Y)':>18} | {'9 -> 28 N (X, Y)':>18} | {'2WD -> AWD (X, Y)':>18}"
    )
    for bs in (1.0, 0.8, 0.6, 0.45):
        ms = 1.0 if bs == 1.0 else fit_mount(bs, y0)
        cx, cy = run(bs, ms)
        xf, _ = run(bs, ms, pos=(0.5, 0.1, 0.5))
        xb, _ = run(bs, ms, pos=(0.5, 0.9, 0.5))
        x9, y9 = run(bs, ms, gantry__belt__type="gt2_9mm", gantry__belt__pretension_n=42.0)
        xl, yl = run(bs, ms, gantry__belt__pretension_n=9.15)
        xa, ya = run(bs, ms, gantry__drive_mode="awd")
        print(
            f"{bs:7.2f} {ms:8.1f} | {cx:5.1f} / {cy:5.1f} | {xf:5.1f} -> {xb:5.1f} {100 * (xb / xf - 1):+5.1f}% | "
            f"{100 * (x9 / cx - 1):+7.1f}% {100 * (y9 / cy - 1):+7.1f}% | "
            f"{100 * (cx / xl - 1):+7.1f}% {100 * (cy / yl - 1):+7.1f}% | "
            f"{100 * (xa / cx - 1):+7.1f}% {100 * (ya / cy - 1):+7.1f}%"
        )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "signatures"
    if mode == "fit":
        fit()
    elif mode == "signatures":
        signatures()
    else:
        raise SystemExit("usage: belt_split_study.py [fit|signatures]")
