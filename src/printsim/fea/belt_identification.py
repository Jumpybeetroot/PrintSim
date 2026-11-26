"""Identify a belt's in-place stiffness from measurements a printer owner can make.

The belt drive of one axis is a chain of springs under the moving mass ``m``:

    1 / (m (2 pi f)^2) = C_0 + G / EA

``f``   measured axis resonance (the main input-shaper peak) [Hz]
``G``   belt-path geometry factor [m]: ``1 / sum(1 / L_i)`` over the parallel belt branches
        between carriage and motor pulleys, fixed by frame size and toolhead position
``EA``  small-amplitude tensile stiffness of the belt as installed [N per unit strain]
``C_0`` everything else in series: tooth mesh, motor, mounts, frame, toolhead [m/N]

``G`` is the only term that changes with toolhead position, and ``EA`` the only one that changes
when the belt is swapped or re-tensioned. Three experiments therefore separate the belt from
the rest of the machine without assuming a datasheet-to-dynamic conversion factor:

* **Position sweep** - the same axis measured at several toolhead positions (`G` varies).
* **Belt swap** - the same machine with two belts of known stiffness ratio (`EA` varies).
* **Tension sweep** - the same belt at two pluck frequencies (`EA(T)` varies).

Tension itself comes from the plucked-span frequency, which is exact string physics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from printsim.geometry.builder import AssemblyModel

__all__ = [
    "BeltIdentification",
    "axis_compliance_m_per_n",
    "belt_share_from_swap",
    "belt_stiffness_ratio_from_tension_sweep",
    "cord_geometry_factor_m",
    "identify_belt_from_position_sweep",
    "identify_belt_from_swap",
    "pluck_frequency_hz",
    "tension_from_pluck_n",
]


def pluck_frequency_hz(
    tension_n: float, span_m: float, linear_mass_kg_m: float, flexural_rigidity_nm2: float = 0.0
) -> float:
    """Fundamental transverse frequency of a taut belt span with bending stiffness.

    ``f = (1 / 2L) sqrt(T / mu) sqrt(1 + pi^2 EI / (T L^2))`` (pinned ends). The bending term
    makes a short, slack span read high, as the Gates sonic-meter manual warns.
    """
    t = max(float(tension_n), 1e-9)
    length = max(float(span_m), 1e-6)
    mu = max(float(linear_mass_kg_m), 1e-9)
    ei = max(float(flexural_rigidity_nm2), 0.0)
    return (1.0 / (2.0 * length)) * math.sqrt(t / mu) * math.sqrt(1.0 + (math.pi**2) * ei / (t * length * length))


def tension_from_pluck_n(
    freq_hz: float, span_m: float, linear_mass_kg_m: float, flexural_rigidity_nm2: float = 0.0
) -> float:
    """Belt tension from a plucked-span frequency: ``T = 4 mu L^2 f^2 - pi^2 EI / L^2``.

    Exact inverse of `pluck_frequency_hz`; with ``EI = 0`` it is the Gates sonic-meter formula.
    """
    length = max(float(span_m), 1e-6)
    t = 4.0 * float(linear_mass_kg_m) * length * length * float(freq_hz) ** 2
    t -= (math.pi**2) * max(float(flexural_rigidity_nm2), 0.0) / (length * length)
    return max(t, 0.0)


def axis_compliance_m_per_n(freq_hz: float, moving_mass_kg: float) -> float:
    """Total series compliance seen by the moving mass at its resonance: ``1 / (m (2 pi f)^2)``."""
    return 1.0 / (max(float(moving_mass_kg), 1e-9) * (2.0 * math.pi * max(float(freq_hz), 1e-9)) ** 2)


def cord_geometry_factor_m(
    assembly: AssemblyModel, axis: str, u_rel: float | None = None, v_rel: float | None = None
) -> float:
    """Belt-path geometry factor ``G`` [m] at a toolhead position; cord compliance is ``G / EA``."""
    from printsim.kinematics.solvers import compute_drive_stiffness_details_n_per_m

    details = compute_drive_stiffness_details_n_per_m(assembly, axis, u_rel=u_rel, v_rel=v_rel)
    return float(details["cord_geometry_factor_m"])


@dataclass
class BeltIdentification:
    """Result of separating belt cord stiffness from the rest of the drive chain."""

    method: str
    axis: str
    ea_in_place_n: float
    other_compliance_um_per_n: float
    belt_share_pct: float
    ea_over_datasheet_static: float | None = None
    r_squared: float | None = None
    ea_stderr_n: float | None = None
    n_points: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "axis": self.axis,
            "ea_in_place_n": round(self.ea_in_place_n, 1),
            "other_compliance_um_per_n": round(self.other_compliance_um_per_n, 4),
            "belt_share_pct": round(self.belt_share_pct, 1),
            "ea_over_datasheet_static": (
                None if self.ea_over_datasheet_static is None else round(self.ea_over_datasheet_static, 3)
            ),
            "r_squared": None if self.r_squared is None else round(self.r_squared, 4),
            "ea_stderr_n": None if self.ea_stderr_n is None else round(self.ea_stderr_n, 1),
            "n_points": self.n_points,
            "notes": list(self.notes),
        }


def _axis_mass_kg(assembly: AssemblyModel, axis: str, moving_mass_kg: float | None) -> float:
    if moving_mass_kg is not None:
        return float(moving_mass_kg)
    return float(assembly.x_moving_mass_kg if axis.upper() == "X" else assembly.y_moving_mass_kg)


def identify_belt_from_position_sweep(
    assembly: AssemblyModel,
    axis: str,
    points: list[tuple[float, float, float]],
    moving_mass_kg: float | None = None,
) -> BeltIdentification:
    """Fit ``1 / (m w^2) = C_0 + G(u, v) / EA`` to one axis measured at several positions.

    ``points`` are ``(u_rel, v_rel, freq_hz)`` with positions as fractions of X and Y travel.
    Needs at least three positions whose geometry factors differ; ``C_0`` is taken as position
    independent, so use positions that move the belt branches (not just the frame load path).
    """
    if len(points) < 3:
        raise ValueError("position sweep needs at least 3 (u_rel, v_rel, freq_hz) points")
    m = _axis_mass_kg(assembly, axis, moving_mass_kg)
    g = np.array([cord_geometry_factor_m(assembly, axis, u, v) for u, v, _ in points], dtype=float)
    y = np.array([axis_compliance_m_per_n(f, m) for _, _, f in points], dtype=float)
    spread = float((g.max() - g.min()) / max(g.mean(), 1e-12))
    notes: list[str] = []
    if spread < 0.05:
        notes.append(
            f"Geometry factor varies only {100.0 * spread:.1f}% across these positions; "
            "measure nearer the travel limits for a usable fit."
        )
    a_mat = np.column_stack([np.ones_like(g), g])
    coef, *_ = np.linalg.lstsq(a_mat, y, rcond=None)
    c0, inv_ea = float(coef[0]), float(coef[1])
    resid = y - a_mat @ coef
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0.0 else 1.0
    if inv_ea <= 0.0:
        raise ValueError(
            "frequency does not fall as the belt branches lengthen: the belt is not the "
            "position-dependent compliance in these measurements"
        )
    ea = 1.0 / inv_ea
    ea_stderr = None
    dof = len(points) - 2
    if dof > 0:
        sigma2 = float(np.sum(resid**2)) / dof
        cov = sigma2 * np.linalg.inv(a_mat.T @ a_mat)
        ea_stderr = math.sqrt(max(float(cov[1, 1]), 0.0)) * ea * ea
    if c0 < 0.0:
        notes.append("Fitted non-belt compliance is negative; check the moving mass and positions.")
    belt_c = float(np.mean(g)) * inv_ea
    static = float(assembly.spec.gantry.belt.static_stiffness_ea_n)
    return BeltIdentification(
        method="position_sweep",
        axis=axis.upper(),
        ea_in_place_n=ea,
        other_compliance_um_per_n=c0 * 1e6,
        belt_share_pct=100.0 * belt_c / max(belt_c + c0, 1e-18),
        ea_over_datasheet_static=ea / static if static > 0.0 else None,
        r_squared=r2,
        ea_stderr_n=ea_stderr,
        n_points=len(points),
        notes=notes,
    )


def belt_share_from_swap(freq_a_hz: float, freq_b_hz: float, ea_ratio_b_over_a: float) -> float:
    """Fraction of axis compliance that scales with belt stiffness, from a belt swap.

    Same machine, tension and position; belt B is ``rho = EA_B / EA_A`` times as stiff:
    ``(f_A / f_B)^2 = 1 - s (1 - 1 / rho)`` gives the share ``s`` carried by belt A.
    """
    rho = float(ea_ratio_b_over_a)
    if abs(1.0 - 1.0 / rho) < 1e-9:
        raise ValueError("belt swap needs two belts of different stiffness")
    return (1.0 - (float(freq_a_hz) / float(freq_b_hz)) ** 2) / (1.0 - 1.0 / rho)


def identify_belt_from_swap(
    assembly: AssemblyModel,
    axis: str,
    freq_a_hz: float,
    freq_b_hz: float,
    ea_ratio_b_over_a: float,
    u_rel: float | None = None,
    v_rel: float | None = None,
    moving_mass_kg: float | None = None,
) -> BeltIdentification:
    """In-place stiffness of belt A from the resonance with belt A and with a stiffer belt B."""
    share = belt_share_from_swap(freq_a_hz, freq_b_hz, ea_ratio_b_over_a)
    notes: list[str] = []
    if not 0.0 < share <= 1.0:
        notes.append(
            f"Belt share {share:.2f} is outside 0-1: the assumed stiffness ratio or one of the "
            "frequencies is inconsistent with a pure belt change."
        )
    m = _axis_mass_kg(assembly, axis, moving_mass_kg)
    c_tot = axis_compliance_m_per_n(freq_a_hz, m)
    g = cord_geometry_factor_m(assembly, axis, u_rel, v_rel)
    belt_c = max(share, 1e-9) * c_tot
    ea = g / belt_c
    static = float(assembly.spec.gantry.belt.static_stiffness_ea_n)
    return BeltIdentification(
        method="belt_swap",
        axis=axis.upper(),
        ea_in_place_n=ea,
        other_compliance_um_per_n=(c_tot - belt_c) * 1e6,
        belt_share_pct=100.0 * share,
        ea_over_datasheet_static=ea / static if static > 0.0 else None,
        n_points=2,
        notes=notes,
    )


def belt_stiffness_ratio_from_tension_sweep(
    freq_low_tension_hz: float, freq_high_tension_hz: float, belt_share_at_high_tension: float
) -> float:
    """``EA(T_low) / EA(T_high)`` from the axis resonance at two tensions on one machine.

    With belt share ``s`` at the higher tension, ``(f_hi / f_lo)^2 = 1 + s (EA_hi / EA_lo - 1)``.
    """
    s = float(belt_share_at_high_tension)
    if not 0.0 < s <= 1.0:
        raise ValueError("belt share must be within (0, 1]")
    ratio_hi_over_lo = 1.0 + ((float(freq_high_tension_hz) / float(freq_low_tension_hz)) ** 2 - 1.0) / s
    return 1.0 / max(ratio_hi_over_lo, 1e-9)
