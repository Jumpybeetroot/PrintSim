"""Klipper-exact input-shaper physics, synthetic ADXL345 PSD synthesis, and calibration.

References & Prior Art:
- Klipper firmware `klippy/extras/shaper_defs.py` (`ZV`, `MZV`, `EI`, `2HUMP_EI`,
  `3HUMP_EI` impulse amplitudes/timings and `SHAPER_VIBRATION_REDUCTION = 20.0`).
- Klipper firmware `klippy/extras/shaper_calibrate.py` (`_estimate_shaper`,
  `_get_shaper_smoothing`, `find_shaper_max_accel`, and Klipper's composite
  smoothing/vibration scoring function).
- Monolith / Kalico community input-shaper benchmarks (`docs/monolith_discord_benchmarks.md`):
  frequency grid extended from Klipper's stock 150 Hz limit up to 220 Hz so high-stiffness
  AWD CoreXY builds (160–190+ Hz resonances) calibrate cleanly without clipping.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

SHAPER_NAMES: tuple[str, ...] = ("ZV", "MZV", "EI", "2HUMP_EI", "3HUMP_EI")
ADVANCED_SHAPER_NAMES: tuple[str, ...] = ("ZVD", "ZVDD", "SMOOTH_ZV", "SMOOTH_EI", "SMOOTH_FLATNESS")
DEFAULT_SHAPER_DAMPING_RATIO: float = 0.1
TEST_DAMPING_RATIOS: tuple[float, ...] = (0.075, 0.1, 0.15)
DEFAULT_TARGET_SMOOTHING: float = 0.12
SHAPER_VIBRATION_REDUCTION: float = 20.0  # Klipper shaper_defs.SHAPER_VIBRATION_REDUCTION
EI_SHAPER_VIB_TOL: float = 0.05  # Klipper shaper_defs default shaper_vib_tol
NONE_SHAPER_ALLOWABLE_RINGING_MM: float = 0.0228  # ~22.8 um allowable unshaped ringing (0.9 / (2*pi)^2)
MIN_FREQ_HZ: float = 5.0
MAX_FREQ_HZ: float = 220.0
NUM_FREQ_BINS: int = 108


@dataclass
class ShaperCandidate:
    """Fitted candidate Klipper input shaper evaluated against an axis PSD."""

    name: str
    freq_hz: float
    vibrations_pct: float
    smoothing: float
    max_accel_mm_s2: float
    score: float
    attenuation_curve: np.ndarray

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "freq_hz": round(float(self.freq_hz), 2),
            "vibrations_pct": round(float(self.vibrations_pct), 2),
            "smoothing": round(float(self.smoothing), 5),
            "max_accel_mm_s2": round(float(self.max_accel_mm_s2), 0),
            "score": round(float(self.score), 6),
            "attenuation_curve": [round(float(v), 4) for v in self.attenuation_curve],
        }


@dataclass
class AxisShaperGraph:
    """Full ADXL345-style PSD and Klipper input-shaper calibration report for one axis."""

    axis: str
    freqs_hz: np.ndarray
    raw_psd: np.ndarray
    shaped_psd: np.ndarray
    peak_freq_hz: float
    damping_ratio: float
    recommended_shaper: str
    recommended_freq_hz: float
    recommended_max_accel_mm_s2: float
    candidates: list[ShaperCandidate]

    def to_dict(self) -> dict:
        return {
            "axis": self.axis,
            "freqs_hz": [round(float(v), 2) for v in self.freqs_hz],
            "raw_psd": [round(float(v), 3) for v in self.raw_psd],
            "shaped_psd": [round(float(v), 3) for v in self.shaped_psd],
            "peak_freq_hz": round(float(self.peak_freq_hz), 2),
            "damping_ratio": round(float(self.damping_ratio), 4),
            "recommended_shaper": self.recommended_shaper,
            "recommended_freq_hz": round(float(self.recommended_freq_hz), 2),
            "recommended_max_accel_mm_s2": round(float(self.recommended_max_accel_mm_s2), 0),
            "candidates": [c.to_dict() for c in self.candidates],
        }


def get_zvd_shaper(
    shaper_freq_hz: float,
    damping_ratio: float = DEFAULT_SHAPER_DAMPING_RATIO,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact Klipper ZVD `(A, T)` impulse amplitudes and times (s) from `shaper_defs.py`."""
    if shaper_freq_hz <= 0.0:
        raise ValueError(f"shaper_freq_hz must be positive, got {shaper_freq_hz}")
    if not (0.0 <= damping_ratio < 1.0):
        raise ValueError(f"damping_ratio must be in [0, 1), got {damping_ratio}")

    df = math.sqrt(1.0 - damping_ratio * damping_ratio)
    k = math.exp(-damping_ratio * math.pi / df)
    t_d = 1.0 / (shaper_freq_hz * df)
    amps = np.array([1.0, 2.0 * k, k * k], dtype=float)
    times = np.array([0.0, 0.5 * t_d, t_d], dtype=float)
    return amps, times


def get_shaper_pulses(
    name: str,
    shaper_freq_hz: float,
    damping_ratio: float = DEFAULT_SHAPER_DAMPING_RATIO,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact Klipper `(A, T)` impulse amplitudes and times (s) from `shaper_defs.py`."""
    if shaper_freq_hz <= 0.0:
        raise ValueError(f"shaper_freq_hz must be positive, got {shaper_freq_hz}")
    if not (0.0 <= damping_ratio < 1.0):
        raise ValueError(f"damping_ratio must be in [0, 1), got {damping_ratio}")

    key = name.strip().upper()
    if key == "NONE":
        return np.array([1.0], dtype=float), np.array([0.0], dtype=float)

    if key == "ZVD":
        return get_zvd_shaper(shaper_freq_hz, damping_ratio=damping_ratio)

    df = math.sqrt(1.0 - damping_ratio * damping_ratio)
    t_d = 1.0 / (shaper_freq_hz * df)

    if key == "ZV":
        k = math.exp(-damping_ratio * math.pi / df)
        amps = np.array([1.0, k], dtype=float)
        times = np.array([0.0, 0.5 * t_d], dtype=float)
        return amps, times

    if key == "MZV":
        k = math.exp(-0.75 * damping_ratio * math.pi / df)
        a1 = 1.0 - 1.0 / math.sqrt(2.0)
        a2 = (math.sqrt(2.0) - 1.0) * k
        a3 = a1 * k * k
        amps = np.array([a1, a2, a3], dtype=float)
        times = np.array([0.0, 0.375 * t_d, 0.75 * t_d], dtype=float)
        return amps, times

    v_tol = EI_SHAPER_VIB_TOL
    k = math.exp(-damping_ratio * math.pi / df)

    if key == "EI":
        a1 = 0.25 * (1.0 + v_tol)
        a2 = 0.5 * (1.0 - v_tol) * k
        a3 = a1 * k * k
        amps = np.array([a1, a2, a3], dtype=float)
        times = np.array([0.0, 0.5 * t_d, t_d], dtype=float)
        return amps, times

    if key == "2HUMP_EI":
        v2 = v_tol * v_tol
        x = pow(v2 * (math.sqrt(1.0 - v2) + 1.0), 1.0 / 3.0)
        a1 = (3.0 * x * x + 2.0 * x + 3.0 * v2) / (16.0 * x)
        a2 = (0.5 - a1) * k
        a3 = a2 * k
        a4 = a1 * k * k * k
        amps = np.array([a1, a2, a3, a4], dtype=float)
        times = np.array([0.0, 0.5 * t_d, t_d, 1.5 * t_d], dtype=float)
        return amps, times

    if key == "3HUMP_EI":
        k2 = k * k
        a1 = 0.0625 * (1.0 + 3.0 * v_tol + 2.0 * math.sqrt(2.0 * (v_tol + 1.0) * v_tol))
        a2 = 0.25 * (1.0 - v_tol) * k
        a3 = (0.5 * (1.0 + v_tol) - 2.0 * a1) * k2
        a4 = a2 * k2
        a5 = a1 * k2 * k2
        amps = np.array([a1, a2, a3, a4, a5], dtype=float)
        times = np.array([0.0, 0.5 * t_d, t_d, 1.5 * t_d, 2.0 * t_d], dtype=float)
        return amps, times

    if key == "ZVDD":
        # Singhose (1994) 4-impulse Zero-Vibration Double-Derivative shaper
        denom = (1.0 + k) ** 3
        amps = np.array([1.0, 3.0 * k, 3.0 * k * k, k * k * k], dtype=float) / denom
        times = np.array([0.0, 0.5 * t_d, t_d, 1.5 * t_d], dtype=float)
        return amps, times

    if key in ("SMOOTH_ZV", "SMOOTH_EI"):
        base_name = "ZV" if key == "SMOOTH_ZV" else "EI"
        return _build_smooth_shaper(base_name, shaper_freq_hz, damping_ratio)

    if key == "SMOOTH_FLATNESS":
        return _build_smooth_flatness_shaper(shaper_freq_hz, damping_ratio)

    valid = SHAPER_NAMES + ADVANCED_SHAPER_NAMES
    raise ValueError(f"Unknown Klipper shaper name '{name}'. Expected one of {valid}.")


SHAPER_REGISTRY: dict[str, dict[str, object]] = {
    "NONE": {"min_freq_hz": 5.0, "hi_mult": 1.0, "builder": lambda f, d=DEFAULT_SHAPER_DAMPING_RATIO: get_shaper_pulses("NONE", f, d)},
    "ZV": {"min_freq_hz": 21.0, "hi_mult": 1.35, "builder": lambda f, d=DEFAULT_SHAPER_DAMPING_RATIO: get_shaper_pulses("ZV", f, d)},
    "MZV": {"min_freq_hz": 23.0, "hi_mult": 1.35, "builder": lambda f, d=DEFAULT_SHAPER_DAMPING_RATIO: get_shaper_pulses("MZV", f, d)},
    "ZVD": {"min_freq_hz": 29.0, "hi_mult": 1.50, "builder": get_zvd_shaper},
    "EI": {"min_freq_hz": 29.0, "hi_mult": 1.50, "builder": lambda f, d=DEFAULT_SHAPER_DAMPING_RATIO: get_shaper_pulses("EI", f, d)},
    "2HUMP_EI": {"min_freq_hz": 39.0, "hi_mult": 2.00, "builder": lambda f, d=DEFAULT_SHAPER_DAMPING_RATIO: get_shaper_pulses("2HUMP_EI", f, d)},
    "3HUMP_EI": {"min_freq_hz": 48.0, "hi_mult": 2.00, "builder": lambda f, d=DEFAULT_SHAPER_DAMPING_RATIO: get_shaper_pulses("3HUMP_EI", f, d)},
    "ZVDD": {"min_freq_hz": 29.0, "hi_mult": 1.50, "builder": lambda f, d=DEFAULT_SHAPER_DAMPING_RATIO: get_shaper_pulses("ZVDD", f, d)},
    "SMOOTH_ZV": {"min_freq_hz": 21.0, "hi_mult": 1.35, "builder": lambda f, d=DEFAULT_SHAPER_DAMPING_RATIO: get_shaper_pulses("SMOOTH_ZV", f, d)},
    "SMOOTH_EI": {"min_freq_hz": 29.0, "hi_mult": 1.50, "builder": lambda f, d=DEFAULT_SHAPER_DAMPING_RATIO: get_shaper_pulses("SMOOTH_EI", f, d)},
    "SMOOTH_FLATNESS": {"min_freq_hz": 21.0, "hi_mult": 1.50, "builder": lambda f, d=DEFAULT_SHAPER_DAMPING_RATIO: get_shaper_pulses("SMOOTH_FLATNESS", f, d)},
}


def _flatness_quadrature(n_taps: int = 65) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric quadrature nodes on `[-1/2, 1/2]` and composite Simpson/trapezoidal weights."""
    if n_taps < 3:
        raise ValueError(f"n_taps must be >= 3, got {n_taps}")
    tau = np.linspace(-0.5, 0.5, n_taps, dtype=float)
    weights = np.ones(n_taps, dtype=float)
    if n_taps % 2 == 1:
        weights[1:-1:2] = 4.0
        weights[2:-2:2] = 2.0
    else:
        weights[1:-1] = 2.0
    return tau, weights


def get_flatness_toolhead_bell(
    shaper_freq_hz: float,
    n_taps: int = 65,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pure septic bell `w0(tau) = (35/16)(1 - 4*tau^2)^3` traced by the nozzle."""
    if shaper_freq_hz <= 0.0:
        raise ValueError(f"shaper_freq_hz must be positive, got {shaper_freq_hz}")
    alpha = 0.70
    t_sm = alpha / float(shaper_freq_hz)
    tau, weights = _flatness_quadrature(n_taps)
    w0 = (35.0 / 16.0) * ((1.0 - 4.0 * tau * tau) ** 3)
    amps = w0 * weights
    amps /= float(np.sum(amps))
    times = (tau + 0.5) * t_sm
    return amps, times


def _build_smooth_flatness_shaper(
    shaper_freq_hz: float,
    damping_ratio: float,
    n_taps: int = 65,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Kalico `SMOOTH_FLATNESS` inverse-plant kernel over `t_sm = 0.70 / f_n`."""
    alpha = 0.70
    t_sm = alpha / float(shaper_freq_hz)
    c = 1.0 / ((2.0 * math.pi * alpha) ** 2)
    d = float(damping_ratio) / (math.pi * alpha)
    tau, weights = _flatness_quadrature(n_taps)
    w_flat = (35.0 / 16.0) * (
        -64.0 * (tau**6)
        - 384.0 * d * (tau**5)
        + (48.0 - 1920.0 * c) * (tau**4)
        + 192.0 * d * (tau**3)
        + (-12.0 + 576.0 * c) * (tau**2)
        - 24.0 * d * tau
        + (1.0 - 24.0 * c)
    )
    amps = w_flat * weights
    amps /= float(np.sum(amps))
    times = (tau + 0.5) * t_sm
    return amps, times


def _build_smooth_shaper(
    base_name: str,
    shaper_freq_hz: float,
    damping_ratio: float,
    tau_cycles: float = 0.75,
    steps_per_quarter_cycle: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Kalico-style C1/C2 smooth kernel shaper over `[0, T_s]` eliminating jerk jumps & high harmonics."""
    amps_base, times_base = get_shaper_pulses(base_name, shaper_freq_hz, damping_ratio=damping_ratio)
    df = math.sqrt(1.0 - damping_ratio * damping_ratio)
    t_d = 1.0 / (shaper_freq_hz * df)
    dt = (0.25 * t_d) / float(steps_per_quarter_cycle)

    n_win = int(round(tau_cycles * t_d / dt)) + 1
    u = np.linspace(0.0, 1.0, n_win)
    kernel = np.sin(math.pi * u) ** 2
    if n_win > 2 and kernel[0] <= 1e-12 and kernel[-1] <= 1e-12:
        kernel = kernel[1:-1]
    kernel /= float(np.sum(kernel))

    m = len(kernel)
    t_last_base = float(times_base[-1])
    n_total = int(round((t_last_base + (m - 1) * dt) / dt)) + 1
    amps = np.zeros(n_total, dtype=float)
    for a_b, t_b in zip(amps_base, times_base):
        idx0 = int(round(float(t_b) / dt))
        amps[idx0 : idx0 + m] += float(a_b) * kernel

    times = np.arange(n_total, dtype=float) * dt
    amps /= float(np.sum(amps))
    return amps, times


def convolve_shapers(
    shaper_a: tuple[np.ndarray | Sequence[float], np.ndarray | Sequence[float]],
    shaper_b: tuple[np.ndarray | Sequence[float], np.ndarray | Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Convolve two discrete input shapers for multi-mode vibration cancellation."""
    amps_a = np.asarray(shaper_a[0], dtype=float)
    times_a = np.asarray(shaper_a[1], dtype=float)
    amps_b = np.asarray(shaper_b[0], dtype=float)
    times_b = np.asarray(shaper_b[1], dtype=float)
    if (
        amps_a.size == 0
        or amps_b.size == 0
        or amps_a.size != times_a.size
        or amps_b.size != times_b.size
    ):
        raise ValueError("Both shapers must contain matching non-empty amplitude and time arrays.")

    raw_amps = (amps_a[:, None] * amps_b[None, :]).ravel()
    raw_times = (times_a[:, None] + times_b[None, :]).ravel()
    order = np.argsort(raw_times, kind="mergesort")
    sorted_t = raw_times[order]
    sorted_a = raw_amps[order]

    merged_t: list[float] = []
    merged_a: list[float] = []
    for t_val, a_val in zip(sorted_t, sorted_a):
        if merged_t and abs(float(t_val) - merged_t[-1]) <= 1e-12:
            merged_a[-1] += float(a_val)
        else:
            merged_t.append(float(t_val))
            merged_a.append(float(a_val))

    amps_out = np.asarray(merged_a, dtype=float)
    times_out = np.asarray(merged_t, dtype=float)
    times_out -= times_out[0]
    total_a = float(np.sum(amps_out))
    if total_a > 0.0:
        amps_out /= total_a
    return amps_out, times_out


def apply_shaper_to_trajectory(
    t_s: np.ndarray | Sequence[float],
    pos_m: np.ndarray | Sequence[float],
    shaper: tuple[np.ndarray | Sequence[float], np.ndarray | Sequence[float]],
) -> np.ndarray:
    """Convolve trajectory `pos_m(t_s)` with shaper impulses via continuous linear interpolation."""
    t_arr = np.asarray(t_s, dtype=float)
    p_arr = np.asarray(pos_m, dtype=float)
    amps = np.asarray(shaper[0], dtype=float)
    times = np.asarray(shaper[1], dtype=float)
    if t_arr.ndim != 1 or t_arr.size == 0:
        raise ValueError("t_s must be a non-empty 1D time array.")
    if p_arr.shape[0] != t_arr.size:
        raise ValueError("pos_m first dimension must match len(t_s).")
    amps_sum = float(np.sum(amps))
    if amps_sum <= 0.0:
        raise ValueError("Shaper impulse amplitudes must sum to a positive value.")
    weights = amps / amps_sum

    if p_arr.ndim == 1:
        out = np.zeros_like(p_arr, dtype=float)
        for w_i, tau_i in zip(weights, times):
            out += w_i * np.interp(t_arr - tau_i, t_arr, p_arr, left=p_arr[0], right=p_arr[-1])
        return out

    flat = p_arr.reshape(t_arr.size, -1)
    out_flat = np.zeros_like(flat, dtype=float)
    for col in range(flat.shape[1]):
        p_col = flat[:, col]
        acc = np.zeros(t_arr.size, dtype=float)
        for w_i, tau_i in zip(weights, times):
            acc += w_i * np.interp(t_arr - tau_i, t_arr, p_col, left=p_col[0], right=p_col[-1])
        out_flat[:, col] = acc
    return out_flat.reshape(p_arr.shape)


def apply_biquad_notch(
    freqs_hz: np.ndarray | Sequence[float] | float,
    f_notch_hz: float,
    zeta_zero: float = 0.03,
    zeta_pole: float = 0.35,
) -> np.ndarray:
    """Return 2nd-order biquad notch magnitude `|H_notch(jw)|` over `freqs_hz`."""
    if f_notch_hz <= 0.0:
        raise ValueError(f"f_notch_hz must be positive, got {f_notch_hz}")
    freqs = np.asarray(freqs_hz, dtype=float)
    r = freqs / float(f_notch_hz)
    one_m_r2 = 1.0 - r * r
    num = one_m_r2 * one_m_r2 + (2.0 * float(zeta_zero) * r) ** 2
    den = one_m_r2 * one_m_r2 + (2.0 * float(zeta_pole) * r) ** 2
    return np.sqrt(num / np.maximum(den, 1e-18))


def biquad_notch_params_from_db_q(depth_db: float, q: float) -> tuple[float, float, float]:
    """Return `(zeta_zero, zeta_pole, max_phase_lag_deg)` for depth-limited TMC4671 `BIQUAD_T` notch."""
    if q <= 0.0:
        raise ValueError(f"q must be positive, got {q}")
    if depth_db < 0.0:
        raise ValueError(f"depth_db must be non-negative, got {depth_db}")
    zeta_pole = 1.0 / (2.0 * float(q))
    zeta_zero = zeta_pole * (10.0 ** (-float(depth_db) / 20.0))
    if zeta_zero >= zeta_pole:
        return zeta_zero, zeta_pole, 0.0
    # Exact peak lag over r in (0, 1) occurs at 2r/(1-r^2) = 1 / sqrt(zeta_zero * zeta_pole)
    ratio = zeta_pole / max(zeta_zero, 1e-15)
    max_phase_lag_deg = math.degrees(math.atan(math.sqrt(ratio)) - math.atan(1.0 / math.sqrt(ratio)))
    return zeta_zero, zeta_pole, max_phase_lag_deg


def estimate_shaper(
    shaper: tuple[np.ndarray, np.ndarray],
    test_freqs: np.ndarray | Sequence[float] | float,
    test_damping_ratio: float = DEFAULT_SHAPER_DAMPING_RATIO,
) -> np.ndarray:
    """Compute Klipper exact residual vibration ratio `V(f)` in `[0, 1]` over `test_freqs`."""
    amps, times = shaper
    amps_arr = np.asarray(amps, dtype=float)
    times_arr = np.asarray(times, dtype=float)
    freqs = np.atleast_1d(np.asarray(test_freqs, dtype=float))

    omega = 2.0 * math.pi * freqs
    damping = test_damping_ratio * omega
    omega_d = omega * math.sqrt(1.0 - test_damping_ratio * test_damping_ratio)
    inv_d = 1.0 / float(np.sum(amps_arr))
    t_last = float(times_arr[-1])

    dt_tail = t_last - times_arr[None, :]
    decay = amps_arr[None, :] * np.exp(-damping[:, None] * dt_tail)
    phase = omega_d[:, None] * times_arr[None, :]
    s_sum = np.sum(decay * np.sin(phase), axis=1)
    c_sum = np.sum(decay * np.cos(phase), axis=1)
    return inv_d * np.sqrt(s_sum * s_sum + c_sum * c_sum)


def get_shaper_smoothing(
    shaper: tuple[np.ndarray, np.ndarray],
    accel: float = 5000.0,
    scv: float = 5.0,
) -> float:
    """Compute Klipper trajectory smoothing (mm) for 90-deg and 180-deg corners."""
    amps, times = shaper
    amps_arr = np.asarray(amps, dtype=float)
    times_arr = np.asarray(times, dtype=float)

    inv_d = 1.0 / float(np.sum(amps_arr))
    ts = float(np.sum(amps_arr * times_arr)) * inv_d

    mask = times_arr >= ts
    dt_pos = times_arr[mask] - ts
    offset_90 = (
        math.sqrt(2.0)
        * inv_d
        * float(np.sum(amps_arr[mask] * (scv + 0.5 * accel * dt_pos) * dt_pos))
    )
    offset_180 = 0.5 * accel * inv_d * float(np.sum(amps_arr * ((times_arr - ts) ** 2)))
    return max(offset_90, offset_180)


def find_shaper_max_accel(
    shaper: tuple[np.ndarray, np.ndarray],
    scv: float = 5.0,
    target_smoothing: float = DEFAULT_TARGET_SMOOTHING,
) -> float:
    """Bisection solver matching Klipper `find_shaper_max_accel`, rounded to nearest 100 mm/s^2."""
    _, times = shaper
    times_arr = np.asarray(times, dtype=float)
    if times_arr.size <= 1 or np.allclose(times_arr, times_arr[0]):
        return 0.0

    if get_shaper_smoothing(shaper, accel=100.0, scv=scv) > target_smoothing:
        return 0.0

    lo = 100.0
    hi = 10000.0
    while get_shaper_smoothing(shaper, accel=hi, scv=scv) <= target_smoothing and hi < 1e7:
        hi *= 2.0

    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if get_shaper_smoothing(shaper, accel=mid, scv=scv) <= target_smoothing:
            lo = mid
        else:
            hi = mid

    return float(round(lo / 100.0) * 100.0)


def _mobility_psd(freqs_hz: np.ndarray, f_mode_hz: float, zeta: float) -> np.ndarray:
    """Normalized 2nd-order velocity-mobility resonance PSD peaking at 1.0 at `f_mode_hz`."""
    r = freqs_hz / max(f_mode_hz, 1.0)
    r2 = r * r
    denom = (1.0 - r2) ** 2 + (2.0 * zeta * r) ** 2
    return (r2 / np.maximum(denom, 1e-12)) * ((2.0 * zeta) ** 2)


def synthesize_axis_psd(
    axis: str,
    primary_freq_hz: float,
    secondary_modes: Sequence[tuple[float, float]] = (),
    damping_primary: float = 0.055,
    damping_secondary: float = 0.055,
    freqs_hz: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize a multi-mode ADXL345-style receptance/mobility PSD normalized to max=100.0.

    Returns `(freqs_hz, raw_psd)` over `[5, 220] Hz` (108 bins by default).
    """
    _ = axis  # reserved for future axis-specific sensor noise profiles
    freqs = (
        np.linspace(MIN_FREQ_HZ, MAX_FREQ_HZ, NUM_FREQ_BINS, dtype=float)
        if freqs_hz is None
        else np.asarray(freqs_hz, dtype=float).copy()
    )

    psd = np.full_like(freqs, 0.008, dtype=float)
    f_primary = max(float(primary_freq_hz), 1.0)
    psd += _mobility_psd(freqs, f_primary, damping_primary)

    for mode_freq_hz, participation in secondary_modes:
        if 10.0 <= mode_freq_hz <= 215.0 and abs(mode_freq_hz - f_primary) >= 2.0 and participation > 0.15:
            # Align with Klipper's 1/f-normalized acceleration sweep PSD (linear in f/f_primary),
            # rolled off above ~1.8 * f_primary where belt/gantry compliance isolates local modes.
            freq_ratio = float(mode_freq_hz) / f_primary
            drive_isolation = 1.0 / (1.0 + (freq_ratio / 1.80) ** 8)
            weight = min(0.85, (float(participation) ** 2) * 0.45 * freq_ratio * drive_isolation)
            psd += weight * _mobility_psd(freqs, float(mode_freq_hz), damping_secondary)

    peak = float(np.max(psd))
    if peak > 0.0:
        psd *= 100.0 / peak
    return freqs, psd


def estimate_psd_damping_ratio(freqs_hz: np.ndarray, raw_psd: np.ndarray) -> float:
    """Estimate primary peak damping ratio via -3 dB (half-power = peak / 2) bandwidth.

    Uses 3-point inverse-PSD (Lorentzian) sub-bin peak interpolation so narrow low-frequency
    peaks sitting between discrete frequency bins do not overestimate bandwidth.
    """
    pk_idx = int(np.argmax(raw_psd))
    pk_val = float(raw_psd[pk_idx])
    pk_freq = float(freqs_hz[pk_idx])
    if 0 < pk_idx < len(freqs_hz) - 1:
        f0, f1, f2 = float(freqs_hz[pk_idx - 1]), pk_freq, float(freqs_hz[pk_idx + 1])
        y0 = 1.0 / max(float(raw_psd[pk_idx - 1]), 1e-9)
        y1 = 1.0 / max(pk_val, 1e-9)
        y2 = 1.0 / max(float(raw_psd[pk_idx + 1]), 1e-9)
        d0, d2 = f0 - f1, f2 - f1
        if abs(d0) > 1e-9 and abs(d2) > 1e-9 and abs(d2 - d0) > 1e-9:
            a = ((y2 - y1) / d2 - (y0 - y1) / d0) / (d2 - d0)
            b = (y2 - y1) / d2 - a * d2
            if a > 1e-12:
                d_min = -b / (2.0 * a)
                y_min = y1 + b * d_min + a * d_min * d_min
                if 0.0 < y_min <= y1 and abs(d_min) < max(abs(d0), abs(d2)):
                    pk_freq = f1 + d_min
                    pk_val = 1.0 / y_min

    half_power = 0.5 * pk_val
    y_min_val = 1.0 / max(pk_val, 1e-12)
    s_target = math.sqrt(y_min_val)

    def _interp_half_power(f_a: float, p_a: float, f_b: float, p_b: float) -> float:
        # For a Lorentzian resonance peak, s(f) = sqrt(1/P(f) - 1/P_max) is linear in |f - f_peak|,
        # eliminating discrete 2 Hz bin-grid bias when the peak lands halfway between frequency bins.
        s_a = math.sqrt(max(1.0 / max(p_a, 1e-12) - y_min_val, 0.0))
        s_b = math.sqrt(max(1.0 / max(p_b, 1e-12) - y_min_val, 0.0))
        denom = s_a - s_b
        if abs(denom) > 1e-12:
            alpha = (s_target - s_b) / denom
            if 0.0 <= alpha <= 1.0:
                return float(f_b + alpha * (f_a - f_b))
        span = max(abs(p_b - p_a), 1e-12)
        alpha = (half_power - p_a) / (p_b - p_a) if abs(p_b - p_a) > 1e-12 else 0.5
        return float(f_a + max(0.0, min(1.0, alpha)) * (f_b - f_a))

    f_lo = float(freqs_hz[0])
    for idx in range(pk_idx, 0, -1):
        if raw_psd[idx - 1] <= half_power <= raw_psd[idx]:
            f_lo = _interp_half_power(
                float(freqs_hz[idx - 1]),
                float(raw_psd[idx - 1]),
                float(freqs_hz[idx]),
                float(raw_psd[idx]),
            )
            break

    f_hi = float(freqs_hz[-1])
    for idx in range(pk_idx, len(freqs_hz) - 1):
        if raw_psd[idx + 1] <= half_power <= raw_psd[idx]:
            f_hi = _interp_half_power(
                float(freqs_hz[idx + 1]),
                float(raw_psd[idx + 1]),
                float(freqs_hz[idx]),
                float(raw_psd[idx]),
            )
            break

    zeta_est = (f_hi - f_lo) / (2.0 * max(pk_freq, 1.0))
    return float(max(0.02, min(0.25, zeta_est)))


_SHAPER_HI_MULT: dict[str, float] = {
    "ZV": 1.35,
    "MZV": 1.35,
    "ZVD": 1.50,
    "EI": 1.50,
    "2HUMP_EI": 2.00,
    "3HUMP_EI": 2.00,
    "ZVDD": 1.50,
    "SMOOTH_ZV": 1.35,
    "SMOOTH_EI": 1.50,
    "SMOOTH_FLATNESS": 1.50,
}


def _shaper_candidate_freqs(name: str, center_f: float) -> np.ndarray:
    """Build Stage 1 coarse frequency grid sized proportionally to each shaper's search span."""
    lo_f = min(max(12.0, center_f * 0.7), 210.0)
    hi_mult = _SHAPER_HI_MULT.get(name.strip().upper(), 1.50)
    hi_f = min(215.0, max(lo_f + 4.0, center_f * hi_mult))
    n_coarse = max(35, int(math.ceil((hi_f - lo_f) / 1.25)) + 1)
    return np.linspace(lo_f, hi_f, n_coarse, dtype=float)


def _select_klipper_best_shaper(candidates: Sequence[ShaperCandidate]) -> ShaperCandidate:
    """Select best shaper across candidates using exact Klipper `find_best_shaper` tradeoffs."""
    if not candidates:
        raise ValueError("candidates must be non-empty.")
    best = candidates[0]
    for shaper in candidates[1:]:
        if (
            shaper.score * 1.2 < best.score
            or (
                shaper.score * 1.05 < best.score
                and shaper.smoothing * 1.1 < best.smoothing
            )
        ):
            best = shaper
    return best


def _fit_single_shaper(
    name: str,
    freqs_hz: np.ndarray,
    raw_psd: np.ndarray,
    candidate_freqs: np.ndarray,
    vibr_threshold: float,
    all_vibrations: float,
    scv: float,
    damping_ratio: float,
    soft_tiebreak_weight: float = 0.0,
    test_damping_ratios: Sequence[float] = TEST_DAMPING_RATIOS,
) -> ShaperCandidate:
    """Fit optimal shaper frequency using Klipper `fit_shaper` damping pessimisation & rules."""
    is_flatness = name.strip().upper() == "SMOOTH_FLATNESS"
    test_drs = np.asarray(
        tuple(float(dr) for dr in test_damping_ratios) if test_damping_ratios else TEST_DAMPING_RATIOS,
        dtype=float,
    )
    omega = 2.0 * math.pi * freqs_hz
    damping_mat = test_drs[:, None] * omega[None, :]  # (N_dr, F)
    omega_d_mat = np.sqrt(1.0 - test_drs[:, None] ** 2) * omega[None, :]  # (N_dr, F)
    evaluated: dict[
        float, tuple[float, float, float, float, np.ndarray, tuple[np.ndarray, np.ndarray]]
    ] = {}
    seen_fcs: set[float] = set()

    def _eval_grid(grid: np.ndarray) -> None:
        new_fcs: list[float] = []
        shapers: list[tuple[np.ndarray, np.ndarray]] = []
        for fc in grid:
            fc_f = round(float(fc), 6)
            if fc_f in seen_fcs:
                continue
            seen_fcs.add(fc_f)
            new_fcs.append(fc_f)
            shapers.append(get_shaper_pulses(name, fc_f, damping_ratio=damping_ratio))
        if not new_fcs:
            return

        amps_batch = np.asarray([s[0] for s in shapers], dtype=float)  # (M, P)
        times_batch = np.asarray([s[1] for s in shapers], dtype=float)  # (M, P)
        inv_d = 1.0 / np.sum(amps_batch, axis=1, keepdims=True)
        w_batch = amps_batch * inv_d  # (M, P)
        dt_tail = times_batch[:, -1:] - times_batch  # (M, P)

        # Vectorized evaluation across (M, N_dr, F, P)
        decay = w_batch[:, None, None, :] * np.exp(
            -damping_mat[None, :, :, None] * dt_tail[:, None, None, :]
        )
        phase = omega_d_mat[None, :, :, None] * times_batch[:, None, None, :]
        s_sum = np.sum(decay * np.sin(phase), axis=3)
        c_sum = np.sum(decay * np.cos(phase), axis=3)
        vals_all = np.sqrt(s_sum * s_sum + c_sum * c_sum)  # (M, N_dr, F)

        rem_vibs = np.sum(np.maximum(vals_all * raw_psd[None, None, :] - vibr_threshold, 0.0), axis=2)
        max_vib_ratios = np.max(rem_vibs / all_vibrations, axis=1)  # (M,)
        shaper_vals_all = np.max(vals_all, axis=1)  # (M, F)

        if is_flatness:
            sm_batch = [get_flatness_toolhead_bell(fc_f) for fc_f in new_fcs]
            sm_amps = np.asarray([s[0] for s in sm_batch], dtype=float)
            sm_times = np.asarray([s[1] for s in sm_batch], dtype=float)
            sm_w = sm_amps / np.sum(sm_amps, axis=1, keepdims=True)
        else:
            sm_w = w_batch
            sm_times = times_batch

        ts = np.sum(sm_w * sm_times, axis=1, keepdims=True)  # (M, 1)
        dt_pos = np.maximum(sm_times - ts, 0.0)
        offset_90 = math.sqrt(2.0) * np.sum(sm_w * (scv + 2500.0 * dt_pos) * dt_pos, axis=1)
        offset_180 = 2500.0 * np.sum(sm_w * ((sm_times - ts) ** 2), axis=1)
        smoothings = np.maximum(offset_90, offset_180)

        vr_eff = max_vib_ratios.copy()
        if soft_tiebreak_weight > 0.0:
            vr_eff += soft_tiebreak_weight * np.sum(shaper_vals_all * raw_psd[None, :], axis=1) / all_vibrations
        scores = smoothings * (vr_eff ** 1.5 + vr_eff * 0.2 + 0.01)

        for idx, fc_f in enumerate(new_fcs):
            evaluated[fc_f] = (
                fc_f,
                float(max_vib_ratios[idx]),
                float(smoothings[idx]),
                float(scores[idx]),
                shaper_vals_all[idx],
                shapers[idx],
            )

    def _pick_within_shaper(
        rows: list[tuple[float, float, float, float, np.ndarray, tuple[np.ndarray, np.ndarray]]],
    ) -> tuple[float, float, float, float, np.ndarray, tuple[np.ndarray, np.ndarray]]:
        best_vib_row = min(rows, key=lambda r: r[1])
        selected = best_vib_row
        for res in reversed(rows):
            if res[1] < best_vib_row[1] * 1.1 + 0.0005 and res[3] < selected[3]:
                selected = res
        return selected

    # Stage 1: coarse sweep across [lo_f, hi_f]
    _eval_grid(candidate_freqs)

    # Stage 2: 0.25 Hz fine local refinement around min-vibration and min-score basins
    if len(candidate_freqs) >= 2:
        coarse_rows = sorted(evaluated.values(), key=lambda r: r[0])
        best_vib_coarse = min(coarse_rows, key=lambda r: r[1])
        sel_coarse = _pick_within_shaper(coarse_rows)
        min_score_coarse = min(coarse_rows, key=lambda r: r[3])
        step_coarse = float(candidate_freqs[1] - candidate_freqs[0])
        lo_bound = float(candidate_freqs[0])
        hi_bound = float(candidate_freqs[-1])
        for anchor_f in {best_vib_coarse[0], sel_coarse[0], min_score_coarse[0]}:
            fine_lo = max(lo_bound, anchor_f - step_coarse)
            fine_hi = min(hi_bound, anchor_f + step_coarse)
            if fine_hi > fine_lo + 0.1:
                fine_grid = np.arange(fine_lo, fine_hi + 0.125, 0.25, dtype=float)
                _eval_grid(fine_grid)

    all_rows = sorted(evaluated.values(), key=lambda r: r[0])
    best_freq, best_vib_ratio, best_smoothing, best_score, best_attenuation, best_shaper = (
        _pick_within_shaper(all_rows)
    )
    accel_shaper = get_flatness_toolhead_bell(best_freq) if is_flatness else best_shaper
    max_accel = find_shaper_max_accel(accel_shaper, scv=scv, target_smoothing=DEFAULT_TARGET_SMOOTHING)
    return ShaperCandidate(
        name=name,
        freq_hz=best_freq,
        vibrations_pct=best_vib_ratio * 100.0,
        smoothing=best_smoothing,
        max_accel_mm_s2=max_accel,
        score=best_score,
        attenuation_curve=best_attenuation,
    )


def calibrate_axis_shapers(
    axis: str,
    primary_freq_hz: float,
    secondary_modes: Sequence[tuple[float, float]] = (),
    scv: float = 5.0,
    shaper_choice: str = "auto",
    damping_ratio: float = DEFAULT_SHAPER_DAMPING_RATIO,
    freqs_hz: np.ndarray | None = None,
    raw_psd: np.ndarray | None = None,
    psd_damping_ratio: float | None = None,
    normalize_psd: bool = True,
    ref_peak: float = 100.0,
    ref_all_vibrations: float | None = None,
    measured_damping_override: float | None = None,
    test_damping_ratios: Sequence[float] = TEST_DAMPING_RATIOS,
    shaper_names: Sequence[str] | None = None,
) -> AxisShaperGraph:
    """Run full Klipper `shaper_calibrate` optimization over `[5, 220] Hz` for one axis."""
    psd_zeta = 0.055 if psd_damping_ratio is None else float(psd_damping_ratio)
    if raw_psd is None:
        freqs, psd = synthesize_axis_psd(
            axis=axis,
            primary_freq_hz=primary_freq_hz,
            secondary_modes=secondary_modes,
            damping_primary=psd_zeta,
            damping_secondary=psd_zeta,
            freqs_hz=freqs_hz,
        )
    else:
        freqs = (
            np.linspace(MIN_FREQ_HZ, MAX_FREQ_HZ, len(raw_psd), dtype=float)
            if freqs_hz is None
            else np.asarray(freqs_hz, dtype=float).copy()
        )
        psd = np.asarray(raw_psd, dtype=float).copy()
        if len(freqs) != len(psd):
            raise ValueError("freqs_hz and raw_psd must have identical lengths.")
        if normalize_psd:
            peak = float(np.max(psd))
            if peak > 0.0:
                psd *= 100.0 / peak

    peak_freq = float(freqs[int(np.argmax(psd))]) if float(np.max(psd)) > 0.0 else float(primary_freq_hz)
    if measured_damping_override is not None:
        measured_zeta = float(max(0.02, min(0.25, measured_damping_override)))
    elif raw_psd is None:
        # Measure half-power bandwidth on a fine 0.15 Hz local grid around primary_freq_hz
        # so narrow low-frequency peaks (~15-35 Hz, FWHM ~2-4 Hz) never alias on the 2 Hz display grid.
        f_p = max(float(primary_freq_hz), 5.0)
        fine_lo = max(2.0, f_p * 0.5)
        fine_hi = min(MAX_FREQ_HZ, max(f_p * 1.6, f_p + 12.0))
        fine_freqs = np.arange(fine_lo, fine_hi + 0.05, 0.15, dtype=float)
        _, fine_psd = synthesize_axis_psd(
            axis=axis,
            primary_freq_hz=primary_freq_hz,
            secondary_modes=secondary_modes,
            damping_primary=psd_zeta,
            damping_secondary=psd_zeta,
            freqs_hz=fine_freqs,
        )
        measured_zeta = estimate_psd_damping_ratio(fine_freqs, fine_psd)
    else:
        measured_zeta = estimate_psd_damping_ratio(freqs, psd)

    if normalize_psd:
        vibr_threshold = float(np.max(psd)) / SHAPER_VIBRATION_REDUCTION
        all_vibrations = max(float(np.sum(np.maximum(psd - vibr_threshold, 0.0))), 1e-12)
        tie_w = 0.0
    else:
        vibr_threshold = float(ref_peak) / SHAPER_VIBRATION_REDUCTION
        all_vibrations = (
            max(float(ref_all_vibrations), 1e-12)
            if ref_all_vibrations is not None
            else max(float(np.sum(np.maximum(psd - vibr_threshold, 0.0))), 1e-12)
        )
        tie_w = 0.02

    center_f = float(primary_freq_hz) if not normalize_psd else peak_freq
    eval_names = tuple(shaper_names) if shaper_names is not None else SHAPER_NAMES

    candidates = [
        _fit_single_shaper(
            name=name,
            freqs_hz=freqs,
            raw_psd=psd,
            candidate_freqs=_shaper_candidate_freqs(name, center_f),
            vibr_threshold=vibr_threshold,
            all_vibrations=all_vibrations,
            scv=scv,
            damping_ratio=damping_ratio,
            soft_tiebreak_weight=tie_w,
            test_damping_ratios=test_damping_ratios,
        )
        for name in eval_names
    ]

    choice_norm = shaper_choice.strip().upper()
    if choice_norm == "AUTO":
        selected = _select_klipper_best_shaper(candidates)
        shaped_psd = psd * selected.attenuation_curve
        rec_name = selected.name
        rec_freq = selected.freq_hz
        rec_accel = selected.max_accel_mm_s2
    elif choice_norm == "NONE":
        shaped_psd = psd.copy()
        rec_name = "NONE"
        rec_freq = center_f
        omega_none = 2.0 * math.pi * max(float(primary_freq_hz), 5.0)
        rec_accel = NONE_SHAPER_ALLOWABLE_RINGING_MM * (omega_none * omega_none)
    else:
        matched = next((c for c in candidates if c.name == choice_norm), None)
        if matched is None and (choice_norm in ADVANCED_SHAPER_NAMES or choice_norm in SHAPER_NAMES):
            matched = _fit_single_shaper(
                name=choice_norm,
                freqs_hz=freqs,
                raw_psd=psd,
                candidate_freqs=_shaper_candidate_freqs(choice_norm, center_f),
                vibr_threshold=vibr_threshold,
                all_vibrations=all_vibrations,
                scv=scv,
                damping_ratio=damping_ratio,
                soft_tiebreak_weight=tie_w,
                test_damping_ratios=test_damping_ratios,
            )
        if matched is None:
            raise ValueError(f"Unsupported shaper_choice '{shaper_choice}'.")
        shaped_psd = psd * matched.attenuation_curve
        rec_name = matched.name
        rec_freq = matched.freq_hz
        rec_accel = matched.max_accel_mm_s2

    return AxisShaperGraph(
        axis=axis.upper(),
        freqs_hz=freqs,
        raw_psd=psd,
        shaped_psd=shaped_psd,
        peak_freq_hz=peak_freq,
        damping_ratio=measured_zeta,
        recommended_shaper=rec_name,
        recommended_freq_hz=rec_freq,
        recommended_max_accel_mm_s2=rec_accel,
        candidates=candidates,
    )
