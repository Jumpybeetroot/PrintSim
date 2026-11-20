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
DEFAULT_SHAPER_DAMPING_RATIO: float = 0.1
DEFAULT_TARGET_SMOOTHING: float = 0.12
SHAPER_VIBRATION_REDUCTION: float = 20.0  # Klipper shaper_defs.SHAPER_VIBRATION_REDUCTION
EI_SHAPER_VIB_TOL: float = 0.05  # Klipper shaper_defs default shaper_vib_tol
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

    raise ValueError(f"Unknown Klipper shaper name '{name}'. Expected one of {SHAPER_NAMES}.")


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
        return 100.0

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
            # Modal receptance scales inversely with mode stiffness (f_primary / f_mode)^2
            stiffness_atten = min(1.0, (f_primary / mode_freq_hz) ** 2)
            weight = (float(participation) ** 2) * 0.45 * stiffness_atten
            psd += weight * _mobility_psd(freqs, float(mode_freq_hz), damping_secondary)

    peak = float(np.max(psd))
    if peak > 0.0:
        psd *= 100.0 / peak
    return freqs, psd


def estimate_psd_damping_ratio(freqs_hz: np.ndarray, raw_psd: np.ndarray) -> float:
    """Estimate primary peak damping ratio via -3 dB (half-power = peak / 2) bandwidth."""
    pk_idx = int(np.argmax(raw_psd))
    pk_val = float(raw_psd[pk_idx])
    pk_freq = float(freqs_hz[pk_idx])
    half_power = 0.5 * pk_val

    f_lo = float(freqs_hz[0])
    for idx in range(pk_idx, 0, -1):
        if raw_psd[idx - 1] <= half_power <= raw_psd[idx]:
            span = max(float(raw_psd[idx] - raw_psd[idx - 1]), 1e-12)
            alpha = (half_power - float(raw_psd[idx - 1])) / span
            f_lo = float(freqs_hz[idx - 1] + alpha * (freqs_hz[idx] - freqs_hz[idx - 1]))
            break

    f_hi = float(freqs_hz[-1])
    for idx in range(pk_idx, len(freqs_hz) - 1):
        if raw_psd[idx + 1] <= half_power <= raw_psd[idx]:
            span = max(float(raw_psd[idx] - raw_psd[idx + 1]), 1e-12)
            alpha = (half_power - float(raw_psd[idx + 1])) / span
            f_hi = float(freqs_hz[idx + 1] - alpha * (freqs_hz[idx + 1] - freqs_hz[idx]))
            break

    zeta_est = (f_hi - f_lo) / (2.0 * max(pk_freq, 1.0))
    return float(max(0.02, min(0.25, zeta_est)))


def _fit_single_shaper(
    name: str,
    freqs_hz: np.ndarray,
    raw_psd: np.ndarray,
    candidate_freqs: np.ndarray,
    vibr_threshold: float,
    all_vibrations: float,
    scv: float,
    damping_ratio: float,
) -> ShaperCandidate:
    """Fit optimal shaper frequency on `candidate_freqs` minimizing Klipper's score."""
    best_score = math.inf
    best_freq = float(candidate_freqs[0])
    best_vib_ratio = 1.0
    best_smoothing = 0.0
    best_attenuation: np.ndarray | None = None
    best_shaper: tuple[np.ndarray, np.ndarray] | None = None

    for fc in candidate_freqs:
        shaper = get_shaper_pulses(name, float(fc), damping_ratio=damping_ratio)
        attenuation = estimate_shaper(shaper, freqs_hz, test_damping_ratio=damping_ratio)
        rem_vib = float(np.sum(np.maximum(raw_psd * attenuation - vibr_threshold, 0.0)))
        vib_ratio = rem_vib / all_vibrations
        smoothing = get_shaper_smoothing(shaper, accel=5000.0, scv=scv)
        score = smoothing * (vib_ratio ** 1.5 + vib_ratio * 0.2 + 0.01)

        if score < best_score:
            best_score = score
            best_freq = float(fc)
            best_vib_ratio = vib_ratio
            best_smoothing = smoothing
            best_attenuation = attenuation
            best_shaper = shaper

    assert best_shaper is not None and best_attenuation is not None
    max_accel = find_shaper_max_accel(best_shaper, scv=scv, target_smoothing=DEFAULT_TARGET_SMOOTHING)
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
) -> AxisShaperGraph:
    """Run full Klipper `shaper_calibrate` optimization over `[5, 220] Hz` for one axis."""
    if raw_psd is None:
        freqs, psd = synthesize_axis_psd(
            axis=axis,
            primary_freq_hz=primary_freq_hz,
            secondary_modes=secondary_modes,
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
        peak = float(np.max(psd))
        if peak > 0.0:
            psd *= 100.0 / peak

    peak_freq = float(freqs[int(np.argmax(psd))])
    measured_zeta = estimate_psd_damping_ratio(freqs, psd)

    vibr_threshold = float(np.max(psd)) / SHAPER_VIBRATION_REDUCTION
    all_vibrations = max(float(np.sum(np.maximum(psd - vibr_threshold, 0.0))), 1e-12)

    lo_f = min(max(12.0, peak_freq * 0.7), 210.0)
    hi_f = min(215.0, max(lo_f + 4.0, peak_freq * 1.35))
    candidate_freqs = np.linspace(lo_f, hi_f, 35, dtype=float)

    candidates = [
        _fit_single_shaper(
            name=name,
            freqs_hz=freqs,
            raw_psd=psd,
            candidate_freqs=candidate_freqs,
            vibr_threshold=vibr_threshold,
            all_vibrations=all_vibrations,
            scv=scv,
            damping_ratio=damping_ratio,
        )
        for name in SHAPER_NAMES
    ]

    choice_norm = shaper_choice.strip().upper()
    if choice_norm == "AUTO":
        acceptable = [c for c in candidates if c.vibrations_pct <= 6.0]
        selected = min(acceptable, key=lambda c: c.score) if acceptable else min(candidates, key=lambda c: c.score)
        shaped_psd = psd * selected.attenuation_curve
        rec_name = selected.name
        rec_freq = selected.freq_hz
        rec_accel = selected.max_accel_mm_s2
    elif choice_norm == "NONE":
        shaped_psd = psd.copy()
        rec_name = "NONE"
        rec_freq = peak_freq
        rec_accel = 0.9 * (max(float(primary_freq_hz), 5.0) ** 2)
    else:
        matched = next((c for c in candidates if c.name == choice_norm), None)
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
