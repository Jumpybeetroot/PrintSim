"""Klipper ADXL345 / Shake&Tune CSV import and measured-vs-simulated PSD calibration."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.signal import welch

from printsim.fea.klipper_shaper import (
    MIN_FREQ_HZ,
    AxisShaperGraph,
    calibrate_axis_shapers,
    estimate_psd_damping_ratio,
)
from printsim.fea.modal import solve_modal_fea
from printsim.geometry.builder import build_assembly
from printsim.schema import PrinterSpec

SourceFormat = Literal["klipper_psd_csv", "raw_accel_csv"]

_FREQ_COL_NAMES = frozenset({"freq", "frequency", "freq_hz", "frequency_hz", "hz"})
_TIME_COL_NAMES = frozenset({"time", "t", "t_s", "time_s", "timestamp", "time_ms", "t_ms"})
_TIME_MS_COL_NAMES = frozenset({"time_ms", "t_ms"})

_AXIS_ACCEL_CANDIDATES: dict[str, tuple[str, ...]] = {
    "X": ("accel_x", "x", "ax", "a_x"),
    "Y": ("accel_y", "y", "ay", "a_y"),
}
_ALL_ACCEL_COL_NAMES = frozenset(
    {
        "accel_x",
        "accel_y",
        "accel_z",
        "x",
        "y",
        "z",
        "ax",
        "ay",
        "az",
        "a_x",
        "a_y",
        "a_z",
        "accel",
        "acc",
        "a",
        "signal",
    }
)
_HEADER_TOKEN_RE = re.compile(r"^[a-z0-9_().%/-]+$")


@dataclass
class ParsedAdxlData:
    """Parsed frequency-domain PSD extracted from a Klipper PSD or raw ADXL345 CSV."""

    axis: str
    source_format: SourceFormat
    freqs_hz: np.ndarray
    psd: np.ndarray
    sample_count: int
    selected_column: str
    sample_rate_hz: float | None = None


@dataclass
class AxisCalibrationResult:
    """Measured-vs-simulated resonance and input-shaper comparison for one axis."""

    axis: str
    source_format: SourceFormat
    freqs_hz: list[float]
    measured_psd_norm: list[float]
    simulated_psd_norm: list[float]
    measured_peak_hz: float
    simulated_peak_hz: float
    peak_delta_hz: float
    peak_error_pct: float
    measured_damping_ratio: float
    simulated_damping_ratio: float
    measured_shaper: str
    measured_shaper_freq_hz: float
    measured_max_accel_mm_s2: float
    simulated_shaper: str
    simulated_shaper_freq_hz: float
    simulated_max_accel_mm_s2: float
    spectral_correlation: float
    sample_count: int
    diagnostics: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "source_format": self.source_format,
            "freqs_hz": [round(float(v), 2) for v in self.freqs_hz],
            "measured_psd_norm": [round(float(v), 4) for v in self.measured_psd_norm],
            "simulated_psd_norm": [round(float(v), 4) for v in self.simulated_psd_norm],
            "measured_peak_hz": round(float(self.measured_peak_hz), 2),
            "simulated_peak_hz": round(float(self.simulated_peak_hz), 2),
            "peak_delta_hz": round(float(self.peak_delta_hz), 2),
            "peak_error_pct": round(float(self.peak_error_pct), 2),
            "measured_damping_ratio": round(float(self.measured_damping_ratio), 4),
            "simulated_damping_ratio": round(float(self.simulated_damping_ratio), 4),
            "measured_shaper": self.measured_shaper,
            "measured_shaper_freq_hz": round(float(self.measured_shaper_freq_hz), 2),
            "measured_max_accel_mm_s2": round(float(self.measured_max_accel_mm_s2), 0),
            "simulated_shaper": self.simulated_shaper,
            "simulated_shaper_freq_hz": round(float(self.simulated_shaper_freq_hz), 2),
            "simulated_max_accel_mm_s2": round(float(self.simulated_max_accel_mm_s2), 0),
            "spectral_correlation": round(float(self.spectral_correlation), 4),
            "sample_count": int(self.sample_count),
            "diagnostics": list(self.diagnostics),
        }


@dataclass
class AdxlCalibrationReport:
    """Full printer measured-vs-simulated ADXL345 calibration report."""

    printer_name: str
    axes: dict[str, AxisCalibrationResult]
    overall_summary: str
    recommended_adjustments: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "printer_name": self.printer_name,
            "axes": {k: v.to_dict() for k, v in self.axes.items()},
            "overall_summary": self.overall_summary,
            "recommended_adjustments": list(self.recommended_adjustments),
        }


def _detect_delimiter(line: str) -> str:
    for delim in (",", ";", "\t"):
        if delim in line:
            return delim
    return ","


def _is_psd_col(name: str) -> bool:
    return name == "psd" or name.startswith("psd_")


def _detect_csv_header(
    lines: list[str],
) -> tuple[int, list[str], str, SourceFormat]:
    """Locate the CSV header row and classify it as `klipper_psd_csv` or `raw_accel_csv`."""
    for idx, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not stripped:
            continue
        candidate = stripped.lstrip("#").strip()
        if not candidate:
            continue
        delim = _detect_delimiter(candidate)
        cols = [c.strip().lower() for c in next(csv.reader([candidate], delimiter=delim))]
        if len(cols) < 2:
            continue
        # Reject prose comments that happen to contain keywords like "freq" or "time"
        if not all((not c or _HEADER_TOKEN_RE.match(c)) for c in cols):
            continue

        has_freq = any(c in _FREQ_COL_NAMES for c in cols)
        has_psd = any(_is_psd_col(c) for c in cols)
        if has_freq and has_psd:
            return idx, cols, delim, "klipper_psd_csv"

        has_time = any(c in _TIME_COL_NAMES for c in cols)
        has_accel = any(c in _ALL_ACCEL_COL_NAMES for c in cols)
        if has_time and has_accel:
            return idx, cols, delim, "raw_accel_csv"

    raise ValueError(
        "Unrecognized CSV format: expected Klipper PSD columns (freq,psd_x,...) "
        "or raw ADXL345 time-series columns (time,accel_x,accel_y,accel_z)."
    )


def _parse_float_cell(row: list[str], idx: int) -> float | None:
    if idx >= len(row):
        return None
    val_str = row[idx].strip()
    if not val_str:
        return None
    try:
        val = float(val_str)
    except ValueError:
        return None
    return val if math.isfinite(val) else None


def _parse_klipper_psd_rows(
    data_lines: list[str],
    cols: list[str],
    delim: str,
    axis: str,
) -> ParsedAdxlData:
    freq_col = next(i for i, c in enumerate(cols) if c in _FREQ_COL_NAMES)
    psd_cols = [(i, c) for i, c in enumerate(cols) if _is_psd_col(c)]

    freqs: list[float] = []
    col_series: dict[str, list[float]] = {name: [] for _, name in psd_cols}

    for row in csv.reader(data_lines, delimiter=delim):
        if not row:
            continue
        first = row[0].strip()
        if not first or first.startswith("#"):
            continue
        f_val = _parse_float_cell(row, freq_col)
        if f_val is None:
            continue

        parsed_psds: dict[str, float] = {}
        valid_row = True
        for c_idx, c_name in psd_cols:
            p_val = _parse_float_cell(row, c_idx)
            if p_val is None:
                valid_row = False
                break
            parsed_psds[c_name] = p_val
        if not valid_row:
            continue

        freqs.append(f_val)
        for c_name, p_val in parsed_psds.items():
            col_series[c_name].append(p_val)

    if len(freqs) < 3:
        raise ValueError(
            f"Klipper PSD CSV for axis {axis} requires at least 3 numeric frequency rows (got {len(freqs)})."
        )

    sample_count = len(freqs)
    freq_arr = np.asarray(freqs, dtype=float)
    series_arrays = {name: np.maximum(np.asarray(vals, dtype=float), 0.0) for name, vals in col_series.items()}

    axis_col = f"psd_{axis.lower()}"
    selected_col: str | None = None
    for candidate_name in (axis_col, "psd_xyz", "psd"):
        if candidate_name in series_arrays and float(np.max(series_arrays[candidate_name])) > 0.0:
            selected_col = candidate_name
            break

    if selected_col is None:
        dominant_name = max(series_arrays, key=lambda k: float(np.max(series_arrays[k])))
        if float(np.max(series_arrays[dominant_name])) > 0.0:
            selected_col = dominant_name

    if selected_col is None:
        raise ValueError(f"Klipper PSD CSV for axis {axis} contains no positive PSD signal energy.")

    psd_arr = series_arrays[selected_col]
    order = np.argsort(freq_arr, kind="mergesort")
    freq_arr = freq_arr[order]
    psd_arr = psd_arr[order]

    # Deduplicate identical frequency bins (if any) by averaging
    unique_freqs, inverse = np.unique(freq_arr, return_inverse=True)
    if unique_freqs.size != freq_arr.size:
        avg_psd = np.zeros_like(unique_freqs, dtype=float)
        counts = np.zeros_like(unique_freqs, dtype=float)
        np.add.at(avg_psd, inverse, psd_arr)
        np.add.at(counts, inverse, 1.0)
        freq_arr = unique_freqs
        psd_arr = avg_psd / np.maximum(counts, 1.0)

    return ParsedAdxlData(
        axis=axis,
        source_format="klipper_psd_csv",
        freqs_hz=freq_arr,
        psd=psd_arr,
        sample_count=sample_count,
        selected_column=selected_col,
    )


def _compute_welch_nperseg(n_samples: int, fs_hz: float) -> int:
    """Select Hann-window `nperseg` tuned for ~1-2 Hz frequency resolution."""
    if n_samples >= int(round(1.5 * fs_hz)):
        target_df = 1.0
    elif n_samples >= int(round(fs_hz)):
        target_df = 1.5
    else:
        target_df = 2.0
    nperseg = int(round(fs_hz / target_df))
    return max(min(n_samples, 32), min(n_samples, nperseg))


def _parse_raw_accel_rows(
    data_lines: list[str],
    cols: list[str],
    delim: str,
    axis: str,
    min_freq_hz: float,
) -> ParsedAdxlData:
    time_col_idx = next(i for i, c in enumerate(cols) if c in _TIME_COL_NAMES)
    time_scale = 1e-3 if cols[time_col_idx] in _TIME_MS_COL_NAMES else 1.0
    accel_cols = [(i, c) for i, c in enumerate(cols) if c in _ALL_ACCEL_COL_NAMES]

    times: list[float] = []
    col_series: dict[str, list[float]] = {name: [] for _, name in accel_cols}

    for row in csv.reader(data_lines, delimiter=delim):
        if not row:
            continue
        first = row[0].strip()
        if not first or first.startswith("#"):
            continue
        t_val = _parse_float_cell(row, time_col_idx)
        if t_val is None:
            continue

        parsed_acc: dict[str, float] = {}
        valid_row = True
        for c_idx, c_name in accel_cols:
            a_val = _parse_float_cell(row, c_idx)
            if a_val is None:
                valid_row = False
                break
            parsed_acc[c_name] = a_val
        if not valid_row:
            continue

        times.append(t_val * time_scale)
        for c_name, a_val in parsed_acc.items():
            col_series[c_name].append(a_val)

    if len(times) < 16:
        raise ValueError(
            f"Raw ADXL CSV for axis {axis} requires at least 16 time-series samples (got {len(times)})."
        )

    sample_count = len(times)
    t_arr = np.asarray(times, dtype=float)
    order = np.argsort(t_arr, kind="mergesort")
    t_arr = t_arr[order]
    dt_vals = np.diff(t_arr)
    pos_dt = dt_vals[dt_vals > 0.0]
    if pos_dt.size == 0:
        raise ValueError(f"Raw ADXL CSV for axis {axis} has non-increasing timestamps.")

    dt_med = float(np.median(pos_dt))
    fs_hz = 1.0 / dt_med
    if fs_hz <= 2.0 * max(min_freq_hz, 5.0):
        raise ValueError(
            f"Raw ADXL sample rate ({fs_hz:.2f} Hz) is too low for resonance analysis."
        )

    series_arrays = {
        name: np.asarray(vals, dtype=float)[order] for name, vals in col_series.items()
    }

    selected_col: str | None = None
    for candidate_name in _AXIS_ACCEL_CANDIDATES[axis] + ("accel", "acc", "a", "signal"):
        if candidate_name in series_arrays and float(np.std(series_arrays[candidate_name])) > 1e-9:
            selected_col = candidate_name
            break

    if selected_col is None:
        dominant_name = max(series_arrays, key=lambda k: float(np.std(series_arrays[k])))
        if float(np.std(series_arrays[dominant_name])) > 1e-12:
            selected_col = dominant_name

    if selected_col is None:
        raise ValueError(f"Raw ADXL CSV for axis {axis} contains no dynamic acceleration signal.")

    signal = series_arrays[selected_col]

    # Resample onto a uniform grid if timestamps exhibit jitter or dropped SPI packets
    rel_dt_jitter = float(np.std(pos_dt)) / dt_med if pos_dt.size > 1 else 0.0
    if rel_dt_jitter > 0.15 or np.any(dt_vals <= 0.0):
        unique_t, unique_idx = np.unique(t_arr, return_index=True)
        if unique_t.size < 16:
            raise ValueError(f"Raw ADXL CSV for axis {axis} has insufficient unique timestamps.")
        uniform_t = np.arange(unique_t[0], unique_t[-1] + 0.25 * dt_med, dt_med, dtype=float)
        signal = np.interp(uniform_t, unique_t, signal[unique_idx])

    nperseg = _compute_welch_nperseg(len(signal), fs_hz)
    welch_freqs, welch_psd = welch(
        signal,
        fs=fs_hz,
        window="hann",
        nperseg=nperseg,
        detrend="constant",
        scaling="density",
    )

    return ParsedAdxlData(
        axis=axis,
        source_format="raw_accel_csv",
        freqs_hz=np.asarray(welch_freqs, dtype=float),
        psd=np.maximum(np.asarray(welch_psd, dtype=float), 0.0),
        sample_count=sample_count,
        selected_column=selected_col,
        sample_rate_hz=fs_hz,
    )


def parse_adxl_csv(
    csv_text: str,
    axis: str = "X",
    min_freq_hz: float = MIN_FREQ_HZ,
) -> ParsedAdxlData:
    """Parse either a Klipper `TEST_RESONANCES` PSD CSV or a raw ADXL345 time-series CSV."""
    axis_norm = axis.strip().upper()
    if axis_norm not in ("X", "Y"):
        raise ValueError(f"axis must be 'X' or 'Y', got {axis!r}")
    if not csv_text or not csv_text.strip():
        raise ValueError(f"CSV input for axis {axis_norm} is empty.")

    lines = csv_text.lstrip("\ufeff").splitlines()
    header_idx, cols, delim, source_format = _detect_csv_header(lines)
    data_lines = lines[header_idx + 1 :]

    if source_format == "klipper_psd_csv":
        return _parse_klipper_psd_rows(data_lines, cols, delim, axis_norm)
    return _parse_raw_accel_rows(data_lines, cols, delim, axis_norm, min_freq_hz=min_freq_hz)


def _pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return 0.0
    std_a = float(np.std(a))
    std_b = float(np.std(b))
    if std_a < 1e-12 or std_b < 1e-12:
        return 0.0
    r = float(np.corrcoef(a, b)[0, 1])
    if not math.isfinite(r):
        return 0.0
    return max(-1.0, min(1.0, r))


def _build_axis_diagnostics(
    spec: PrinterSpec,
    axis: str,
    measured_peak_hz: float,
    simulated_peak_hz: float,
    peak_delta_hz: float,
    peak_error_pct: float,
    measured_damping: float,
    simulated_damping: float,
    measured_graph: AxisShaperGraph,
    simulated_graph: AxisShaperGraph,
    spectral_correlation: float,
) -> tuple[list[str], list[str]]:
    """Generate physical root-cause diagnostics and actionable PrintSim/hardware adjustments."""
    diags: list[str] = []
    adjustments: list[str] = []

    belt = spec.gantry.belt
    pluck_hz = belt.pluck_freq_hz(150.0)

    if peak_error_pct < -10.0:
        diags.append(
            f"{axis}-axis measured peak ({measured_peak_hz:.1f} Hz) is {abs(peak_error_pct):.1f}% lower "
            f"than simulated ({simulated_peak_hz:.1f} Hz, {peak_delta_hz:+.1f} Hz): real machine has lower "
            f"effective stiffness or higher moving mass. Check belt tension / pluck frequency "
            f"(modeled {belt.pretension_n:.0f} N ≈ {pluck_hz:.0f} Hz on 150 mm span), linear rail preload/quality "
            f"(modeled {spec.gantry.rail_preload} / {spec.gantry.rail_quality}), or loose corner/XY-joint/motor-mount "
            f"fasteners (modeled '{spec.frame.corner_joint_type}', '{spec.gantry.bracket_material}')."
        )
        adjustments.append(
            f"[{axis}] Measure XY belt pluck frequency against {pluck_hz:.0f} Hz (150 mm span) and inspect "
            f"corner/XY-joint/motor fasteners; or lower gantry.belt.pretension_n / step down gantry.rail_preload "
            f"or frame.corner_joint_type in PrintSim if hardware tension is already at spec."
        )
    elif peak_error_pct > 10.0:
        diags.append(
            f"{axis}-axis measured peak ({measured_peak_hz:.1f} Hz) is {peak_error_pct:.1f}% higher "
            f"than simulated ({simulated_peak_hz:.1f} Hz, {peak_delta_hz:+.1f} Hz): real build is stiffer or lighter "
            f"than modeled. Likely tighter belt tension than {belt.pretension_n:.0f} N, stiffer/metal XY-joint or motor "
            f"brackets than '{spec.gantry.bracket_material}', stiffer structural panels/corner joints, or lighter "
            f"actual toolhead/gantry mass than {spec.toolhead.total_mass_g:.0f} g."
        )
        adjustments.append(
            f"[{axis}] Verify real belt tension is safely below {belt.max_working_tension_n:.0f} N working limit "
            f"and update gantry.belt.pretension_n, gantry.bracket_material, or toolhead.total_mass_g in PrintSim to match."
        )
    elif abs(peak_error_pct) >= 5.0:
        diags.append(
            f"{axis}-axis measured peak ({measured_peak_hz:.1f} Hz) has a minor {peak_error_pct:+.1f}% offset "
            f"vs simulated ({simulated_peak_hz:.1f} Hz): check fine belt tension balance, carriage preload "
            f"({spec.gantry.rail_preload}), and umbilical/wiring mass on toolhead."
        )
        adjustments.append(
            f"[{axis}] Fine-tune gantry.belt.pretension_n (~{belt.pretension_n:.0f} N) or toolhead.total_mass_g "
            f"to close the {abs(peak_delta_hz):.1f} Hz gap."
        )
    else:
        diags.append(
            f"{axis}-axis resonance frequency matches digital twin closely ({measured_peak_hz:.1f} Hz measured "
            f"vs {simulated_peak_hz:.1f} Hz simulated, {peak_error_pct:+.1f}% error)."
        )

    zeta_diff = measured_damping - simulated_damping
    if zeta_diff >= 0.015:
        diags.append(
            f"{axis}-axis measured damping (zeta={measured_damping:.4f}) is noticeably higher than simulated "
            f"({simulated_damping:.4f}, {zeta_diff:+.4f}): likely extra dissipation from frame feet "
            f"('{spec.frame.feet_type}'), enclosure panel foam seals, tight linear rail wipers/heavy grease, "
            f"or extrusion cavity fill ('{spec.frame.extrusion_filling}'). Verify rail carriages glide smoothly without binding."
        )
        adjustments.append(
            f"[{axis}] Check rail seal drag/grease viscosity and enclosure panel foam seals; update frame.feet_type "
            f"or frame.extrusion_filling in PrintSim if benchtop/panel damping exceeds baseline."
        )
    elif zeta_diff <= -0.015:
        diags.append(
            f"{axis}-axis measured damping (zeta={measured_damping:.4f}) is noticeably lower than simulated "
            f"({simulated_damping:.4f}, {zeta_diff:+.4f}): sharp lightly-damped resonance peak typical of "
            f"rigid bench mounting, bare unsealed panels/extrusions, or lightly-lubricated rail seals."
        )
        adjustments.append(
            f"[{axis}] Sharp resonance peak: consider anti_vib_pads/paver_on_foam feet, panel foam tape, or "
            f"MZV/EI shaper for wider frequency robustness; or set frame.feet_type='rigid_bolted' in PrintSim."
        )

    if spectral_correlation < 0.70:
        if abs(peak_error_pct) <= 10.0:
            diags.append(
                f"{axis}-axis spectral curve correlation is low (r={spectral_correlation:.2f}): secondary peaks or "
                f"broad/split modal energy detected (e.g. panel drum vibration, umbilical slap, or belt-span asymmetry)."
            )
            adjustments.append(
                f"[{axis}] Inspect secondary peaks in the PSD overlay (panels, Z-rails, toolhead umbilical) and prefer "
                f"MZV/EI over ZV if multiple modes sit within +/-15 Hz."
            )
        else:
            diags.append(
                f"{axis}-axis spectral curve correlation is low (r={spectral_correlation:.2f}), primarily driven by "
                f"the {peak_delta_hz:+.1f} Hz primary resonance shift (and any secondary structural/panel modes)."
            )

    if (
        measured_graph.recommended_shaper != simulated_graph.recommended_shaper
        or abs(measured_graph.recommended_max_accel_mm_s2 - simulated_graph.recommended_max_accel_mm_s2)
        > 0.15 * max(simulated_graph.recommended_max_accel_mm_s2, 100.0)
    ):
        diags.append(
            f"{axis}-axis shaper recommendation differs: measured recommends {measured_graph.recommended_shaper} "
            f"@ {measured_graph.recommended_freq_hz:.1f} Hz ({measured_graph.recommended_max_accel_mm_s2:.0f} mm/s^2) "
            f"vs simulated {simulated_graph.recommended_shaper} @ {simulated_graph.recommended_freq_hz:.1f} Hz "
            f"({simulated_graph.recommended_max_accel_mm_s2:.0f} mm/s^2)."
        )

    return diags, adjustments


def _calibrate_single_axis(
    spec: PrinterSpec,
    axis: str,
    csv_text: str,
    sim_graph: AxisShaperGraph,
    max_freq_hz: float,
) -> tuple[AxisCalibrationResult, list[str]]:
    parsed = parse_adxl_csv(csv_text, axis=axis, min_freq_hz=MIN_FREQ_HZ)

    band_mask = (parsed.freqs_hz >= MIN_FREQ_HZ - 1e-9) & (parsed.freqs_hz <= max_freq_hz + 1e-9)
    meas_freqs_band = parsed.freqs_hz[band_mask]
    meas_psd_band = parsed.psd[band_mask]
    if meas_freqs_band.size < 3:
        raise ValueError(
            f"Axis {axis} CSV has fewer than 3 samples in [{MIN_FREQ_HZ}, {max_freq_hz}] Hz."
        )
    if float(np.max(meas_psd_band)) <= 0.0:
        raise ValueError(
            f"Axis {axis} CSV has zero PSD signal energy within [{MIN_FREQ_HZ}, {max_freq_hz}] Hz."
        )

    measured_damping = estimate_psd_damping_ratio(meas_freqs_band, meas_psd_band)
    raw_peak_guess = float(meas_freqs_band[int(np.argmax(meas_psd_band))])
    meas_graph = calibrate_axis_shapers(
        axis=axis,
        primary_freq_hz=raw_peak_guess,
        scv=spec.kinematics.square_corner_velocity_mm_s,
        shaper_choice="auto",
        freqs_hz=meas_freqs_band,
        raw_psd=meas_psd_band,
    )

    sim_mask = sim_graph.freqs_hz <= max_freq_hz + 1e-9
    grid_freqs = sim_graph.freqs_hz[sim_mask]
    sim_psd_grid = sim_graph.raw_psd[sim_mask]
    if grid_freqs.size < 3:
        raise ValueError(f"max_freq_hz={max_freq_hz} yields fewer than 3 simulated frequency bins.")

    meas_psd_interp = np.interp(grid_freqs, meas_freqs_band, meas_psd_band, left=0.0, right=0.0)
    meas_peak_grid = float(np.max(meas_psd_interp))
    sim_peak_grid = float(np.max(sim_psd_grid))
    meas_norm = meas_psd_interp / max(meas_peak_grid, 1e-12)
    sim_norm = sim_psd_grid / max(sim_peak_grid, 1e-12)

    measured_peak_hz = float(meas_graph.peak_freq_hz)
    simulated_peak_hz = (
        float(sim_graph.peak_freq_hz)
        if sim_graph.peak_freq_hz <= max_freq_hz + 1e-9
        else float(grid_freqs[int(np.argmax(sim_psd_grid))])
    )
    peak_delta_hz = measured_peak_hz - simulated_peak_hz
    peak_error_pct = (peak_delta_hz / max(simulated_peak_hz, 1e-6)) * 100.0
    spectral_corr = _pearson_correlation(meas_norm, sim_norm)

    diags, adjustments = _build_axis_diagnostics(
        spec=spec,
        axis=axis,
        measured_peak_hz=measured_peak_hz,
        simulated_peak_hz=simulated_peak_hz,
        peak_delta_hz=peak_delta_hz,
        peak_error_pct=peak_error_pct,
        measured_damping=measured_damping,
        simulated_damping=float(sim_graph.damping_ratio),
        measured_graph=meas_graph,
        simulated_graph=sim_graph,
        spectral_correlation=spectral_corr,
    )

    result = AxisCalibrationResult(
        axis=axis,
        source_format=parsed.source_format,
        freqs_hz=[float(v) for v in grid_freqs],
        measured_psd_norm=[float(v) for v in meas_norm],
        simulated_psd_norm=[float(v) for v in sim_norm],
        measured_peak_hz=measured_peak_hz,
        simulated_peak_hz=simulated_peak_hz,
        peak_delta_hz=peak_delta_hz,
        peak_error_pct=peak_error_pct,
        measured_damping_ratio=measured_damping,
        simulated_damping_ratio=float(sim_graph.damping_ratio),
        measured_shaper=meas_graph.recommended_shaper,
        measured_shaper_freq_hz=float(meas_graph.recommended_freq_hz),
        measured_max_accel_mm_s2=float(meas_graph.recommended_max_accel_mm_s2),
        simulated_shaper=sim_graph.recommended_shaper,
        simulated_shaper_freq_hz=float(sim_graph.recommended_freq_hz),
        simulated_max_accel_mm_s2=float(sim_graph.recommended_max_accel_mm_s2),
        spectral_correlation=spectral_corr,
        sample_count=parsed.sample_count,
        diagnostics=diags,
    )
    return result, adjustments


def calibrate_printer_against_adxl(
    spec: PrinterSpec,
    axis_x_csv: str | None = None,
    axis_y_csv: str | None = None,
    max_freq_hz: float = 200.0,
) -> AdxlCalibrationReport:
    """Compare real Klipper ADXL345 / Shake&Tune CSV measurements against PrintSim FEA PSDs."""
    has_x = bool(axis_x_csv and axis_x_csv.strip())
    has_y = bool(axis_y_csv and axis_y_csv.strip())
    if not has_x and not has_y:
        raise ValueError("Provide at least one measured CSV string (`axis_x_csv` and/or `axis_y_csv`).")
    if max_freq_hz < 20.0:
        raise ValueError(f"max_freq_hz must be >= 20.0 Hz, got {max_freq_hz}")

    assembly = build_assembly(spec)
    modal_res = solve_modal_fea(assembly)

    axes_out: dict[str, AxisCalibrationResult] = {}
    all_adjustments: list[str] = []

    if has_x and axis_x_csv is not None:
        if modal_res.shaper_graph_x is None:
            raise RuntimeError("Modal FEA did not produce shaper_graph_x.")
        res_x, adj_x = _calibrate_single_axis(
            spec, "X", axis_x_csv, modal_res.shaper_graph_x, max_freq_hz
        )
        axes_out["X"] = res_x
        all_adjustments.extend(adj_x)

    if has_y and axis_y_csv is not None:
        if modal_res.shaper_graph_y is None:
            raise RuntimeError("Modal FEA did not produce shaper_graph_y.")
        res_y, adj_y = _calibrate_single_axis(
            spec, "Y", axis_y_csv, modal_res.shaper_graph_y, max_freq_hz
        )
        axes_out["Y"] = res_y
        all_adjustments.extend(adj_y)

    # Deduplicate adjustments while preserving order
    dedup_adjustments: list[str] = []
    seen: set[str] = set()
    for item in all_adjustments:
        if item not in seen:
            seen.add(item)
            dedup_adjustments.append(item)
    if not dedup_adjustments:
        dedup_adjustments.append(
            "Digital twin matches measured ADXL345 resonance within 5%; apply the recommended Klipper shaper settings directly."
        )

    axis_snippets = [
        (
            f"{ax}: {res.measured_peak_hz:.1f} Hz meas vs {res.simulated_peak_hz:.1f} Hz sim "
            f"({res.peak_error_pct:+.1f}%, r={res.spectral_correlation:.2f}, "
            f"rec {res.measured_shaper} @ {res.measured_max_accel_mm_s2:.0f} mm/s^2)"
        )
        for ax, res in axes_out.items()
    ]
    mean_abs_err = sum(abs(r.peak_error_pct) for r in axes_out.values()) / len(axes_out)
    fit_quality = (
        "excellent"
        if mean_abs_err <= 5.0
        else ("moderate" if mean_abs_err <= 10.0 else "divergent")
    )
    overall_summary = (
        f"{spec.name} ADXL345 calibration ({fit_quality} fit, mean |err|={mean_abs_err:.1f}%): "
        + " | ".join(axis_snippets)
    )

    return AdxlCalibrationReport(
        printer_name=spec.name,
        axes=axes_out,
        overall_summary=overall_summary,
        recommended_adjustments=dedup_adjustments,
    )
