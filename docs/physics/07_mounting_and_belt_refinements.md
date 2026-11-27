# 07 — Mounting (Feet, Ballast, Suspension) and Belt Refinements

Scope: how the printer's mount and extra frame mass change the synthesized input-shaper result,
the `suspended_lines` mount, and two belt refinements (hot-chamber tension rise, CoreXY A/B
tension mismatch). Code: `fea/modal.py`, `geometry/builder.py`, `fea/components.py`,
`kinematics/solvers.py`, `schema/printer.py`, `schema/overrides.py`.
Tests: `tests/test_mounting_and_belt_refinements.py`.

## 1. Mount-mode coupling (replaces the B-14 feet heuristic)

A printer on compliant feet or lines has six whole-printer rigid-body modes (two sway, yaw,
bounce, two rock). The FE model already contains them, because the feet are springs. What was
missing was how they act on a drive axis that resonates far above them.

The static nozzle-to-bed compliance on a compliant mount is the complete modal sum
`c_static = Σ_k R_k`, with `R_k = φ_rel,k (φ_kᵀF) / (m_gen,k ω_k²)`. A mount mode well below the
axis resonance does not respond quasi-statically there, and the frame it releases recoils against
the moving mass. For each mount mode (at least 50% of its strain energy in the mount springs):

| Quantity | Expression |
| :--- | :--- |
| Decoupling factor | `d(r) = r⁴ / (1 + r⁴)`, `r = f_axis / f_mount` |
| Inertia-relieved compliance | `c_dyn = c_static − Σ_mount R_k d(r_k)` |
| Point inertance at the toolhead | `A_k = (φ_kᵀF)² / m_gen,k` [1/kg] |
| Reduced moving mass | `m_red = m (1 − m Σ_mount A_k d(r_k))` |
| Axis frequency | `f = 1 / (2π √(m_red (c_dyn + c_drive)))`, fixed-point on `r_k` |

`d(r)` is the regularised limit of the undamped receptance factor `1/(1 − r²)`: 0 when the mount
is stiff compared with the axis (quasi-static spring, the old series-compliance picture), 1 when
the frame is inertially free. The singularity at `r = 1` is removed because the mount mode is
drawn as its own peak. For a rigid frame of mass `M_f`, `A = 1/(M_f + m)` and the result is the
textbook two-body value `f = f_fixed √(1 + m/M_f)`.

Consequences (Voron Trident 300 preset, values move with the preset):

| Mount | Whole-printer modes | Frame recoil X / Y | Effective frequency vs bolted |
| :--- | :--- | :--- | :--- |
| `rigid_bolted` | none | 0 | reference |
| `rubber_feet` | 23–65 Hz | 5% / 10% | +5% |
| `anti_vib_pads` | 11–32 Hz | 6% / 11% | +6% |
| `paver_on_foam` (18 kg) | 15–28 Hz | 4% / 8% | +4% |
| `suspended_lines` (paracord, 600 mm) | 0.64–7.3 Hz | 6% / 11% | +6% |

A heavier frame recoils less, so the result converges on the bolted case as mass is added.
`ModalResult.mounting_dynamics` reports the modes, the recoil ratio per axis, and the peak a
toolhead accelerometer reads for each mode, `m A_k ζ_main / ζ_mount`. That peak is absolute
motion: nozzle and bed move together in it, which is why soft mounts can make accelerometer
graphs look worse without changing prints.

Limits: mount stiffness and damping values are estimates (no load-deflection data exists for
common printer feet); elastomer dynamic-to-static stiffness ratio (1.3–1.7 for rubber, up to 4 for
soft Sorbothane) and load stiffening are not modelled; the bench itself is rigid.

## 2. Frame ballast

`frame.ballast_mass_kg` and `frame.ballast_location` (`base`, `top`, `distributed`) lump dead
weight on the frame corner nodes. On `rigid_bolted`, base ballast sits on grounded nodes and is
inert, as weight bolted to the bench would be; top ballast lowers the racking modes. On compliant
mounts it lowers the rigid-body modes as `√(k/M)` and cuts recoil. On lines it raises tension
and leaves the pendulum frequency unchanged.

## 3. Suspended mount (`feet_type: suspended_lines`)

One line per foot, attached at the frame corner directly above it (top corners of a CoreXY cube
or delta towers; frames with no upright there hang by the base corners), anchored
`suspension_length_mm` overhead.

| Property | Model | Check |
| :--- | :--- | :--- |
| Axial stiffness per line | `EA / L` | bounce `(1/2π)√(n EA / (L M))` within 3% of FE |
| Lateral stiffness per line | `T / L`, `T = M g / n` (geometric) | sway `(1/2π)√(g/L)` within 1.5% of FE, mass-invariant |
| Rotational stiffness | 0 (a line carries no moment) | yaw = sway × (attach radius / yaw gyradius) |
| Safety | `break strength / T` vs the line's design factor | flagged in diagnostics |

Line catalog (`SUSPENSION_LINE_CATALOG`, working-load secant `EA`):

| Line | EA (N) | Break (N) | Basis |
| :--- | ---: | ---: | :--- |
| `paracord_550` | 4,000 | 2,447 | MIL-C-5040H Type III: ≥550 lbf at ≥30% elongation (≤8.2 kN secant at break); estimate at light load |
| `uhmwpe_braid_200lb` | 26,000 | 890 | Samson AmSteel-Blue elastic elongation 0.46/0.70/0.96% at 10/20/30% of break (EA ≈ 29× break); Hangprinter firmware uses 20–100 kN |
| `nylon_mono_100lb` | 1,700 | 445 | 20–30% elongation at break; estimate |
| `steel_cable_1_5mm` | 101,000 | 1,500 | S3i 7x7 stainless: 57.3 GPa on nominal area |
| `bungee_cord_6mm` | 85 | 534 | MIL-C-5651 Type I 1/4 in: 10–18 lbf at 75% elongation |

All mount modes of a suspended printer are reported under `mounting_dynamics`, never as the
structural fundamental. Not modelled: the gravity righting moment `M g h` of a centre of gravity
below the attachment plane (matters only for bungee-class lines), splayed lines, line creep, and
swing build-up during printing (only the resonant reversal length `v / (2 f_sway)` is reported).
No accelerometer data from a hung printer was found to calibrate against. Published tests
(Sanladerer 2022, Ender-3 on elastic tie-downs) found prints identical to stock feet.

## 4. Belt refinements

### 4.1 Hot-chamber tension rise

A belt tensioned on a cold frame is stretched further when the frame expands more than the cord:

`ΔT = EA_static · crimp(T₀) · (α_frame − α_cord) · (T_chamber − T_set)`

with `α_cord` = 5.0 (E-glass), −2.0 (aramid), 11.5 (steel) ppm/K. Opt-in through
`gantry.belt.pretension_set_temp_c`; when it is unset, `pretension_n` is taken as the operating
tension, so existing presets are unchanged. Check: Voron 2.4 owners measure about 100 Hz → 145 Hz
on a 150 mm span of 6 mm 2GT from 21 °C to a >50 °C chamber (7.6 N → 15.9 N); the model gives
14.6 N with the datasheet static `EA` of 16.9 kN (15.4 N with the earlier 18.9 kN).

### 4.2 CoreXY A/B tension mismatch

With the motors holding, belt A stretches by `x + y` and belt B by `x − y`:

`K = [[k_A + k_B, k_A − k_B], [k_A − k_B, k_A + k_B]]`, `M = diag(m_x, m_y)`.

`gantry.belt.tension_mismatch_pct` sets `T_A,B = T (1 ± p/200)`. With a constant `EA` a tension
difference would not change stiffness; the loops differ because they sit at different points of
the low-tension toe. The coupling `κ = (k_A − k_B)/(k_A + k_B)`, scaled by the belt's share of
each axis' compliance, is solved as an exact 2-DOF problem. Each axis' graph gains the other
axis' resonance as a secondary peak, and `ModalResult.belt_dynamics` reports both tensions, pluck
frequencies, the coupling and the cross peaks. At tensions past the toe (≥4 N/mm of width) the
coupling is a few percent; near the Voron-recommended 110 Hz it is larger.

### 4.3 Static `EA` correction

The catalog's Gates static `EA` used the 2M / 3M unit-strain modulus (18,000 / 30,000 lb/in per
inch of width) at every width. Gates' own long-length belting table (Light Power & Precision
manual, Table 6, tension at 0.1% elongation) and the narrow-belt rule give lower values for the
widths printers use. `ea_static_n` now equals PrintDB `tensile_stiffness_n_per_strain`
(parity-tested). An independent source agrees: Mitsuboshi patent US12247639B2 measures 29.8 N/%
per mm of width (2.98 kN/mm) on E-glass 2.0–2.5 mm pitch belts, within 6% of the Gates 2M figure.

| Belt | Static `EA` before → after (kN) | Basis |
| :--- | :--- | :--- |
| 2GT 6 mm | 18.9 → 16.9 | Table 6 LL2MGT06, 3.8 lb |
| 2GT 9 mm | 28.35 → 25.5 | Table 6 LL2MGT09, 5.7 lb (25.4 kN) |
| 2GT 10 mm, GT3 2MGT 10 mm | 31.5 → 28.3 | 10/9 × the 9 mm value |
| 2GT 12 mm | 37.8 → 34.0 | 2M modulus × 0.90 narrow-belt factor |
| 2GT 15 mm | 47.25 (unchanged) | wider than 1/2 in: no factor |
| 2GT dual 9 mm | 56.7 → 51.0 | 2 × the 9 mm value |
| 3GT 9 / 12 / 15 mm | 47.25 / 63.0 / 78.75 → 39.6 / 56.5 / 73.4 | Table 6 LL3MGT09 (8.9 lb), midpoint, LL3MGT15 (16.5 lb) |

The dynamic stiffness as installed is `ea_static_n ×` one family ratio
(`GATES_FIBERGLASS_DYNAMIC_STIFFENING`; GT3 2MGT × 0.97). That ratio is set from controlled
measurements in section 5.2: **1.85**, replacing the earlier fitted 2.0 and 2.23.

### 4.4 First-principles belt micromechanics, 3D Tet10 pitch-cell FEA & nonlinear shaker dynamics (2026-11-27)

All four legacy empirical belt sub-models (`0.52 / 2.3` exponential crimp fit, `0.0015 / 0.0145 / 0.036` flexural rigidity constants, `5.5e4` + `0.22 / 2.2` tooth seating, and `modal.py`'s `1 - 0.42 u_pen^1.5` / `0.30 + 0.85 exp(-q0/1.4)` / `0.82 f_eff` shaker heuristics) have been replaced by first-principles continuum/analytical models (`fea/belt_micromechanics.py`, `fea/belt_nonlinear.py`):

| Subsystem | First-Principles Model | Calibrated Constants / Validation |
| :--- | :--- | :--- |
| **Cord crimp toe** (`belt_crimp_factor`) | Costello helical-cord radial compaction against fiber/matrix confinement: `EA_tan(T) = EA_inf / (1 + β_0 exp(-λ_comp T))`, saturating at `N_c E_f A_f cos³(α)` | 6 mm 2GT E-glass 135 tex: `12.1 kN` @ 0 N (`+3.7%` vs Wang 2018 `11.7 kN`) → `22.5 kN` @ 100 N (`-0.5%` vs Wang `22.6 kN`); `0.1%` secant matches Gates Table 6 `16.9 kN`. |
| **Tooth-mesh shear & seating** (`gerbert_tooth_mesh_stiffness_n_m`) | 3D Tet10 active-set unilateral pulley-groove seating FEA (`scripts/calibrate_belt_pitch_cell_fea.py`, `data/touchstones/belt_pitch_cell_fea.json`) + Gerbert discrete spring chain with wrap pressure ratio `20 / Z_pulley` & Ling et al. (2020) DMA filled-EPDM small-strain dynamic factor (`EPDM_DYNAMIC_TOOTH_MODULUS_FACTOR = 1.45` at 40–150 Hz; `1.0` for TPU) | Static Tet10: `2GT`: `k_t_full = 32,934 N/(m·mm)`, `s_0 = 0.76`, `q_s = 5.85 N/mm`; `3GT`: `41,378`, `0.32`, `1.50`; `AT3`: `108,743`, `0.05`, `2.40` (mesh convergence `≤ 2.1%`). Dynamic 2GT EPDM: `47,755 N/(m·mm)`. |
| **Composite bending `EI_b` & wrap radius** (`belt_flexural_rigidity_nm2`, `belt_effective_ea_n`) | Exact 3D Tet10 2×2 ABD static-condensation bending FEA partitioned into cord-dominated & thermally softening matrix fractions + serpentine back-idler/90° corner effective bend radius `R_eff = max(0.5·(R_p + R_back + h_belt), R_p + 0.75·h_belt)` | `2GT`: `0.001917 N·m²/m` (`70.6%` cord / `29.4%` elastomer); `3GT` (`2.41 mm`): `0.01289`; `AT3` (`1.90 mm`): `0.006225` (proving thick `3GT > AT3 > 2GT` in bending). |
| **Elastomer DMA thermal retention & damping** (`BeltSpec.tooth_modulus_retention`, `belt_damping_ratio`) | Ling et al. (2020) DMA on EPDM/CSM timing-belt compound (`tan δ ≈ 0.06–0.08` → `ζ ≈ 0.03–0.04` base; `E'` drops `44%` from `25 → 120 °C`) | EPDM rated plateau `≤ 80 °C` (`1.0`), linear DMA decline to `0.56` at `120 °C` (floor `0.55`); TPU softens `> 45 °C` (`0.72` @ 65 °C, floor `0.35`). |
| **Coupled nonlinear shaker dynamics** (`solve_nonlinear_belt_shaker_response` in `modal.py`) | Vectorized harmonic-balance / describing-function solver (`~2 ms/axis`, Pyodide-WASM safe) on 2-DOF `[m_rot, m_car]` + unilateral tension-only Costello `EA(T)` branches + Beikmann–Perkins–Ulsoy (1996) 1:1 sag/runout & 2:1 parametric Mathieu–Duffing transverse whip | Cross-validated against full time-domain swept-sine ODE (`simulate_swept_sine_time_domain`). Eliminates double-damping by feeding pure DMA viscoelastic `zeta_belt_mat` into the nonlinear solver and mapping intermittent slack duty cycle `D > 0.28` onto the exact unloaded-branch frequency `f_slack`. |

**Calibrated empirical benchmark summary (`printsim shaper-benchmark`, baselines refreshed):**
- **37 CoreXY machines (65 axes):** Overall MAPE improved **`9.88% → 9.73%`** (`-0.15%`; X `10.00%`, Y `9.50%`; `≤10%`: `39/65`, `≤15%`: `48/65`, `≤20%`: `58/65`).
- **All 40 machines (71 axes):** Overall MAPE improved **`10.69% → 10.61%`** (`-0.08%`).
- **17 controlled same-machine A/B sweeps:** **`17/17` direction match**, Ratio MAPE improved **`3.95% → 3.65%`** (`-0.30%`):
  - `jeff.k` Trident `6.2 → 9.8 N` tension sweep: X **`+0.8%`** err (`1.141×` vs `1.132×`, was `+6.1%`), Y `-6.1%` err (`1.132×` vs `1.206×`).
  - CloakedWayne Monolith AWD `1.0 → 2.8 A` current sweep: **`+0.7%`** err (`1.178×` vs `1.170×`).
  - CloakedWayne `10 mm` GT3 vs GT2 EPDM swap: **`-1.1%`** err (`0.961×` vs `0.972×`); Aramid vs Fiberglass `+3.6%`.
  - LH 3-point tension sweep (`13.3 → 19.2 → 25.0 N`; measured `66.4 → 67.6 → 68.2 Hz` MZV): reproduces strictly concave diminishing frequency rise (`+4.5%` then `+2.2%`) and PSD sharpening.
  - CloakedWayne `12 mm` GT3 `89 → 200 N` (`+4.0% ZV / +5.75% MZV`): pure 2GT fiberglass Costello crimp is already `>99.9%` saturated at `89 N / 12 mm = 7.4 N/mm` (`+0.8%` on standard 2GT), confirming that the larger measured `89 → 200 N` jump on GT3 reflects GT3's softer compound (`e_seat = 0.90` at 89 N) seating further toward GT2 EPDM plus idler-bearing preload at extreme tension.

### 4.5 Still open

| Item | Finding |
| :--- | :--- |
| Dynamic ratio `1.85` | Controlled testbeds support `1.6–2.0×` over Gates Table 6 static `EA`; no published 40–150 Hz dynamic `EA(f, T_0)` bench curve for glass-cord timing belts yet |
| Linear mass | Gates sonic-meter constant `1.4 g/m` per mm → `8.4 g/m` (6 mm), `12.6 g/m` (9 mm); catalog agrees (`8.3 / 12.45 g/m`) |
| 3GT / AT3 real CoreXY ADXL data | 3D Tet10 pitch-cell FEA & Costello cord models now anchor `k_t`, `EI_b`, and crimp, but zero real CoreXY ADXL sweeps exist for 3GT or AT3 |

## 5. Identifying belt stiffness from what an owner can measure

The catalog's dynamic `EA` is a fitted number. It cannot be read from a datasheet: Gates quotes
one low-load point (tension at 0.1% elongation, 16.9 N on a 6 mm belt), and the published pull
tests sample too coarsely to resolve the 5–90 N range printers run in. The belt's stiffness as
installed has to be measured, and three things an owner can already measure are enough.

**Tension from the plucked span** is exact string physics, with a bending correction for short
spans:

`f = (1 / 2L) √(T / μ) √(1 + π² EI / (T L²))`, so `T = 4 μ L² f² − π² EI / L²`.

Check: the Monolith matched-tension shootout set four 10 mm belts to about 89 N. Their different
masses (12.36, 14.99, 13.99, 12.88 g/m) predict 282.9, 256.8, 265.9 and 277.1 Hz on a 150 mm
span; measured 283, 257, 266 and 277 Hz.

**Stiffness from the axis resonance.** One axis is a chain of springs under the moving mass:

`1 / (m (2π f)²) = C₀ + G / EA`

`G = 1 / Σ(1 / Lᵢ)` is the belt-path geometry factor over the parallel branches between carriage
and motor pulleys. It is fixed by frame size and toolhead position, and it is the only term that
moves with position; `EA` is the only term that moves when the belt is swapped or re-tensioned.
`C₀` is everything else in series (tooth mesh, motor, mounts, frame, toolhead).

| Experiment | What varies | What it returns |
| :--- | :--- | :--- |
| Position sweep: same axis at 3+ toolhead positions | `G` | `EA` and `C₀` by a straight-line fit |
| Belt swap: two belts of known stiffness ratio `ρ` | `EA` | belt share `s = (1 − (f_A/f_B)²) / (1 − 1/ρ)` |
| Tension sweep: same belt at two pluck frequencies | `EA(T)` | `EA_hi / EA_lo = 1 + ((f_hi/f_lo)² − 1) / s` |

`printsim belt-identify <spec> --axis X` prints where to measure and what the model predicts;
adding `--point u,v,freq_hz` for each position returns the in-place `EA`, the belt share and the
fit quality. Code: `fea/belt_identification.py`.

Belt geometry in the forward model follows the toolhead position (it used centre-of-bed
geometry everywhere). With the perimeter routing of section 5.2 a 2WD machine's geometry factor
changes little across the bed, so the position sweep is informative mainly on AWD machines,
where the toolhead gets close to a motor; on 2WD a belt swap or an AWD comparison separates the
belt better.

What the available observations say when put through these relations:

| Observation | Result |
| :--- | :--- |
| One machine, 6 mm → 12 mm belt: 40.9 → 47.2 Hz | Belt carries 50% of the axis compliance at 6 mm |
| One machine, pluck 80 → 120 Hz: shaper 32 → 34 Hz | `EA` rises 20–37% for 2.25× tension (`EA ∝ T^0.2–0.4`), for a belt share of 0.35–0.65 |
| Gates break strength / `EA` (556 N / 16.9 kN, 845 N / 25.4 kN) | 3.3% strain if linear to break, the failure strain of glass filament: the datasheet figure is close to the elastic stiffness of the glass itself |
The break-strength row bounds the answer from above: a cord cannot be much stiffer than its own
glass. Glass-cord datasheets (NGF) give a 135 tex cord a break load of 78–88 N at 2.6%
elongation, about 3.0–3.4 kN per cord, so the elastic stiffness of the glass in a belt is
`break strength / 2.6%`: 21–24 kN for a 6 mm 2GT at its minimum rated strength, more at typical
strength. An installed stiffness of 1.6–2.0× the low-load datasheet point is consistent with that.

### 5.1 Cross-machine data cannot settle it

`scripts/belt_split_study.py fit` sweeps belt stiffness and mount stiffness against five
community machines with documented shaper results. The error is nearly flat across belt values
because per-machine errors of 10–25% (opposite sign on X and Y, different on each machine) swamp
the difference; one Trident with raw accelerometer files in the project measures 50.3 / 36.1 Hz
against a preset of 67 / 47 Hz. Only a controlled change on one machine separates the belt from
the rest.

### 5.2 Controlled same-machine measurements (Monolith 300 mm testbed)

CloakedWayne's testbed (Monolith Discord `#updates`; 300 mm, 4040 / 2040 frame, Carbonara beam,
10 mm Gates EPDM belts at about 89 N, LDO 42STH48-2504AH at 36 V, Beacon accelerometer, room
temperature) was measured with one thing changed at a time. ZV frequencies read from the posted
Klipper graphs:

| Experiment | Measured | Model before | Model now |
| :--- | :--- | :--- | :--- |
| AWD, 2.5 A (2025-07-10) | X 178.6, Y 114.4 Hz | 168.6, 119.0 | 162.9, 113.2 |
| 2WD, same machine and day | X 129.0, Y 83.2 Hz | 139.3, 95.8 | 125.0, 83.8 |
| AWD / 2WD frequency ratio | 1.384, 1.375 | 1.210, 1.242 | 1.30, 1.35 |
| Current sweep, AWD Y at 1 / 1.5 / 2 / 2.5 / 2.8 A (2025-08-12) | 97.4 / 104.0 / 107.6 / 110.4 / 111.6 Hz | 100.4 / 110.1 / 115.8 / 119.0 / 120.2 | 97.8 / 106.0 / 110.6 / 113.2 / 114.2 |
| All nine points, rms error | | 5.9% | 3.5% |

Two results follow.

**Belt routing.** Going from AWD to 2WD multiplies the non-motor compliance by 1.9–2.0 on both
axes. The earlier 2WD model had one short branch straight from the gantry to the motor and one
long return branch, `EA (1 / 0.5P + 1 / 1.5P) = 2.67 EA / P` at bed centre, only 1.5× softer than
AWD. In the real perimeter routing (Voron, Monolith, classic CoreXY) neither branch runs straight
to the motor: one goes forward to the front idler and back along the side, the other to the far
rear idler and across the rear. Both are about `P` long at centre, giving `2 EA / P`, exactly
half the AWD value of `4 EA / P`. The branch lengths in `compute_drive_stiffness_details_n_per_m`
now follow that routing. Consequences: both layouts are softest at bed centre; a 2WD machine
changes little front to back; an AWD machine changes more, because the toolhead gets closer to a
motor. The developer of that testbed describes the same behaviour independently.

**Motor and belt shares.** In the current sweep `1 / f²` is linear in `1 / I` to within 0.6%, the
signature of a magnetic spring in series, and its slope puts the motor at 19% of the Y-axis
compliance at 2.5 A. With the routing fixed, the one remaining free number is the belt's dynamic
stiffness. Values from 1.6 to 2.0 times the datasheet figure reproduce all nine measurements
about equally well (3.3–3.8% rms depending on whether motor stiffness is taken at standstill);
**1.85** is used. For the 10 mm belt at 89 N that is 52 kN as installed against 28.3 kN on the
datasheet, and 63 kN in the earlier model.

Remaining gap: the modelled AWD / 2WD ratio is still 3–6% under the measurement.

### 5.3 Stock Voron check at measured belt tension

The stock Voron presets looked unbalanced against the curated touchstone rows (X high, Y low).
That was not a model error. Those rows are typical values rather than measurements, with X / Y
ratios of 1.14–1.21 that a stock Voron's moving-mass ratio (about 2.1) cannot produce, and the
presets ran 28 N of belt tension where Voron documents a 110 Hz pluck (9–10 N). Against
individually measured stock Tridents at their measured tension:

| Measurement | Measured | Model, lumped bracket estimate | Model, CAD kit brackets (current tree) |
| :--- | :--- | :--- | :--- |
| Trident 300 hot, 2.2 lb (9.8 N), MZV | X 61.4–64.0, Y 40.8–42.2 Hz | 58.1, 41.1 | 64.3, 45.3 |
| Same machine cold, 1.4 lb (6.2 N), MZV | X 55.4, Y 34.4 Hz | 49.6, 37.0 | 54.5, 41.1 |
| Trident R2 + Tap, peaks, 110 Hz pluck assumed | X 50.3, Y 36.1 Hz | 52.6, 37.4 | 57.5, 41.0 |

X / Y is 1.39–1.61 measured and 1.34–1.42 modelled. Tightening from 1.4 to 2.2 lb raised that
machine 11–23%; the model gives 10–18%. The gantry-bracket model changed while this was being
checked (a lumped printed-bracket estimate of about 3.4 µm/N was replaced by CAD kit brackets
at about 0.5 µm/N); the stiffer brackets fit X slightly better and Y worse, and the Tap machine
reads about 14% high with them.

**Preset change (2026-11-26).** The four stock Voron presets (`voron_trident_300`,
`corexy_voron_style`, `ldo_voron_trident_revd_300`, `ldo_voron_24_revd_300`) now ship
`pretension_n: 9.0`, which is a 110 Hz pluck on a 150 mm span with the catalog belt mass.

| Preset | X / Y effective (Hz) | X / Y MZV (Hz) | Recommended accel (mm/s²) |
| :--- | :--- | :--- | :--- |
| `voron_trident_300` | 60.6 / 42.6 | 62.8 / 44.2 | 7100 |
| `corexy_voron_style` | 59.1 / 41.6 | 61.7 / 43.2 | 6800 |
| `ldo_voron_trident_revd_300` | 63.7 / 44.0 | 66.3 / 45.3 | 7600 |
| `ldo_voron_24_revd_300` | 62.4 / 43.1 | 64.6 / 45.3 | 7200 |
| `corexy_flying_gantry_300` (generic starter) | 57.7 / 39.9 | 60.2 / 41.4 | 6400 |
| `corexy_fixed_gantry_300` (generic starter, Studio default) | 59.1 / 40.7 | 61.6 / 42.4 | 5300 |

The measured Trident (2.2 lb, hot) is MZV 61.4–64.0 / 40.8–42.2 Hz. The two generic 6 mm 2WD
starters were at 28 N as well (X 63.9 / Y 44.3 and X 65.8 / Y 45.4 Hz, 6700 mm/s² each) and
take the same 9.0 N. The tuned builds keep their own tensions.

**Literature check (2026-11-26).** No published work identifies GT2 belt stiffness in place on a
printer or stage and reports it against the datasheet, and no measured dynamic-to-static ratio
exists for a glass- or aramid-cord timing belt. What the search did establish:

| Chain element | Independent source | Model (Trident, X axis) |
| :--- | :--- | :--- |
| Stepper magnetic spring | Jones: `k = N_r T_hold (I / I_rated)` = 16.0 N·m/rad at standstill | 14.5 N·m/rad per motor |
| Tooth mesh | Koyama 1979: 17–29 N/mm per tooth per mm width (L-section), scaled to 2.1–3.4 kN/mm | 4.0 kN/mm, 6% of cord compliance |
| Belt cord | controlled testbed data: 1.6–2.0× datasheet (section 5.2) | 1.85× datasheet at working tension |
| Mounts and brackets | none | 3.4 µm/N lumped estimate |

Motor and tooth mesh agree with independent figures. Synthetic-fibre ropes show dynamic
stiffness of 2–3× static that rises with mean load and falls with amplitude (ABS mooring
guidance); that comes from the polymer's amorphous phase, which glass lacks, so it is an analogy
only. The belt's dynamic ratio therefore rests on the controlled measurements in section 5.2.

## 6. References

- Barry Controls, isolator selection guide (damping ratios; coupled sway/rock modes).
- Mason UK AB104 (natural frequency from static deflection, dynamic/static ratio).
- Sorbothane data sheet 101 (tan δ and dynamic modulus by durometer).
- Structural Concepts, University of Manchester (pendulum on parallel and inclined lines).
- Samson Rope AmSteel-Blue; S3i wire rope technical data; MIL-C-5040H; MIL-C-5651.
- Gates Light Power & Precision Drive Design Manual, Table 6 (long-length belting); Gates/SDP "Belt Tensile Properties"; Gates Mectrol "Timing Belt Theory"; Gates sonic tension meter manual.
- Wang et al. 2018, Robotics 7(4):75 (GT2 belt nonlinear stiffness).
- Voron forum, "Belt tension vs temperature"; Voron secondary printer tuning guide.
- Klippain Shake&Tune belts-comparison documentation.
- CloakedWayne, Monolith Discord `#updates`: "LDO Gantry Kit in 2WD Config", "SLM MGN12H vs LDO kit MGN9H XY-joints", "Stepper Currents" input-shaper comparisons (2025).
- Mitsuboshi US12247639B2 (belt modulus by cord type); NGF Europe glass cord brochure.
- D. W. Jones, "Stepping Motor Physics" (University of Iowa); Koyama et al. 1979, Bull. JSME 22(169); ABS Guidance Notes on Fiber Rope for Offshore Mooring (2011).
- T. Sanladerer 2022, "Wobbly 3D printer makes for better parts"; CNC Kitchen paver noise test.
