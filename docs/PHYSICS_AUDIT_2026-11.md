# PrintSim Math & Physics Audit — 2026-11-24

Read-only audit of every physics-bearing module. No source file was changed by the audit.
This document is written so that another agent can pick up any finding and fix it without
re-deriving the problem.

## 0. How to use this document

- **Baseline.** Audit ran against the working tree on top of commit `45e7e40`, which had
  uncommitted edits in `engine/evaluator.py`, `fea/bed_plate_fem.py`, `fea/klipper_shaper.py`,
  `fea/mechatronics.py`, `fea/modal.py`, `gcode/ballbar.py`, `geometry/bom.py`,
  `geometry/builder.py`, `kinematics/solvers.py`, `schema/printer.py` and others.
  **Line numbers refer to that working tree and will drift.** Each finding quotes the offending
  expression; grep for the expression, do not trust the line number.
- **Finding IDs** (`A-01`, `B-03`, …) are stable handles. Refer to them in commits and PRs.
- **Groups.**
  - **A** — local bugs. The fix is confined to a few lines and does not need recalibration.
  - **B** — model errors. The formula or model is wrong or inconsistent, but other constants
    have probably been tuned around it. Fixing one of these **changes headline outputs** and
    needs a recalibration pass against `docs/community_touchstones.md`,
    `tests/test_calibration_touchstones.py` and `tests/test_community_realism.py`.
  - **C** — scoring and optimizer design. The physics underneath is fine; the metric measures
    the wrong thing.
  - **D** — catalog values to check against datasheets.
  - **E** — minor inconsistencies and naming.
- **Project rules still apply** (`AGENTS.md`): strict SI inside solvers, suffixed units at the
  boundary, and every physics change needs a test in `tests/` against a closed-form result or a
  monotonic sanity check. Run `uv run pytest -v` and
  `uv run printsim evaluate examples/corexy_voron_style.yaml` before declaring anything done.
- **Tests that pin current behaviour** are listed in section 9. Several findings are deliberate
  calibration choices guarded by a test; changing them means changing the test on purpose.

### Evidence legend

| Tag | Meaning |
|---|---|
| **REPRODUCED** | The lead reviewer re-ran it independently and got the stated numbers. |
| **CODE-CONFIRMED** | The lead reviewer read the quoted code and it is as described; the numbers come from the domain reviewer's own run. |
| **REVIEWER-RUN** | Confirmed numerically by the domain reviewer only; not independently re-checked. |
| **REASONED** | Derived on paper; no numeric confirmation. Treat as a hypothesis until tested. |
| **FROM-MEMORY** | Reference value recalled, not looked up in a datasheet or source. Verify before acting. |

The audit did **not** run the pytest suite, and made no web or datasheet lookups.

---

## 1. Headline

The core mathematics is correct. Verified against closed form (see section 8): the 12-DOF beam
element and consistent mass matrix, the generalized eigen-solve, Klipper shaper definitions and
smoothing/max-accel formulas, FK/IK for all three kinematics, Hagen–Poiseuille and power-law
pressure drop, the thermal resistance network, MITC4 plate bending, Guyan and Craig–Bampton
reduction, LM-FBS receptance coupling, SIMP sensitivities and Fox–Kapoor/adjoint sensitivities.
**No unit slip (mm/m, g/kg, MPa/Pa, N/mm vs N/m, Hz vs rad/s, °C vs K, deg vs rad) was found in
any module.**

The errors are in the layers that turn those solvers into recommendations and scores: load
patterns, the frequency post-processing in `modal.py`, the motor model, voxel meshing of
brackets, and what the composite score and Pareto objectives actually measure.

## 2. Summary table

| ID | Sev | Area | One line | Evidence |
|---|---|---|---|---|
| A-01 | High | static FEA | Y inertial load share omits hotend mass | CODE-CONFIRMED |
| A-02 | High | static/modal | Deltas get the CoreXY gantry load split | REPRODUCED |
| A-03 | High | kinematics | Delta workspace check tests square bed corners | REPRODUCED |
| A-04 | Med | collision | No tower-height check for deltas | REVIEWER-RUN |
| A-05 | Med | thermal | Chamber temperature never reaches the hotend model | REPRODUCED (grep) |
| A-06 | Med | thermal | Arrhenius shift applied to `K`, should be `K·a_T^n` | CODE-CONFIRMED |
| A-07 | Med | modal | Eigen-solve is not deterministic | REPRODUCED |
| A-08 | Med | shaper | Shaper frequency search capped at 1.35× resonance | CODE-CONFIRMED |
| A-09 | Med | G-code | Parser drops `G2`/`G3` and extrude-only moves | CODE-CONFIRMED |
| A-10 | High | Pareto | "Safe accel" is 1.25× motor limit; constraints at wrong accel | CODE-CONFIRMED |
| A-11 | Low | builder | Torsion constant set from `iy_m4` (probable typo) | CODE-CONFIRMED |
| A-12 | Med | BOM | Z rails sized from the Y-rail profile | CODE-CONFIRMED |
| A-13 | Med | mechatronics | FOC gains sized on 2× the per-motor load | CODE-CONFIRMED |
| A-14 | Med | ballbar | FOC tracking lag counted twice | CODE-CONFIRMED |
| A-15 | Low | sensitivity | Nozzle gyration radius disagrees with mass matrix | REVIEWER-RUN |
| A-16 | Low | mechatronics | Delta reflected rotor mass over-reported | REVIEWER-RUN |
| A-17 | Low | ballbar | Corner benchmark snaps last trapezoid sample | REVIEWER-RUN |
| A-18 | Low | optimizer | Collision penalty applied twice | REASONED |
| A-19 | Low | sensitivity | Compliance share uses `abs(u_rel)` (latent) | REASONED |
| A-20 | Med | ADXL | Raw accelerometer path skips Klipper normalisation | REVIEWER-RUN / FROM-MEMORY |
| A-21 | Low | beam FEA | Member self-weight has no fixed-end moments | REVIEWER-RUN |
| A-22 | Low | static FEA | Stress envelope includes pseudo-profile members | REVIEWER-RUN |
| B-01 | High | modal | Structure + drive combined as frequencies with different masses | CODE-CONFIRMED |
| B-02 | Med | modal | "Structural frequency" is not a resonance | REVIEWER-RUN |
| B-03 | High | kinematics | Delta motor torque/speed model strongly optimistic | CODE-CONFIRMED |
| B-04 | Med | kinematics | Stepper back-EMF constant mixes peak and RMS | CODE-CONFIRMED |
| B-05 | Med | kinematics | Motor max velocity is a hard corner-speed cut-off | CODE-CONFIRMED |
| B-06 | Med | kinematics | CoreXY motor-limited accel conservative and mislabelled | CODE-CONFIRMED |
| B-07 | High | thermal | Filament specific heat on mixed bases | REVIEWER-RUN / FROM-MEMORY |
| B-08 | High | solid FEA | Bracket wall thickness quantised to whole voxels | REPRODUCED |
| B-09 | Med | solid FEA | Stress read at element centroid / mean | CODE-CONFIRMED |
| B-10 | Med | solid FEA | Hex8 fully integrated, shear-locks in bending | REVIEWER-RUN |
| B-11 | Med | beam FEA | 25 mm gyration radius for every lumped mass incl. bed | CODE-CONFIRMED |
| B-12 | Med | components | Hertz row factor 2× high for 4-row carriages | REASONED |
| B-13 | Low | components | Taco-bow model has a unit-ambiguous knob, no buckling threshold | REVIEWER-RUN |
| B-14 | Low | modal | Feet rocking penalty is not a series-compliance term | REVIEWER-RUN |
| B-15 | Med | ballbar | Stepper radial lag is ~90% a hard-coded 11 ms | CODE-CONFIRMED |
| B-16 | Med | G-code sim | "Ringing severity" is not a ringing model | CODE-CONFIRMED |
| B-17 | Low | shaper | Calibration is close to Klipper, not exact | REVIEWER-RUN / FROM-MEMORY |
| B-18 | Low | shaper | Synthetic PSD secondary-mode weighting | REASONED |
| B-19 | Low | shell | MITC4 membrane locks in in-plane bending | REVIEWER-RUN |
| B-20 | Low | shell | Drilling stabilisation ~6 orders below physical | REVIEWER-RUN |
| B-21 | Low | builder | Frame nodes at outer dimensions, spans one profile too long | REASONED |
| B-22 | Low | builder | Moving-mass bookkeeping (bedslinger, delta) | REASONED |
| C-01 | High | scoring | "Deflection under acceleration" is mostly gravity sag | REPRODUCED |
| C-02 | High | scoring | Composite score rises when the machine is derated | REPRODUCED |
| C-03 | Med | bed | Flatness and sag include rigid tilt and mount settlement | REVIEWER-RUN |
| C-04 | Med | schema | `internal_stiffness_n_per_mm` double-counted and remapped | CODE-CONFIRMED |
| C-05 | Med | BOM | BOM mass, FEA mass and two cost models not reconciled | REVIEWER-RUN |
| C-06 | Low | optimizer | Free upgrades and a weak time bonus | REVIEWER-RUN |
| C-07 | Low | Pareto | Gene encoding does not reproduce the baseline | REVIEWER-RUN |
| C-08 | Low | sensitivity | `est_hz_gain` extrapolates a log-derivative linearly | REASONED |
| C-09 | Low | scoring | Dynamics score, flow reference and mode labels | REVIEWER-RUN |
| D-01…09 | — | catalogs | Belt EA, CF shear modulus, rail masses, etc. | FROM-MEMORY |

---

## 3. Group A — local bugs

### A-01 — Y inertial load share omits the hotend mass
- **Where:** `src/printsim/fea/solver.py:406`
  `th_share = assembly.spec.toolhead.total_mass_g * 1e-3 / max(assembly.y_moving_mass_kg, 1e-6)`
- **Problem:** `toolhead.total_mass_g` excludes the hotend by definition (schema note near
  `schema/printer.py:2388`). `y_moving_mass_kg` is built from `toolhead_moving_mass_kg`
  (toolhead + hotend). `modal.py:1813` already uses the consistent
  `assembly.toolhead_moving_mass_kg`.
- **Effect:** `accel_deflection_y_um`, the combined envelope and Y-case stresses are under-reported.
  Recomputed with the consistent share: voron_style 22.77 → 24.42 µm (+7%), ratrig +10.5%,
  doomcube +27.5%, hyper_corexy +36.5%, monolith +42.5%, vzbot 1.77 → 2.71 µm (+53%).
- **Fix:** use `assembly.toolhead_moving_mass_kg`.
- **Related improvement:** the `(1 − th_share)` remainder goes entirely to the two Y-carriage
  nodes (`solver.py:410-414`, `modal.py:1815-1823`), so X-beam bow under its own inertia is
  missed. A consistent distributed beam load gives 25.9 µm on voron_style (+15% over as-coded).
  Cleanest form is `M @ a` over the moving DOFs.

### A-02 — Deltas get the CoreXY gantry load split
- **Where:** `src/printsim/fea/solver.py:404-416` and `src/printsim/fea/modal.py:1812-1825`.
- **Problem:** for any non-bedslinger the `(1 − th_share)` part of the Y load is put on "other
  gantry nodes" in Y. For a delta, `gantry_node_ids = [c_0, c_1, c_2, toolhead]`, so the tower
  carriages (which only move in Z) receive a Y inertial force. The X case puts everything at the CoG.
- **Evidence:** REPRODUCED. `examples/delta_racer.yaml` is 3-fold symmetric (static
  `kx = ky = 227.3 N/mm`, `x_moving = y_moving`) yet reports `accel_deflection` X/Y =
  **4.19 / 2.86 µm**. The domain reviewer also saw structural 169.0 / 164.6 Hz, shaper MZV vs ZV,
  recommended accel 28 600 vs 34 900 mm/s², and flsun_v400 6.68 / 5.30 µm.
- **Effect:** Y deflection ~20–32% low and a spurious X/Y anisotropy on deltas.
- **Fix:** for `architecture == "delta"` apply the full load at the CoG in both axes (the X pattern).
- **Test idea:** a symmetric delta must give equal X and Y accel deflection and equal structural
  frequencies within tolerance.

### A-03 — Delta workspace check tests the square bed corners
- **Where:** `src/printsim/kinematics/solvers.py:558-566`
  `for px in (0.0, bx): for py in (0.0, by): inverse_kinematics(spec, (px, py, 0.0))`
- **Problem:** a delta's build area is a circle. The square corners are outside it.
- **Evidence:** REPRODUCED. `uv run printsim evaluate examples/delta_racer.yaml` prints
  "Delta arm workspace geometry limit: arm length (310 mm) cannot reach full…". The corner needs
  331.8 mm horizontal reach against L = 310 (racer) and 383.9 against 350 (flsun). The inscribed
  circle is fully reachable with minimum arm angles of 23.2° and 21.8°.
- **Effect:** both shipped delta examples report `delta_workspace_valid = False`.
- **Fix:** test the circle of radius `min(bx, by) / 2` at `z = 0` and `z = bz`, and enforce a
  minimum arm angle (about 20°). `check_trajectory_bounds` should be cylindrical for delta.

### A-04 — No tower-height check for deltas
- **Where:** `src/printsim/kinematics/collision.py:43-47` (generic
  `overhead_z = 85.0 + 0.5 * ez` and rectangular X/Y clearances for every architecture).
- **Evidence:** REVIEWER-RUN. Carriages must reach 627 mm (racer) and 759 mm (flsun) above the
  bed. With the racer frame cut to 600 mm, `check_frame_clearance` still returns no collision and
  `clear_z = +155 mm`. Stock `clear_z` reads 335 mm where the true figure is about 50 mm.
- **Fix:** for delta, require `max(z_i)` + carriage height + effector height to fit the frame height.

### A-05 — Chamber temperature never reaches the hotend thermal model
- **Where:** `src/printsim/thermal/hotend.py:150` and `:386` (`t_amb = he.ambient_temp_c`).
- **Problem:** `frame.chamber_temp_c` ("Ambient/heated enclosure chamber air temperature",
  `schema/printer.py:1750`) is read by belts, brackets, components and RCSA but by nothing under
  `thermal/`. Every example hard-codes `ambient_temp_c: 25`.
- **Evidence:** REPRODUCED by grep (`grep -rn chamber_temp_c src/printsim` shows no hit in
  `thermal/`). Reviewer run: ABS cold side is 37.50 °C at both chamber 25 and chamber 65; setting
  hotend ambient to 65 by hand gives 75.3 °C (ABS) and 73.2 °C with `heat_creep_risk = True` (PLA).
- **Effect:** heat-creep risk, steady loss, heat-up time and Graetz inlet temperature are wrong
  for any enclosed build; cold side under-predicted by about the chamber rise.
- **Fix:** use `max(he.ambient_temp_c, frame.chamber_temp_c)`, or drop the duplicate field. Check
  how the hotend solver is called so the frame value is actually available there.

### A-06 — Arrhenius shift applied to `K` instead of `K·a_T^n`
- **Where:** `src/printsim/thermal/hotend.py:476-477`
  `k_melt_pa_sn = fil.power_law_k_pa_sn * a_t`
- **Problem:** time–temperature superposition gives `η(γ̇, T) = a_T · η_ref(a_T · γ̇)`. For a power
  law `η = K γ̇^(n−1)` this is `K(T) = K_ref · a_T^n`. The catalog activation energies
  (65–95 kJ/mol) are zero-shear magnitudes, so `a_T` is a true shift factor.
- **Evidence:** formula REASONED; numbers REVIEWER-RUN. PLA at +20 °C: code multiplies `K` by 0.46
  and pressure-limited flow by 8.6×; correct is 0.756 and 2.2×. At −20 °C: flow 0.10× vs 0.43×.
- **Effect:** none at the default temperature (`t_ref` equals print temperature, so `a_T = 1`).
  Pressure limit, push force and ΔP become over-sensitive whenever `target_temp_c` differs.
- **Fix:** `k_melt_pa_sn = fil.power_law_k_pa_sn * a_t ** fil.power_law_n`. The existing `test_t3`
  only checks monotonicity and will still pass.

### A-07 — Eigen-solve is not deterministic
- **Where:** `src/printsim/fea/modal.py:1245-1252`
  `eigsh(k_csc, k=k_scan, M=m_csc, sigma=0.0, which="LM", OPinv=op)` with no `v0`
  (path taken when `n_free > 250`).
- **Evidence:** REPRODUCED. Three consecutive `evaluate_printer` calls on
  `corexy_voron_style.yaml` produced three different hashes of the modal result; `f0` agreed only
  to ~1e-11. The domain reviewer saw the nozzle component of mode 1 flip between +0.4594 and
  −0.4594 (scipy 1.17.1). Scalars are stable to rounding.
- **Effect:** `nodal_vectors` / `nodal_rotations` signs flip run to run (animation phase, snapshot
  tests, caching, API diffs); degenerate pairs can rotate arbitrarily. Violates the determinism
  rule in `.agents/rules/physics-and-fea.md`.
- **Fix:** pass a fixed `v0` (for example `ones / sqrt(n)`) and apply a sign convention (largest
  component, or nozzle component, positive).

### A-08 — Shaper frequency search capped at 1.35× the resonance
- **Where:** `src/printsim/fea/klipper_shaper.py:708-710`
  `lo_f = min(max(12.0, center_f * 0.7), 210.0)`
  `hi_f = min(215.0, max(lo_f + 4.0, center_f * 1.35))`
- **Problem:** Klipper scans every shaper over its whole frequency range. Multi-hump shapers are
  optimal well above the resonance (about 1.55× for 2HUMP_EI, about 1.85× for 3HUMP_EI). The cap
  pins both at the upper bound.
- **Evidence:** numbers REVIEWER-RUN against a from-scratch Klipper `fit_shaper` on the repo's own
  synthetic PSD (80 Hz peak):

  | Shaper | Repo freq (Hz) | Repo max accel | Reference freq (Hz) | Reference max accel | Smoothing repo vs ref |
  |---|---|---|---|---|---|
  | 2HUMP_EI | 107.1 | 12 800 | 123.4 | 16 931 | 0.049 vs 0.039 |
  | 3HUMP_EI | 107.1 | 8 400 | 147.6 | 15 950 | 0.071 vs 0.041 |

  The same pinning at exactly 1.35× appears at 45, 52+66 and 120 Hz.
- **Effect:** max accel for those two shapers under-reported 25–47%, score inflated 1.3–1.9×. Hits
  specs with `input_shaper: 2hump_ei/3hump_ei` directly and biases AUTO against them.
- **Fix:** scan per shaper up to about 2.0× `center_f` for multi-hump shapers, or the full
  `[MIN_FREQ, MAX_FREQ]` range; the existing fine-grid refinement keeps cost down.

### A-09 — G-code parser drops arcs and extrude-only moves
- **Where:** `src/printsim/gcode/parser.py:81` (`elif cmd in ("G0", "G1"):`) and `:105`
  (`if float(np.linalg.norm(new_pos - pos)) > 1e-5:` discards zero-length XYZ moves).
- **Problem:** (a) an arc is neither tessellated nor used to update position, so the next `G1`
  becomes a straight chord from the arc start, and in `M83` mode the arc's E is lost.
  (b) Retract/unretract moves vanish, so the planner joins travel and print directly. In Klipper a
  non-kinematic move forces the junction speed to zero and the retract time is real.
- **Evidence:** REVIEWER-RUN. An 8-line test with one `G2` gave a chord (20,0)→(30,20) and parsed
  E = 1.3 mm against 1.9 mm true. The print-time effect is REASONED.
- **Fix:** tessellate `G2`/`G3` (I/J/R forms); keep E-only moves as zero-length entries with
  duration `|dE| / F` that force `junction_v = 0`. Lower priority: parse
  `SET_VELOCITY_LIMIT ACCEL=` and `M204 T`.

### A-10 — Pareto "safe max accel" exceeds the motor limit
- **Where:** `src/printsim/engine/pareto.py:548-551`
  `safe_accel = min(recommended_max_accel_mm_s2, max_safe_accel_motor_mm_s2 * 1.25)`;
  constraints `g1..g3` at `:555-557`.
- **Problem:**
  - `max_safe_accel_motor_mm_s2` is already the torque-balance limit including the safety factor
    (`kinematics/solvers.py:501-513`). Multiplying by 1.25 removes most of that margin.
  - Inconsistent with `evaluator._compute_motion_limits` (min of shaper, motor, belt at 1.0×) and
    with the simulator's step-skip threshold (1.15×).
  - `belt_max_accel_mm_s2` is ignored.
  - Overtension, shaft-bending and deflection are evaluated at the baseline's
    `kinematics.max_accel_mm_s2`, not at the advertised accel.
- **Evidence:** numbers REVIEWER-RUN (pymoo, pop 16, 6 gen, seed 42). "Sweet Spot (Knee)"
  advertises 36 610 mm/s² = 1.25 × motor limit 29 288. Re-evaluated at 36 610: torque margin 1.10,
  `step_skip_risk = True`, 1020 of 1020 extrusion moves flagged, composite 82.0 → 64.1.
  "Fastest Safe Accel" advertises 47 700 against a motor limit of 44 942.
- **Fix:** `safe_accel = min(shaper, motor, belt)` by reusing `_compute_motion_limits`, then re-run
  kinematics and static FEA at `accel_override = safe_accel` for the constraints and the
  deflection objective.

### A-11 — Torsion constant set from `iy_m4` on dual-belt-sync side rails
- **Where:** `src/printsim/geometry/builder.py:3825` and `:3832`
  `"j_m4": m_side.profile.iy_m4 * 2.5` (the adjacent line is `"iz_m4": … iz_m4 * 4.0`).
- **Evidence:** numbers REVIEWER-RUN on `vzbot_330_awd`: J becomes 1.7e-8 m⁴, 15.5× the catalogue
  1.1e-9 instead of 2.5×. `f1` is 111.6 Hz as coded and 101.2 Hz with `j_m4 * 2.5`; recommended
  accel unchanged.
- **Caveat:** may be a deliberate rod-holder clamp stiffening; the comment does not say. Confirm
  intent, then either fix to `m_side.profile.j_m4 * 2.5` or document the choice.

### A-12 — BOM sizes Z rails from the Y-rail profile
- **Where:** `src/printsim/geometry/bom.py:1010`
  `_add_rail_and_carriage("Z", "Z-Axis", y_rail_prof.id, z_rail_qty, z_rail_len, z_rail_qty)`;
  same pattern at `:1029` (bedslinger) and `:1033` (delta).
- **Problem:** the FEA builder uses `spec.bed.z_rail_profile` (`builder.py:4097-4106`).
- **Evidence:** numbers REVIEWER-RUN. Example spec (z_rail mgn12, y_rail mgn9): BOM lists
  4 × 128 g = 512 g of MGN9H; the FEA carries 4 × 325 g = 1.30 kg of MGN12. voron_trident_300:
  BOM 384 g MGN9 against MGN12 in the FEA. delta_racer: BOM 1.209 kg MGN12 towers against 0.749 kg
  MGN9 in the FEA. Masked wherever Y and Z profiles coincide.
- **Fix:** use `spec.bed.z_rail_profile` and skip the line when it is `"none"`.

### A-13 — FOC gain sizing uses twice the true per-motor load
- **Where:** `src/printsim/fea/mechatronics.py:585-600`
  (`m_car_per_mot = 0.5 * (m_cx + m_cy)` for 2WD, `0.25 * (m_cx + m_cy)` for AWD, the X/Y average
  for Cartesian), consumed at `:455-458` (`m_eff = m_ref_1 + m_carriage_per_motor`).
- **Problem:** with `a = x + y`, `b = x − y`, the carriage mass matrix in motor coordinates is
  `(1/4)[[mx+my, mx−my],[mx−my, mx+my]]`. Per-motor load is `m_cx / 2` for an X move and
  `m_cy / 2` for a Y move (diagonal entry `(m_cx + m_cy) / 4`); AWD is half again.
- **Evidence:** numbers REVIEWER-RUN:

  | Machine | `m_eff` used (kg) | True load X move | True load Y move | Achieved velocity-loop BW X / Y (spec 120 Hz) |
  |---|---|---|---|---|
  | Voron 2WD | 0.921 | 0.393 | 0.696 | 281 / 159 Hz |
  | Monolith AWD | 0.471 | 0.242 | 0.396 | 233 / 142 Hz |
  | Bedslinger | 0.999 | 0.585 | 1.413 | 205 / 85 Hz |

- **Effect:** FOC `k_em` and `b_em` are 1.3–2.3× too high relative to the labelled
  `servo_vel_bw_hz` / `servo_pos_bw_hz`; FOC coupled damping and frequency are optimistic. The
  closed-loop stepper `b_em` floor is affected the same way.
- **Fix:** `(m_cx + m_cy) / (2 * n_mot)` for CoreXY, or size gains per Cartesian axis from
  `Tᵀ M T`. Use per-axis mass for Cartesian.

### A-14 — FOC servo tracking lag counted twice in the ballbar
- **Where:** `src/printsim/gcode/ballbar.py:383-396`
  (`omega_track = 2π · servo_pos_bw_hz` → `alpha_trk` pre-filter) together with
  `src/printsim/fea/mechatronics.py:452-458`
  (`b_em = m_eff * omega_v; k_em = m_eff * omega_v * omega_p`).
- **Problem:** the plant already contains the position/velocity cascade lag, because `b_em` acts
  on absolute motor velocity with no velocity feed-forward, so following error
  `= b_em / k_em · v = v / ω_p`. The ballbar then filters the command with a second first-order
  lag of the same time constant.
- **Evidence:** numbers REVIEWER-RUN (voron_style, `foc_servo`, R = 50 mm, 120 mm/s): plant alone
  −3.47 µm; pre-filter alone −5.84 µm; simulated `mean_radial_lag` −13.29 µm (remainder is shaper
  smoothing). voron_monolith_awd: −4.39 / −5.84 / −10.31 µm.
- **Effect:** FOC mean radial lag over-stated roughly 2–3×; FOC-vs-stepper comparisons biased
  against FOC.
- **Fix:** for `foc_servo` skip the pre-filter, or keep it and add velocity feed-forward so the
  plant has no following error.

### A-15 — Nozzle gyration radius in sensitivities disagrees with the mass matrix
- **Where:** `src/printsim/fea/sensitivity.py:970`
  (`rot_i = node.lumped_mass_kg * (0.025**2)` for every node) vs `fea/beam3d.py:248` and
  `fea/modal.py:782` (`r_gyr = 0.007 if node.role == "nozzle" else 0.025`).
- **Evidence:** REVIEWER-RUN via the Fox–Kapoor sum rule (mass sensitivities should sum to
  `−f/2`): ratio 1.0003 on the bedslinger, 1.0001 on deltas, 1.0000 elsewhere.
- **Fix:** share one helper. Do this together with B-11.

### A-16 — Delta `reflected_rotor_mass_g` over-reported
- **Where:** `src/printsim/fea/mechatronics.py:600` (`reflected_per_axis = 1.5 * m_ref_1`).
- **Problem:** omits the arm Jacobian factor. The dynamics use `T_cart` correctly; only the
  reported number is wrong.
- **Evidence:** REVIEWER-RUN: the model's own `Tᵀ T` diagonal is 0.593 (delta_racer) and 0.500
  (flsun_v400) against the reported 1.5.
- **Fix:** report `mean(diag(T_cartᵀ T_cart)) * m_ref_1`.

### A-17 — Corner benchmark snaps the last trapezoid sample
- **Where:** `src/printsim/gcode/ballbar.py:108-120`
  (`t_arr = arange(0, t_tot + 0.5*dt, dt)` then `s_arr[-1] = dist_m`).
- **Evidence:** REVIEWER-RUN: last-step velocity 9.5 mm/s instead of 5 mm/s, a 2.25 µm position
  step at the corner. Small against 96 µm unshaped ringing, not negligible against the 4.6 µm
  shaped residual.
- **Fix:** sample both legs on one continuous time base (evaluate `s(t)` analytically across the
  junction).

### A-18 — Optimizer applies the collision penalty twice
- **Where:** `src/printsim/engine/optimizer.py:75-76` returns `composite − 30` although composite
  already carries the 30-point collision penalty (`evaluator.py:522`) and is clamped at 0; it
  also bypasses the selected objective.
- **Evidence:** REASONED.
- **Fix:** one collision penalty, or a hard constraint.

### A-19 — Compliance share divides by `abs(u_rel)` and clips negatives
- **Where:** `src/printsim/fea/sensitivity.py:742-744`, `:830-832`, `:874-876`
  (`max(0.0, -dux / max(abs(u_rel_x), 1e-12))`).
- **Problem:** the exact decomposition is `share_e = λᵀ K_e u / u_rel`, which sums to 1 with the
  sign of `u_rel`. With `abs()`, a negative `u_rel` flips every share negative and they all clip
  to zero.
- **Evidence:** REASONED for the failure. The reviewer confirmed `u_rel_x` and `u_rel_y` are
  positive on all 10 example specs, so it does not bite today.
- **Fix:** divide by the signed `u_rel`.

### A-20 — Raw accelerometer CSV path skips Klipper's normalisation
- **Where:** `src/printsim/fea/adxl_calibration.py:467-485` (Welch output used as-is) and
  `:443-446` (takes the single `accel_x` / `accel_y` column).
- **Problem:** FROM-MEMORY of Klipper's `shaper_calibrate.py` (fairly confident, not certain):
  `process_accelerometer_data` sums `psd_x + psd_y + psd_z`, and `normalize_to_frequencies()` does
  `psd /= (freq_bins + 0.1)` before fitting. Exported `resonances_*.csv` files are already
  normalised, so the PSD-CSV path is fine; the raw path is not equivalent. **Verify against the
  Klipper source before changing.**
- **Evidence:** size of effect REVIEWER-RUN on a synthetic 5–133 Hz sweep with modes at 48 and
  110 Hz: secondary/primary peak ratio 0.62 raw vs 0.27 normalised; recommendation shifts from
  EI @ 39.6 Hz / 2900 mm/s² to EI @ 42.6 Hz / 3400 mm/s².
- **Also:** the column pick accepts `accel_x` whenever its std > 1e-9, which is always true with
  noise, so a rotated sensor is silently analysed on the wrong axis.
- **Fix:** for raw input compute Welch on all three columns, sum, divide by `(f + 0.1)`, zero
  below `MIN_FREQ`.

### A-21 — Member self-weight lumped as two nodal forces
- **Where:** `src/printsim/fea/beam3d.py:201-204` (no fixed-end moments).
- **Evidence:** REVIEWER-RUN. Cantilever self-weight tip sag: 1 element 26.2 µm vs exact 19.7
  (+33%), 2 elements +8%, 4 elements +2%, 16 elements +0.1%. A single fixed–fixed element gives
  zero sag.
- **Effect:** small; gravity sag is dominated by lumped masses.
- **Fix:** add consistent moments `±wL²/12` in local axes.

### A-22 — Stress envelope includes pseudo-profile members
- **Where:** `src/printsim/fea/beam3d.py:269-318` / `src/printsim/fea/solver.py:526-551`.
- **Problem:** `toolhead_mount` / `nozzle_stem` use stiffness-calibrated pseudo-profiles, so
  `M·c/I` on them is not a stress, yet they enter `max_von_mises` / `min_safety_factor`. Also
  `tau_torsion = T * r_eff / J` uses the outer half-width with the open-section torsion constant
  (correct for closed tubes, about 5× conservative for T-slot).
- **Evidence:** REVIEWER-RUN: the min-SF member is `toolhead_mount` on the bedslinger (SF 159) and
  both deltas (SF 201 / 169).
- **Effect:** cosmetic at these stress levels (all < 6 MPa), but the "peak stress part" label can
  point at a pseudo-member.
- **Fix:** exclude pseudo-profile members from stress envelopes.

---

## 4. Group B — model errors (need recalibration)

### B-01 — Structure and drive combined as frequencies with different masses
- **Where:** `src/printsim/fea/modal.py:292`
  `f_eff = 1.0 / math.sqrt(1.0 / structural_hz**2 + 1.0 / drive_hz**2)` and the compliance shares
  at `:251-274` (`(1/f_i²) / Σ(1/f_j²)`).
- **Problem:** `1/f²` is a compliance proxy only if every `f` uses the same mass. `drive_hz` uses
  the moving mass (0.3–0.45 kg in X); `structural_hz` comes from locked-belt frame modes whose
  modal mass is many kg. Correct series form is
  `1/f_eff² = 4π² · m_moving · (c_struct + 1/k_drive)`, or better, put `k_drive` into `K` and
  solve the coupled eigenproblem. Two-DOF illustration: a 10 kg frame mode at 60 Hz carrying
  0.5 kg on a 60 Hz drive spring gives 54.4 / 67.9 Hz exactly; the Dunkerley form gives 42.4 Hz.
- **Evidence:** numbers REVIEWER-RUN. The toolhead was split from the X beam with the repo's own
  `k_drive` and re-solved with the repo's `K`, `M` (X axis):

  | Machine | Coupled dominant nozzle mode (Hz) | Code `f_eff` (Hz) | True frame share | Reported frame share |
  |---|---|---|---|---|
  | bedslinger | 70.0 | 57.6 | — | — |
  | voron_style | 79.6 (plus frame mode 40.7) | 49.7 | 33% | 69% |
  | ratrig | 132.0 | 72.6 | 22% | 67% |
  | monolith | 175.2 / 227.5 | 141.8 | 18% | 54% |
  | vzbot | — | — | 13% | 49% |

  Y agrees within ~5–10% because the moving mass is large there.
- **Effect:** X effective frequency 18–45% low, recommended X accel (∝ f²) 1.5–3× low, and
  `primary_bottleneck = frame` reported where drive compliance dominates.
- **Caveat — read before fixing:** this is a documented approximation, and the current outputs
  (Voron X ≈ 50 Hz) sit near field values. `k_drive` and frame stiffness are probably calibrated
  around it (see D-01: belt EA is about 2.6× the cited Gates figure). **A formula swap alone will
  move every X number away from the touchstones.**
- **Fix:** compute shares from compliances (static `c_struct` from the existing unit-load solve,
  `1/k` for drive parts) and `f_eff` from `m(c_s + c_d)` or a coupled solve; then recalibrate.

### B-02 — "Structural frequency" is not a resonance
- **Where:** `src/printsim/fea/modal.py:1322-1353` (`_geometric_car_freq`) and `:1390-1433`
  (`_dunkerley_overhang_struct`).
- **Problem:** the value is `exp(Σ c_k ln f_k / Σ c_k)` over modes with positive carriage
  compliance, then reduced by `4π²[(m_cog + m_noz)·Δc_cog + m_noz·Δc_noz]`.
  (i) Modes with negative modal compliance are dropped rather than summed (monolith X: −0.052 of
  +0.292 µm/N). (ii) The result depends on how many modes are scanned. (iii) The FE model already
  contains the CoG and nozzle masses on flexible links, so adding overhang compliance again
  double-counts it.
- **Evidence:** REVIEWER-RUN. voron_style X: the racking mode at 40.8 Hz carries 1.36 of
  2.48 µm/N, yet reported structural is 59.8 Hz. Bedslinger X: structural 87.9 Hz is below the
  lowest X mode in the model (91.9 Hz). Mode-count sensitivity: voron_trident X 47.1 Hz (16 modes)
  vs 70.5 Hz (56, default); doomcube X 202.8 (default) vs 217.0 (100 modes), accel 76 600 vs
  81 300.
- **Effect:** `structural_freq_hz` can sit 5–50% from the mode holding most of the compliance;
  ~5–7% truncation drift. An explicit calibrated heuristic — an inconsistency, not a coding slip.
- **Fix:** use nozzle receptance directly (the nozzle node exists), sum signed residues, and feed
  all `(f_k, R_k)` into the PSD synthesis instead of collapsing to one frequency.

### B-03 — Delta motor torque and speed model is strongly optimistic
- **Where:** `src/printsim/kinematics/solvers.py:476-485`
  (`share = 0.65`, `alpha_per_a = 1.0 / r_p`, `f_eff = m_x * a_max_m_s2 + f_fric`) and `:447-448`
  (`motor_lin_v = v_max_m_s * (1.4142 if corexy else 1.0)`).
- **Problem:** carriage motion follows the Jacobian `dz_i/dx = −(x − x_i)/h_i` with
  `h_i = sqrt(L² − d_i²)`. At centre the carriage/effector speed ratio is `R/h` and the peak
  single-carriage force is `(2/3)(h/R)·m·a` — about 1.06·m·a for the shipped geometry, not 0.65.
  Near the perimeter the speed ratio is `d/h`, well above 1. Gravity hold and carriage mass
  (`gantry.carriage_mass_g`) are missing altogether.
- **Evidence:** numbers REVIEWER-RUN (analytic Jacobian matches finite-difference IK to 2.5e-10).
  delta_racer: speed ratio 0.63 at centre, 1.06 at half radius, 2.34 at perimeter; peak motor
  torque at perimeter 0.068 N·m with gravity (0.061 without) against code `t_req` 0.0373; static
  hold alone 0.013–0.023 N·m. flsun_v400: ratio up to 2.50; 0.0685 N·m against code 0.0332.
- **Effect:** required torque under-stated about 1.8–2×, peak motor rpm about 2.3–2.5× (code
  900 rpm, real about 2100 rpm at the perimeter). The reported 6.0× margin and 69 396 mm/s²
  `motor_max_accel` are not credible.
- **Fix:** evaluate
  `τ_i = r·(J⁻ᵀ·m_e·(a + g) + m_c·(J·a + g)) + (J_rot/r)·J·a` over a perimeter/direction sweep,
  and use `max|J·u|` for `max_rpm` and the velocity limit.

### B-04 — Stepper back-EMF constant mixes peak and RMS
- **Where:** `src/printsim/schema/printer.py:2037` and `src/printsim/kinematics/solvers.py:248`
  (`k_e = holding_torque_nm / rated_current_a`), used with RMS run current at
  `printer.py:2038-2045` and `solvers.py:269`.
- **Problem:** holding torque is two-phases-on, so `T_hold = √2·K_t·I_rated` and the per-phase EMF
  constant is `T_hold / (√2·I_rated)`. The code's EMF is √2 too high and its `I·R` and
  `ω_e·L·I` terms are √2 too low because they use RMS against the peak bus voltage. The smooth
  `eta` formula also sheds torque well below the corner.
- **Evidence:** numbers REVIEWER-RUN against a two-phase phasor model
  (`|I| ≤ √2·I_run`, `|V| ≤ V_bus`). Code/physical torque: 1.00 at 10 rpm, 0.87 at 500, 0.79 at
  700, 0.67 at 1000, 0.61 at 1500, 0.74 at 4000 rpm. Zero-speed limit and `1/ω` asymptote are
  correct. Code corner 814 rpm; consistent-convention corner about 950 rpm.
- **Caveat:** this compares against the ideal pull-out limit. Open-loop mid-band losses are real,
  so part of the gap may be intended conservatism (the 0.72 utilisation already covers some).
- **Effect:** 20–40% torque under-estimate from mid speed upward; feeds `motor_max_accel` and
  torque margin.
- **Fix:** `Kt_phase = T_hold / (√2·I_rated)` with peak current `√2·I_run`, and a piecewise curve
  (constant torque to the corner, then voltage-limited).

### B-05 — Motor max velocity is a hard corner-speed cut-off on an axial basis
- **Where:** `src/printsim/kinematics/solvers.py:266-273`
  (`omega_m = v_m_s / r_p`; `if v_req > v_bus_eff: return False`).
- **Problem:** (a) crossing `V_bus` is where torque starts falling as `1/ω`, not where motion
  stops; the torque-reserve branch never binds, so the result is just the corner speed.
  (b) `ω` uses axial belt speed, while the reserve torque in the same function and `max_rpm` at
  `:447` use the CoreXY diagonal (√2).
- **Evidence:** numbers REVIEWER-RUN. Default LDO-2504 at 24 V: solver returns 542.6 mm/s, exactly
  the gate speed, where available/sf torque is still 0.104 N·m against 0.0144 needed.
  corexy_voron_style: configured 450 mm/s means 636 mm/s diagonal belt speed against a reported
  ceiling of 520, yet the evaluator warning (`max_velocity > 1.05 × ceiling`) does not fire.
- **Fix:** drop the hard gate and solve `T_avail(ω)/sf ≥ T_required` with the correct diagonal
  factor; or rename the field to a corner speed and divide by √2 for CoreXY.

### B-06 — CoreXY motor-limited acceleration is conservative and mislabelled
- **Where:** `src/printsim/kinematics/solvers.py:465-475` and `:506-512`.
- **Problem:** with `A = x + y`, `B = x − y`, `F_A = (F_x + F_y)/2`. The 45° effective mass is
  `(m_x + m_y)/2`, and the true worst direction gives
  `a_max = T / (sqrt(M_x² + M_y²)·r/2)` with `M = m + 2J/r²`. The code applies the diagonal share
  to `m_y` alone. The fields named `motor_max_accel_x` / `_y` are diagonal-share values, not axis
  capabilities.
- **Evidence:** numbers REVIEWER-RUN (voron_style, `m_x` 0.451, `m_y` 1.056). Exact: pure X
  50 102; pure Y 28 315; 45° 24 912; worst 24 077 mm/s² at 59°. Code: x 34 497; y 19 496; min
  19 496. Code/exact worst = 0.81; the x and y fields are 0.69 of the true axis values.
- **Caveat:** `test_m1` pins the current `t_req_y` expression — a deliberate conservative choice,
  19% low (bounded by √2).
- **Fix:** use the exact worst-direction expression and report true per-axis values; update
  `test_m1` deliberately.

### B-07 — Filament specific heat is on mixed bases
- **Where:** `src/printsim/thermal/filaments.py:68-71` (PLA k = 0.13, cp = 1800), `:88-90`
  (PETG 0.20 / 1450), `:108-110` (ABS 0.18 / 1400), ASA and PA12-CF similar; consumed at
  `src/printsim/thermal/hotend.py:462-470`.
- **Problem:** the Graetz formula is right but linear in `α = k/(ρ·cp)`. PLA uses a melt-averaged
  cp; ABS, PETG and ASA use room-temperature solid cp (their melt-range average is roughly
  1800–2100 J/kg·K — FROM-MEMORY). The `C_core = 3.1` calibration and the tests cover PLA only.
- **Evidence:** REVIEWER-RUN on the default hotend (16 mm melt zone): PLA 21.7, PETG 42.6,
  ABS 53.5, ASA 53.6, PA12-CF 88.7 mm³/s (the last capped by pressure at 51.9). α is 5.8e-8 for
  PLA against 1.24e-7 for ABS. Reference ranking (FROM-MEMORY, medium-high confidence): on the
  same hotend ABS ≈ PLA, PETG < PLA, PA-CF much lower.
- **Effect:** `max_volumetric_flow_mm3_s`, flow-limited speed and the thermal score are about
  2–2.5× too high for every non-PLA filament.
- **Fix:** put cp on one temperature-averaged basis and calibrate `C_core` per filament, or add a
  per-filament melt-flow factor; add a cross-filament ranking test.

### B-08 — Bracket wall thickness is quantised to whole voxels
- **Where:** `src/printsim/fea/bracket_designer.py:159-199`
  (`w_eff = max(w, hx*1.05, hy*1.05)`; `grid |= Y <= w_eff` centre-in test; same pattern for
  ribs/flanges at `:217`, `:231-247`, `:283`) and `src/printsim/fea/superelement.py:643-665`
  (`h_mm = clip(h_vol, max(2.0, min_dim/6), max(3.0, min_dim/2.5))`).
- **Problem:** a wall of thickness `w` is represented by however many voxel centres fall inside
  it, so thickness snaps to multiples of the pitch. Bending stiffness goes as `t³`, so the
  response is a staircase in the design variable.
- **Evidence:** REPRODUCED (`corner_gusset`, `aluminum_6061_t6`, no rib/hole/fillet, Hex8):

  | Wall (mm) | Stiffness (N/mm) | Volume (cm³) | Max von Mises (MPa) |
  |---|---|---|---|
  | 4.9 | 484 | 6.08 | 17.22 |
  | 5.0 | 1613 | 8.88 | 9.54 |
  | 6.0 | 1613 | 8.88 | 9.54 |
  | 6.9 | 1613 | 8.88 | 9.54 |
  | 7.0 | 3930 | 11.52 | 5.92 |

  Reviewer run: Tet10 gives 801 and 854 N/mm at 4.9 / 5.0 mm. Superelement at its default
  (`target_elements = 380`, gusset 40/20/6/5): pitch 4.44 mm, so the 6 mm wall is one element;
  mass 20.27 g vs ~25.1 g analytic (−19%); port `k_x` 2.01e6 N/m vs 2.75e6 at 1 mm pitch (−27%),
  `k_z` 1.54e6 vs 3.14e6 (−51%). Convergence is non-monotonic (2.01, 3.54, 2.65, 3.00, 2.75 e6
  for pitch 4.44, 3.33, 2.0, 1.33, 1.0 mm). `target_elements` above ~3000 has no effect because
  of the `min_dim/6` floor.
- **Effect:** `design_and_evaluate_bracket` (Hex8, meant for agent optimisation loops) gives zero
  gradient then 2–3× jumps in stiffness, mass and safety factor. The default superelement feeds
  `/api/brackets/inject-evaluate` (`api/server.py:811-818`), so injected bracket stiffness is
  roughly 25–50% off depending on direction.
- **Fix:** align grid planes with feature planes (per-axis `hx = w / round(w / h_target)`, or a
  rectilinear grid with a few `Ke` size classes), or scale E and ρ of cut voxels by volume
  fraction; loosen the `min_dim/6` floor; for optimisation loops default to Tet10 or snap
  parameters and report the realised thickness.

### B-09 — Stress recovered only at element centroid / mean
- **Where:** `src/printsim/fea/solid_fem.py:437-448` (`DB = D @ B_center`);
  `src/printsim/fea/tet_fem.py:372-388` (`sigma_elem = np.mean(sigma_qp, axis=2)`).
- **Problem:** bending stress peaks on the surface; a centroid value under-reads by at least
  `(1 − 1/n_through_thickness)`, plus half an element back from the root. For Tet10 the stress is
  linear in the element and the mean throws away the surface value.
- **Evidence:** numbers REVIEWER-RUN. `cantilever_bar` preset (20×4×4): max von Mises 6.92 MPa vs
  `6PL/bh²` = 10.42 MPa (ratio 0.665); 10×2×2: 0.386; 40×8×8: 0.81. Corner gusset w = 6: Hex8
  default 9.54 MPa, SF 28.9; at 0.667 mm 16.9 MPa, SF 16.3; nominal beam root stress 17.0 MPa;
  Tet10 default 12.6 MPa, SF 21.9.
- **Effect:** `safety_factor` and `failure_mode` optimistic by about 1.5–1.8× for Hex8 at default
  resolution and about 1.35× for Tet10; `max_z_tensile_mpa` (FDM delamination check) has the same
  bias.
- **Fix:** evaluate `D·B` at the 8 Gauss points or element corners and take the max, or use
  nodal-averaged / SPR recovery; for Tet10 use vertex or quadrature-point values without
  averaging. A clamped re-entrant corner is singular, so a fine-mesh max diverges — a
  nodal-averaged value is the usual compromise.

### B-10 — Hex8 is fully integrated with no shear-locking remedy
- **Where:** `src/printsim/fea/solid_fem.py:180-195`.
- **Evidence:** REVIEWER-RUN. Tip deflection FE/Timoshenko, 60×12×12 bar: 5×1×1 mesh = 0.625,
  10×2×2 = 0.852, 20×4×4 = 0.951, 40×8×8 = 0.982. 40×20×6 mm wall in bending: 1 element through
  thickness 0.664, 2 elements 0.72–0.83, 3 elements 0.88, 6 elements 0.95. The element itself is
  correct (patch test exact, 6 rigid-body modes, mass exact).
- **Effect:** voxelised STL parts and the default superelement have 1–2 elements through walls,
  so 20–50% over-stiff. In the superelement this partly cancels the under-thickness error of
  B-08, which is why its results look plausible but do not converge.
- **Fix:** incompatible modes (Wilson–Taylor) or EAS; `Ke` is computed once per brick size so cost
  is nil. **Fix B-08 and B-10 together** — fixing one alone removes the accidental cancellation.

### B-11 — 25 mm gyration radius for every lumped mass, including the bed
- **Where:** `src/printsim/fea/beam3d.py:248`
  (`r_gyr = 0.007 if node.role == "nozzle" else 0.025`), same constant at
  `src/printsim/fea/modal.py:782`.
- **Problem:** the whole bed mass sits on one node; a 300 mm plate has `r_gyr` ≈ 87 mm (tilt) and
  ≈ 122 mm (yaw), so its rotary inertia is 12–24× too small.
- **Evidence:** numbers REVIEWER-RUN (what-if with `m·b²/12`, MAC-matched). Bed-rotation modes
  move: vzbot 129.4 → 114.0 Hz (−12%), bedslinger 150.4 → 147.5, doomcube 220.5 → 213.5;
  frame/gantry modes unchanged. Through secondary-mode selection it also moved recommended X
  accel on doomcube (76 600 → 61 400) and hyper_corexy (53 500 → 42 900).
- **Fix:** per-node inertia tensor (plate formula for the bed, box estimate for toolhead and
  motors). Keep `sensitivity.py` in step (A-15).

### B-12 — Hertz row factor is 2× high for 4-row carriages
- **Where:** `src/printsim/fea/components.py:121-137` (`row_factor = 0.5 * n_rows`).
- **Problem:** each ball has two contacts in series (`k_n/2`). For 2-row Gothic arch the second
  loaded diagonal doubles it back, so `n_b·k_n` is right. For 4-row circular-arc (HGH) there is
  one diagonal per ball: `4·n_b·(k_n/2)·sin²45° = n_b·k_n`, not `2·n_b·k_n`. `k_skirt` inherits
  the same factor. LMxUU bushings reuse the path with groove conformity `f = 0.525`, but a ball
  on a convex shaft has `R_e` ≈ 0.73 mm vs 3.6 mm assumed (another ~1.7×).
- **Evidence:** REASONED. Computed outputs: hgh15 Z1 = 572 (C) / 786 (H) N/µm, Z0 = 413 / 568;
  recalled catalog range for size-15 four-row guides is roughly 200–300 N/µm (FROM-MEMORY).
  mgn12H Z1 = 231 N/µm.
- **Effect:** HGH15 and rod-bushing carriage stiffness about 2× and 3× high. No example YAML uses
  them.
- **Fix:** row factor `n_rows / 4` for 2-point-contact rows; separate contact geometry for
  bushings. Check a HIWIN rigidity table first.

### B-13 — Taco-bow model: unit-ambiguous knob, no buckling threshold
- **Where:** `src/printsim/fea/components.py:400-425` and `BED_MOUNT_CATALOG`
  `k_restraint_thermal`.
- **Problem:** `arch_fraction = k_restraint / (k_restraint + k_plate_taco)` compares an
  unsuffixed catalog number (8, 22, 8000) with an out-of-plane plate stiffness in N/m
  (2e5–1.4e7). 8000 N/m for a "rigid 4-pt" bolted mount is not physical, so it is a calibration
  constant, and the field breaks the unit-suffix convention. The elastica amplitude is applied to
  all excess expansion, with no Euler critical strain.
- **Evidence:** numbers REVIEWER-RUN, physics REASONED. `rigid_4pt`, 12 mm plate, 60 °C: 48 µm
  "buckling" reported although excess expansion is 44 µm vs 583 µm critical shortening; 8 mm at
  110 °C: 239 µm with 321 vs 259 µm.
- **Fix:** suffix the field; compute restraint force from series(mount lateral k, plate EA/L);
  below `P_cr` use eccentric-load bow with `1/(1 − P/P_cr)`; above it use only the excess beyond
  `ε_cr`.

### B-14 — Feet rocking penalty is not a series-compliance term
- **Where:** `src/printsim/fea/modal.py:1484-1490`
  `pen = 0.052 * (m/M)^1.5 * m (2π·45)² / k_feet * damp_scale`, then
  `struct = struct_int / sqrt(1 + pen)`.
- **Problem:** a series term would scale with `f_int²`, not a fixed 45 Hz; damping is used to
  scale a frequency. `k_feet`, `z_cog` and the `K_θ` lever arms are correct.
- **Evidence:** REVIEWER-RUN. voron_style `rubber_feet`: code lowers structural X 59.8 → 56.6 Hz;
  the FE mode list (which already contains the feet springs) replaces the clamped 40.8 Hz racking
  mode with a 22.9 Hz rocking mode and internal modes at 73.9 / 98.4 Hz. ratrig: 57.9 Hz racking
  mode disappears, internal X modes at 115.8 / 146.8 Hz, code reports 82.4.
- **Effect:** sign and size of the feet effect are heuristic (−5% to −28% across feet types).
  Only active when `feet_type != rigid_bolted`; no example uses it.
- **Fix:** take frequencies from the feet-sprung FE model, weighting modes by nozzle-to-bed
  receptance.

### B-15 — Stepper ballbar "radial lag" is ~90% a hard-coded 11 ms
- **Where:** `src/printsim/gcode/ballbar.py:388-390`
  (`tau_eff = tau_e + 0.011; omega_track = 1.0 / tau_eff`); `:386` for closed-loop
  (`max(pos_bw * 1.25, 22.0)` Hz).
- **Problem:** an open-loop stepper has no servo following error. The commanded field is followed
  to within the load angle, already in the plant as `k_em`. A spring-driven mass on a circle
  below resonance grows the radius by `R(ω/ω_n)²`; it does not shrink. L/R (1.25 ms) is
  defensible as voltage-mode current lag; the extra 11 ms has no Klipper counterpart.
- **Evidence:** numbers REVIEWER-RUN (voron, open loop, 120 mm/s): plant alone +1.64 µm
  (outward); pre-filter alone (τ = 12.2 ms) −21.60 µm; simulated −24.22 µm. Bedslinger:
  +1.54 / −23.39 / −24.23 µm.
- **Caveat:** `tests/test_phase2_mechatronics_ballbar.py` asserts `M_SE < 0` scaling with v², and
  the code comment says the filter exists to produce that. Deliberate calibration fudge. Shaper
  smoothing alone (about −3.6 µm here) may satisfy the sign assertion — not tested.
- **Effect:** `mean_radial_lag_um` for steppers is essentially `−v²τ²/(2R)` with τ ≈ 12 ms,
  independent of the machine design. A design agent cannot improve it by changing hardware.
- **Fix:** drop the 11 ms constant (keep L/R at most), let shaper smoothing plus plant compliance
  produce the radius error, and document any remaining empirical term as a named, cited
  parameter.

### B-16 — Simulator "ringing severity" is not a ringing model
- **Where:** `src/printsim/gcode/simulator.py:279-287`
  `ringing_sev = max(0.0, accels[i] / max(allow_a, 200.0) − 0.85) * (0.45 + 0.55 * corner_factor)`
  with `allow_a` from `modal.shaper_*.max_accel_mm_s2`.
- **Problem:** Klipper's shaper max accel is a smoothing limit (`TARGET_SMOOTHING = 0.12`), not a
  ringing limit. With a shaper on, exceeding it rounds corners; it does not ring more. The
  expression ignores resonance frequency, damping, the actual velocity change at the corner and
  move length, and uses the configured accel even on moves that never accelerate.
- **Evidence:** REASONED. Physical form: residual amplitude ≈ `(a/ω_n²) · V_shaper(f_n, ζ)`, up to
  2× depending on ramp timing; wavelength `v/f`; decay length `v/(ζ ω_n)`.
- **Effect:** `ringing_segments`, `max_ringing_severity` and 30 points of `print_quality_score`
  follow accel/limit only.
- **Fix:** per junction compute Δv, then amplitude = Δv-derived impulse response × the
  `estimate_shaper` residual at the modal frequency, in µm, and threshold on µm. Report smoothing
  as a separate defect class.

### B-17 — Shaper calibration is close to Klipper, not exact
- **Where:** `src/printsim/fea/klipper_shaper.py:587` (one test damping ratio, equal to the
  shaper's own), `:595-603` (takes minimum score directly), `:729-730` (AUTO = lowest score with
  vibrations ≤ 6%), `:22` (no ZVD), `:431-432` (floor of 100).
- **What differs (FROM-MEMORY of Klipper source — verify):** Klipper pessimises vibrations over
  test damping ratios `[0.075, 0.1, 0.15]`; picks the lowest-vibration frequency and then the
  best score within 1.1× of that; its cross-shaper rule is "20% better score, or 5% score and 10%
  smoothing"; it includes ZVD; and it returns max accel 0 when smoothing cannot be met.
- **Evidence:** REVIEWER-RUN. Across 28 one- and two-peak synthetic PSDs the recommended shaper
  name matched the reference 28/28. Repo frequency 1–2% higher, max accel 2–4% higher. ZV
  vibrations reported as 0.00–0.01% vs 0.4–0.6%.
- **Effect:** small on clean synthetic PSDs; on real noisy PSDs narrow shapers look better than
  Klipper would rate them.
- **Fix:** take the max over the three test damping ratios, port the two selection rules, add
  ZVD. Otherwise soften the "exact" wording in the docstring.

### B-18 — Synthetic PSD secondary-mode weighting
- **Where:** `src/printsim/fea/klipper_shaper.py:481-485`
  `stiffness_atten = min(1.0, (f_primary / mode_freq_hz) ** 2)`;
  `weight = participation² * 0.45 * stiffness_atten`.
- **Problem:** in a Klipper sweep the excitation acceleration is ∝ f, so resonant acceleration
  PSD scales as `p²Q²f²`, or as `f` after Klipper's `1/f` normalisation. Higher modes are
  amplified, not attenuated by `1/f²`. The `1/f²` weighting is reasonable for displacement, but
  this PSD is fed to the Klipper scorer and overlaid on measured ADXL data.
- **Evidence:** REASONED. For `f2/f1 = 1.5` the relative secondary height differs by about
  `1.5³/0.45` ≈ 7×. The 0.45 is a declared heuristic, so this may be partly intentional.
- **Effect:** AUTO on simulated PSDs leans toward ZV where a real machine would get MZV/EI;
  `spectral_correlation` is depressed for multi-mode machines.
- **Fix:** decide which quantity the synthetic PSD represents (acceleration divided by f, to
  match Klipper) and weight modes accordingly.

### B-19 — MITC4 membrane locks in in-plane bending
- **Where:** `src/printsim/fea/shell_mitc4.py:198-205`, `:226` (plain fully integrated bilinear Q4
  membrane).
- **Evidence:** REVIEWER-RUN. Strip cantilever in in-plane bending, one element across the width,
  FE/beam tip deflection: aspect 1 = 0.674, aspect 2 = 0.379, aspect 3.3 = 0.186; 4 across =
  0.968. The open-centre brace builder (`geometry/builder.py:2767-2900`) meshes each ring leg
  with one element across the flange. **System impact is small:** swapping in a selectively
  reduced-integration membrane changed the fundamental by −0.0% to −0.4% on four examples, with
  and without `top_brace = cnc_halo_ring`. Perimeter-bolted panels work mostly in shear, which Q4
  represents exactly.
- **Fix:** incompatible-mode (QM6) or EAS membrane. Low priority.

### B-20 — Drilling stabilisation ~6 orders below a physical drilling stiffness
- **Where:** `src/printsim/fea/shell_mitc4.py:13` (`_ALPHA_DRILL = 1e-6`), `:228`.
- **Evidence:** REVIEWER-RUN. 3 mm Al plate, 8×8 mesh, clamped edges: a moment about the normal
  at the centre node sees 0.038 N·m/rad; the same moment as an in-plane force couple sees
  1.27e6 N·m/rad. Strain definition and frame invariance are correct.
- **Caveat:** fine as a pure singularity guard, but then the normal component of fastener
  `k_rot` and any beam torsion landing on an interior shell node carries essentially nothing.
- **Fix:** raise alpha to 1e-3…1e-2 (1-point integrated), or document that drilling is
  non-structural.

### B-21 — Frame nodes placed at outer dimensions
- **Where:** `src/printsim/geometry/builder.py:3249-3251` and `:3331-3392` (nodes at 0 and
  w / d / h).
- **Problem:** every span is one profile width too long (460 vs 440 mm for 2020; 9% for 4040).
  Bending stiffness scales as `1/L³`, so about 13% softer for 2020 and up to about 27% for 4040,
  roughly 6% lower frame frequencies. X-beam and rail mass is also taken over the full 460 mm.
- **Evidence:** REASONED. Conservative, undocumented.

### B-22 — Moving-mass bookkeeping
- **Evidence:** REASONED throughout.
- `src/printsim/geometry/_builder_bedslinger.py:48`, `:68-72`, `:202`:
  `gantry.carriage_mass_g` is lumped on the X-gantry Z-carriage nodes and also counted in
  `y_moving` as the bed carriage; `bkt_delta_kg` is added to both X and Y moving mass.
- `src/printsim/geometry/_builder_delta.py:246`, `:265-266`: moving mass is toolhead + 0.09 kg.
  The three tower carriages are excluded although they contribute roughly
  `1.5 · m_car · tan²θ` of horizontal inertia and are what the motors drive (ties into B-03).
- `src/printsim/geometry/builder.py:4237` adds a 0.04 kg belt allowance to X only;
  `engine/evaluator.py:668` then reports "Toolhead Total 451 g" where the toolhead is 411 g.

---

## 5. Group C — scoring and optimizer

### C-01 — "Deflection under acceleration" is mostly static gravity sag
- **Where:** `src/printsim/engine/evaluator.py:154`
  `defl_pen = min(45.0, static_fea.combined_max_deflection_um * 0.65)`; diagnostic at `:165-169`
  ("High peak deflection … under acceleration"); `src/printsim/engine/pareto.py:553` and `:590`
  (third objective). Source value: `src/printsim/fea/solver.py:455-483`
  (`u_base = u_grav + u_belt`, 3D norm of nozzle-to-bed displacement).
- **Problem:** `combined_max_deflection_um = |gravity + belt preload + inertial|`. Gravity sag
  (Z-belt stretch under gantry weight, bed settling on its mounts) is a constant offset removed
  by homing, QGL or Z-offset. Only the inertial part, and the variation of sag across XY, affects
  a print.
- **Evidence:** REPRODUCED on `corexy_voron_style.yaml`: combined 74.8 µm with accel components
  12.8 (X) / 22.9 (Y) µm at 8000 mm/s²; at 4000 mm/s² combined is still **69.4 µm** while the
  accel components halve to 6.4 / 11.4 µm. Reviewer run: gravity 65.1 µm; with
  `accel_override_mm_s2 = 1` combined is 65.5 µm. The penalty saturates at 69.2 µm, so the
  structural score is 31.9 for every max accel from 4000 to 18 000. Swapping only `z_drive` to
  `leadscrew_triple`: gravity 2.1 µm, structural 31.9 → 58.6, with near-identical dynamic
  stiffness. Pareto front: "Stiffest Nozzle" has 45.3 µm (gravity 45.2, accel 3.2 / 2.7);
  "Fastest Safe Accel" (AWD, 4040) has 78.0 µm (gravity 77.9, accel 0.9 / 2.4) — the dynamically
  stiffest design is ranked worse than baseline. `springs_4pt` beds show 364–383 µm, all spring
  settle.
- **Effect:** about 43 of the 45 deflection-penalty points on Voron-2.4-style machines (about 11
  composite points); the third Pareto axis and its "Stiffest Nozzle" label are mis-ranked.
- **Caveat:** the solver comment documents the combined envelope as intentional; the error is
  using it as an "under acceleration" metric.
- **Fix:** score and optimise on the inertial-only envelope
  (`max(accel_deflection_x_um, accel_deflection_y_um)` or the existing `u_xy_comb`), ideally at
  the candidate's achievable accel, or on stiffness in N/µm. Report gravity sag separately; if
  scored, score its peak-to-valley across XY (48.9 to 65.1 µm here). Reword the diagnostic.
  Replace the hard `min(45, …)` clamp with a smooth saturating function.

### C-02 — Composite score rises when the machine is derated
- **Where:** `src/printsim/engine/evaluator.py:1004-1006` (benchmark speeds
  `min(220, 0.65*max_velocity)` and `min(320, 0.85*max_velocity)`, not capped by hotend flow);
  `:962` (static FEA at configured `max_accel_mm_s2`); `:221` (torque margin at configured
  accel/velocity); `src/printsim/engine/optimizer.py:237-240` (`accel` mutation exposes this to
  the optimiser).
- **Problem:** the default benchmark asks for 320 mm/s × 0.42 × 0.25 mm = 33.6 mm³/s on a
  23.8 mm³/s hotend, then penalises the resulting under-extrusion, which a slicer's volumetric
  cap would prevent. The same 40 mm cube always breaches the minimum layer time (2.4 s per layer
  against 5.25 s), costing a flat 15 points; faster machines breach it more.
- **Evidence:** REPRODUCED: composite **62.7 → 66.7** at `max_accel_mm_s2 = 4000` and
  **62.7 → 66.6** at `max_velocity_mm_s = 300`. Reviewer run: print quality 66.5 → 85.0 at 4000;
  composite 67.1 at 2000; 33.5 of the 100 print-quality points on the example are lost to these
  two effects (18.5 flow + 15 cooling).
- **Effect:** 4–5 composite points; `optimize_design` can "improve" a printer by derating it.
- **Fix:** cap benchmark feedrates at `thermal.max_volumetric_flow / (w·h)`; run the benchmark at
  the achievable accel and velocity from `_compute_motion_limits`; reward print time. Scale the
  cube with the build volume or drop the layer-time term. Remove `accel` from the mutation pool,
  or treat it as an output.
- **Test idea:** composite must be non-increasing when only `max_accel_mm_s2` or
  `max_velocity_mm_s` is lowered below the achievable limits.

### C-03 — Bed "flatness" and "gravity sag" include rigid tilt and mount settlement
- **Where:** `src/printsim/fea/bed_plate_fem.py:514`
  (`gravity_sag_um = max(abs(z_gravity_um))`) and `:520-523` (only the mean anchor Z is
  subtracted before `total_flatness_error_um`, `mesh_bed_leveling_required`,
  `first_layer_risk`).
- **Evidence:** REVIEWER-RUN. CoreXY `leadscrew_single`, 300×300×8, 2.8 kg: total flatness
  1171.6 µm ("critical"); after best-fit plane removal 22.1 µm; front–rear tilt 1164.7 µm.
  235×235×4: 498 µm vs 33 µm. `springs_4pt`: `gravity_sag_um` 435 µm, of which plate
  peak-to-valley is 6.6 µm (the rest is `m·g/k` spring compression). `silicone_4pt`: 149–176 µm
  vs 6.6 µm.
- **Caveat:** `tests/test_bed_plate_fem.py:126`, `:154-155` assert these hierarchies — a
  deliberate definition. But static tilt/settlement is removed by tramming or Z-offset, and the
  evaluator prints it as "gravity sag" and as flatness risk (`evaluator.py:469-473`).
- **Fix:** report three quantities — mount settlement, rigid tilt, and plate flatness after plane
  removal — and drive `first_layer_risk` and mesh-levelling from the last.

### C-04 — `toolhead.internal_stiffness_n_per_mm` double-counted and silently remapped
- **Where:** `src/printsim/geometry/builder.py:1385-1389`
  (`k_int_scaled = 350e3 * ((k_raw/350)**1.90 if k_raw < 350 else k_raw/350)`), `:1395-1398` and
  `:1458-1462` (series with carriage plate and hotend). Schema:
  `src/printsim/schema/printer.py:2361-2366` describes the field as "Overall internal toolhead
  structural stiffness from linear carriage to nozzle tip … body, carriage plate, and hotend
  clamp".
- **Problem:** the builder treats the field as stage 2 (housing) only and adds stage 1 and
  stage 3 in series again. Below 350 N/mm the entered value is replaced by a power-law value, so
  the `_n_per_mm` number is not what is used.
- **Evidence:** numbers REVIEWER-RUN. Entered 350 gives carriage-to-nozzle 92.9 N/mm (stages
  173 / 350 / 469); FEM nozzle stiffness `kx` 132, `ky` 68 N/mm. Entered 175 gives body
  93.8 N/mm. Entered 5000 gives combined only 123 N/mm; recommended accel moves 7500 → 7800. With
  `body_construction` set (all `TOOLHEAD_PRESETS`) the field is ignored entirely, which makes the
  optimiser's writes (`optimizer.py:203`, `pareto.py:503`) no-ops for preset-based specs.
- **Caveat:** the builder docstring (`builder.py:1300-1307`) describes the 3-stage intent, so
  this may be a stale schema description rather than a coding slip.
- **Fix:** either rename/redefine the field as body-only stiffness and document the sub-350
  calibration, or honour it as the overall value and back out stages 1 and 3.

### C-05 — BOM mass, FEA mass and the two cost models are not reconciled
- **Where:** `src/printsim/geometry/bom.py:1826` total vs
  `src/printsim/geometry/builder.py:4342` `total_mass_kg`; `engine/evaluator.py:518` uses
  `assembly.estimated_material_cost_usd`; `engine/pareto.py:552` uses `bom.total_cost_usd`.
- **Evidence:** REVIEWER-RUN on the example. FEA mass 13.80 kg (equals the mass-matrix sum and
  gravity load / 9.80665) vs BOM 15.32 kg. Bare extrusions agree exactly (3.289 kg). Differences:
  rails FEA 1.89 kg vs BOM 1.03 kg (FEA uses full frame lengths, plus A-12); panels FEA 3.22 kg
  vs BOM 4.04 kg (front door missing from FEA); bed FEA 2.80 kg (`bed.mass_g`) vs BOM 2.66 kg
  (computed from plate volume); hotend BOM `block_mass_g + 48` = 61 g vs spec `hotend.mass_g`
  46 g (`bom.py:1469`); fasteners, Z gears and idlers (about 1.4 kg) are BOM-only. **Cost:**
  efficiency score sees $303, Pareto sees $1096 for the same printer.
- **Also:** BOM cut lengths equal FEA centreline lengths (460 mm horizontals and 500 mm columns),
  which overlap at the corners — about 0.2 kg over-count and a cut list that is not buildable as
  printed.
- **Fix:** one mass ledger feeding both; derive `bed.mass_g` from the plate when not overridden;
  use `hotend.mass_g` in the BOM; use BOM cost in the efficiency score.

### C-06 — Optimiser free upgrades and a weak time bonus
- **Where:** `src/printsim/engine/optimizer.py:193-196`, `:180-188`;
  `src/printsim/engine/pareto.py:490-492`.
- **Problem:** the optimiser can change toolhead mass and hotend melt zone / heater power with no
  mass, length, stiffness or meaningful cost consequence. The "speed" time bonus
  `max(0, 40 − 0.25 t)` is 0 for t ≥ 160 s and only 9/100 at the example's 146 s.
- **Evidence:** REVIEWER-RUN: a 14-iteration balanced run picked
  `hotend.melt=24.0mm/copper_c110` as best.
- **Fix:** tie melt length and power to hotend preset swaps; rescale the time bonus.

### C-07 — Pareto encoding does not reproduce the baseline
- **Where:** `src/printsim/engine/pareto.py:243-315` and `:373-522`.
- **Evidence:** REVIEWER-RUN: decoding the baseline's own gene vector changes the design (run
  current 1.45 → 1.2 A, hotend 17.5 mm / 50 W → 16 mm / 60 W; flow 23.8 → 21.7, accel
  7500 → 7200). REASONED: the toolhead preset gene (`:475-481`) copies mass and offsets but not
  `body_construction`, `z_probe_type` or `envelope_mm`; SBX/PM with rounding treats unordered
  categories as ordinal and gives end categories half weight.
- **Fix:** add an "as-baseline" option per gene, apply full presets, use categorical operators.

### C-08 — `est_hz_gain` extrapolates a log-derivative linearly
- **Where:** `src/printsim/fea/sensitivity.py:1307`
  (`est_hz_gain = 0.5 * eff_stiff_hz`, described as "+50% stiffness").
- **Problem:** `stiffness_sens` is `df/d(ln E)`. For +50% the first-order step is
  `ln(1.5) = 0.405`, and the frozen-mode Rayleigh bound is `f·(sqrt(1 + 0.5 s) − 1)`. If a member
  holds all strain energy the code gives 0.25 f vs 0.2247 f (11% high).
- **Evidence:** REASONED. Labelled "potential", so arguably intentional.
- **Fix:** use the Rayleigh expression or `ln(1.5)`.

### C-09 — Smaller scoring inconsistencies
- Dynamics score uses the shaper accel uncapped by motor and belt limits
  (`engine/evaluator.py:217-220`); use `achievable_max_accel_mm_s2`.
- `flow_max_velocity_mm_s` assumes a 0.2 × 0.4 line; the benchmark uses 0.25 × 0.42.
- Low-frequency bed modes (12.6 Hz with `springs_4pt`) are reported as "frame natural frequency …
  outer frame racking" (`engine/evaluator.py:160-164`).
- Two stiffness definitions for brackets: `fea/bracket_designer.py:611` uses `F / max|u|`
  anywhere, `fea/solid_fem.py:469` uses `F / mean|u|` on loaded nodes (1612.9 vs 1623.4 N/mm on
  one case). Pick one, ideally displacement along the load direction.

---

## 6. Group D — catalog values to check against datasheets

All reference figures here are **FROM-MEMORY**. Look up the datasheet before changing a value.

| ID | Where | Current | Reference recalled | Confidence | Notes |
|---|---|---|---|---|---|
| D-01 | `schema/printer.py:134` (comment at `:116`) | GT2 6 mm EA = 50 kN (9 mm 75 kN, 12 mm 100 kN) | ~18.9 kN for 6 mm from the Gates "18k lbf/in" figure the file itself cites (18 000 × 4.448 × 6/25.4); community range 10–35 kN | Medium | Documented as a "dynamic secant modulus … empirically calibrated against ADXL logs" and pinned by `tests/test_industrial_belts.py:34`. Likely absorbing error elsewhere (see B-01). Internal inconsistency: gt2_9mm fibreglass (75 kN) ≈ AT3 10 mm steel-cord (80 kN). Sensitivity on the Voron example: belt cord 306 → 116 N/mm, total drive 141 → 80 N/mm, drive resonance × 0.75. **Do not change without recalibrating.** |
| D-02 | `schema/materials.py:151-165` (`carbon_fiber_tube`, also `carbon_fiber_high_tg`) | "0/90 Roll-Wrapped", E = 115 GPa, G = 18 GPa | 0/90 laminate in-plane G ≈ 4–5 GPa; 18 GPa needs substantial ±45 content, which would pull E to ~60–70 GPa | Medium | GJ of CF gantry beams over-stated about 3–4×; affects toolhead-offset twist modes. |
| D-03 | `schema/profiles.py:191-199` | MGN9 0.32, MGN15 0.92 kg/m | HIWIN: MGN7 0.22, MGN9 0.38, MGN12 0.65, MGN15 1.06 kg/m | Medium-high | MGN9 ~16% low, MGN15 ~13% low, MGN12 right. Implied fill ratios (0.70 / 0.86 / 0.78) are non-monotonic. Iy, Iz, J are still those of the full solid rectangle. |
| D-04 | `schema/profiles.py:107-110` | "Misumi HFS5-2020", A = 162 mm² (0.437 kg/m), I = 0.68 cm⁴ | Misumi ≈ 0.48 kg/m, 0.74 cm⁴ | Medium | Both 8–9% low; values look like an item/Bosch 20×20. 2040/3030/4040 within ~10%. |
| D-05 | `schema/profiles.py:213` | `al_flat_bar_25x5` J = w·t³/3 = 1041.7 mm⁴ | Roark (as `_solid_rect` in the same file) = 910.4 mm⁴ | High | 14% high. |
| D-06 | `thermal/filaments.py` | PETG, ABS, ASA latent heat 32–38 kJ/kg | Amorphous polymers: ≈ 0 | High | ~10% of enthalpy; only affects the heater-power limit, which rarely binds. |
| D-07 | `schema/printer.py:498` | E3D V6 structural stiffness 45 N/mm | ~300 N/mm from heatbreak-neck beam theory | Low | Flagged as uncertain. |
| D-08 | `schema/materials.py` (ACM/Dibond) | E = 22 GPa | Membrane ~14 GPa, flexural ~55 GPa for 3 mm panels | Low | Consider separate membrane and bending moduli. |
| D-09 | `schema/profiles.py` T-slot J | J/(Iy+Iz) = 0.06–0.08 | No authoritative figure recalled | — | Fixed by a prior audit and pinned by `test_t1`. Plausible, unverified. |

Also uncited: RCSA joint constants `k_t1 = 150` and `k_t2 = 170` N·m/rad per 2.1 GPa
(`fea/rcsa.py:398-401`), `I_mount = 4.5e-9`, `A_mount = 1.8e-4` (`:291-292`). On
voron_trident_300 the hard-coded J1 rotation contributes 30.2 of 57.0 µm/N while the user's
`internal_stiffness_n_per_mm = 350` contributes 6.2 µm/N; result is
`hotend_bending_stiffness` 17.5 N/mm, and the stock ABS Voron trips `whip_risk`
(ratio 1.393 > 1.35). The maths is right; the calibration needs a cited source.

---

## 7. Group E — minor inconsistencies and naming

- `thermal/hotend.py:416` lumps block loss at 0.032 W/K; the FVM at `:190-193` sums to 0.05 W/K
  (5 × 0.004 + 10 × 0.003).
- `thermal/hotend.py:443` `bed_target` is computed and never used; the loop always runs to 100 °C.
- `heatup_time_215c_s` reports time to `t_target` (255 °C for ABS), not 215 °C.
- `thermal/hotend.py:514` `max_linear_speed_02x04` divides by 0.20 × 0.42.
- `kinematics/collision.py:45-46`: comment says about 75 + 80 mm, code uses `85 + ez/2`.
- `kinematics/solvers.py:552-555`: `shaft_bending_risk` is an OR of two conditions, so the
  evaluator prints "risk at 36 N belt pretension (> 45 N)" for delta_racer.
- `available_torque_at_vmax_nm` is evaluated at 70% of max rpm.
- `fea/klipper_shaper.py:739` `rec_accel = 0.9 * f²` for NONE is an undocumented constant
  (equivalent to allowing `a/ω²` ≈ 23 µm); write it as `delta_allow * (2πf)²` with a named
  tolerance.
- `gcode/simulator.py:339` hard-codes PLA 1.24 g/cm³ for filament mass.
- `fea/sensitivity.py:145-169` "Dailey" guard projects the repeated-mode subspace onto the
  excitation vector — sensible, but not Dailey's sub-eigenproblem. Rename or implement properly.
- Bed mount inset is 0.375 in the FE and 0.38 in `components.py`.

---

## 8. Improvements (not bugs)

**Structural / modal**
- Put drive stiffness into the eigenproblem instead of combining frequencies afterwards (B-01).
- Evaluate over a small pose sweep. Scores come from the build-volume centre only; on the example
  recommended accel is 10 500 at `z_rel` 0.05, 7 400 at centre and 5 400 at `z_rel` 0.95
  (f1 44.7 / 41.6 / 37.3 Hz).
- Timoshenko shear deformation in the beam element. What-if with κ = 0.40 (extrusion) / 0.45
  (tube) / 5/6 (solid): raw modes drop 1–4.6%; structural Y drops 4.5–6% on doomcube, monolith,
  vzbot. `φ = 12EI/(κGAL²)`; keep Euler–Bernoulli as the `φ → 0` limit so benchmarks still pass.
- Member orientation: `beam_rotation_matrix` has no roll vector; the 0.99 threshold swaps Iy/Iz
  at 8.1° from vertical. Fine for square sections, limiting for 2040/3060.
- Smooth the accel recommendation. Secondary-mode filters (3 Hz exclusion, participation > 0.15,
  1% weight cut, 10–215 Hz window) create jumps: a 1.4% structural change moved doomcube X accel
  76 600 → 60 500, and delta_racer accel rose 28 600 → 35 300 when its frequency fell 1.5%. This
  hurts optimizer use.
- Track gravity-sag peak-to-valley over XY (16 µm on the example) and gantry tilt rather than the
  absolute value.
- Z-belt stiffness is a constant 0.18 N/µm per corner (`builder.py:3314-3315`) while leadscrews
  use EA/L; make it span-dependent.
- Static inertial loads go through locked sliders into the X-beam and Y-rails; in a real CoreXY
  they react through the belts at the motor and idler corners.
- Corner-joint knockdown (`builder.py:1271-1273`) applies to columns only and is skipped with
  bolted panels. Panel membrane stiffness is scaled by the shear-buckling factor for all membrane
  action (`builder.py:2217`). Bed mount rotation uses a scalar geometric mean of roll and pitch
  (`builder.py:1823`), which loses 3-point anisotropy.
- Bed workpiece point: `_bed_wp_disp` ignores bed rotation unless feet exist. Negligible at bed
  centre; matters off-centre.

**Continuum**
- Incompatible-mode / EAS Hex8 (B-10) and QM6 membrane (B-19).
- RBE3 vs RBE2 default: on the default gusset, port stiffness differs 1.6× in `k_x` and 6.3× in
  `k_z`. A face bolted flat to an extrusion is closer to RBE2 in the normal direction; report
  both as bounds or weight RBE3 by bolt locations.
- Apply bed gradient bow in the FE as thermal moments `M_T = D(1+ν)·α·ΔT/t` so mount restraint is
  captured (`bed_plate_fem.py:301-334`).
- FDM strength: z-layer strength is a hard-coded 0.55 × yield for every material
  (`solid_fem.py:451`, `tet_fem.py:391`), only σ_zz tension is checked, and print direction is
  assumed to be grid Z.
- `hz_per_100g` for members assumes stiffness scales linearly with added mass
  (`sensitivity.py:759-760`, `:779`); for section growth I scales faster than A.

**Kinematics / mechatronics**
- Position-dependent belt stiffness (`k = EA/L` with real span lengths) for a workspace-dependent
  drive resonance.
- Integrate a torque–speed curve over the accel ramp instead of one 70%-speed point; add
  pulley/idler inertia and belt mass.
- Friction is a constant 2.5 / 3.6 N; rail preload and seal drag from `fea/components` could feed
  it. Ballbar friction uses the X-rail value on both axes.
- FOC model has an ideal current loop. Add an R/L or current-loop-bandwidth pole and voltage
  saturation so `servo_vel_bw_hz` cannot be set arbitrarily high for free.
- The notch filter is cascaded on the position command; a TMC4671 biquad sits inside the
  velocity/torque loop where its 36.8° lag costs phase margin.

**Thermal / extrusion**
- Cross or Carreau–Yasuda viscosity with a WLF/Arrhenius shift. A pure power law has no Newtonian
  plateau; in the 2 mm chamber at 15 mm³/s it gives about 2 600 Pa·s for PLA, inflating the
  chamber term (about 25% of ΔP).
- Graetz-number-dependent Nu instead of constant Nu = 3.66 × `C_core = 3.1`; the limit is
  currently independent of bore and filament diameter.
- Bed heat-up uses one-sided h = 14 W/m²·K with no radiation or underside loss.

**G-code / shaper / ballbar**
- Lookahead: add Klipper's `minimum_cruise_ratio` / `max_accel_to_decel` pass; scale Z moves by
  `z_ratio`; use each move's own accel in the centripetal limit.
- Volumetric flow uses the segment average; peak flow at `v_peak` triggers under-extrusion on
  short accelerating moves.
- Ballbar `G_xy` is measured about the nominal centre; ISO 230-4 uses a least-squares or
  minimum-zone centre. Add squareness, scale-mismatch and microstep-periodic terms.
- `ringing_reduction_pct` compares different time windows, which flatters long shapers.
- Welch with Hann at 1 Hz biases the half-power damping estimate about 8% high for a 48 Hz,
  ζ = 0.05 peak; a 3-parameter SDOF `|H|²` fit removes the window bias.
- Benchmark G-code uses a rectangular bead area (`w·h`); slicers use
  `(w − h)·h + π·h²/4`, about 13% less at 0.42 × 0.25.

---

## 9. Verified correct (coverage)

Do not re-audit these unless the code has changed.

**Beam FEA / modal** — `beam3d.py`: all Euler–Bernoulli stiffness terms in both planes incl. the
θ_y sign convention; consistent mass (156/420, 22L, 54, 13L, 4L², 3L²); axial 1/3–1/6; torsional
inertia uses polar `Ip = Iy + Iz` while stiffness uses torsion constant J; rotation matrix
right-handed in every orientation. K symmetric, exactly 6 rigid-body zero modes for 5
orientations, rigid-translation mass = ρAL, M positive definite. 16-element cantilever in three
orientations: tip deflection and twist exact to 1e-12; bending f1/f2 within 0.0005%; torsion and
axial within 0.04%. `solver.py`: unit conversions, CoG moment signs, belt preload
self-equilibrated, rigid-body least-squares fit for bed deflection. `modal.py`: generalized
problem and shift-invert usage, mass normalisation, 2π factors, modal compliance residue, feet
strain-energy share, Fox–Kapoor constants (8π²f), PSD scaling, damping bookkeeping.
`components.py`: Hertz constant, 2-row Gothic-arch factor, roll/pitch/yaw integrals, Ernst
catenary EA, Harris bearing deflection and Stribeck 5F/Z, 48EI/L³ and 3EI/L³ shaft cases,
Timoshenko bimetal curvature, gradient bow L²αq/(8k), elastica amplitude.

**Continuum** — `solid_fem`: isotropic D, transversely isotropic compliance, Hex8 B matrix,
Jacobian, 2×2×2 weights (patch test to 7 digits), consistent mass. `tet_fem`: Voigt strains,
quadrature, consistent traction, unit conversions. `shell_mitc4`: plate D, 5/6 shear factor,
MITC tying rows, HRZ mass; 6 rigid-body modes on a distorted tilted element; membrane patch test
to 1e-15; simply supported plate w/Navier = 0.996 (8×8), 1.0007 (16×16), 0.998 at a/t = 1000 (no
shear locking); f1/exact 0.991–0.998. `bed_plate_fem`: mount/carrier stiffness conversions,
series combination, in-plane spring construction incl. `kelvin_cone_vee_flat`, thermal bow sign.
`topology`: modified SIMP, analytic/FD sensitivity ratio 1.000000, Sigmund filter, OC update and
bisection (volume fraction exact). `superelement`: Guyan K (5.5e-16), Craig–Bampton full basis
reproduces all eigenvalues (1.5e-15), rigid-translation mass. `rcsa`: Timoshenko state matrix,
LM-FBS with joint flexibility, free-free rigid-body limits, clamped-free first frequency
176.28 Hz vs 176.53 Hz. `sensitivity`: Fox–Kapoor and adjoint sum rules hold on all 10 examples.

**Shaper / mechatronics / G-code** — ZV, MZV, EI, 2HUMP_EI, 3HUMP_EI amplitudes and times
bit-identical to the Klipper formulas (reference reproduced FROM-MEMORY); ZVDD; `estimate_shaper`,
`get_shaper_smoothing`, `find_shaper_max_accel` match the reference to rounding. SMOOTH_FLATNESS
polynomial re-derived (kernel sums to 1, V(f_n) = 0). Biquad notch depth, Q and peak-lag.
Half-power damping estimator within 1% for ζ 0.02–0.2. Trapezoid distance integrates to 7e-15
over 2000 random cases. Junction speed matches `a·jd·sin(θ/2)/(1 − sin(θ/2))` at six angles (90°
gives exactly SCV); centripetal limit matches. Stepper `k_θ = p·T_hold`, `k_lin = k_θ/r²`,
reflected mass J/r². CoreXY `T_cart`, `K_cart`, reflected inertia 2× / 4×, DC gain = I. ZOH
discretisation, circle kinematics.

**Kinematics / thermal** — FK/IK round trips, 5000 random points each: CoreXY 6e-14 mm,
Cartesian 0, delta 1.7e-13 mm. Pitch radius, rpm conversions, `pole_pairs = steps/4`, copper loss
2I²R. Belt tight/slack split `T0 ± F/2`, pluck frequency `(1/2L)·sqrt(T/μ)`, GT2 linear mass.
Pressure drop: Newtonian limit equals Hagen–Poiseuille to machine precision (254 647.9 Pa both);
power-law pipe and cone formulas match numerical integration. Thermal: `R_hb = L/(kA)`
(1013.5 K/W), lumped cold-side temperature agrees with a node balance to 0.03 °C, heater energy
balance and Graetz expression reproduce by hand. Section formulas for square tube (Bredt), round
rod, solid rectangle (Roark). Metals in `materials.py` (6063-T5, 6061-T6, 7075, MIC-6, 4140, 304,
Ti-6Al-4V, C360, C110) within datasheet scatter.

**Engine / builder** — all unit conversions at evaluator/builder interfaces. Gravity
`g = 9.80665`, −Z, applied to all mass (gravity load / g equals the mass-matrix total,
13.797 kg). Rail and filling mass folded into member density without double counting. Composite
rail section parallel-axis formula (hand 2.15e-8 m⁴ vs 2.166e-8). Shear-panel equivalent diagonal
area, ACM sandwich rigidity, cantilever stem, leadscrew series stiffness, bed-mount second
moments. Damping ratios 0.037–0.046 across examples. Score arithmetic reproduces
31.9 / 72.8 / 80.0 / 66.5 / 78.7 / 62.7; composite weights sum to 1.00. Pareto domination,
feasibility-first sorting, crowding distance, zero-range guard, knee selection.

**Smoke run plausibility** (`examples/corexy_voron_style.yaml`): f1 41.6 Hz; shaper X/Y 50.4 Hz
MZV / 43.7 Hz ZV; recommended accel 7 400 mm/s²; moving mass 0.411 / 0.451 / 1.056 kg; nozzle
stiffness 132 / 68 / 156 N/mm; hotend flow 23.8 mm³/s; heat-up to 215 °C in 38 s; cold side
35.8 °C; panel drum mode 19.2 Hz (hand formula ~21 Hz); bed first mode 138 Hz. All plausible for
a Voron-2.4-class machine.

---

## 10. Tests that pin current behaviour

Changing these findings means changing the test on purpose, with a reason in the commit.

| Test | Pins | Related finding |
|---|---|---|
| `tests/test_audit_fixes_thermal_kinematics.py::test_m1_corexy_diagonal_45deg_torque_and_rotor_inertia` | Current `t_req_y` CoreXY diagonal-share expression | B-06 |
| `tests/test_industrial_belts.py:34` | GT2 EA values | D-01 |
| `tests/test_phase2_mechatronics_ballbar.py` | `M_SE < 0` scaling with v² | B-15, A-14 |
| `tests/test_bed_plate_fem.py:126`, `:154-155` | Flatness/sag hierarchies incl. tilt and settlement | C-03 |
| `tests/test_audit_fixes_thermal_kinematics.py::test_t1_open_tslot_extrusion_torsional_constants` | T-slot torsion constants | D-09 |
| `tests/test_audit_fixes_thermal_kinematics.py::test_t3_rabinowitsch_power_law_and_arrhenius_pressure_flow` | Viscosity shift monotonic only | A-06 (reviewer expects the fix to still pass; confirm by running) |
| Bracket taco tests | `> 100 µm` and orderings only | B-13 |
| `tests/test_calibration_touchstones.py`, `tests/test_community_realism.py` | Headline frequencies and accels vs field data | every Group B item |

Coverage gap worth closing: `test_fea`, `test_audit_fixes_core_fea` and
`test_modal_limiting_causes` (31 tests) cover none of A-01, A-02, A-07, B-01, B-02, B-11.

---

## 11. Suggested order of work

1. **Group A, physics-facing first:** A-01, A-02, A-03, A-04, A-05, A-06, A-07. Each is a few
   lines plus one test. They do not interact.
2. **Group A, tooling:** A-08, A-09, A-10, A-12, then A-13 + A-14 together (both touch the FOC
   path), then the Low items.
3. **Scoring (Group C):** C-01 and C-02 first — they change what the optimizer chases, so they
   should land before any recalibration. Then C-03, C-04, C-05.
4. **Group B as one recalibration project, in this order:** B-11 (inertia) → B-01 + B-02 (coupled
   drive, receptance-based frequency) → D-01 (belt EA, re-fit) → B-03…B-06 (motor model) → B-07
   (filament data). Re-run the touchstone tests after each step and record the deltas.
5. **Solid FEA:** B-08 + B-10 together, then B-09.
6. **Datasheet pass (Group D)** with real lookups.

Commit prefixes per `AGENTS.md`: `fix:` for Group A, `phys:` for Group B, `test:` for added
coverage. Keep commits small and one finding each where possible.

---

## 12. Reproduction snippets

Run from the repo root.

**C-02 and C-01 — derating raises the score; deflection is mostly gravity**
```python
# uv run python this_file.py
import yaml
from printsim.schema import PrinterSpec
from printsim.engine import evaluate_printer

spec = PrinterSpec(**yaml.safe_load(open("examples/corexy_voron_style.yaml")))

def run(**kw):
    s = spec.model_copy(deep=True)
    for k, v in kw.items():
        setattr(s.kinematics, k, v)
    r = evaluate_printer(s)
    sf = r.static_fea
    comp = getattr(r, "composite_score", None) or getattr(r.scores, "composite", None)
    print(kw, round(sf.combined_max_deflection_um, 1),
          round(sf.accel_deflection_x_um, 1), round(sf.accel_deflection_y_um, 1), comp)

run()                        # 74.8, 12.8, 22.9, composite 62.7
run(max_accel_mm_s2=4000)    # 69.4,  6.4, 11.4, composite 66.7
run(max_velocity_mm_s=300)   # 74.8, 12.8, 22.9, composite 66.6
```

**A-02 — delta X/Y anisotropy; A-07 — non-determinism**
```python
import yaml, hashlib, json
from printsim.schema import PrinterSpec
from printsim.engine import evaluate_printer

def load(n):
    return PrinterSpec(**yaml.safe_load(open(f"examples/{n}.yaml")))

sf = evaluate_printer(load("delta_racer")).static_fea
print(sf.accel_deflection_x_um, sf.accel_deflection_y_um)   # 4.19 vs 2.86

for _ in range(3):
    m = evaluate_printer(load("corexy_voron_style")).modal_fea
    d = m.to_dict() if hasattr(m, "to_dict") else m.model_dump()
    print(hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10])
# three different hashes
```

**A-03 — false delta workspace warning**
```bash
uv run printsim evaluate examples/delta_racer.yaml | grep -i workspace
```

**A-05 — chamber temperature unused by the thermal model**
```bash
grep -rn "chamber_temp_c" src/printsim/thermal/
```

**B-08 — bracket stiffness staircase**
```python
from printsim.fea.bracket_designer import design_and_evaluate_bracket

for w in [4.9, 5.0, 6.0, 6.9, 7.0]:
    r = design_and_evaluate_bracket(
        "corner_gusset", wall_thickness_mm=w, material_id="aluminum_6061_t6",
        engine="hex8", rib_thickness_mm=0.0, bolt_hole_mm=0.0, fillet_mm=0.0,
    )
    print(w, round(r["stiffness_n_per_mm"]), round(r["volume_cm3"], 3),
          round(r["max_von_mises_mpa"], 2))
# 484, 1613, 1613, 1613, 3930 N/mm
```

The domain reviewers' fuller repro scripts were written to a temporary scratch directory and are
not part of the repo; the snippets above cover the findings that were independently reproduced.
For the rest, the quoted expressions and numbers in each finding are enough to rebuild a check.
