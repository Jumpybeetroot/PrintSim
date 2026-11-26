# CAD Superelement, Multiphysics & Studio Viewport Engineering Playbook

This playbook captures the hard-won engineering pitfalls, mathematical invariants, and verification checklists discovered while building PrintSim's 3D CAD Craig–Bampton superelement pipeline, Hertzian linear-rail mechanics, and interactive 3D Studio visualizations. **Review this checklist before adding new CAD hardware assets, editing structural/modal solvers, or modifying Studio 3D deformation and sidebar CSS.**

---

## 1. CAD Asset Baking & GLB Visual Quality (`cad_assets/baker.py`)

### Pitfall 1A: Small Fasteners Starving Large Surfaces of Triangles
- **Symptom**: High-count M3 bolts/threads consume 80–85% of a fixed GLB triangle budget, turning smooth cylindrical hotend shells and heatsinks into faceted polygons.
- **Invariant**: Allocate GLB triangles **proportionally to surface area**, cap individual fastener solids (`<= 160` triangles per screw), target `~24,000` total triangles for hero hardware assemblies, and compute **angle-creased smooth vertex normals (`~34°` crease threshold)** so curved bodies shade smoothly while chamfers and bolt heads stay crisp.

### Pitfall 1B: Integration Models vs. Real Hardware Geometry
- **Symptom**: Manufacturer "integration" STEP files sometimes contain simplified clearance envelopes or omit cooling fins/heaters/shields.
- **Invariant**: Always visually inspect rendered GLBs against real hardware photos and manufacturer drawings (e.g., Luke's Lab Chube product specs, Okano Tricorn STEP, Microwaved Bowden `mbt_tricorn_v2.step`) before committing baked assets.

### Pitfall 1C: Multi-Mount RBE3 Nullspace Collapse
- **Symptom**: Condensing a braced assembly (e.g., top 4× M3 crown + lower Ti bottom brace lugs) with a single merged RBE3 patch creates kinematic rank deficiency or over-constrains separated bolt faces.
- **Invariant**: Condense **each physically distinct bolted patch** (`p0_top`, `lug_left`, `lug_right`) with its own independent RBE3 filter, then tie fixed patches via congruence transformation `Q`. Always verify rigid-body zero-mode eigenvalues stay `< 0.1 Hz` (`rigid_body_max_freq_hz < 0.05 Hz` typical).

### Pitfall 1D: Double-Drawing Shared Meshes in Coupled Assemblies
- **Symptom**: Loading both `toolheads/microwaved_bowden/visual.glb` (which contains `hotend_brace` + `mbt_shield`) AND `assemblies/microwaved_bowden__tricorn_long/visual.glb` (which also contains `hotend_brace`) or `hotends/tricorn_long/visual.glb` (`ti_shield_bucket`) causes Z-fighting or overlapping shields.
- **Invariant**: Keep `hotend_visual_glb_url` pointing to the standalone hotend GLB when the toolhead GLB already includes the shared brace/shield, verify relative part orientation (standalone `tricorn_*` spins 180° around CAD Z on `microwaved_bowden`, whereas coupled `/assemblies/` GLBs are already oriented), and toggle individual sub-mesh visibility (`hotend_brace`, `fastener_11..13`, `mbt_shield`, `fastener_19`, `ti_shield_bucket`) from resolved accessory state.

---

## 2. Superelement Port Transforms, Series Coupling & Metadata (`builder.py`)

### Pitfall 2A: Unshifted Port-1 Compliance (`dp_cad_m` vs. `dp_asm_m`)
- **Symptom**: Dropping a CAD superelement's raw `C_cad` (baked at CAD Port 1, e.g. `Z = +30.5 mm` *above* the MGN12 rail on Microwaved) directly onto the assembly's `toolhead_cog` node (`Z = -1.64 mm` *below* the rail) produces an **inverted-sign lateral-to-pitch cross-coupling (`C_rx_y`)**, making Mode 6 rock backwards with the nozzle moving less than the CoG!
- **Invariant**: Whenever `dp_cad_m != dp_asm_m` on a port-0-anchored superelement (`toolhead_mount`), rigidly translate compliance to the assembly node before series summation:
  $$T_{\text{shift}} = \begin{bmatrix} I_3 & -[\Delta\mathbf{p}_{\text{asm}} - \Delta\mathbf{p}_{\text{cad}}]_\times \\ 0 & I_3 \end{bmatrix}, \qquad C_{\text{cad,asm}} = T_{\text{shift}}\, C_{\text{cad}}\, T_{\text{shift}}^T$$
- **Caution for `nozzle_stem`**: On hotends, Port 1 **already is** `nozzle_tip`, so do NOT apply a Port-1 shift to `C_cad` of `nozzle_stem`; only shift `toolhead_mount` (or shift `C_clamp` from Port 0 to Port 1).

### Pitfall 2B: Top-Ring Local Collar Compliance vs. Bulk Monoblock Chassis Stiffness
- **Symptom**: Baking a CAD toolhead with Port 1 on a small top hotend mounting ring (`Z = +30.5 mm` above the MGN12 rail) includes local collar-lip rotational compliance in `krx`; projecting that compliance 66 mm down to the nozzle makes the whole toolhead body appear 3× too soft (`705 N/mm`), and gating a stiffness fix only on `will_use_braced_combo` makes toggling the hotend bottom brace OFF artificially soften the entire 130 g chassis body instead of isolating the compliance drop to the hotend.
- **Invariant**: Keep `toolhead_mount` bulk chassis stiffness **invariant** to hotend-brace toggles (`has_elevated_hotend_port` pre-scaling separates bulk chassis stiffness from Stage 3 `hotend_clamp`), so removing the Ti bottom brace leaves Microwaved's body bend unchanged (`~1.2 rad/m`) while making **only** the unbraced hotend & nozzle tip whip dramatically (`~4.6×` more hotend bend).

### Pitfall 2C: Double-Counting Stage 1 Backplate or Unscaled Manifest Metadata
- **Symptom**: Self-contained CAD toolheads bolt directly onto the MGN carriage; adding a generic procedural backplate compliance in series double-counts Stage 1 flex, and copying raw `manifest["mass_kg"]` or `manifest["port_stiffness_n_per_mm"]` ignores `mass_scale_applied` / `stiffness_scale_applied` when the user switches chassis material (e.g., Ti-6Al-4V → Al 6061-T6).
- **Invariant**:
  1. For self-contained CAD toolheads, place **only** Hertzian MGN carriage ball-raceway compliance (`C_rail_1`) in series and set `backplate_only_n_per_mm = 0.0`.
  2. Always propagate `mass_scale_applied` and `stiffness_scale_applied` into `AttachedSuperelement.mass_kg` and `to_summary_dict()`, and always forward `name`, `element_type`, `num_elements`, `port_stiffness_n_per_mm`, `port0_mm`, and `port1_mm`.

---

## 3. Inertial Excitation & Linear Rail Mechanics (`modal.py`, `solver.py`, `sensitivity.py`, `components.py`)

### Pitfall 3A: Putting 100% Inertial Force on `toolhead_cog` and 0% on `nozzle_tip`
- **Symptom**: Removing a stiff structural hotend brace (dropping hotend stiffness 8.5× from `2,231 → 263 N/mm`) causes input-shaper frequency recommendations to go **UP** (from the `-1.67 g` mass reduction alone), and leaves `nozzle_stem` with `0.0%` static adjoint compliance share.
- **Invariant**: Across **all** solvers (`modal.py`, `solver.py`, `sensitivity.py`), split toolhead X/Y inertial excitation across `toolhead_cog` and `nozzle_tip` in exact proportion to their physical lumped masses ($F_i = m_i a$):
  $$w_{\text{noz}} = \frac{m_{\text{noz}}}{m_{\text{cog}} + m_{\text{noz}}}, \qquad w_{\text{cog}} = 1 - w_{\text{noz}}$$
- **Verification**: Always include a monotonic regression test confirming that disabling a hotend brace lowers both static nozzle tip stiffness, static accel deflection resistance, and X/Y input-shaper frequency.

### Pitfall 3B: Double-Counting $\sin 45^\circ$ and Wing-Flex Factors in Hertzian Rail Roll Stiffness
- **Symptom**: Using $r_y = (W_R / 2)\sin 45^\circ$ (e.g., `4.2 mm` on MGN12) together with `k_zz` (which already contains the 45° contact-angle projection and Hung's 15% skirt compliance) plus an extra `/ 1.5` factor makes MGN12 roll stiffness `k_rx` ~2.5× too soft (`k_rx / k_ry = 0.30` vs. HIWIN's official $M_R / M_P = 1.86$).
- **Invariant**: Set `row_half_pitch_m = W_R / 2` matching HIWIN $M_R / C_0$ (`4.5 / 6.0 / 7.5 / 7.5 mm` for MGN9/12/15/HGH15) with a single C-block skirt factor (`1.25`). Ensure **every** caller (`builder.py`, `rcsa.py`, `ballbar.py`, `overrides.py`) forwards `preload_fraction_override` and `resolve_rail_quality(spec)` and selects roll (`k_rx`) vs. pitch/yaw (`0.5*(k_ry+k_rz)`) matching the actual load axis.

### Pitfall 3C: Single-Source Accessory Mass & CoG Accounting
- **Symptom**: UI headers, BOM, Model Confidence, RCSA, or FEA read raw `spec.toolhead.total_mass_g` / `hotend.mass_g` / `cog_offset_mm` and disagree whenever optional CAD hardware accessories (Ti bottom brace, nozzle shields) are toggled.
- **Invariant**: Route **all** backend mass/CoG consumers through `resolve_toolhead_hardware_accessories(spec)` and **all** frontend displays through `sim.geometry.toolhead_structure` (`effective_cog_offset_mm`, `toolhead_body_mass_g`, `hotend_mass_g`, `assembled_toolhead_mass_g`), resetting `nozzle_shield_mode = "auto"` when switching presets.

---

## 4. 3D Studio Viewport Placement & Modal/FEA Deformation (`PrinterViewport.tsx`)

### Pitfall 4A: Aligning CAD Toolheads by Bounding-Box `min.z`
- **Symptom**: Parts extending *behind* the MGN carriage (e.g., Beacon RevH probe, rear belt wings) shift `-bounds.min.z` forward by 30–35 mm, pushing the entire toolhead off the carriage and detaching the hotend.
- **Invariant**: Always anchor CAD toolheads by their **exact CAD datum ports** (`Port 0` flush onto the MGN carriage face, `Port 1` directly mating Stage 2 / hotend top flange) — never by overall mesh bounding-box extremes.

### Pitfall 4B: `kPlateStage === 0` Falling Into Legacy Backplate-Bending Branch
- **Symptom**: On self-contained CAD toolheads (`backplate_only_n_per_mm = 0`), a guard checking `kPlateStage > 0` falls into the fallback branch (`stageRailFrac = 0`, `stagePlateFrac = 67%`), freezing the MGN carriage and bending solid metal CAD bodies like rubber around a non-existent backplate hinge.
- **Invariant**: Explicitly handle `kRailStage > 0 && (!kPlateStage || kPlateStage <= 0)` by splitting compliance strictly between `stageRailFrac` and `stageBodyFrac` with `stagePlateFrac = 0`.

### Pitfall 4C: Carriage Roll Pivot, X-Beam Twist & 4-Bolt Face Clamping
- **Symptom**: Rotating the MGN carriage block around the X-beam centerline `(0, 0, 0)` instead of the MGN ball-raceway contact center detaches the carriage off the steel rail into a V-gap and clips it into the X-beam; failing to clamp the full 4× M3 bolt footprint (`±13.5 mm × 3.8 mm`) or adding translational `tmpUBodyShear` makes the bolted toolhead slide across the carriage face.
- **Invariant**:
  1. Rotate `modalRailShoe` rigidly with the X-beam's modal twist `tmpBeamQuat`.
  2. Roll `modalToolhead` strictly by the **relative** carriage-to-rail rotation (`stageRailFrac * (rot_cog - rot_th)`) pivoted around `modalRailPivotLocal` (the MGN ball-raceway contact line).
  3. Keep the entire 4× M3 bolted pad clamped (`s2 === 0`, `tmpUBodyShear = 0` when `plateFrac === 0`), with $C^1$-smooth flexure emerging only outside the bolted zone and staying bit-for-bit seam-locked at Port 1.
  4. In `registerElasticGroup`, always use `traverseVisible` so hidden accessory sub-meshes are never cloned or deformed.

---

## 5. Studio Sidebar CSS & Flex-Direction Traps (`styles.css`, `DesignPanel.tsx`)

### Pitfall 5A: `flex: 1 0 100%` Inside `flex-direction: column` Containers
- **Symptom**: Reusing a horizontal flex-wrap full-row utility (`flex: 1 0 100%`) inside a vertical `flex-direction: column` `.field-row` or `.section-card` sets `flex-basis: 100%` of **vertical height**, exploding the row and clipping sibling controls.
- **Invariant**: Use `width: 100%; box-sizing: border-box; flex: 0 0 auto;` on standalone toggle rows (scope `flex: 1 0 100%` strictly to horizontal wrap parents like `.studio-mode-bar .cad-cms-toggle-row`), never nest `.field-row` inside `.field-row`, and ensure every sidebar header row containing title + pills has `flex-wrap: wrap` so narrow 320–360 px panels never clip.

---

## Pre-Finish Verification Checklist for Every New CAD Hardware Asset

1. **Baker**: `rigid_body_max_freq_hz < 0.05 Hz`, watertight volume/mass/CoG checked, 24k-tri area-weighted GLB with 34° creased normals, no double-drawn brace/shield meshes.
2. **Backend**: Port-1 → CoG rigid-arm shift verified, `mass_scale_applied` & `stiffness_scale_applied` propagated, inertial excitation split across `cog` + `nozzle_tip`, monotonic brace/stiffness test passing in `pytest`.
3. **Viewport**: Inspected live via `capture_studio_viewport` in **Assembly**, **FEA Stress**, and **Modal** modes (2 frames across modal phase) — zero rail-carriage gap, zero 4-bolt sliding, zero shield overlap, clean Port 0/Port 1 alignment.
4. **Studio UI**: All header pills, preset dropdowns, Part Inspector, and Model Confidence report live effective `N/mm`, `g`, and `CoG`, with `npm run build --prefix web` passing `check:style` cleanly.
