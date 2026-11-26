# 06 — Hardware Construction & Preset Fidelity Audit

**Scope:** `src/printsim/geometry/builder.py`, `src/printsim/schema/printer.py`, `src/printsim/geometry/bom.py`, and all 8 presets in `examples/*.yaml`.
**Question:** Do the structural models and presets reflect how real 3D printers are actually built?
**Method:** I read the builder topology for each architecture, ran one inspection script over all 8 presets (geometry, masses, belts, toolheads, steppers, BOM), and ran targeted modal sensitivity checks with `solve_modal_fea`. I compared the results against published build documentation: the Voron docs and Klipper reference configs, the VzBot docs, DoomCube kit listings, and Gates/Misumi belt data. This audit is read-only; no source files were changed.

---

## 0. Evidence collected

### 0.1 Inspection script output (all 8 presets)

| Preset | Frame (mm, centreline) | Build vol. | Plate `Lx·Ly·t·ρ` | `bed.mass_g` (ratio) | Moving X / Y | Frame / Total mass | Belt (N → pluck @150 mm) | Toolhead `preset_id` / mass | XY stepper (rated / **run** A, Ω, steps) | Builder cost / BOM cost |
|---|---|---|---|---|---|---|---|---|---|---|
| `cartesian_bedslinger` | 360×380×420 (2040) | 220×220×250 | 596 g | 1350 (2.26×) | 490 g / **1700 g** | 2.9 / 5.7 kg | 6 mm 22 N → 146 Hz | `stealthburner` (default!) / 460 g | 1.5 / **1.2**, 1.2 Ω, 200 | $207 / $630 |
| `corexy_voron_style` | 460×460×500 (2020) | 300³ | 2016 g | 2500 (1.24×) | 420 / 1073 g | 4.5 / 12.7 kg | 6 mm 28 N → **164 Hz** | `stealthburner` / 380 g | 2.0 / **1.2**, 1.2 Ω, 200 | $331 / $1181 |
| `delta_racer` | 420×420×780 (3030) | 240×240×320 | 1136 g | 1600 (1.41×) | 285 / 285 g | 4.7 / 7.7 kg | 9 mm 36 N → 152 Hz | `stealthburner` (default!) / 195 g | 2.2 / **1.2**, 1.2 Ω, **200 ("0.9°")** | $275 / $840 |
| `doomcube_4040` | 500×500×520 (**4040 everywhere**) | 300×300×250 | 2394 g | 3100 (1.29×) | 310 / 969 g | **14.0 / 31.2 kg** | 9 mm 65 N → 205 Hz | `a4t_beacon` / 270 g | 2.8 / 2.0, 0.7 Ω, 200 | $732 / $1829 |
| `hyper_corexy_carbon` | 470×470×490 (3030) | 300×300×280 | 2016 g | 2400 (1.19×) | 265 / 641 g | 8.4 / 16.6 kg | 9 mm 42 N → 164 Hz | `stealthburner` (default!) / 225 g | 2.5 / **1.2**, **1.2 Ω**, 200 | $428 / $1345 |
| `voron_monolith_awd` | 460×460×500 (2020) | 300×300×250 | 2016 g | 2500 (1.24×) | 330 / 896 g | 4.4 / **19.6 kg** (9.7 kg panels) | 9 mm 65 N → 205 Hz | `stealthburner` (default!) / 290 g | 2.5 / 2.2, 0.7 Ω, 200 | $554 / $1546 |
| `voron_trident_300` | 460×460×500 (2020) | 300×300×250 | 2016 g | 2500 (1.24×) | 420 / 1068 g | 4.2 / 12.3 kg | 6 mm 28 N → **164 Hz** | `stealthburner` / 380 g | 2.0 / 1.2, 1.2 Ω, 200 | $292 / $993 |
| `vzbot_330_awd` | 480×480×540 (2020) | 330×330×300 | 1738 g | 2350 (1.35×) | 265 / 841 g | 4.9 / 15.0 kg | 9 mm 65 N → 205 Hz | `archetype_cpap` / 225 g | 2.8 / 2.0, 0.7 Ω, 200 | $560 / $1473 |

Delta check: `delta_radius_mm=165` and `delta_arm_length_mm=310` in the spec, but the builder produces **3 arms of 340 mm, profile `brace_rod_10mm`**.

### 0.2 Modal sensitivity checks (`solve_modal_fea`, first 4 modes)

| Experiment | Result | What it shows |
|---|---|---|
| V2.4 (`corexy_voron_style`), toolhead Z = 0.1 vs 0.9 | **Identical** (92.8 / 120.1 / 142.7 / 169.4 Hz) | The flying gantry never moves in Z. Column lengths and bed gap stay frozen. |
| Trident, toolhead Z = 0.1 vs 0.9 | 31.8 → 41.4 Hz | The fixed-gantry case responds to bed height, as it should. |
| V2.4 vs Trident fundamental | **92.8 Hz vs 38.5 Hz** | The model rates the flying gantry about 2.4× *stiffer* than the fixed gantry. Real-world experience says the opposite. |
| V2.4, `corner_gussets` → `none` | 92.8 → 82.7 Hz (−11 %) | The phantom rear-face diagonal adds about 12 % stiffness. |
| Trident, `corner_gussets` → `none` | New 37.2 Hz X-racking mode appears | The phantom diagonal hides the X-racking mode. |
| Bedslinger, bed Y = 0.1 vs 0.9 | **Identical** (102.7 / 113.1 Hz) | The bed's Y travel and overhang are ignored. |
| Bedslinger fundamental (unbraced 2040 portal) | 102.7 Hz "Whole-Frame Y Sway" | Too high. Unbraced i3 portals typically ring at about 30–60 Hz. |
| Delta lowest modes | 88–99 Hz, all "Toolhead & Nozzle Pitch Whip" | With pin-ended single rods, effector tilt is held only by artificial stabilisation, not by parallelogram geometry. |

---

## (a) Per-architecture construction comparison: real printer vs `builder.py`

### A1. Voron 2.4: flying gantry (`belted_quad_z`)

| Aspect | Real Voron 2.4 | `builder.py` | Verdict |
|---|---|---|---|
| Gantry–column joint | The gantry is a separate 2020 rectangle that hangs on 4 MGN9 Z carriages, one per column, and on 4 GT2 Z belts driven by 80T:16T reduced motors at the base. The printed Z-joints let the gantry tilt so it can be quad-gantry-levelled. | The 4 gantry corners `g_fl/g_fr/g_rl/g_rr` are **frame nodes on the columns themselves** (`builder.py:1765-1768`). The Y-rail extrusions are welded between them (`:1903-1906`). There are no Z-carriage, Z-belt or Z-joint springs. | **CRITICAL.** The gantry's Z, pitch and roll compliance (Z-belt EA/L, carriage moments, joint release) is missing. The gantry is welded rigid. |
| Gantry Z position | The gantry travels about 250–350 mm in Z. The lower column span changes as it moves. | `z_gantry = h − 0.055` is fixed (`:1724`). The bed is pinned at `orig_z` (`:1744-1745`). The `w_rel` input is ignored. | **CRITICAL.** Modes are identical across the full Z travel (§0.2). |
| A/B motors | Mounted on the gantry's rear corners, so they **move in Z with the gantry** and add about 0.7 kg to the hanging mass. | `xy_motor_kg` is lumped on `g_rl/g_rr` column nodes (`:1767-1768`), which act as rigid frame mass. | MODERATE. The mass sits in the right place in XY but is carried by the wrong load path. |
| Z motors | 4 motors at the base. | Motor mass is lumped on all 4 feet (`:1760-1763`). Those nodes are `fixed`, so the lumps only enter `total_mass`. | ACCURATE for 4 motors (but see A2/A3). |
| Fixed bed | The bed bolts to 2 × 2020 bed extrusions on the lower frame, so its inertia couples into the frame. | 4 synthetic solid-bar "arms" run diagonally from the **ground-fixed feet** to `bed_frame` (`:1922-1944`), with made-up sections (`iy = 4e-8 + 1.2e-7·(t/8)³`). | MODERATE. The bed is grounded, not frame-coupled, so the bed–frame interaction is lost. |
| Panels | 3 mm acrylic or PC on clips. They are not bolted to the gantry. | `panels_bolted_to_gantry_effective` correctly returns False for a flying gantry (`printer.py:930-938`). | ACCURATE (but the preset's material and mounting fall back to the defaults; see B2). |

**Net effect:** the model reproduces a fixed-gantry CoreXY with a grounded bed. That is why it calls the 2.4 stiffer than the Trident (92.8 vs 38.5 Hz). Real builds behave the other way round: the gantry hanging on 4 belts is the 2.4's known weak direction.

### A2. Voron Trident: triple leadscrew (`leadscrew_triple`)

| Aspect | Real Trident | `builder.py` | Verdict |
|---|---|---|---|
| Z layout | 3 TR8 leadscrews and 3 MGN9 Z rails: front-left, front-right and rear-centre (per the Klipper `z_tilt` reference config). The gantry is fixed. | Front Z nodes sit on the front-left/front-right columns (split at `z_bed_*`), and there is a rear-centre column (`:1783-1791`, `:1836-1843`, `:1960-1980`). | ACCURATE topology. |
| Bed–Z coupling | Each bed corner rides an MGN9 carriage (stiff laterally) plus a leadscrew nut (axial stiffness plus backlash or preload). The joint is nearly kinematic. | The bed arms are **rigid beam elements** fused to the column nodes (`:1983-2004`). Their section constants are tuned (`iy_triple_arm = 1.8e-8 + 7.2e-8·(t/8)³`, `:1918`) and the material is massless (`_make_bed_virtual_material`). | MODERATE. There is no leadscrew-nut axial spring and no carriage-moment spring, and the numbers can't be traced to a datasheet. |
| Rear Z rail stiffness | A 2020 extrusion plus an **MGN9** rail. | `bed_z_rail_profile` is named "+ MGN12", uses `iy×2.6` and takes its rail area from `gantry.y_rail_profile` (`:1947-1959`). | MINOR. The rail size is borrowed from the Y axis and the stiffness multiplier is a guess. |
| Z motors | 3 | The motor mass is lumped on **4** feet (`:1760-1763`). | MINOR (adds +1 phantom motor to total mass). |
| Rear-face bracing | None. A Voron frame has no diagonals; it relies on blind joints and clip-on panels. | `corner_gussets` adds a **full-face 10 mm diagonal** from `b_rl` to `g_rr` (`:2094-2096`). | **CRITICAL** for every Voron-family preset. It hides the X-racking mode (§0.2). |

### A3. Dual-Z CoreXY (`leadscrew_dual`; VzBot preset)

| Aspect | Real VzBot 330 | `builder.py` | Verdict |
|---|---|---|---|
| Z hardware | **1** NEMA17 drives **2** TR8 leadscrews through a closed GT2 sync belt. The bed is guided by **4 × Ø10 mm smooth rods** with LMK10LUU bearings (vzbot.org kit spec). | Adds side mid-depth vertical members (`bed_dual_z_rail_profile`, set to 0.22× the frame profile, `:2006-2040`) and splits the base and Y rails at mid-depth (`:1793-1803`, `:1827-1831`). | MODERATE. Real rods are 4 × Ø10 mm steel with I = 491 mm⁴ each. The model uses 2 guides with an invented section. |
| Gantry–Z tie | The gantry and the Z guides are separate parts. | When the gantry is within 2 mm of mid-depth, the moving `y_car_l/r` is tied to `g_left_mid/right_mid` with **1e9 N/m, 1e7 N·m/rad** springs (`:1868-1901`). | MODERATE. A moving carriage gets welded rigidly to the frame at mid-travel, so stiffness jumps at Y ≈ 0.5. |
| Motor / rail count | 1 Z motor, 0 Z rails | BOM: **2** Z motors (`bom.py:658-668`) and **2 MGN12 Z rails** (`bom.py:884-890, 910`). Builder: 4 Z motor lumps. | MODERATE (cost and mass). |

### A4. Single-Z cantilever (`leadscrew_single`)

| Aspect | Typical real build (Ender-5 class) | `builder.py` | Verdict |
|---|---|---|---|
| Bed support | One leadscrew between **2 guide rods or rails** about 150–250 mm apart on the rear face. The rod spacing gives roll stiffness; the cantilever gives pitch. | One rear-centre column, `z_rail_prof` (`:1804-1811`, `:1960-1980`), and one cantilever arm (`:2063-2084`). | MINOR. Roll stiffness comes from a single synthetic section, not from the pair spacing. The BOM correctly lists 2 Z rails (`bom.py:888`), which doesn't match the builder's single column. |

### A5. Cartesian bedslinger (`cartesian_bedslinger`)

| Aspect | Real i3 / Ender-3 class | `builder.py` | Verdict |
|---|---|---|---|
| Portal base | 2040 uprights are bolted to the base extrusions with 2–4 M5 screws and angle brackets, and the base stands on feet. That joint is the dominant source of portal Y-nod compliance. | The portal base nodes `p_bl/p_br` are **`fixed=True`** (`:2295-2296`). The uprights are clamped to the ground, so the base H-frame (`:2314-2320`) contributes nothing. | **CRITICAL.** The unbraced portal fundamental comes out at 102.7 Hz; real portals sit around 30–60 Hz. |
| X gantry | X-ends ride V-wheels or LM8UU bearings on the uprights and are lifted by 1–2 leadscrews. | The X beam is welded straight to the upright nodes `z_l/z_r` (`:2298-2299`, `:2332-2333`). | MODERATE. There is no X-end/wheel compliance. |
| Moving bed | The bed rides a Y carriage on a central 2040/4040 spine (V-wheels or rails), and its Y position varies along with the overhang. | The bed node is fixed at `(w/2, d/2)` (`:2407`). The `v` input is discarded (`u, _, w_rel`, `:2270`). Y compliance comes from a belt-calibrated arm (`:2456-2469`). | MODERATE. Modes don't change with bed Y (§0.2). The reduced-order belt arm is a reasonable idea. |
| Y moving mass | The bed assembly (plate, carriage, wheels, sheet, heater) is about 1.0–1.3 kg. | `y_moving = bed_kg + 0.35 + bkt` (`:2404`). The preset's 1350 g already includes the carriage (2.26× the plate mass). | MINOR. Y moving mass is about 25 % too high. |
| `z_drive` | Not applicable; Z lifts the gantry. | The preset says `leadscrew_dual`, which the builder ignores. | MINOR (semantic). |

### A6. Delta (`delta`)

| Aspect | Real Kossel / delta | `builder.py` | Verdict |
|---|---|---|---|
| Arms | **6 rods in 3 parallel pairs** (parallelograms) with ball joints. The pairs hold the effector parallel to the bed and lock pitch, roll and yaw. Effector tilt stiffness comes from pair spacing × rod EA. | **3 single pin-ended truss rods**, all meeting at one `toolhead` node (`:2554`). | **CRITICAL.** The 3 rotational DOFs are physically unconstrained. The 88–99 Hz "pitch whip" modes are numerical artefacts. The BOM already lists 6 rods (`bom.py:1018-1031`). |
| Arm geometry | Tower-carriage pivots sit at `DELTA_RADIUS` (here 165 mm) from the effector pivots. Rod length is `delta_arm_length_mm`. | The tower radius is `max(w, d)/2 = 210 mm` (`:2511`), so `delta_radius_mm` is **ignored**. Carriage height is computed with `radius·0.75` (`:2521`), so the built arm is **340 mm vs the 310 mm spec**. | **CRITICAL.** The kinematic geometry doesn't match the spec. |
| Arm section | 2 × Ø6 mm CF tube per tower (A ≈ 31 mm² total). | Hardcoded `brace_rod_10mm` **solid** (A = 78.5 mm²) in `carbon_fiber_tube`. `gantry.x_beam_profile` (`cf_tube_20x20x1_5`) is ignored. | MODERATE. Axial stiffness is about 2.5× too high. |
| Effector nozzle | Nozzle on the effector axis, offset about (0, 0, −30…−45) mm. | The preset has no `nozzle_offset_mm`, so it inherits the CoreXY default `(0, −22, −62)` (`printer.py:793-796`). | MODERATE. The 22 mm lateral lever is spurious. |
| Bed | Fixed to the base triangle. | Fixed; preset says `z_drive: leadscrew_triple`, which is ignored. The schema has no `fixed` option (`printer.py:898-900`). | MINOR. |

### A7. Cross-cutting construction issues

| Issue | Location | Verdict |
|---|---|---|
| One `frame.profile` is used for **every** frame member: columns, horizontals, Y-gantry extrusions and base. | `builder.py:1813-1814` (and similarly in the bedslinger and delta builders) | MODERATE. DoomCube uses 4040 corner uprights with 2020 horizontals; i3 frames mix 2040, 2020 and 4040. |
| `corner_gussets` is modelled as a full-length diagonal, not as corner gussets. | `builder.py:2094-2096`; bedslinger `:2336-2338` | **CRITICAL** (see A2). |
| The Z-motor lump is always ×4 on the CoreXY feet, regardless of `z_drive`. | `builder.py:1760-1763` | MINOR. It only matters for total mass, or once `feet_type` stops being rigid (`:1278-1306`), at which point it becomes dynamic mass. |
| Builder cost and BOM cost differ by 2.5–3.4×. | `builder.py:2682-2685` (flat $140 baseline) vs `bom.py` | MINOR. These are two inconsistent cost models. |

---

## (b) Per-preset accuracy scorecard

Grades: A means a faithful match to the real machine; D means a significant mismatch.

| Preset | Frame & topology | Z-system | Gantry / belts | Toolhead | Steppers | Bed | Panels | **Overall** | Key problems |
|---|---|---|---|---|---|---|---|---|---|
| `voron_trident_300` | B− | B | B | A− | B | A− | A | **B** | Phantom rear diagonal (`yaml:10`). Belt at 28 N = 164 Hz vs the Voron spec of 110 Hz (about 9 N per Voron docs) (`yaml:32`). **Resolved 2026-11-26: preset set to 9.0 N (110 Hz).** Z motor named TR8x4 (`yaml:51`), but the BOM says TR8x8 4-start and stock Trident ships TR8x8. CoG (0, −18, −25) differs from the `stealthburner` catalog value (0, −22, −28) (`printer.py:821`). |
| `corexy_voron_style` (V2.4-style) | C | **D** | B− | B | B− | C | C | **C−** | The flying gantry is welded and frozen in Z (A1). Y rail is MGN12 (`yaml:19`); a real 2.4 uses MGN9 on Y and Z, and the BOM also puts MGN12 on the Z rails (`bom.py:910`). Panels fall back to the defaults `bolted_structural` + `acm_dibond` (`yaml:13-14`, `printer.py:458-465`); real 2.4 panels are clipped acrylic. Z stepper defaults to a 40 mm, 280 g motor; real 2.4 uses the same 48 mm motor as A/B. Run current defaults to 1.2 A. |
| `voron_monolith_awd` | B− | B | **C** | C | **D** | A− | C | **C** | `preset_id: ldo_2804ah` but the name is "2804AC", with 0.65 N·m / 2.5 A / 6.8e-6 / 360 g vs the catalog's 0.42 / 2.8 / 6.2e-6 / 350 (`yaml:43-52` vs `printer.py:664-675`). `belt_routing` is left at the default `stock_corexy` (`yaml:17-20`), even though Monolith *is* the short-path routing (DoomCube and VzBot set `monolith_short`). Toolhead `preset_id` is the default `stealthburner` and `internal_stiffness` is the default 450 despite a 6061 mount. 4 mm 5052 bolted panels (9.7 kg) are not part of the Monolith gantry mod. |
| `doomcube_4040` | **D** | B | A− | A | A− | A | B | **C+** | `profile: extrusion_4040` (`yaml:8`) is applied to every member, but real DoomCube uses 4040 corner uprights with 2020 horizontals. Frame mass is overstated (14.0 kg) and stiffness is inflated. Total 31.2 kg. |
| `vzbot_330_awd` | B | **D** | A− | A | A− | B+ | B | **B−** | `leadscrew_dual` (`yaml:100`) becomes 2 Z motors and 2 MGN12 Z rails. Real VzBot: 1 motor, 2 belt-synced TR8 screws, 4 × Ø10 rods. Rigid 1e9 tie at mid-Y (A3). |
| `hyper_corexy_carbon` (generic) | B (`full_truss` is explicit) | B | B+ | **C** | **C** | B+ | C | **B−** | Toolhead block (`yaml:45-49`) has no `nozzle_offset_mm`, `preset_id`, `mount_material` or `internal_stiffness`, so it inherits (0, −22, −62), `stealthburner`, `abs_printed` and 450 N/mm. That contradicts "Ultra-Light CNC/PA12-CF, CoG (0, −12, −18)". Stepper block (`yaml:37-44`) has no `run_current_a` or `phase_resistance_ohm`, so it inherits 1.2 A (48 % of rated) and 1.2 Ω (a 1.1 mH, 48 V motor is about 0.6–0.9 Ω). |
| `cartesian_bedslinger` | **D** | n/a (B) | B | C | B− | B− | n/a | **C−** | Portal clamped to ground (A5). Bed position frozen. Single 2040 profile everywhere (`yaml:8`). `z_drive: leadscrew_dual` is semantic noise (`yaml:76`). Toolhead `preset_id` defaults to `stealthburner`; nozzle offset is the default. +0.35 kg added to Y mass (`builder.py:2404`). BOM total of $630 is about 2–3× a real i3-class machine. |
| `delta_racer` | B | n/a | **D** | **D** | **C** | B | n/a | **D+** | 3 single rods (A6). `delta_radius_mm: 165` (`yaml:37`) ignored; arms 340 vs 310 mm. Stepper named "0.9deg" (`yaml:39`) but `steps_per_rev` defaults to 200. Run current defaults to 1.2 A. No `nozzle_offset_mm` (`yaml:46-50`), so the effector nozzle sits 22 mm off-axis. `x_beam_profile: cf_tube_20x20x1_5` (`yaml:15`) is unused. `z_drive: leadscrew_triple` (`yaml:78`) is meaningless. |

### Catalog and schema observations

- **Belt linear mass** `μ = 11.5 g/m` for 6 mm fiberglass GT2 (`printer.py:510-514`). Supplier data gives about **7.5–10 g/m**, and Misumi scales to about 7.8 g/m. Using Voron's own calibration (110 Hz over 150 mm ≈ 2 lbf ≈ 8.9 N) gives μ = T/(2Lf)² ≈ **8.2 g/m**. So the pluck-to-tension conversion is off by about 40 %, and the Voron presets are tensioned about 3× above the Voron spec.
- **`preset_id` is a label only.** Nothing resolves fields from `STEPPER_CATALOG` or `TOOLHEAD_PRESETS`, yet `engine/pareto.py:278-288` uses `preset_id` to pick optimizer variables. Presets that omit it are encoded as `stealthburner` / `ldo_2504ac`. That affects 5 of 8 toolheads, including a delta effector and a bedslinger V6.
- **`StepperSpec` defaults** `run_current_a=1.2`, `phase_resistance_ohm=1.2`, `steps_per_rev=200` (`printer.py:602-610`) silently apply to 4 presets with very different motors.
- The `ldo_2504ac` catalog says rated 2.5 A (`printer.py:656`), while the Trident and V2.4-style presets say 2.0 A.
- **Bed mass ratios** of 1.19–1.41× the plate mass are realistic for CoreXY beds (plate + spring-steel sheet + heater + magnet). **ACCURATE.**
- **Toolhead masses** (SB 380 g, A4T 270 g, CPAP 225 g, delta effector 195 g) and **CoG/nozzle offsets** for the presets that set them explicitly are realistic. **ACCURATE.**
- **BOM Z hardware:** the 80T/16T reduction kit for `belted_quad_z` matches the V2.4 (**ACCURATE**). The BOM always lists the leadscrew as "TR8x8, 2 mm pitch, 4-start" (`bom.py:1072-1073`), but the presets name TR8x4 motors.

---

## (c) Prioritized findings

| # | Severity | Finding | Location | Recommended fix |
|---|---|---|---|---|
| 1 | **CRITICAL INACCURACY** | The Voron 2.4 flying gantry is welded to the frame columns and fixed in Z. There are no Z-belt, MGN9 carriage or Z-joint compliances. A/B motors are carried as frame mass. The bed is grounded via arms to the fixed feet. Result: 2.4 fundamental 92.8 Hz vs Trident 38.5 Hz, the inverse of reality, and modes don't change with Z. | `builder.py:1724, 1742-1745, 1765-1768, 1903-1906, 1922-1944` | Add separate gantry nodes `gz_fl…gz_rr` at `z_gantry(w_rel) = z_top − w_rel·bz`. Connect each to its column node with a `FastenerSpring`: lateral k from `hertzian_rail_carriage_stiffness("mgn9_rail")`, vertical k = `belt.stiffness_ea_n / L_belt(w_rel)` (Z belt, 80:16 reduction), and k_rot ≈ 0 about the axes the Z-joints release. Move `xy_motor_kg` onto the rear gantry nodes. Hang the bed on two 2020 bed extrusions between the lower frame rails instead of on ground arms. |
| 2 | **CRITICAL INACCURACY** | `bracing: corner_gussets` inserts a full-face Ø10 mm diagonal (b_rl→g_rr). All 4 Voron-family presets (and DoomCube) use it, but real Vorons have no diagonals. It adds about 12 % to the V2.4 fundamental and hides the Trident X-racking mode (37.2 Hz). | `builder.py:2094-2096, 2336-2338`; `corexy_voron_style.yaml:10`, `voron_trident_300.yaml:10`, `voron_monolith_awd.yaml:10`, `doomcube_4040.yaml:10` | Model real gussets as a rotational-stiffness boost at the corner joints, via `corner_joint_stiffness` / `_apply_corner_joint_compliance`, not as a truss member. Set the Voron presets to `bracing: none` with `corner_joint_type: blind_joint`. |
| 3 | **CRITICAL INACCURACY** | The delta uses 3 single pin-ended rods instead of 3 parallelogram pairs. Effector pitch/roll/yaw are unconstrained, and the reported 88–99 Hz "pitch whip" modes are artefacts. | `builder.py:2554`; BOM already lists 6 rods at `bom.py:1018-1031` | Build 2 effector pivot nodes per tower, about ±`rod_spacing/2` (e.g. 50 mm) along the carriage-perpendicular direction, rigidly tied to an effector body. Add 6 truss rods (2 per tower) using a real CF-tube profile (e.g. Ø6/4 mm), driven by a new `delta_rod_profile` field or `x_beam_profile`. |
| 4 | **CRITICAL INACCURACY** | Delta geometry ignores `delta_radius_mm` (uses frame half-width, 210 vs 165 mm) and computes carriage height with `radius·0.75`. Built arm length is 340 mm vs the 310 mm spec. | `builder.py:2511, 2521`; `delta_racer.yaml:36-37` | Place towers at `R_tower = delta_radius_mm + effector_offset + carriage_offset`. Set `z_car = z_eff + sqrt(L² − delta_radius²)`. Assert built rod length == `delta_arm_length_mm` in a unit test (IK round-trip). |
| 5 | **CRITICAL INACCURACY** | The bedslinger portal base is ground-clamped (`p_bl/p_br fixed=True`), so the upright-to-base bolted joint and base H-frame give no compliance. The unbraced 2040 portal reads 102.7 Hz; real i3 portal Y-nod is about 30–60 Hz. | `builder.py:2295-2296` | Make only the 4 corner feet fixed. Leave `p_bl/p_br` free on the base H-frame and add a bolted upright-to-base joint spring (`corner_joint_stiffness(fp, "l_bracket")`). Validate against published Ender-3 ADXL Y/X resonances (about 35–55 Hz). |
| 6 | MODERATE DISCREPANCY | Belt μ = 11.5 g/m is about 40 % too high (real 7.5–10 g/m). Voron presets run 28 N = 164 Hz vs the Voron spec of 110 Hz / about 9 N. | `printer.py:510-514`; `voron_trident_300.yaml:32`, `corexy_voron_style.yaml:28` | Set μ ≈ 0.0080 kg/m for 6 mm fiberglass (scale by width) and cite Gates/Misumi. Retune the Voron presets to the pluck target: about 10–12 N for 6 mm. Add a test that 110 Hz at 150 mm gives about 9 N. **Resolved 2026-11-26:** catalog belt mass corrected earlier; the four stock Voron presets now ship 9.0 N (109.8 Hz pluck), see `docs/physics/07_mounting_and_belt_refinements.md` §5.3. |
| 7 | MODERATE DISCREPANCY | The VzBot Z system is modelled as 2 motors + 2 MGN12 rails. Real: 1 motor, belt-synced 2 × TR8, 4 × Ø10 mm rods. A rigid 1e9 N/m / 1e7 N·m/rad tie welds the moving Y carriages to the frame at mid-Y. | `builder.py:1793-1803, 1868-1901, 2006-2040`; `bom.py:658-668, 884-890`; `vzbot_330_awd.yaml:100` | Add `z_drive: leadscrew_dual_belt_sync` (or a `z_motor_count` / `z_guide_type: rods_4x10` field). Model 4 rod members with a real Ø10 section. Remove the carriage-to-frame tie; let the shared node split handle coincidence. |
| 8 | MODERATE DISCREPANCY | A single `frame.profile` is used for every member, so DoomCube's 2020 horizontals and gantry extrusions are modelled as 4040. Frame mass 14 kg, total 31 kg, stiffness inflated. | `builder.py:1813-1814`; `doomcube_4040.yaml:8` | Add an optional `frame.upright_profile` (or `horizontal_profile`). Use it for `col_*`, and `profile` for horizontals and `y_rail_*`. Set DoomCube to `upright_profile: extrusion_4040`, `profile: extrusion_2020`. |
| 9 | MODERATE DISCREPANCY | Leadscrew Z beds (triple/dual/single) are rigidly beam-fused to the columns with tuned, massless sections. There is no leadscrew-nut axial or carriage-moment compliance. | `builder.py:1918-1920, 1983-2084` | Replace the bed arms with `FastenerSpring`s: axial k from TR8 nut stiffness (POM about 5–15 N/µm, brass higher), lateral k and moment k from `hertzian_rail_carriage_stiffness("mgn9_rail")`. Keep a plate-bending member for the bed itself. |
| 10 | MODERATE DISCREPANCY | The bedslinger bed Y position is frozen (`v` discarded), and the X gantry is welded to the uprights with no X-end/V-wheel compliance. | `builder.py:2270, 2298-2299, 2407` | Place `bed_frame` at `orig_y + v·by`. Add X-end carriage springs (V-wheel or LM8UU lateral and moment stiffness). |
| 11 | MODERATE DISCREPANCY | `preset_id` is a label that doesn't drive any fields. 5 of 8 presets silently get `toolhead.preset_id = stealthburner` (delta, bedslinger, hyper, monolith, voron_style), and the optimizer keys off it. Monolith's stepper `preset_id: ldo_2804ah` contradicts its own values. | `printer.py:598, 786`; `engine/pareto.py:278-288`; `voron_monolith_awd.yaml:43-52` | Add a `model_validator` that fills unspecified fields from the catalog when `preset_id` is set, and warns when explicit fields diverge. Set `preset_id: custom` where no catalog entry fits. |
| 12 | MODERATE DISCREPANCY | Stepper defaults (`run_current_a=1.2`, `phase_resistance_ohm=1.2`, `steps_per_rev=200`) leak into 4 presets. The delta's "0.9deg" motor runs at 200 steps/rev; the hyper 2.5 A motor runs at 1.2 A. | `printer.py:602-610`; `delta_racer.yaml:38-45`, `hyper_corexy_carbon.yaml:37-44`, `corexy_voron_style.yaml:37-44`, `cartesian_bedslinger.yaml:36-43` | Make every preset explicitly specify `run_current_a`, `phase_resistance_ohm` and `steps_per_rev` (delta: 400). Consider defaulting `run_current_a` to about 0.7×`rated_current_a`. |
| 13 | MODERATE DISCREPANCY | Missing `nozzle_offset_mm` inherits the CoreXY default (0, −22, −62): wrong for the delta effector (should be on-axis) and the ultra-light hyper toolhead. | `printer.py:793-796`; `delta_racer.yaml:46-50`, `hyper_corexy_carbon.yaml:45-49` | Delta: `[0, 0, -40]`. Hyper: about `[0, -15, -50]`. Also set `mount_material` and `internal_stiffness_n_per_mm` consistent with "CNC/PA12-CF". |
| 14 | MODERATE DISCREPANCY | The Monolith preset leaves `belt_routing` at `stock_corexy`, even though Monolith is the short-path routing. It also adds 4 mm Al bolted panels (9.7 kg) that aren't part of the Monolith mod. | `voron_monolith_awd.yaml:17-20, 13-16` | Add `belt_routing: monolith_short`. Use stock Voron panels (`clip_enclosure`, `acrylic_pmma`, 3 mm), or rename the preset to make the panel upgrade explicit. |
| 15 | MODERATE DISCREPANCY | The V2.4-style preset falls back to `panel_mounting: bolted_structural` + `acm_dibond` and an MGN12 Y rail. Real 2.4: clipped acrylic, MGN9 Y/Z. | `corexy_voron_style.yaml:13-19`; `printer.py:458-465` | Set `panel_mounting: clip_enclosure`, `panel_material: acrylic_pmma`, `y_rail_profile: mgn9_rail`. Specify `z_stepper` as the 48 mm motor. |
| 16 | MINOR / POLISH | The Z-motor mass is always lumped ×4 on the CoreXY feet (Trident 3, dual 2, VzBot 1). It becomes dynamic mass when `feet_type` isn't rigid. | `builder.py:1760-1763` | Lump `n_z = _z_motor_count(...)` motors at the actual Z node bases. |
| 17 | MINOR / POLISH | The Z-rail virtual profile is named "MGN12" and takes its area from `y_rail_profile`. The BOM also uses `y_rail_profile` for the Z rails. | `builder.py:1947-1959`; `bom.py:910, 929, 933` | Add `gantry.z_rail_profile` (default `mgn9_rail`). |
| 18 | MINOR / POLISH | `z_drive` has no `fixed`/`none` option. The delta and bedslinger presets carry meaningless leadscrew values, and the default is `belted_quad_z` even for non-CoreXY specs. | `printer.py:898-900`; `delta_racer.yaml:78`, `cartesian_bedslinger.yaml:76` | Add `"fixed_bed"` (delta) and `"gantry_leadscrew_single"` / `"_dual"` (bedslinger). Validate per architecture. |
| 19 | MINOR / POLISH | Bedslinger Y moving mass adds +0.35 kg on top of a `bed.mass_g` that already includes the carriage (2.26× the plate). | `builder.py:2404` | Drop the constant, or document that `bed.mass_g` is plate-only. |
| 20 | MINOR / POLISH | Leadscrew naming: the BOM is fixed at "TR8x8, 2 mm pitch, 4-start", the presets name TR8x4 Z motors, and stock Trident uses TR8x8. | `bom.py:1072-1073`; `voron_trident_300.yaml:51` (+ DoomCube, Monolith, VzBot) | Add `z_stepper.leadscrew_lead_mm` and drive both the BOM and the Z-force math from it. |
| 21 | MINOR / POLISH | Builder `estimated_material_cost_usd` ($207–732) disagrees with BOM `total_cost_usd` ($630–1829) by 2.5–3.4×. | `builder.py:2682-2685` | Have the builder call `generate_bom(...).total_cost_usd`, or rename the builder value "raw structural material cost". |
| 22 | MINOR / POLISH | Delta arm profile is hardcoded `brace_rod_10mm` solid. `x_beam_profile` is ignored. | `builder.py:2554`; `delta_racer.yaml:15` | Covered by fix #3. |
| 23 | MINOR / POLISH | Frame centreline spans of 460 mm for 300-class Vorons. Real horizontals are 470 mm (outer ≈ 510 mm, centreline ≈ 490 mm). | `voron_trident_300.yaml:5-6`, `corexy_voron_style.yaml:5-6`, `voron_monolith_awd.yaml:5-6` | Use 490 mm centreline for 300-class Vorons. |
| 24 | ACCURATE | Trident triple-Z topology (FL/FR/rear-centre). Panels not bolted to a flying gantry. AWD front-motor placement. Bed-to-plate mass ratios for CoreXY. Explicit toolhead mass/CoG/nozzle values (Trident, DoomCube, VzBot, Monolith). 48 V low-inductance stepper values for DoomCube/VzBot. V2.4 80T:16T Z kit in the BOM. Delta BOM's 6 rods. | `builder.py:1783-1791, 1752-1754`; `printer.py:930-938`; `bom.py:1035-1059, 1018-1031` | No change. |

---

### Validation tests to add after the fixes

1. **Flying gantry:** V2.4 fundamental < Trident fundamental at the same frame, and V2.4 modes change monotonically with gantry Z.
2. **Delta:** built rod length == `delta_arm_length_mm` (±0.1 mm). Effector tilt stiffness ∝ (rod spacing)² · EA/L. No rigid-body rotation modes.
3. **Bracing:** `bracing: none` adds no truss members; `corner_gussets` adds no full-length diagonals.
4. **Belt:** μ_6mm ∈ [7, 10] g/m; 110 Hz at 150 mm ↔ 7–11 N.
5. **Bedslinger:** modes change with bed Y. The unbraced 2040 portal fundamental falls in about 30–70 Hz.

---

### 2026-11-26 update: `vzbot_330_awd` rebuilt to the official spec

The audit row above (480×480×540 2020 frame, 9 mm / 65 N belts, MGN12 X rail, `archetype_cpap`, 300 mm Z) described a placeholder, not a VzBot. The preset is now built from the official VzBoT3D BOM, the `Vz-330_v1.2` full-printer STEP, and `Firmware/330AWD-printer.cfg`:

| Aspect | Old preset | Real VzBot 330 AWD (now modelled) |
|---|---|---|
| Frame | 480×480×540, all 2020, rear cross bracing, 4 mm Al panels | 530×500×570, 2040 uprights + 2020 rings on blind joints, no diagonal bracing, 3 mm Al rear + PC sides (`side_panel_material`) |
| Gantry plane | separate ring 55 mm below top | Y rails on the inner faces of the top-ring Y extrusions (`gantry_on_top_frame`) |
| Gantry kit | auto → Monolith CNC (via `monolith_short`) | `vzbot_330_awd`: CAD-baked rear motor mounts, AWD front mounts, XY joints; stock CoreXY routing |
| X beam / rail | high-Tg CF, MGN12 | 20×20×1 CF tube (440 mm), MGN9H on top |
| Belts | 9 mm, 65 N | 6 mm GT2, 22 N (150–189 Hz over 150 mm) |
| Toolhead | Archetype CPAP, 344 g | `vz_printhead_cnc`: CAD-measured 343 g dressed, nozzle −54.2 mm, CoG +13.4 mm |
| Bed / Z | 6 mm ATP-5, kinematic 3-pt, 300 mm Z, TR8x4 | 8 mm cast 330×330 on 4 bolted silicone spacers, 400 mm Z, belt-synced 2× TR8x8, 4× Ø10 mm rods |

Findings #7 and #16 above (dual-Z modelled as 2 motors + MGN12 rails) were already resolved by `leadscrew_dual_belt_sync`. Result: X ≈ 106 Hz / Y ≈ 65 Hz versus the cited community band X 75–118 / Y 47–80 Hz (`tests/test_vzbot_330.py`).
