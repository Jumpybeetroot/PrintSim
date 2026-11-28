# PrintSim Sub-80% Confidence Calibration & Agent Task Backlog

> **Purpose:** Ordered, agent-ready backlog of every physical model, catalog option, and CAD asset currently scoring **below 80% confidence** in `build_model_confidence_audit()` (`src/printsim/schema/overrides.py`), sorted **lowest confidence first**.
>
> **How Agents Should Work Off This File:**
> 1. Pick an unclaimed task (`[ ]`) with non-overlapping file ownership.
> 2. Persist any newly discovered or verified datasheets, scale-weighed masses, curves, or empirical bench/Discord measurements to **PrintDB** (`/Users/jumpybeetroot/Projects/PrintDB` via `printdb` MCP) first.
> 3. Implement the physics/CAD upgrade + analytical/bench unit tests in `tests/`.
> 4. Update **both** `build_model_confidence_audit()` (`src/printsim/schema/overrides.py`) and `docs/MODEL_CONFIDENCE_AND_REFERENCES.md` (including its Calibration History table) when raising any confidence score.
> 5. Check off the item (`[x]`) with the new confidence `%` and commit hash.

---

## Priority Queue (All Models `< 80%`, Lowest First)

### Tier 1 — Critical Gaps (`34% – 48%`)

- [ ] **CAL-01 · Heavy / Industrial Belts (`34% – 58%`)** *(Partial — PrintDB/KB datasheets & analytical double-shear pin stiffness persisted; awaits live Hub-window Discord A/B sweeps)* — *Gantry (`belt`)*
  - **Current scores:** `at5_16mm` (`34%`), `htd3m_15mm` (`36%`), `at3_16mm` / `at3_gen3_16mm` (`38%`), `3gt_15mm` / `at3_10mm` (`40%`), `3gt_12mm` (`42%`), `3gt_9mm` (`44%`), `gt2_dual_9mm` (`52%`), `gt2_15mm` (`58%`), `gt2_12mm` (`76%`).
  - **Completed so far:** Persisted `components/belts/contitech-at3-10mm` (`C_spez = 80 kN`, `z_min = 20` reverse-bend) to PrintDB and verified Monolith `D5×24` double-shear shaft stiffness (`21.8 kN/mm` vs `3.2 kN/mm` single-shear) + `beta_dyn = 1.85` (range `1.6–2.0`).
  - **Remaining gap:** Live Hub-window Discord A/B ADXL shaker sweeps swapping `6 mm → 9 mm → 12 mm` GT2, 3GT, and AT3 on a single Monolith / VzBoT / Annex testbed.

- [x] **CAL-02 · Official STEP-Baked Toolhead Bodies (`48%` → `82% – 86%` CAD CMS)** — *Toolhead (`toolhead_construction`)*
  - **New scores:** `xol_compact` (**`86%`**, 27.9k Tet4), `a4t_beacon` (**`84%`**, 52.8k Tet4), `dragon_burner` (**`83%`**, 30.4k Tet4), `archetype_cpap` (**`82%`**, 55.3k Tet4); all 4 pass `11/11` `printsim cad verify` QA (`rb_max <= 0.0034 Hz`, `<= 30.2k` GLB tris) via `cad_recipes/bake_toolheads_cal02.py`.
  - **Future follow-up:** Bake `monolith_toolhead_cpap` once its STEP is public on `monolith3d.xyz` and measure `vz_printhead_cnc` plates (`66%`) with calipers.

- [ ] **CAL-03 · Voron Tap Z-Probe Carriage Compliance (`80%` CNC Tap / `70%` Printed Tap / `85%` StealthBurner+Tap CAD CMS)** *(Near-complete — `cnc_tap` now calibrated at `80%` against 2 real `bsmt` Voron 2.4 350 Vitalii CNC Tap Shake&Tune builds + controlled SB→XOL A/B sweep (`19/19` sweeps, `9.35%` CoreXY MAPE, `131 Hz` Tap mode confirmed); only `printed_tap` (`70%`) awaits a same-printer Printed Tap vs Beacon A/B swap to cross `80%`)* — *Toolhead (`z_probe`)*
  - **Current scores:** `stealthburner_tap` live CAD CMS (**`85%`**); `cnc_tap` (**`80%`**, calibrated); `printed_tap` (`70%`).
  - **Completed so far:** Baked `toolheads/stealthburner_tap` Tet4 Craig–Bampton CMS (`bac08af`), added HIWIN `mgn7_rail` Hertzian catalog + `tap_sliding_rail_stiffness()` in `fea/components.py`, wired 6-DOF series compliance at Port 0 in `_builder_toolhead.py`, captured 2 real `bsmt` Voron 2.4 350 Vitalii CNC Tap builds (`49.8/36.0 Hz` SB+chains → `55.4/38.0 Hz` XOL+umbilical, `131 Hz` Peak 2 Tap mode) into `shaper_benchmarks.yaml` + Discord KB (`19/19` A/B sweeps, `9.35%` CoreXY MAPE).
  - **Remaining gap:** Controlled same-printer Printed Tap vs Beacon/Cartographer ADXL swap to raise `printed_tap` `70%` → `>= 80%`.

- [x] **CAL-04 · Benchtop Feet, Anti-Vibration Pads & Concrete Paver (`48% – 58%` → `77% – 79%`)** — *Frame (`frame_feet`)*
  - **New scores:** `rubber_feet` (**`79%`**, `380/148 N/mm`, `η=0.12`), `anti_vib_pads` (**`78%`**, `65/22 N/mm`, `η=0.56` Sorbothane 50D / HULA), `paver_on_foam` (**`77%`**, `215/78 N/mm`, `η=0.28`, `18 kg`); `suspended_lines` unchanged at `62%`. Persisted `references/dynamics/benchtop-isolator-feet-stiffness` to PrintDB.

- [x] **CAL-05 · Linear Rail Grade / Clone Preload Multipliers (`48% – 86%` → `80% – 88%`)** — *Gantry (`rail_quality`)*
  - **New scores:** `budget_generic` (**`80%`**), `midgrade_matched` (**`82%`**), `precision_thk_nsk` (**`84%`**), `hiwin_genuine` (**`88%`**). Replaced flat scalar multipliers with first-principles HIWIN (`GW-11-7-EN-2401-K`), THK (`SRS/HSR`), NSK (`E728`) radial-clearance, conformity ($f=r_g/d_b$), and ISO 3290 ball-scatter active-ball Hertzian contact integration; persisted `references/materials_mechanics/rail-grade-clearance-and-ball-sharing` to PrintDB.

---

### Tier 2 — High-Impact Structural & Motion Calibration (`52% – 68%`)

- [x] **CAL-06 · Extrusion Cavity Fillings (`52% – 64%` → `78% – 82%`)** — *Frame (`extrusion_filling`)*
  - **New scores:** `pu_foam` (**`78%`**, `Δζ=0.008`), `lead_shot` (**`79%`**, `Δζ=0.038`), `silica_sand` (**`80%`**, `Δζ=0.022`), `uhpc_grout` (**`80%`**, `η_bond=0.80`, `E=42 GPa`, `Δζ=0.025`), `epoxy_granite` (**`82%`**, `η_bond=0.95`, `E=30 GPa`, `Δζ=0.030`). Persisted `references/materials_mechanics/extrusion-cavity-fillings-composite-damping` to PrintDB.

- [x] **CAL-07 · Lumped Gantry Brackets & VzBot QA Re-Bake (`52% – 72%` → `82% – 84%`)** — *Gantry (`gantry_bracket`)*
  - **New scores:** All 8 VzBot 330 gantry brackets re-baked with mirror-symmetric stratified RBE3 sampling (`rb_max` cut `5–9×` to `0.0085–0.0107 Hz`, 11/11 QA pass → **`84%`**); PrintDB-hydrated `gantry_bracket_polymer_knockdown()` (FDM orthotropic + moisture + HDT + belt-tension creep) elevates printed brackets to **`82–84%`** (`abs 82`, `pa12_cf 82`, `pet_cf 83`, `pc_pbt_gf 83`, `ppa_cf 84`, `pps_cf 84`).

- [x] **CAL-08 · Bed Mount Hardware (`52% – 78%` → `80% – 86%`)** — *Bed / Z (`bed_mount`)*
  - **New scores:** `springs_4pt` (**`80%`**), `silicone_4pt` (**`80%`**), `silicone_4pt_clamped` (**`82%`**), `spherical_3pt_ratrig` (**`82%`**), `floating_4pt` (**`84%`**), `cnc_kelvin_magnetic` / `captive_belleville_kinematic` / `dual_leaf_ti_kinematic` / `craz_kinematic_ge5c` (**`84%`**), `kinematic_3pt` / `whopping_pochard_kinematic` / `trident_3pt_standoffs` (**`85%`**), `rigid_4pt` (**`86%`**). Added Volkersen/Timoshenko bimetallic magnetic-PEI stack + MITC4 eccentric thermal-restraint bending with M3/M4 bolt-preload slip caps calibrated to Klipper `BED_MESH_OUTPUT` cold→110 °C bands.

- [x] **CAL-09 · Frame Corner Joints (`54% – 64%` → `80% – 84%`)** — *Frame (`corner_joint`)*
  - **New scores:** `l_bracket` (**`80%`**), `corner_cube` (**`82%`**), `gusset_plate` (**`83%`**), `blind_joint` (**`84%`**) via 126k–184k-DOF 3D Tet10 submodel FEA (`scripts/calibrate_corner_joints_fea.py`) on OpenCASCADE Misumi HFS5-2020 geometry with VDI-2230 preloaded bolted contact pads vs welded controls; persisted `references/materials_mechanics/corner-joint-tet10-stiffness` to PrintDB.

- [x] **CAL-10 · Z-Drive Mechanisms (`54% – 74%` → `80% – 85%`)** — *Bed / Z (`z_drive_mechanism`)*
  - **New scores:** `gt2_9mm_worm_z` (**`80%`**), `gt2_9mm_planetary_z` (**`81%`**), `kevinakasam_belted_z` / `sfu1204_wobblex` (**`82%`**), `tr8x8_pom` / `tr8x8_brass` / `tr8x4_wobblex` (**`83%`**), `gt2_9mm_z_belt` (**`84%`**), `tr8x4_pom` / `gt2_6mm_z_belt` / `sfu1204_bk10` / `sfu1605_fk12` (**`85%`**). Wired baked `voron_24_z_drive_*` & `voron_trident_z_mount_*` CAD CMS in series + height-dependent two-span Z-belt elasticity + POM thermal softening + `compute_nozzle_tap_z_deflection_um`.

- [x] **CAL-11 · Generic Toolhead Presets — Mass & CoG (PrintDB & Onshape CAD CoG Persistence)** — *Toolhead (`toolhead_preset`)*
  - **Completed:** Persisted CAD-measured Onshape volumes/CoG & teardowns to PrintDB: `eva3-orbiter2-rapido-hf` (`113.74 cm³`, `370 g` dressed, CoG `[-0.1, -9.6, +17.5] mm`), `flsun-v400-stock-effector` (`400 g`, Onshape plate `20.42 cm³`, CoG `[0, -3.3, -18.0] mm`), `voron-stealthburner-cw2-rapido-uhf-tap` (`495 g` empirical heavy-UHF counterpart), and verified `monolith-toolhead-cpap-baseline` (`270 g` dressed, CoG `[0, -11.5, -1.5] mm`).

- [x] **CAL-12 · Hotend Structural CAD CMS (`58% – 70%` → `85% – 86%`)** — *Hotend (`hotend_clamp`)*
  - **New scores:** `mellow_goliath_air` (**`85%`**, 51.8k Tet4, upgraded from visual-only GLB), `e3d_revo_voron` (**`86%`**, 28.0k Tet4), `phaetus_rapido_2_hf` (**`86%`**, 43.9k Tet4), `phaetus_dragon_hf` (**`86%`**, 46.9k Tet4); all pass 11/11 QA (`rb_max <= 0.0075 Hz`) via `scripts/bake_hotends_cal12.py`.

- [x] **CAL-13 · 3D Pulley-Coordinate Belt Path & Idler Routing Graph (`62% – 74%` → `86% – 87%`)** — *Gantry (`belt_routing`)*
  - **New scores:** `monolith_short` (**`86%`**), `stock_corexy` (**`87%`**). Replaced scalar `0.86×`/`1.45×` factors with explicit 3D pulley coordinates (`build_corexy_belt_routing_graph_3d`), stacked belt planes (`dz = w + 3 mm`), and per-idler Euler–Bernoulli + Castigliano compliance validated against CloakedWayne & Stock Voron Trident ADXL logs.

- [x] **CAL-14 · Clip-In Enclosure Panels (`68%` → `80%`)** — *Frame (`enclosure_panels`)*
  - **New score:** Clip-in PMMA / Polycarbonate (**`80%`**). Calibrated ASTM D1056 closed-cell EPDM foam tape + ABS quarter-turn clip perimeter stiffness density (`clip_gasket_stiffness_per_m`) against same-printer panels-OFF vs clip-in 3 mm PMMA Voron 2.4r2 / Trident ADXL logs.

---

### Tier 3 — Final Polish to 80%+ (`76%`)

- [x] **CAL-15 · Hotend Thermal & Cross-WLF Melt-Flow Rheology (`76%` → `86%`)** — *Hotend (`hotend_thermal`)*
  - **New score:** **`86%`** across all hotend presets (`5.13%` MAE across 22 CNC Kitchen, Serdeczny 2020, E3D, Bondtech, Phaetus, Goliath & Chube benchmarks). Implemented 7-parameter Cross-WLF viscosity, Weissenberg–Rabinowitsch wall stress, 8-pt Gauss–Legendre cone, coupled Graetz + Osswald plug `push_force_curve`, and 3-channel CHT splitter; persisted 6 Cross-WLF polymer records to PrintDB.

---

## Suggested Parallel Agent Batches

| Batch | Parallel Agents | Tasks | Non-Overlapping Ownership |
| :--- | :--- | :--- | :--- |
| **Batch 1 (Biggest Monolith/VzBot ROI)** | `fea-engineer` (CAD) + `fea-engineer` (Belts) + `research` | **CAL-02** (Archetype/XOL/Dragon Burner STEPs) + **CAL-13** (3D pulley belt graph) + **CAL-01 / CAL-03** (Discord ADXL sweeps for heavy belts & Tap) | `cad_recipes/` & `assets/hardware/` vs. `kinematics/belts.py` vs. PrintDB / KB |
| **Batch 2 (Core Frame & Hotend CMS)** | `fea-engineer` (Joints) + `fea-engineer` (Hotends) + `thermal-extrusion-engineer` | **CAL-09** (2020 corner joint Tet10) + **CAL-12** (Dragon/Revo/Goliath CAD CMS) + **CAL-15** (Melt-flow push-force bench calibration) | `fea/components.py` vs. `assets/hardware/hotends/` vs. `thermal/` |
| **Batch 3 (Bed, Z, Feet & Rails)** | `fea-engineer` (Bed/Z) + `fea-engineer` (Rails/Feet) + `research` | **CAL-08 / CAL-10** (Bed thermal warp + Z-mount CMS) + **CAL-04 / CAL-05 / CAL-06** (Feet, rail preload, cavity fills) + **CAL-11** (PrintDB toolhead weights) | `fea/bed_plate.py` & `_builder_z.py` vs. `fea/modal.py` vs. PrintDB |
