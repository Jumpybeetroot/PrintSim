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

- [ ] **CAL-01 · Heavy / Industrial Belts (`34% – 58%`)** — *Gantry (`belt`)*
  - **Current scores:** `at5_16mm` (`34%`), `htd3m_15mm` (`36%`), `at3_16mm` / `at3_gen3_16mm` (`38%`), `3gt_15mm` / `at3_10mm` (`40%`), `3gt_12mm` (`42%`), `3gt_9mm` (`44%`), `gt2_dual_9mm` (`52%`), `gt2_15mm` (`58%`), `gt2_12mm` (`76%`).
  - **Target:** `75–85%` (for `3GT` / `AT3` / `GT2 12–15 mm` / `Dual 9 mm`).
  - **Files:** `src/printsim/schema/catalogs.py` (`BELT_CATALOG`), `src/printsim/kinematics/belts.py`, `src/printsim/schema/overrides.py`.
  - **Recommended Agent:** `fea-engineer` + `research` (`discord-research` skill).
  - **What to do:** Gather real Monolith / VzBoT / Annex community ADXL shaker sweeps comparing 6 mm vs. 9 mm vs. 12 mm GT2, 3GT, and AT3 on the same frame; validate dynamic $EA$ multiplier, steel-cord vs. fiberglass wrap $EI_b$ bend loss over 20T–30T pulleys, and double-shear idler stiffness at `> 80 N` tension; persist verified curves to PrintDB.

- [ ] **CAL-02 · Non-STEP-Baked Toolhead Bodies (`40% – 50%`) & VzBot CNC Proxy (`62%`)** — *Toolhead (`toolhead_construction`)*
  - **Current scores:** `extruder_as_chassis` / Monolith CPAP (`40%`), `cnc_monoblock` (`42%`), `cantilever_bowden` / `plate_sandwich_cf` (`44%`), `compact_cage` / `archetype_cpap` / XOL / Dragon Burner / A4T (`48%`), `clamshell` unbaked fallback (`50%`), `vz_printhead_cnc` (`62%` — printed STEP scaled to 6061).
  - **Target:** `80–88%` via CAD Craig–Bampton `.npz` + `.glb` baking.
  - **Files:** `cad_recipes/`, `src/printsim/assets/hardware/toolheads/`, `src/printsim/cad_assets/registry.py`, `src/printsim/schema/overrides.py`.
  - **Recommended Agent:** `fea-engineer` (using `uv run printsim cad bake`).
  - **What to do:** Source official `.STEP` assemblies for **Archetype CPAP**, **XOL**, **Dragon Burner**, **A4T**, and **Monolith Toolhead** (once public on `monolith3d.xyz`) and bake Tet4 Craig–Bampton `.npz` superelements + `visual.glb` meshes. Note: switching `voron_monolith_awd.yaml` to `microwaved_bowden` + `tricorn_long` already achieves `85–89%` (`86%` overall).

- [ ] **CAL-03 · Voron Tap Z-Probe Carriage Compliance (`48% – 56%`)** — *Toolhead (`z_probe`)*
  - **Current scores:** Printed Voron Tap (`48%`), CNC Voron Tap / ChaoticLab (`56%`).
  - **Target:** `78–84%`.
  - **Files:** `src/printsim/fea/components.py`, `src/printsim/schema/catalogs.py`, `src/printsim/schema/overrides.py`.
  - **Recommended Agent:** `fea-engineer` + `research` (`discord-research` skill).
  - **What to do:** Pull same-printer before/after Klipper ADXL logs (Tap vs. Beacon/Cartographer/Klicky) from Voron/Monolith Discord to calibrate the Tap MGN9 rail + magnet preload pitch/yaw compliance knockdown; optionally bake official Voron Tap R8 STEP into a 2-port carriage superelement.

- [ ] **CAL-04 · Benchtop Feet, Anti-Vibration Pads & Concrete Paver (`48% – 62%`)** — *Frame (`frame_feet`)*
  - **Current scores:** `anti_vib_pads` HULA / Sorbothane (`48%`), `paver_foam` (`55%`), `rubber_feet` (`58%`), `suspended_lines` (`62%`).
  - **Target:** `75–82%`.
  - **Files:** `src/printsim/fea/components.py`, `src/printsim/fea/modal.py`, `src/printsim/schema/overrides.py`.
  - **Recommended Agent:** `fea-engineer` + `research`.
  - **What to do:** Calibrate HULA / Sorbothane / TPU foot axial & lateral spring rates and loss factors against published HULA / Klipper accelerometer isolation tests (including sub-30 Hz desk rocking modes) and persist shore-durometer / stiffness data to PrintDB.

- [ ] **CAL-05 · Linear Rail Grade / Clone Preload Multipliers (`48% – 68%`)** — *Gantry (`rail_quality`)*
  - **Current scores:** `budget_generic` clone (`48%`), `precision_ground` THK/NSK Grade P (`60%`), `matched` LDO/HoneyBadger/Fysetc (`68%`).
  - **Target:** `80–85%`.
  - **Files:** `src/printsim/fea/components.py`, `src/printsim/schema/catalogs.py`, `src/printsim/schema/overrides.py`.
  - **Recommended Agent:** `fea-engineer` + `research`.
  - **What to do:** Incorporate HIWIN Z0/Z1/ZA and THK/NSK radial clearance / preload tables from official catalogs into Hertzian contact stiffness scaling (replacing flat scalar multipliers), and cross-check against community rail-swap ADXL data.

---

### Tier 2 — High-Impact Structural & Motion Calibration (`52% – 68%`)

- [ ] **CAL-06 · Extrusion Cavity Fillings (`52% – 64%`)** — *Frame (`extrusion_filling`)*
  - **Current scores:** PU Foam (`52%`), Lead Shot / UHPC Grout (`58%`), Epoxy Granite (`62%`), Silica Sand (`64%`).
  - **Target:** `78–82%`.
  - **Files:** `src/printsim/schema/catalogs.py`, `src/printsim/fea/beam3d.py`, `src/printsim/fea/modal.py`, `src/printsim/schema/overrides.py`.
  - **Recommended Agent:** `fea-engineer`.
  - **What to do:** Calibrate composite bending stiffness $EI_{\text{eff}}$ and granular/viscoelastic modal damping $\zeta$ against published 2020/4040 sand-filled & epoxy-granite bar tap-test / ring-down literature.

- [ ] **CAL-07 · Lumped Gantry Brackets & VzBot CNC 6061 Proxy (`52% – 72%`)** — *Gantry (`gantry_bracket`)*
  - **Current scores:** Non-kit lumped printed brackets (`52–58%`: PA12-CF15, PET-CF15, PPS-CF, PC-PBT-GF, PPA-CF, ABS/ASA), lumped CNC 6061 (`66%`), CAD VzBot 330 AWD (`72%` — official printed STEP scaled to 6061-T6; also 4 VzBot brackets have minor `0.051–0.091 Hz` rigid-body residual vs `<0.05 Hz` QA gate).
  - **Target:** `82–90%`.
  - **Files:** `scripts/bake_vzbot_kit.py`, `src/printsim/assets/hardware/gantry_brackets/vzbot_*`, `src/printsim/fea/components.py`, `src/printsim/schema/overrides.py`.
  - **Recommended Agent:** `fea-engineer`.
  - **What to do:** Re-bake the 4 VzBot brackets with tighter solver tolerance to clear the `< 0.05 Hz` rigid-body QA gate; if public Mellow CNC 6061 STEPs or measured wall-thickness specs become available, bake true CNC geometry; calibrate lumped printed-bracket creep/modulus knockdown against PrintDB polymer HDT/creep records.

- [ ] **CAL-08 · Bed Mount Hardware (`52% – 78%`)** — *Bed / Z (`bed_mount`)*
  - **Current scores:** Yellow die springs (`52%`), VzBot torqued silicone spacers (`54%`), 70A silicone spacers (`56%`), CNC magnetic Kelvin kinematic (`64%`), 4-pt floating tabs (`66%`), 3-pt Maxwell kinematic (`70%`), Trident Bakelite + GE5C (`74%`), 4-pt rigid standoffs (`78%`).
  - **Target:** `80–86%`.
  - **Files:** `src/printsim/fea/bed_plate.py`, `src/printsim/fea/components.py`, `src/printsim/schema/overrides.py`.
  - **Recommended Agent:** `fea-engineer` + `research`.
  - **What to do:** Calibrate thermal restraint stiffness $k_{\text{therm}}$ and bimetallic bed warp against real Klipper `BED_MESH_OUTPUT` cold-vs-110 °C deltas (Trident Bakelite, Voron 2.4r2 standoffs, VzBot silicone, kinematic mounts).

- [ ] **CAL-09 · Frame Corner Joints (`54% – 64%`)** — *Frame (`corner_joint`)*
  - **Current scores:** 3-way corner cube (`54%`), internal L-bracket (`56%`), external CNC gusset (`58%`), Voron blind joint (`64%`).
  - **Target:** `80–85%`.
  - **Files:** `src/printsim/fea/components.py`, `src/printsim/fea/beam3d.py`, `src/printsim/schema/overrides.py`.
  - **Recommended Agent:** `fea-engineer`.
  - **What to do:** Run 3D contact/Tet10 submodel FEA (`scikit-fem` / `gmsh`) on 2020 blind joint vs. 5-hole CNC gusset vs. cast L-bracket vs. 3-way cube (with realistic M5 bolt preload & T-slot bearing area) to replace lumped joint rotational springs with FEA-derived $K_\theta, K_t$ curves.

- [ ] **CAL-10 · Z-Drive Mechanisms (`54% – 74%`)** — *Bed / Z (`z_drive_mechanism`)*
  - **Current scores:** 9 mm belted-Z worm (`54%`), KevinAkaSam belted-Z (`58%`), SFU1204 + WobbleX (`60%`), 9 mm belted quad-Z (`68%`), SFU1204 + BK10 (`72%`), 6 mm belted quad-Z / VzBot belt-sync dual-Z (`74%`).
  - **Target:** `80–85%`.
  - **Files:** `src/printsim/fea/components.py`, `src/printsim/geometry/_builder_z.py`, `src/printsim/schema/overrides.py`.
  - **Recommended Agent:** `fea-engineer`.
  - **What to do:** Wire baked `voron_24_z_drive_*` and `voron_trident_z_mount_*` superelements directly into the Z-drive vertical stiffness chain (currently displayed as 3D CAD stations while vertical compliance uses catalog springs), and calibrate POM nut backlash / Z-belt elasticity against nozzle-tap vertical deflection tests.

- [ ] **CAL-11 · Generic Toolhead Presets — Mass & CoG (`56% – 66%`)** — *Toolhead (`toolhead_preset`)*
  - **Current scores:** Generic/custom presets (`56–65%`, e.g., EVA 3 + Orbiter 2, FLSUN V400 effector, generic CPAP/direct-drive toolheads).
  - **Target:** `80–88%`.
  - **Files:** `src/printsim/schema/catalogs.py`, `src/printsim/schema/printdb_bridge.py`, `src/printsim/schema/overrides.py`.
  - **Recommended Agent:** `research` (`discord-research` skill) + PrintDB ingestion.
  - **What to do:** Replace estimated masses and nozzle-to-CoG offsets with scale-weighed community teardown measurements stored in PrintDB (with cited sources & verification tier).

- [ ] **CAL-12 · Analytical Hotend Structural Clamps & Goliath Air (`58% – 70%`)** — *Hotend (`hotend_clamp`)*
  - **Current scores:** `mellow_goliath_air` (`58%` — visual-only GLB, no `.npz`), generic high-flow / Volcano (`58%`), `e3d_revo_voron` (`64%`), `e3d_v6` (`68%`), `dragon_hf` (`70%`).
  - **Target:** `82–88%` (matching CAD-baked Chube / Tricorn at `86–89%`).
  - **Files:** `cad_recipes/`, `src/printsim/assets/hardware/hotends/`, `src/printsim/schema/overrides.py`.
  - **Recommended Agent:** `fea-engineer`.
  - **What to do:** Bake Tet4 Craig–Bampton `.npz` superelements from official STEPs for **Phaetus Dragon HF**, **E3D Revo Voron**, **Phaetus Rapido 2**, and **Mellow Goliath Air** (rebuilding a watertight solid heatsink/heatbreak/block for Goliath Air so it upgrades from visual-only GLB to full CAD CMS).

- [ ] **CAL-13 · Belt Path & Idler Routing (`62% – 74%`)** — *Gantry (`belt_routing`)*
  - **Current scores:** `monolith_short` (`62%`), `stock_corexy` (`74%`).
  - **Target:** `85–90%`.
  - **Files:** `src/printsim/kinematics/belts.py`, `src/printsim/geometry/_builder_gantry.py`, `src/printsim/schema/overrides.py`.
  - **Recommended Agent:** `fea-engineer`.
  - **What to do:** Replace the scalar `0.86×` Monolith span factor and lumped belt-plane Z-offset approximation with an explicit 3D pulley-coordinate belt graph (exact pulley `(x, y, z)` centers from each hardware kit's CAD stations, stacked upper/lower belt plane lever arms, and per-idler bolt cantilever compliance).

- [ ] **CAL-14 · Clip-In Enclosure Panels (`68%`)** — *Frame (`enclosure_panels`)*
  - **Current scores:** Clip-in PMMA / Polycarbonate (`68%`; bolted panels are already `86%`).
  - **Target:** `80%`.
  - **Files:** `src/printsim/fea/panels.py`, `src/printsim/schema/overrides.py`.
  - **Recommended Agent:** `fea-engineer`.
  - **What to do:** Validate clip-in foam-tape shear stiffness against same-printer panels-on vs. panels-off Klipper ADXL logs on stock Voron 2.4r2 / Trident.

---

### Tier 3 — Final Polish to 80%+ (`76%`)

- [ ] **CAL-15 · Hotend Thermal & Melt-Flow Rheology (`76%`)** — *Hotend (`hotend_thermal`)*
  - **Current scores:** `76%` across all 13 hotend presets.
  - **Target:** `84–88%`.
  - **Files:** `src/printsim/thermal/hotend.py`, `src/printsim/thermal/rheology.py`, `src/printsim/schema/overrides.py`.
  - **Recommended Agent:** `thermal-extrusion-engineer`.
  - **What to do:** Calibrate Cross-WLF / Arrhenius melt push-force and max volumetric flow (`mm³/s`) curves against CNC Kitchen / community extruder push-force bench tests (PLA, ABS/ASA, PETG, PA12-CF) across Chube, Goliath, Rapido 2, Dragon HF, and Revo, and persist curves to PrintDB.

---

## Suggested Parallel Agent Batches

| Batch | Parallel Agents | Tasks | Non-Overlapping Ownership |
| :--- | :--- | :--- | :--- |
| **Batch 1 (Biggest Monolith/VzBot ROI)** | `fea-engineer` (CAD) + `fea-engineer` (Belts) + `research` | **CAL-02** (Archetype/XOL/Dragon Burner STEPs) + **CAL-13** (3D pulley belt graph) + **CAL-01 / CAL-03** (Discord ADXL sweeps for heavy belts & Tap) | `cad_recipes/` & `assets/hardware/` vs. `kinematics/belts.py` vs. PrintDB / KB |
| **Batch 2 (Core Frame & Hotend CMS)** | `fea-engineer` (Joints) + `fea-engineer` (Hotends) + `thermal-extrusion-engineer` | **CAL-09** (2020 corner joint Tet10) + **CAL-12** (Dragon/Revo/Goliath CAD CMS) + **CAL-15** (Melt-flow push-force bench calibration) | `fea/components.py` vs. `assets/hardware/hotends/` vs. `thermal/` |
| **Batch 3 (Bed, Z, Feet & Rails)** | `fea-engineer` (Bed/Z) + `fea-engineer` (Rails/Feet) + `research` | **CAL-08 / CAL-10** (Bed thermal warp + Z-mount CMS) + **CAL-04 / CAL-05 / CAL-06** (Feet, rail preload, cavity fills) + **CAL-11** (PrintDB toolhead weights) | `fea/bed_plate.py` & `_builder_z.py` vs. `fea/modal.py` vs. PrintDB |
