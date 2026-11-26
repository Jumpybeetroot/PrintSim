# Monolith Community Empirical Engineering Benchmarks

Read-only community test data compiled from the **Monolith** Discord server (`#panels-and-frame`, `#monolith-builds`, `#updates`, `#making-it-work`, `#printed-gantry`, `#sheet-metal-gantry`) to calibrate PrintSim's structural panel, AWD belt, stepper motor, and input-shaper physics models against real-world builds.

## 1. Structural Panels — Materials, Mounting & Measured Gains
- **Consensus Sweet Spot:** **4 mm – 5 mm (3/16") Aluminum 5052-H32** laser-cut panels bolted rigidly directly to the extrusion T-slots (**no compressible foam gasket** between structural panels and extrusions, as foam decouples shear transfer).
- **Fixed Gantry Advantage (Voron Trident):** Side panels bolt directly across the fixed upper gantry beams and act as linear rail backers (`MegaMooks`, `Alx`).
- **Measured Before/After Input Shaper Gains (`Yak`, pinned in `#panels-and-frame`):**
  - Bare frame → Deck + Top + Back panels: **+5%** IS frequency
  - Adding 1 side panel: **+16%** total gain
  - Both side panels + all others: **+19%** total gain (side panels dominate Y-axis racking stiffness)
- **Material Comparison from Real Builds:**
  - **5 mm Al 5052-H32 ≈ 3 mm Mild/304 Steel** in stiffness, at **~half the weight** (~6 kg vs ~11 kg per panel). Mild steel also often arrives non-flat and can pull frames out of square.
  - **Thin ACM (3 mm):** Widely noted as offering minimal frame stiffening compared to 4–5 mm solid aluminum (*"doesn't do anything but stop air"* unless thick/composite).
  - **Polycarbonate / Acrylic:** Unsuitable for structural shear in heated chambers (PC loses ~50% strength by 65°C and has ~3× the CTE of aluminum; thin sheets drum noisily).

## 2. Belts, Tensions & Core Material Tests
- **Real 150 mm Pluck Frequencies:**
  - Standard 9 mm EPDM: **~190 Hz**
  - High-tension 9–10 mm AWD: **217–285 Hz** (`CloakedWayne`: 257–283 Hz ≈ 20 lbf / ~89 N on 10 mm belts)
- **Matched-Tension Core Shootout (`CloakedWayne` — *Road to 50k*, ~20 lbf on 10 mm belts):**
  - Gates GT2 EPDM (Fiberglass): **113.8 Hz** ZV (@ 283 Hz pluck)
  - Gates GT3 2MGT: **110.6 Hz** ZV (@ 257 Hz pluck)
  - Rubber + Aramid (Kevlar) 2GT: **119.4 Hz** ZV (@ 266 Hz pluck) (~+5% higher resonance than fiberglass at matched tension)
  - TPU + Kevlar 2GT: **120.4 Hz** ZV (@ 277 Hz pluck), but **loses tension at 60–70°C** chamber temps
  - **Steel-Core Belts:** Strongly avoided on 20T pulleys/idlers due to tight bend radius and rapid fatigue fracture.
- **Controlled Same-Machine Belt-Path & Motor Tests (`CloakedWayne`, `#updates`, 300 mm testbed: 4040/2040 frame, Carbonara, `LDO 2504 S45R` @ 36 V, `10 mm` Gates EPDM @ `~20 lbf`, Beacon, room temp):**
  - **AWD vs 2WD, same machine and day (`2025-07-10`, `2.5 A`):** AWD X ZV **`178.6 Hz`** / MZV `178.4 Hz`, Y ZV **`114.4 Hz`** / MZV `117.4 Hz`; 2WD X ZV **`129.0 Hz`** / MZV `127.8 Hz`, Y ZV **`83.2 Hz`** / MZV `84.6 Hz`. Frequency ratio **`1.38`** on both axes ("average driven belt length is just a tad over 2x compared to AWD … 90+% scaling").
  - **Stepper run-current sweep, AWD Y axis (`2025-08-12`):** ZV / MZV **`97.4 / 98.8 Hz`** @ `1.0 A`, **`104.0 / 105.4`** @ `1.5 A`, **`107.6 / 111.0`** @ `2.0 A`, **`110.4 / 113.8`** @ `2.5 A`, **`111.6 / 115.6`** @ `2.8 A`. `1/f²` is linear in `1/I`; the motor magnetic spring is about `19%` of Y compliance at `2.5 A`.
  - **Notes from the same author:** catalogue belt stiffness is too low to reach measured frequencies even with everything else rigid; input-shaper performance scales about linearly with belt width for otherwise identical belts; branch stiffness varies more across the bed in AWD because the toolhead gets closer to the steppers; EPDM belts are stiffest when new.
  - Used in PrintSim to correct the 2WD CoreXY belt routing (two about-equal perimeter branches) and to set the Gates fiberglass dynamic belt ratio to `1.85` (`docs/physics/07_mounting_and_belt_refinements.md` §5.2).

## 3. Motors, Drivers & Failure Modes
- **48V + TMC5160 / BTT Kraken:** Standard AWD setup with **LDO 42STH48-2504AH (S45R)** (~2.5 A @ 48V) or **Kraken steppers** (~1.8–2.0 A @ 48V); long 42–45 mm shafts required for double-shear 9 mm stacks.
- **Printed Mount Creep:** Standard ABS/ASA and even annealed PA-CF creep under sustained high belt tension; builders step up to **annealed PPA-CF/GF, PPS-CF/GF, PET-CF**, or **CNC/SLM aluminum**.
- **Shake&Tune / Klipper Notes:** Stock Klipper caps shaper calibration at 150 Hz (stiff AWD builds use dynamic >150 Hz scans); Shake&Tune had a known `peak / sqrt(2)` vs `peak / 2` PSD damping-ratio quirk.

## 4. Measured Real-World Input Shaper Benchmarks

| Build | Frame | Drive & Belts | Gantry / Panels | Measured X Shaper | Measured Y Shaper |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hanarius** (VT 250) | 2020 Trident 250 | 2WD, 6 mm | Printed Monolith | MZV @ **86.8 Hz** (≤22k) | MZV @ **56.8 Hz** (≤9.4k) |
| **onewayne** (Doomish) | 4040+2020 Trident | 2WD/AWD, 9 mm | 4 mm Al + PIR panels | MZV @ **100.4 Hz** (29.7k) | MZV @ **67.6 Hz** (13.5k) |
| **Stas_Robotmaker** (HM350) | 3030 350³ | 2WD, 9 mm @ 190 Hz | Laser-cut gantry | MZV (~27k) | `smooth_zv` (~9k) |
| **Beattune** | Stiff CoreXY | AWD, 9–10 mm (217–285 Hz) | Kalico >150 Hz scan | Raw **~136 Hz** (46k) | Raw **~83 Hz** |
| **CloakedWayne** (300mm) | Stiff 300mm | AWD, 10 mm | Monolith testbed | MZV @ **175 Hz** | ZV @ **115 Hz** |
| **dderg** (VT 300 AWD) | 4040 Trident 300 | AWD, 12 mm, 48V 5160Plus | K3Z SLM milled gantry | `smooth_mzv` @ **186 Hz** (50k) | `smooth_zv` @ **122 Hz** (50k) |

**Key Calibration Rule of Thumb:** Across real CoreXY builds, **X resonance is consistently ~1.4×–1.6× higher than Y resonance** (toolhead-only mass vs full X-beam + rail + carriage mass), and **side structural panels (4–5 mm Al 5052 or 3 mm steel)** provide ~+16–19% real-world IS frequency gain.

## 5. Scale-Measured CNC AWD XY-Joint & Moving Mass Benchmarks (0.1 g Scale)
Source: `#updates` — *"The XY-Joint Weigh-In - The Road to 50k"* & *"A 750g Comparison"* (`CloakedWayne`, `#ldo-gantry-kits`, `#monolith-builds`):

| Component / Assembly | Configuration | Each (g) | Pair (g) |
| :--- | :--- | :---: | :---: |
| **LDO MGN9H carriage** | Bare block (no fasteners) | 25.2 | 50.4 |
| **LDO MGN12H carriage** | Bare block (no fasteners) | 48.3 | 96.6 |
| **LDO Monolith CNC 6061 XY-Joint** | Assembled w/ stock live-shaft idlers (minus X-beam fasteners) | 75.9 | **151.8** |
| **LDO CNC XY-Joint + MGN9H** | Dressed pair (add ~8 g X-beam fasteners → ~210 g) | 101.1 | **202.2** |
| **LDO CNC XY-Joint + MGN12H** | Dressed pair (add ~8 g X-beam fasteners → **~256 g**) | 124.2 | **248.4** |
| **SLM Al MGN12H XY-Joint + MGN12H** | Steel hardware (minus X-beam fasteners) | 96.2 | **192.4** |
| **dderg Lightweight Milled + MGN9H** | Ti hardware + hollow Ti live-shaft (−67.6 g/pair vs. LDO) | 67.3 | **134.6** |
| **Carbonara 350 mm CF X-Beam** | Bare 350 mm CF beam (`Eric Zimmerman`, 0.01 g scale) | 84.3 | — |
| **Complete LDO CNC AWD Moving Y-Gantry** | 300 mm AWD, LDO CNC + Carbonara + Monolith TH (`CloakedWayne`) | **~750 g** | *(~682 g w/ dderg XY-joints)* |

## 6. Carbonara Carbon Fiber X-Beam & Two-Wall Clamped Internal Inserts
Source: `#user-projects` thread **`[Read pins] Carbonara questions and help`** (`Brazuka`, `CloakedWayne`, `Scarecrow`, `Denald`, `HVFDesigns`, `Ingstronmentalist`, `Panzarkatten`, `count_mm`, `dderg`):

- **Tube Geometry, Manufacturer & Thermal Spec:**
  - **Manufacturer**: Gen 1 Carbonara was co-developed by **Brazuka** and **Carbontime** (higher-QC flat-wall 2 mm spec vs generic older Carbontime 1.5 mm "noodle" tubes); **Carbonara Gen 2** (`10–11/2026`) moved to a **new manufacturer** targeting even higher temperature capability.
  - **Exact CAD geometry (`Carbonara_assembly_v7.STEP`, `github.com/Brazuka-1/Carbonara_CF_tube`):** **20 × 20 mm OD**, **16 × 16 mm ID**, **2.0 mm nominal wall** (`144.0 mm²` nominal solid cross-section; measured user variance `1.75–1.90 mm`, matching `Eric Zimmerman`'s `64.7 / 74.5 / 84.3 g` at `330 / 380 / 430 mm` = `196.1 g/m`; `~58.8%` lighter than bare 2020 Al extrusion at `~476 g/m`).
  - **Quantitative Thermal Torsion Bench Test (`The_Adeo`, `monolith-gantry-updates.zen3d.eu`, Test Logs 1 & 2, `4/14/25` & `5/30/25`):**
    - Prototype resin datasheet: **`125 °C` onset $T_g$** (rated `90 °C` continuous chamber / `110 °C` short-term; endurance tested `800 hrs` at `500–1300 mm/s`, `50k–100k mm/s²`, `80 °C` chamber).
    - Quantitative torsion test (`20 cm` sample, `6 kg @ 17 cm` ≈ `10.0 N·m` off-axis load): **`+2.4°` at Ambient**, **`+2.4°` after 10 hours at `100 °C`** (`100%` stiffness retention!), and **`+2.6°` after 2 hours at `110 °C`** (`92.3%` stiffness retention!).
  - **Fiber layup note:** Smooth 3K carbon fiber exterior; exact fiber grade & ply angles remain proprietary/undisclosed in the thread and README.
- **Two-Wall Clamped Internal Printed Inserts (`cf_rail_mounting = "two_wall_clamped_inserts"`):**
  - **Why single-wall bare CF fails:** Bolting a linear rail through only one thin CF wall causes inward wall bowing/oil-canning under roll moments and rapid nut pull-through as resin softens (`CloakedWayne`: *"only using one wall for rail mounting will make this worse than a 2020"*, *"A single wall mounting should always fail at a lower temp"*).
  - **Exact `Carbonara_assembly_v7.STEP` mounting architecture:** Individual snug **7.0 × 16.0 × 16.0 mm** (`1,293.7 mm³` each) internal 3D-printed rail inserts spaced every **25.0 mm** (`12 / 14 / 16×` for 250 / 300 / 350 mm; **100% infill** annealed **PET-CF / PC-CF / HTPA-CF / PC-PBT-GF**, HDT ≥ `110 °C`, indented side facing linear rail, optional `18 mm` hollow alignment spacers between them) + **M3×28 Grade 5 Ti bolts** (`2.0–2.5 N·m`, `Ø10 × 1 mm` washers + M3 nuts; Grade 2 Ti sheared at `1.76–2.0 N·m`) clamping **rail → front CF wall → solid compressed insert → rear CF wall**, engaging **2 CF walls (`~3.6–4.0 mm` total bearing)** and directing clamping shear into the tube corners ("stress concentrators"). No glue/epoxy is used.
  - **XY-joint end mounting & scale weights:** **31.0 × 16.0 × 16.0 mm** (`5,647.2 mm³` each) printed internal end inserts housing female–female **M5×15 hex standoffs** (`7 mm` or `8 mm` across flats; Beta v7 switches to aluminium hex standoff + centering stop) + M5 nut. Scale-weighed printed insert + standoff kit (`Denald`, 97% annealed PC-CF): **`16 g`** (Al standoff + Ti nut) to **`20 g`** (brass standoff); complete 300 mm Carbonara BOM (`Brazuka` / `Denald`, tube + Ti bolts + inserts + standoffs): **`121 g`**.
- **Controlled Before/After Klipper ADXL345 Benchmark (`HVFDesigns`, Siboor printer, `9/6/26 → 9/8/26`, 2020 + milled joints → Carbonara):**
  - **Y-Axis:** $\omega_0 = 61.2 \to \mathbf{65.8\text{ Hz}}$ (`+7.5%`), $\zeta = 0.044 \to \mathbf{0.063}$ (`+43%` damping), MZV `62.2 → 65.0 Hz` (`11,300 → 12,360 mm/s²`), ZV `63.4 → 68.6 Hz` (`15,480 → 18,180 mm/s²`).
  - **X-Axis:** $\omega_0 = 84.2 \to \mathbf{88.8\text{ Hz}}$ (`+5.5%`), $\zeta = 0.034 \to \mathbf{0.041}$ (`+21%` damping), MZV `85.8 → 89.4 Hz` (`21,480 → 23,330 mm/s²`), ZV `87.6 → 91.4 Hz` (`29,520 → 32,160 mm/s²`).

## 7. Official Website & Monolith Builder Benchmarks (`monolith3d.xyz` & `build.monolith3d.xyz`)
Source: 195 public development updates (`Mar 2025 – Nov 2026`, `CloakedWayne`, `Adeo`, `Zruncho`, `dderg`, `Brazuka`, `FrankRamm`) and live Monolith Builder BOMs:

- **Scale-Measured 10 mm Belt Pairs (`2 × 1658 mm = 3.316 m` at `~20 lbf` / `89 N`):**
  - `Gates GT2 EPDM 10 mm`: `41.0 g` (`12.36 g/m`), `283 Hz` pluck → **`113.8 Hz` ZV**
  - `Gates GT3 2MGT 10 mm`: `49.7 g` (`14.99 g/m`), `257 Hz` pluck → **`110.6 Hz` ZV** (softer base elastomer increases elongation needed & rolling drag)
  - `2GT 10 mm TPU + Kevlar`: `42.7 g` (`12.88 g/m`), `277 Hz` pluck → **`120.4 Hz` ZV** (loses tension at `60–70 °C`)
  - `2GT 10 mm Rubber + Aramid`: `46.4 g` (`13.99 g/m`), `266 Hz` pluck → **`119.4 Hz` ZV**
- **Thermal Press-Fit & Stepper Gasket Tests:**
  - `F695` steel bearing (`Ø13 mm` OD) in 6061 Al housing (`P6` fit, `26 µm` max interference) drops to zero interference above **`~66 °C`** (bearing reaches `~100 °C` in `80 °C` chamber); `M5` Belleville disc springs (`10×5.2×0.25 mm`) restore axial preload across thermal cycles.
  - `2 mm` PTFE stepper thermal isolation gasket (`25 W` / `115 °C` stepper in `80 °C` chamber) reduces mount surface temp from **`109.1 °C max / 104.1 °C avg` → `95.7 °C max / 90.0 °C avg`** (`-13.4 °C` peak / `-14.1 °C` avg).
- **CPAP Blowers, Monolith Toolhead & Chamber Heaters:**
  - Ducted CPAP flow: `WS7040-24-V200` (`5.8 dm³/s`, `+5.5 °C` exhaust heating @ 100% PWM), `WM9290-24V` (`9.2 dm³/s`, throttles to `1.0 dm³/s` @ 25%, quietest), `WM7575-220VAC` (`11.6 dm³/s`, loud HF); `15 mm` HT hose vs `18 mm` ID (`~20%` worst-case drop, lower mass/stiffness).
  - Monolith Toolhead (`37:1` double-start worm extruder, `rotation_distance: 55`, belt-coplanar CoG centered on `MGN12H` raceways) + `50 mm`+V6 HF hotend (`80 W` + `PT1000`): **`83 mm³/s`** LDO ABS (`300 °C` nozzle, `85 °C` chamber, `0.5 A`, `~90%` PWM).
  - Active SLM PTC Chamber Heater kits: `100 W 24V` (`98×32 mm` PTC + `2× 3628` fans) and `500 W AC` (`140×50 mm` PTC + `2× 5032` fans) paired with `10/20 mm` PIR insulation.
- **Monolith Zero (`M0`) & Fixed vs. Flying Parity:**
  - `M0` (`1515` `1.1 mm` wall frame, `6 mm` belts @ `10 lbf`, `500–600 g` moving mass): unsupported bare `MGN12` X-rail hits **`54k–63k` MZV** (`59k–62k` w/ `5 mm` backer; 2WD `30–40k` MZV Y).
  - Matched `750 g` 300 mm comparison: **Fixed gantry (`2040` Y)** vs. **Flying gantry (`2020` Y + `2 mm` SS backers + double SLM Z-joints)** achieved identical IS frequencies within run-to-run variance (`~37–41k` MZV baseline).
  - `Cra-Z` (`2+2+2` double-wishbone `GE5C` gravity-preloaded) kinematic Z suspension drop-in for 3-screw Trident & 4-belt fixed V2.

## 7. BFE (Big Fucking Extruder) — Dual-NEMA17 Dual-LGX/HGX Bowden Extruder
Source: `#user-projects` thread **`BFE (Big Fucking Extruder)`** (`1567989441379700766`, 139 messages, `2026-11-05` to `2026-11-25`, OP `Microwave` / `micreathonn`; pinned GitHub repo [`github.com/micreathonn/BFE`](https://github.com/micreathonn/BFE), PrintDB ID `components/extruders/microwaved-bfe`):

- **Architecture & CAD Geometry (`CAD/BFEV1.STEP`, 66 solids):**
  - **Envelope:** `114.9 × 102.5 × 80.9 mm`; two 48 mm NEMA17 motors mounted side-by-side at `43.0 mm` center-to-center (`X = ±21.5 mm`).
  - **Geartrain:** 2× **17T stainless steel motor hobs** (Mod 0.5, `Ø9.9 mm` OD; stainless required when pairing with metal LGX gears; AliExpress `1005006755339441`) directly driving 2× synchronized pairs of **37T LGX (preferred) or HGX large drive gears** (`Ø19.5 mm` OD, `13.50 mm` pinion-to-gear C-C, `18.60 mm` meshed pair C-C) → exact gear ratio **`37:17` (`2.1765:1`)** (`~25.3 mm/motor-rev` at `~17.5 mm` effective LGX wheel diameter, keeping high-acceleration Bowden PA velocity spikes well inside the flat torque band of NEMA17 steppers).
  - **Equal-Tension Double-Shear Mechanics:** Two main printed housing pieces (**Motor Plate** `30.1 cm³` + **Main Body** `26.3 cm³`) plus 2× **Idler Arms** (`2.28 / 2.30 cm³`) and printed thumbscrew knob (`2.26 cm³`, `63.2 cm³` total printed volume). Static LGX gears and idler arms assemble onto the Motor Plate first to set backlash, then the Main Body caps and double-shears all pins (`2× M3×24 mm`, `4× M3×16 mm`). One shared **`M3×50 mm` SHCS + `3 mm` ID `8–12 mm` spring** tensions both idlers equally for synchronous dual-stage drive. Note: thinner budget AliExpress HGX/LGX gears (`3256808239002515`) require axial shims; do not mix LGX and HGX gear pairs.
  - **Bowden Interface & Cooling:** `2× PC4-M10` couplers + `1× 23 mm` internal `1.9 mm` ID Capricorn PTFE guide; optional extruder-side screw-tensioned Bowden sheath (toolhead side uses rigid mount, retaining string loops for Colphaer sheath). Mounted outside the heated chamber with motor heatsinks + `6015` fan (gears stay cool).
  - **CAD-Derived Mass Estimate:** `~175 g` excl. motors (`~50–66 g` ABS/ASA printed parts + `~74 g` 4× steel LGX gears + `~8.4 g` 2× stainless hobs + `~37.5 g` hardware/couplers) → **`~895 g`** with 2× Moons `MS17HD6P420I-04` (`360 g` each) and **`~950–1000 g`** with heatsinks + 6015 fan (consistent with `Microwave`'s *"weigh 1 kg"* note; **0 g** added to moving gantry/toolhead mass since frame-mounted).
  - **BOM Cost:** `~$50` (budget HGX gears + generic NEMA17s) to `~$140` (steel LGX gears + Moons NEMA17s); reputable motors recommended (`Moons / Wantai / OMC / LDO`).
- **Empirical Volumetric Flow, Grip & High-Accel Bowden Benchmarks (`Microwave`, 350 mm Voron Trident + MBT Tricorn):**
  - **Max Flow / Grip Limit (`ABS @ 300 °C`, `2.0 A` per NEMA17):** Starts skipping/grinding only at **`100 mm³/s`** (vs `~65 mm³/s` on DiodeKing's extruder and `50–60 mm³/s` on Boombox). Failure mode is **filament grinding** (dual-LGX grip `~130 N` est.), not motor stall (dual NEMA17 torque headroom `~150–200 N`).
  - **High-Speed Bowden Print Tests (`350 mm` Trident, ABS & PET-GF):**
    - Test cube 1 (ABS): `500 mm/s`, `25,000 mm/s²`, `PA = 0.08 s`, `smooth_time = 0.02 s` — crisp corners.
    - Test cube 2 (ABS): `500 mm/s`, `40,000 mm/s²` — extruder keeps up cleanly.
    - Dial-in runs (ABS): `500 mm/s`, **`50,000 mm/s²`**, **`20 SCV`** (`25k` top layers), `PA = 0.09 s` / `0.02 s` smooth time → refined to `PA = 0.085 s`, `smooth_time = 0.01 s` (also tested `PA ≈ 0.12 s @ 25k` and half bridging PA in OrcaSlicer; rebuild with tighter backlash eliminated sawtooth PA artifact).
    - Functional parts (PET-GF PSU enclosures): `500 mm/s`, `30,000 mm/s²` everything.
  - **Mounting & Routing Guidance:** Mount **top center** on panel/frame (5 mm steel panel alone easily supports BFE without extra extrusions), angled backwards or forwards to ease PTFE bend radius. Stock Trident `~80 mm` headroom above gantry works, though **`+50 mm` extra top clearance (`~130 mm` total)** is ideal for Bowden bend radius. Keep Bowden tube as short as possible (critical for PA); CPAP hose length is less sensitive and can be routed cleanly alongside.

## 8. CW's Fixed Gantry Conversion (`MONOLITH 300`) & Simpla Isolation Feet
Source: `#members-projects` thread **`CW's Fixed Gantry Conversion`** (`1564391176369606726`, 184 messages, `2026-10-26` to `2026-11-24`, OP `CloakedWayne`, plus `zruncho` M0 foot tests; preset [`examples/cw_fixed_gantry_300.yaml`](../examples/cw_fixed_gantry_300.yaml), PrintDB `assemblies/printers/monolith-cw-fixed-gantry-300`):

- **Machine Architecture (`MONOLITH 300` — *Road to 50k* Testbed):**
  - Converted from a **Voron V2.4 300 LDO kit** (`300×300 mm` bed / `305 mm` PEI sheet retained; Mainsail `X=150`, `Y=298`) to a **fixed-gantry AWD CoreXY** at *"standard Trident height −20 mm insulation"* with the entire electronics bay (**BTT Octopus Pro** + **Raspberry Pi**) relocated **above the gantry** behind a perforated top front cover and filament spools stored inside the heated chamber.
  - **Thermal Performance (Mainsail log, `2026-11-10`):** **`19.5 °C → 80.0 °C` chamber in 9 minutes** with **20 mm foil-faced PIR insulation** and active PTC chamber heating (`Heater Fins` thermistor; snapshot at 9 min: Bed `99.4/100.0 °C @ 36%`, Chamber `80.0/80.0 °C @ 78%`, Extruder `150.0 °C @ 10%`, Beacon Coil `70.9 °C`).
  - **Gantry, Toolhead & Motion:** **Carbonara 20×20 CF X-beam** + front-mounted rail, **inner-mounted Y-rails**, metallic **CNC/SLM XY-joints**, metallic **SLM/CNC CPAP toolhead** + **Beacon** eddy probe + **WS9290 CPAP** blower (`40–50%` PWM + Mellow pot intake filter, front-extrusion umbilical chain), printing functional parts at **`50,000 mm/s²` accel / `50 mm/s` SCV**.
  - **Experimental `Cra-Z` Suspension:** Vertical Z-rails + vertical leadscrews with top-mounted Z steppers driving curved **GE5C double-wishbone arms** to the bed; CW notes the current linkage still exhibits an **~5× stiffness ratio** between stiffest and weakest directions and high sensitivity to rod length.
- **Simpla Isolation Feet (`Isolation foot.step`, 481 KB, pinned):**
  - Replaces Voron BOM rubber feet with **2 printed parts/corner** + **`F8-22M` (`8×22×7 mm`) thrust bearing** flipped for lateral seismic isolation + **`3070` natural latex tubing (`3 mm ID × 7 mm OD`, `~119 mm`/foot)** + **`45×38×3.5 mm` O-ring** base (`~25%` perceived noise reduction by CW ear test).
  - **`zruncho` Monolith Zero (`M0`) ADXL Benchmarks (high-CG narrow printer comparison):**
    - `2026-05-25` M0 2WD (`OMC 2504 @ 3.3 A`, baseline before feet): **X `MZV 142.2 Hz` (`≤59.6k`)** / `ZV 144.6 Hz` (`81.5k`); **Y `ZV 101.0 Hz` (`≤39.8k`)** / `MZV 104.0 Hz` (`31.9k`).
    - `2025-10-04` M0 AWD (`4× 2504 @ 2.5 A`, old gantry): **X `MZV 185.2 Hz` (`≤101.0k`)**; **Y `MZV 132.0 Hz` (`≤51.3k`)**.
    - `2026-11-23` M0 2WD (`FOC + Simpla feet`): tall/top-heavy M0 exhibited coupled frame-rocking split mode on Y (`~86 Hz` + `~235 Hz`, `MZV 62.6 Hz ≤11.5k` / `3HUMP_EI 174.6 Hz ≤22.3k`) while X shifted up (`ZV 243.4 Hz` / `MZV 253.0 Hz ≤188.6k`), confirming that low-lateral-stiffness isolation feet can couple with tall, top-heavy printers (`Sayomi` saw zero IS shift on a wider Trident).
