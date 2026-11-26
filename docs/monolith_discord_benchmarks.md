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
