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
