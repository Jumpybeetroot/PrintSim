# Community 3D Printer Engineering Touchstones

Curated, source-cited real-world empirical baselines compiled from **Klipper Discourse**, **Voron Design** community shaper logs, **RatRig** / **VzBot** build benchmarks, **Armchair Heavy Industries** toolhead repos, and **MyTechFun / Bambu Lab / Siraya Tech / Polymaker / MatWeb** material technical datasheets (TDS).

These touchstones complement `docs/monolith_discord_benchmarks.md` by filling baseline coverage across stock Voron Trident, Voron 2.4 flying-gantry, 3030 9 mm CoreXY, VzBot AWD, Cartesian bedslinger, and Delta printers, alongside scale-weighed component masses and structural polymer TDS values.

## 1. Measured Community Shaper Baselines

> **Provenance note (2026-11-26).** The five "Stock Voron" rows below are curated typical values,
> not individual measurements, and their X / Y ratios (1.14–1.21) are inconsistent with the
> moving-mass ratio of a stock Voron (about 2.1, giving X / Y near 1.4–1.6). Individually measured
> stock Tridents from the VORONDesign resonance forum read lower on Y:
>
> | Machine | Belt tension | X | Y |
> | :--- | :--- | :--- | :--- |
> | Voron Trident 300, LDO 2004MAH (`jeff.k`), hot chamber | 2.2 lb (PF Makes meter), about 9.8 N | MZV 61.4–64.0 Hz | MZV 40.8–42.2 Hz |
> | Same machine, cold 26 °C | 1.4 lb, about 6.2 N | MZV 55.4 Hz | MZV 34.4 Hz |
> | Formbot Trident R2 + Tap (`AchillesUltimate`), raw CSVs in `data/touchstones/adxl_csvs/` | not stated | peak 50.3 Hz | peak 36.1–37.7 Hz |
>
> Use the measured rows for calibration. The stock Voron presets (`voron_trident_300`,
> `corexy_voron_style`, `ldo_voron_trident_revd_300`, `ldo_voron_24_revd_300`) ship the
> Voron-documented belt tension, 9 N (110 Hz pluck on a 150 mm span). At that tension
> `voron_trident_300` gives MZV 62.8 Hz on X and 44.2 Hz on Y. Before 2026-11-26 they ran 28 N.
> The generic 6 mm starters `corexy_flying_gantry_300` and `corexy_fixed_gantry_300` use the same 9 N.

| Build | Frame | Drive & Belts | Gantry / Panels | Measured X Shaper | Measured Y Shaper |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stock Voron Trident 300** (Baseline) | 2020 Trident 300 | 2WD, 6 mm Gates 2GT @ 110 Hz pluck | StealthBurner + CW2, clip-in acrylic + bolted bottom deck | MZV @ **52.5 Hz** (<=8000) | MZV @ **46.0 Hz** (<=6200) |
| **Stock Voron Trident 300** (LDO Kit) | 2020 Trident 300 | 2WD, 6 mm Gates 2GT @ 110 Hz pluck | StealthBurner + CW2, LDO stainless rails, clip-in acrylic | MZV @ **56.0 Hz** (<=9200) | MZV @ **48.2 Hz** (<=6800) |
| **Stock Voron 2.4 300** (Flying Gantry) | 2020 Voron 2.4 300 Flying Gantry | 2WD, 6 mm Gates 2GT @ 110 Hz pluck | StealthBurner + CW2, clip-in acrylic enclosure | MZV @ **58.4 Hz** (<=10000) | MZV @ **48.2 Hz** (<=6800) |
| **Stock Voron 2.4 350** (Flying Gantry) | 2020 Voron 2.4 350 Flying Gantry | 2WD, 6 mm Gates 2GT @ 110 Hz pluck | StealthBurner + CW2, clip-in acrylic enclosure | MZV @ **51.0 Hz** (<=7600) | MZV @ **41.8 Hz** (<=5100) |
| **Voron 2.4 300 Lightweight Toolhead** | 2020 Voron 2.4 300 Flying Gantry | 2WD, 6 mm Gates 2GT @ 110 Hz pluck | XOL 2 (~250g) lightweight toolhead, clip-in acrylic | MZV @ **66.2 Hz** (<=12900) | MZV @ **51.4 Hz** (<=7800) |
| **RatRig V-Core 3.1 300** | 3030 RatRig V-Core 3.1 300 | 2WD, 9 mm Gates 2GT | EVA 3 toolhead, 3030 cast corner brackets | MZV @ **74.0 Hz** (<=16100) | MZV @ **54.0 Hz** (<=8600) |
| **RatRig V-Core 3.1 400** | 3030 RatRig V-Core 3.1 400 | 2WD, 9 mm Gates 2GT | EVA 3 toolhead, 3030 frame with enclosure | MZV @ **64.0 Hz** (<=12000) | MZV @ **46.5 Hz** (<=6400) |
| **VzBot 330 AWD** (Mellow CNC kit) | 2040 uprights + 2020 VzBot 330 (530×500×570) | AWD 4× LDO 2804AC @ 48 V, 6 mm GT2 @ 150–189 Hz / 150 mm | 20×20×1 CF tube + top MGN9H, Vz-Printhead CNC + HextrudORT + Goliath + Beacon | ZV @ **118.0 Hz** | MZV @ **77.0 Hz** |
| **VzBot 330 AWD** (Beacon, Klipper cfg) | 2040 + 2020 VzBot 330 | AWD, 6 mm GT2 | CF tube + MGN9H, Vz-Printhead | MZV @ **107.4 Hz** | MZV @ **76.0 Hz** |
| **VzBot 330 AWD** (eddy toolhead) | 2040 + 2020 VzBot 330 | AWD, 6 mm GT2 | CF tube + MGN9H, Beacon → SHT36 eddy | MZV @ **75.6 Hz** | MZV @ **47.0 Hz** |
| **VzBot 330 2WD** (Klicky) | 2040 + 2020 VzBot 330 | 2WD, 6 mm GT2 | CF tube + MGN9H, Klicky probe | ZV @ **83.8 Hz** | ZV @ **67.8 Hz** |
| **VzBot 235 AWD** (andersevenrud) | 2040 + 2020 VzBot 235 | AWD, 6 mm GT2 | CF tube + MGN9H, Vz-Printhead | MZV @ **110.0 Hz** | MZV @ **84.8 Hz** |
| **VzBot 235 AWD** (water-cooled) | 2040 + 2020 VzBot 235 | AWD, 6 mm GT2 | CF tube + MGN9H, water-cooled Goliath | MZV @ **149.8 Hz** | MZV @ **85.6 Hz** |
| **Ender 3 V2 Klipper** (Bedslinger) | 2020 / 4020 V-Slot Bedslinger | 2WD, 6 mm GT2 | Stock gantry + moving glass/PEI Y bed | MZV @ **56.0 Hz** (<=9200) | EI @ **37.5 Hz** (<=3400) |
| **Prusa MK4S Input Shaper** (Bedslinger) | 3030 / Sheet Metal Bedslinger | 2WD, 6 mm GT2 | Nextruder toolhead + moving Y bed carriage | MZV @ **54.0 Hz** (<=8600) | MZV @ **42.0 Hz** (<=5200) |
| **FLSUN V400 Klipper** (Delta) | 3-Column Aluminum Delta | 6 mm–10 mm GT2 Delta | Carbon effector arms, direct-drive effector | MZV @ **68.0 Hz** (<=13600) | MZV @ **67.0 Hz** (<=13200) |
| **Doron Velta** (Delta) | 2020 3-Tower Delta | 6 mm GT2 Delta | Lightweight carbon arms + compact effector | MZV @ **88.0 Hz** (<=22800) | MZV @ **86.0 Hz** (<=21800) |

**Sources:**
- Voron Trident / Voron 2.4 baselines: Voron Design community input-shaper logs and Klipper Discourse `SHAPER_CALIBRATE` threads.
- RatRig V-Core 3.1 baselines: RatRig community documentation and Shake&Tune calibration archives.
- VzBot 330 / 235 rows: `[input_shaper]` sections of published Klipper configs (no official VzBoT shaper values exist; earlier uncited 128 / 82 Hz row removed 2026-11-26): [Travis90x VzBot_330_Mellow](https://github.com/Travis90x/Klipper-configuration/blob/HEAD/config/PRINTER_%26_START_CONFIG_examples/VzBot_330_Mellow_printer.cfg) (ZV 118 / MZV 77, 20k accel), [techyg/VzBot](https://github.com/techyg/VzBot/blob/HEAD/Config/printer.cfg) (MZV 107.4 / 76.0), [keefo/vzbot](https://github.com/keefo/vzbot/blob/HEAD/new_config/printer.cfg) (MZV 75.6 / 47.0), [FlashSavage/VZbot-config](https://github.com/FlashSavage/VZbot-config/blob/HEAD/printer_data/config/printer.cfg) (2WD ZV 83.8 / 67.8), [ckvsoft/KlipperConfigs](https://github.com/ckvsoft/KlipperConfigs/tree/HEAD/VzBot330) (2WD MZV 82.4 / 58.2), andersevenrud/my-vz235 & lukascechovic (235 AWD). Hardware per the official [VzBoT-Vz330 BOM](https://github.com/VzBoT3D/VzBoT-Vz330/tree/master/BOM), `Vz-330_v1.2` full-printer CAD and [belt tensioning guide](https://docs.vzbot.org/general/belt-tensioning). Spread is large: 330 AWD X ≈ 75–118 Hz, Y ≈ 47–80 Hz, Y always lower.
- Cartesian Bedslinger & Delta baselines: Klipper Discourse `SHAPER_CALIBRATE` threads, Prusa MK4S official firmware shaper defaults, FLSUN V400 Klipper community logs, and Doron Velta open-source delta benchmarks.

## 2. Verified Component Weights

| Category | Component | Mass (g) | Source / Notes |
| :--- | :--- | :--- | :--- |
| toolhead | StealthBurner + CW2 + Dragon HF | 415 | Voron Design standard ABS toolhead with Clockwork 2, 5015/4010 fans, LEDs, and Dragon HF |
| toolhead | XOL 2 + Sherpa Mini + Rapido | 250 | Armchair Heavy Industries Xol-Toolhead repo / community scale-weighed assembly |
| toolhead | A4T + Wristwatch G2 + Dragon | 250 | Armchair Heavy Industries A4T repo / community scale-weighed assembly |
| toolhead | Dragon Burner + Orbiter 2 + Dragon | 245 | Chirpy2605 Dragon Burner community scale-weighed build |
| toolhead | Archetype CPAP + Goliath Air | 235 | Armchair Heavy Industries Archetype CPAP lightweight remote-cooling toolhead |
| toolhead | El Fuego SLM (Aire Fuego / Agua Fuego) + Sherpa Micro + Chube Conduction | 352 | KhaymKhaym Monolith Discord release STLs (`83.75 g` SLM AlSi10Mg main body + `6.87 g` Al belt plate [`19.97 g` steel] + `3.62 g` Ti extruder plate = `94.24 g` bare CAD; `262 g` excl. hotend w/ MGN12H + Sherpa Micro + Beacon; GOM Scan 1 `<=20 µm` contact faces; Aire Fuego `2.5 mm` CPAP bleed cooling) |
| carriage | Stock Voron ABS XY-Joints (pair) | 125 | Voron Trident / 2.4 printed ABS XY-joints with F695 bearings and M5 hardware |
| carriage | CNC 6061 AWD XY-Joints (pair) | 195 | CNC 6061-T6 billet AWD XY-joint pair with double-shear idler hardware |
| carriage | MGN12H Carriage Block + Printed Shuttle | 115 | MGN12H stainless carriage block (~65 g) + printed ABS/CF X-carriage shuttle (~50 g) |
| gantry | 2020 Aluminum 350mm + MGN12H Rail | 460 | Misumi 2020 350 mm extrusion (~195 g) + 350 mm MGN12H steel rail (~265 g) |
| gantry | 2020 Carbon Tube 350mm + MGN12H Rail | 345 | 20x20x2 mm carbon fiber square tube 350 mm (~80 g) + 350 mm MGN12H steel rail (~265 g) |
| gantry | Carbonara 20x20 CF X-Beam (330 / 380 / 430 mm bare) | 64.7 / 74.5 / 84.3 | Monolith / Brazuka / Fabreeko high-Tg ±45°/0° CF beam (196.1 g/m, 0.01 g scale); requires internal PET-CF/PPA-CF M3 load-distribution inserts |
| gantry | CNC Lightweight Al Beam 350mm + MGN12H Rail | 375 | Pocketed CNC 6061-T6 lightweight X-beam 350 mm (~110 g) + 350 mm MGN12H steel rail (~265 g) |
| hotend | Phaetus Dragon HF | 45 | Phaetus datasheet / community scale check (assembled block, heatbreak, heatsink) |
| hotend | Phaetus Rapido 2 UHF | 52 | Phaetus datasheet (ceramic heater cartridge + UHF melt zone) |
| hotend | E3D Revo Voron | 40 | E3D datasheet (rigid Voron heatsink + HeaterCore + brass nozzle) |
| hotend | Mellow Goliath Air | 96 | Mellow / VzBot datasheet (air-cooled ultra-high-flow hotend assembly; [VzBoT3D/Goliath](https://github.com/VzBoT3D/Goliath) photo shows 95 g incl. ~1 m sleeved leads) |
| toolhead | VzBot Vz-Printhead CNC (Al) | 27 | [VzBoT3D/Vz-Printhead-CNC](https://github.com/VzBoT3D/Vz-Printhead-CNC) stated (22 g hybrid CF + Al) |
| hotend | Luke's Lab Chube Compact | 63 | Official Luke's Lab STEP/datasheet: 50 g bare + 6×16 mm 80W heater + PT1000 + V6 nozzle + silicone sock (47.2 mm OAL, 4×M3 16 mm PCD) |
| hotend | Luke's Lab Chube Air | 79 | Official Luke's Lab Rev A drawing: 69 g bare (CoM 36.15 mm) + 6×20 mm heater + thermistor + V6 nozzle (66.0 mm OAL, 6×M3 18 mm PCD) |
| hotend | Luke's Lab Chube Conduction | 90 | Official Luke's Lab Rev A drawing: 74 g bare (CoM 34.82 mm) + 6×47.5 mm 120W Supervolcano heater + thermistor + V6 nozzle (66.0 mm OAL, 6×M3 18 mm PCD) |

## 3. Verified Structural Filament & Material TDS

| Material | Modulus (GPa) | HDT (C) | Creep Limit (N) | Source / Notes |
| :--- | :--- | :--- | :--- | :--- |
| PPA-CF (Annealed) | 10.8 | 220 | 120 | Bambu Lab / Siraya Tech / MyTechFun TDS; annealed high-stiffness AWD motor & idler mount standard |
| PPS-CF (Annealed) | 9.2 | 250 | 140 | Bambu Lab / Siraya Tech / MyTechFun TDS; superior high-temp chamber creep resistance (70–100 C) |
| PET-CF15 | 6.2 | 110 | 85 | Bambu Lab / Polymaker / MyTechFun TDS; low moisture uptake and strong sustained-load creep resistance |
| PA12-CF (Conditioned) | 6.2 | 160 | 65 | Polymaker / MyTechFun TDS; moisture-sensitive and exhibits gradual creep under ~20 lbf AWD preload |
| ABS / ASA | 2.2 | 95 | 35 | MatWeb / Voron standard structural parts; suitable for 2WD 6 mm belts, creeps under high AWD tension |
