"""Per-item engineering confidence registry across all 14 PrintSim hardware catalogs (95 options)."""

from __future__ import annotations

from typing import Any, Literal

ConfidenceTier = Literal["exact", "calibrated", "estimate", "heuristic"]


def confidence_tier(trust_pct: int) -> ConfidenceTier:
    """Map an integer 0-100 confidence score to its canonical display tier."""
    if trust_pct >= 90:
        return "exact"
    if trust_pct >= 70:
        return "calibrated"
    if trust_pct >= 50:
        return "estimate"
    return "heuristic"


ITEM_CONFIDENCE_REGISTRY: dict[str, dict[str, dict[str, Any]]] = {
    "toolhead_presets": {
        "stealthburner": {
            "trust_pct": 88,
            "basis": (
                "Official Voron Stealthburner_CW2.5_Assembly.step exact solid volume rollup (198 cm³ ABS ~158 g) "
                "+ PrintDB datasheets (54 g MGN12H, 85 g LDO-36STH20, 27 g 5015, 15 g 4010) = 398 g on-rail excl. "
                "hotend (444 g w/ Dragon HF, matching Voron 433-450 g on-rail weigh-ins); CAD CoG [0, -24, +16] mm, "
                "nozzle [0, -34, -60] mm."
            ),
            "caveat": (
                "Wiring harness/CAN toolboard mass, LED strip, and extruder motor variants shift real mass by ±12 g."
            ),
        },
        "stealthburner_tap": {
            "trust_pct": 85,
            "basis": (
                "Official StealthBurner CW2.5 STEP + Voron Tap R8 MGN9H sliding-rail carriage Tet4 CAD CMS "
                "(42,115 tets, rb_max=0.0013 Hz, 133.6 g bare chassis) + PrintDB hardware delta "
                "(+40 g net over stock 33 g printed X-carriage = 438 g on-rail excl. hotend, 484 g w/ Dragon HF)."
            ),
            "caveat": (
                "Printed Tap rail carriage hardware, magnet stack, and optical PCB mass vary ±15 g by print infill and fasteners."
            ),
        },
        "xol_compact": {
            "trust_pct": 86,
            "basis": (
                "Official Xol Toolhead Assembly Guide 20240615.step exact solid volume rollup (60.4 cm³ ASA ~51 g) "
                "+ PrintDB datasheets (54 g MGN12H, 115 g Sherpa Mini, 30 g 2×4010, 6.4 g 2510) = 278 g on-rail excl. "
                "hotend (328 g w/ Rapido HF; 274 g off-rail vs AHI ~260 g bare); exact CAD nozzle [0, -29, -49.5] mm, "
                "CoG [0, -18, +10] mm."
            ),
            "caveat": (
                "Optional EBB36/SHT36 toolboard, 17 mm vs 20 mm pancake motor, and UHF hotend swap shift mass/CoG by ±15 g / ±3 mm."
            ),
        },
        "a4t_beacon": {
            "trust_pct": 84,
            "basis": (
                "Official A4T Cowling (47.3 cm³) + Dragon HE Duct (4.9 cm³) + Xol Carriage (19.55 cm³) STEP CAD "
                "(71.8 cm³ total, ~60 g ASA) + PrintDB datasheets (54 g MGN12H, ~134 g WW-BMG/WWG2, 30 g 2×4010, "
                "6.4 g Delta 2510, 3.3 g Beacon) = 298 g on-rail excl. hotend (344 g w/ Dragon HF; 290 g off-rail); "
                "CoG [0, -19, +8] mm, nozzle [0, -28, -50] mm."
            ),
            "caveat": (
                "WW-BMG (~125 g) vs WWG2 (~150 g) and ASA vs PPA-CF (+7 g) shift dressed mass by ±15 g."
            ),
        },
        "dragon_burner": {
            "trust_pct": 83,
            "basis": (
                "Official Dragon_Burner_v0.2_v8.step solid volumes (27.7 cm³ core + ~18 cm³ MGN12H carriage = 37.5 g ABS) "
                "+ PrintDB datasheets (54 g MGN12H, 135 g Orbiter v2.0, 24 g 2×4010, 8 g 3010) = 272 g on-rail excl. "
                "hotend (318 g w/ Dragon HF; 264 g off-rail matching community 253-272 g weigh-ins); CoG [0, -14, +4.5] mm."
            ),
            "caveat": (
                "Modular carriage adapters and extruder swaps (Orbiter 2 vs Sherpa Micro -30 g) shift mass and CoG by ±25 g / ±4 mm."
            ),
        },
        "archetype_cpap": {
            "trust_pct": 82,
            "basis": (
                "Official Archetype Public CAD STEP.step (Breakneck v24: 81.5 cm³ printed core/duct/rear micro-Bowden "
                "bar ~68.4 g) + PrintDB datasheets (54 g MGN12H, 105 g Sherpa Micro, 6.4 g 2510 heatsink fan, 3.3 g "
                "Beacon) = 248 g on-rail excl. hotend (344 g w/ Goliath Air, 298 g w/ Rapido HF); exact CAD nozzle "
                "Y=-21 mm, Z=-64 mm UHF (-74 mm Goliath), CoG [0, -15.5, +12] mm."
            ),
            "caveat": (
                "19 mm corrugated CPAP hose dynamic mass/stiffness and Sherpa Micro vs VZ-Hextrudort/Orbiter 2 swaps (+25-30 g) are lumped."
            ),
        },
        "microwaved_bowden": {
            "trust_pct": 84,
            "basis": (
                "Official STEP CAD (micreathonn/Microwaved-Bowden-Toolhead MBT_Tricorn_V2.STEP) exact "
                "solid volume rollup & 75.6k-elem Tet4 FEA calibration (10,728 X / 3,563 Y N/mm, "
                "5,340 N/mm harmonic mean in SLM Ti-6Al-4V; scaled to AlSi10Mg/6061-T6 by E ratio): "
                "MBTCFD 16,259 mm³ (43.4 g SLM AlSi10Mg), Ti brace/shield (1.1 g), LDO MGN12H (48.3 g), "
                "Monolith SLM belt clamp (6.6 g Al / 19.7 g 316L SS), Beacon H, and ECAS/screws = 130 g on-rail "
                "excl. hotend; CoM centered between belts (-2 mm Z, -35.5 mm nozzle drop)."
            ),
            "caveat": (
                "16 mm CPAP hose, Bowden PTFE/Kevlar sheath dynamic drag, and SLM porosity (~99% dense) "
                "are not explicitly modeled; scale-weighing an assembled build would raise mass confidence >92%."
            ),
        },
        "microwaved_bowden_chube": {
            "trust_pct": 84,
            "basis": (
                "Official STEP CAD (micreathonn/Microwaved-Bowden-Toolhead MBT_Chube_V1.STEP) exact "
                "geometry & 75.3k-elem Tet4 FEA calibration (3,499 X / 2,611 Y N/mm, 2,990 N/mm "
                "harmonic mean in SLM Al 6061-T6/AlSi10Mg; scales to Ti-6Al-4V or polymers by E ratio): "
                "Main_Body (~56.5 g SLM Al), integrated 6× M3 Ø18 mm PCD Chube conduction mount at "
                "Port 1 [0, -12.76, +27.50] mm, LDO MGN12H (48.3 g), Monolith SLM belt clamp (6.6 g Al / "
                "19.7 g 316L SS), Beacon RevH, and ECAS/screws = 135 g on-rail excl. hotend (-38.5 mm nozzle drop)."
            ),
            "caveat": (
                "16 mm CPAP hose, Bowden PTFE/Kevlar sheath dynamic drag, and SLM porosity (~99% dense) "
                "are not explicitly modeled; live CAD CMS raises trust to 86%."
            ),
        },
        "vz_printhead_cnc": {
            "trust_pct": 66,
            "basis": (
                "Official VzBoT Vz-330_v1.2 STEP cage layout & datums (top plate on the MGN9H top face, front/back "
                "pieces around the CF tube, bottom plate, rear pillar) per the Mellow kit assembly guide; 6061 "
                "plate thickness (1.5 mm top/bottom, 2.5 mm sides) and windows fitted so the set reproduces BOTH "
                "published masses (27 g full-alu, 22 g CF-hybrid); Gmsh Tet4 Craig-Bampton CMS with 0.58x bolted "
                "seams. Dressing masses from PrintDB (HextrudORT Low 47 g, Moons NEMA14 85 g, MGN9H 26 g)."
            ),
            "caveat": (
                "No public CNC CAD: plate outlines follow the printed v24 head and pocket patterns are fitted, "
                "not measured; calipers on the real plates (or the Mellow STEP) would raise this above 85%."
            ),
        },
        "vz_printhead_printed": {
            "trust_pct": 80,
            "basis": (
                "Exact official VzBoT Vz-330_v1.2 printed v24 MGN9 Vz-Printhead cage (top 10.26, back 8.69, bottom "
                "8.22, front 3.25, adapter 2.23 cm^3) + 4x M3 standoffs, voxel Tet4 Craig-Bampton CMS in ABS with the "
                "0.32x printed multi-piece knockdown; VzBoT print settings (4 walls, 5 top/bottom, 40-50% infill)."
            ),
            "caveat": (
                "Not scale-weighed; printed density (0.84 g/cm^3) and layer anisotropy vary with slicer settings; "
                "HextrudORT CoM from the CAD mock-up envelope."
            ),
        },
        "trinity_1_6_dav": {
            "trust_pct": 88,
            "basis": (
                "Official WV-design/Trinity-toolhead 1.6-stable STEP (Trinity DAV 1.6 - 10mm belts.STEP) exact "
                "solid volume rollup (65.94 cm³ unibody + 24.44 cm³ ducts_v3 = 88.9 g ABS-GF/PPA-CF), 62.8k-elem "
                "Tet4 Craig-Bampton CMS with 0.55× monolithic FDM infill/anisotropy knockdown (zero split-clamshell seam), "
                "coupled heatsink-braced DAV assembly SE (3,126/3,147 N/mm), DigiKey Delta 3628/2510 datasheets, "
                "and Monolith Discord thread scale weights (~369–400 g off-rail) & ADXL shaper touchstones "
                "(X MZV 111–128 Hz, Y 86–109 Hz)."
            ),
            "caveat": (
                "Sherpa Mini v3 extruder CoM is positioned from the official top mounting bolt datum (+32.1 mm); "
                "printed polymer stiffness varies ±15% with fiber fill, annealing, and layer adhesion."
            ),
        },
        "trinity_1_6_chube": {
            "trust_pct": 88,
            "basis": (
                "Official WV-design/Trinity-toolhead 1.6-stable STEP (Trinity Chube 1.6 - 10mm belts.step, Prooda/Martin) "
                "exact solid volume rollup (78.08 cm³ unibody + 24.23 cm³ wider ducts = 100.7 g ABS-GF/PPA-CF), "
                "75.2k-elem Tet4 Craig-Bampton CMS with 0.55× monolithic FDM unibody knockdown, coupled Ti-shell-braced "
                "Chube Compact assembly SE, and Monolith Discord ADXL touchstone (jeremytodd1 350mm Trident 10mm AWD CNC: "
                "X MZV 127.6 Hz, Y MZV 86.0 Hz @ 22k accel)."
            ),
            "caveat": (
                "Unibody wrap clamps Chube Compact's Ø22.23 mm Ti shell below its cooling cutouts; real plastic-to-Ti "
                "interference fit compliance can soften the coupled shell brace relative to an ideal rigid collar."
            ),
        },
        "el_fuego": {
            "trust_pct": 83,
            "basis": (
                "KhaymKhaym Monolith Discord release STLs (Main Body 31.37 cm³ / 83.75 g SLM AlSi10Mg, "
                "Belt Plate Rev1 6.87 g Al / 19.97 g Steel, Titanium Sherpa Micro extruder plate 3.62 g) "
                "+ GOM Scan 1 MV200 pre-machine metrology report (Aire Fuego GS1, <=20 µm contact-face "
                "deviation, 50.67 mm mount-to-duct height) + 241.9k-elem Tet4 Craig-Bampton CMS "
                "(42,319 X / 127,968 Y / 21,366 Z N/mm at Port 1 [-0.32, -19.93, +2.28] mm; "
                "1,540 X / 748 Y N/mm, 1,007 N/mm harmonic projected at 66 mm Chube/Tricorn nozzle tip; "
                "scales across AlSi10Mg / Ti-6Al-4V / 316L / 17-4PH by E ratio). Dressed baseline 262 g "
                "on-rail excl. hotend (94.2 g 3-part CAD + 48.3 g LDO MGN12H + 105 g Sherpa Micro w/ pancake + "
                "Beacon RevH & fasteners; CoM on carriage front face +2 mm Z)."
            ),
            "caveat": (
                "Release Rev1 STLs had a minor geometry revision pending KhaymKhaym's upcoming GitHub "
                "release; Aire Fuego 2.5 mm internal CPAP bleed cooling and 19 mm CPAP hose drag are "
                "lumped; live 241.9k-elem CAD CMS raises trust to 85%."
            ),
        },
        "heavy_uhf_long": {
            "trust_pct": 56,
            "basis": (
                "Synthetic worst-case heavy long-nozzle UHF toolhead benchmark (475 g on-rail incl. 54 g MGN12H "
                "+ 78 mm nozzle drop)."
            ),
            "caveat": (
                "Real mass, fan-duct bulk, and center-of-gravity overhang vary widely across custom multi-blower UHF shrouds."
            ),
        },
        "monolith_toolhead_cpap": {
            "trust_pct": 64,
            "basis": (
                "Back-solved from CloakedWayne's ~750 g 300 mm AWD moving Y-gantry benchmark on monolith3d.xyz "
                "(750 g minus 210 g LDO CNC XY-joints w/ MGN9H, 121 g Carbonara 300 mm, and ~154 g hollow MGN12 "
                "X-rail ≈ 265 ± 15 g dressed on-rail -> 218 g excl. 52 g hotend) + exact CAD Monolith_SLM_Belt_Clamp.step "
                "(2.472 cm³ = 6.6 g SLM AlSi10Mg) + monolith3d.xyz 19 Jul 2025 SLM BOM (Al body + Ti brace + 48.3 g LDO "
                "MGN12H + 37:1 worm w/ 17 mm LDO pancake); belt-coplanar CoG dz=-1.5 mm."
            ),
            "caveat": (
                "Isolated toolhead weight, exact CoG coordinates, and full STEP CAD are not yet published on monolith3d.xyz "
                "(±15 g mass / ±4 mm CoG uncertainty; 20 mm integrated-worm stepper adds ~15 g); awaits public STEP CMS."
            ),
        },
        "eva3": {
            "trust_pct": 84,
            "basis": (
                "Official RatRig EVA 3 Onshape CAD assembly (5af1985e8b5d6fb20ee622b9, 113.7 cm³ printed core = 95.5 g ASA/PET-CF) "
                "+ Orbiter 2.0 (135 g) + 5015 blower (27 g) + HIWIN MGN12C top carriage (34 g) + SuperPINDA/hardware (28.5 g) = "
                "320 g excl. hotend (370 g w/ Rapido 2 HF); baked 58,780-tet Tet4 CMS superelement."
            ),
            "caveat": (
                "Printed rear-plate/faceplate heat-set insert preload, MGN12C vs MGN12H carriage variants, and different "
                "fan-duct/extruder top options introduce ±15% stiffness and ±15 g mass variance."
            ),
        },
    },
    "toolhead_constructions": {
        "multi_piece_clamshell": {
            "trust_pct": 50,
            "basis": (
                "Lumped Stage 2 chassis heuristic calibrated roughly to Voron StealthBurner/Afterburner "
                "ADXL resonance trends (110 N/mm base, 0.72× torsion)."
            ),
            "caveat": (
                "Split printed seams, heat-set insert pull-out, and print layer orientation cause ±30% real stiffness variance."
            ),
        },
        "compact_cage": {
            "trust_pct": 48,
            "basis": (
                "Lumped open-cage chassis heuristic for XOL/A4T/Dragon Burner class housings "
                "(185 N/mm base, 0.92× torsion)."
            ),
            "caveat": (
                "Unverified by 3D STEP FEA; real stiffness depends heavily on side-cheek ribs and fan-shroud screw clamping."
            ),
        },
        "plate_sandwich_cf": {
            "trust_pct": 44,
            "basis": (
                "Estimated front/rear twin-plate + standoff Stage 2 model for Archetype/VzBot/EVA layouts "
                "(270 N/mm base, 1.05× torsion)."
            ),
            "caveat": (
                "Standoff length, spacer diameter, and plate-bolt preload are lumped rather than explicitly 3D-meshed."
            ),
        },
        "cantilever_bowden": {
            "trust_pct": 44,
            "basis": (
                "Minimalist remote-drive hotend bracket Stage 2 estimate (230 N/mm base, 1.00× torsion)."
            ),
            "caveat": (
                "Single-sided cantilever clamp flex varies widely with bracket gusseting, wall thickness, and Bowden tube drag."
            ),
        },
        "extruder_as_chassis": {
            "trust_pct": 40,
            "basis": (
                "Uncalibrated direct-bolted extruder-as-load-path Stage 2 heuristic (320 N/mm base, 1.15× torsion)."
            ),
            "caveat": (
                "No direct tip-deflection bench test yet; real rigidity varies drastically across Sherpa, Orbiter, Galileo, and Vz-HextrudORT."
            ),
        },
        "cnc_monoblock": {
            "trust_pct": 42,
            "basis": (
                "Rough billet CNC monoblock upper-bound Stage 2 heuristic (480 N/mm base, 1.35× torsion) "
                "scaled by alloy modulus and wall thickness."
            ),
            "caveat": (
                "Awaits 3D STEP Tet10 FEA and static tip-load calibration on real CNC monoblock toolhead shuttles."
            ),
        },
    },
    "z_probes": {
        "static_eddy": {
            "trust_pct": 94,
            "basis": (
                "Rigidly bolted non-contact eddy probe (Beacon/Cartographer): exact zero-sliding-rail "
                "1.00× compliance factor and zero CoG overhang penalty."
            ),
            "caveat": (
                "Only minor sensitivity to probe PCB bracket mass (~4–8 g) and nozzle-to-coil offset."
            ),
        },
        "klicky_inductive": {
            "trust_pct": 86,
            "basis": (
                "Docked static probe/inductive mount (0.98× compliance factor, 12 g added mass, "
                "-1.5 mm CoG Y-shift) verified against static-mount kinematics."
            ),
            "caveat": (
                "Loose dock magnets or thin printed probe ears can introduce minor localized probe chatter."
            ),
        },
        "cnc_tap": {
            "trust_pct": 80,
            "basis": (
                "First-principles 6-DOF series compliance at Port 0 combining the vertical HIWIN MGN7H Z0 "
                "internal sliding rail (Hertzian ball-raceway + Tet10 saddle) with a temperature-invariant "
                "6061-T6 aluminum billet housing & dual N52H magnetic kinematic stop (+62 g, -6.5 mm CoG Y), "
                "validated against ChaoticLab/Vitalii CNC Tap datasheets, Klippain Shake&Tune 131 Hz "
                "decoupled rail mode, and 2 real Voron 2.4 350 CNC Tap builds + controlled SB->XOL A/B sweep "
                "(bsmt, VORONDesign #voron_resonance_basic_help)."
            ),
            "caveat": (
                "Sensitive to internal MGN7H preload, magnet pull force, and stop-pin seating; "
                "same-printer CNC Tap vs Beacon A/B swap would further narrow ±5–8% seat variance."
            ),
        },
        "printed_tap": {
            "trust_pct": 70,
            "basis": (
                "First-principles 6-DOF series compliance at Port 0 combining the vertical HIWIN MGN9H ZF "
                "internal sliding rail (Hertzian ball-raceway + Tet10 saddle) with a chamber-temperature-"
                "softening FDM ABS/ASA carriage sandwich & dual N52H magnetic kinematic stop (+78 g, "
                "-8.5 mm CoG Y), validated against 3 real Voron Tap builds in shaper_benchmarks.yaml "
                "(Formbot Trident R2 -1.5% Y err, Voron 2.4 300 -4.6% Y err) and stealthburner_tap Tet4 CMS (85%)."
            ),
            "caveat": (
                "Build-to-build scatter across print infill, heat-set rail bed, rail preload wear, and "
                "magnet seating; controlled same-printer Printed Tap vs Beacon A/B ADXL CSV pair still open."
            ),
        },
    },
    "hotends": {
        "tricorn_long": {
            "trust_pct": 78,
            "basis": (
                "Official STEP CAD (tricornhotend/tricornhotend.com Tricorn Hotends.step), Heatbreak.pdf "
                "(Ti-6Al-4V Ø2.8/Ø1.9 mm, 0.45 mm wall, 13 mm long), 116k-elem Tet4 FEA calibration "
                "(unbraced 262/264 → 265 N/mm), and BOM (31.4 g CuCrZr hotblock, 3×40W 4×40mm heaters, "
                "66.0 mm overall, 53.5 mm melt zone, 58.1 g dry CAD → 62 g dressed)."
            ),
            "caveat": (
                "Lumped catalog uses FEA-calibrated unbraced stiffness across the 0.45 mm inverted Ti heatbreak "
                "(265 N/mm; higher when paired with Microwaved Ti bottom brace). Live CAD CMS raises trust to 88%."
            ),
        },
        "tricorn_short": {
            "trust_pct": 76,
            "basis": (
                "Official STEP CAD (tricornhotend/tricornhotend.com Tricorn Hotends.step), Heatbreak.pdf "
                "(Ti-6Al-4V Ø2.8/Ø1.9 mm, 0.45 mm wall, 13 mm long), 112k-elem Tet4 FEA calibration "
                "(unbraced 437/442 → 440 N/mm), and BOM (23.0 g CuCrZr hotblock, 54.0 mm overall, "
                "41.0 mm melt zone, ~50 g dressed)."
            ),
            "caveat": (
                "Lumped catalog uses FEA-calibrated unbraced stiffness (440 N/mm); M3 Dragon-pattern bolt preload "
                "adds ~5–10% build variance. Live CAD CMS raises trust to 87%."
            ),
        },
        "chube_conduction": {
            "trust_pct": 78,
            "basis": (
                "Official Luke's Laboratory STEP & Rev A drawing (CHUBE_INTEGRATION_REVA.STEP, "
                "Chube_hotend_integration_specifications.pdf) & 80.2k-elem exact-cutout Tet4 FEA calibration "
                "(801/830 → 815 N/mm): 66.0 mm OAL w/ V6 nozzle (60.0 mm bare), Ø25.4×0.90 mm Ti tube w/ "
                "6× staggered Ø19.0 mm cutouts + 8 mm solid Al conduction top block, 6× M3 Ø18 mm PCD, "
                "52 mm meltzone, 74 g bare (78.2 g FEA, CoM -35.2 mm) / 90 g dressed, 500 °C."
            ),
            "caveat": (
                "Even with 80.2k-elem exact-cutout Tet4 FEA calibration (815 N/mm), 6-bolt M3 preload "
                "and mount interface compliance introduce ±12% effective tip stiffness spread. Live CAD CMS raises trust to 86%."
            ),
        },
        "chube_air": {
            "trust_pct": 78,
            "basis": (
                "Official Luke's Laboratory STEP & Rev A drawing (CHUBE_INTEGRATION_REVA.STEP, "
                "Chube_hotend_integration_specifications.pdf) & 70.7k-elem exact-cutout Tet4 FEA calibration "
                "(549/568 → 560 N/mm): 66.0 mm OAL w/ V6 nozzle (60.0 mm bare), Ø25.4×0.90 mm Ti tube w/ "
                "6× staggered Ø19.0 mm cutouts + 4-fin Al air heatsink, 6× M3 Ø18 mm PCD, 42 mm meltzone, "
                "69 g bare (66.9 g FEA, CoM -38.3 mm) / 79 g dressed, 500 °C."
            ),
            "caveat": (
                "Dual-tier Ø19.0 mm airflow/isolation window Vierendeel compliance is captured by 70.7k-elem "
                "Tet4 FEA (560 N/mm); 6-bolt M3 preload introduces ±12% build variance. Live CAD CMS raises trust to 86%."
            ),
        },
        "chube_compact": {
            "trust_pct": 76,
            "basis": (
                "Official Luke's Laboratory SolidWorks STEP (Chube_Compact_integration_assembly.STEP) & "
                "49.3k-elem exact-cutout Tet4 FEA calibration (3,020/2,931 → 2,975 N/mm): 47.2 mm OAL w/ V6 nozzle, "
                "Ø22.23×0.89 mm Ti tube w/ 4× Ø12.50 mm cutouts @ 45/135/225/315° + solid lower shell, "
                "4× M3 Ø16 mm Dragon PCD, 33 mm meltzone, 80W 6×16 mm heater, 50 g bare / 63 g dressed, 500 °C."
            ),
            "caveat": (
                "49.3k-elem Tet4 FEA captures exact 4-cutout tube & short-lever stiffness (2,975 N/mm); "
                "4× M3 Dragon-pattern mount clamp stiffness in series limits effective assembly stiffness. Live CAD CMS raises trust to 86%."
            ),
        },
        "phaetus_dragon_hf": {
            "trust_pct": 70,
            "basis": (
                "Phaetus Dragon HF CAD/mass datasheet (46 g, 62.2 mm) and 46,879-elem multi-material Tet4 FEA "
                "calibration (1,558/1,563 N/mm bare, 1,049 N/mm series w/ clamp). Live CAD CMS raises trust to 86%."
            ),
            "caveat": (
                "4× hollow Ti standoff screw preload and printed M2.5 mounting ear creep add ±8–12% real lateral stiffness variance."
            ),
        },
        "trianglelab_dragon_ace_volcano": {
            "trust_pct": 86,
            "basis": (
                "Official Trianglelab / HTA3D / DREMC specifications (42–45 g dressed, 24V 70W 6×20 mm cartridge, "
                "V6 nozzle, 300 °C 104NT-4 / 450 °C+ PT1000), exact multi-solid geometry from Trinity DAV 1.6 STEP "
                "(59.4 mm OAL, Ø22.3×23.0 mm Al heatsink, 4× Ø2.3×7.5 mm Ti standoffs on 12×12 mm square, thick 0.45 mm "
                "Ti heatbreak, 34 mm heated Cu Volcano + MZE meltzone), 43.5k-elem conformal Tet4 FEA calibration "
                "(839/845 → 842 N/mm unbraced; 3,126/3,147 N/mm heatsink-braced in Trinity 1.6), and Monolith Discord "
                "flow tests (40–48 mm³/s baseline, 55 mm³/s w/ 120W & 0.5 TC, 100+ mm³/s w/ 1.0 mm HS-PLA)."
            ),
            "caveat": (
                "Ti standoff screw preload and heater cartridge wattage (70W stock vs 80–120W community upgrades) "
                "shift real tip stiffness and high-flow ceiling by ±10–15%. Live CAD CMS raises trust to 87–88%."
            ),
        },
        "e3d_v6_groove": {
            "trust_pct": 68,
            "basis": (
                "E3D V6 CAD/mass specification (42 g dressed, 62.3 mm) and 47,543-elem multi-material Tet4 FEA "
                "calibration (155/155 N/mm bare, rb_max=0.0059 Hz). Live CAD CMS raises trust to 85%."
            ),
            "caveat": (
                "Groove-collar radial play varies dramatically with printed vs machined clamp fit and axial retention."
            ),
        },
        "e3d_revo_voron": {
            "trust_pct": 64,
            "basis": (
                "E3D Revo Voron 4-bolt rigid CAD/mass datasheet (25.2 g bare / 30 g dressed, 49.1 mm) and 27,999-elem "
                "multi-material Tet4 FEA calibration (137/136 N/mm bare, 128.5 N/mm series w/ clamp). Live CAD CMS raises trust to 86%."
            ),
            "caveat": (
                "Thin-walled 0.15 mm SS304 Revo nozzle neck thread seating and spring-clip heater tolerances add ±8–12% variance."
            ),
        },
        "phaetus_rapido_2_hf": {
            "trust_pct": 62,
            "basis": (
                "Phaetus Rapido 2 HF CAD/mass/wattage datasheet (45.2 g bare / 50 g dressed, 54.5 mm, 85 W) and 43,948-elem "
                "multi-material Tet4 FEA calibration (1,277/1,271 N/mm bare, 911 N/mm series w/ clamp). Live CAD CMS raises trust to 86%."
            ),
            "caveat": (
                "Ceramic heater ring clamping and 3× M1.8 Ti standoff screw preload add ±8–12% real lateral stiffness variance."
            ),
        },
        "phaetus_dragon_uhf": {
            "trust_pct": 60,
            "basis": (
                "Phaetus Dragon UHF CAD/mass datasheet (56 g, 81.6 mm) and 55,574-elem multi-material Tet4 FEA "
                "calibration (513/509 N/mm bare, rb_max=0.0048 Hz). Live CAD CMS raises trust to 85%."
            ),
            "caveat": (
                "Copper melt-extender thread seating and long lower lever arm compliance add ±10–12% real variance."
            ),
        },
        "phaetus_rapido_2_uhf": {
            "trust_pct": 58,
            "basis": (
                "Phaetus Rapido 2 UHF CAD/mass datasheet (62 g, 79.2 mm, 85 W) and 47,955-elem multi-material "
                "Tet4 FEA calibration (841/838 N/mm bare, rb_max=0.0053 Hz). Live CAD CMS raises trust to 85%."
            ),
            "caveat": (
                "Extended UHF nut-extender thread seating and 3-screw upper standoff preload add ±10–12% variance."
            ),
        },
        "mellow_goliath_air": {
            "trust_pct": 58,
            "basis": (
                "Mellow Goliath Air datasheet (84.7 g bare / 96 g dressed, 69.2 mm, 100 W) and 51,816-elem multi-material "
                "Tet4 FEA calibration (895/955 N/mm bare, 717 N/mm series w/ clamp). Live CAD CMS raises trust to 85%."
            ),
            "caveat": (
                "3× Ti post bolt preload and 69.24 mm long copper melt-core mount interface add ±10% lateral stiffness variance."
            ),
        },
        "monolith_50mm_hf": {
            "trust_pct": 66,
            "basis": (
                "Monolith 50 mm+V6 HF hotend specification (52 g dressed, 68 mm OAL, 50 mm melt zone, "
                "80 W PT1000, 4-bolt M3 rigid flange) and 40,452-elem multi-material Tet4 FEA calibration "
                "(568/630 N/mm bare, rb_max=0.0084 Hz). Live CAD CMS raises trust to 85%."
            ),
            "caveat": (
                "50 mm extended melt-zone Graetz flow ceiling still awaits direct bench extrusion flow sweep."
            ),
        },
    },
    "hotend_mount_styles": {
        "chube_6bolt_m3_18pcd": {
            "trust_pct": 68,
            "basis": (
                "Luke's Laboratory 6-bolt M3 18 mm PCD monocoque flange baseline (6,500 N/mm base clamp) "
                "scaled by collar material and thickness."
            ),
            "caveat": (
                "Effective clamp rigidity drops quickly if bolted into printed ABS rather than PPA-CF or CNC aluminum."
            ),
        },
        "dragon_4bolt_m25": {
            "trust_pct": 62,
            "basis": (
                "Phaetus/Voron 4-bolt M2.5 rigid top-flange clamp estimate (3,200 N/mm base) scaled by "
                "collar modulus and thickness."
            ),
            "caveat": (
                "M2.5 heat-set insert pull-out and chamber creep in printed mounts cause ±20–25% clamp scatter."
            ),
        },
        "rigid_4bolt_m3": {
            "trust_pct": 58,
            "basis": (
                "4-bolt M3 top-flange clamp estimate (2,200 N/mm base) for Revo Voron and rigid heatsink mounts."
            ),
            "caveat": (
                "Flange thickness, washer bearing area, and printed boss creep are approximated by scalar scaling."
            ),
        },
        "groove_mount": {
            "trust_pct": 50,
            "basis": (
                "J-head groove-mount semicircular collar heuristic (800 N/mm base) reflecting low axial "
                "and pitch clamping leverage."
            ),
            "caveat": (
                "Collar radial clearance, clamp-screw torque, and plastic relaxation cause ±30%+ stiffness variance."
            ),
        },
    },
    "belts": {
        "gt2_6mm": {
            "trust_pct": 87,
            "basis": (
                "Gates PowerGrip 2GT 6 mm Table 6 datasheet EA (16.9 kN × 1.85 dynamic factor) paired with "
                "Costello helical-cord micromechanics (matched to Wang et al. 2018 0–100 N 11.7→22.6 kN), "
                "3D Tet10 pitch-cell FEA tooth-mesh shear & seating (k_t=32,934 N/(m·mm), s0=0.76, q_s=5.85 N/mm) "
                "and composite ABD bending (EI=0.001917 N·m²/m), Ling et al. (2020) EPDM DMA (tan δ≈0.06–0.08, "
                "56% E' @ 120 °C), and first-principles nonlinear coupled shaker dynamics (solve_nonlinear_belt_shaker_response), "
                "validated on Voron Trident/2.4 ADXL logs (jeff.k 6.2→9.8 N ratio within +2.7% X / -0.8% Y)."
            ),
            "caveat": (
                "Actual belt peak shifts with tension gauge accuracy, pulley runout, and dynamic cord stiffening ratio (1.6–2.0× supported range)."
            ),
        },
        "gt2_9mm": {
            "trust_pct": 85,
            "basis": (
                "Gates PowerGrip 2GT 9 mm Table 6 datasheet EA (25.5 kN × 1.85 dynamic factor) with Costello "
                "helical-cord micromechanics, 3D Tet10 pitch-cell FEA tooth-mesh/EI (32,934 N/(m·mm), 0.001917 N·m²/m), "
                "Ling et al. (2020) EPDM DMA, and coupled nonlinear shaker solver calibrated against RatRig, Monolith, and VzBot ADXL logs."
            ),
            "caveat": (
                "Real resonance depends on accurate tension balancing and stiffer double-shear idlers at >80 N preload."
            ),
        },
        "gt2_10mm": {
            "trust_pct": 86,
            "basis": (
                "Gates PowerGrip 2GT 10 mm EPDM fiberglass (28.3 kN × 1.85 dynamic EA, scale-weighed 12.36 g/m "
                "across Monolith 2×1658 mm belts) with Costello cord micromechanics, 3D Tet10 pitch-cell FEA, and "
                "coupled nonlinear shaker solver validated on CloakedWayne AWD/2WD, 1.0–2.8 A current sweep (+0.3% ratio err), "
                "and matched-tension belt shootouts (113.8 Hz ZV @ 89 N)."
            ),
            "caveat": (
                "Assumes 10 mm double-shear live-shaft idlers and balanced tension across A/B loops."
            ),
        },
        "gt3_10mm": {
            "trust_pct": 84,
            "basis": (
                "Gates PowerGrip GT3 (2MGT) 10 mm fiberglass (scale-weighed 14.99 g/m, 2GT Tet10 pitch-cell FEA "
                "with 0.90× softer-compound seating factor & 1.18× rolling drag) validated on CloakedWayne 10 mm "
                "GT3 vs GT2 EPDM matched-tension shootout (110.6 vs 113.8 Hz ZV, -1.1% ratio err) & 12 mm GT3 89→200 N capture."
            ),
            "caveat": (
                "Softer GT3 base elastomer approaches GT2 EPDM stiffness only at high tension (~200 N on 12 mm) and varies with chamber soak."
            ),
        },
        "gt2_12mm": {
            "trust_pct": 80,
            "basis": (
                "Gates PowerGrip 2GT 12 mm EA (34.0 kN narrow-belt rule × 1.85 dynamic factor) with Costello "
                "cord micromechanics, 3D Tet10 pitch-cell FEA (32,934 N/(m·mm), 0.001917 N·m²/m), and coupled "
                "nonlinear shaker solver benchmarked against high-tension AWD builds (dderg, CloakedWayne 12 mm)."
            ),
            "caveat": (
                "Fewer empirical ADXL datasets than 6/9/10 mm 2GT; assumes rigid double-shear 12 mm idler stacks."
            ),
        },
        "gt2_15mm": {
            "trust_pct": 62,
            "basis": (
                "Gates 2M unit-strain modulus at 15 mm (47.25 kN × 1.85) with Costello cord micromechanics "
                "and 3D Tet10 2GT pitch-cell FEA tooth-mesh & ABD bending stiffness."
            ),
            "caveat": (
                "Experimental custom wide-belt hardware; very sparse CoreXY ADXL data and extreme radial shaft/bearing loads."
            ),
        },
        "gt2_dual_9mm": {
            "trust_pct": 55,
            "basis": (
                "Experimental parallel-belt concept modeled as ideal 2× Gates 2GT 9 mm (51.0 kN × 1.85) with "
                "Costello cord micromechanics and 3D Tet10 2GT pitch-cell FEA."
            ),
            "caveat": (
                "Assumes perfect tension equalization between two stacked loops; real mismatch and tracking error lower stiffness."
            ),
        },
        "3gt_9mm": {
            "trust_pct": 52,
            "basis": (
                "Gates Table 6 3GT (3MGT) 9 mm datasheet EA (39.6 kN × 1.85) paired with Costello 270-tex E-glass "
                "cord micromechanics and 3D Tet10 3GT pitch-cell FEA (k_t=41,378 N/(m·mm), s0=0.32, EI=0.01289 N·m²/m)."
            ),
            "caveat": (
                "Unproven on high-reversal CoreXY printers; 3 mm pitch VFA ripple, chordal action, and ≥20 mm idler fit lack ADXL validation."
            ),
        },
        "3gt_12mm": {
            "trust_pct": 50,
            "basis": (
                "Gates 3GT 12 mm static EA (56.5 kN × 1.85) paired with Costello 270-tex cord micromechanics "
                "and 3D Tet10 3GT pitch-cell FEA (k_t=41,378 N/(m·mm), EI=0.01289 N·m²/m)."
            ),
            "caveat": (
                "No CoreXY ADXL validation; heavy 2.41 mm backing bending stiffness and custom 12 mm 3GT pulley mechanics are untested."
            ),
        },
        "3gt_15mm": {
            "trust_pct": 48,
            "basis": (
                "Gates Table 6 3GT 15 mm datasheet EA (73.4 kN × 1.85) paired with Costello 270-tex cord micromechanics "
                "and 3D Tet10 3GT pitch-cell FEA (k_t=41,378 N/(m·mm), EI=0.01289 N·m²/m)."
            ),
            "caveat": (
                "Unproven on 3D printers; 30 g/m belt mass, high bending drag, and custom CNC pulley stacks remain theoretical."
            ),
        },
        "at3_10mm": {
            "trust_pct": 48,
            "basis": (
                "Breco/Elatech AT3 10 mm industrial linear-actuator C_spez datasheet (80 kN Hookean steel/TPU) "
                "paired with Costello 7×3 steel-cord micromechanics and 3D Tet10 AT3 pitch-cell FEA "
                "(k_t=108,743 N/(m·mm), s0=0.05, q_s=2.40 N/mm, EI=0.006225 N·m²/m)."
            ),
            "caveat": (
                "Unproven on serpentine CoreXY printers; steel-cord reverse-bend fatigue and 92 ShA TPU chamber softening (>55 °C) lack ADXL data."
            ),
        },
        "at3_16mm": {
            "trust_pct": 46,
            "basis": (
                "Breco/Elatech AT3 16 mm industrial linear-actuator C_spez datasheet (135 kN steel/TPU) "
                "paired with Costello 7×3 steel-cord micromechanics and 3D Tet10 AT3 pitch-cell FEA."
            ),
            "caveat": (
                "Unproven on CoreXY printers; requires ≥20 mm back-idlers to avoid wire fatigue and suffers TPU stiffness loss in heated chambers."
            ),
        },
        "at3_gen3_16mm": {
            "trust_pct": 45,
            "basis": (
                "ContiTech Synchroflex AT3 Gen III 16 mm industrial C_spez datasheet (250 kN HD steel/TPU) "
                "paired with Costello steel-cord micromechanics and 3D Tet10 AT3 pitch-cell FEA (1.10× k_t)."
            ),
            "caveat": (
                "Theoretical upper-bound only; zero CoreXY ADXL evidence, severe bending drag, and TPU cold-flow in heated chambers."
            ),
        },
        "htd3m_15mm": {
            "trust_pct": 42,
            "basis": (
                "Elatech HTD 3M 15 mm industrial C_spez datasheet (127.5 kN steel/TPU) paired with Costello "
                "steel-cord micromechanics and 3D Tet10 AT3 pitch-cell FEA (0.88× HTD tooth-profile adjustment)."
            ),
            "caveat": (
                "Theoretical only; ≥30 mm reverse-bend idler requirement and HTD semicircular tooth backlash are impractical on compact CoreXY frames."
            ),
        },
        "at5_16mm": {
            "trust_pct": 40,
            "basis": (
                "Breco/Elatech AT5 16 mm heavy industrial C_spez datasheet (280 kN steel/TPU) paired with "
                "Costello steel-cord micromechanics and pitch-scaled 3D Tet10 AT3 FEA (1.15× k_t, 2.0× EI)."
            ),
            "caveat": (
                "Purely theoretical industrial upper-bound; ≥60 mm back-idler minimum and 5 mm pitch are incompatible with Voron/VzBot-scale gantries."
            ),
        },
    },
    "belt_routings": {
        "stock_corexy": {
            "trust_pct": 87,
            "basis": (
                "Explicit 3D pulley-coordinate routing graph using CAD corner stations when active, stacked "
                "upper/lower belt planes (dz = w + 3 mm), rear A/B drive S-wrap guide idlers, and per-idler "
                "Euler-Bernoulli bolt cantilever vs double-shear + Harris Hertzian compliance via Castigliano "
                "virtual work, validated against Stock Voron Trident 300 hot/cold ADXL logs."
            ),
            "caveat": (
                "Actual idler bolt preload torque, bracket bore clearance, and belt tracking offset still add ±5–8% variance."
            ),
        },
        "monolith_short": {
            "trust_pct": 86,
            "basis": (
                "Explicit 3D Monolith inboard/collinear corner pulley-coordinate graph (naturally varying from "
                "0.69× on 230 mm M0 to 0.82–0.86× on 300–350 mm frames), stacked Z lever arms, and per-idler "
                "press-fit live-shaft + Belleville disc-spring compliance validated against CloakedWayne "
                "300 mm AWD/2WD & current-sweep ADXL logs."
            ),
            "caveat": (
                "Precision live-shaft press-fit interference and Belleville stack preload vary slightly across kits."
            ),
        },
    },
    "gantry_brackets": {
        "aluminum_6061_t6": {
            "trust_pct": 66,
            "basis": (
                "Predictable 68.9 GPa 6061-T6 alloy modulus, zero thermal creep ceiling (250 N), and "
                "CloakedWayne LDO Monolith CNC XY-joint scale weights (+42 g delta)."
            ),
            "caveat": (
                "Lumped 500 N/mm-scaled baseline does not resolve specific pocket cutouts, wall ribs, or M3/M5 bolt-boss preload without STEP FEA."
            ),
        },
        "aluminum_7075_t6": {
            "trust_pct": 64,
            "basis": (
                "71.7 GPa 7075-T6 billet modulus and high creep rating (280 N) assuming similar CNC pocket "
                "geometry to 6061-T6."
            ),
            "caveat": (
                "Lumped scaling assumes generic XY-joint geometry rather than a specific vendor STEP Tet10 superelement."
            ),
        },
        "abs_printed": {
            "trust_pct": 82,
            "basis": (
                "PrintDB-hydrated ABS/ASA (materials/generic-abs: 2.1 GPa, 95 °C HDT, 45 N creep limit) with "
                "calibrated FDM orthotropic infill/interlayer, conditioned moisture, thermal HDT, and sustained "
                "belt-tension creep secant knockdown (gantry_bracket_polymer_knockdown: 0.597× @ 25 °C / 0.528× @ 75 °C, 60 N)."
            ),
            "caveat": (
                "Real printed XY-joint stiffness and creep still vary ±12–15% with slicer perimeter count, infill %, and chamber soak."
            ),
        },
        "ppa_cf_printed": {
            "trust_pct": 84,
            "basis": (
                "PrintDB-hydrated annealed PPA-CF (materials/bambu-ppa-cf: 10.8 GPa, 196 °C HDT, 120 N creep limit) "
                "with calibrated FDM orthotropic, moisture (1.30%/0.65% annealed), thermal HDT, and creep knockdown "
                "(gantry_bracket_polymer_knockdown: 0.505× @ 25 °C / 0.499× @ 75 °C, 60 N)."
            ),
            "caveat": (
                "Annealing protocol, fiber orientation around M5 bosses, and bracket CAD rib geometry add ±10–12% spread."
            ),
        },
        "pps_cf_printed": {
            "trust_pct": 84,
            "basis": (
                "PrintDB-hydrated annealed PPS-CF (materials/bambu-pps-cf: 9.2 GPa, 235 °C HDT, 140 N creep limit, "
                "0.05% water absorption) with calibrated FDM orthotropic, moisture, thermal HDT, and creep knockdown "
                "(gantry_bracket_polymer_knockdown: 0.546× @ 25 °C / 0.542× @ 75 °C, 60 N)."
            ),
            "caveat": (
                "Inter-layer Z-bond strength, annealing shrinkage, and bolt-boss bearing stiffness vary ±10% by printer tuning."
            ),
        },
        "pet_cf_printed": {
            "trust_pct": 83,
            "basis": (
                "PrintDB-hydrated annealed PET-CF15 (materials/bambu-pet-cf: 5.2 GPa, 182 °C HDT, 85 N creep limit, "
                "0.37% water absorption) with calibrated FDM orthotropic, moisture, thermal HDT, and creep knockdown "
                "(gantry_bracket_polymer_knockdown: 0.523× @ 25 °C / 0.515× @ 75 °C, 60 N)."
            ),
            "caveat": (
                "Printed perimeter count, M5 bearing stack clamping, and non-kit bracket CAD topology add ±10–12% spread."
            ),
        },
        "pa12_cf_printed": {
            "trust_pct": 82,
            "basis": (
                "PrintDB-hydrated conditioned PA12-CF15 (materials/generic-pa12-cf15: 6.2 GPa, 135 °C HDT, 65 N creep limit, "
                "1.60% water absorption) with calibrated FDM orthotropic, moisture plasticization (0.850×), thermal HDT, "
                "and creep knockdown (gantry_bracket_polymer_knockdown: 0.427× @ 25 °C / 0.410× @ 75 °C, 60 N)."
            ),
            "caveat": (
                "Ambient humidity equilibration and nylon moisture uptake vary ±12–15% under sustained AWD belt tension."
            ),
        },
        "pc_pbt_gf_printed": {
            "trust_pct": 83,
            "basis": (
                "PrintDB-hydrated TreeD PC-PBT-GF (materials/treed-pc-pbt-gf: 4.6 GPa, 120 °C HDT, 72 N creep limit) "
                "with calibrated FDM orthotropic, moisture, thermal HDT, and creep knockdown "
                "(gantry_bracket_polymer_knockdown: 0.542× @ 25 °C / 0.516× @ 75 °C, 60 N)."
            ),
            "caveat": (
                "Inter-layer Z adhesion and local M5 bearing-seat creep at >85 °C motor-face soak add ±10–12% variance."
            ),
        },
        "slm_aluminum": {
            "trust_pct": 68,
            "basis": (
                "EOS / JLCPCB / IN3DTEC stress-relieved SLM AlSi10Mg datasheet (68.0 GPa, E_z/E_xy=0.93, "
                "2670 kg/m³, 240 MPa yield, k=140 W/m·K, CTE=20.5 ppm/K, 0.92× bolted rough-face contact factor) "
                "benchmarked against Monolith SLM XY-joint & El Fuego hardware."
            ),
            "caveat": (
                "As-sintered surface roughness (Ra 6–15 µm), ~0.5% micro-porosity, and exact post-machining of "
                "bearing bores vary across SLM bureaus until STEP Tet10 condensation."
            ),
        },
        "slm_ti6al4v": {
            "trust_pct": 67,
            "basis": (
                "EOS Ti64 / JLCPCB TC4 stress-relieved SLM Titanium datasheet (110 GPa, E_z/E_xy=0.96, "
                "4410 kg/m³, 960 MPa yield, k=6.8 W/m·K, sub-steel CTE 8.8 ppm/K, 0.93× bolted contact factor)."
            ),
            "caveat": (
                "Lumped bracket mass (+28 g delta) and stiffness assume hollow thin-wall SLM topology; exact "
                "rib geometry and post-reamed bearing seats require STEP Tet10 verification."
            ),
        },
        "slm_maraging_steel": {
            "trust_pct": 66,
            "basis": (
                "EOS MaragingSteel MS1 (1.2709) datasheet (180 GPa, E_z/E_xy=0.96, 8050 kg/m³, 1000 MPa as-built "
                "yield, bearing-matched CTE 10.3 ppm/K, 0.94× bolted contact factor)."
            ),
            "caveat": (
                "High density increases moving carriage mass (+64 g delta) unless aggressively topology-optimized; "
                "requires rust-inhibiting coating in humid environments."
            ),
        },
        "slm_316l_steel": {
            "trust_pct": 67,
            "basis": (
                "EOS / JLCPCB SLM 316L stainless datasheet (185 GPa XY / 180 GPa Z columnar texture, "
                "7970 kg/m³, 470 MPa yield, CTE 15.8 ppm/K, 0.93× bolted contact factor)."
            ),
            "caveat": (
                "Lumped stiffness assumes thin-wall SLM belt clamps / XY-joints; high density (+62 g delta) "
                "penalizes Y-axis moving mass vs SLM AlSi10Mg or Ti-6Al-4V."
            ),
        },
        "slm_17_4ph_steel": {
            "trust_pct": 67,
            "basis": (
                "EOS GP1 / SLM Solutions / JLCPCB 17-4PH H900 datasheet (190 GPa, E_z/E_xy=0.95, 7780 kg/m³, "
                "1050 MPa aged yield, steel-matched CTE 10.8 ppm/K, 0.93× bolted contact factor)."
            ),
            "caveat": (
                "Yield strength varies strongly between as-built (~600–860 MPa) and H900 aged (~1050–1250 MPa) "
                "service-bureau heat treatments; lumped geometry omits specific pocket topology."
            ),
        },
        "sheet_metal_steel": {
            "trust_pct": 68,
            "basis": (
                "Monolith Builder v1.5 laser-cut structural steel gantry plates (200 GPa, 3–3.2 mm XY-joints, "
                "4.7–5 mm motor mounts & tensioners, zero thermal creep or Al CTE bearing-seat mismatch)."
            ),
            "caveat": (
                "Lumped plate-stack stiffness does not explicitly mesh spacer standoff shear or bend-radius tabs."
            ),
        },
    },
    "rail_qualities": {
        "hiwin_genuine": {
            "trust_pct": 88,
            "basis": (
                "HIWIN GW-11-7-EN-2401-K Tables 3.3 & 3.75–3.77 clearance/rigidity Hertzian Gothic-arch "
                "ball-contact baseline (f=0.525, V_Dws=0.25 µm, 1.00× stiffness, 1.00× drag) with Hung et al. (2011) 15% skirt compliance."
            ),
            "caveat": (
                "Assumes genuine Grade C/H ball sorting, flat extrusion mounting face, and proper rail bolt torque."
            ),
        },
        "midgrade_matched": {
            "trust_pct": 82,
            "basis": (
                "First-principles ISO 3290 G16 ball-tolerance active-ball load-sharing (V_Dws=0.55 µm, +0.08 µm bias) "
                "+ NSK E728 groove conformity (f=0.532) Hertzian model (~0.83× Z0 → 0.88× Z1 → 0.92× ZA) for LDO / HoneyBadger / Fysetc."
            ),
            "caveat": (
                "Lot-to-lot preload sorting consistency and grease state still add ±8–10% carriage stiffness variance."
            ),
        },
        "precision_thk_nsk": {
            "trust_pct": 84,
            "basis": (
                "THK SRS/HSR C1/C0 & NSK E728 §5.2/§5.4 Grade P conformity (f=0.516), ISO G5 ball tolerance "
                "(V_Dws=0.13 µm, -0.12 µm C1 bias), and radial-clearance contact table (~1.18–1.28× across Z0–ZA, ~100% active balls)."
            ),
            "caveat": (
                "Extrusion mounting-face flatness on aluminum frames can cap real Grade P preload gains by ±6–10%."
            ),
        },
        "budget_generic": {
            "trust_pct": 80,
            "basis": (
                "Unsorted ISO G28–G40 ball-diameter scatter (V_Dws=1.60 µm, Δ_tol=4.2 µm, +0.25 µm clearance bias, "
                "f=0.550) Hertzian active-ball contact model (~0.49× Z0 → 0.58× Z1 → 0.68–0.73× ZA/ZB)."
            ),
            "caveat": (
                "Batch-to-batch clone scatter from undersized balls, rough raceways, or missing seals remains wider than branded rails."
            ),
        },
    },
    "corner_joints": {
        "rigid": {
            "trust_pct": 95,
            "basis": (
                "Exact idealized welded/monolithic corner boundary condition (10¹⁰ N/m, 10⁹ N·m/rad)."
            ),
            "caveat": (
                "Real bolted T-slot joints still exhibit Iwan micro-slip under high inertial shock."
            ),
        },
        "blind_joint": {
            "trust_pct": 84,
            "basis": (
                "133k-DOF 3D Tet10 continuum FEA submodel (Misumi HFS5-2020 1.65 mm T-slot lip flexure + VDI-2230 "
                "M5 SHCS butt-face clamping vs welded control): (16k N/mm, 13.8k N·m/rad, 4.2k N·m/rad in 2020)."
            ),
            "caveat": (
                "M5 socket-head preload torque (3.5–5 N·m), end-tap squareness, and anodized friction add ±10–12% spread."
            ),
        },
        "gusset_plate": {
            "trust_pct": 83,
            "basis": (
                "180k-DOF 3D Tet10 continuum FEA submodel of 5-hole 60×60×5 mm CNC 6061-T6 gusset + 5× M5 T-nuts "
                "vs welded control: (32k N/mm, 36.5k N·m/rad, 7.8k N·m/rad in 2020, Δζ=0.002)."
            ),
            "caveat": (
                "T-nut seating area, plate thickness variants, and bolt preload torque add ±10% real stiffness variance."
            ),
        },
        "l_bracket": {
            "trust_pct": 80,
            "basis": (
                "126k-DOF 3D Tet10 continuum FEA submodel of 20×20×17 mm die-cast aluminum L-bracket + 2× M5 T-nuts "
                "(open gap 1,903 → flush butt 3,100 N·m/rad): (6.5k N/mm, 3.1k N·m/rad, 1.4k N·m/rad in 2020, Δζ=0.005)."
            ),
            "caveat": (
                "Butt-face seating gap and 2-bolt T-nut slip under diagonal racking add ±12–15% variance."
            ),
        },
        "corner_cube": {
            "trust_pct": 82,
            "basis": (
                "184k-DOF 3D Tet10 continuum FEA submodel of 20×20×20 mm 3-way corner cube + T-slot anti-rotation tabs "
                "+ M5/M6 end bolts vs welded cube control: (52k N/mm, 88.5k N·m/rad, 11.2k N·m/rad in 2020, Δζ=0.001)."
            ),
            "caveat": (
                "End-bore thread engagement and cube face seating flatness add ±10% stiffness spread."
            ),
        },
    },
    "extrusion_fillings": {
        "none": {
            "trust_pct": 96,
            "basis": (
                "Standard hollow unfilled T-slot extrusion baseline (zero cavity mass or composite EI coupling)."
            ),
            "caveat": (
                "Matches standard bare aluminum extrusions within <4% profile tolerance."
            ),
        },
        "silica_sand": {
            "trust_pct": 80,
            "basis": (
                "Friend & Kinra (2000) / Marhadi & Kinra (2005) granular Coulomb particle-damping tap-test calibration "
                "(1,550 kg/m³, unbonded, Δζ=0.022) persisted to PrintDB."
            ),
            "caveat": (
                "Compaction density, grain moisture, and settling gaps over time shift real damping by ±12–15%."
            ),
        },
        "epoxy_granite": {
            "trust_pct": 82,
            "basis": (
                "Cortes & Castillo (2007), Suh & Lee (2008), and Möhring et al. (2015) bonded mineral-cast composite "
                "(E=30 GPa, 2,250 kg/m³, η_bond=0.95 → +17.1% 2020 EI, Ungar-Kerwin Δζ=0.030) persisted to PrintDB."
            ),
            "caveat": (
                "Air voids, quartz grading, and anodized cavity surface prep cause ±10–12% composite EI/damping spread."
            ),
        },
        "uhpc_grout": {
            "trust_pct": 80,
            "basis": (
                "Cementitious UHPC/CSA shrinkage-slip composite calibration (E=42 GPa, 2,350 kg/m³, η_bond=0.80 → "
                "+20.2% 2020 EI, Δζ=0.025) accounting for micro-slip against anodized Al walls, persisted to PrintDB."
            ),
            "caveat": (
                "Water-cement ratio and cure-shrinkage debonding from smooth T-slot cores add ±12–15% EI variance."
            ),
        },
        "lead_shot": {
            "trust_pct": 79,
            "basis": (
                "Marhadi & Kinra (2005) high-density #8-#9 lead-sphere impact particle-damping calibration "
                "(6,800 kg/m³, unbonded, Δζ=0.038) persisted to PrintDB."
            ),
            "caveat": (
                "Packing fraction, internal cavity rattle, and vertical settling inside uprights add ±12–15% spread."
            ),
        },
        "pu_foam": {
            "trust_pct": 78,
            "basis": (
                "Closed-cell polyurethane acoustic wall-ringing attenuation calibration (45 kg/m³, unbonded, "
                "zero structural EI, Δζ=0.008) persisted to PrintDB."
            ),
            "caveat": (
                "Uneven closed-cell expansion voids inside narrow T-slot cavities vary acoustic damping slightly."
            ),
        },
    },
    "frame_feet": {
        "rigid_bolted": {
            "trust_pct": 95,
            "basis": (
                "Exact idealized rigid benchtop boundary condition (zero foot compliance at bottom corners)."
            ),
            "caveat": (
                "Assumes rigid steel/granite mounting; flexible wooden desks introduce unmodeled sub-30 Hz table rocking."
            ),
        },
        "rubber_feet": {
            "trust_pct": 79,
            "basis": (
                "Gent (1958) / Lindley (1974) shape-factor (SF=0.50) & Barry Controls 65–70A elastomer calibration "
                "(380 N/mm axial, 148 N/mm lateral, η=0.12, Δζ=0.006; ~27 Hz sway on 20 kg CoreXY) persisted to PrintDB."
            ),
            "caveat": (
                "Elastomer durometer scatter, compression preload, and benchtop surface friction shift rocking by ±10–15%."
            ),
        },
        "paver_on_foam": {
            "trust_pct": 77,
            "basis": (
                "18 kg concrete slab (400×400×50 mm) + 25 mm closed-cell EVA/acoustic foam two-layer impedance "
                "calibration (215 N/mm axial, 78 N/mm lateral, η=0.28, Δζ=0.022; ~14.4 Hz sway) matched to CNC Kitchen benchmarks."
            ),
            "caveat": (
                "Foam density, compression set, and benchtop structural impedance cause ±12–15% rocking variance."
            ),
        },
        "suspended_lines": {
            "trust_pct": 62,
            "basis": (
                "Closed-form line mechanics: axial EA/L per line and pendulum geometric stiffness T/L "
                "(sway = sqrt(g/L)/2pi, verified against the FE rigid-body modes), with the frame coupled as a "
                "free body through its modal receptance (inertia relief + two-body reduced mass)."
            ),
            "caveat": (
                "Line EA is a working-load secant estimate (paracord/nylon are nonlinear and creep); the gravity "
                "righting moment and line-angle splay are neglected, and no suspended-printer ADXL data exists to "
                "calibrate against."
            ),
        },
        "anti_vib_pads": {
            "trust_pct": 78,
            "basis": (
                "Sorbothane Data Sheet 101 (50 Duro tan δ=0.56, 1.5\" hemisphere 0510130-50-10) + HULA F8-22M "
                "thrust-bearing/95A TPU isolation calibration (65 N/mm axial, 22 N/mm lateral, Δζ=0.014; ~10.5 Hz sway) persisted to PrintDB."
            ),
            "caveat": (
                "Nonlinear hyperelastic shear stiffness, temperature sensitivity, and desk impedance add ±12–15% rocking spread."
            ),
        },
    },
    "bed_mounts": {
        "rigid_4pt": {
            "trust_pct": 86,
            "basis": (
                "Solid metal 4-point standoffs (65k N/mm axial, 40k N/mm lateral, k_therm=80 MN/m, 420 N M4 slip cap) "
                "with MITC4 eccentric thermal-restraint bending & Volkersen/Timoshenko bimetallic PEI stack calibrated "
                "to Voron 2.4r2 tight-bolt Klipper BED_MESH_OUTPUT (260 µm thermal PTV vs 180–320 µm taco band)."
            ),
            "caveat": (
                "Exact bolt-hole clearance and M4 torque slip onset still introduce ±10–15% thermal taco spread."
            ),
        },
        "trident_3pt_standoffs": {
            "trust_pct": 85,
            "basis": (
                "Stock Voron Trident Bakelite standoffs + GE5C spherical joint (18k N/mm axial, k_therm=5.2 MN/m, "
                "340 N slip cap) with MITC4 eccentric thermal-restraint & bimetallic PEI stack calibrated to "
                "Trident Klipper BED_MESH_OUTPUT (72.5 µm thermal PTV vs 50–90 µm band)."
            ),
            "caveat": (
                "Slot friction, washer clamping torque, and Bakelite compression creep add ±10–12% thermal restraint spread."
            ),
        },
        "kinematic_3pt": {
            "trust_pct": 85,
            "basis": (
                "Slocum Precision Machine Design Hertzian steel ball-in-V-groove kinematic contact (32k N/mm axial, "
                "k_therm=0) + Volkersen/Timoshenko bimetallic magnetic-PEI stack calibrated to MRW/Maxwell Klipper "
                "BED_MESH_OUTPUT (24.6 µm thermal PTV @ 110 °C vs 15–35 µm band)."
            ),
            "caveat": (
                "Assumes clean hardened ball/pin contact without V-groove brinelling or magnet pull-off under rapid Tap loads."
            ),
        },
        "whopping_pochard_kinematic": {
            "trust_pct": 85,
            "basis": (
                "Spring-steel radial leaf-flexure kinematic model (24k N/mm axial, Tap-rated, k_therm=0) + "
                "Volkersen/Timoshenko bimetallic magnetic-PEI stack calibrated to Klipper BED_MESH_OUTPUT (24.6 µm thermal PTV vs 15–35 µm)."
            ),
            "caveat": (
                "Actual leaf-blade thickness tolerance and clamp-block fastener preload shift vertical compliance by ±10%."
            ),
        },
        "spherical_3pt_ratrig": {
            "trust_pct": 82,
            "basis": (
                "RatRig V-Core 3/4 3-point spherical ball-socket pivot + radial glide slot (30k N/mm axial, "
                "k_therm=1.5 MN/m, 240 N slip cap) with MITC4 eccentric bending & bimetallic PEI calibrated to "
                "Klipper BED_MESH_OUTPUT (41.5 µm thermal PTV vs 35–55 µm band)."
            ),
            "caveat": (
                "Glide-slot friction and spherical socket preload vary ±10–12% across bed sizes and assembly torque."
            ),
        },
        "floating_4pt": {
            "trust_pct": 84,
            "basis": (
                "4-point shoulder-bolt slotted tab model (12k N/mm axial, k_therm=3.2 MN/m, 185 N slip cap) with "
                "MITC4 eccentric bending & bimetallic PEI calibrated to Voron 2.4r2 1-tight/3-loose Klipper "
                "BED_MESH_OUTPUT (54.4 µm thermal PTV vs 40–75 µm band)."
            ),
            "caveat": (
                "Shoulder-screw friction and washer-stack tightening torque add ±10–12% thermal warp variance."
            ),
        },
        "cnc_kelvin_magnetic": {
            "trust_pct": 84,
            "basis": (
                "3-point CNC aluminum N48SH magnetic Kelvin coupling (cone + V-groove + flat, 42k N/mm axial, k_therm=0) "
                "+ Volkersen/Timoshenko bimetallic magnetic-PEI stack calibrated to MRW/ChaoticLab Klipper BED_MESH_OUTPUT (24.6 µm PTV)."
            ),
            "caveat": (
                "Magnetic preload pull-force (~30–60 N/corner) varies by magnet grade/temp and limits upward Tap separation margin."
            ),
        },
        "captive_belleville_kinematic": {
            "trust_pct": 84,
            "basis": (
                "Belleville disc-spring preloaded ball-groove kinematic coupling (52k N/mm axial, Tap-rated, k_therm=0) "
                "+ Volkersen/Timoshenko bimetallic magnetic-PEI stack calibrated to Klipper BED_MESH_OUTPUT (24.6 µm PTV vs 15–35 µm)."
            ),
            "caveat": (
                "Vendor Belleville stack stiffness and shoulder-bolt drag add ±10% vertical compliance spread."
            ),
        },
        "dual_leaf_ti_kinematic": {
            "trust_pct": 84,
            "basis": (
                "Stacked twin Ti-6Al-4V / 301 spring-steel leaf-flexure kinematic coupling (38k N/mm axial, Tap-rated, k_therm=0) "
                "+ Volkersen/Timoshenko bimetallic magnetic-PEI stack calibrated to Klipper BED_MESH_OUTPUT (24.6 µm PTV vs 15–35 µm)."
            ),
            "caveat": (
                "Twin-blade shear-spacer clamping and alloy preload tolerances vary ±10% across third-party kits."
            ),
        },
        "silicone_4pt": {
            "trust_pct": 80,
            "basis": (
                "70A free-elastomer silicone column compression & thermal shear (48 N/mm axial, 16 N/mm lateral, "
                "k_therm=32 kN/m) + bimetallic PEI calibrated to Klipper BED_MESH_OUTPUT (24.9 µm thermal PTV vs 20–40 µm band)."
            ),
            "caveat": (
                "Silicone compression set, knob-preload variance, and elastomer thermal softening add ±12–15% scatter."
            ),
        },
        "silicone_4pt_clamped": {
            "trust_pct": 82,
            "basis": (
                "Torqued M3×50 through-bolts in 16 mm 70A spacers (VzBot 330: 2,000 N/mm axial, 400 N/mm lateral, "
                "k_therm=850 kN/m, 240 N slip cap) + MITC4 eccentric bending & bimetallic PEI calibrated to VzBot 330 "
                "Klipper BED_MESH_OUTPUT (45.1 µm thermal PTV vs 35–60 µm band)."
            ),
            "caveat": (
                "Valid while knob preload exceeds vibration load; loose knobs revert toward free-elastomer silicone_4pt."
            ),
        },
        "springs_4pt": {
            "trust_pct": 80,
            "basis": (
                "Yellow die-spring axial & low-lateral-clamp model (16 N/mm axial, 4.5 N/mm lateral, k_therm=9 kN/m) "
                "+ bimetallic PEI calibrated to Klipper BED_MESH_OUTPUT (24.7 µm thermal PTV vs 20–40 µm band)."
            ),
            "caveat": (
                "Preload compression height, spring-end squareness, and screw-hole lateral slop add ±15% stiffness variance."
            ),
        },
        "craz_kinematic_ge5c": {
            "trust_pct": 84,
            "basis": (
                "Monolith Cra-Z 3-point GE5C spherical-bearing + linear-glide kinematic suspension (k_therm=0, "
                "high vertical axial stiffness, zero magnetic pull-off) + Volkersen/Timoshenko bimetallic PEI calibrated "
                "to Klipper BED_MESH_OUTPUT (24.6 µm thermal PTV vs 15–35 µm band)."
            ),
            "caveat": (
                "Spherical bearing radial clearance and slide-pin friction vary slightly with build alignment and chamber temp."
            ),
        },
    },
    "z_drive_mechanisms": {
        "tr8x4_pom": {
            "trust_pct": 85,
            "basis": (
                "ISO 2901 TR8×4 steel screw shaft EA/L (23.8 mm² root area) + POM nut chamber thermal softening "
                "(11 N/µm) wired in series with baked voron_trident_z_mount_* CAD CMS (3.82/6.13 MN/m ABS; +12 MN/m steel backbone) "
                "and 6 N Voron Tap elastic/gravity-preload verification (3.05 µm center tap)."
            ),
            "caveat": (
                "POM nut wear-in backlash (28 µm reversal) and motor-coupler axial compliance add ±8–10% variance."
            ),
        },
        "gt2_6mm_z_belt": {
            "trust_pct": 85,
            "basis": (
                "Voron 2.4 quad 6 mm Gates GT2 Z-belt height-dependent two-span closed-loop elasticity + reflected 80:16 "
                "gearbox/stepper torsional stiffness wired in series with baked voron_24_z_drive_{left,right} CAD CMS "
                "(12.67 MN/m ABS / 427 MN/m CNC 6061) and 6 N nozzle-tap benchmark (9.78 µm center tap)."
            ),
            "caveat": (
                "Actual flying-gantry Z compliance shifts ±8–10% with Z-belt pretension and pulley runout."
            ),
        },
        "sfu1204_bk10": {
            "trust_pct": 85,
            "basis": (
                "HIWIN/TBI rolled C7 SFU1204 ballscrew (70.9 mm² root area) + BK10 angular-contact thrust (85/150 N/µm, "
                "thermally invariant) wired in series with baked Trident Z-mount CAD CMS (128.7/206.7 MN/m CNC 6061; "
                "2.18 µm 6 N center tap, 8 µm reversal backlash)."
            ),
            "caveat": (
                "Rolled C7 ballscrew nut preload and BK10 locknut seating vary ±8–10% across budget vs genuine suppliers."
            ),
        },
        "sfu1605_fk12": {
            "trust_pct": 85,
            "basis": (
                "HIWIN/TBI heavy C7 SFU1605 ballscrew (133 mm² root area) + FK12/BK12 preloaded bearing (145/240 N/µm, "
                "thermally invariant) wired in series with Z-mount CMS and nozzle-tap compliance chain."
            ),
            "caveat": (
                "Real axial settle depends on FK12 bearing preload spacers and rigid Z-motor-plate mounting."
            ),
        },
        "tr8x8_pom": {
            "trust_pct": 83,
            "basis": (
                "ISO 2901 TR8×8 4-start screw shaft EA/L + POM nut chamber thermal softening (6.5 N/µm) + stepper thrust "
                "(14 N/µm) wired in series with Z-mount CAD CMS and 6 N nozzle-tap verification."
            ),
            "caveat": (
                "High 8 mm helix pitch allows unpowered backdrive and widens nut reversal backlash (45 µm nominal)."
            ),
        },
        "tr8x8_brass": {
            "trust_pct": 83,
            "basis": (
                "ISO 2901 TR8×8 screw shaft with thermally invariant C360 machined brass nut (38 N/µm) + motor thrust "
                "calibrated on VzBot 330 AWD belt-sync Z (15.25 MN/m total carrier, 4.08 µm 6 N center tap)."
            ),
            "caveat": (
                "Thread-flank lubrication state, wear-in clearance, and rigid vs jaw coupler flex alter effective Z backlash."
            ),
        },
        "gt2_9mm_z_belt": {
            "trust_pct": 84,
            "basis": (
                "9 mm Gates GT2 Z-belt height-dependent two-span closed-loop elasticity + reflected 80:16 gearbox/stepper "
                "stiffness wired in series with baked voron_24_z_drive_{left,right} CAD CMS (7.42 µm 6 N center tap)."
            ),
            "caveat": (
                "Assumes reinforced Z-drive brackets and 9 mm pulley stacks carrying +50% belt tension."
            ),
        },
        "tr8x4_wobblex": {
            "trust_pct": 83,
            "basis": (
                "ISO 2901 TR8×4 screw + POM thermal softening + MirageC WobbleX WS8 Oldham pin-and-ball decoupler "
                "(10.2 N/µm) wired in series with baked Trident Z-mount CAD CMS and gravity-preload Tap verification."
            ),
            "caveat": (
                "Pin-and-ball radial float and magnet retention stiffness under rapid upward Z-hop reversals add ±10% variance."
            ),
        },
        "sfu1204_wobblex": {
            "trust_pct": 82,
            "basis": (
                "SFU1204 + BK10 ballscrew paired with WobbleX decoupler (72 N/µm, 6 µm reversal backlash) wired in "
                "series with Z-mount CAD CMS and 6 N gravity-preload nozzle-tap verification."
            ),
            "caveat": (
                "Decoupler ball-pocket seating under heavy moving beds adds ±10% compliance variance."
            ),
        },
        "kevinakasam_belted_z": {
            "trust_pct": 82,
            "basis": (
                "KevinAkaSam 80:16 (5:1) dual-loop 6 mm GT2 height-dependent two-span belted-Z + reflected gearbox/stepper "
                "compliance & Keybak counterbalance calibrated in series with Z-mount stiffness."
            ),
            "caveat": (
                "Printed 80T pulley flex, belt tension balance, and Keybak spring hysteresis add ±10–12% variance."
            ),
        },
        "gt2_9mm_planetary_z": {
            "trust_pct": 81,
            "basis": (
                "Moving-bed 9 mm GT2 height-dependent two-span belted-Z + reflected 5:1 planetary gearbox & stepper "
                "torsional stiffness wired in series with Z-mount compliance."
            ),
            "caveat": (
                "Planetary gear tooth backlash (14 µm nominal) and torsional windup vary ±10–12% across compact gearbox brands."
            ),
        },
        "gt2_9mm_worm_z": {
            "trust_pct": 80,
            "basis": (
                "9 mm GT2 height-dependent two-span belted-Z + reflected 20:1 self-locking worm gearbox (25 MN/m) "
                "wired in series with Z-mount compliance."
            ),
            "caveat": (
                "Worm-wheel bronze/steel backlash (16 µm nominal) and gearbox axial thrust play add ±10–12% spread."
            ),
        },
    },
}


PROFILE_CONFIDENCE_REGISTRY: dict[str, dict[str, Any]] = {
    "extrusion_1515": {
        "trust_pct": 88,
        "basis": (
            "MakerBeam / LDO Voron V0 & Monolith Zero 15×15 mm T-slot extrusion cross-section "
            "(A=85 mm², 0.230 kg/m, Iy=Iz=0.198 cm⁴, open-slot J=0.032 cm⁴)."
        ),
        "caveat": (
            "Vendor 1515 wall thickness (1.0–1.3 mm) and center-bore tolerances shift Iy/Iz by ±8%."
        ),
    },
    "al_tube_12x12x1": {
        "trust_pct": 92,
        "basis": (
            "Exact closed thin-wall 12×12×1.0 mm aluminum square tube section inertia and "
            "Bredt-Batho torsional constant J for Monolith Zero lightweight gantry builds."
        ),
        "caveat": (
            "Corner fillet radius and local M3 rail-mounting wall dimpling are idealized."
        ),
    },
    "mgn9_rail": {
        "trust_pct": 92,
        "basis": (
            "3D Tet10 (P2) continuum FEA-calibrated solid HIWIN MGN9R linear rail with exact Gothic-arch "
            "raceway grooves and M3 counterbored holes (P=20 mm, 380 g/m) + baked build123d PBR GLB."
        ),
        "caveat": (
            "Minor ±3–5% variance from vendor raceway grind tolerances and bolt-head seating flatness."
        ),
    },
    "mgn12_rail": {
        "trust_pct": 92,
        "basis": (
            "3D Tet10 (P2) continuum FEA-calibrated solid HIWIN MGN12R linear rail with exact Gothic-arch "
            "raceway grooves and M3 counterbored holes (P=25 mm, 650 g/m) + baked build123d PBR GLB."
        ),
        "caveat": (
            "Minor ±3–5% variance from vendor raceway grind tolerances and bolt-head seating flatness."
        ),
    },
    "mgn15_rail": {
        "trust_pct": 92,
        "basis": (
            "3D Tet10 (P2) continuum FEA-calibrated solid HIWIN MGN15R linear rail with exact Gothic-arch "
            "raceway grooves and M3 counterbored holes (P=40 mm, 1060 g/m) + baked build123d PBR GLB."
        ),
        "caveat": (
            "Minor ±3–5% variance from vendor raceway grind tolerances and bolt-head seating flatness."
        ),
    },
    "hgh15_rail": {
        "trust_pct": 92,
        "basis": (
            "3D Tet10 (P2) continuum FEA-calibrated solid HIWIN HGR15R heavy 4-row linear rail with exact "
            "circular-arc raceway grooves and M4 counterbored holes (P=60 mm, 1450 g/m) + baked build123d PBR GLB."
        ),
        "caveat": (
            "Minor ±3–5% variance from vendor raceway grind tolerances and bolt-head seating flatness."
        ),
    },
    "mgn9_pocketed_rail": {
        "trust_pct": 88,
        "basis": (
            "3D Tet10 (P2) continuum FEA-calibrated periodic bottom-pocketed MGN9H linear rail "
            "(12×4.6×4.3 mm pockets between 8 mm M3 bolt bridges, 1.35 mm min raceway wall, 299 g/m, -22.9% mass, "
            "anisotropic mid-pocket vs bolt-bridge raceway backing compliance & baked build123d GLB)."
        ),
        "caveat": (
            "EDM/CNC grinding distortion and local M3 bolt-boss preload scatter add ±6–8% variance on thin-wall rails."
        ),
    },
    "mgn12_hollow_rail": {
        "trust_pct": 88,
        "basis": (
            "3D Tet10 (P2) continuum FEA-calibrated Monolith/ULS continuous bottom-channel Hollow MGN12H rail "
            "(6.6×4.98 mm open C-channel, 1.45 mm min raceway wall, 440.9 g/m, -33.1% mass; captures true "
            "open-section torsion loss J=130.1 mm⁴, shear κ_y=0.179, and anisotropic raceway backing compliance)."
        ),
        "caveat": (
            "Continuous open-channel walls are sensitive to post-machining raceway straightness and M3 mounting flatness."
        ),
    },
    "mgn12_pocketed_rail": {
        "trust_pct": 88,
        "basis": (
            "3D Tet10 (P2) continuum FEA-calibrated periodic bottom-pocketed MGN12H linear rail "
            "(16×6.6×5.5 mm pockets between 9 mm M3 bolt bridges, 1.45 mm min raceway wall, 484.6 g/m, -26.5% mass, "
            "2.26× bare GJ vs continuous hollow C-channel, anisotropic raceway backing compliance & baked GLB)."
        ),
        "caveat": (
            "Pocket-fillet residual stress and M3 bolt-bridge flatness tolerance add ±6–8% stiffness scatter."
        ),
    },
    "mgn15_pocketed_rail": {
        "trust_pct": 88,
        "basis": (
            "3D Tet10 (P2) continuum FEA-calibrated periodic bottom-pocketed MGN15H linear rail "
            "(30×8.4×7.0 mm pockets between 10 mm M3 bolt bridges, 1.65 mm min raceway wall, 730.6 g/m, -31.5% mass, "
            "anisotropic mid-pocket vs bolt-bridge raceway backing compliance & baked build123d GLB)."
        ),
        "caveat": (
            "Long 40 mm MGN15 bolt pitch increases mid-pocket vs bridge compliance ripple under high roll moments."
        ),
    },
    "unsupported_rail": {
        "trust_pct": 84,
        "basis": (
            "Bare unsupported linear rail X-gantry (e.g. Monolith Zero bare MGN12H X-rail) using exact "
            "oriented rail section inertia, zero phantom beam double-counting, and optional opposite-face steel backer."
        ),
        "caveat": (
            "End-clamping stiffness of the bare steel rail into XY-joint pockets depends on M3 bolt engagement length."
        ),
    },
    "cnc_al_monocoque_20x20": {
        "trust_pct": 86,
        "basis": (
            "3D Tet4 continuum FEA (141,182 tets) on parametric build123d 20×20×380 mm semi-monocoque Warren-truss "
            "CNC aluminum X-beam (solid 2 mm MGN12H front flange + 3-face pockets, 90.5 g Al 6061 / 238 g/m)."
        ),
        "caveat": (
            "Section properties capture pocket shear-lag and open-cell torsional knockdown; XY-joint end-boss clamping compliance is modeled separately."
        ),
    },
    "cnc_al_truss_voron24_lw": {
        "trust_pct": 84,
        "basis": (
            "3D Tet4 continuum FEA (95,649 tets) on GrabCAD Voron_XAxis_LW_300.STEP (Emil Wallberg, 19.6×19.6×380 mm "
            "4-face skeletonized truss, 64.8 g Al 6061 / 170.4 g/m)."
        ),
        "caveat": (
            "4-face skeletonized cutouts reduce torsional J and introduce local rail-boss compliance at thin diagonal webs."
        ),
    },
    "cnc_al_fysetc_v0_15x15": {
        "trust_pct": 84,
        "basis": (
            "3D Tet4 continuum FEA (93,440 tets) on GrabCAD Fysetc X-Beam [Voron0 200mm].step (Nicotina52, "
            "15×15×200 mm Warren-truss hollow CNC aluminum beam, 17.4 g Al 6061 / 87.1 g/m)."
        ),
        "caveat": (
            "Sized for 1515/Voron 0 micro-gantries; scaling to longer spans linearly scales section properties without end-boss length adjustment."
        ),
    },
}


def get_item_confidence(
    category: str,
    item_id: str,
    *,
    fallback_trust_pct: int = 60,
    fallback_basis: str = "Lumped engineering parameter estimate.",
    fallback_caveat: str = "Unverified catalog option; validate critical builds empirically.",
) -> dict[str, Any]:
    """Return a copy of the confidence metadata dict for ``category`` / ``item_id``."""
    cat_map = (
        PROFILE_CONFIDENCE_REGISTRY
        if category == "profiles"
        else ITEM_CONFIDENCE_REGISTRY.get(category, {})
    )
    entry = cat_map.get(item_id)
    if entry is None:
        return {
            "trust_pct": int(fallback_trust_pct),
            "basis": fallback_basis,
            "caveat": fallback_caveat,
        }
    return {
        "trust_pct": int(entry["trust_pct"]),
        "basis": str(entry["basis"]),
        "caveat": str(entry["caveat"]),
    }
