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
            "trust_pct": 86,
            "basis": (
                "Voron Design official StealthBurner + CW2 CAD geometry and extensive community "
                "scale-weighed assemblies (~415 g dressed with Dragon HF) paired with stock ADXL baselines."
            ),
            "caveat": (
                "Wiring harness/CAN toolboard mass, LED strip, and extruder motor variants shift real mass by ±12 g."
            ),
        },
        "stealthburner_tap": {
            "trust_pct": 84,
            "basis": (
                "Voron StealthBurner + Printed Tap official CAD dimensions, scale weights, and Voron "
                "community ADXL345 logs."
            ),
            "caveat": (
                "Printed Tap rail carriage hardware, magnet stack, and optical PCB mass vary ±15 g by print infill and fasteners."
            ),
        },
        "xol_compact": {
            "trust_pct": 82,
            "basis": (
                "Armchair Heavy Industries XOL repository CAD and community scale-weighed assemblies "
                "(~250 g dressed with Sherpa Mini + Rapido)."
            ),
            "caveat": (
                "Swapping 5015 vs 4010 blowers, extruder motors, or umbilical mounts shifts mass and CoG by ±10 g / ±3 mm."
            ),
        },
        "a4t_beacon": {
            "trust_pct": 80,
            "basis": (
                "Armchair Heavy Industries A4T repository CAD and community scale-weighed builds "
                "(~250 g with Wristwatch G2); CoG offset derived from CAD assembly layout."
            ),
            "caveat": (
                "CoG Z/Y overhang and dressed mass vary ±12 g across extruder motors and Beacon vs Cartographer mounts."
            ),
        },
        "dragon_burner": {
            "trust_pct": 78,
            "basis": (
                "Chirpy2605 Dragon Burner community scale-weighed builds (~245 g with Orbiter 2 + Dragon) "
                "and public CAD envelope."
            ),
            "caveat": (
                "Modular fan shrouds and extruder swaps (Orbiter 2 vs Sherpa vs Galileo 2) shift CoG by ±4 mm."
            ),
        },
        "archetype_cpap": {
            "trust_pct": 72,
            "basis": (
                "Armchair Heavy Industries Archetype CPAP lightweight build weights (~235 g with "
                "remote blower ducting) and CAD nozzle-drop layout."
            ),
            "caveat": (
                "Corrugated CPAP hose stiffness, pneumatic back-pressure drag, and dynamic hose mass coupling are not modeled."
            ),
        },
        "microwaved_bowden": {
            "trust_pct": 84,
            "basis": (
                "Official STEP CAD (micreathonn/Microwaved-Bowden-Toolhead MBT_Tricorn_V2.STEP) exact "
                "solid volume rollup & 75.6k-elem Tet4 FEA calibration (10,728 X / 3,563 Y N/mm, "
                "5,340 N/mm harmonic mean in SLM Ti-6Al-4V; scaled to AlSi10Mg/6061-T6 by E ratio): "
                "MBTCFD 16,259 mm³ (43.4 g SLM AlSi10Mg), Ti brace/shield (1.1 g), MGN12H (48.3 g), "
                "Monolith SLM belt clamp (~16 g), Beacon H, and ECAS/screws = 130 g excl. hotend; "
                "CoM centered between belts (-2 mm Z, -35.5 mm nozzle drop)."
            ),
            "caveat": (
                "16 mm CPAP hose, Bowden PTFE/Kevlar sheath dynamic drag, and SLM porosity (~99% dense) "
                "are not explicitly modeled; scale-weighing an assembled build would raise mass confidence >92%."
            ),
        },
        "heavy_uhf_long": {
            "trust_pct": 56,
            "basis": (
                "Generic long-nozzle UHF toolhead envelope estimate (440 g housing + 78 mm nozzle drop)."
            ),
            "caveat": (
                "Real mass, fan-duct bulk, and center-of-gravity overhang vary widely across custom multi-blower UHF shrouds."
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
            "trust_pct": 56,
            "basis": (
                "Lumped sliding-rail pitch knockdown (0.58×, +62 g, -6.5 mm CoG Y) matched to community "
                "CNC Tap ADXL drops (~12–20 Hz vs Beacon)."
            ),
            "caveat": (
                "Depends strongly on MGN9 rail preload, magnet pull force, and optical trigger clearance; no controlled A/B ADXL pair archived yet."
            ),
        },
        "printed_tap": {
            "trust_pct": 48,
            "basis": (
                "Lumped printed ABS/ASA Tap carriage knockdown (0.34×, +78 g, -8.5 mm CoG Y) matched to "
                "community Printed Tap ADXL drops (~18–30 Hz)."
            ),
            "caveat": (
                "Severe scatter across print stiffness, rail preload wear, magnet seating, and chamber thermal softening."
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
                "Chube_hotend_integration_specifications.pdf) & 56k-elem Tet4 FEA calibration "
                "(5,529/5,820 → 5,670 N/mm): 66.0 mm OAL w/ V6 nozzle (60.0 mm bare), Ø25.4×0.90 mm solid-top Ti tube, "
                "6× M3 Ø18 mm PCD, 52 mm meltzone, 74 g bare (CoM 34.82 mm) / 90 g dressed, 500 °C."
            ),
            "caveat": (
                "Even with 56k-elem Tet4 FEA-calibrated stiffness (5,670 N/mm), carriage-interface 6-bolt M3 preload "
                "and mount material introduce ±12% effective tip stiffness spread. Live CAD CMS raises trust to 86%."
            ),
        },
        "chube_air": {
            "trust_pct": 78,
            "basis": (
                "Official Luke's Laboratory STEP & Rev A drawing (CHUBE_INTEGRATION_REVA.STEP, "
                "Chube_hotend_integration_specifications.pdf) & 49.5k-elem Tet4 FEA calibration "
                "(2,140/2,357 → 2,250 N/mm): 66.0 mm OAL w/ V6 nozzle (60.0 mm bare), Ø25.4×0.90 mm Ti tube, "
                "6× M3 Ø18 mm PCD, 42 mm meltzone, 69 g bare (CoM 36.15 mm) / 79 g dressed, 500 °C."
            ),
            "caveat": (
                "Side cooling-window stiffness knockdown is captured by 49.5k-elem Tet4 FEA (2,250 N/mm); "
                "carriage-interface 6-bolt M3 preload introduces ±12% build variance. Live CAD CMS raises trust to 86%."
            ),
        },
        "chube_compact": {
            "trust_pct": 76,
            "basis": (
                "Official Luke's Laboratory SolidWorks STEP (Chube_Compact_integration_assembly.STEP) & "
                "31k-elem Tet4 FEA calibration (3,015/2,277 → 2,600 N/mm): 47.2 mm OAL w/ V6 nozzle, "
                "Ø22.23 mm Ti tube, 4× M3 Ø16 mm Dragon PCD, 33 mm meltzone, 80W 6×16 mm heater, "
                "50 g bare / 63 g dressed w/ sock, 500 °C."
            ),
            "caveat": (
                "31k-elem Tet4 FEA captures asymmetric 4-bolt flange compliance (2,600 N/mm); 4× M3 Dragon-pattern "
                "mount clamp stiffness in series limits effective assembly stiffness. Live CAD CMS raises trust to 86%."
            ),
        },
        "phaetus_dragon_hf": {
            "trust_pct": 70,
            "basis": (
                "Phaetus Dragon HF CAD/mass datasheet (46 g, 62.2 mm) and widely tested 4-Ti-standoff rigid "
                "tip-stiffness comparisons (~550 N/mm)."
            ),
            "caveat": (
                "Standoff screw tightness and printed M2.5 mounting ear creep alter real lateral nozzle stiffness."
            ),
        },
        "e3d_v6_groove": {
            "trust_pct": 68,
            "basis": (
                "E3D V6 CAD/mass specification (42 g, 62.3 mm) and classic J-head groove-mount lateral "
                "neck deflection measurements (~45 N/mm)."
            ),
            "caveat": (
                "Groove-collar radial play varies dramatically with printed vs machined clamp fit and axial retention."
            ),
        },
        "e3d_revo_voron": {
            "trust_pct": 64,
            "basis": (
                "E3D Revo Voron 4-bolt rigid CAD/mass datasheet (30 g, 52 mm); thin-wall 0.15 mm SS304 "
                "integrated nozzle-neck stiffness (~165 N/mm) estimated from geometry."
            ),
            "caveat": (
                "Thin-walled Revo neck bending stiffness has not been independently bench-tested under lateral 10 N load."
            ),
        },
        "phaetus_rapido_2_hf": {
            "trust_pct": 62,
            "basis": (
                "Phaetus Rapido 2 HF CAD/mass/wattage datasheet (50 g, 71 mm, 85 W); 3-Ti-screw mount "
                "tip stiffness (~500 N/mm) scaled from Dragon architecture."
            ),
            "caveat": (
                "Ceramic heater ring clamping and 3-screw standoff asymmetry lack direct static tip-deflection measurements."
            ),
        },
        "phaetus_dragon_uhf": {
            "trust_pct": 60,
            "basis": (
                "Phaetus Dragon UHF CAD/mass datasheet (56 g, 81.6 mm); reduced tip stiffness (~340 N/mm) "
                "estimated from unsupported copper UHF extender cantilever."
            ),
            "caveat": (
                "Copper melt-extender thread seating and long lower lever arm compliance are analytically estimated."
            ),
        },
        "phaetus_rapido_2_uhf": {
            "trust_pct": 58,
            "basis": (
                "Phaetus Rapido 2 UHF CAD/mass datasheet (62 g, 79.2 mm, 85 W); lower UHF nozzle stiffness "
                "(~320 N/mm) scaled from cantilever length."
            ),
            "caveat": (
                "Extended UHF nut-extender flex and 3-screw upper standoff stiffness are unbenchmarked."
            ),
        },
        "mellow_goliath_air": {
            "trust_pct": 58,
            "basis": (
                "Mellow Goliath Air datasheet (96 g, 69.2 mm, 100 W); dual-post structural stiffness "
                "(~750 N/mm) estimated from CAD geometry."
            ),
            "caveat": (
                "Long 50 mm copper melt core and wide Ti support posts have not been statically deflection-tested in-repo."
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
            "trust_pct": 84,
            "basis": (
                "Gates PowerGrip 2GT 6 mm catalog EA (18.9 kN × 2.0 dynamic factor) extensively validated "
                "against Voron Trident/2.4 CoreXY ADXL resonance logs."
            ),
            "caveat": (
                "Actual belt peak shifts with tension gauge accuracy, wrap angle, and EPDM thermal soak; scalar 2.0× dynamic factor awaits shaker curves."
            ),
        },
        "gt2_9mm": {
            "trust_pct": 82,
            "basis": (
                "Gates PowerGrip 2GT 9 mm catalog EA (28.35 kN × 2.0 dynamic factor) calibrated against "
                "RatRig, Monolith, and VzBot community ADXL logs."
            ),
            "caveat": (
                "Real resonance depends on accurate tension balancing and stiffer double-shear idlers at >80 N preload."
            ),
        },
        "gt2_12mm": {
            "trust_pct": 76,
            "basis": (
                "Gates PowerGrip 2GT 12 mm catalog EA (37.8 kN × 2.0 dynamic factor) benchmarked against "
                "high-tension AWD builds (e.g. dderg)."
            ),
            "caveat": (
                "Fewer empirical ADXL datasets than 6/9 mm 2GT; assumes rigid double-shear 12 mm idler stacks."
            ),
        },
        "gt2_15mm": {
            "trust_pct": 58,
            "basis": (
                "Linear per-mm width extrapolation from Gates PowerGrip 2GT (47.25 kN × 2.0 dynamic factor)."
            ),
            "caveat": (
                "Experimental custom wide-belt hardware; very sparse CoreXY ADXL data and extreme radial shaft/bearing loads."
            ),
        },
        "gt2_dual_9mm": {
            "trust_pct": 52,
            "basis": (
                "Experimental parallel-belt concept modeled as ideal 2× Gates 2GT 9 mm (56.7 kN × 2.0)."
            ),
            "caveat": (
                "Assumes perfect tension equalization between two stacked loops; real mismatch and tracking error lower stiffness."
            ),
        },
        "3gt_9mm": {
            "trust_pct": 44,
            "basis": (
                "Theoretical Gates 3GT (3MGT) tensile modulus ratio extrapolation (47.25 kN × 2.0) from 2GT data."
            ),
            "caveat": (
                "Unproven on high-reversal CoreXY printers; 3 mm pitch VFA ripple, chordal action, and ≥20 mm idler fit are uncalibrated."
            ),
        },
        "3gt_12mm": {
            "trust_pct": 42,
            "basis": (
                "Theoretical Gates 3GT 12 mm width-scaled tensile extrapolation (63.0 kN × 2.0)."
            ),
            "caveat": (
                "No CoreXY ADXL validation; heavy bending stiffness and custom 12 mm 3GT pulley mechanics are untested."
            ),
        },
        "3gt_15mm": {
            "trust_pct": 40,
            "basis": (
                "Theoretical Gates 3GT 15 mm width-scaled tensile extrapolation (78.75 kN × 2.0)."
            ),
            "caveat": (
                "Unproven on 3D printers; 30 g/m belt mass, bending drag, and custom CNC pulley stacks remain theoretical."
            ),
        },
        "at3_10mm": {
            "trust_pct": 40,
            "basis": (
                "Breco/Elatech AT3 10 mm industrial linear-actuator C_spez datasheet (80 kN Hookean steel/TPU)."
            ),
            "caveat": (
                "Unproven on serpentine CoreXY printers; steel-cord reverse-bend fatigue and 92 ShA TPU chamber softening (>55 °C) are uncalibrated."
            ),
        },
        "at3_16mm": {
            "trust_pct": 38,
            "basis": (
                "Breco/Elatech AT3 16 mm industrial linear-actuator C_spez datasheet (135 kN steel/TPU)."
            ),
            "caveat": (
                "Unproven on CoreXY printers; requires ≥20 mm back-idlers to avoid wire fatigue and suffers TPU stiffness loss in heated chambers."
            ),
        },
        "at3_gen3_16mm": {
            "trust_pct": 38,
            "basis": (
                "ContiTech Synchroflex AT3 Gen III 16 mm industrial C_spez datasheet (250 kN HD steel/TPU)."
            ),
            "caveat": (
                "Theoretical upper-bound only; zero CoreXY ADXL evidence, severe bending drag, and TPU cold-flow in heated chambers."
            ),
        },
        "htd3m_15mm": {
            "trust_pct": 36,
            "basis": (
                "Elatech HTD 3M 15 mm industrial C_spez datasheet (127.5 kN steel/TPU)."
            ),
            "caveat": (
                "Theoretical only; ≥30 mm reverse-bend idler requirement and HTD semicircular tooth backlash are impractical on compact CoreXY frames."
            ),
        },
        "at5_16mm": {
            "trust_pct": 34,
            "basis": (
                "Breco/Elatech AT5 16 mm heavy industrial C_spez datasheet (280 kN steel/TPU)."
            ),
            "caveat": (
                "Purely theoretical industrial upper-bound; ≥60 mm back-idler minimum and 5 mm pitch are incompatible with Voron/VzBot-scale gantries."
            ),
        },
    },
    "belt_routings": {
        "stock_corexy": {
            "trust_pct": 74,
            "basis": (
                "Exact inner pulley-to-pulley envelope (W_p, D_p, u, v) baseline with 1.00× free-span "
                "and conventional M5 bolt idler stiffness."
            ),
            "caveat": (
                "Approximates exact 3D stacked belt-plane Z-offsets and individual XY-joint idler bolt overhangs."
            ),
        },
        "monolith_short": {
            "trust_pct": 62,
            "basis": (
                "Scalar 0.86× belt free-span and 1.45× live-shaft idler stiffness approximation from "
                "Monolith short-path CAD comparisons."
            ),
            "caveat": (
                "Replaces explicit 3D pulley coordinates with scalar multipliers; exact span reduction depends on frame aspect ratio."
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
            "trust_pct": 58,
            "basis": (
                "Voron standard ABS/ASA (2.1 GPa, 95 °C HDT, 45 N creep limit) calibrated loosely against "
                "stock Voron Trident/2.4 ADXL baselines."
            ),
            "caveat": (
                "Real printed XY-joint stiffness and creep vary ±25% with infill %, perimeter walls, chamber temp, and heat-set inserts."
            ),
        },
        "ppa_cf_printed": {
            "trust_pct": 56,
            "basis": (
                "Bambu/Siraya/MyTechFun annealed PPA-CF TDS (10.8 GPa, 196 °C HDT, 105 N creep limit) "
                "with lumped double-shear/AWD bracket scaling."
            ),
            "caveat": (
                "FDM layer anisotropy, annealing protocol, and unmeshed XY-joint rib geometry add ±20% real stiffness spread."
            ),
        },
        "pps_cf_printed": {
            "trust_pct": 54,
            "basis": (
                "Bambu/Siraya/MyTechFun annealed PPS-CF TDS (9.2 GPa, 252 °C HDT, 115 N creep limit) "
                "with lumped gantry bracket scaling."
            ),
            "caveat": (
                "Inter-layer Z-bond strength, annealing shrinkage, and bolt-boss bearing stiffness vary by printer tuning."
            ),
        },
        "pet_cf_printed": {
            "trust_pct": 54,
            "basis": (
                "Bambu/Polymaker/MyTechFun annealed PET-CF15 TDS (5.2 GPa, 182 °C HDT, 75 N creep limit) "
                "with lumped bracket scaling."
            ),
            "caveat": (
                "Printed perimeter count, M5 bearing stack clamping, and actual bracket CAD topology are not explicitly meshed."
            ),
        },
        "pa12_cf_printed": {
            "trust_pct": 52,
            "basis": (
                "Polymaker/MyTechFun conditioned PA12-CF15 TDS (6.2 GPa, 155 °C HDT, 65 N creep limit) "
                "with lumped bracket scaling."
            ),
            "caveat": (
                "Nylon moisture absorption lowers real modulus by up to 30–40% and accelerates preload creep under AWD belt tension."
            ),
        },
    },
    "rail_qualities": {
        "hiwin_genuine": {
            "trust_pct": 86,
            "basis": (
                "HIWIN MG/HG catalog Hertzian Gothic-arch ball-contact baseline (1.00× stiffness, 1.00× drag) "
                "with Hung et al. (2011) 15% skirt compliance."
            ),
            "caveat": (
                "Assumes genuine Grade C/H ball sorting, flat extrusion mounting face, and proper rail bolt torque."
            ),
        },
        "midgrade_matched": {
            "trust_pct": 68,
            "basis": (
                "Estimated ball-sharing knockdown (0.88× stiffness, 1.05× drag) for hand-matched 3D printer "
                "rails (LDO / HoneyBadger / Fysetc)."
            ),
            "caveat": (
                "No direct roll-moment bench curve yet; lot-to-lot preload consistency varies across vendors."
            ),
        },
        "precision_thk_nsk": {
            "trust_pct": 60,
            "basis": (
                "Estimated high-conformity Grade P ground raceway multiplier (1.22× stiffness, 0.82× drag) "
                "for THK/NSK/IKO blocks."
            ),
            "caveat": (
                "Unbenchmarked on 3D printer aluminum frames where extrusion face flatness often limits true Grade P gains."
            ),
        },
        "budget_generic": {
            "trust_pct": 48,
            "basis": (
                "Lumped loose-tolerance generic clone knockdown (0.58× stiffness, 1.45× drag) assuming ~60% "
                "active load-sharing balls."
            ),
            "caveat": (
                "Massive batch-to-batch scatter from undersized balls, rough raceways, missing seals, or zero effective preload."
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
            "trust_pct": 64,
            "basis": (
                "Voron standard M5 blind-joint lumped spring (12k N/mm, 22k N·m/rad in 2020) calibrated "
                "loosely to unpaneled frame frequency trends."
            ),
            "caveat": (
                "Depends strongly on M5 socket-head preload torque, extrusion end-tapping squareness, and anodized friction."
            ),
        },
        "gusset_plate": {
            "trust_pct": 58,
            "basis": (
                "External CNC/cast gusset plate lumped joint estimate (22k N/mm, 42k N·m/rad in 2020, Δζ=0.002)."
            ),
            "caveat": (
                "Untested on a static torque-angle bench; real stiffness varies with plate thickness and T-nut seating."
            ),
        },
        "l_bracket": {
            "trust_pct": 56,
            "basis": (
                "Internal/stamped L-bracket lumped joint estimate (3.5k N/mm, 4.5k N·m/rad in 2020, Δζ=0.005)."
            ),
            "caveat": (
                "Die-cast/stamped brackets have low torsional restraint and slip easily under diagonal racking loads."
            ),
        },
        "corner_cube": {
            "trust_pct": 54,
            "basis": (
                "3-way cast/CNC corner cube lumped joint estimate (38k N/mm, 85k N·m/rad in 2020, Δζ=0.001)."
            ),
            "caveat": (
                "No static moment-angle bench data; real stiffness varies with end-bore thread bite and cube seating."
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
            "trust_pct": 64,
            "basis": (
                "Kiln-dried silica sand (1,550 kg/m³, unbonded, Δζ=0.028) granular mass and Coulomb "
                "particle-friction damping estimate."
            ),
            "caveat": (
                "Compaction density, grain moisture, and settling gaps over time shift real damping and added mass."
            ),
        },
        "epoxy_granite": {
            "trust_pct": 62,
            "basis": (
                "85% quartz + epoxy mineral-cast CNC literature (E=30 GPa, 2,250 kg/m³, bonded, Δζ=0.036) "
                "coupled via cavity void fraction."
            ),
            "caveat": (
                "Air voids, cure shrinkage, and wall-bond adhesion to anodized aluminum cavities cause ±20% EI/damping spread."
            ),
        },
        "uhpc_grout": {
            "trust_pct": 58,
            "basis": (
                "Non-shrink UHPC/CSA cementitious grout estimate (E=40 GPa, 2,350 kg/m³, bonded, Δζ=0.030)."
            ),
            "caveat": (
                "No direct filled-extrusion tap test; micro-shrinkage debonding from smooth T-slot cores can drop composite EI."
            ),
        },
        "lead_shot": {
            "trust_pct": 58,
            "basis": (
                "Fine #8-#9 lead shot granular impact damping estimate (6,800 kg/m³, unbonded, Δζ=0.042)."
            ),
            "caveat": (
                "Packing fraction, internal cavity rattle, and vertical settling inside uprights vary widely."
            ),
        },
        "pu_foam": {
            "trust_pct": 52,
            "basis": (
                "Expanding polyurethane acoustic foam estimate (45 kg/m³, E=0.02 GPa, Δζ=0.010)."
            ),
            "caveat": (
                "Unbenchmarked acoustic deadening model; minimal structural stiffness effect and uneven closed-cell expansion."
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
            "trust_pct": 58,
            "basis": (
                "Typical 60-70A Voron rubber/TPU foot stiffness estimate (360 N/mm axial, 140 N/mm lateral, Δζ=0.006)."
            ),
            "caveat": (
                "Elastomer durometer scatter, compression preload, and benchtop surface friction alter whole-frame rocking frequency."
            ),
        },
        "paver_on_foam": {
            "trust_pct": 55,
            "basis": (
                "18 kg concrete paver + dense acoustic foam base-inertia and impedance estimate (240 N/mm axial, "
                "95 N/mm lateral, Δζ=0.022)."
            ),
            "caveat": (
                "Foam thickness, open-cell vs EVA compression set, and frame-to-paver coupling vary greatly across builds."
            ),
        },
        "anti_vib_pads": {
            "trust_pct": 48,
            "basis": (
                "Soft isolator pad estimate (HULA / Sorbothane: 85 N/mm axial, 32 N/mm lateral, Δζ=0.015)."
            ),
            "caveat": (
                "Highly nonlinear hyperelastic shear stiffness, temperature sensitivity, and desk impedance cause wide rocking variance."
            ),
        },
    },
    "bed_mounts": {
        "rigid_4pt": {
            "trust_pct": 78,
            "basis": (
                "Solid metal 4-point standoffs (65k N/mm axial, 40k N/mm lateral) with Timoshenko radial "
                "thermal-buckling taco model."
            ),
            "caveat": (
                "Mechanical Z stiffness is high-confidence, but real thermal taco onset varies with bolt-hole clearance and slip."
            ),
        },
        "trident_3pt_standoffs": {
            "trust_pct": 74,
            "basis": (
                "Stock Voron Trident Bakelite standoffs + GE5C spherical joint CAD dimensions and slotted "
                "radial thermal relief (18k N/mm axial)."
            ),
            "caveat": (
                "Slot friction, washer clamping torque, and Bakelite compression creep add ±15–20% thermal restraint spread."
            ),
        },
        "kinematic_3pt": {
            "trust_pct": 70,
            "basis": (
                "Slocum Precision Machine Design Hertzian steel ball-in-V-groove kinematic contact stiffness "
                "(32k N/mm axial, zero radial taco)."
            ),
            "caveat": (
                "Assumes clean hardened ball/pin contact without V-groove brinelling or magnet pull-off under rapid Z/Tap loads."
            ),
        },
        "whopping_pochard_kinematic": {
            "trust_pct": 70,
            "basis": (
                "Spring-steel radial leaf-flexure kinematic model (24k N/mm axial, Tap-rated, zero radial taco) "
                "from Whopping Pochard geometry."
            ),
            "caveat": (
                "Actual leaf-blade thickness tolerance and clamp-block fastener preload shift vertical compliance by ±15%."
            ),
        },
        "spherical_3pt_ratrig": {
            "trust_pct": 68,
            "basis": (
                "RatRig V-Core 3/4 3-point spherical ball-socket pivot + radial glide slot estimate "
                "(30k N/mm axial, 1.5 MN/m thermal restraint)."
            ),
            "caveat": (
                "Glide-slot friction and spherical socket preload vary across bed sizes and build assembly torque."
            ),
        },
        "floating_4pt": {
            "trust_pct": 66,
            "basis": (
                "4-point shoulder-bolt slotted tab estimate (12k N/mm axial, 2.8 MN/m thermal restraint)."
            ),
            "caveat": (
                "Shoulder-screw friction and tab-bending stiffness depend heavily on washer stack and tightening torque."
            ),
        },
        "cnc_kelvin_magnetic": {
            "trust_pct": 64,
            "basis": (
                "3-point CNC aluminum N48SH magnetic Kelvin coupling (cone + V-groove + flat, 42k N/mm axial) "
                "Hertzian estimate."
            ),
            "caveat": (
                "Magnetic preload pull-force (~30–60 N/corner) varies by magnet grade/temp and limits upward Tap separation margin."
            ),
        },
        "captive_belleville_kinematic": {
            "trust_pct": 62,
            "basis": (
                "Belleville disc-spring preloaded ball-groove kinematic estimate (52k N/mm axial, Tap-rated)."
            ),
            "caveat": (
                "Vendor Belleville stack stiffness, shoulder-bolt drag, and V-groove contact wear are unbenchmarked."
            ),
        },
        "dual_leaf_ti_kinematic": {
            "trust_pct": 62,
            "basis": (
                "Stacked twin Ti-6Al-4V / 301 spring-steel leaf-flexure kinematic estimate (38k N/mm axial, Tap-rated)."
            ),
            "caveat": (
                "Twin-blade shear-spacer clamping and alloy yield/preload tolerances vary across third-party kits."
            ),
        },
        "silicone_4pt": {
            "trust_pct": 56,
            "basis": (
                "Lumped 70A silicone column compression and thermal shear estimate (48 N/mm axial, 16 N/mm lateral)."
            ),
            "caveat": (
                "Silicone compression set, knob-preload variance, and elastomer thermal softening cause ±25–35% stiffness scatter."
            ),
        },
        "springs_4pt": {
            "trust_pct": 52,
            "basis": (
                "Lumped yellow die-spring axial and lateral shear estimate (16 N/mm axial, 4.5 N/mm lateral)."
            ),
            "caveat": (
                "Preload compression height, spring-end squareness, and screw-hole lateral slop cause ±30–40% stiffness variance."
            ),
        },
    },
    "z_drive_mechanisms": {
        "tr8x4_pom": {
            "trust_pct": 74,
            "basis": (
                "ISO 2901 TR8×4 steel screw shaft EA/L (23.8 mm² root area) in series with Voron Trident "
                "POM flanged nut (11 N/µm) and thrust baseline."
            ),
            "caveat": (
                "POM nut backlash wear (28 µm nominal) and motor-coupler axial compliance vary by build."
            ),
        },
        "gt2_6mm_z_belt": {
            "trust_pct": 74,
            "basis": (
                "Voron 2.4 quad 6 mm Gates GT2 Z-belt baseline (0.18 N/µm = 180 N/mm per corner, 5:1 "
                "reduction, 18 µm backlash)."
            ),
            "caveat": (
                "Actual flying-gantry Z corner compliance depends on Z-belt pretension and printed Z-drive/idler housing flex."
            ),
        },
        "sfu1204_bk10": {
            "trust_pct": 72,
            "basis": (
                "HIWIN/TBI rolled C7 SFU1204 ballscrew (70.9 mm² root area) + BK10 angular-contact thrust "
                "catalog stiffnesses (85 / 150 N/µm)."
            ),
            "caveat": (
                "Rolled C7 ballscrew nut preload and BK10 locknut seating vary across budget vs genuine suppliers."
            ),
        },
        "sfu1605_fk12": {
            "trust_pct": 72,
            "basis": (
                "HIWIN/TBI heavy C7 SFU1605 ballscrew (133 mm² root area) + FK12/BK12 preloaded bearing "
                "catalog stiffnesses (145 / 240 N/µm)."
            ),
            "caveat": (
                "Real axial settle depends on FK12 bearing preload spacers and rigid Z-motor-plate mounting."
            ),
        },
        "tr8x8_pom": {
            "trust_pct": 70,
            "basis": (
                "ISO 2901 TR8×8 4-start screw shaft EA/L in series with POM nut (6.5 N/µm) and stepper "
                "thrust bearing (14 N/µm)."
            ),
            "caveat": (
                "High 8 mm helix pitch allows unpowered backdrive and widens nut axial backlash (45 µm nominal)."
            ),
        },
        "tr8x8_brass": {
            "trust_pct": 68,
            "basis": (
                "ISO 2901 TR8×8 screw shaft with C360 machined brass nut (38 N/µm) and motor thrust bearing estimate."
            ),
            "caveat": (
                "Thread-flank lubrication state, wear-in clearance, and rigid vs jaw coupler flex alter effective Z backlash."
            ),
        },
        "gt2_9mm_z_belt": {
            "trust_pct": 68,
            "basis": (
                "9 mm Gates GT2 Z-belt width-scaled drive estimate (0.27 N/µm per corner, 5:1 reduction)."
            ),
            "caveat": (
                "Assumes reinforced Z-drive brackets and 9 mm pulley stacks capable of carrying +50% belt tension."
            ),
        },
        "tr8x4_wobblex": {
            "trust_pct": 60,
            "basis": (
                "ISO 2901 TR8×4 screw + MirageC WobbleX WS8 Oldham pin-and-ball decoupler axial compliance "
                "estimate (10.2 N/µm)."
            ),
            "caveat": (
                "Pin-and-ball radial float and magnet retention stiffness under rapid Tap reversals are unbenchmarked."
            ),
        },
        "sfu1204_wobblex": {
            "trust_pct": 60,
            "basis": (
                "SFU1204 + BK10 ballscrew paired with lumped WobbleX decoupler axial compliance estimate "
                "(72 N/µm, 6 µm backlash)."
            ),
            "caveat": (
                "Decoupler ball-pocket seating and axial play under heavy moving beds lack direct static deflection data."
            ),
        },
        "kevinakasam_belted_z": {
            "trust_pct": 58,
            "basis": (
                "KevinAkaSam 80:16 (5:1) dual-loop 6 mm GT2 belted-Z + Keybak counterbalance estimate "
                "(0.21 N/µm per corner)."
            ),
            "caveat": (
                "Printed 80T pulley flex, belt tension balance, and Keybak spring hysteresis vary widely across builds."
            ),
        },
        "gt2_9mm_planetary_z": {
            "trust_pct": 56,
            "basis": (
                "Moving-bed 9 mm GT2 belted-Z + 5:1 planetary gearbox lumped compliance estimate (0.31 N/µm per corner)."
            ),
            "caveat": (
                "Planetary gear tooth backlash (14 µm nominal) and torsional windup vary markedly between compact gearbox brands."
            ),
        },
        "gt2_9mm_worm_z": {
            "trust_pct": 54,
            "basis": (
                "9 mm GT2 belted-Z + 20:1 self-locking worm gearbox lumped compliance estimate (0.34 N/µm per corner)."
            ),
            "caveat": (
                "Worm-wheel bronze/steel backlash, housing flex, and gearbox axial thrust play are not directly bench-measured."
            ),
        },
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
    cat_map = ITEM_CONFIDENCE_REGISTRY.get(category, {})
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
