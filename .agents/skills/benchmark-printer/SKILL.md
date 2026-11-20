---
name: benchmark-printer
description: Run a structured multiphysics engineering evaluation, trade study, and AI optimization loop on a 3D printer YAML specification using PrintSim. Use when testing or comparing printer frames, gantries, or hotends.
---

# Benchmark & Optimize a 3D Printer Design

Follow this procedure when evaluating or iterating on a 3D printer design in PrintSim.

## Steps

1. **Validate & Evaluate the Baseline**
   Run the CLI evaluator to obtain full multiphysics metrics (FEA static/modal, kinematics, thermal, virtual print):
   ```bash
   uv run printsim evaluate <spec.yaml>
   ```
   For programmatic comparison, pass `--json --no-segments`.

2. **Diagnose Bottlenecks**
   Inspect the five subsystem scores and diagnostics:
   - **Collisions**: Any `COLLISION:` diagnostic is a hard failure — widen/raise the frame or shrink the toolhead envelope first.
   - **Structural & Modal**: If `fundamental_freq_hz < 50 Hz` or `recommended_max_accel_mm_s2` is low, check whether frame bracing (`rear_cross` / `full_truss`), frame profile (`extrusion_3030`), X gantry beam (`cf_tube_20x20x2`), or belt stiffness (`gt2_9mm` / `aramid`) is the limiting factor.
   - **Kinematics**: Check `torque_safety_margin >= 1.5x` and `belt_slip_risk == false`.
   - **Hotend Thermal**: Check `heat_creep_risk == false` (`cold_side_temp_c < 52°C`) and whether `max_volumetric_flow_mm3_s` exceeds the G-code `peak_volumetric_flow_mm3_s`.

3. **Run Automated Design Exploration (Optional)**
   ```bash
   uv run printsim optimize <spec.yaml> --iterations 25 --objective balanced --out <improved.yaml>
   ```
   Objectives: `balanced`, `speed`, `rigidity`, `budget`.

4. **Verify & Export**
   Re-run `uv run printsim evaluate <improved.yaml>` to confirm improvements, and optionally export a 3D mesh:
   ```bash
   uv run printsim export-stl <improved.yaml> --out <printer.stl>
   ```
