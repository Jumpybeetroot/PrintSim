"""Autonomous 2-stage AI agent discovery loop using PrintSim's MCP tool dispatcher.

Demonstrates how an external AI agent (or Python optimization script) can plug into
PrintSim via ``dispatch_tool`` to:
  1. Discover available presets and materials from ``list_catalog`` / ``get_preset_yaml``.
  2. Batch-evaluate and rank 5 structural/layout mutations on ``voron_trident_300``
     via ``mutate_and_compare_printer``.
  3. Sweep 4 parametric ``corner_gusset`` bracket designs via ``design_and_test_bracket``
     to identify the highest specific stiffness (N/mm/g).

Run with:
    uv run python examples/agent_discovery_loop.py
"""

from __future__ import annotations

from typing import Any

from printsim.mcp_server import dispatch_tool


def main() -> None:
    # ------------------------------------------------------------------
    # Stage 0: Capability & preset discovery (what an MCP client sees)
    # ------------------------------------------------------------------
    catalog = dispatch_tool("list_catalog", {})
    print(f"PrintSim Catalog Loaded: {len(catalog['presets'])} presets, "
          f"{len(catalog['materials'])} materials, "
          f"{len(catalog['extrusion_fillings'])} fillings, "
          f"{len(catalog['gantry_brackets'])} bracket alloys/polymers")

    base_preset = dispatch_tool("get_preset_yaml", {"preset_id": "voron_trident_300"})
    base_yaml = base_preset["yaml_text"]
    print(f"\nLoaded baseline preset: {base_preset['name']} ({base_preset['architecture']})\n")

    # ------------------------------------------------------------------
    # Stage 1: Batch-compare 5 hardware & layout mutations
    # ------------------------------------------------------------------
    candidates: list[dict[str, Any]] = [
        {
            "name": "PPA-CF Brackets + Epoxy Granite",
            "overrides": {
                "frame.extrusion_filling": "epoxy_granite",
                "gantry.bracket_material": "ppa_cf_printed",
                "gantry.belt.pretension_n": 45.0,
            },
        },
        {
            "name": "Bottom X-Rail + Inner Y-Rail + Low CoG",
            "overrides": {
                "gantry.x_rail_mount": "bottom",
                "gantry.y_rail_mount": "inner",
                "toolhead.cog_offset_mm": [0.0, -10.0, -12.0],
                "toolhead.nozzle_offset_mm": [0.0, -14.0, -50.0],
            },
        },
        {
            "name": "4mm Bolted Al-5052 Panels",
            "overrides": {
                "frame.enclosed_panels": True,
                "frame.panel_mounting": "bolted_structural",
                "frame.panel_material": "aluminum_5052_sheet",
                "frame.panel_thickness_mm": 4.0,
                "frame.bolt_panels_to_gantry": True,
            },
        },
        {
            "name": "AWD 9mm Fiberglass + 48V + PPA-CF",
            "overrides": {
                "gantry.drive_mode": "awd",
                "gantry.double_shear_pulleys": True,
                "gantry.bracket_material": "ppa_cf_printed",
                "gantry.belt.type": "gt2_9mm",
                "gantry.belt.core": "fiberglass",
                "gantry.belt.pretension_n": 65.0,
                "kinematics.xy_stepper.supply_voltage_v": 48.0,
                "kinematics.xy_stepper.run_current_a": 1.8,
            },
        },
        {
            "name": "Monolith Stack (AWD + Al Panels + UHPC)",
            "overrides": {
                "frame.extrusion_filling": "uhpc_grout",
                "frame.enclosed_panels": True,
                "frame.panel_mounting": "bolted_structural",
                "frame.panel_material": "aluminum_5052_sheet",
                "frame.panel_thickness_mm": 4.0,
                "frame.bolt_panels_to_gantry": True,
                "gantry.drive_mode": "awd",
                "gantry.double_shear_pulleys": True,
                "gantry.bracket_material": "ppa_cf_printed",
                "gantry.x_rail_mount": "bottom",
                "gantry.y_rail_mount": "inner",
                "gantry.belt.type": "gt2_9mm",
                "gantry.belt.core": "fiberglass",
                "gantry.belt.pretension_n": 65.0,
                "kinematics.xy_stepper.supply_voltage_v": 48.0,
                "kinematics.xy_stepper.run_current_a": 1.8,
            },
        },
    ]

    cmp_res = dispatch_tool(
        "mutate_and_compare_printer",
        {
            "yaml_text": base_yaml,
            "candidates": candidates,
            "rank_by": "composite",
            "filament_id": "abs",
            "toolhead_rel_pos": [0.5, 0.5, 0.5],
        },
    )

    base = cmp_res["baseline"]
    print("=== Stage 1: Printer Architecture Trade Study (Voron Trident 300) ===")
    print(
        f"Baseline -> Score: {base['composite_score']:.1f} | "
        f"f1: {base['fundamental_freq_hz']:.1f} Hz | "
        f"Rec Accel: {base['recommended_max_accel_mm_s2']:.0f} mm/s^2 | "
        f"Nozzle Flex: {base['nozzle_deflection_um']:.1f} um | "
        f"Cost: ${base['estimated_cost_usd']:.0f}\n"
    )
    for c in cmp_res["ranked_candidates"]:
        print(
            f"  #{c['rank']} {c['name']:<40s} | "
            f"Score {c['composite_score']:5.1f} ({c['delta_score']:+5.1f}) | "
            f"f1 {c['fundamental_freq_hz']:5.1f} Hz ({c['delta_freq_hz']:+5.1f}) | "
            f"Accel {c['recommended_max_accel_mm_s2']:6.0f} ({c['delta_accel_mm_s2']:+6.0f}) | "
            f"Flex {c['nozzle_deflection_um']:5.1f} um ({c['delta_deflection_um']:+5.1f}) | "
            f"Cost ${c['estimated_cost_usd']:6.0f} ({c['delta_cost_usd']:+5.0f})"
        )
    print(f"\nStage 1 Winner: {cmp_res['best_candidate_name']}")

    # ------------------------------------------------------------------
    # Stage 2: Parametric Corner Gusset Bracket FEA Sweep
    # ------------------------------------------------------------------
    bracket_variants: list[dict[str, Any]] = [
        {
            "label": "ABS Baseline (5mm wall, no rib)",
            "material_id": "abs_printed",
            "wall_thickness_mm": 5.0,
            "rib_thickness_mm": 0.0,
        },
        {
            "label": "PA12-CF Thin Ribbed (4mm wall, 4mm rib)",
            "material_id": "pa12_cf_printed",
            "wall_thickness_mm": 4.0,
            "rib_thickness_mm": 4.0,
        },
        {
            "label": "PPA-CF Optimized (4.5mm wall, 6mm rib)",
            "material_id": "ppa_cf_printed",
            "wall_thickness_mm": 4.5,
            "rib_thickness_mm": 6.0,
        },
        {
            "label": "CNC Al 6061-T6 (6mm wall, 6mm rib)",
            "material_id": "aluminum_6061_t6",
            "wall_thickness_mm": 6.0,
            "rib_thickness_mm": 6.0,
        },
    ]

    print("\n=== Stage 2: Parametric Corner-Gusset Bracket FEA Sweep (Hex8) ===")
    bracket_results: list[dict[str, Any]] = []
    for variant in bracket_variants:
        res = dispatch_tool(
            "design_and_test_bracket",
            {
                "bracket_type": "corner_gusset",
                "arm_length_mm": 40.0,
                "height_mm": 20.0,
                "wall_thickness_mm": variant["wall_thickness_mm"],
                "rib_thickness_mm": variant["rib_thickness_mm"],
                "bolt_hole_mm": 5.5,
                "fillet_mm": 2.0,
                "material_id": variant["material_id"],
                "force_n": [60.0, 0.0, 0.0],
                "engine": "hex8",
            },
        )
        res["label"] = variant["label"]
        bracket_results.append(res)
        print(
            f"  {variant['label']:<42s} | "
            f"Mass {res['mass_g']:5.2f} g | "
            f"Disp {res['max_displacement_um']:6.2f} um | "
            f"K {res['stiffness_n_per_mm']:7.1f} N/mm | "
            f"Spec.K {res['specific_stiffness_n_per_mm_per_g']:6.1f} N/mm/g | "
            f"SF {res['safety_factor']:5.1f}x"
        )

    winner = max(bracket_results, key=lambda r: r["specific_stiffness_n_per_mm_per_g"])
    printed_results = [r for r in bracket_results if "printed" in r["material_id"]]
    printed_winner = max(printed_results, key=lambda r: r["specific_stiffness_n_per_mm_per_g"])
    print(
        f"\nWinning Specific Stiffness (Overall): {winner['label']} "
        f"({winner['specific_stiffness_n_per_mm_per_g']:.1f} N/mm/g)"
    )
    print(
        f"Winning Specific Stiffness (3D-Printed): {printed_winner['label']} "
        f"({printed_winner['specific_stiffness_n_per_mm_per_g']:.1f} N/mm/g)"
    )


if __name__ == "__main__":
    main()
