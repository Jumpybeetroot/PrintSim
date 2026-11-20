"""Polymer filament thermal, glass-transition, and non-Newtonian rheology catalog."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FilamentSpec(BaseModel):
    id: str
    name: str
    print_temp_c: float
    bed_temp_c: float
    glass_transition_c: float = Field(..., description="Tg (C) — cold-throat heat-creep threshold")
    density_kg_m3: float
    specific_heat_j_kgk: float
    latent_heat_j_kg: float
    viscosity_factor: float = Field(
        1.0, gt=0, description="Relative melt viscosity multiplier vs PLA (1.0)"
    )
    power_law_n: float = Field(0.38, gt=0.1, le=1.0, description="Shear-thinning index n")
    cooling_demand: float = Field(
        1.0, ge=0.1, le=2.0, description="Relative part-cooling demand for overhangs (PLA=1.0, ABS=0.35)"
    )


FILAMENT_CATALOG: dict[str, FilamentSpec] = {
    "pla": FilamentSpec(
        id="pla",
        name="PLA (Standard High-Speed)",
        print_temp_c=215.0,
        bed_temp_c=60.0,
        glass_transition_c=58.0,
        density_kg_m3=1240.0,
        specific_heat_j_kgk=1800.0,
        latent_heat_j_kg=45000.0,
        viscosity_factor=1.0,
        power_law_n=0.36,
        cooling_demand=1.0,
    ),
    "petg": FilamentSpec(
        id="petg",
        name="PETG (Functional Structural)",
        print_temp_c=245.0,
        bed_temp_c=80.0,
        glass_transition_c=80.0,
        density_kg_m3=1270.0,
        specific_heat_j_kgk=1450.0,
        latent_heat_j_kg=38000.0,
        viscosity_factor=1.35,
        power_law_n=0.40,
        cooling_demand=0.75,
    ),
    "abs": FilamentSpec(
        id="abs",
        name="ABS (Voron / Enclosed Parts)",
        print_temp_c=255.0,
        bed_temp_c=105.0,
        glass_transition_c=105.0,
        density_kg_m3=1040.0,
        specific_heat_j_kgk=1400.0,
        latent_heat_j_kg=32000.0,
        viscosity_factor=1.18,
        power_law_n=0.38,
        cooling_demand=0.40,
    ),
    "asa": FilamentSpec(
        id="asa",
        name="ASA (UV-Resistant Structural)",
        print_temp_c=260.0,
        bed_temp_c=105.0,
        glass_transition_c=100.0,
        density_kg_m3=1070.0,
        specific_heat_j_kgk=1420.0,
        latent_heat_j_kg=34000.0,
        viscosity_factor=1.22,
        power_law_n=0.38,
        cooling_demand=0.42,
    ),
    "pa12_cf": FilamentSpec(
        id="pa12_cf",
        name="PA12-CF15 (Carbon Nylon)",
        print_temp_c=285.0,
        bed_temp_c=100.0,
        glass_transition_c=95.0,
        density_kg_m3=1180.0,
        specific_heat_j_kgk=1650.0,
        latent_heat_j_kg=55000.0,
        viscosity_factor=1.75,
        power_law_n=0.44,
        cooling_demand=0.35,
    ),
    "tpu_95a": FilamentSpec(
        id="tpu_95a",
        name="TPU 95A (Flexible Elastomer)",
        print_temp_c=230.0,
        bed_temp_c=50.0,
        glass_transition_c=65.0,
        density_kg_m3=1210.0,
        specific_heat_j_kgk=1700.0,
        latent_heat_j_kg=35000.0,
        viscosity_factor=2.30,
        power_law_n=0.48,
        cooling_demand=0.85,
    ),
}


def get_filament(filament_id: str) -> FilamentSpec:
    if filament_id not in FILAMENT_CATALOG:
        valid = ", ".join(sorted(FILAMENT_CATALOG))
        raise KeyError(f"Unknown filament '{filament_id}'. Available: {valid}")
    return FILAMENT_CATALOG[filament_id]
