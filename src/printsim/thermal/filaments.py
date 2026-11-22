"""Polymer filament thermal, glass-transition, and non-Newtonian rheology catalog."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FilamentSpec(BaseModel):
    id: str
    name: str
    print_temp_c: float
    bed_temp_c: float
    glass_transition_c: float = Field(..., description="Glass transition temperature Tg (C)")
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
    thermal_conductivity_w_mk: float = Field(
        0.18, gt=0.01, description="Polymer thermal conductivity k (W/(m*K))"
    )
    melt_temp_c: float = Field(
        170.0, description="Crystalline melting or glass-flow onset temperature Tm (C)"
    )
    extrudable_temp_c: float = Field(
        180.0, description="Minimum extrudable core temperature T_core for Graetz melt limit (C)"
    )
    heat_creep_temp_c: float | None = Field(
        default=None,
        description="Cold-throat softening / filament buckling threshold (C); defaults to glass_transition_c",
    )
    power_law_k_pa_sn: float = Field(
        18000.0, gt=0.0, description="Power-law consistency index K at t_ref_c (Pa*s^n)"
    )
    arrhenius_ea_j_mol: float = Field(
        75000.0, ge=0.0, description="Arrhenius flow activation energy Ea (J/mol)"
    )
    t_ref_c: float = Field(
        220.0, description="Reference temperature T_ref for power_law_k_pa_sn (C)"
    )

    @property
    def effective_heat_creep_temp_c(self) -> float:
        """Cold-throat softening / filament buckling threshold (C)."""
        return self.heat_creep_temp_c if self.heat_creep_temp_c is not None else self.glass_transition_c

    @property
    def thermal_diffusivity_m2_s(self) -> float:
        """Polymer thermal diffusivity alpha = k / (rho * cp) in strict SI (m^2/s)."""
        return self.thermal_conductivity_w_mk / max(self.density_kg_m3 * self.specific_heat_j_kgk, 1.0)


FILAMENT_CATALOG: dict[str, FilamentSpec] = {
    "pla": FilamentSpec(
        id="pla",
        name="PLA (Standard High-Speed)",
        print_temp_c=215.0,
        bed_temp_c=60.0,
        glass_transition_c=58.0,
        heat_creep_temp_c=58.0,
        melt_temp_c=165.0,
        extrudable_temp_c=175.0,
        thermal_conductivity_w_mk=0.13,
        density_kg_m3=1240.0,
        specific_heat_j_kgk=1800.0,
        latent_heat_j_kg=45000.0,
        viscosity_factor=1.0,
        power_law_n=0.36,
        power_law_k_pa_sn=20000.0,
        arrhenius_ea_j_mol=80000.0,
        t_ref_c=215.0,
        cooling_demand=1.0,
    ),
    "petg": FilamentSpec(
        id="petg",
        name="PETG (Functional Structural)",
        print_temp_c=245.0,
        bed_temp_c=80.0,
        glass_transition_c=80.0,
        heat_creep_temp_c=80.0,
        melt_temp_c=180.0,
        extrudable_temp_c=195.0,
        thermal_conductivity_w_mk=0.20,
        density_kg_m3=1270.0,
        specific_heat_j_kgk=1450.0,
        latent_heat_j_kg=38000.0,
        viscosity_factor=1.35,
        power_law_n=0.40,
        power_law_k_pa_sn=16000.0,
        arrhenius_ea_j_mol=65000.0,
        t_ref_c=245.0,
        cooling_demand=0.75,
    ),
    "abs": FilamentSpec(
        id="abs",
        name="ABS (Voron / Enclosed Parts)",
        print_temp_c=255.0,
        bed_temp_c=105.0,
        glass_transition_c=105.0,
        heat_creep_temp_c=105.0,
        melt_temp_c=185.0,
        extrudable_temp_c=195.0,
        thermal_conductivity_w_mk=0.18,
        density_kg_m3=1040.0,
        specific_heat_j_kgk=1400.0,
        latent_heat_j_kg=32000.0,
        viscosity_factor=1.18,
        power_law_n=0.38,
        power_law_k_pa_sn=17000.0,
        arrhenius_ea_j_mol=72000.0,
        t_ref_c=255.0,
        cooling_demand=0.40,
    ),
    "asa": FilamentSpec(
        id="asa",
        name="ASA (UV-Resistant Structural)",
        print_temp_c=260.0,
        bed_temp_c=105.0,
        glass_transition_c=100.0,
        heat_creep_temp_c=100.0,
        melt_temp_c=185.0,
        extrudable_temp_c=195.0,
        thermal_conductivity_w_mk=0.18,
        density_kg_m3=1070.0,
        specific_heat_j_kgk=1420.0,
        latent_heat_j_kg=34000.0,
        viscosity_factor=1.22,
        power_law_n=0.38,
        power_law_k_pa_sn=17500.0,
        arrhenius_ea_j_mol=72000.0,
        t_ref_c=260.0,
        cooling_demand=0.42,
    ),
    "pa12_cf": FilamentSpec(
        id="pa12_cf",
        name="PA12-CF15 (Carbon Nylon)",
        print_temp_c=285.0,
        bed_temp_c=100.0,
        glass_transition_c=56.0,
        heat_creep_temp_c=56.0,
        melt_temp_c=178.0,
        extrudable_temp_c=205.0,
        thermal_conductivity_w_mk=0.35,
        density_kg_m3=1180.0,
        specific_heat_j_kgk=1650.0,
        latent_heat_j_kg=55000.0,
        viscosity_factor=1.75,
        power_law_n=0.44,
        power_law_k_pa_sn=20000.0,
        arrhenius_ea_j_mol=65000.0,
        t_ref_c=285.0,
        cooling_demand=0.35,
    ),
    "tpu_95a": FilamentSpec(
        id="tpu_95a",
        name="TPU 95A (Flexible Elastomer)",
        print_temp_c=230.0,
        bed_temp_c=50.0,
        glass_transition_c=-35.0,
        heat_creep_temp_c=55.0,
        melt_temp_c=165.0,
        extrudable_temp_c=185.0,
        thermal_conductivity_w_mk=0.19,
        density_kg_m3=1210.0,
        specific_heat_j_kgk=1700.0,
        latent_heat_j_kg=35000.0,
        viscosity_factor=2.30,
        power_law_n=0.48,
        power_law_k_pa_sn=35000.0,
        arrhenius_ea_j_mol=95000.0,
        t_ref_c=230.0,
        cooling_demand=0.85,
    ),
}


def get_filament(filament_id: str) -> FilamentSpec:
    if filament_id not in FILAMENT_CATALOG:
        valid = ", ".join(sorted(FILAMENT_CATALOG))
        raise KeyError(f"Unknown filament '{filament_id}'. Available: {valid}")
    return FILAMENT_CATALOG[filament_id]
