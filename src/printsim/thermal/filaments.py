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
    melt_transfer_factor: float = Field(
        1.0,
        gt=0.05,
        le=2.0,
        description=(
            "Relative Graetz shear-film core-melting coupling multiplier vs PLA (1.0), "
            "capturing amorphous viscous boundary resistance and fiber-core bridging"
        ),
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
        16200.0, gt=0.0, description="High-shear power-law consistency index K at t_ref_c (Pa*s^n)"
    )
    arrhenius_ea_j_mol: float = Field(
        75000.0, ge=0.0, description="Arrhenius flow activation energy Ea (J/mol)"
    )
    t_ref_c: float = Field(
        220.0, description="Reference temperature T_ref for power_law_k_pa_sn (C)"
    )
    # Cross-WLF 7-parameter melt viscosity model (strict SI):
    #   eta0(T, p) = D1 * exp(-A1 * (T - T*) / (A3 + (T - T*))), T* = D2 + D3 * p
    #   eta(gamma_dot, T, p) = eta0 / (1 + (eta0 * gamma_dot / tau*)^(1 - n))
    # High-shear limit matches K_eff(T) = eta0(T)^n * (tau*)^(1 - n).
    wlf_d1_pa_s: float = Field(
        2.45e13, gt=0.0, description="Cross-WLF zero-shear reference viscosity scale D1 (Pa*s)"
    )
    wlf_d2_k: float = Field(
        331.15, gt=150.0, description="Cross-WLF reference temperature D2 (K; Tg or Tm onset)"
    )
    wlf_d3_k_per_pa: float = Field(
        0.0, ge=0.0, description="Cross-WLF pressure shift coefficient D3 (K/Pa)"
    )
    wlf_a1: float = Field(
        32.0, gt=0.0, description="Cross-WLF dimensionless temperature sensitivity A1"
    )
    wlf_a3_k: float = Field(
        51.6, gt=1.0, description="Cross-WLF free-volume temperature constant A3 / A2_tilde (K)"
    )
    cross_tau_star_pa: float = Field(
        85000.0, gt=100.0, description="Cross-WLF critical shear stress tau* for shear thinning (Pa)"
    )
    cross_n: float = Field(
        0.36, gt=0.1, le=1.0, description="Cross-WLF high-shear power-law index n"
    )
    bagley_factor: float = Field(
        3.0, ge=0.0, le=12.0, description="Viscoelastic / extensional Bagley entry-loss factor e_B"
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
        # Melt-range averaged DSC specific heat (~1900-2050 J/(kg*K) over 25-215 C)
        specific_heat_j_kgk=1950.0,
        latent_heat_j_kg=45000.0,
        viscosity_factor=1.0,
        melt_transfer_factor=1.0,
        # Calibrated Cross-WLF bridging Ingeo 4032D (Di Lorenzo 2025), Prusa PLA (Bakrani Balani 2023),
        # Colon Quintana (2021, Ea=80.6 kJ/mol), and Serdeczny et al. (2019/2020) E3D V6 push-force data.
        # eta0(215 C) = 850 Pa*s, eta(1000 s^-1) = 158 Pa*s, K_eff(215 C) = 16,196 Pa*s^0.36.
        power_law_n=0.36,
        power_law_k_pa_sn=16200.0,
        arrhenius_ea_j_mol=80000.0,
        t_ref_c=215.0,
        wlf_d1_pa_s=2.4498e13,
        wlf_d2_k=331.15,
        wlf_d3_k_per_pa=0.0,
        wlf_a1=32.0,
        wlf_a3_k=51.6,
        cross_tau_star_pa=85000.0,
        cross_n=0.36,
        bagley_factor=3.0,
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
        # Melt-range averaged DSC specific heat; strictly amorphous (D-06: latent_heat = 0)
        specific_heat_j_kgk=1950.0,
        latent_heat_j_kg=0.0,
        viscosity_factor=1.35,
        melt_transfer_factor=0.58,
        # Calibrated Cross-WLF consistent with Schuller et al. (2023, eta0 ~ 600 Pa*s @ 240 C)
        # and CNC Kitchen PETG flow tests: eta0(245 C) = 750 Pa*s, eta(1000 s^-1) = 168 Pa*s.
        power_law_n=0.40,
        power_law_k_pa_sn=13700.0,
        arrhenius_ea_j_mol=65000.0,
        t_ref_c=245.0,
        wlf_d1_pa_s=1.3753e12,
        wlf_d2_k=353.15,
        wlf_d3_k_per_pa=0.0,
        wlf_a1=28.0,
        wlf_a3_k=51.6,
        cross_tau_star_pa=95000.0,
        cross_n=0.40,
        bagley_factor=3.4,
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
        # Melt-range averaged DSC specific heat; strictly amorphous (D-06: latent_heat = 0)
        specific_heat_j_kgk=2050.0,
        latent_heat_j_kg=0.0,
        viscosity_factor=1.18,
        melt_transfer_factor=0.58,
        # Calibrated Cross-WLF for amorphous ABS (Colon Quintana 2021 n=0.285; Kazmer et al. 2021;
        # CNC Kitchen & Monolith high-flow ABS benchmarks): eta0(255 C) = 1600 Pa*s, n = 0.30.
        power_law_n=0.30,
        power_law_k_pa_sn=20800.0,
        arrhenius_ea_j_mol=72000.0,
        t_ref_c=255.0,
        wlf_d1_pa_s=1.6647e13,
        wlf_d2_k=378.15,
        wlf_d3_k_per_pa=0.0,
        wlf_a1=31.0,
        wlf_a3_k=51.6,
        cross_tau_star_pa=62000.0,
        cross_n=0.30,
        bagley_factor=3.0,
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
        # Melt-range averaged DSC specific heat; strictly amorphous (D-06: latent_heat = 0)
        specific_heat_j_kgk=2050.0,
        latent_heat_j_kg=0.0,
        viscosity_factor=1.22,
        melt_transfer_factor=0.56,
        # Amorphous styrenic close to ABS with slightly higher zero-shear viscosity: eta0(260 C) = 1700 Pa*s.
        power_law_n=0.31,
        power_law_k_pa_sn=20900.0,
        arrhenius_ea_j_mol=72000.0,
        t_ref_c=260.0,
        wlf_d1_pa_s=2.5734e13,
        wlf_d2_k=373.15,
        wlf_d3_k_per_pa=0.0,
        wlf_a1=31.0,
        wlf_a3_k=51.6,
        cross_tau_star_pa=65000.0,
        cross_n=0.31,
        bagley_factor=3.1,
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
        extrudable_temp_c=220.0,
        thermal_conductivity_w_mk=0.35,
        density_kg_m3=1180.0,
        # Melt-range averaged DSC specific heat (PA12 melt diluted with 15 wt% CF); semi-crystalline
        specific_heat_j_kgk=2100.0,
        latent_heat_j_kg=55000.0,
        viscosity_factor=1.75,
        melt_transfer_factor=0.28,
        # Semi-crystalline melt-onset WLF (D2 = Tm = 451.15 K) with fiber-loaded entry resistance (e_B = 5.2):
        # eta0(285 C) = 1900 Pa*s, eta(1000 s^-1) = 307 Pa*s, calibrated to Voron/Bambu PA12-CF flow benches.
        power_law_n=0.38,
        power_law_k_pa_sn=25450.0,
        arrhenius_ea_j_mol=65000.0,
        t_ref_c=285.0,
        wlf_d1_pa_s=6.4886e6,
        wlf_d2_k=451.15,
        wlf_d3_k_per_pa=0.0,
        wlf_a1=16.5,
        wlf_a3_k=110.0,
        cross_tau_star_pa=125000.0,
        cross_n=0.38,
        bagley_factor=5.2,
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
        # Melt-range averaged DSC specific heat; negligible crystalline fusion peak (D-06)
        specific_heat_j_kgk=1950.0,
        latent_heat_j_kg=0.0,
        viscosity_factor=2.30,
        melt_transfer_factor=0.70,
        # Hard-segment melt-onset WLF (D2 = 438.15 K = 165 C): eta0(230 C) = 2800 Pa*s, n = 0.46, e_B = 4.2.
        power_law_n=0.46,
        power_law_k_pa_sn=22500.0,
        arrhenius_ea_j_mol=95000.0,
        t_ref_c=230.0,
        wlf_d1_pa_s=7.1604e5,
        wlf_d2_k=438.15,
        wlf_d3_k_per_pa=0.0,
        wlf_a1=14.5,
        wlf_a3_k=105.0,
        cross_tau_star_pa=130000.0,
        cross_n=0.46,
        bagley_factor=4.2,
        cooling_demand=0.85,
    ),
}

from printsim.schema.printdb_bridge import hydrate_filament_catalog as _hydrate_filament_catalog

_hydrate_filament_catalog(FILAMENT_CATALOG)


def get_filament(filament_id: str) -> FilamentSpec:
    if filament_id not in FILAMENT_CATALOG:
        valid = ", ".join(sorted(FILAMENT_CATALOG))
        raise KeyError(f"Unknown filament '{filament_id}'. Available: {valid}")
    return FILAMENT_CATALOG[filament_id]
