"""Engineering material property catalog for structural and thermal simulation."""

from __future__ import annotations

import math

from pydantic import BaseModel, Field


class MaterialSpec(BaseModel):
    """Isotropic / effective orthotropic structural and thermal material properties."""

    id: str
    name: str
    youngs_modulus_pa: float = Field(..., gt=0, description="Young's modulus E (Pa)")
    poisson_ratio: float = Field(0.3, gt=-0.5, lt=0.5, description="Poisson's ratio nu")
    shear_modulus_pa: float | None = Field(
        None, gt=0, description="Shear modulus G (Pa); defaults to E / (2*(1+nu))"
    )
    density_kg_m3: float = Field(..., gt=0, description="Density rho (kg/m^3)")
    yield_strength_pa: float = Field(..., gt=0, description="Yield / ultimate working stress (Pa)")
    thermal_conductivity_w_mk: float = Field(..., gt=0, description="Thermal conductivity k (W/m·K)")
    specific_heat_j_kgk: float = Field(..., gt=0, description="Specific heat capacity cp (J/kg·K)")
    thermal_expansion_1_k: float = Field(1e-5, ge=0, description="CTE alpha (1/K)")
    damping_ratio: float = Field(0.015, gt=0, lt=0.5, description="Structural damping ratio zeta")
    cost_per_kg_usd: float = Field(10.0, ge=0, description="Estimated raw material cost (USD/kg)")
    color_hex: str = "#8a95a5"

    # Orthotropic FDM / composite ratios and Mahieux-Reifsnider (2002) viscoelastic parameters
    e2_ratio: float = Field(1.0, gt=0, le=1.5, description="Transverse in-plane E2 / E1 ratio")
    e3_ratio: float = Field(1.0, gt=0, le=1.5, description="Interlayer Z-axis E3 / E1 ratio")
    g12_ratio: float | None = Field(
        None, gt=0, le=1.5, description="Optional in-plane shear G12 / E1 ratio"
    )
    tg_c: float | None = Field(None, description="Glass transition temperature Tg (C)")
    hdt_c: float | None = Field(None, description="Heat deflection temperature HDT (C)")
    e_rubbery_ratio: float = Field(
        0.05, gt=0, le=1.0, description="Post-Tg rubbery plateau modulus ratio E_r / E_g"
    )
    mahieux_m: float = Field(
        14.0, gt=1.0, le=40.0, description="Mahieux-Reifsnider Weibull transition exponent m"
    )
    is_printed_polymer: bool = False

    @property
    def effective_shear_modulus_pa(self) -> float:
        if self.shear_modulus_pa is not None:
            return self.shear_modulus_pa
        return self.youngs_modulus_pa / (2.0 * (1.0 + self.poisson_ratio))

    def modulus_retention_at_temp(self, temp_c: float) -> float:
        """Mahieux-Reifsnider (2002) Weibull E(T)/E(25 C) modulus retention factor.

        Uses Kelvin ratio ``(1 - e_r) * exp(-(T_K / T_ref_K)^m) + e_r`` normalized to 1.0 at 25 C.
        For semicrystalline CF-reinforced polymers where HDT far exceeds Tg (e.g. annealed PPS-CF),
        crystalline/fiber load bridging shifts the effective viscoelastic inflection ``T_ref_K``.
        Metals and thermoset CF plates (``tg_c is None``) return ``1.0``.
        """
        if self.tg_c is None:
            return 1.0

        hdt_val = self.hdt_c if self.hdt_c is not None else self.tg_c
        t_ref_c = self.tg_c + 0.32 * max(0.0, hdt_val - self.tg_c - 80.0)
        t_ref_k = max(t_ref_c + 273.15, 200.0)
        e_r = self.e_rubbery_ratio
        m = self.mahieux_m

        def _weibull_val(tc: float) -> float:
            t_k = max(tc + 273.15, 100.0)
            return (1.0 - e_r) * math.exp(-((t_k / t_ref_k) ** m)) + e_r

        ref_25 = max(_weibull_val(25.0), 1e-6)
        return _weibull_val(temp_c) / ref_25

    def effective_modulus_pa(self, temp_c: float = 25.0) -> float:
        """Temperature-dependent effective Young's modulus E(T) in Pascals."""
        return self.youngs_modulus_pa * self.modulus_retention_at_temp(temp_c)


MATERIAL_CATALOG: dict[str, MaterialSpec] = {
    "aluminum_6063_t5": MaterialSpec(
        id="aluminum_6063_t5",
        name="Aluminum 6063-T5 (Extrusion Alloy)",
        youngs_modulus_pa=68.9e9,
        poisson_ratio=0.33,
        density_kg_m3=2700.0,
        yield_strength_pa=145e6,
        thermal_conductivity_w_mk=200.0,
        specific_heat_j_kgk=900.0,
        thermal_expansion_1_k=23.4e-6,
        damping_ratio=0.012,
        cost_per_kg_usd=8.5,
        color_hex="#b8c2cc",
    ),
    "aluminum_6061_t6": MaterialSpec(
        id="aluminum_6061_t6",
        name="Aluminum 6061-T6 (Machined Plate/Block)",
        youngs_modulus_pa=68.9e9,
        poisson_ratio=0.33,
        density_kg_m3=2700.0,
        yield_strength_pa=276e6,
        thermal_conductivity_w_mk=167.0,
        specific_heat_j_kgk=896.0,
        thermal_expansion_1_k=23.6e-6,
        damping_ratio=0.012,
        cost_per_kg_usd=11.0,
        color_hex="#c6d0db",
    ),
    "aluminum_7075_t6": MaterialSpec(
        id="aluminum_7075_t6",
        name="CNC Aluminum 7075-T6 (Aerospace Billet)",
        youngs_modulus_pa=71.7e9,
        poisson_ratio=0.33,
        density_kg_m3=2810.0,
        yield_strength_pa=503e6,
        thermal_conductivity_w_mk=130.0,
        specific_heat_j_kgk=960.0,
        thermal_expansion_1_k=23.6e-6,
        damping_ratio=0.011,
        cost_per_kg_usd=24.0,
        color_hex="#cbd5e1",
    ),
    "mic6_aluminum": MaterialSpec(
        id="mic6_aluminum",
        name="MIC-6 Cast Aluminum Tooling Plate",
        youngs_modulus_pa=71.0e9,
        poisson_ratio=0.33,
        density_kg_m3=2800.0,
        yield_strength_pa=105e6,
        thermal_conductivity_w_mk=142.0,
        specific_heat_j_kgk=900.0,
        thermal_expansion_1_k=24.3e-6,
        damping_ratio=0.014,
        cost_per_kg_usd=18.0,
        color_hex="#d1d8e0",
    ),
    "carbon_fiber_tube": MaterialSpec(
        id="carbon_fiber_tube",
        name="Carbon Fiber Structural Tube (0/90 Roll-Wrapped)",
        youngs_modulus_pa=115.0e9,
        poisson_ratio=0.28,
        shear_modulus_pa=18.0e9,
        density_kg_m3=1550.0,
        yield_strength_pa=480e6,
        thermal_conductivity_w_mk=8.0,
        specific_heat_j_kgk=1050.0,
        thermal_expansion_1_k=1.2e-6,
        damping_ratio=0.022,
        cost_per_kg_usd=95.0,
        color_hex="#262a30",
    ),
    "steel_4140": MaterialSpec(
        id="steel_4140",
        name="Alloy Steel 4140 (Rails/Rods/Frame)",
        youngs_modulus_pa=205.0e9,
        poisson_ratio=0.29,
        density_kg_m3=7850.0,
        yield_strength_pa=655e6,
        thermal_conductivity_w_mk=42.6,
        specific_heat_j_kgk=473.0,
        thermal_expansion_1_k=12.2e-6,
        damping_ratio=0.008,
        cost_per_kg_usd=6.0,
        color_hex="#687382",
    ),
    "stainless_304": MaterialSpec(
        id="stainless_304",
        name="Stainless Steel 304 (Heatbreak/Fasteners)",
        youngs_modulus_pa=193.0e9,
        poisson_ratio=0.29,
        density_kg_m3=8000.0,
        yield_strength_pa=215e6,
        thermal_conductivity_w_mk=16.2,
        specific_heat_j_kgk=500.0,
        thermal_expansion_1_k=17.3e-6,
        damping_ratio=0.009,
        cost_per_kg_usd=9.0,
        color_hex="#9ba3ad",
    ),
    "titanium_grade5": MaterialSpec(
        id="titanium_grade5",
        name="Titanium Ti-6Al-4V (Bi-Metal Heatbreak / Backer)",
        youngs_modulus_pa=113.8e9,
        poisson_ratio=0.34,
        density_kg_m3=4430.0,
        yield_strength_pa=880e6,
        thermal_conductivity_w_mk=6.7,
        specific_heat_j_kgk=526.0,
        thermal_expansion_1_k=8.6e-6,
        damping_ratio=0.010,
        cost_per_kg_usd=75.0,
        color_hex="#8b8896",
    ),
    "brass_c360": MaterialSpec(
        id="brass_c360",
        name="Free-Machining Brass C360 (Nozzle)",
        youngs_modulus_pa=97.0e9,
        poisson_ratio=0.31,
        density_kg_m3=8500.0,
        yield_strength_pa=200e6,
        thermal_conductivity_w_mk=115.0,
        specific_heat_j_kgk=380.0,
        thermal_expansion_1_k=20.5e-6,
        damping_ratio=0.011,
        cost_per_kg_usd=16.0,
        color_hex="#d4af37",
    ),
    "copper_c110": MaterialSpec(
        id="copper_c110",
        name="Plated Copper C110 (High-Flow Heater Block/Nozzle)",
        youngs_modulus_pa=117.0e9,
        poisson_ratio=0.34,
        density_kg_m3=8940.0,
        yield_strength_pa=195e6,
        thermal_conductivity_w_mk=385.0,
        specific_heat_j_kgk=385.0,
        thermal_expansion_1_k=17.0e-6,
        damping_ratio=0.011,
        cost_per_kg_usd=22.0,
        color_hex="#c8754b",
    ),
    "hardened_steel": MaterialSpec(
        id="hardened_steel",
        name="Hardened Tool Steel (Abrasive Nozzle)",
        youngs_modulus_pa=210.0e9,
        poisson_ratio=0.29,
        density_kg_m3=7800.0,
        yield_strength_pa=900e6,
        thermal_conductivity_w_mk=24.0,
        specific_heat_j_kgk=460.0,
        thermal_expansion_1_k=11.5e-6,
        damping_ratio=0.008,
        cost_per_kg_usd=25.0,
        color_hex="#3d424a",
    ),
    "abs_printed": MaterialSpec(
        id="abs_printed",
        name="3D-Printed ABS (Voron Structural Parts)",
        youngs_modulus_pa=2.1e9,
        poisson_ratio=0.35,
        density_kg_m3=1040.0,
        yield_strength_pa=35e6,
        thermal_conductivity_w_mk=0.18,
        specific_heat_j_kgk=1400.0,
        thermal_expansion_1_k=85e-6,
        damping_ratio=0.045,
        cost_per_kg_usd=24.0,
        color_hex="#e63946",
        e2_ratio=0.88,
        e3_ratio=0.75,
        tg_c=102.0,
        hdt_c=88.0,
        e_rubbery_ratio=0.02,
        mahieux_m=18.0,
        is_printed_polymer=True,
    ),
    "pa12_cf_printed": MaterialSpec(
        id="pa12_cf_printed",
        name="3D-Printed PA12-CF15 (Stiff High-Temp Parts)",
        youngs_modulus_pa=6.2e9,
        poisson_ratio=0.36,
        density_kg_m3=1180.0,
        yield_strength_pa=68e6,
        thermal_conductivity_w_mk=0.35,
        specific_heat_j_kgk=1500.0,
        thermal_expansion_1_k=30e-6,
        damping_ratio=0.038,
        cost_per_kg_usd=85.0,
        color_hex="#333840",
        e2_ratio=0.42,
        e3_ratio=0.32,
        tg_c=56.0,
        hdt_c=135.0,
        e_rubbery_ratio=0.30,
        mahieux_m=11.0,
        is_printed_polymer=True,
    ),
    "pet_cf_printed": MaterialSpec(
        id="pet_cf_printed",
        name="3D-Printed PET-CF15 (Annealed Low-Creep)",
        youngs_modulus_pa=5.2e9,
        poisson_ratio=0.36,
        density_kg_m3=1290.0,
        yield_strength_pa=74e6,
        thermal_conductivity_w_mk=0.32,
        specific_heat_j_kgk=1350.0,
        thermal_expansion_1_k=28e-6,
        damping_ratio=0.036,
        cost_per_kg_usd=55.0,
        color_hex="#3b4252",
        e2_ratio=0.45,
        e3_ratio=0.34,
        tg_c=78.0,
        hdt_c=185.0,
        e_rubbery_ratio=0.35,
        mahieux_m=14.0,
        is_printed_polymer=True,
    ),
    "ppa_cf_printed": MaterialSpec(
        id="ppa_cf_printed",
        name="3D-Printed PPA-CF (Annealed High-Stiffness)",
        youngs_modulus_pa=11.8e9,
        poisson_ratio=0.35,
        density_kg_m3=1250.0,
        yield_strength_pa=165e6,
        thermal_conductivity_w_mk=0.42,
        specific_heat_j_kgk=1450.0,
        thermal_expansion_1_k=20e-6,
        damping_ratio=0.032,
        cost_per_kg_usd=135.0,
        color_hex="#2d333b",
        e2_ratio=0.42,
        e3_ratio=0.33,
        tg_c=130.0,
        hdt_c=210.0,
        e_rubbery_ratio=0.35,
        mahieux_m=15.0,
        is_printed_polymer=True,
    ),
    "pps_cf_printed": MaterialSpec(
        id="pps_cf_printed",
        name="3D-Printed PPS-CF (Annealed 260°C HDT)",
        youngs_modulus_pa=8.5e9,
        poisson_ratio=0.35,
        density_kg_m3=1310.0,
        yield_strength_pa=98e6,
        thermal_conductivity_w_mk=0.45,
        specific_heat_j_kgk=1200.0,
        thermal_expansion_1_k=18e-6,
        damping_ratio=0.030,
        cost_per_kg_usd=165.0,
        color_hex="#23272e",
        e2_ratio=0.44,
        e3_ratio=0.35,
        tg_c=98.0,
        hdt_c=252.0,
        e_rubbery_ratio=0.42,
        mahieux_m=15.0,
        is_printed_polymer=True,
    ),
    # Structural & Enclosure Panel Materials (MatWeb / 3A Composites Dibond datasheets)
    "acrylic_pmma": MaterialSpec(
        id="acrylic_pmma",
        name="Cast Acrylic Sheet (PMMA)",
        youngs_modulus_pa=3.1e9,
        poisson_ratio=0.37,
        shear_modulus_pa=1.13e9,
        density_kg_m3=1190.0,
        yield_strength_pa=65e6,
        thermal_conductivity_w_mk=0.19,
        specific_heat_j_kgk=1470.0,
        thermal_expansion_1_k=70e-6,
        damping_ratio=0.040,
        cost_per_kg_usd=14.0,
        color_hex="#93c5fd",
    ),
    "polycarbonate": MaterialSpec(
        id="polycarbonate",
        name="Polycarbonate Sheet (Lexan/PC)",
        youngs_modulus_pa=2.35e9,
        poisson_ratio=0.37,
        shear_modulus_pa=0.86e9,
        density_kg_m3=1200.0,
        yield_strength_pa=62e6,
        thermal_conductivity_w_mk=0.20,
        specific_heat_j_kgk=1200.0,
        thermal_expansion_1_k=68e-6,
        damping_ratio=0.045,
        cost_per_kg_usd=19.0,
        color_hex="#7dd3fc",
    ),
    "acm_dibond": MaterialSpec(
        id="acm_dibond",
        name="ACM / Dibond (Aluminum Composite Panel)",
        youngs_modulus_pa=22.0e9,
        poisson_ratio=0.33,
        shear_modulus_pa=8.2e9,
        density_kg_m3=1350.0,
        yield_strength_pa=85e6,
        thermal_conductivity_w_mk=1.2,
        specific_heat_j_kgk=1100.0,
        thermal_expansion_1_k=24e-6,
        damping_ratio=0.032,
        cost_per_kg_usd=24.0,
        color_hex="#475569",
    ),
    "aluminum_5052_sheet": MaterialSpec(
        id="aluminum_5052_sheet",
        name="Aluminum 5052-H32 Structural Sheet",
        youngs_modulus_pa=70.3e9,
        poisson_ratio=0.33,
        shear_modulus_pa=25.9e9,
        density_kg_m3=2680.0,
        yield_strength_pa=193e6,
        thermal_conductivity_w_mk=138.0,
        specific_heat_j_kgk=880.0,
        thermal_expansion_1_k=23.8e-6,
        damping_ratio=0.013,
        cost_per_kg_usd=12.0,
        color_hex="#94a3b8",
    ),
    "carbon_fiber_plate": MaterialSpec(
        id="carbon_fiber_plate",
        name="Carbon Fiber 3K Structural Sheet",
        youngs_modulus_pa=58.0e9,
        poisson_ratio=0.30,
        shear_modulus_pa=22.0e9,
        density_kg_m3=1550.0,
        yield_strength_pa=520e6,
        thermal_conductivity_w_mk=6.5,
        specific_heat_j_kgk=1050.0,
        thermal_expansion_1_k=2.0e-6,
        damping_ratio=0.025,
        cost_per_kg_usd=115.0,
        color_hex="#1e293b",
    ),
}


def get_material(material_id: str) -> MaterialSpec:
    """Look up a material from the catalog by ID."""
    if material_id not in MATERIAL_CATALOG:
        valid = ", ".join(sorted(MATERIAL_CATALOG))
        raise KeyError(f"Unknown material '{material_id}'. Available: {valid}")
    return MATERIAL_CATALOG[material_id]
