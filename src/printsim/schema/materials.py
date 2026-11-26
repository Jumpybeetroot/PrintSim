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
    thermal_expansion_1_k: float = Field(1e-5, ge=-10e-6, description="CTE alpha (1/K)")
    damping_ratio: float = Field(0.015, gt=0, lt=0.5, description="Structural damping ratio zeta")
    cost_per_kg_usd: float = Field(10.0, ge=0, description="Estimated raw material cost (USD/kg)")
    color_hex: str = "#8a95a5"

    # Orthotropic FDM / composite / SLM build-direction ratios and Mahieux-Reifsnider (2002) viscoelastic parameters
    e2_ratio: float = Field(1.0, gt=0, le=1.5, description="Transverse in-plane E2 / E1 ratio")
    e3_ratio: float = Field(1.0, gt=0, le=1.5, description="Interlayer / build-direction Z-axis E3 / E1 ratio")
    z_strength_ratio: float = Field(
        1.0,
        gt=0,
        le=1.2,
        description="Build-direction Z-axis yield / interlayer strength ratio sigma_z / sigma_xy",
    )
    g12_ratio: float | None = Field(
        None, gt=0, le=1.5, description="Optional in-plane shear G12 / E1 ratio"
    )
    tg_c: float | None = Field(None, description="Glass transition temperature Tg (C)")
    hdt_c: float | None = Field(None, description="Heat deflection temperature HDT (C)")
    e_rubbery_ratio: float = Field(
        0.05, gt=0, le=1.0, description="Post-Tg rubbery plateau modulus ratio E_r / E_g"
    )
    shear_rubbery_ratio: float | None = Field(
        None,
        gt=0,
        le=1.0,
        description="Optional post-Tg shear modulus plateau G_r / G_g for continuous-fiber composites",
    )
    mahieux_m: float = Field(
        14.0, gt=1.0, le=40.0, description="Mahieux-Reifsnider Weibull transition exponent m"
    )
    is_printed_polymer: bool = False
    is_slm_metal: bool = False
    bolted_joint_stiffness_factor: float = Field(
        1.0,
        gt=0.1,
        le=1.2,
        description="Bolted mating-face contact stiffness factor vs CNC-machined billet (0.92-0.94 for as-sintered/bead-blasted SLM)",
    )
    surface_roughness_ra_um: float | None = Field(
        None,
        ge=0.0,
        description="Typical as-delivered surface roughness Ra (um; ~0.8-1.6 CNC vs ~6-15 SLM)",
    )
    porosity_pct: float = Field(
        0.0,
        ge=0.0,
        le=15.0,
        description="Typical residual internal micro-porosity (%)",
    )

    @property
    def effective_shear_modulus_pa(self) -> float:
        if self.shear_modulus_pa is not None:
            return self.shear_modulus_pa
        return self.youngs_modulus_pa / (2.0 * (1.0 + self.poisson_ratio))

    def _weibull_retention_at_temp(self, temp_c: float, rubbery_ratio: float) -> float:
        """Mahieux-Reifsnider (2002) Weibull retention normalized to 1.0 at 25 C."""
        if self.tg_c is None:
            return 1.0

        hdt_val = self.hdt_c if self.hdt_c is not None else self.tg_c
        t_ref_c = self.tg_c + 0.32 * max(0.0, hdt_val - self.tg_c - 80.0)
        t_ref_k = max(t_ref_c + 273.15, 200.0)
        e_r = rubbery_ratio
        m = self.mahieux_m

        def _weibull_val(tc: float) -> float:
            t_k = max(tc + 273.15, 100.0)
            return (1.0 - e_r) * math.exp(-((t_k / t_ref_k) ** m)) + e_r

        ref_25 = max(_weibull_val(25.0), 1e-6)
        return _weibull_val(temp_c) / ref_25

    def modulus_retention_at_temp(self, temp_c: float) -> float:
        """Mahieux-Reifsnider (2002) Weibull E(T)/E(25 C) modulus retention factor.

        Uses Kelvin ratio ``(1 - e_r) * exp(-(T_K / T_ref_K)^m) + e_r`` normalized to 1.0 at 25 C.
        For semicrystalline CF-reinforced polymers where HDT far exceeds Tg (e.g. annealed PPS-CF),
        crystalline/fiber load bridging shifts the effective viscoelastic inflection ``T_ref_K``.
        Metals and thermoset CF plates (``tg_c is None``) return ``1.0``.
        """
        return self._weibull_retention_at_temp(temp_c, self.e_rubbery_ratio)

    def shear_modulus_retention_at_temp(self, temp_c: float) -> float:
        """Temperature-dependent shear modulus retention G(T)/G(25 C).

        Continuous-fiber laminates with ``shear_rubbery_ratio`` model matrix-dominated shear
        softening near epoxy Tg separately from fiber-dominated longitudinal modulus retention.
        """
        ratio = self.shear_rubbery_ratio if self.shear_rubbery_ratio is not None else self.e_rubbery_ratio
        return self._weibull_retention_at_temp(temp_c, ratio)

    def effective_modulus_pa(self, temp_c: float = 25.0) -> float:
        """Temperature-dependent effective Young's modulus E(T) in Pascals."""
        return self.youngs_modulus_pa * self.modulus_retention_at_temp(temp_c)

    def effective_shear_modulus_at_temp_pa(self, temp_c: float = 25.0) -> float:
        """Temperature-dependent effective shear modulus G(T) in Pascals."""
        return self.effective_shear_modulus_pa * self.shear_modulus_retention_at_temp(temp_c)


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
        surface_roughness_ra_um=1.2,
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
        surface_roughness_ra_um=1.2,
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
    "atp5_aluminum": MaterialSpec(
        id="atp5_aluminum",
        name="ATP-5 / Alca 5 Cast Aluminum Tooling Plate (5083 DC-Cast)",
        youngs_modulus_pa=70.3e9,
        poisson_ratio=0.33,
        density_kg_m3=2660.0,
        yield_strength_pa=145.0e6,
        thermal_conductivity_w_mk=117.0,
        specific_heat_j_kgk=900.0,
        thermal_expansion_1_k=23.8e-6,
        damping_ratio=0.014,
        cost_per_kg_usd=18.0,
        color_hex="#d1d5db",
    ),
    # Carbon Fiber X-Gantry Tube Laminates — Classical Lamination Theory (CLT) derived from
    # Torayca T300 / T700S / M40J epoxy lamina datasheets (~58-60% Vf) and verified against
    # Rock West Composites, DragonPlate, and Carbonara (Monolith/Brazuka) structural specs.
    "carbon_fiber_3k_twill": MaterialSpec(
        id="carbon_fiber_3k_twill",
        name="Carbon Fiber Tube — 3K Woven 0/90 (Budget, Tg 78°C)",
        youngs_modulus_pa=64.0e9,  # CLT 50% 0° / 50% 90° Toray T300 3K 2x2 twill (~62-67 GPa)
        poisson_ratio=0.05,
        shear_modulus_pa=4.6e9,  # 0/90 woven laminate matrix-dominated in-plane G_xy
        density_kg_m3=1520.0,
        yield_strength_pa=420e6,
        thermal_conductivity_w_mk=5.5,
        specific_heat_j_kgk=1050.0,
        thermal_expansion_1_k=2.0e-6,  # CLT 0/90 T300 woven axial CTE ~ +2.0 ppm/K
        damping_ratio=0.024,
        cost_per_kg_usd=65.0,
        color_hex="#2c3138",
        e2_ratio=1.0,
        tg_c=78.0,
        hdt_c=72.0,
        e_rubbery_ratio=0.68,
        shear_rubbery_ratio=0.18,
        mahieux_m=22.0,
    ),
    "carbon_fiber_tube": MaterialSpec(
        id="carbon_fiber_tube",
        name="Carbon Fiber Tube — UD-Rich 0/90 (Std Epoxy, Tg 80°C)",
        youngs_modulus_pa=115.0e9,  # CLT ~80% 0° T700S UD / 20% 90° 3K outer weave
        poisson_ratio=0.28,
        shear_modulus_pa=5.0e9,  # D-02: 0/90 roll-wrapped CF laminate in-plane G_xy ~ 3.5-6.0 GPa
        density_kg_m3=1550.0,
        yield_strength_pa=480e6,
        thermal_conductivity_w_mk=8.0,
        specific_heat_j_kgk=1050.0,
        thermal_expansion_1_k=0.6e-6,  # CLT 80/20 T700S/epoxy axial CTE ~ +0.5-0.6 ppm/K
        damping_ratio=0.022,
        cost_per_kg_usd=95.0,
        color_hex="#262a30",
        e2_ratio=0.31,
        tg_c=80.0,
        hdt_c=75.0,
        e_rubbery_ratio=0.72,
        shear_rubbery_ratio=0.18,
        mahieux_m=22.0,
    ),
    "carbon_fiber_high_tg": MaterialSpec(
        id="carbon_fiber_high_tg",
        name="Carbon Fiber Tube — UD-Rich 0/90 (High-Tg 125°C)",
        youngs_modulus_pa=115.0e9,  # CLT ~80% 0° T700S UD / 20% 90° 3K outer weave
        poisson_ratio=0.28,
        shear_modulus_pa=5.0e9,  # D-02: 0/90 roll-wrapped CF laminate in-plane G_xy ~ 3.5-6.0 GPa
        density_kg_m3=1550.0,
        yield_strength_pa=480e6,
        thermal_conductivity_w_mk=8.0,
        specific_heat_j_kgk=1050.0,
        thermal_expansion_1_k=0.6e-6,
        damping_ratio=0.022,
        cost_per_kg_usd=135.0,
        color_hex="#1f242b",
        e2_ratio=0.31,
        tg_c=125.0,
        hdt_c=125.0,
        e_rubbery_ratio=0.72,
        shear_rubbery_ratio=0.18,
        mahieux_m=22.0,
    ),
    "carbon_fiber_torsional_45": MaterialSpec(
        id="carbon_fiber_torsional_45",
        name="Carbon Fiber Tube — ±45°/0° Torsional High-Tg (Carbonara/T700)",
        youngs_modulus_pa=87.0e9,  # CLT 55% 0° / 35% ±45° / 10% 90° T700S/epoxy (Rock West / Carbonara)
        poisson_ratio=0.39,
        shear_modulus_pa=15.7e9,  # ±45° torsional plies raise G_xy >3x vs 0/90 tubes
        density_kg_m3=1560.0,
        yield_strength_pa=540e6,
        thermal_conductivity_w_mk=7.2,
        specific_heat_j_kgk=1050.0,
        thermal_expansion_1_k=0.2e-6,  # CLT balanced ±45°/0° T700S axial CTE ~ +0.17 ppm/K
        damping_ratio=0.023,
        cost_per_kg_usd=165.0,
        color_hex="#1b2028",
        e2_ratio=0.37,
        tg_c=125.0,
        hdt_c=115.0,
        e_rubbery_ratio=0.62,
        shear_rubbery_ratio=0.55,
        mahieux_m=22.0,
    ),
    "carbon_fiber_m40j_hm": MaterialSpec(
        id="carbon_fiber_m40j_hm",
        name="Carbon Fiber Tube — M40J High-Modulus ±45°/0° (Tg 140°C)",
        youngs_modulus_pa=136.0e9,  # CLT 55% 0° / 35% ±45° / 10% 90° Toray M40J (377 GPa fiber)
        poisson_ratio=0.42,
        shear_modulus_pa=22.6e9,  # High-modulus ±45° plies approach aluminum G at 59% density
        density_kg_m3=1600.0,
        yield_strength_pa=580e6,
        thermal_conductivity_w_mk=18.0,
        specific_heat_j_kgk=980.0,
        thermal_expansion_1_k=-0.5e-6,  # CLT M40J negative fiber CTE (-0.83 ppm/K) -> ~ -0.5 ppm/K
        damping_ratio=0.018,
        cost_per_kg_usd=340.0,
        color_hex="#151920",
        e2_ratio=0.33,
        tg_c=140.0,
        hdt_c=135.0,
        e_rubbery_ratio=0.65,
        shear_rubbery_ratio=0.55,
        mahieux_m=22.0,
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
    # Bambu / Siraya Tech PPA-CF annealed TDS: ~10.8 GPa tensile/flexural modulus (dry annealed)
    "ppa_cf_printed": MaterialSpec(
        id="ppa_cf_printed",
        name="3D-Printed PPA-CF Annealed (10.8 GPa)",
        youngs_modulus_pa=10.8e9,
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
    # Bambu / Siraya Tech / Fiberon PPS-CF annealed TDS: ~9.2 GPa flexural/tensile modulus
    "pps_cf_printed": MaterialSpec(
        id="pps_cf_printed",
        name="3D-Printed PPS-CF Annealed (9.2 GPa, 252°C HDT)",
        youngs_modulus_pa=9.2e9,
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
    # TreeD PC-PBT-GF high-HDT structural filament (TreeD PCPBT GF Data Sheet rev. 1.0, synced with PrintDB)
    "pc_pbt_gf_printed": MaterialSpec(
        id="pc_pbt_gf_printed",
        name="3D-Printed PC-PBT-GF (6.1 GPa, 120°C HDT)",
        youngs_modulus_pa=6.1e9,
        poisson_ratio=0.36,
        density_kg_m3=1360.0,
        yield_strength_pa=78e6,
        thermal_conductivity_w_mk=0.20,
        specific_heat_j_kgk=1280.0,
        thermal_expansion_1_k=32e-6,
        damping_ratio=0.038,
        cost_per_kg_usd=85.0,
        color_hex="#334155",
        e2_ratio=0.52,
        e3_ratio=0.40,
        tg_c=138.0,
        hdt_c=120.0,
        e_rubbery_ratio=0.32,
        mahieux_m=14.0,
        is_printed_polymer=True,
    ),
    # SLM / LPBF 3D-Printed Metals (EOS / SLM Solutions / JLCPCB / IN3DTEC datasheets)
    # Distinct from CNC wrought billet: mild Z-build anisotropy (e3_ratio 0.93-0.97, z_strength_ratio 0.90-0.95),
    # rougher as-sintered/bead-blasted mating faces (Ra 8-10 um -> bolted_joint_stiffness_factor 0.92-0.94),
    # micro-porosity (0.3-0.8%), alloy-specific CTE (AlSi10Mg 20.5 vs 6061-T6 23.6 ppm/K), and lower k.
    "slm_aluminum": MaterialSpec(
        id="slm_aluminum",
        name="SLM 3D-Printed Aluminum (AlSi10Mg)",
        youngs_modulus_pa=68.0e9,
        poisson_ratio=0.33,
        shear_modulus_pa=25.6e9,
        density_kg_m3=2670.0,
        yield_strength_pa=240e6,
        thermal_conductivity_w_mk=140.0,
        specific_heat_j_kgk=900.0,
        thermal_expansion_1_k=20.5e-6,
        damping_ratio=0.015,
        cost_per_kg_usd=180.0,
        color_hex="#b8c4d0",
        e3_ratio=0.93,
        z_strength_ratio=0.90,
        is_slm_metal=True,
        bolted_joint_stiffness_factor=0.92,
        surface_roughness_ra_um=10.0,
        porosity_pct=0.5,
    ),
    "slm_ti6al4v": MaterialSpec(
        id="slm_ti6al4v",
        name="SLM 3D-Printed Titanium (Ti-6Al-4V Gr.5/23)",
        youngs_modulus_pa=110.0e9,
        poisson_ratio=0.34,
        shear_modulus_pa=41.0e9,
        density_kg_m3=4410.0,
        yield_strength_pa=960e6,
        thermal_conductivity_w_mk=6.8,
        specific_heat_j_kgk=526.0,
        thermal_expansion_1_k=8.8e-6,
        damping_ratio=0.012,
        cost_per_kg_usd=360.0,
        color_hex="#7c7d8c",
        e3_ratio=0.96,
        z_strength_ratio=0.95,
        is_slm_metal=True,
        bolted_joint_stiffness_factor=0.93,
        surface_roughness_ra_um=8.0,
        porosity_pct=0.3,
    ),
    # Laser-cut mild / 304 steel sheet metal gantry plates (Monolith Sheet Metal v1.5: 3-3.2mm & 4.7-5mm)
    "sheet_metal_steel": MaterialSpec(
        id="sheet_metal_steel",
        name="Laser-Cut Structural Steel Plate (3-5mm)",
        youngs_modulus_pa=200.0e9,
        poisson_ratio=0.29,
        shear_modulus_pa=77.5e9,
        density_kg_m3=7850.0,
        yield_strength_pa=250e6,
        thermal_conductivity_w_mk=45.0,
        specific_heat_j_kgk=490.0,
        thermal_expansion_1_k=12.0e-6,
        damping_ratio=0.008,
        cost_per_kg_usd=8.0,
        color_hex="#64748b",
        surface_roughness_ra_um=3.2,
    ),
    # SLM 316L Stainless, 17-4PH H900, Maraging Steel MS1, & CuCrZr Copper
    "slm_316l_steel": MaterialSpec(
        id="slm_316l_steel",
        name="SLM 316L Stainless Steel",
        youngs_modulus_pa=185.0e9,
        poisson_ratio=0.30,
        shear_modulus_pa=71.2e9,
        density_kg_m3=7970.0,
        yield_strength_pa=470e6,
        thermal_conductivity_w_mk=15.0,
        specific_heat_j_kgk=500.0,
        thermal_expansion_1_k=15.8e-6,
        damping_ratio=0.010,
        cost_per_kg_usd=210.0,
        color_hex="#94a3b8",
        e3_ratio=0.97,
        z_strength_ratio=0.92,
        is_slm_metal=True,
        bolted_joint_stiffness_factor=0.93,
        surface_roughness_ra_um=8.0,
        porosity_pct=0.3,
    ),
    "slm_17_4ph_steel": MaterialSpec(
        id="slm_17_4ph_steel",
        name="SLM 17-4PH Stainless Steel (H900 Aged)",
        youngs_modulus_pa=190.0e9,
        poisson_ratio=0.29,
        shear_modulus_pa=73.6e9,
        density_kg_m3=7780.0,
        yield_strength_pa=1050e6,
        thermal_conductivity_w_mk=16.5,
        specific_heat_j_kgk=460.0,
        thermal_expansion_1_k=10.8e-6,
        damping_ratio=0.010,
        cost_per_kg_usd=235.0,
        color_hex="#7c8798",
        e3_ratio=0.95,
        z_strength_ratio=0.94,
        is_slm_metal=True,
        bolted_joint_stiffness_factor=0.93,
        surface_roughness_ra_um=8.0,
        porosity_pct=0.3,
    ),
    "slm_maraging_steel": MaterialSpec(
        id="slm_maraging_steel",
        name="SLM Maraging Steel MS1 (1.2709)",
        youngs_modulus_pa=180.0e9,
        poisson_ratio=0.30,
        shear_modulus_pa=69.2e9,
        density_kg_m3=8050.0,
        yield_strength_pa=1000e6,
        thermal_conductivity_w_mk=20.0,
        specific_heat_j_kgk=450.0,
        thermal_expansion_1_k=10.3e-6,
        damping_ratio=0.009,
        cost_per_kg_usd=260.0,
        color_hex="#6b7280",
        e3_ratio=0.96,
        z_strength_ratio=0.94,
        is_slm_metal=True,
        bolted_joint_stiffness_factor=0.94,
        surface_roughness_ra_um=8.0,
        porosity_pct=0.3,
    ),
    "slm_cucrzr": MaterialSpec(
        id="slm_cucrzr",
        name="SLM Copper Alloy CuCrZr (Aged Conformal-Cooled)",
        youngs_modulus_pa=120.0e9,
        poisson_ratio=0.34,
        shear_modulus_pa=44.8e9,
        density_kg_m3=8860.0,
        yield_strength_pa=255e6,
        thermal_conductivity_w_mk=320.0,
        specific_heat_j_kgk=385.0,
        thermal_expansion_1_k=17.2e-6,
        damping_ratio=0.013,
        cost_per_kg_usd=290.0,
        color_hex="#b96d4b",
        e3_ratio=0.95,
        z_strength_ratio=0.90,
        is_slm_metal=True,
        bolted_joint_stiffness_factor=0.93,
        surface_roughness_ra_um=9.0,
        porosity_pct=0.8,
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

from printsim.schema.printdb_bridge import hydrate_material_catalog as _hydrate_material_catalog

_hydrate_material_catalog(MATERIAL_CATALOG)


def get_material(material_id: str) -> MaterialSpec:
    """Look up a material from the catalog by ID."""
    if material_id not in MATERIAL_CATALOG:
        valid = ", ".join(sorted(MATERIAL_CATALOG))
        raise KeyError(f"Unknown material '{material_id}'. Available: {valid}")
    return MATERIAL_CATALOG[material_id]
