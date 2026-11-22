# PrintSim Physics, Mathematics & Real-World Hardware Reference

This document is the master index for PrintSim's internal multiphysics reference and first-principles engineering audit. Every equation, coordinate/sign convention, numerical verification benchmark, and real-world 3D printer hardware comparison is documented in six self-contained chapters under [`docs/physics/`](physics/).

---

## Reference Chapters

| Chapter | Domain | Audited Modules |
| :--- | :--- | :--- |
| [01 — Structural FEM & Static Solver](physics/01_structural_fem.md) | 12-DOF 3D beams, 24-DOF MITC4 shells, CBUSH springs, 2D bed plate FEM, static load cases & structural builder | `fea/beam3d.py`, `fea/shell_mitc4.py`, `fea/bed_plate_fem.py`, `fea/solver.py`, `geometry/builder.py` |
| [02 — Modal, Adjoint Sensitivity, RCSA & Shapers](physics/02_modal_adjoint_shaper.md) | Generalized eigenproblem, static adjoint & Fox–Kapoor derivatives, Dailey guard, RCSA LM-FBS, Klipper shapers | `fea/modal.py`, `fea/sensitivity.py`, `fea/rcsa.py`, `fea/klipper_shaper.py` |
| [03 — Kinematics, Components, Mechatronics & Ballbar](physics/03_kinematics_mechatronics_components.md) | FK/IK, stepper/belt drives, Hertz/Newmark/Iwan/Timoshenko/Mahieux models, LPV state-space, virtual ballbar & G-code | `kinematics/solvers.py`, `kinematics/collision.py`, `fea/components.py`, `fea/mechatronics.py`, `gcode/ballbar.py`, `gcode/simulator.py` |
| [04 — Solid FEM (Hex8/Tet10), CMS & SIMP Topology](physics/04_solid_fem_cms_topology.md) | Isoparametric Hex8, conformal `scikit-fem` Tet10, FDM Z-anisotropy, Craig–Bampton CMS + RBE3 VPT, 3D SIMP | `fea/solid_fem.py`, `fea/tet_fem.py`, `fea/superelement.py`, `fea/topology.py`, `fea/bracket_designer.py` |
| [05 — Hotend Thermal FVM, Rheology & Catalogs](physics/05_thermal_rheology_materials.md) | Multi-node hotend FVM, heat creep, melt-zone flow, nozzle pressure drop, filament/material/profile datasheets | `thermal/hotend.py`, `thermal/filaments.py`, `schema/materials.py`, `schema/profiles.py`, `schema/printer.py` |
| [06 — Real-World Printer Construction & Presets](physics/06_hardware_and_presets_audit.md) | Real printer topology vs. `builder.py` across Voron 2.4, Trident, Monolith AWD, DoomCube, VzBot, Bedslinger, Delta + all 8 YAMLs | `examples/*.yaml`, `geometry/builder.py`, `schema/printer.py`, `geometry/bom.py` |

---

## Consolidated Audit Triage — Confirmed Critical Bugs (15) — All Fixed & Verified

Every critical finding below was verified mathematically and numerically (via patch tests, central finite differences, closed-form Roark/Timoshenko/Leissa comparisons, or datasheet checks), and **all 15 critical issues plus all 5 moderate issue groups have been fixed and locked down with regression tests** (`tests/test_audit_fixes_*.py`, `258 passed`).

### Group A — Core Structural & Dynamics Solver Bugs (Fixed)

| ID | Status | File & Line | Bug Summary | Implemented Fix |
| :--- | :---: | :--- | :--- | :--- |
| **F1** | Fixed | `src/printsim/fea/shell_mitc4.py` | Grounded diagonal drilling penalty $k_{\text{drill}}$ on $\theta_z$ violated 3D rigid-body rotational invariance ($1{,}523\,\text{J/rad}^2$ per 3 mm Al quad under rigid $R_z$). | Replaced with frame-invariant Hughes–Brezzi $(\theta_z - \omega_z)^2$ penalty ($R_z$ energy $8.6\times 10^{-10}\,\text{J/rad}^2$, 6 zero RBMs). |
| **F2** | Fixed | `src/printsim/geometry/builder.py` | `bed_mount_spring` used `offset_m=(0,0,0)` across a 14 mm Z-gap (`bed_frame` to `bed_surface`), creating spurious rigid pitch energy. | Set `offset_m=(0.0, 0.0, -0.014)` matching physical Z separation. |
| **F3** | Fixed | `src/printsim/fea/bed_plate_fem.py` | Thermal gradient bow `w_grad` had inverted sign (concave-down dome instead of concave-up bowl for bottom heating). | Flipped sign to concave-up bowl for bottom heating and scaled curvature for $L_x \ne L_y$. |
| **F4** | Fixed | `src/printsim/fea/solver.py` | CoreXY belt-preload FBD applied spurious $-T_0$ Y pull on moving `y_car_l/r` and unbalanced $-T_0$ at rear corners. | Zeroed Y-preload on moving carriages and applied balanced $-2T_0$ at rear corners ($\sum F = 0, \sum M = 0$). |
| **F5** | Fixed | `src/printsim/fea/solver.py`, `modal.py` | Cartesian bedslinger `F_y` applied phantom $+m_x a_y$ on stationary `toolhead_cog` and omitted carriage mass from bed inertia. | Applied Y inertial load strictly to moving bed (`y_moving_mass_kg`) and removed phantom toolhead Y load. |
| **D1** | Fixed | `src/printsim/fea/rcsa.py` | Carriage rotational damping multiplied by `w / w0` instead of `w`, cutting damping by $1/\omega_0 \approx 583\times$. | Multiplied by `w` and filtered bare-carriage `f0` extraction (`nozzle_whip_ratio <= 1.45`). |
| **T1** | Fixed | `src/printsim/schema/profiles.py` | Open T-slot extrusion torsional constants $J$ were 3–5× too high ($J/I_p \approx 0.31$). | Replaced with Bosch/Misumi open-slot datasheet $I_t$ values ($J/I_p \approx 0.06\text{–}0.08 < 0.15$). |
| **M1** | Fixed | `src/printsim/gcode/simulator.py` | Junction-speed formula used $\sin(\phi/2)$ and omitted $(\sqrt{2}-1)$ factor. | Ported exact Klipper `toolhead.py` centripetal `junction_deviation` formula. |

### Group B — 3D Solid Continuum, CMS & Anisotropy Bugs (Fixed)

| ID | Status | File & Line | Bug Summary | Implemented Fix |
| :--- | :---: | :--- | :--- | :--- |
| **C1** | Fixed | `src/printsim/fea/solid_fem.py`, `bracket_designer.py` | Voxel sampler passed scalar `h_mm` to cubic Hex8, distorting thin bracket height ($10.3\times$ too stiff). | Generalized Hex8 `B`/`Ke`/`Me` to rectangular brick elements `(hx, hy, hz)` ($5.2\%$ parity vs. Tet10). |
| **C2** | Fixed | `src/printsim/fea/superelement.py` | `_build_coarse_bracket_grid` collapsed `(hx, hy, hz)` to `h_cubic`. | Switched CMS superelement grid to exact `(hx, hy, hz)` rectangular bricks and consistent ARPACK regularization. |
| **C3** | Fixed | `src/printsim/fea/solid_fem.py`, `tet_fem.py`, `topology.py` | Hardcoded `E_z = 0.68 * E` instead of `MaterialSpec.e3_ratio`. | Read `mat.e3_ratio` for printed polymers (`mat.is_printed_polymer`), verified strictly SPD across all materials. |

### Group C — Real-World Printer Architecture & Preset Fidelity Bugs (Fixed)

| ID | Status | File & Line | Bug Summary | Implemented Fix |
| :--- | :---: | :--- | :--- | :--- |
| **H1** | Fixed | `src/printsim/geometry/builder.py` | Voron 2.4 flying gantry (`belted_quad_z`) welded gantry corners straight to outer columns at fixed Z. | Decoupled `g_*` from `col_g_*` via 4× 6-DOF `z_joint_*` Z-belt/rail CBUSH springs, moved gantry dynamically with $Z(w_{\text{rel}})$, and anchored bed to twin base rails. |
| **H2** | Fixed | `src/printsim/geometry/builder.py` | `bracing: corner_gussets` inserted a phantom Ø10 mm full-face rear diagonal rod. | Removed diagonal rod from `corner_gussets` (retaining corner joint stiffening), reserving diagonal rods for `rear_cross` / `full_truss`. |
| **H3** | Fixed | `src/printsim/geometry/builder.py` | Delta modeled 3 single rods instead of 3 parallelogram pairs and ignored `delta_radius_mm`. | Modeled 6-rod parallelogram effector linkages honoring `delta_radius_mm` directly. |
| **H4** | Fixed | `src/printsim/geometry/builder.py` | Cartesian bedslinger clamped Z-portal base nodes directly to ground (`fixed=True`). | Anchored Z-portal uprights to flexible base extrusions (leaving only corner feet grounded) and moved Y-bed dynamically with `v_rel`. |

---

## Moderate Issues — Also Fixed & Verified

1. **Thermal Melt Flow & Rheology (Fixed — `hotend.py`, `filaments.py`)**: Replaced arithmetic FVM face conductance with Patankar harmonic mean, replaced inverted melt flow with true Graetz conduction (V6 `15.8`, Rapido `24.5`, UHF `38.8 mm³/s`, strictly monotonic with nozzle temp), added Power-Law/Rabinowitsch + Arrhenius pressure drop, and corrected PA12-CF (`56 °C`) & TPU (`−35 °C`, `heat_creep_temp_c=55 °C`) $T_g$.
2. **Dense Eigensolver Conditioning (Fixed — `modal.py`, `sensitivity.py`)**: Inverted dense `eigh(K_ff, M_ff)` to `eigh(M_ff, K_ff)` (`1.5e-11 Hz` match to shift-invert `eigsh`).
3. **CoreXY 45° Diagonal Torque & Circular Import (Fixed — `kinematics/solvers.py`)**: Applied exact $1/\sqrt{2}$ diagonal force share and $\sqrt{2}a/r$ rotor inertia, and broke the `solvers ↔ mechatronics` circular import.
4. **Belt Mass, Stepper Saturation & Preset Cleanup (Fixed — `schema/printer.py`, `examples/*.yaml`)**: Updated GT2-6mm linear mass (`11.5 → 8.3 g/m`), fixed `effective_holding_torque_nm` saturation curve, and calibrated all 8 reference YAMLs (`preset_id`, realistic `bed.mass_g`, `steps_per_rev: 400` on `delta_racer.yaml`, `belt_routing: monolith_short` on `voron_monolith_awd.yaml`, `clip_enclosure` acrylic + `mgn9_rail` Y-rails on `corexy_voron_style.yaml`).
5. **Static Envelope & SIMP OC Bracket (Fixed — `solver.py`, `topology.py`)**: Expanded `solver.py` combined load envelope to 8 unit-circle directions (pure X/Y + ±45°), and scaled `topology.py` `_oc_update` bisection upper bound dynamically.

---

## Verified Correct Subsystems

- **3D Beam FEM (`beam3d.py`)**: Exact 12×12 Euler–Bernoulli stiffness/mass/rotation signs (cantilever static error $\le 5.7\times 10^{-16}$, 4-element $f_1$ error $+0.0033\%$).
- **MITC4 Shell Patch Tests (`shell_mitc4.py`)**: Distorted-quad membrane ($2\times 10^{-16}$) and bending ($1.3\times 10^{-12}$) patch tests pass with zero shear locking; eccentric CBUSH matrices pass symmetry, PSD, 6-RBM, and equilibrium checks.
- **Static Adjoint & Fox–Kapoor Derivatives (`sensitivity.py`)**: Exact central-FD agreement (`0.0000%–0.03%` relative error on beams/lumped mass; Euler homogeneity sums `0.0000%`).
- **Timoshenko Free–Free RCSA & Klipper Shapers (`rcsa.py`, `klipper_shaper.py`)**: Cantilever tip receptance `0.00005%` error, zero reciprocity error, exact Klipper $V(f_n)$, $S_{90}/S_{180}$, `find_shaper_max_accel`, and Lorentzian PSD damping estimator (`0.75%` error).
- **FK/IK, Stepper Pole-Pairs & LPV Stability (`kinematics/`, `mechatronics.py`)**: Exact FK/IK round-trips ($< 2\times 10^{-13}\,\text{mm}$), exact 50/100 pole-pair $k_\theta$ scaling, stable LPV eigenvalues across all presets.
- **Solid FEM & Craig–Bampton CMS (`solid_fem.py`, `tet_fem.py`, `superelement.py`)**: All 23 materials SPD, Tet10 cantilever `-0.9%` vs. Timoshenko, VPT left-inverse `2.1e-14`, RBE3 wrench conservation `1.5e-14`, Guyan static condensation exact (`6.6e-20 m`), CB modal parity `≤ 0.41%` across 5 modes, SIMP sensitivity vs. FD `6.9e-5`.
