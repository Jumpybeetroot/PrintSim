---
trigger: always_on
---

# Research & Reuse First Rule

Before implementing any new physics solver, geometry kernel, optimization algorithm, firmware model, or UI visualization in PrintSim, **always research existing open-source libraries, published engineering models, and prior art first**:

## 1. Reuse Before Reinventing
- **Check existing packages first**: Search PyPI, npm, GitHub, and academic/engineering references (`search_web` or a `research` subagent) before writing custom numerical solvers or CAD/optimization code from scratch.
- **Prefer proven optional backends + clean adapters**: Where heavy native libraries (e.g., `gmsh`, `build123d`, `scikit-fem`, `pymoo`, `optuna`, `CalculiX`) provide higher fidelity than a lightweight built-in solver, integrate them via optional dependency extras in `pyproject.toml` and a clean backend interface, keeping a fast pure-NumPy/SciPy fallback when helpful for zero-setup usage.
- **Cite prior art & datasheets**: Document the exact library, paper, firmware source (e.g., Klipper `shaper_defs.py`), or manufacturer datasheet (Misumi, Gates, E3D, MatWeb, Voron) behind any physical model or constant.

## 2. Do Deeper Research When Needed
- Whenever a task touches non-trivial engineering physics (e.g., bolted joint compliance, Cross-WLF melt rheology, Klipper input-shaper convolution, conformal meshing, CFD cooling, multi-objective optimization), run targeted research **before** coding to compare 2–3 established approaches and pick the most accurate, maintainable one.
- Validate custom models against either closed-form textbook benchmarks or outputs from established open-source reference tools.

## 3. Curated Open-Source Ecosystem for PrintSim
Evaluate these proven tools first when expanding each subsystem:
- **3D Meshing & Continuum FEA**: `gmsh` (conformal Tet4/Tet10 meshing from STEP/STL), `scikit-fem` + `meshio` (pure-Python weak-form FEM), `SfePy`, `CalculiX` (`pygccx`), `sectionproperties` (exact custom extrusion cross-section $A, I_x, I_y, J$).
- **Parametric CAD & STEP B-Rep**: `build123d` (OpenCASCADE/OCP) for exact STEP/STL brackets, drilled extrusions, and manufacturable parts; `trimesh` for lightweight mesh I/O.
- **Firmware, Kinematics & G-code**: Klipper `klippy/extras/shaper_defs.py` & `shaper_calibrate.py` (exact input-shaper math & PSD calibration), `FullControl` (`fullcontrol`) for programmatic G-code toolpaths, OrcaSlicer / PrusaSlicer CLI for realistic sliced benchmarks.
- **Multi-Objective Optimization**: `pymoo` (NSGA-II / NSGA-III Pareto fronts with constraints) and `optuna` (sample-efficient multi-objective TPE for expensive FEA evaluations).
- **Thermal & Fluid**: `ht` (Python heat-transfer correlations for convection/radiation/fins), published FDM melt-flow & Cross-WLF rheology literature, OpenFOAM / surrogate CFD datasets for 5015/CPAP cooling ducts.
