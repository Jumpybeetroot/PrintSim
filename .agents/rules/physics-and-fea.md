---
trigger: glob
globs: src/printsim/fea/**,src/printsim/thermal/**,src/printsim/kinematics/**,src/printsim/gcode/**
---

# Physics & FEA Solver Rules

1. **Strict SI Internal Representation**: Convert all `mm`, `g`, `MPa`, and `mm/s` inputs to `m`, `kg`, `Pa`, and `m/s` at solver entry. Convert outputs back to suffixed display units only in report serialization (`to_dict()`).
2. **Matrix Symmetry & Conditioning**:
   - Stiffness (`K`) and mass (`M`) matrices must remain symmetric (`0.5 * (A + A.T)` before eigensolves).
   - Guard against zero-length members, singular/under-constrained trusses, and division by zero in compliance or damping ratios.
3. **Sign Conventions (3D Beam Elements)**:
   - Local `x` runs from `start_node` to `end_node`.
   - Bending in local `x-y` uses `Iz` with `v'(x) = +theta_z`.
   - Bending in local `x-z` uses `Iy` with `w'(x) = -theta_y` (watch opposite off-diagonal signs for `theta_y`!).
4. **Analytical Benchmark Requirement**:
   - Every modification to `fea/beam3d.py`, `fea/solver.py`, or `fea/modal.py` must continue passing the analytical cantilever deflection and natural frequency tests in `tests/test_fea.py` within `< 0.5%` error.
5. **Determinism for AI Optimization**:
   - Solvers must be deterministic (no unseeded randomness) and fast (< 100 ms per full evaluation for frame FEM) so evolutionary/RL agents can run hundreds of evaluations cleanly.
