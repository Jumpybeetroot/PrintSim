# PrintSim Physics Reference & Audit: 3D Solid Continuum FEM, Craig-Bampton CMS, and SIMP Topology Optimization

> **Scope**: `src/printsim/fea/solid_fem.py`, `src/printsim/fea/tet_fem.py` (with `_gmsh_worker.py`), `src/printsim/fea/superelement.py`, `src/printsim/fea/topology.py`, and `src/printsim/fea/bracket_designer.py`.
> **Units**: Strict SI internally (`m`, `kg`, `s`, `N`, `Pa`, `J`, `rad/s`); explicit suffixed units at API/DSL boundaries (`_mm`, `_g`, `_um`, `_mpa`, `_hz`, `_mj`, `_n_per_mm`).

---

## 1. Tensor Conventions & 3D Constitutive Models (`solid_fem.py`, `tet_fem.py`)

### 1.1 Engineering Voigt Notation
Both the cubic Hex8 solver (`solid_fem.py`) and the `scikit-fem` tetrahedral solver (`tet_fem.py`) use consistent **6-component engineering-shear Voigt ordering**:

$$
\boldsymbol{\varepsilon} = \begin{bmatrix} \varepsilon_{xx} & \varepsilon_{yy} & \varepsilon_{zz} & \gamma_{xy} & \gamma_{yz} & \gamma_{xz} \end{bmatrix}^T, \qquad
\boldsymbol{\sigma} = \begin{bmatrix} \sigma_{xx} & \sigma_{yy} & \sigma_{zz} & \tau_{xy} & \tau_{yz} & \tau_{xz} \end{bmatrix}^T
$$

where $\gamma_{ij} = \partial u_i / \partial x_j + \partial u_j / \partial x_i = 2 \varepsilon_{ij}$ for $i \neq j$. Constitutive law: $\boldsymbol{\sigma} = \mathbf{D} \boldsymbol{\varepsilon}$.

### 1.2 Isotropic 3D Linear Elasticity
For isotropic solids (`fdm_anisotropic=False`, `solid_fem.py:69-81`), with Young's modulus $E$ and Poisson's ratio $\nu \in (-1, 0.5)$:

$$
\mathbf{D}_{\text{iso}} = \frac{E}{(1+\nu)(1-2\nu)}
\begin{bmatrix}
1-\nu & \nu & \nu & 0 & 0 & 0 \\
\nu & 1-\nu & \nu & 0 & 0 & 0 \\
\nu & \nu & 1-\nu & 0 & 0 & 0 \\
0 & 0 & 0 & \frac{1-2\nu}{2} & 0 & 0 \\
0 & 0 & 0 & 0 & \frac{1-2\nu}{2} & 0 \\
0 & 0 & 0 & 0 & 0 & \frac{1-2\nu}{2}
\end{bmatrix}
$$

with shear modulus $G = \frac{E}{2(1+\nu)}$. The eigenvalues of $\mathbf{D}_{\text{iso}}$ are $3K = \frac{E}{1-2\nu}$ (multiplicity 1), $2G = \frac{E}{1+\nu}$ (multiplicity 2), and $G = \frac{E}{2(1+\nu)}$ (multiplicity 3), all strictly positive for $E>0$ and $-1 < \nu < 0.5$.

### 1.3 Transversely Isotropic FDM Layer-Bond Elasticity
For FDM-printed parts with build direction along global $+Z$ (`fdm_anisotropic=True`, `solid_fem.py:83-102`), the $XY$ print-bed plane is isotropic ($E_x = E_y = E_{xy}$, $\nu_{xy}$) and the inter-layer $Z$-axis is compliant ($E_z < E_{xy}$, $G_{yz} = G_{xz}$). Compliance matrix $\mathbf{S} = \mathbf{D}^{-1}$:

$$
\mathbf{S} =
\begin{bmatrix}
\frac{1}{E_{xy}} & -\frac{\nu_{xy}}{E_{xy}} & -\frac{\nu_{xz}}{E_{xy}} & 0 & 0 & 0 \\
-\frac{\nu_{xy}}{E_{xy}} & \frac{1}{E_{xy}} & -\frac{\nu_{xz}}{E_{xy}} & 0 & 0 & 0 \\
-\frac{\nu_{xz}}{E_{xy}} & -\frac{\nu_{xz}}{E_{xy}} & \frac{1}{E_z} & 0 & 0 & 0 \\
0 & 0 & 0 & \frac{1}{G_{xy}} & 0 & 0 \\
0 & 0 & 0 & 0 & \frac{1}{G_{xz}} & 0 \\
0 & 0 & 0 & 0 & 0 & \frac{1}{G_{xz}}
\end{bmatrix}, \qquad \mathbf{D} = \mathbf{S}^{-1}
$$

**Thermodynamic Stability & Maxwell–Betti Reciprocity** (Lempriere 1968; Jones 1999):
1. Symmetry requires $\frac{\nu_{xz}}{E_{xy}} = \frac{\nu_{zx}}{E_z}$, so $S_{13} = S_{31} = -\nu_{xz}/E_{xy}$ enforces reciprocity with implied $\nu_{zx} = \nu_{xz}(E_z/E_{xy})$.
2. Positive-definiteness ($\mathbf{S} \succ 0 \iff \mathbf{D} \succ 0$) holds iff $E_{xy}, E_z, G_{xy}, G_{xz} > 0$, $|\nu_{xy}| < 1$, and:
   $$1 - \nu_{xy} - 2 \nu_{xz}^2 \frac{E_z}{E_{xy}} > 0$$

**Implementation values (`solid_fem.py:84-89`)**: $E_{xy} = E$, $E_z = 0.68 E$ (**hardcoded**, ignoring `MaterialSpec.e3_ratio`!), $\nu_{xy} = \nu$, $\nu_{xz} = 0.85\nu$, $G_{xy} = \frac{E_{xy}}{2(1+\nu_{xy})}$, $G_{xz} = \frac{E_z}{2(1+\nu_{xz})}$.

### 1.4 Failure Criteria & Stress Invariants
Equivalent von Mises stress (`solid_fem.py:375-378`, `tet_fem.py:158-168`):

$$
\sigma_{\text{vM}} = \sqrt{\frac{1}{2}\left[(\sigma_{xx}-\sigma_{yy})^2 + (\sigma_{yy}-\sigma_{zz})^2 + (\sigma_{zz}-\sigma_{xx})^2\right] + 3\left(\tau_{xy}^2 + \tau_{yz}^2 + \tau_{xz}^2\right)}
$$

Combined safety factor (`solid_fem.py:382-394`, `tet_fem.py:390-401`):

$$
\mathrm{SF} = \min\!\left(\frac{\sigma_y}{\max(\sigma_{\text{vM}})}, \; \frac{\sigma_{z,\text{layer}}}{\max(0, \max(\sigma_{zz}))}\right), \qquad \sigma_{z,\text{layer}} = \begin{cases} 0.55\,\sigma_y & \text{if FDM anisotropic} \\ \sigma_y & \text{otherwise} \end{cases}
$$

---

## 2. 3D Continuum Finite Element Formulations

### 2.1 Regular Cubic Hex8 Element (`solid_fem.py:105-152`, `superelement.py:628-660`)
Trilinear Lagrange shape functions on parent cube $(\xi,\eta,\zeta) \in [-1,1]^3$ with corner sign table $\mathbf{s}_i = (s_{x,i}, s_{y,i}, s_{z,i}) \in \{-1, +1\}^3$ matching corner offsets `[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,0,1], [1,0,1], [1,1,1], [0,1,1]`:

$$
N_i(\xi,\eta,\zeta) = \frac{1}{8}(1 + s_{x,i}\xi)(1 + s_{y,i}\eta)(1 + s_{z,i}\zeta), \qquad i = 0,\dots,7
$$

For a regular cube of edge length $h$, the Jacobian is $\mathbf{J} = \frac{h}{2}\mathbf{I}_3$, $\det\mathbf{J} = (h/2)^3$, and $\nabla_{\mathbf{x}} N_i = \frac{2}{h}\nabla_{\boldsymbol{\xi}} N_i$. The element stiffness and consistent mass matrices are integrated via $2 \times 2 \times 2$ Gauss–Legendre quadrature ($\xi_q, \eta_q, \zeta_q = \pm 1/\sqrt{3}$, $w_q = 1$), which is **algebraically exact** for polynomials up to degree 3 in each coordinate (integrands $\mathbf{B}^T \mathbf{D} \mathbf{B}$ and $N_i N_j$ have max degree 2 per coordinate):

$$
\mathbf{K}_e = \sum_{q=1}^{8} \mathbf{B}(\boldsymbol{\xi}_q)^T \mathbf{D} \mathbf{B}(\boldsymbol{\xi}_q) \left(\frac{h}{2}\right)^3, \qquad
\mathbf{M}_e = \mathbf{I}_3 \otimes \left[\rho \sum_{q=1}^{8} \mathbf{N}(\boldsymbol{\xi}_q)\mathbf{N}(\boldsymbol{\xi}_q)^T \left(\frac{h}{2}\right)^3\right]
$$

**Boundary Traction Weighting (`solid_fem.py:344-357`)**: For a flat exterior bounding-box face, counting incident occupied elements per loaded node yields exact $1:2:4$ (corner:edge:interior) consistent nodal force weights for uniform traction $\mathbf{t} = \mathbf{F}/A$.

**Key Assumptions & Validity Limits**:
- Small-strain linear elasticity; uniform cubic voxels of edge $h$ (no non-cubic brick support).
- Full $2\times 2\times 2$ integration on trilinear Hex8 suffers from mild **parasitic shear locking** in coarse bending-dominated members (e.g., $-4.9\%$ deflection error at 4 elements through thickness on `cantilever_bar`).
- Stress recovery uses only the element centroid $\mathbf{B}(0,0,0)$, systematically underestimating outer-fiber bending stress on coarse meshes.

### 2.2 Conformal Tetrahedral Continuum FEA (`tet_fem.py`, `_gmsh_worker.py`)
Powered by OpenCASCADE STEP import + Gmsh 3D unstructured Delaunay tetrahedralization (`_gmsh_worker.py`) and `scikit-fem` weak-form assembly (`tet_fem.py:244-341`):

- **Weak Form**: Find $\mathbf{u} \in [H^1_0(\Omega; \Gamma_D)]^3$ such that $\forall \mathbf{v} \in [H^1_0(\Omega; \Gamma_D)]^3$:
  $$\int_\Omega \boldsymbol{\varepsilon}(\mathbf{v})^T \mathbf{D} \boldsymbol{\varepsilon}(\mathbf{u}) \, d\Omega = \int_{\Gamma_N} \mathbf{t} \cdot \mathbf{v} \, d\Gamma, \qquad \mathbf{t} = \frac{\mathbf{F}}{\int_{\Gamma_N} d\Gamma}$$
- **Elements**: Affine **Tet10 (`ElementTetP2`, default `order=2`)** with `intorder=2` (4-point symmetric Keast rule, exact for degree 2 stiffness integrands on straight-sided tetrahedra) or **Tet4 (`ElementTetP1`, `order=1`)** with `intorder=1`.
- **Note**: `_gmsh_worker.py:54` extracts linear 4-node tetrahedra (Gmsh type 4), and `scikit-fem` places P2 mid-edge nodes at straight-edge midpoints (subparametric/affine Tet10 — constant Jacobian per element, linear faceting of curved holes/fillets).

---

## 3. Craig-Bampton CMS & Virtual Point Transformation (`superelement.py`)

### 3.1 Virtual Point Transformation (VPT) & RBE3 Filter (van der Seijs et al., 2016)
For a mounting patch of $n_p$ solid nodes at coordinates $\mathbf{r}_j \in \mathbb{R}^3$ coupled to a 6-DOF virtual point at $\mathbf{p}_{\text{vp}}$ with $\mathbf{d}_j = \mathbf{r}_j - \mathbf{p}_{\text{vp}}$ and $\mathbf{q}_{\text{vp}} = [u_x, u_y, u_z, \theta_x, \theta_y, \theta_z]^T$:

$$
\mathbf{u}_j = \mathbf{u}_{\text{vp}} + \boldsymbol{\theta} \times \mathbf{d}_j = \begin{bmatrix} \mathbf{I}_3 & -[\mathbf{d}_j]_\times \end{bmatrix} \mathbf{q}_{\text{vp}}, \qquad
-[\mathbf{d}_j]_\times = \begin{bmatrix} 0 & d_{z,j} & -d_{y,j} \\ -d_{z,j} & 0 & d_{x,j} \\ d_{y,j} & -d_{x,j} & 0 \end{bmatrix}
$$

Stacking over all $n_p$ patch nodes defines $\mathbf{R}_u \in \mathbb{R}^{3n_p \times 6}$ (`superelement.py:79-110`). The weighted RBE3 displacement filter (`superelement.py:113-155`) is the weighted left pseudoinverse:

$$
\mathbf{T}_u = \left(\mathbf{R}_u^T \mathbf{W} \mathbf{R}_u\right)^{-1} \mathbf{R}_u^T \mathbf{W} \in \mathbb{R}^{6 \times 3n_p}, \qquad \mathbf{T}_u \mathbf{R}_u = \mathbf{I}_6
$$

By virtual work $\delta W = \mathbf{f}_p^T \delta \mathbf{u}_p$, the equivalent virtual-point wrench is $\mathbf{f}_{\text{vp}} = \mathbf{R}_u^T \mathbf{f}_p = \left[\sum_j \mathbf{f}_j;\; \sum_j \mathbf{d}_j \times \mathbf{f}_j\right]^T$.

### 3.2 RBE3 Nullspace Relaxation vs. RBE2 Rigid-Patch Locking (`superelement.py:496-592`)
- **RBE3 (`interface_mode="rbe3"`, default)**: Introduces an invertible square basis transformation $\mathbf{B}$ on each patch:
  $$\mathbf{u}_p = \mathbf{R}_u \mathbf{q}_{\text{vp}} + \mathbf{V}_p \mathbf{w}_{\text{warp}}, \qquad \mathbf{T}_u \mathbf{V}_p = \mathbf{0} \;\; (\mathbf{V}_p = \operatorname{null}(\mathbf{T}_u) \in \mathbb{R}^{3n_p \times (3n_p - 6)})$$
  Condensing $\mathbf{w}_{\text{warp}}$ alongside interior DOFs $\mathbf{u}_{\text{int}}$ enforces $\mathbf{V}_p^T \mathbf{f}_p = \mathbf{0} \iff \mathbf{f}_p = \mathbf{T}_u^T \mathbf{f}_{\text{vp}}$, which guarantees **exact force/moment conservation** ($\mathbf{R}_u^T \mathbf{T}_u^T \mathbf{f}_{\text{vp}} = \mathbf{f}_{\text{vp}}$) while leaving cross-sectional warping and Poisson contraction completely unconstrained.
- **RBE2 (`interface_mode="rbe2"`)**: Enforces $\mathbf{w}_{\text{warp}} \equiv \mathbf{0}$ ($\mathbf{u}_p = \mathbf{R}_u \mathbf{q}_{\text{vp}}$), rigidly locking the mounting face. By Rayleigh's monotonicity theorem, $\mathbf{K}_{\text{RBE2}} \succeq \mathbf{K}_{\text{RBE3}}$.

### 3.3 Craig-Bampton Fixed-Interface Substructuring (Craig & Bampton, 1968)
Partitioning DOFs into boundary ($b$) and interior ($i$) sets (`superelement.py:222-397`):

1. **Static constraint modes** (Guyan reduction):
   $$\boldsymbol{\Psi}_{cb} = -\mathbf{K}_{ii}^{-1} \mathbf{K}_{ib}$$
2. **Fixed-interface normal modes** ($n_m$ lowest modes):
   $$(\mathbf{K}_{ii} - \omega_k^2 \mathbf{M}_{ii}) \boldsymbol{\phi}_k = \mathbf{0}, \qquad \boldsymbol{\Phi}_{ii}^T \mathbf{M}_{ii} \boldsymbol{\Phi}_{ii} = \mathbf{I}_{n_m}$$
3. **Congruent projection** with $\mathbf{T}_{cb} = \begin{bmatrix} \mathbf{I}_{n_b} & \mathbf{0} \\ \boldsymbol{\Psi}_{cb} & \boldsymbol{\Phi}_{ii} \end{bmatrix}$:
   $$\mathbf{K}_{cb} = \mathbf{T}_{cb}^T \mathbf{K} \mathbf{T}_{cb} = \begin{bmatrix} \mathbf{K}_{bb} - \mathbf{K}_{bi}\mathbf{K}_{ii}^{-1}\mathbf{K}_{ib} & \mathbf{0} \\ \mathbf{0} & \operatorname{diag}(\omega_k^2) \end{bmatrix}, \qquad \mathbf{M}_{cb} = \mathbf{T}_{cb}^T \mathbf{M} \mathbf{T}_{cb}$$

Because $\mathbf{K}_{bm} = \mathbf{0}$ identically and $\mathbf{K}_{bb,\text{cb}}$ is the exact Schur complement, static boundary responses are **algebraically exact** for any $n_m \ge 0$, while dynamic modes below $\omega_{n_m}$ converge rapidly as $n_m$ increases.

---

## 4. 3D SIMP Topology Optimization (`topology.py`)

### 4.1 Modified SIMP & Compliance Sensitivity (Bendsøe & Sigmund, 2003; Andreassen et al., 2011)
For design densities $\rho_e \in [\rho_{\min}, 1]$ ($\rho_{\min} = 10^{-3}$) on $N$ cubic Hex8 voxels (`topology.py:464, 500-512`):

$$
s_e(\rho_e) = \epsilon_{\min} + \rho_e^p (1 - \epsilon_{\min}), \qquad \mathbf{K}(\boldsymbol{\rho}) = \sum_{e=1}^N s_e(\rho_e) \mathbf{K}_0^{(e)}, \qquad \epsilon_{\min} = 10^{-3}
$$

Compliance objective $c(\boldsymbol{\rho}) = \mathbf{F}^T \mathbf{u} = \sum_{e=1}^N s_e(\rho_e) \, c_e$ with $c_e = \mathbf{u}_e^T \mathbf{K}_0 \mathbf{u}_e \ge 0$. Because compliance is self-adjoint ($\boldsymbol{\lambda} = -\mathbf{u}$), the exact analytical sensitivity is:

$$
\frac{\partial c}{\partial \rho_e} = -\mathbf{u}^T \frac{\partial \mathbf{K}}{\partial \rho_e} \mathbf{u} = -p\,\rho_e^{p-1}(1 - \epsilon_{\min})\, c_e \le 0
$$

### 4.2 Conical Sensitivity Filter & Optimality Criteria (Sigmund, 2001)
Mesh-independency cone weights $H_{ef} = \max(0, r_{\min} - \|\mathbf{x}_e - \mathbf{x}_f\|_2)$ (`topology.py:261-295`), with Sigmund's (2001) heuristic sensitivity filter (`topology.py:512`):

$$
\widehat{\frac{\partial c}{\partial \rho_e}} = \frac{1}{\max(10^{-12}, \rho_e \sum_f H_{ef})} \sum_f H_{ef} \, \rho_f \frac{\partial c}{\partial \rho_f}
$$

Optimality Criteria (OC) update (`topology.py:298-334`) with move limit $m = 0.2$ and damping exponent $\eta = 1/2$:

$$
\rho_e^{\text{new}} = \max\!\left(\rho_{\min},\, \max\!\left(\rho_e - m,\, \min\!\left(1,\, \min\!\left(\rho_e + m,\, \rho_e \sqrt{\frac{\max(0, -\widehat{\partial c / \partial \rho_e})}{\lambda}}\right)\right)\right)\right)
$$

where $\lambda > 0$ is solved via bisection to satisfy $\sum_e \rho_e^{\text{new}} = V_f N$, with passive boundary voxels pinned to $\rho_e = 1$ when `preserve_boundaries=True`. Post-processing applies top-$k$ binary thresholding followed by 6-neighbor BFS pruning and Dijkstra bridging of disconnected loaded faces (`topology.py:367-433`).

---

## 5. Quantitative Verification Summary

Executed via `uv run python` against `MATERIAL_CATALOG` (all 23 materials), `cantilever_bar` ($60 \times 12 \times 12\text{ mm}$, $F_z = -50\text{ N}$), an $8\times 2\times 2$ Hex8 CMS bar ($h = 5\text{ mm}$), SIMP finite-difference gradients, and parametric bracket parity checks:

| Benchmark / Check | Reference / Exact | Measured in PrintSim | Verdict |
| :--- | :--- | :--- | :--- |
| `D` symmetry & SPD (all 23 materials, iso & FDM) | $\|D - D^T\| \approx 0$, $\lambda_{\min} > 0$ | $\max\|D-D^T\| = 1.53\times 10^{-5}\text{ Pa}$ ($7\times 10^{-17}$ rel), $\lambda_{\min,\text{iso}} = 7.78\times 10^8\text{ Pa}$, $\lambda_{\min,\text{fdm}} = 5.50\times 10^8\text{ Pa}$ | **VERIFIED** (SPD), but `E_z = 0.68 E` ignores `mat.e3_ratio` |
| Single Hex8 `Ke` & `Me` (`solid_fem.py`, `superelement.py`) | 6 zero RB modes, $\sum M_x = \rho h^3$ | $\max|\lambda_{1..6}| = 4.85\times 10^{-8}$ vs $\lambda_7 = 1.30\times 10^7$; $\sum M_x = 7.290000\times 10^{-5} = \rho h^3$ | **VERIFIED** |
| Al6061 Cantilever Tip Disp ($60\times 12\times 12\text{ mm}, 50\text{ N}$) | EB: $30.24\,\mu\text{m}$; Timoshenko ($\kappa=0.8509$): $31.18\,\mu\text{m}$ | Tet10 ($h=4\text{ mm}$): $30.90\,\mu\text{m}$ ($-0.9\%$); Hex8 ($20\times 4\times 4$): $29.66\,\mu\text{m}$ ($-4.9\%$); Hex8 ($40\times 8\times 8$): $30.64\,\mu\text{m}$ ($-1.7\%$); Tet4: $23.79\,\mu\text{m}$ ($-23.7\%$) | **VERIFIED** (Tet10 & Hex8); mild Hex8 shear locking |
| Al6061 Cantilever Peak Bending Stress | Root outer fiber $\sigma_{xx} = 10.417\text{ MPa}$ | Hex8 centroid: $6.923\text{ MPa}$ ($-33.5\%$); Tet10 centroid: $8.645\text{ MPa}$ ($-17.0\%$) | **MODERATE ISSUE** (centroid stress underprediction) |
| VPT Left-Inverse & RBE3 Load/Moment Conservation | $T_u R_u = I_6$, $R_u^T T_u^T f_{vp} = f_{vp}$ | $\max|T_u R_u - I_6| = 2.13\times 10^{-14}$; $\|R_u^T T_u^T f_{vp} - f_{vp}\| = 1.47\times 10^{-14}$ | **VERIFIED** |
| Craig-Bampton Guyan Static Exactness & $K_{bm}=0$ | $\|u_{\text{cb}} - u_{\text{full}}\| = 0$, $K_{bm} = 0$ | $\max|u_{\text{cb}} - u_{\text{full}}| = 6.61\times 10^{-20}\text{ m}$; $\max|K_{bm}| = 2.44\times 10^{-6}$ ($2\times 10^{-14}$ rel) | **VERIFIED** |
| Craig-Bampton Cantilever Eigenfrequencies (first 5 modes) | Full: `[5350.07, 5350.07, 19389.40, 27591.49, 27591.49] Hz` | CB ($n_m=6$): `[5351.25, 5351.25, 19468.05, 27692.14, 27692.14] Hz` (max rel err $0.41\%$, mode 1: $0.022\%$) | **VERIFIED** |
| VPT RBE3 vs RBE2 RB Modes, Mass & Boss Locking | 6 zero RB modes, $m = 0.010800\text{ kg}$, $k_{\text{RBE2}} > k_{\text{RBE3}}$ | Both: $\max|\lambda_{1..6}| \le 1.73\times 10^{-5}$ ($\lambda_7 \ge 2.14\times 10^3$), $m = 0.010800\text{ kg}$; $k_{z,\text{RBE2}}/k_{z,\text{RBE3}} = 1.0909$ ($+9.1\%$ locking removed) | **VERIFIED** |
| SIMP Sensitivity Analytical vs Central FD ($\epsilon=10^{-6}$) | Exact derivative | Max rel error $= 6.93\times 10^{-5}$ | **VERIFIED** |
| OC Bisection Volume Conservation (`dc ~ 1e-17`) | Target $V_f = 0.50000$ | Returns $V_f = 0.30000$ (40% error — bisection exhausts 80 iters from `l2 = 1e9`) | **MODERATE ISSUE** |
| Bracket Designer `motor_mount(arm=90, H=10)` Hex8 vs Tet10 | Requested $121.5\times 108.0\times 10.0\text{ mm}$ | Hex8 simulates $121.5\times 109.4\times 24.3\text{ mm}$ ($+143\%$ $Z$-height!); mass $181.87\text{ g}$ vs $84.13\text{ g}$ ($+116\%$); disp $169.8\,\mu\text{m}$ vs $1740.7\,\mu\text{m}$ ($10.3\times$ too stiff!) | **CRITICAL BUG** |
| Coarse CMS Bracket Grid (`superelement.py`) Geometry | `corner_gusset(40,20)`: `[40,40,20]`; `(60,12)`: `[60,60,12]` | Simulates `[38.6, 38.6, 21.5]` ($-3.5\%, -3.5\%, +7.3\%$) and `[55.7, 55.7, 13.9]` ($-7.2\%, -7.2\%, +16.0\%$) | **CRITICAL BUG** |

---

## 6. Audit Findings Table

| Severity | Module & Location | Finding & Quantitative Impact | Recommended Concrete Fix |
| :--- | :--- | :--- | :--- |
| **CRITICAL BUG** | `src/printsim/fea/bracket_designer.py:176-190, 295` | **Cubic-voxel dimension distortion in `_build_bracket_voxel_grid`**: Occupancy is sampled in anisotropic coordinates `(hx, hy, hz) = (Lx/nx, Ly/ny, Lz/nz)` with floor `nz = max(4, round(Lz/h_mm))`, then uniform `h_mm` is returned to cubic Hex8 `solve_solid_voxel_fea`. Actual simulated dimensions are `(nx*h_mm, ny*h_mm, nz*h_mm)`. For valid input `motor_mount(arm=90, H=10)`, simulated thickness is `24.3 mm` instead of `10.0 mm` (**+143% height error**, **+116% mass error**: `181.87 g` vs Tet10 `84.13 g`, and **10.3× displacement underprediction**: `169.8 µm` vs Tet10 `1740.7 µm`). | Generalize `_hex8_b_matrix` and `hex8_element_stiffness` (in `solid_fem.py:105-151`) to accept `(hx, hy, hz)` (`inv_jx=2/hx, inv_jy=2/hy, inv_jz=2/hz`, `det_j = hx*hy*hz/8`), and pass exact `(hx, hy, hz)` from `_build_bracket_voxel_grid` (or keep cubic voxels strictly matching `min_dim/4` without clamping `nz` independently of `h_mm`). |
| **CRITICAL BUG** | `src/printsim/fea/superelement.py:674-681, 740, 889-893` | **Aspect-ratio distortion in `_build_coarse_bracket_grid`**: Samples geometry on `(hx, hy, hz)` then collapses to scalar `h_cubic = (hx*hy*hz)**(1/3)` for cubic Hex8 `K` and `M`. Preserves bounding-box volume only by distorting all three axes: even default `corner_gusset(40, 20)` simulates `[38.6, 38.6, 21.5] mm` (`[-3.5%, -3.5%, +7.3%]`); `corner_gusset(60, 12)` simulates `[55.7, 55.7, 13.9] mm` (`[-7.2%, -7.2%, +16.0%]`), artificially inflating bending stiffness and corrupting VP coordinates `node_coords_m`. | Extend `hex8_element_stiffness` and `hex8_element_mass` to take `(hx_m, hy_m, hz_m)` and compute `node_coords_m = node_grid_coords * [hx_m, hy_m, hz_m]` and element volume `hx_m * hy_m * hz_m`. |
| **CRITICAL BUG** | `src/printsim/fea/solid_fem.py:84-89` (shared by `tet_fem.py:259`, `topology.py:470`, `superelement.py:891`) | **Hardcoded `E_z = E * 0.68` ignores `MaterialSpec.e3_ratio`**: `MATERIAL_CATALOG` (`materials.py:280-366`) defines calibrated interlayer ratios `e3_ratio = 0.32..0.35` for CF-reinforced printed polymers (`pa12_cf_printed`, `pet_cf_printed`, `ppa_cf_printed`, `pps_cf_printed`) and `0.75` for `abs_printed`. Hardcoding `0.68` overpredicts Z-axis stiffness of CF-printed parts by **1.94×–2.12×** (and ignores `e2_ratio`). Confirmed numerically that using `mat.e3_ratio` preserves strict positive-definiteness (`min eigvalsh(D) >= 6.07e8 Pa`) across all 5 printed polymers. | Use `E_z = E * mat.e3_ratio` (clamped for SPD stability: verify $1 - \nu_{xy} - 2\nu_{xz}^2 (E_z/E_{xy}) > 0.05$, or scale $\nu_{xz} = \min(\nu \cdot 0.85, 0.9 \sqrt{(1-\nu_{xy})E_{xy}/(2E_z)})$). Symmetrize `D = 0.5 * (D + D.T)` after `np.linalg.inv(S)`. |
| **MODERATE ISSUE** | `src/printsim/fea/solid_fem.py:369-380`, `src/printsim/fea/tet_fem.py:372-388` | **Centroid-only stress recovery systematically underpredicts peak bending & Z-delamination stress**: `solid_fem.py` evaluates stress only at `B_center` ($\xi=\eta=\zeta=0$), and `tet_fem.py` averages the 4 P2 quadrature-point stresses (`np.mean(sigma_qp, axis=2)`) before taking `max_vm_mpa` and `max_z_tensile_mpa`, throwing away Tet10's linear intra-element stress variation. On `cantilever_bar` (exact root outer fiber `10.417 MPa`), Hex8 (`20×4×4`) reports `6.923 MPa` (**−33.5%**) and Tet10 (`h=4mm`) reports `8.645 MPa` (**−17.0%**). | In `tet_fem.py`, compute `vm_qp = _von_mises_from_voigt(sigma_qp)` and report peak stress from Gauss points (or nodal $L^2$ projection / Superconvergent Patch Recovery), and use quadrature-weighted averaging (`basis.dx`) for centroid export. In `solid_fem.py`, evaluate `B` at the $2\times 2\times 2$ Gauss points (or extrapolated nodal stresses) for `max_vm_mpa` and `max_z_tensile_mpa`. |
| **MODERATE ISSUE** | `src/printsim/fea/topology.py:312-318` | **Non-scale-invariant OC bisection bracket (`l2 = max(1e9, ...)`) fails at small compliance sensitivities**: In SI units (Joules), element sensitivities `neg_dc` scale as $F^2 / (E h)$ and can be $10^{-17}\text{–}10^{-10}\text{ J}$ for stiff metals or small forces. Starting bisection at `l2 = 1e9` with a 80-iteration cap exhausts iterations before reaching $\lambda^*$, breaking volume conservation (`dc ~ 1e-17` yields `vol_frac = 0.300` instead of `0.500`). | Replace line 313 with a scale-invariant upper bound: `dc_max = float(np.max(neg_dc))`; if `dc_max <= 0.0`, return `rho.copy()`; else set `l1 = 0.0`, `l2 = dc_max * 1e4` (or bracket dynamically by doubling/halving from `np.median(neg_dc[neg_dc > 0])`). |
| **MODERATE ISSUE** | `src/printsim/fea/superelement.py:804` (and `solid_fem.py:267`, `tet_fem.py:237`, `topology.py:467`, `bracket_designer.py:512`) | **Inconsistent FDM anisotropy auto-detection**: `condense_parametric_bracket` defaults `fdm_anisotropic = False`, silently treating printed brackets (`abs_printed`, `ppa_cf_printed`, etc.) as isotropic unless explicitly overridden, whereas the other 4 modules check `"printed" in material_id` instead of the authoritative `MaterialSpec.is_printed_polymer` flag. | Default `fdm_anisotropic = mat.is_printed_polymer` when `None` across all 5 modules (including `superelement.py:804`). |
| **MODERATE ISSUE** | `src/printsim/fea/solid_fem.py:399-401`, `src/printsim/fea/tet_fem.py:355-357`, `src/printsim/fea/bracket_designer.py:612` | **Inconsistent and non-work-conjugate stiffness metric across modules**: `solid_fem.py` and `tet_fem.py` compute `||F|| / mean(||u_loaded||)` (unweighted over corner vertices in `tet_fem.py`, biasing results toward Gmsh curvature-refined regions and including displacements orthogonal to $\mathbf{F}$), whereas `bracket_designer.py:612` overwrites stiffness with `||F|| / max(||u_all||)`. | Standardize all three modules on exact work-conjugate effective stiffness $k_{\text{eff}} = \|\mathbf{F}\|^2 / (\mathbf{f}^T \mathbf{u})$ (in N/m $\to$ N/mm), which is mesh-refinement invariant and directionally consistent with compliance $C = \mathbf{f}^T \mathbf{u}$. |
| **MODERATE ISSUE** | `src/printsim/fea/solid_fem.py:383`, `src/printsim/fea/tet_fem.py:391` | **Hardcoded 55% Z-layer delamination strength**: `z_layer_mpa = yield_mpa * 0.55` is applied uniformly to all printed polymers. While reasonable for unreinforced ABS/ASA (50–65%), CF-filled FDM polymers (`pa12_cf_printed`, `ppa_cf_printed`, `pps_cf_printed`) exhibit Z-layer tensile strengths of only 25–38% of XY tensile strength, making delamination safety factors overly optimistic by ~1.5–2×. | Add an explicit `z_strength_ratio` field to `MaterialSpec` (e.g., `0.55` for ABS, `0.32–0.38` for CF-filled polymers) and read `mat.z_strength_ratio` in both `solid_fem.py` and `tet_fem.py`. |
| **MODERATE ISSUE** | `src/printsim/fea/topology.py:464, 512` | **High void stiffness floor (`emin_ratio = 1e-3`) & heuristic sensitivity filter**: `emin_ratio = 1e-3` is $10^6\times$ larger than standard modified SIMP (`1e-9`, Andreassen et al. 2011), allowing void zones to transmit non-trivial load and widening the gap between continuous SIMP and binary-thresholded compliance. Additionally, Sigmund's heuristic sensitivity filter (`line 512`) is not the exact chain-rule gradient of a filtered density field, leaving gray regions prior to hard binary thresholding. | Lower `emin_ratio` to `1e-6`–`1e-9` (safe with `diag_floor` stabilization) and consider upgrading to a density filter $\tilde{\boldsymbol{\rho}} = \mathbf{H}\boldsymbol{\rho}/\mathbf{H}_s$ with exact chain-rule sensitivity $\frac{\partial c}{\partial \boldsymbol{\rho}} = \mathbf{H}^T\!\left(\frac{1}{\mathbf{H}_s}\frac{\partial c}{\partial \tilde{\boldsymbol{\rho}}}\right)$ and optional Heaviside $\beta$-continuation projection. |
| **MODERATE ISSUE** | `src/printsim/fea/bracket_designer.py:196-295` vs `316-481` (and `superelement.py:721-736`) | **Geometry mismatch between voxel (`hex8`) and B-Rep (`tet10`) bracket builders**: Voxel `motor_mount` omits the 4× NEMA17 M3 mounting holes (and `superelement.py` also omits the 2× rear M5 flange holes) and omits front corner fillets; voxel `corner_gusset`/`xy_joint_l_bracket` add an inner concave root fillet at `(w, w)` even when `r_eff > 0` while omitting the outer corner fillets that B-Rep applies. Contributes to the 17.5% displacement / 27.5% stress gap observed on `corner_gusset`. | Align cutout/fillet logic between `_build_bracket_voxel_grid`, `_build_coarse_bracket_grid`, and `_build_parametric_bracket_brep` (add M3/M5 cutouts where hole diameter $> 1.2 h$ and match outer vs. inner fillet placement). |
| **MODERATE ISSUE** | `src/printsim/fea/superelement.py:299-303, 333-342` | **Potential ARPACK shift-invert operator mismatch on singular `K_ii`**: If `splu(K_ii_csc)` raises `RuntimeError` and falls back to factorizing `K_ii_reg = K_ii_csc + reg*I`, line 336 passes unregularized `K_ii_csc` to `eigsh(..., OPinv=op)` while `op` solves `K_ii_reg`, violating ARPACK's `OPinv == inv(A - sigma*M)` contract. | Store `K_ii_solve = K_ii_reg` when regularization triggers and pass `K_ii_solve` to `eigsh`. |
| **MINOR / LIMITATION** | `src/printsim/fea/superelement.py:481, 513` | **Unweighted RBE3 filter & dense nullspace scaling**: `condense_solid_patch_to_vp` calls `virtual_point_filter_rbe3(R_u)` with `weights=None` instead of face tributary-area weights (over-weighting corner nodes relative to interior nodes on Hex8 faces), and `scipy.linalg.null_space(T_u)` is $O(n_p^3)$ dense (fast for coarse ~380-element brackets, not suitable for fine meshes with $n_p > 500$). | Optionally pass face-tributary nodal area weights to `virtual_point_filter_rbe3(R_u, weights=area_w)`. Document mesh-size limit for dense RBE3 nullspace condensation. |
| **MINOR / LIMITATION** | `src/printsim/fea/topology.py:430-431, 526-533, 592-597` | **Post-processing & logging quirks**: (1) `_prune_disconnected_islands` returns `visited | (kept & is_loaded_elem)`, which can re-attach disconnected loaded voxels if the input domain itself is disconnected; (2) `history` pairs pre-update `compliance_mj` with post-update `volume_fraction`/`mass_g`; (3) `optimize_solid_topology` solves the final optimized Hex8 system twice (`solve_solid_voxel_fea` + `_solve_system_displacements`). | Return `visited` directly in `_prune_disconnected_islands` (Dijkstra already bridges reachable loaded faces); log `compliance_mj`, `volume_fraction`, and `mass_g` from the same iterate; return `compliance_mj` from `solve_solid_voxel_fea` to eliminate the duplicate solve. |
| **MINOR / LIMITATION** | `src/printsim/fea/solid_fem.py:140-151`, `src/printsim/fea/tet_fem.py:54, 245` | **Element kinematics limitations**: (1) Full-integration Hex8 exhibits mild bending shear locking ($-4.9\%$ at $4$ elements/thickness on `cantilever_bar`); (2) `_gmsh_worker.py` exports straight-sided Tet4 meshes promoted to affine Tet10 in `scikit-fem`, so curved fillets/holes are linearly faceted; (3) isotropic `constitutive_matrix_3d` ignores `MaterialSpec.shear_modulus_pa` for non-isotropic materials like `carbon_fiber_tube`. | Document as known discretization limits; for high-accuracy bending on coarse Hex8 grids, consider Selective Reduced Integration (SRI) or Wilson incompatible modes; for curved CAD fillets, export Gmsh order-2 curved meshes if needed. |
| **VERIFIED** | `src/printsim/fea/solid_fem.py:64-152, 344-357` | **Hex8 kinematics, 2×2×2 Gauss quadrature, Voigt ordering, and flat-face traction weighting**: Exact $K_e$ symmetry ($0.0$), 6 zero RB modes ($\le 4.85\times 10^{-8}$ vs $\lambda_7 = 1.30\times 10^7$), exact Voigt alignment with `D`, exact 1:2:4 consistent nodal weighting on planar faces, and SPD `D` for all 23 catalog materials. | None required (except `mat.e3_ratio` and non-cubic `(hx, hy, hz)` extensions above). |
| **VERIFIED** | `src/printsim/fea/tet_fem.py:145-168, 244-341` | **`scikit-fem` isotropic & FDM-anisotropic bilinear forms and Tet10 cantilever accuracy**: Symmetric Voigt strain extraction is transpose-invariant; bilinear form and consistent surface traction integration match analytical Timoshenko cantilever deflection to **−0.9%** ($30.90\,\mu\text{m}$ vs $31.18\,\mu\text{m}$), and ABS FDM Hex8/Tet10 displacements agree within $2.7\%$. | None required for assembly/traction weak forms. |
| **VERIFIED** | `src/printsim/fea/superelement.py:79-156, 222-397, 496-661` | **Craig-Bampton CMS, consistent Hex8 mass, and VPT RBE2/RBE3 physics**: Exact `hex8_element_mass` ($\sum M_x = \rho h^3$), exact VPT left-inverse ($2.1\times 10^{-14}$) and RBE3 wrench conservation ($1.5\times 10^{-14}$), exact Guyan static reduction ($6.6\times 10^{-20}\text{ m}$), block-diagonal $K_{cb}$ ($2\times 10^{-14}$ rel), cantilever eigenfrequency parity within $0.41\%$ across 5 modes ($0.022\%$ on mode 1), exact 6 zero RB modes and total mass conservation on both RBE2 & RBE3, and physical $+9.1\%$ boss-locking reduction from RBE2 to RBE3. | None required for core CMS/VPT algebra. |
| **VERIFIED** | `src/printsim/fea/topology.py:500-511` | **SIMP interpolation and analytical compliance sensitivity**: Analytical sensitivity `-p * rho**(p-1) * (1 - emin_ratio) * ce` matches central finite differences across all elements to $6.93\times 10^{-5}$ relative error with correct negative sign and modified-SIMP chain-rule factor. | None required for sensitivity derivation. |

---

## 7. References

1. Craig, R. R., & Bampton, M. C. C. (1968). *Coupling of substructures for dynamic analyses*. AIAA Journal, 6(7), 1313–1319.
2. van der Seijs, M. V., de Klerk, D., & Rixen, D. J. (2016). *General framework for dynamic substructuring: History, review and classification of techniques*. Mechanical Systems and Signal Processing, 68–69, 217–244.
3. Lempriere, B. M. (1968). *Poisson's ratio in orthotropic materials*. AIAA Journal, 6(11), 2226–2227.
4. Jones, R. M. (1999). *Mechanics of Composite Materials* (2nd ed.). Taylor & Francis.
5. Bendsøe, M. P., & Sigmund, O. (2003). *Topology Optimization: Theory, Methods and Applications*. Springer.
6. Sigmund, O. (2001). *A 99 line topology optimization code written in Matlab*. Structural and Multidisciplinary Optimization, 21(2), 120–127.
7. Andreassen, E., Clausen, A., Schevenels, M., Lazarov, B. S., & Sigmund, O. (2011). *Efficient topology optimization in MATLAB using 88 lines of code*. Structural and Multidisciplinary Optimization, 43(1), 1–16.
8. Cowper, G. R. (1966). *The shear coefficient in Timoshenko's beam theory*. Journal of Applied Mechanics, 33(2), 335–340.
