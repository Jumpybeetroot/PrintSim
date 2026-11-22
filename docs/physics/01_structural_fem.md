# PrintSim Physics Audit 01: Structural FEM, Shells, Bed Plate, Static Solver & Assembly Mechanics

**Scope (Read-Only Source Audit):**
- `src/printsim/fea/beam3d.py` — 12-DOF 3D Euler-Bernoulli frame element, global assembly, internal stress recovery
- `src/printsim/fea/shell_mitc4.py` — 24-DOF Bathe & Dvorkin (1984) MITC4 flat shell element & 12-DOF eccentric CBUSH fastener springs
- `src/printsim/fea/bed_plate_fem.py` — 2D MITC4 heated bed plate statics, thermal dome/taco warp, and modal solver
- `src/printsim/fea/solver.py` — Multi-loadcase static 3D FEA solver (gravity, unit-probe stiffness, peak XY acceleration, CoreXY belt preload, combined envelope)
- `src/printsim/geometry/builder.py` — Structural mechanics helpers (Newmark composite rail/backer, Huth-Schwarmann bolted panels, ACM sandwich, corner-joint compliance, toolhead link stiffness, extrusion cavity filling, bed Z-drive & isolator feet springs)

---

## Executive Summary

The core finite element kernels (`beam3d.py` 12×12 stiffness/mass matrices, `shell_mitc4.py` Bathe-Dvorkin covariant shear tying & Mindlin-Reissner bending operators, and `fastener_spring_matrices` eccentric CBUSH transformation) are mathematically sound and match closed-form analytical solutions (Euler-Bernoulli beam theory, distorted-quad membrane/bending patch tests, Timoshenko/Leissa plate theory) to machine precision or discretization accuracy.

However, this first-principles audit uncovered **5 `CRITICAL BUG`s** and **8 `MODERATE ISSUE`s** affecting rigid-body frame invariance, CoreXY belt-preload equilibrium, bedslinger Y-acceleration physics, heated-bed thermal curvature sign/magnitude, and eccentric spring coupling:

1. **Grounded drilling stiffness in MITC4 (`shell_mitc4.py:225-228`)** violates 3D rigid-body rotational invariance, adding fictitious rotational springs to inertial ground (`1,523.4 J/rad²` per 3 mm Al quad) and making all 3 free-free rotational modes of an unconstrained paneled CoreXY frame non-zero (`0.76, 0.81, 1.13 Hz`).
2. **Zero-offset CBUSH across 14 mm vertical gap on `bed_mount_spring` (`builder.py:1230, 1259`)** breaks rigid-body invariance (`7,056.0 J/rad²` spurious rigid pitch energy for `kinematic_3pt`) and drops the `14 mm` horizontal force-couple transfer between `bed_surface` and `bed_frame`.
3. **Inverted sign and $0.5\times$ curvature factor on bed thermal gradient dome (`bed_plate_fem.py:206`)**: steady upward heat loss $q_{\text{top}} > 0$ makes the bottom hotter than the top, producing a **concave-up bowl** ($\nabla^2 w > 0$) that *adds* to center gravity sag; the coded `1 - 2(r_x^2 + r_y^2)` produces a **concave-down dome** with half the physical 2D spherical curvature, partially *cancelling* gravity sag.
4. **CoreXY belt preload FBD error (`solver.py:351-358`)**: although total $\sum F = 0, \sum M = 0$, the moving Y-carriages (`y_car_l`, `y_car_r`) receive a spurious `-t0` force in Y (total `-2*t0` pulling the gantry forward under equal belt tension!) while rear corners (`g_rl`, `g_rr`) receive `-t0` instead of `-2*t0`.
5. **Cartesian bedslinger Y-acceleration applies a phantom horizontal kick to the stationary toolhead (`solver.py:321-327`)**: `+m_x * a_y` is applied directly to `toolhead_cog`, bending the stationary Z-portal even on rigidly bolted feet, while `bed_surface` uses bare `bed.mass_g` rather than `y_moving_mass_kg`.

---

## 1. 3D Euler-Bernoulli Beam FEM (`src/printsim/fea/beam3d.py`)

### 1.1 Coordinate Frame & Transformation
For a member from $\mathbf{p}_0$ to $\mathbf{p}_1$ of length $L = \|\mathbf{p}_1 - \mathbf{p}_0\|_2 > 10^{-9}\text{ m}$, the local right-handed orthonormal triad $\mathbf{R} = [\mathbf{e}_x;\; \mathbf{e}_y;\; \mathbf{e}_z] \in SO(3)$ (`beam3d.py:11-32`) is constructed as:
$$\mathbf{e}_x = \frac{\mathbf{p}_1 - \mathbf{p}_0}{L}, \qquad
\mathbf{r}_{\text{ref}} = \begin{cases} (0,1,0)^T & \text{if } |\mathbf{e}_x \cdot \hat{\mathbf{z}}| > 0.99 \\ (0,0,1)^T & \text{otherwise} \end{cases}, \qquad
\mathbf{e}_y = \frac{\mathbf{r}_{\text{ref}} \times \mathbf{e}_x}{\|\mathbf{r}_{\text{ref}} \times \mathbf{e}_x\|}, \qquad
\mathbf{e}_z = \mathbf{e}_x \times \mathbf{e}_y$$
The 12×12 block-diagonal rotation matrix $\mathbf{T} = \operatorname{diag}(\mathbf{R}, \mathbf{R}, \mathbf{R}, \mathbf{R})$ (`beam3d.py:35-41`) maps global nodal DOFs $\mathbf{u}_g$ to local DOFs $\mathbf{u}_\ell = \mathbf{T}\mathbf{u}_g$, with global matrices $\mathbf{K}_g = \mathbf{T}^T \mathbf{K}_\ell \mathbf{T}$ and $\mathbf{M}_g = \mathbf{T}^T \mathbf{M}_\ell \mathbf{T}$.

- **Gimbal-lock safety:** Switching at $|\mathbf{e}_x \cdot \hat{\mathbf{z}}| > 0.99$ ($\approx 8.11^\circ$ cone around $\pm Z$) guarantees $\|\mathbf{r}_{\text{ref}} \times \mathbf{e}_x\| \ge \sin(8.11^\circ) \approx 0.141$, eliminating division-by-zero for vertical beams along $\pm Z$.
- **Limitation:** No explicit cross-section roll angle $\beta$ is stored; asymmetric profiles (`extrusion_2040`, `extrusion_3060`) undergo an abrupt $90^\circ$ cross-section orientation flip across the $8.11^\circ$ vertical cone.

### 1.2 Local DOF Ordering & Right-Hand-Rule Sign Conventions
Local nodal DOF ordering (`beam3d.py:56-57`):
$$\mathbf{u}_\ell = \begin{bmatrix} u_{x1} & u_{y1} & u_{z1} & \theta_{x1} & \theta_{y1} & \theta_{z1} & u_{x2} & u_{y2} & u_{z2} & \theta_{x2} & \theta_{y2} & \theta_{z2} \end{bmatrix}^T$$
Under the right-hand rule:
- Bending in the local $x$-$y$ plane (about $\mathbf{e}_z$, governed by $I_z = \int y^2\,dA$): $v'(x) = +\theta_z$.
- Bending in the local $x$-$z$ plane (about $\mathbf{e}_y$, governed by $I_y = \int z^2\,dA$): $w'(x) = -\theta_y$.

### 1.3 Stiffness & Consistent Mass Matrices
With $s_b = 0.08$ for `is_truss=True` and $s_b = 1.0$ otherwise (`beam3d.py:175`), the non-zero blocks of $\mathbf{K}_\ell$ (`beam3d.py:44-100`) are:
- **Axial** $(u_{x1}, u_{x2})$: $\frac{EA}{L}\begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}$
- **Torsion** $(\theta_{x1}, \theta_{x2})$: $\frac{s_b GJ}{L}\begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}$
- **$x$-$y$ bending** $(u_{y1}, \theta_{z1}, u_{y2}, \theta_{z2})$:
$$\mathbf{K}_{xy} = \frac{s_b E I_z}{L^3} \begin{bmatrix} 12 & 6L & -12 & 6L \\ 6L & 4L^2 & -6L & 2L^2 \\ -12 & -6L & 12 & -6L \\ 6L & 2L^2 & -6L & 4L^2 \end{bmatrix}$$
- **$x$-$z$ bending** $(u_{z1}, \theta_{y1}, u_{z2}, \theta_{y2})$, transformed by $\mathbf{S}=\operatorname{diag}(1,-1,1,-1)$ from $w' = -\theta_y$:
$$\mathbf{K}_{xz} = \frac{s_b E I_y}{L^3} \begin{bmatrix} 12 & -6L & -12 & -6L \\ -6L & 4L^2 & 6L & 2L^2 \\ -12 & 6L & 12 & 6L \\ -6L & 2L^2 & 6L & 4L^2 \end{bmatrix}$$

With element mass $m_e = \rho A L$ and polar radius of gyration squared $r_x^2 = (I_y + I_z)/A$, the consistent mass matrix $\mathbf{M}_\ell$ (`beam3d.py:103-150`) has blocks:
- **Axial:** $\frac{m_e}{6}\begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$; **Torsion:** $\frac{m_e r_x^2}{6}\begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$.
- **$x$-$y$ transverse** $(u_{y1}, \theta_{z1}, u_{y2}, \theta_{z2})$:
$$\mathbf{M}_{xy} = \frac{m_e}{420}\begin{bmatrix} 156 & 22L & 54 & -13L \\ 22L & 4L^2 & 13L & -3L^2 \\ 54 & 13L & 156 & -22L \\ -13L & -3L^2 & -22L & 4L^2 \end{bmatrix}$$
- **$x$-$z$ transverse** $(u_{z1}, \theta_{y1}, u_{z2}, \theta_{y2})$: $\mathbf{M}_{xz} = \mathbf{S}^T \mathbf{M}_{xy} \mathbf{S}$ (exact opposite signs on translation-rotation off-diagonals).

**References:** Przemieniecki (1968), *Theory of Matrix Structural Analysis*, Ch. 5 & 11; Cook et al. (2002), *Concepts and Applications of Finite Element Analysis*, 4th ed.

### 1.4 Gravity Assembly, Lumped Inertia & Stress Recovery
- **Element gravity (`beam3d.py:201-204`):** Lumped nodal forces $F_{z1} = F_{z2} = -\frac{1}{2}\rho A L g$ ($g = 9.80665\text{ m/s}^2$) without consistent fixed-end bending moments $\pm w L^2 / 12$.
- **Lumped mass rotary inertia (`beam3d.py:247-250`):** Every lumped nodal mass $m_i$ receives $I_{xx}=I_{yy}=I_{zz}=m_i (0.025\text{ m})^2$.
- **Internal forces & von Mises (`beam3d.py:267-316`):** Computes $\mathbf{f}_\ell = \mathbf{K}_\ell \mathbf{T}\mathbf{u}_g$, takes end-wise maxima $(N, V_y, V_z, T, M_y, M_z)$, and evaluates the conservative upper-bound envelope:
$$\sigma_{\text{norm}} = \frac{|N|}{A} + \frac{|M_y| c_z}{I_y} + \frac{|M_z| c_y}{I_z}, \qquad
\tau_{\text{tot}} = \frac{1.5\sqrt{V_y^2 + V_z^2}}{A} + \frac{|T| \max(c_y, c_z)}{J}, \qquad
\sigma_{\text{vM}} = \sqrt{\sigma_{\text{norm}}^2 + 3 \tau_{\text{tot}}^2}$$
where $c_y = \text{width}/2$, $c_z = \text{height}/2$.

---

## 2. 24-DOF MITC4 Shell & 12-DOF CBUSH Springs (`src/printsim/fea/shell_mitc4.py`)

### 2.1 Local Planar Frame & Mindlin-Reissner Kinematics
For CCW quad corners $[\mathbf{p}_0, \mathbf{p}_1, \mathbf{p}_2, \mathbf{p}_3]$ (`shell_mitc4.py:14-47`), $\mathbf{e}_3 = \text{norm}((\mathbf{p}_2-\mathbf{p}_0)\times(\mathbf{p}_3-\mathbf{p}_1))$, $\mathbf{e}_1 = \text{norm}((\mathbf{p}_1-\mathbf{p}_0)_\perp)$, $\mathbf{e}_2 = \mathbf{e}_3 \times \mathbf{e}_1$, with planar coordinates $(x_k, y_k)$ relative to centroid $\mathbf{p}_c = \frac{1}{4}\sum \mathbf{p}_k$.

Matching `beam3d.py`'s right-hand rule (`shell_mitc4.py:120-127`), section normal rotations $(\beta_x, \beta_y)$ satisfy $\beta_x = +\theta_y$ and $\beta_y = -\theta_x$, yielding:
$$\mathbf{u}(x,y,z) = \begin{bmatrix} u_0 + z\theta_y \\ v_0 - z\theta_x \\ w_0 \end{bmatrix}, \quad
\boldsymbol{\kappa} = \begin{bmatrix} \frac{\partial \theta_y}{\partial x} \\ -\frac{\partial \theta_x}{\partial y} \\ \frac{\partial \theta_y}{\partial y} - \frac{\partial \theta_x}{\partial x} \end{bmatrix}, \quad
\boldsymbol{\gamma} = \begin{bmatrix} \frac{\partial w}{\partial x} + \theta_y \\ \frac{\partial w}{\partial y} - \theta_x \end{bmatrix}$$

### 2.2 Constitutive Matrices & Bathe-Dvorkin (1984) Covariant Shear Tying
With isotropic plane-stress matrix $\mathbf{C}_{\text{iso}} = \begin{bmatrix} 1 & \nu & 0 \\ \nu & 1 & 0 \\ 0 & 0 & \frac{1-\nu}{2} \end{bmatrix}$ (`shell_mitc4.py:74-105`):
$$\mathbf{D}_m = \frac{(Et)_{\text{mem}}}{1-\nu^2}\mathbf{C}_{\text{iso}}, \qquad
\mathbf{D}_b = D_b \mathbf{C}_{\text{iso}} \;\Big(D_b = \frac{E t^3}{12(1-\nu^2)}\Big), \qquad
\mathbf{D}_s = D_s \mathbf{I}_2 \;\Big(D_s = \frac{5}{6} G t\Big)$$
with optional sandwich/buckling overrides `(e_mem_h_n_m, d_bend_nm, g_transverse_h_n_m)`.

To eliminate transverse shear locking on thin plates, covariant shear strains $(\gamma_{\xi z}, \gamma_{\eta z})^T = \mathbf{J}(\gamma_{xz}, \gamma_{yz})^T$ are sampled at the four edge midpoints $A(0,-1)$, $B(1,0)$, $C(0,1)$, $D(-1,0)$ and interpolated linearly across the element (`shell_mitc4.py:149-170, 207-213`):
$$\gamma_{\xi z}(\xi,\eta) = \frac{1-\eta}{2}\gamma_{\xi z}\big|_A + \frac{1+\eta}{2}\gamma_{\xi z}\big|_C, \qquad
\gamma_{\eta z}(\xi,\eta) = \frac{1-\xi}{2}\gamma_{\eta z}\big|_D + \frac{1+\xi}{2}\gamma_{\eta z}\big|_B, \qquad
\mathbf{B}_s = \mathbf{J}^{-1}\mathbf{G}_{\text{cov}}$$
where for edge $i \to j$ (e.g. $0 \to 1$ at $A$):
$$\gamma_{\xi z}\big|_A = \frac{w_1 - w_0}{2} - \frac{y_1 - y_0}{4}(\theta_{x0}+\theta_{x1}) + \frac{x_1 - x_0}{4}(\theta_{y0}+\theta_{y1})$$
Integrated via $2\times 2$ Gauss quadrature ($\xi,\eta = \pm 1/\sqrt{3}$):
$$\mathbf{K}_m = \int_{\Omega} \mathbf{B}_m^T \mathbf{D}_m \mathbf{B}_m\,dA, \qquad
\mathbf{K}_b = \int_{\Omega} \left(\mathbf{B}_b^T \mathbf{D}_b \mathbf{B}_b + \mathbf{B}_s^T \mathbf{D}_s \mathbf{B}_s\right)dA$$

**Reference:** Bathe, K.-J. & Dvorkin, E. N. (1984), "A continuum mechanics based four-node shell element for general non-linear analysis," *Eng. Comput.*, 1(1), 77–88.

### 2.3 Drilling Stabilization (Bug), HRZ Mass & Stress Recovery
- **Drilling DOF (`shell_mitc4.py:225-228`):** Adds an uncoupled diagonal spring $k_{\text{drill}} = 10^{-4} (Et)_{\text{mem}} A$ directly to each node's $\theta_{zk}$ diagonal. Because this penalizes absolute $\theta_{zk}$ rather than relative spin $(\theta_{zk} - \omega_z)$, it violates rigid-body invariance about $\mathbf{e}_3$ (see Critical Bug #1).
- **HRZ Lumped Mass (`shell_mitc4.py:231-244`):** Distributes total mass $M_e = \rho_A A$ proportionally to $\int N_k^2\,dA$, with rotary inertia $I_{\theta x} = I_{\theta y} = \frac{1}{12} m_k t^2$ and $I_{\theta z} = \max(10^{-6} I_{\theta x}, 10^{-12})$.
- **Surface von Mises (`shell_mitc4.py:259-323`):** Evaluates $\mathbf{N} = \mathbf{D}_m \mathbf{B}_m \mathbf{u}_m$ and $\mathbf{M} = \mathbf{D}_b \mathbf{B}_b \mathbf{u}_b$ at $(\xi,\eta)=(0,0)$, then combines $\boldsymbol{\sigma}_{\pm} = \mathbf{N}/t \pm 6\mathbf{M}/t^2$ using plane-stress von Mises $\sqrt{\sigma_x^2 - \sigma_x\sigma_y + \sigma_y^2 + 3\tau_{xy}^2}$. Exact for homogeneous isotropic plates; underestimates ACM Al skin stresses (see Moderate Issue #8).

### 2.4 12-DOF Eccentric CBUSH Fastener Spring
Connecting Node 0 (shell at $\mathbf{p}_0$) to Node 1 (frame at $\mathbf{p}_1$) with local normal $\mathbf{e}_3$ and offset $\mathbf{r} = \mathbf{p}_0 - \mathbf{p}_1$ (`shell_mitc4.py:338-385`):
$$\mathbf{K}_t = \mathbf{R}^T \operatorname{diag}(k_s, k_s, k_n)\mathbf{R}, \qquad
\mathbf{K}_r = k_{\text{rot}} \mathbf{I}_3, \qquad
\mathbf{B}_{\text{off}} = \begin{bmatrix} \mathbf{I}_3 & \mathbf{0} & \mathbf{0} & \mathbf{0} \\ \mathbf{0} & \mathbf{I}_3 & \mathbf{0} & \mathbf{0} \\ \mathbf{0} & \mathbf{0} & \mathbf{I}_3 & -[\mathbf{r}]_\times \\ \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{I}_3 \end{bmatrix}$$
$$\mathbf{K}_{12} = \mathbf{B}_{\text{off}}^T \begin{bmatrix} \mathbf{K}_t & \mathbf{0} & -\mathbf{K}_t & \mathbf{0} \\ \mathbf{0} & \mathbf{K}_r & \mathbf{0} & -\mathbf{K}_r \\ -\mathbf{K}_t & \mathbf{0} & \mathbf{K}_t & \mathbf{0} \\ \mathbf{0} & -\mathbf{K}_r & \mathbf{0} & \mathbf{K}_r \end{bmatrix} \mathbf{B}_{\text{off}}$$
Since $(\mathbf{B}_{\text{off}}\mathbf{u})_{u1} = \mathbf{u}_1 - \mathbf{r}\times\boldsymbol{\theta}_1 = \mathbf{u}_1 + \boldsymbol{\theta}_1 \times \mathbf{r}$, rigid-body motion $\mathbf{u}_0 = \mathbf{u}_1 + \boldsymbol{\theta}_1 \times \mathbf{r}$ produces $\Delta \mathbf{u} = \mathbf{0}$ (exact 6 zero RBMs) and exact moment equilibrium $\mathbf{r}\times\mathbf{f}_{u0} + \mathbf{f}_{\theta 0} + \mathbf{f}_{\theta 1} = \mathbf{0}$.

---

## 3. 2D Heated Bed Plate FEM (`src/printsim/fea/bed_plate_fem.py`)

### 3.1 Plate Mesh, Mount Springs & Modal Solver
Constructs a uniform $n \times n$ ($n = \text{mesh\_n}$, default $4$) grid of MITC4 elements over $[-L_x/2, L_x/2] \times [-L_y/2, L_y/2]$ with effective areal mass $\rho_A = m_{\text{bed}} / (L_x L_y)$ (`bed_plate_fem.py:326-368`). Grounded springs are placed at snapped anchor nodes (`bed_plate_fem.py:118-150`):
$$K_{z,\text{eff}} = \frac{K_{z,\text{mount}} K_{z,\text{carrier}}}{K_{z,\text{mount}} + K_{z,\text{carrier}}}, \qquad
k_{z,\text{each}} = \frac{K_{z,\text{eff}}}{n_{\text{anc}}}, \qquad
k_{xy,\text{each}} = \frac{n_{\text{pts}} k_{\text{lat}}}{n_{\text{anc}}}$$
plus pitch spring $k_{\theta x}$ for collinear 1-Z/2-Z CoreXY leadscrew layouts and rotational regularization $10^{-2}\text{ N}\cdot\text{m/rad}$. Static solves (`bed_plate_fem.py:217-256`) compute gravity sag and $-2.0\text{ N}$ probe deflections at center and all 4 corners; modal solves (`bed_plate_fem.py:259-318`) solve $\mathbf{K}\boldsymbol{\phi} = \omega^2 \mathbf{M}\boldsymbol{\phi}$ via `scipy.linalg.eigh` and select modes with $Z$-energy fraction $> 25\%$.

### 3.2 Thermal Gradient & Post-Buckling Taco Warp Field
Analytical superposition (`bed_plate_fem.py:152-215`):
- **Through-thickness thermal flux:** $q_{\text{top}} = 12.0 \max(0, T_{\text{bed}} - T_{\text{ch}})\;[\text{W/m}^2]$, 1D thermal sagitta $w_{\text{grad},0} = \frac{L_{\text{bed}}^2 \alpha q_{\text{top}}}{8 k}$.
- **Physical 2D spherical thermal bow** for bottom-heated plate ($T_{\text{bot}} > T_{\text{top}} \implies \frac{\partial T}{\partial z} = -q_{\text{top}}/k < 0$):
$$\frac{\partial^2 w}{\partial x^2} = \frac{\partial^2 w}{\partial y^2} = -\alpha \frac{\partial T}{\partial z} = +\frac{\alpha q_{\text{top}}}{k} > 0 \implies w_{\text{phys}}(r_x, r_y) = w_0 + 4 w_{\text{grad},0} \left(\frac{L_x^2}{L_{\text{bed}}^2} r_x^2 + \frac{L_y^2}{L_{\text{bed}}^2} r_y^2\right)$$
  *(Note: coded equation at line 206 has opposite sign and factor `2.0` instead of `4.0` — see Critical Bug #3.)*
- **Rigid-mount elastica taco warp:** With $\Delta L_{\text{ex}} = \max(0, (\alpha_b \Delta T_b - \alpha_f \Delta T_f) L_{\text{mount}} - 0.15\text{ mm})$, $L_{\text{mount}} = 0.85 L_{\text{bed}}$:
$$w_{\text{elastica}} = \frac{2}{\pi}\sqrt{L_{\text{mount}} \Delta L_{\text{ex}}}, \qquad
k_{\text{plate}} = \frac{48 D L_{\text{bed}}}{L_{\text{mount}}^3}, \qquad
w_{\text{taco},0} = w_{\text{elastica}} \sqrt{\frac{k_{\text{restraint}}}{k_{\text{restraint}} + k_{\text{plate}}}}$$
superposed with $w_{\text{taco}}(r_x, r_y) = w_{\text{taco},0}[\cos(\pi r_x)\cos(\pi r_y) - 0.35(r_x^2 - r_y^2)]$ and zeroed to anchor mean.

**References:** Timoshenko & Woinowsky-Krieger (1959), *Theory of Plates and Shells*, Table 41; Leissa (1969), *Vibration of Plates*, NASA SP-160; Boley & Weiner (1960), *Theory of Thermal Stresses*.

---

## 4. Static 3D Solver (`src/printsim/fea/solver.py`)

Solves $\mathbf{K}_{ff}\mathbf{u}_f = \mathbf{F}_f$ via pre-factored sparse LU (`splu`, $n_{\text{free}} > 250$) or Cholesky (`cho_factor`, $n_{\text{free}} \le 250$) across 5 load cases (`solver.py:249-384`):
1. **Gravity ($\mathbf{u}_{\text{grav}}$):** Nozzle-to-bed relative Z sag $|u_{\text{noz},z} - u_{\text{bed},z}|$.
2. **Unit-probe structural loop stiffness:** Self-equilibrating $+1\text{ N}$ at `nozzle_tip` and $-1\text{ N}$ at `bed_surface` along $i \in \{x,y,z\}$:
$$K_{\text{loop},i} = \frac{1\text{ N}}{|u_{\text{noz},i} - u_{\text{bed},i}|\cdot 10^3}\;[\text{N/mm}]$$
3. **Peak XY inertial loads ($\mathbf{u}_x, \mathbf{u}_y$):** $F_x = m_{x,\text{mov}} a_{\max}$ at `toolhead_cog`; $F_y = m_{y,\text{mov}} a_{\max}$ distributed across `toolhead_cog` and `other_gantry` (CoreXY/Delta) or bed/toolhead (bedslinger — see Critical Bug #5).
4. **CoreXY belt preload ($\mathbf{u}_{\text{belt}}$):** Static pulley loads at pretension $T_0$ (`solver.py:348-364`). True CoreXY Belt A + Belt B free-body diagram gives:
   - `g_rl`: $(+2T_0, -2T_0)$, `g_rr`: $(-2T_0, -2T_0)$, `g_fl`: $(0, +2T_0)$, `g_fr`: $(0, +2T_0)$, `y_car_l`: $(+2T_0, 0)$, `y_car_r`: $(-2T_0, 0)$.
   *(Coded values misassign $-T_0$ in Y from `g_rl/g_rr` onto `y_car_l/y_car_r` — see Critical Bug #4.)*
5. **Combined envelope ($\mathbf{u}_{\text{comb}}$):** Maximizes $\|\mathbf{u}_{\text{noz}} - \mathbf{u}_{\text{bed}}\|_2$ over $\mathbf{u}_{\text{grav}} + \mathbf{u}_{\text{belt}} + \frac{s_x}{\sqrt{2}}\mathbf{u}_x + \frac{s_y}{\sqrt{2}}\mathbf{u}_y$ for $s_x, s_y \in \{\pm 1\}$ (omits pure-axis $(\pm 1, 0), (0, \pm 1)$ — see Moderate Issue #6).

---

## 5. Structural Assembly Mechanics (`src/printsim/geometry/builder.py`)

### 5.1 Composite Linear Rails — Newmark (1951) Partial Interaction
(`builder.py:722-825`, calling `components.py:166-197`)
$$\gamma = \frac{1}{1 + \frac{\pi^2 (EA)^* s}{L_{\text{eff}}^2 k_s}}, \qquad
(EA)^* = \frac{E_r A_r \cdot E_b A_b}{E_r A_r + E_b A_b}, \qquad
n_E = \frac{E_r}{E_b}$$
For single-face rail at centroid distance $d_c = \frac{1}{2} h_{\text{beam}} + \frac{1}{2} h_{\text{rail}}$:
$$\Delta I_{\text{par}} = \gamma \frac{A_b (n_E A_r)}{A_b + n_E A_r} d_c^2$$
projected onto exact local beam axes $(\mathbf{e}_y, \mathbf{e}_z)$ via $c_y = (\hat{\mathbf{n}} \cdot \mathbf{e}_y)^2$, $c_z = (\hat{\mathbf{n}} \cdot \mathbf{e}_z)^2$, while rail mass is folded into $\rho_{\text{eff}}$ at constant $A_b$ (zero mass double-counting).

**Reference:** Newmark, N. M., Siess, C. P., & Viest, I. M. (1951), *Proc. Soc. Exp. Stress Anal.*, 9(1), 75–92; EN 1995-1-1 (Eurocode 5) Annex B.

### 5.2 Enclosure Panels — ACM Sandwich, Timoshenko Buckling & Huth-Schwarmann (1986) Fasteners
(`builder.py:332-467, 536, 1527-1710`)
- **ACM / Dibond sandwich:** $D_{\text{bend}} = \frac{E_{\text{Al}} t_s (t - t_s)^2}{2(1-\nu_{\text{Al}}^2)}$, Allen (1969) core shear $S_{\text{core}} = G_c (t - t_s)^2 / t_c$, Timoshenko shear knockdown $D_{\text{eff}} = (D_{\text{bend}}^{-1} + (\pi / b_{\min})^2 S_{\text{core}}^{-1})^{-1}$.
- **Timoshenko shear buckling:** $k_s = 5.34 + 4 (b_{\min}/b_{\max})^2$, $N_{xy,\text{cr}} = k_s \pi^2 D_{\text{eff}} / b_{\min}^2$, knockdown $\psi_b = 0.22 + 0.78 N_{xy,\text{cr}}/(N_{xy,\text{cr}} + 40\text{ kN/m})$.
- **Huth-Schwarmann (1986) single-shear bolt compliance:**
$$C_{\text{bolt}} = \left(\frac{t_{\text{lever}} + t_{\text{ext}}}{2 d_b}\right)^{2/3} b_{\text{Huth}} \left(\frac{1}{t_b E_b} + \frac{1}{t_{\text{ext}} E_{\text{ext}}} + \frac{1}{2 t_b E_{\text{bolt}}} + \frac{1}{2 t_{\text{ext}} E_{\text{bolt}}}\right)$$
with $b_{\text{Huth}} = 3.0$ (metallic) or $4.2$ (composite/polymer/ACM), $t_{\text{lever}} = \min(t_b, 1.5 t_{\text{ext}})$, distributed along all four bay edges via tributary perimeter lengths $ds$ into eccentric CBUSH springs (`offset_m = p_shell - p_frame`).
- **Delta equivalent X-brace area (`builder.py:536`):** $A_{\text{eq}} = \frac{(Gt)_{\text{eff}} L_d^3}{2 E b h}$ (analytically exact for horizontal shear equivalence).

**References:** Huth, H. (1986), *ASTM STP 927*, 221–250; Timoshenko & Gere (1961), *Theory of Elastic Stability*; Allen, H. G. (1969), *Analysis and Design of Structural Sandwich Panels*.

### 5.3 Semi-Rigid Corner Joints, Toolhead Links, Cavity Filling & Bed/Feet Springs
- **Corner joint column sway & torsion knockdown (`builder.py:851-894`):** Exact double-curvature column sway factor $\eta_I = (1 + \frac{6 EI}{L k_r})^{-1}$ and torsional end-spring factor $\eta_J = (1 + \frac{2 GJ}{L k_{\text{tor}}})^{-1}$.
- **Toolhead links (`builder.py:902-1153`):** Splits toolhead mass conservatively across `toolhead` ($40\text{ g}$), `toolhead_cog` ($m_{\text{th}} - 60\text{ g}$), and `nozzle_tip` ($20\text{ g}$), connected by calibrated virtual-massless beam links combining Hertzian carriage compliance (`hertzian_rail_carriage_stiffness`) and Mahieux-Reifsnider polymer $E(T)$.
- **Extrusion cavity filling (`builder.py:1155-1215`):** Fills $58\%$ of extrusion void area ($A_{\text{cav}} = 0.58(wh - A_{\text{ext}})$) into density $\rho_{\text{eff}} = \rho_{\text{base}} + \rho_{\text{fill}} A_{\text{cav}}/A_{\text{ext}}$; for bonded fills (`epoxy_granite`, `uhpc_grout`), adds transformed inertia $n_E I_{\text{cav}}$ and $0.7 n_G (I_{\text{cav},y}+I_{\text{cav},z})$.
- **Bed mount & feet isolator springs (`builder.py:1218-1313`):** 3-pt / 4-pt symmetric mount pitch/roll rotational stiffness $K_\theta = \frac{1}{2} n_{\text{pts}} k_{\text{axial}} r_{\text{mount}}^2$ (exact for equilateral 3-pt and square 4-pt layouts on circle radius $r_{\text{mount}}$).

---

## 6. Numerical Verification Results

Executed via single verification script against `printsim.fea` and `printsim.geometry.builder` (Al 6063-T5 $E=68.9\text{ GPa}$, $\nu=0.33$, $\rho=2700\text{ kg/m}^3$):

| Test Case | Analytical / Expected | Numerical Result | Relative Error / Status |
| :--- | :--- | :--- | :--- |
| 3D Beam `K_loc` symmetry & RBMs | Sym $0$, $6$ zero RBMs | Sym $0.00$, $6$ zero RBMs | **PASS** |
| 3D Beam `M_loc` symmetry & PD | Sym $0$, $\lambda_{\min} > 0$ | Sym $0.00$, $\lambda_{\min}=4.50\times 10^{-6}$ | **PASS** |
| Cantilever $F_y=10\text{ N}$ ($u_y, +\theta_z$) | $FL^3/3EI_z$, $+FL^2/2EI_z$ | $4.30\times 10^{-16}$, $5.74\times 10^{-16}$ | **EXACT** |
| Cantilever $F_z=10\text{ N}$ ($u_z, -\theta_y$) | $FL^3/3EI_y$, $-FL^2/2EI_y$ | $2.15\times 10^{-16}$, $2.87\times 10^{-16}$ | **EXACT** |
| Cantilever axial & torsion | $FL/EA$, $TL/GJ$ | $0.00$, $0.00$ | **EXACT** |
| Cantilever 4-el $f_1$ vs Euler-Bernoulli | $71.5136\text{ Hz}$ | $71.5159\text{ Hz}$ | **+0.0033%** |
| `beam_rotation_matrix` ($0^\circ, 81^\circ, 85^\circ, \pm Z$) | $\mathbf{R}^T\mathbf{R}=\mathbf{I}$, $\det=+1$ | $\le 2.22\times 10^{-16}$, $+1.000000$ | **EXACT** |
| Distorted MITC4 membrane uniaxial patch | $68.900000\text{ MPa}$ | $68.900000\text{ MPa}$ | **$2.06\times 10^{-16}$ (EXACT)** |
| Distorted MITC4 bending patch (stress / energy) | $10.335000\text{ MPa}$ / $2.856336\times 10^{-2}\text{ J}$ | $10.335000\text{ MPa}$ / $2.856336\times 10^{-2}\text{ J}$ | **$3.44\times 10^{-16}$ / $1.31\times 10^{-12}$ (EXACT)** |
| Single MITC4 rigid rotation $R_z$ about $\mathbf{e}_3$ | $0.0\text{ J/rad}^2$ | **$1,523.4\text{ J/rad}^2$** | **FAIL (CRITICAL BUG #1)** |
| Unconstrained CoreXY + 3 mm Al panels lowest 9 modes | $6$ zeros (`[0,0,0,0,0,0,...]`) | **`[0, 0, 0, 0.76, 0.81, 1.13, 57.1, ...]` Hz** | **FAIL (CRITICAL BUGS #1 & #2)** |
| Eccentric CBUSH 3D RBM energy & equilibrium | $0\text{ J}$, $\sum F=0, \sum M=0$ | $2.56\times 10^{-13}\text{ J}$, $0.0$, $4.37\times 10^{-11}$ | **EXACT** |
| `bed_mount_spring` (`offset=0`, $14\text{ mm}$ gap) RBM pitch | $0.0\text{ J/rad}^2$ | **$7,056.0\text{ J/rad}^2$** | **FAIL (CRITICAL BUG #2)** |
| 4-corner 300×300×8 mm bed plate (`mesh_n=12`) | Timoshenko $13.190\;\mu\text{m}$ / Leissa $155.11\text{ Hz}$ | $13.203\;\mu\text{m}$ / $148.29\text{ Hz}$ | **+0.10% sag / -4.4% $f_1$ (carrier + shear)** |
| 4-corner 300×300×8 mm bed plate (`mesh_n=4` default) | Converged $13.203\;\mu\text{m}$ / $148.29\text{ Hz}$ | $11.398\;\mu\text{m}$ / $157.96\text{ Hz}$ | **-13.6% sag / +6.5% $f_1$ (coarse mesh)** |

---

## 7. Audit Findings Table

| # | Severity | Location (`file:line`) | Finding & Physical Impact | Concrete Recommended Fix |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **`CRITICAL BUG`** | `src/printsim/fea/shell_mitc4.py:225-228` | **Grounded drilling DOF stabilization violates 3D rigid-body rotational invariance.** `k_loc[6*k+5, 6*k+5] = 1e-4 * e_mem_h * area` penalizes absolute $\theta_{zk}$ relative to inertial ground rather than continuum spin $\omega_z = \frac{1}{2}(\partial_x v - \partial_y u)$. A single 3 mm Al quad stores `1,523.4 J/rad²` under rigid in-plane rotation $R_z$, and an unconstrained paneled CoreXY frame exhibits spurious grounded rotational modes (`0.76, 0.81, 1.13 Hz`). | Replace diagonal `k_drill` with Hughes-Brezzi / MacNeal-Harder frame-invariant drilling penalty $\mathbf{K}_{\text{drill}} = \alpha E t \int_A (\theta_z - \frac{1}{2}(\frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}))^2 dA$ (or centroid spin row $\mathbf{b}_d$ + zero-row-sum relative nodal matrix $\mathbf{I}_4 - \frac{1}{4}\mathbf{1}\mathbf{1}^T$), yielding 6 exact zero RBMs. |
| **2** | **`CRITICAL BUG`** | `src/printsim/geometry/builder.py:1230, 1259` | **`bed_mount_spring` sets `offset_m=(0.0, 0.0, 0.0)` across a 14 mm vertical node gap.** `bed_frame` sits 14 mm below `bed_surface`, yet `FastenerSpring` omits `offset_m`. Any rigid pitch/roll of the bed stores `7,056 J/rad²` spurious energy (`kinematic_3pt`), and horizontal bed acceleration (`F_y` on bedslingers) transfers zero overturning moment $0.014 F_y$ into `bed_frame`. | Set `offset_m=(0.0, 0.0, -0.014)` (`bed_frame_pos - bed_center_m`, matching `shell_node_id="bed_frame"` and `frame_node_id="bed_surface"`). Verify `v_pitch^T K v_pitch == 0`. |
| **3** | **`CRITICAL BUG`** | `src/printsim/fea/bed_plate_fem.py:206` | **Thermal gradient bow has inverted curvature sign and $0.5\times$ 2D spherical curvature.** Bottom-heated bed with steady upward flux $q_{\text{top}} > 0$ has $T_{\text{bot}} > T_{\text{top}} \implies \frac{\partial^2 w}{\partial x^2} = +\frac{\alpha q_{\text{top}}}{k} > 0$ (concave-up **bowl**, center low), which *adds* to gravity sag. Coded `w_grad = bed_grad_bow_um * (1 - 2*(rx^2 + ry^2))` is a concave-down **dome** (partially cancelling gravity sag!) with curvature $\frac{1}{2}\frac{\alpha q_{\text{top}}}{k}$ and ignores $L_x \ne L_y$ aspect ratio. | Replace with `w_grad = 4.0 * bed_grad_bow_um * ((lx_m/l_bed)**2 * rx_nodes**2 + (ly_m/l_bed)**2 * ry_nodes**2)` (positive concave-up bowl for $q_{\text{top}} > 0$, or solve directly as consistent thermal bending moments $M_T = (1+\nu) D \alpha q_{\text{top}} / k$ through the MITC4 stiffness matrix $\mathbf{K}$). |
| **4** | **`CRITICAL BUG`** | `src/printsim/fea/solver.py:351-358` | **CoreXY belt preload FBD puts spurious $-T_0$ Y-force on moving Y-carriages.** Under equal pretension $T_A = T_B = T_0$, Belt A ($-T_0$) and Belt B ($+T_0$) cancel to $0$ in Y on `y_car_l` and `y_car_r`, while rear motors/idlers (`g_rl`, `g_rr`) carry $-2T_0$ in Y. Code puts `-t0` on `g_rl/g_rr` and `-t0` on `y_car_l/y_car_r`, pulling the moving X-gantry forward by $-2T_0$ and halving compressive load in the rear Y-rail segments. | Change `belt_loads` to: `("g_rl", +2*t0, -2*t0)`, `("g_rr", -2*t0, -2*t0)`, `("g_fl", 0.0, +2*t0)`, `("g_fr", 0.0, +2*t0)`, `("y_car_l", +2*t0, 0.0)`, `("y_car_r", -2*t0, 0.0)`. Preserves $\sum F = 0, \sum M = 0$ and zero net Y-force on the moving gantry. |
| **5** | **`CRITICAL BUG`** | `src/printsim/fea/solver.py:321-327` | **Cartesian bedslinger Y-acceleration applies phantom horizontal force $+m_x a_y$ to stationary toolhead.** During pure Y bed acceleration, the X-gantry/toolhead is stationary in Y; the reaction $+m_y a_y$ acts on the Y-motor/base frame, not on `toolhead_cog`. Applying `+m_x * a_y` directly to `toolhead_cog` bends the Z-portal even on rigidly bolted feet, while `F_y[bed_dof+1]` omits carriage mass (`spec.bed.mass_g` instead of `y_moving_mass_kg`). | Apply `-assembly.y_moving_mass_kg * accel_m_s2` at `bed_dof + 1` (and equal-and-opposite reaction `+assembly.y_moving_mass_kg * accel_m_s2` distributed to base Y-motor/idler mount nodes `base_front`/`base_rear` so net external force is zero on compliant feet); set toolhead Y-force to `0.0`. |
| **6** | **`MODERATE ISSUE`** | `src/printsim/fea/solver.py:368-384` | **Combined acceleration envelope omits pure X and pure Y axes.** Only the 4 diagonal combinations $(\pm 0.7071\,\mathbf{u}_x \pm 0.7071\,\mathbf{u}_y)$ are checked. When one axis compliance dominates and gravity sag is small, $0.7071\,\|\mathbf{u}_y\|$ underestimates the true $\|\mathbf{a}\| \le a_{\max}$ worst-case deflection by up to $29.3\%$. | Loop over all 8 unit-circle directions `(sx, sy) in [(1,0), (-1,0), (0,1), (0,-1), (s,s), (s,-s), (-s,s), (-s,-s)]` where `s = math.sqrt(0.5)`. |
| **7** | **`MODERATE ISSUE`** | `src/printsim/fea/solver.py:312-341` | **CoreXY/Delta `F_x`, `F_y` omit Z-probe mass in `th_share`, lump X-beam mass onto end carriages, and lack motor reaction balance.** Line 330 uses `toolhead.total_mass_g` (ignoring Z-probe `added_mass_g`), dumps $100\%$ of X-beam/rail Y-inertia onto `y_car_l/y_car_r` (zero midspan beam inertial bending), and applies unbalanced external forces without opposing belt reactions at A/B motors (inducing unphysical static rocking on soft isolator feet). | Use `_effective_toolhead_mass_kg(spec)` in `th_share`; apply consistent beam inertia ($\mathbf{M}_{\text{gantry}} \mathbf{a}_y$) across X-beam nodes; add opposing motor reaction forces at `g_rl/g_rr` (and `g_fl/g_fr` for AWD) so $\sum \mathbf{F}_{\text{ext}} = \mathbf{0}$. |
| **8** | **`MODERATE ISSUE`** | `src/printsim/fea/shell_mitc4.py:315-316` & `builder.py:1555` | **ACM/Dibond sandwich stress & core shear discrepancies.** `mitc4_shell_von_mises_mpa` divides `n_res` by full thickness $t=3\text{ mm}$ instead of Al skin thickness $2 t_s = 0.6\text{ mm}$ ($5\times$ underestimate of skin membrane stress) and uses $6 M / t^2$ instead of $M / (t_s(t - t_s))$ ($1.85\times$ underestimate of skin bending stress). Separately, `builder.py:1555` sets `g_trans_base = g_core * t_core` instead of Allen's `g_core * (t - t_skin)**2 / t_core` used at `builder.py:372` ($21\%$ lower on 3 mm Dibond). | Pass optional `skin_thickness_m` to `mitc4_shell_von_mises_mpa` for sandwich skin stress recovery, and unify line 1555 with Allen's formula `g_core * ((panel_t_m - t_skin)**2) / max(t_core, 1e-5)`. |
| **9** | **`MODERATE ISSUE`** | `src/printsim/fea/beam3d.py:247-250` | **Uniform 25 mm radius of gyration for all lumped masses.** `rot_i = m * (0.025**2)` severely underestimates rotary inertia of `bed_surface` (a 300×300 mm bed has $r_{g,x}=r_{g,y}=86.6\text{ mm}$, $12\times$ larger $I_{xx}, I_{yy}$, and $24\times$ larger $I_{zz}$), inflating 3D frame bed pitch/roll eigenfrequencies. | Add an optional `rot_inertia_kg_m2: tuple[float, float, float] | None = None` field to `StructuralNode` and populate exact plate inertia $(\frac{1}{12}m L_y^2, \frac{1}{12}m L_x^2, \frac{1}{12}m(L_x^2+L_y^2))$ on `bed_surface`. |
| **10** | **`MODERATE ISSUE`** | `src/printsim/fea/bed_plate_fem.py:189-201` & `printer.py:161` | **Unit-ambiguous `k_restraint_thermal` & missing Euler buckling threshold.** `k_restraint_thermal` lacks a unit suffix (`8000` for `rigid_4pt` vs `k_plate_taco = 2.17e6 N/m`) and uses an empirical `arch_fraction` attenuation on inextensible elastica rather than checking $N_{\text{th}} > N_{\text{cr}}$ with in-plane axial compliance. Additionally, default `mesh_n=4` has `-13.6%` static sag / `+6.5%` $f_1$ discretization stiffness bias vs converged `mesh_n=12`. | Rename to `k_restraint_thermal_n_m` (or replace with physical post-buckling $\sqrt{\max(0, \Delta L - \Delta L_{\text{cr}}) L}$ with axial compression threshold); raise default `mesh_n` to `6` or `8` (still `< 10 ms`). |
| **11** | **`MODERATE ISSUE`** | `src/printsim/geometry/builder.py:913-956` | **Compounded `probe_comp` exponent in `compute_toolhead_link_props`.** `probe_comp` (`0.34` for `printed_tap`) enters `e_mount_eff` ($\times 1$), `k_int_n_m` ($\times 1$), `mat_scale` ($\times 0.5$), `k_r_spring` ($\times 2.5$ combined), and `e_mount_eff * iy_mount` ($\times 2.0$ before joint series knockdown), creating an opaque non-linear scaling ($\sim p^{2.0\text{--}2.5}$). | Apply `probe_comp` cleanly once as a dedicated series carriage-rail compliance spring $k_{\text{probe}}$ (or single multiplier on `k_int_n_m` without also scaling `e_mount_eff` and `k_r_spring`), and recalibrate catalog factors accordingly. |
| **12** | **`MODERATE ISSUE`** | `src/printsim/geometry/builder.py:762-799, 2159-2183` | **Dual-rail & backer Newmark composite approximations.** For `dual_top_bottom` symmetric rails (`builder.py:782`), the beam centroid carries zero axial load in bending so slip depends on $E_r A_r$, not the smaller harmonic $(EA)^*$; using $(EA)^*$ overestimates $\gamma$ (`0.72` vs `0.48` for MGN12 on 2020). Conversely, opposite-face rail + backer (`builder.py:2168`) runs two independent single-side calls, underestimating combined opposite-flange $I_{\text{par}}$ by ~25–40%. | For `is_symmetric_dual`, pass `ea_star = E_r A_r` into Newmark $\gamma$; for rail + opposite backer, compute the exact 3-layer transformed neutral axis $\bar{z}$ and coupled parallel-axis inertia in one step. |
| **13** | **`MODERATE ISSUE`** | `src/printsim/fea/bed_plate_fem.py:292` | **Latent `ValueError` comparing tuples containing `np.ndarray`.** `[m for m in fallback_modes if m not in z_modes]` compares `(freq_hz, uz)` tuples; if `len(z_modes) < 3` and two distinct degenerate square-plate modes share an identical float `freq_hz`, `uz_a == uz_b` raises `ValueError: The truth value of an array with more than one element is ambiguous`. | Track selected eigenmode integer indices `z_indices: set[int]` and filter fallback by `idx not in z_indices`. |
| **14** | **`MINOR / LIMITATION`** | `src/printsim/fea/beam3d.py:201-204` | **Lumped beam gravity omits fixed-end moments.** Single-element fixed-fixed beams exhibit zero self-weight sag/moment; 1-element cantilevers overpredict gravity tip sag by $+33.3\%$ ($\frac{WL^3}{6EI}$ vs $\frac{WL^3}{8EI}$). | Optionally integrate consistent gravity load $\mathbf{f}_{g,\ell} = \int_0^L \mathbf{N}^T \mathbf{R}\,(0,0,-\rho A g)^T dx$ (adds $\pm w_y L^2/12, \pm w_z L^2/12$ fixed-end moments). |
| **15** | **`MINOR / LIMITATION`** | `src/printsim/fea/beam3d.py:121, 291-309` | **Beam torsional inertia & conservative stress envelope.** (a) For composite rail/filled beams, `rx2 = (iy + iz)/area` uses modulus-weighted $I$, inflating torsional rotary inertia; (b) `element_internal_forces` combines end-1/end-2 maxima, corner bending stress, neutral-axis shear, and circular torsion formula $Tr/J$ (upper bound; overestimates rectangular open-bar torsional shear). | Evaluate von Mises at nodes 1 and 2 separately; store geometric `ip_m4` for composite mass if high-frequency torsional modes are required. |
| **16** | **`MINOR / LIMITATION`** | `src/printsim/geometry/builder.py:1173-1205, 1384` | **Extrusion cavity filling applied after corner-joint knockdown & omits axial $EA$ boost.** Bonded fill (`epoxy_granite`, `uhpc_grout`) adds $n_E I_{\text{cav}}$ after `_apply_corner_joint_compliance` has already knocked down column $I$, and does not add axial $E_{\text{fill}} A_{\text{cav}}$. | Call `_apply_extrusion_filling` before `_apply_corner_joint_compliance`, or apply column `eta_iy/eta_iz/eta_j` after bonded fill. |
| **17** | **`MINOR / LIMITATION`** | `src/printsim/fea/shell_mitc4.py:251-255, 282` | **MITC4 uniform `0.25` nodal gravity & centroid-only stress recovery.** Exact on rectangular bays (all PrintSim panels/beds), approximate on distorted quads; centroid-only stress misses edge/corner bending peaks on coarse $2\times 2$ meshes. | Use `m_nodes * (-9.80665)` (or $\int N_k dA$) for gravity on distorted quads; evaluate stress at $2\times 2$ Gauss points when peak corner stress is needed. |
| **18** | **`VERIFIED`** | `src/printsim/fea/beam3d.py:11-150` | **12×12 3D Euler-Bernoulli stiffness, consistent mass & rotation matrix.** Right-hand-rule signs verified in both $x$-$y$ ($v'=+\theta_z$) and $x$-$z$ ($w'=-\theta_y$) planes; cantilever deflections/rotations match exact formulas to $\le 5.7\times 10^{-16}$; 4-el $f_1$ within $+0.0033\%$; vertical $\pm Z$ rotations orthonormal with $\det = +1$. | No fix needed. |
| **19** | **`VERIFIED`** | `src/printsim/fea/shell_mitc4.py:149-217, 326-385` | **Bathe-Dvorkin (1984) MITC4 membrane/bending/shear operators & eccentric CBUSH springs.** Distorted-quad membrane & Kirchhoff bending patch tests pass to $2\times 10^{-16}$ (stress) and $1.3\times 10^{-12}$ (energy, zero shear locking). Eccentric CBUSH matrix is symmetric PSD with 6 exact zero RBMs ($2.6\times 10^{-13}\text{ J}$) and exact $\sum F = 0, \sum M = 0$. | No fix needed (except Fix #1 for drilling DOF and Fix #2 at `bed_mount_spring` call site). |
| **20** | **`VERIFIED`** | `src/printsim/fea/bed_plate_fem.py:320-378` & `builder.py:390-440, 536, 762, 877, 1247` | **Kirchhoff/Leissa plate convergence & structural assembly formulas.** Converged `mesh_n=12` bed sag matches Timoshenko Table 41 to $+0.10\%$ ($13.203$ vs $13.190\;\mu\text{m}$); $f_1$ matches Leissa to $-4.4\%$ (carrier + Mindlin shear). Huth-Schwarmann (1986), Timoshenko $k_s$, Delta X-brace $A_{\text{eq}}$, Newmark $\gamma$, column corner-joint $\eta_I/\eta_J$, symmetric 3/4-pt bed $K_\theta$, and mass/cost non-double-counting all verified exact. | No fix needed. |
