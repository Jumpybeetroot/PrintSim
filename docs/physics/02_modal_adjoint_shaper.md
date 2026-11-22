# PrintSim Physics Audit 02: Modal FEA, Analytical Adjoint / Fox–Kapoor Sensitivities, RCSA / LM-FBS & Klipper Input Shapers

**Audit Scope (Read-Only):**
- `src/printsim/fea/modal.py`
- `src/printsim/fea/sensitivity.py`
- `src/printsim/fea/rcsa.py`
- `src/printsim/fea/klipper_shaper.py`

**Executive Summary:** PrintSim's structural dynamics core rests on sound first-principles formulations: exact 12-DOF/24-DOF generalized modal eigenanalysis, exact static adjoint and Fox–Kapoor (1968) eigenvalue sensitivities (verified to $\le 0.03\%$ against $\pm 0.1\%$ central finite differences and $0.0000\%$ on Euler sum identities), an exact state-space/transfer-matrix Timoshenko free–free beam receptance and LM-FBS coupler (static cantilever error $0.00005\%$, exact reciprocity $0.00$), and Klipper-exact `shaper_defs.py` / `shaper_calibrate.py` mathematics.

The audit identified **1 Critical Bug**, **9 Moderate Issues**, and **7 Minor Limitations**:
- **Critical Bug (`rcsa.py:381`):** Carriage rotational damping in `Y_car[:, 1, 1]` multiplies $2\zeta_f\sqrt{k_{\text{rot}} J_{\text{car}}}$ by dimensionless $\omega/\omega_0$ instead of $\omega$ ($\text{rad/s}$), violating dimensional homogeneity ($\text{N}\cdot\text{m}\cdot\text{s}$ added to $\text{N}\cdot\text{m}$) and suppressing rotational damping by $1/\omega_0 \approx 583\times$.
- **Top Moderate Issues:**
  1. `sensitivity.py:92, 114` & `modal.py:461`: Dense `scipy.linalg.eigh(K_ff, M_ff)` Cholesky-factorizes ill-conditioned $M_{ff}$ ($\kappa(M_{ff}) \approx 7.1\times 10^{11}$ from `m_floor = 1e-12`), causing $10^{-4}$–$10^{-3}\,\text{Hz}$ eigenvalue jitter on all presets with $n_{\text{free}} \le 500$.
  2. `sensitivity.py:825–834, 1039–1042`: Panel bay `stiffness_sens_hz` and `du_rel_dlnE_m` fold perimeter CBUSH fastener springs into `bay_acc` (+172% to +194% above shell-only $E$-sensitivity), and `hz_per_100g` assumes linear $t^1$ scaling instead of cubic $t^3$ plate bending (+322.6% error vs thickness FD).
  3. `sensitivity.py:125–149`: `_dailey_subspace_vector` performs fixed excitation-aligned subspace projection rather than Dailey (1989) $r\times r$ interaction-matrix eigenvalue splitting.
  4. `modal.py:485–488` & `sensitivity.py:487–490`: Bedslinger `fy_exc` applies a $2\,\text{N}$ dipole ($+1\,\text{N}$ on toolhead CoG, $-1\,\text{N}$ on bed) instead of a unit bed inertial load.
  5. `modal.py:614–618`: Subsystem KE shares use $\boldsymbol{\phi}_S^T(M_{ff}\boldsymbol{\phi})_S$, which is not guaranteed non-negative under consistent beam mass cross-terms and differs from `ke_panel` ($\boldsymbol{\phi}_p^T M_{ff}\boldsymbol{\phi}_p$).
  6. `modal.py:900–924`: `print_impact_score` normalizes purely relatively (`c_k / c_max` over only the first `k_ret` modes), assigning `100.0` ("critical") to every printer with any mode $\le 140\,\text{Hz}$ regardless of absolute stiffness.
  7. `sensitivity.py:952–954, 1080–1085`: Negative `"modal"` lens heat scores use different normalization denominators for structural elements (`max_neg_xy_net`) vs lumped masses (`max_mass_pen`).
  8. `klipper_shaper.py:105–158`: `get_shaper_pulses` returns unnormalized amplitudes ($\sum A_i \neq 1$) for the 5 stock Klipper shapers (`ZV`–`3HUMP_EI`) but unit-normalized amplitudes ($\sum A_i = 1$) for `NONE`, `ZVDD`, `SMOOTH_ZV`, `SMOOTH_EI`.
  9. `klipper_shaper.py:508, 589–591`: `_fit_single_shaper` evaluates a single damping ratio ($\zeta=0.10$) instead of Klipper's 3-damping worst-case envelope $[0.075, 0.10, 0.15]$, and searches only $[0.7 f_{\text{pk}}, 1.35 f_{\text{pk}}]$ instead of the full spectrum.

---

## 1. 3D Modal FEA & Structural Receptance (`src/printsim/fea/modal.py`)

### 1.1 Governing Equations & Assumptions

#### Generalized Eigenproblem (`modal.py:409–461`)
After applying Dirichlet boundary conditions (removing `fixed` 6-DOF and `pinned` 3-DOF constraints), the free-DOF structural stiffness $K_{ff} \in \mathbb{R}^{n_f \times n_f}$ and mass $M_{ff} \in \mathbb{R}^{n_f \times n_f}$ matrices are explicitly symmetrized:
$$K_{ff} \leftarrow \tfrac{1}{2}(K_{ff} + K_{ff}^T), \qquad M_{ff} \leftarrow \tfrac{1}{2}(M_{ff} + M_{ff}^T)$$
To prevent singular $M_{ff}$ on zero-mass rotational or massless virtual-link DOFs (`toolhead_mount`, `nozzle_stem`, `bed_arm_*`), a diagonal mass floor is added (`modal.py:420–422`):
$$m_{\text{floor}} = \max\!\left(10^{-12}\max_i M_{ff,ii},\; 10^{-12}\,\text{kg}\right), \qquad M_{ff} \leftarrow M_{ff} + \operatorname{diag}\!\left(\mathbb{1}[M_{ff,ii} < m_{\text{floor}}]\,m_{\text{floor}}\right)$$
The undamped free-vibration eigenproblem is:
$$K_{ff}\boldsymbol{\phi}_k = \lambda_k M_{ff}\boldsymbol{\phi}_k, \qquad \lambda_k = \omega_k^2 = (2\pi f_k)^2, \qquad m_{\text{gen},k} = \boldsymbol{\phi}_k^T M_{ff}\boldsymbol{\phi}_k$$
- For $n_f > 250$ (`modal.py:438–457`), the lowest $k_{\text{scan}}$ modes (`56` with panels, `max(num_modes, 16)` without) are extracted via ARPACK shift-invert Lanczos (`scipy.sparse.linalg.eigsh`) at $\sigma = 0$ with `OPinv = splu(K_ff).solve`, solving $K_{ff}^{-1} M_{ff}\boldsymbol{\phi}_k = \mu_k \boldsymbol{\phi}_k$ for largest $|\mu_k| = 1/\lambda_k$.
- For $n_f \le 250$ (or `RuntimeError` fallback, `modal.py:459–461`), `scipy.linalg.eigh(K_ff, M_ff, subset_by_index=[0, k_scan-1])` is used.

**Conditioning Assessment:** Because $m_{\text{floor}} = 10^{-12}$ and MITC4 shell drilling inertia $I_{\theta_z} = 10^{-6}\cdot\tfrac{1}{12}m_n t^2 \sim 10^{-12}$, $M_{ff}$ is severely ill-conditioned ($\kappa(M_{ff}) = 7.14\times 10^{11}$ on `corexy_voron_style.yaml`), whereas $K_{ff}$ is well-conditioned ($\kappa(K_{ff}) = 7.50\times 10^6$). Shift-invert `eigsh` LU-factorizes $K_{ff}$ and is accurate to machine precision, whereas dense `eigh(K_ff, M_ff)` Cholesky-factorizes $M_{ff}$ and loses 5–6 significant digits on eigenvalue differences (Section 2.2). On `corexy_voron_style.yaml` ($n_f=372$), `eigsh` and `eigh` agree to $1.80\times 10^{-5}\,\text{Hz}$ on mode 1 ($92.8294\,\text{Hz}$).

#### Modal Compliance / DC Receptance & Units (`modal.py:480–503, 661–703`)
For axis $a \in \{X, Y\}$, let $\hat{\mathbf{f}}_a$ be the unit inertial excitation pattern (dimensionless force distribution, $\sum_i \hat{f}_{a,i} = 1$ for CoreXY/Delta) and $\mathbf{c}_a^{\text{noz}} = \mathbf{e}_{\text{noz},a} - \mathbf{e}_{\text{bed},a}$, $\mathbf{c}_a^{\text{car}} = \mathbf{e}_{\text{th},a} - \mathbf{e}_{\text{bed},a}$. Each mode's signed static compliance contribution is:
$$C_{k,a}^{\text{noz}} = \frac{(\mathbf{c}_a^{\text{noz}\,T}\boldsymbol{\phi}_k)(\boldsymbol{\phi}_k^T \hat{\mathbf{f}}_a)}{m_{\text{gen},k}\,\omega_k^2}, \qquad R_{k,a} = \left|C_{k,a}^{\text{noz}}\right|$$
**Dimensional Check:** For arbitrary eigenvector scaling $[u]$:
$$\left[C_{k,a}^{\text{noz}}\right] = \frac{[u]\cdot [u]}{([u]^2\cdot\text{kg})\cdot \text{s}^{-2}} = \frac{1}{\text{kg}\cdot\text{s}^{-2}} = \frac{\text{m}}{\text{N}}$$
By the spectral expansion $K_{ff}^{-1} = \sum_{k=1}^{n_f} \frac{\boldsymbol{\phi}_k\boldsymbol{\phi}_k^T}{m_{\text{gen},k}\omega_k^2}$, $\sum_{k=1}^{n_f} C_{k,a}^{\text{noz}} = \mathbf{c}_a^{\text{noz}\,T} K_{ff}^{-1}\hat{\mathbf{f}}_a$ is exact static compliance in $\text{m/N}$.

**Bedslinger Y-Excitation Issue (`modal.py:485–488`):** For `cartesian_bedslinger`, `fy_exc` sets `+1.0` at `cog_idx` and `-1.0` at `bed_idx` ($\|\hat{\mathbf{f}}_y\|_1 = 2$). Physically, only the heated bed accelerates along Y; its belt reaction enters the base frame, not the stationary Z-portal toolhead. Putting $+1\,\text{N}$ on `cog_idx` artificially excites portal fore-aft bending in Y-receptance and inflates Y compliance relative to unit-force scaling.

#### Signed Rayleigh Carriage-to-Nozzle Compliance Ratio & Series Drive Coupling (`modal.py:206–208, 937–996`)
Carriage effective structural frequency is computed via a compliance-weighted harmonic mean (`modal.py:948–959`):
$$w_k = \frac{R_k}{f_k^2}, \qquad f_{\text{car},a} = \left(\frac{\sum_{k \in \mathcal{S}} w_k / f_k^2}{\sum_{k \in \mathcal{S}} w_k}\right)^{-1/2}$$
where $\mathcal{S} = \{k : w_k \ge 0.01 \max_j w_j\}$. Because $R_k \propto 1/f_k^2$, $w_k \propto 1/f_k^4$ (an effective $1/f_k^6$ weighting on modal residue $|(\mathbf{c}^T\boldsymbol{\phi}_k)(\boldsymbol{\phi}_k^T\hat{\mathbf{f}})|/m_{\text{gen},k}$), which heavily prioritizes the lowest significant mode. Nozzle overhang compliance is then folded in via the truncated signed modal compliance sum ratio (`modal.py:990–994`):
$$\rho_a = \max\!\left(1.0,\; \frac{\left|\sum_{k=1}^{k_{\text{scan}}} C_{k,a}^{\text{noz}}\right|}{\max\!\left(\left|\sum_{k=1}^{k_{\text{scan}}} C_{k,a}^{\text{car}}\right|,\, 10^{-18}\right)}\right), \qquad f_{\text{struct},a} = \frac{f_{\text{car},a}}{\sqrt{\rho_a}}$$
Finally, structural compliance is combined in series (Dunkerley's formula) with belt, motor, and bracket drive resonances (`modal.py:167–208`):
$$\frac{1}{f_{\text{eff},a}^2} = \frac{1}{f_{\text{struct},a}^2} + \frac{1}{f_{\text{belt},a}^2} + \frac{1}{f_{\text{motor},a}^2} + \frac{1}{f_{\text{bracket},a}^2}, \qquad s_i = \frac{1/f_i^2}{\sum_j 1/f_j^2}$$

**Numerical Truncation Check:**
- Paneled Voron 2.4 (`k_scan = 56`): $\sum C_{k,X}^{\text{noz}} = 1.8336\times 10^{-6}\,\text{m/N}$ vs exact $\mathbf{c}_X^T K^{-1}\hat{\mathbf{f}}_X = 1.8428\times 10^{-6}\,\text{m/N}$ ($-0.50\%$ error); $\rho_X = 1.5614$ vs exact $1.5721$ ($-0.68\%$).
- Open-frame CoreXY (`k_scan = 16`): nozzle sum error $+0.73\%$, carriage error $+0.06\%$, $\rho_X = 1.2754$ vs exact $1.2669$ ($+0.67\%$).

#### Subsystem Kinetic Energy Shares (`modal.py:576–577, 614–618, 746–759`)
Panel KE fraction uses the block quadratic form:
$$T_{\text{panel}} = \frac{\boldsymbol{\phi}_p^T M_{ff} \boldsymbol{\phi}_p}{m_{\text{gen}}}$$
which is exact because panel shell nodes only attach to the frame via massless CBUSH springs ($M_{p,\neg p} = 0$). However, the four structural subsystem shares (`frame`, `gantry`, `toolhead`, `bed`) use the row-partitioned form:
$$\tilde{T}_\alpha = \frac{\max\!\left(0,\; \boldsymbol{\phi}_{S_\alpha}^T (M_{ff}\boldsymbol{\phi})_{S_\alpha} / m_{\text{gen}}\right)}{\max(1 - T_{\text{panel}}, 10^{-6})}$$
Because `beam3d.py` uses a **consistent** 12×12 beam mass matrix (containing negative off-diagonal translation-rotation coupling entries $-13L$, $-3L^2$), any beam bridging two different node roles (`frame`–`gantry` Y-rails, `gantry`–`toolhead` X-gantry, `frame`–`bed` arms) introduces cross-terms $\boldsymbol{\phi}_{S_\alpha}^T M_{S_\alpha, S_\beta}\boldsymbol{\phi}_{S_\beta}$ that are **not guaranteed non-negative**. Although all four audited YAML presets had 0 negative hits in the scanned low modes (due to dominant lumped nodal masses), out-of-phase higher bending modes can go negative and clip to `0.0`. Furthermore, `_classify_mode_diagnosis` (`modal.py:781–795`) consumes un-renormalized $\tilde{T}_\alpha$, whereas `ModeShape.subsystem_share_pct` (`modal.py:756–759`) renormalizes them to $100\%$.

#### Print-Impact Score (`modal.py:815, 900–924`)
For each retained mode $m \in \{1,\dots,k_{\text{ret}}\}$:
$$c_m = \frac{R_{m,X} + R_{m,Y}}{\max(f_m^2, 1)}, \qquad c_{\text{norm},m} = \frac{c_m}{\max_{j \le k_{\text{ret}}} c_j}$$
$$\text{Score}_m = \operatorname{clip}_{[0,100]}\!\left(100\sqrt{c_{\text{norm},m}}\,(0.15 + 0.85\min(1, A_{\text{noz},m}/0.30))\,\min\!\left(1, (140/f_m)^{0.6}\right) \cdot B_{\text{whip},m}\right)$$
Because $c_{\text{norm},m}$ is purely relative to the worst mode in the retained set (and $A_{\text{noz},m}$ is normalized to $\max_i \|\mathbf{u}_i\| = 1$), **every** printer with any mode $\le 140\,\text{Hz}$ and $A_{\text{noz}} \ge 0.30$ receives `100.0` ("critical") regardless of its absolute dynamic stiffness (e.g., Delta Racer modes 1–2 at $88$–$90\,\text{Hz}$ both score `100.0`; Bedslinger mode 3 at $138.4\,\text{Hz}$ scores `100.0`).

---

## 2. Analytical Adjoint Static & Fox–Kapoor Modal Sensitivities (`src/printsim/fea/sensitivity.py`)

### 2.1 Governing Equations & Derivations

#### Static Adjoint Sensitivity (`sensitivity.py:415–466, 688–693`)
Given inertial static load $\mathbf{F}_a$ ($K\mathbf{u}_a = \mathbf{F}_a$, with $\partial \mathbf{F}_a / \partial \ln E_e = \mathbf{0}$) and relative nozzle-to-bed probe $u_{\text{rel},a} = \mathbf{c}_a^T \mathbf{u}_a$, differentiating $K\mathbf{u}_a = \mathbf{F}_a$ and solving the symmetric adjoint system $K\boldsymbol{\lambda}_a = \mathbf{c}_a$ (reusing the single sparse/dense factorization of $K_{ff}$) yields:
$$\frac{\partial u_{\text{rel},a}}{\partial \ln E_e} = -\boldsymbol{\lambda}_{a,e}^T \left(E_e \frac{\partial K_e}{\partial E_e}\right)\mathbf{u}_{a,e} = -\boldsymbol{\lambda}_{a,e}^T K_e \mathbf{u}_{a,e}$$
provided $K_e$ is homogeneous of degree 1 in the stiffness scale parameter (exact for isotropic beams, homogeneous MITC4 shells, and linear CBUSH springs). By Euler's homogeneous function theorem:
$$\sum_{\text{all } e} \frac{\partial u_{\text{rel},a}}{\partial \ln E_e} = -\boldsymbol{\lambda}_a^T K \mathbf{u}_a = -\mathbf{c}_a^T \mathbf{u}_a = -u_{\text{rel},a}$$

#### Fox–Kapoor (1968) Modal Frequency Sensitivities (`sensitivity.py:649–651, 700–705, 787–792, 821–823, 904–918`)
For a distinct eigenvalue $\lambda_k = 4\pi^2 f_k^2$ with $M$-orthonormal eigenvector $\boldsymbol{\phi}_k^T M \boldsymbol{\phi}_k = 1$, differentiating $K\boldsymbol{\phi}_k = \lambda_k M\boldsymbol{\phi}_k$ and premultiplying by $\boldsymbol{\phi}_k^T$ gives Fox & Kapoor (1968):
$$\frac{\partial \lambda_k}{\partial p} = \boldsymbol{\phi}_k^T\!\left(\frac{\partial K}{\partial p} - \lambda_k \frac{\partial M}{\partial p}\right)\!\boldsymbol{\phi}_k \quad\Longrightarrow\quad \frac{\partial f_k}{\partial p} = \frac{1}{8\pi^2 f_k}\,\boldsymbol{\phi}_k^T\!\left(\frac{\partial K}{\partial p} - \lambda_k \frac{\partial M}{\partial p}\right)\!\boldsymbol{\phi}_k$$
Setting $p = \ln E_e$ ($\partial K/\partial p = K_e$) and $p = \ln m_e$ ($\partial M/\partial p = M_e$):
$$\frac{\partial f_k}{\partial \ln E_e} = \frac{\boldsymbol{\phi}_{k,e}^T K_e \boldsymbol{\phi}_{k,e}}{8\pi^2 f_k}, \qquad \frac{\partial f_k}{\partial \ln m_e} = -\frac{\lambda_k\,\boldsymbol{\phi}_{k,e}^T M_e \boldsymbol{\phi}_{k,e}}{8\pi^2 f_k} = -\frac{f_k}{2}\,\boldsymbol{\phi}_{k,e}^T M_e \boldsymbol{\phi}_{k,e}$$
with exact global Euler sum identities $\sum_e \frac{\partial f_k}{\partial \ln E_e} = +\frac{f_k}{2}$ and $\sum_e \frac{\partial f_k}{\partial \ln m_e} = -\frac{f_k}{2}$. For lumped nodal masses (`sensitivity.py:908–918`), $M_i^{\text{lump}} = \operatorname{diag}(m_i, m_i, m_i, m_i r_g^2, m_i r_g^2, m_i r_g^2)$ with $r_g = 0.025\,\text{m}$, matching `beam3d.py:242–250` identically.

#### Critique of Repeated-Root Guard (`sensitivity.py:125–149`)
`_dailey_subspace_vector` collects all modes within relative tolerance `rel_tol = 1e-4` of $f_k$ into $\Phi_R \in \mathbb{R}^{n\times r}$ and projects onto the excitation vector $\mathbf{f}_{\text{exc}}$:
$$\tilde{\boldsymbol{\phi}} = \Phi_R\,\frac{\Phi_R^T\mathbf{f}_{\text{exc}}}{\|\Phi_R^T\mathbf{f}_{\text{exc}}\|_2}$$
While $\tilde{\boldsymbol{\phi}}$ is $M$-normalized ($\tilde{\boldsymbol{\phi}}^T M \tilde{\boldsymbol{\phi}} = 1$) and eliminates arbitrary eigensolver rotation inside a degenerate subspace, **this is not Dailey (1989) repeated-root sensitivity**. True repeated-root perturbation theory (Dailey 1989; Seyranian, Lund & Olhoff 1994) forms the $r\times r$ interaction matrix per design variable $p$:
$$D_p = \Phi_R^T\!\left(\frac{\partial K}{\partial p} - \lambda_0 \frac{\partial M}{\partial p}\right)\!\Phi_R$$
whose eigenvalues $\{\mu_1,\dots,\mu_r\}$ are the exact split directional derivatives. For `phi_fund` (`sensitivity.py:645`, using $\mathbf{f}_{\text{exc}} = \hat{\mathbf{f}}_x + \hat{\mathbf{f}}_y$), on a symmetric square CoreXY ($f_X = f_Y$), $\tilde{\boldsymbol{\phi}}_{\text{fund}} = \frac{1}{\sqrt{2}}(\boldsymbol{\phi}_X + \boldsymbol{\phi}_Y)$ predicts positive $\partial f_{\text{fund}}/\partial \ln E$ for an X-only member even though the true lower branch $f_{\min} = \min(f_X, f_Y)$ has $\partial f_{\min}/\partial \ln E = \min\operatorname{eig}(D_p)/(8\pi^2 f_0) = 0$.

#### Critique of Panel Bay Conflation & Thickness Scaling (`sensitivity.py:825–834, 1039–1042`)
1. **CBUSH Fastener Spring Folding:** Lines `825–834` add the stiffness sensitivities of all panel-to-frame CBUSH springs (`bay_prefix in ("left","right","rear","top")`) directly into `bay_acc[bay_prefix]["stiffness_sens_hz"]` and `du_rel_*_dlnE_m`. Consequently, `panel_bay_sensitivities` reports the combined derivative w.r.t. simultaneous scaling of shell modulus **and** fastener stiffness (which is ~3× larger than the panel sheet alone).
2. **Thickness Scaling Error in `hz_per_100g`:** Line `1039` computes `net_b = b["stiffness_sens_hz"] + b["mass_sens_hz"]` and divides by `b["mass_kg"]`. For a monolithic plate of thickness $t$, membrane and shear stiffnesses scale as $t^1$, mass scales as $t^1$, **but bending rigidity scales as $D_b = Et^3/(12(1-\nu^2)) \propto t^3$**, so $\frac{\partial K_{\text{bend}}}{\partial \ln t} = 3 K_{\text{bend}}$ while fastener stiffness does not scale linearly with $t$.

#### 4-Lens Heat Score Equations (`sensitivity.py:931–1087`)
Using axis compliance weights $w_a = f_a^{-2} / (f_x^{-2} + f_y^{-2})$, each element has:
- **Compliance lens** ($[0,1]$): $h_{\text{comp}} = \operatorname{clip}_{[0,1]}(c_{\%}/c_{\%,\max})$ (`0` for lumped masses).
- **Mass lens** ($[-1,0]$): $h_{\text{mass}} = \max(-1, \min(0, \Delta f_{m,XY}) / M_{\text{pen},\max})$, where $M_{\text{pen},\max}$ spans members, active bays, and lumped masses.
- **Modal lens** ($[-1,+1]$): For members/bays, positive $\Delta f_{\text{net},XY}$ is normalized by `max_pos_xy_net` and negative by `max_neg_xy_net` (**computed over structural elements only**, `sensitivity.py:952–954`). For lumped masses (`sensitivity.py:1080–1085`), `heat_score_modal` is instead normalized by `max_mass_pen`. This scale mismatch lets a structural member with a tiny negative net sensitivity saturate at `-1.0` alongside a much heavier lumped mass.
- **Impact lens** (`heat_score`, $[-1,+1]$): $p = 0.65 h_{\text{comp}} + 0.35(\Delta f_{K,XY}/\Delta f_{K,\max})$; $n = |h_{\text{mass}}|$ if $(\text{is\_moving} \lor \Delta f_{\text{net},XY} < 0) \land h_{\text{comp}} < 0.30$ else $0$; reports $+p/p_{\max}$ when $p \ge 0.45 n$ and $p > 0.02$, else $-n$ when $n > 0.02$, else $0$. Note `sensitivity.py:1002` flags `role == "bed"` as moving even for fixed-bed CoreXY/Trident architectures.

### 2.2 Central Finite-Difference Verification ($\epsilon = \pm 0.1\%$)

Tested via $\frac{g(p(1+\epsilon)) - g(p(1-\epsilon))}{2\epsilon}$ ($\epsilon = 10^{-3}$) on paneled `corexy_voron_style.yaml` ($n_f = 372$), `voron_trident_300.yaml` ($n_f = 396$), and `cartesian_bedslinger.yaml`:

| Element / Parameter | Metric | Analytical | Central FD ($\pm 0.1\%$) | Rel. Error | Verdict |
| :--- | :--- | :---: | :---: | :---: | :--- |
| Beam `brace_rear_1` ($E, G$, Voron 2.4) | $\partial u_{\text{rel},X}/\partial\ln E$ | $-2.325870\times 10^{-7}$ | $-2.325871\times 10^{-7}$ | **0.0000%** | VERIFIED |
| Beam `brace_rear_1` ($E, G$, Voron 2.4) | $\partial u_{\text{rel},Y}/\partial\ln E$ | $-1.274258\times 10^{-10}$ | $-1.274259\times 10^{-10}$ | **0.0000%** | VERIFIED |
| Beam `brace_rear_1` ($\rho$, Voron 2.4, `eigsh`) | $\partial f_{\text{fund}}/\partial\ln m$ | $-2.953859\times 10^{-2}$ | $-2.953860\times 10^{-2}$ | **0.0000%** | VERIFIED (vs 34.3% w/ `eigh`!) |
| Beam `col_fr_bot` ($E, G$, Trident, `eigsh`) | $\partial f_{\text{fund}}/\partial\ln E$ | $+2.427750$ | $+2.427746$ | **0.0002%** | VERIFIED |
| Beam `col_fr_bot` ($\rho$, Trident, `eigsh`) | $\partial f_{\text{fund}}/\partial\ln m$ | $-3.750304\times 10^{-2}$ | $-3.750322\times 10^{-2}$ | **0.0005%** | VERIFIED |
| MITC4 `left` bay (**shell $E$ + bay CBUSH**) | $\partial u_{\text{rel},X}/\partial\ln K$ | $-2.221623\times 10^{-7}$ | $-2.221625\times 10^{-7}$ | **0.0001%** | VERIFIED (combined) |
| MITC4 `left` bay (**shell $E$ + bay CBUSH**) | $\partial f_Y/\partial\ln K$ | $+21.26948$ | $+21.27079$ | **0.0062%** | VERIFIED (combined) |
| MITC4 `left` bay (**shell $E$ ONLY**) | $\partial u_{\text{rel},X}/\partial\ln E$ | $-2.221623\times 10^{-7}$ | $-7.567580\times 10^{-8}$ | **193.57%** | MODERATE ISSUE (CBUSH folded) |
| MITC4 `left` bay (**shell $E$ ONLY**) | $\partial f_{\text{fund}}/\partial\ln E$ | $+7.635348$ | $+2.808017$ | **171.91%** | MODERATE ISSUE (CBUSH folded) |
| MITC4 `left` bay (thickness $t$, $D_b\propto t^3$) | `net_b` vs $\partial f_{\text{fund}}/\partial\ln t$ | $+5.521631$ | $+1.306719$ | **322.56%** | MODERATE ISSUE ($t^1$ vs $t^3$ + CBUSH) |
| MITC4 `left` bay (areal mass $\rho t$) | $\partial f_Y/\partial\ln m$ | $-2.163686$ | $-2.171546$ | **0.3620%** | VERIFIED |
| Spring `bed_mount_spring` ($K$, Trident) | $\partial f_{\text{fund}}/\partial\ln K$ | $+6.886072\times 10^{-3}$ | $+6.886072\times 10^{-3}$ | **0.0000%** | VERIFIED |
| Spring `bed_mount_spring` ($K$, Bedslinger) | $\partial u_{\text{rel},Y}/\partial\ln K$ | $-8.437500\times 10^{-8}$ | $-8.437508\times 10^{-8}$ | **0.0001%** | VERIFIED |
| Lumped mass `toolhead_cog` (Voron 2.4) | $\partial f_{\text{fund}}/\partial\ln m$ | $-4.932095$ | $-4.932607$ | **0.0104%** | VERIFIED |
| Lumped mass `toolhead_cog` (Voron 2.4) | $\partial f_Y/\partial\ln m$ | $-18.10962$ | $-18.10465$ | **0.0274%** | VERIFIED |
| **Euler Sum Identity** (Voron 2.4) | $\sum \partial u_{\text{rel},X}/\partial\ln E_e$ vs $-u_{\text{rel},X}$ | $-6.191897\times 10^{-6}$ | $-6.191897\times 10^{-6}$ | **0.0000%** | VERIFIED |
| **Euler Sum Identity** (Voron 2.4) | $\sum \partial f/\partial\ln K_e$ vs $+f/2$ | $+46.414720$ | $+46.414702$ | **0.0000%** | VERIFIED |
| **Euler Sum Identity** (Voron 2.4) | $\sum \partial f/\partial\ln M_e$ vs $-f/2$ | $-46.414702$ | $-46.414702$ | **0.0000%** | VERIFIED |

---

## 3. RCSA / LM-FBS Toolhead–Hotend–Nozzle Dynamics (`src/printsim/fea/rcsa.py`)

### 3.1 Exact 4×4 Free–Free Timoshenko Beam Receptance (`rcsa.py:57–146`)

Under harmonic $e^{+j\omega t}$ convention with hysteretic complex moduli $E^* = E(1 + j\eta)$, $G^* = G(1 + j\eta)$, $\eta = 2\zeta$, the Timoshenko beam state vector $\mathbf{z}(x) = [u(x), \theta(x), M(x), V(x)]^T$ satisfies the exact first-order ODE (`rcsa.py:70–76`):
$$\frac{d}{dx}\begin{bmatrix} u \\ \theta \\ M \\ V \end{bmatrix} = \underbrace{\begin{bmatrix} 0 & 1 & 0 & (\kappa G^* A)^{-1} \\ 0 & 0 & (E^* I)^{-1} & 0 \\ 0 & -\rho I \omega^2 & 0 & -1 \\ -\rho A \omega^2 & 0 & 0 & 0 \end{bmatrix}}_{A(\omega)} \begin{bmatrix} u \\ \theta \\ M \\ V \end{bmatrix}$$
With transfer matrix $U = \exp(A(\omega)L) = \begin{bmatrix} U_{11} & U_{12} \\ U_{21} & U_{22} \end{bmatrix}$ and external end-force sign relations $\mathbf{s}(0) = [M(0), V(0)]^T = -J[F_1, M_1]^T$, $\mathbf{s}(L) = +J[F_2, M_2]^T$ where $J = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$, eliminating internal forces yields (`rcsa.py:86–97`):
$$\begin{bmatrix} U_{11} & -I_2 \\ U_{21} & 0 \end{bmatrix} H(\omega) = \begin{bmatrix} U_{12} J & 0 \\ U_{22} J & J \end{bmatrix}$$
When $\beta L = (\rho A \omega^2 / |E^*|I)^{1/4} L > 14$, `rcsa.py:105–115` subdivides the beam into $n_{\text{seg}} = \lceil \beta L / 10 \rceil$ segments and couples them recursively in receptance space via LM-FBS to prevent `expm` overflow.

**Numerical Verification:**
- Static cantilever Schur complement $c_{\text{cant}} = H_{22} - H_{2,0:2} H_{0:2,0:2}^{-1} H_{0:2,2}$ at $f=0.1\,\text{Hz}$ ($\zeta=0$, $L=0.05\,\text{m}$, $E=70\,\text{GPa}$, $I=10^{-9}\,\text{m}^4$, $G=26\,\text{GPa}$, $A=10^{-4}\,\text{m}^2$, $\kappa=5/6$): $6.183153\times 10^{-7}\,\text{m/N}$ vs analytical Timoshenko $\frac{L^3}{3EI} + \frac{L}{\kappa GA} = 6.183150\times 10^{-7}\,\text{m/N}$ (**0.00005% error**).
- Complex reciprocity $\|H - H^T\|_F / \|H\|_F = 0.00$ for both direct ($\beta L \le 14$) and subdivided ($\beta L > 14$) branches.

### 3.2 Dual LM-FBS Coupling (`rcsa.py:148–215`)

Following de Klerk, Rixen & Voormeeren (2008, *AIAA J.* 46(5):1169–1181), assembling block-diagonal uncoupled receptances $Y = \operatorname{blkdiag}(Y^{(1)},\dots,Y^{(m)})$ with signed Boolean compatibility $B\mathbf{u} = Y_{\text{joint}}\mathbf{g}$ (where $Y_{\text{joint}} = Z_{\text{joint}}^{-1}$ is interface joint compliance) yields:
$$Y_{\text{coupled}}(\omega) = Y - Y B^T \left(B Y B^T + Y_{\text{joint}}\right)^{-1} B Y$$
Symmetrization uses transpose `0.5 * (Yc + Yc.T)` (not Hermitian conjugate), which is exact for complex-symmetric mechanical receptances ($Y = Y^T$).

### 3.3 Damping Convention & CRITICAL BUG in Carriage Rotational Damping (`rcsa.py:376–382`)

All substructures consistently adopt $e^{+j\omega t}$ (positive imaginary dynamic stiffness $+j\omega C$ or $+j\eta K$):
- Beams: $E^* = E(1 + j 2\zeta)$ (`rcsa.py:141–143`).
- Bolted joints J1, J2: $Y_{\text{joint},ii} = (k_i + j\omega c_i)^{-1}$ (`rcsa.py:410–413`).
- Carriage translation: $Y_{\text{car},00} = (k_{\text{modal}} - \omega^2 m_{\text{bare}} + j 2\zeta_f \omega_0 \omega m_{\text{modal}})^{-1}$ (`rcsa.py:377`, exact SI units $\text{N/m}$).

**CRITICAL BUG (`rcsa.py:378–382`):** Carriage rotational receptance is written as:
$$Y_{\text{car},11}^{-1} = k_{\text{rot,car}} - \omega^2 J_{\text{car}} + j \cdot 2\zeta_f \sqrt{k_{\text{rot,car}} J_{\text{car}}} \cdot \frac{\omega}{\max(\omega_0, 1.0)}$$
**Dimensional Proof of Bug:**
$$\left[k_{\text{rot,car}}\right] = \frac{\text{N}\cdot\text{m}}{\text{rad}}, \qquad \left[\sqrt{k_{\text{rot,car}} J_{\text{car}}}\right] = \sqrt{\frac{\text{N}\cdot\text{m}}{\text{rad}}\cdot \text{kg}\cdot\text{m}^2} = \text{N}\cdot\text{m}\cdot\text{s}$$
The viscous damping coefficient $c_{\text{rot}} = 2\zeta_f\sqrt{k_{\text{rot,car}} J_{\text{car}}}$ MUST be multiplied by $\omega$ ($\text{rad/s}$) to produce $\text{N}\cdot\text{m}$, **not** dimensionless $\omega/\omega_0$. Dividing by $\omega_0$ breaks dimensional consistency and attenuates carriage rotational damping by $1/\omega_0$ (on Voron 2.4, $1/583.3 = 0.00171$, collapsing effective rotational damping from $\zeta_f = 0.046$ to $7.89\times 10^{-5}$!).

### 3.4 Whip Ratio & Quasi-Static Cantilever Stiffness (`rcsa.py:426–469`)
- Sensor FRF $|Y_{\text{coupled}, 0,0}(\omega)|$ and nozzle-tip transfer FRF $|Y_{\text{coupled}, 12,0}(\omega)|$ both excite carriage translation (DOF 0, belt drive force). `whip_ratio = max(1.0, max_f |H_{12,0}| / max_f |H_{0,0}|)` compares independent global peak magnitudes and flags `whip_risk` above `1.35`.
- Quasi-static hotend bending stiffness (`rcsa.py:439–468`) clamps the mount root via exact Schur complement at $f = 0.5\,\text{Hz}$ and adds series J1 translational and moment compliance $1/k_{u1} + L_{\text{tot}}^2 / k_{t1}$, converting correctly via $k_{\text{N/mm}} = 10^{-3} / c_{\text{m/N}}$.

---

## 4. Klipper Input-Shaper Physics & Calibration (`src/printsim/fea/klipper_shaper.py`)

### 4.1 Governing Equations vs. Singer & Seering (1990) & Klipper `shaper_defs.py`

Let $d_f = \sqrt{1-\zeta^2}$, $T_d = (f_s d_f)^{-1}$, $K = \exp(-\zeta\pi/d_f)$, $K_{\text{mzv}} = \exp(-0.75\zeta\pi/d_f)$, and $v = v_{\text{tol}} = 0.05$:

| Shaper | Impulses $A_i$ (`klipper_shaper.py:105–162`) | Timings $T_i$ | Matches Klipper / Literature? |
| :--- | :--- | :--- | :--- |
| `ZV` | $[1,\; K]$ | $[0,\; 0.5 T_d]$ | Exact `shaper_defs.py` (unnormalized) |
| `MZV` | $[1-\tfrac{1}{\sqrt{2}},\; (\sqrt{2}-1)K_{\text{mzv}},\; (1-\tfrac{1}{\sqrt{2}})K_{\text{mzv}}^2]$ | $[0,\; 0.375 T_d,\; 0.75 T_d]$ | Exact Klipper Butyugin MZV (unnormalized) |
| `EI` | $[\tfrac{1+v}{4},\; \tfrac{1-v}{2}K,\; \tfrac{1+v}{4}K^2]$ | $[0,\; 0.5 T_d,\; T_d]$ | Exact `shaper_defs.py` (unnormalized) |
| `2HUMP_EI` | $x=(v^2(\sqrt{1-v^2}+1))^{1/3}$, $a_1=\frac{3x^2+2x+3v^2}{16x}$, $[a_1, (\tfrac{1}{2}-a_1)K, (\tfrac{1}{2}-a_1)K^2, a_1 K^3]$ | $[0,\; 0.5 T_d,\; T_d,\; 1.5 T_d]$ | Exact `shaper_defs.py` (unnormalized) |
| `3HUMP_EI` | $a_1=\tfrac{1}{16}(1+3v+2\sqrt{2(v+1)v})$, $a_2=\tfrac{1-v}{4}K$, $a_3=(\tfrac{1+v}{2}-2a_1)K^2$, $[a_1, a_2, a_3, a_2 K^2, a_1 K^4]$ | $[0,\; 0.5 T_d,\; T_d,\; 1.5 T_d,\; 2 T_d]$ | Exact `shaper_defs.py` (unnormalized) |
| `ZVDD` | $[1,\; 3K,\; 3K^2,\; K^3]/(1+K)^3$ | $[0,\; 0.5 T_d,\; T_d,\; 1.5 T_d]$ | Exact Singhose (1994) (normalized) |
| `SMOOTH_ZV` / `SMOOTH_EI` | Discrete convolution of `ZV`/`EI` with normalized $\sin^2(\pi u)$ Hann kernel ($\tau = 0.75 T_d$, $32\text{ steps}/T_d$) | Uniform grid $\Delta t = T_d / 32$ | Approximation (longer than Kalico compact-support smoothers) |

#### Residual Vibration $V(f)$, Smoothing $S$, and Max Accel (`klipper_shaper.py:295–370`)
With $D = \sum_i A_i$, $\omega = 2\pi f$, $\omega_d = \omega\sqrt{1-\zeta^2}$, $t_{\text{last}} = T_n$, and shaper centroid $t_s = \frac{1}{D}\sum_i A_i T_i$:
$$V(f) = \frac{1}{D}\sqrt{\left(\sum_i A_i e^{-\zeta\omega(t_{\text{last}}-T_i)}\sin(\omega_d T_i)\right)^2 + \left(\sum_i A_i e^{-\zeta\omega(t_{\text{last}}-T_i)}\cos(\omega_d T_i)\right)^2}$$
$$S_{90} = \frac{\sqrt{2}}{D}\sum_{T_i \ge t_s} A_i\!\left(v_{\text{scv}} + \tfrac{1}{2}a(T_i - t_s)\right)(T_i - t_s), \qquad S_{180} = \frac{a}{2D}\sum_{\text{all } i} A_i(T_i - t_s)^2, \qquad S = \max(S_{90}, S_{180})$$
- Both $S_{90}$ and $S_{180}$ (summing over **all** $i$ for $180^\circ$ reversal, as physically required by convolving $x(t) = -\frac{1}{2}at^2$ at $t=t_s$) and bisection `find_shaper_max_accel` targeting $S \le 0.12\,\text{mm}$ (rounded to nearest $100\,\text{mm/s}^2$) match Klipper `shaper_calibrate.py` exactly.
- Synthetic PSD uses normalized velocity-mobility $|H_v(j\omega)|^2 = \frac{4\zeta^2 r^2}{(1-r^2)^2 + (2\zeta r)^2}$ (`klipper_shaper.py:373–378`), and `estimate_psd_damping_ratio` (`klipper_shaper.py:417–483`) recovers $\zeta = 0.05541$ vs true $0.055$ at off-grid $63.3\,\text{Hz}$ (**0.75% error**).

### 4.2 Numerical Shaper Verification Table ($f_n = 50\,\text{Hz}$, `scv = 5 mm/s`)

| Shaper | $\sum A_i$ ($\zeta=0.1$) | $\sum A_i$ ($\zeta=0$) | $T_{\text{last}}/T_d$ | $V(f_n,\zeta=0.1)$ | $V(f_n,\zeta=0)$ | $S(5000)$ (mm) | $a_{\max}$ ($\text{mm/s}^2$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `NONE` | 1.00000 | 1.00000 | 0.0000 | 1.000000 | 1.000000 | 0.00000 | 0 |
| `ZV` | **1.72925** | **2.00000** | 0.5000 | **0.000000** | **0.000000** | 0.06770 | 9700 |
| `MZV` | **0.80217** | 1.00000 | 0.7500 | **0.000000** | **0.000000** | 0.08147 | 7400 |
| `EI` | **0.74849** | 1.00000 | 1.0000 | 0.035525 | **0.050000** | 0.12885 | 4700 |
| `2HUMP_EI` | **0.65078** | 1.00000 | 1.5000 | **0.000000** | **0.000000** | 0.21585 | 2800 |
| `3HUMP_EI` | **0.56913** | 1.00000 | 2.0000 | 0.024846 | **0.050000** | 0.32781 | 1700 |
| `ZVDD` | 1.00000 | 1.00000 | 1.5000 | **0.000000** | **0.000000** | 0.18475 | 3200 |
| `SMOOTH_ZV` | 1.00000 | 1.00000 | 1.1875 | **0.000000** | **0.000000** | 0.08015 | 7500 |
| `SMOOTH_EI` | 1.00000 | 1.00000 | 1.6875 | 0.019705 | 0.034298 | 0.14741 | 4100 |

---

## 5. Structured Audit Findings Table

| Severity | File : Line | Finding & Quantitative Evidence | Concrete Recommended Fix |
| :--- | :--- | :--- | :--- |
| **CRITICAL BUG** | `src/printsim/fea/rcsa.py:378-382` | **Dimensional & $1/\omega_0$ error in carriage rotational damping:** `Y_car[:, 1, 1]` adds `1j * 2.0 * z_f * sqrt(k_rot_car * j_car) * (w / max(w0, 1.0))`. Since $c_{\text{rot}} = 2\zeta_f\sqrt{k_{\text{rot}} J_{\text{car}}}$ has units $\text{N}\cdot\text{m}\cdot\text{s}$, multiplying by dimensionless $\omega/\omega_0$ instead of $\omega$ violates dimensional homogeneity and reduces rotational damping by $1/\omega_0 \approx 583\times$ ($\zeta_{\text{rot,eff}} = 7.89\times 10^{-5}$ vs $0.046$ on Voron 2.4). | Replace `* (w / max(w0, 1.0))` with `* w` (viscous damping $+j\omega c_{\text{rot}}$), or use hysteretic damping `+ 1j * 2.0 * z_f * k_rot_car`. |
| **MODERATE ISSUE** | `src/printsim/fea/sensitivity.py:92, 114`<br>`src/printsim/fea/modal.py:438, 461` | **Ill-conditioned dense `eigh(K_ff, M_ff)` Cholesky on $M_{ff}$:** `m_floor = 1e-12` produces $\kappa(M_{ff}) = 7.14\times 10^{11}$ vs $\kappa(K_{ff}) = 7.50\times 10^6$. Because `sensitivity.py` only uses shift-invert `eigsh` for $n_f > 500$, all paneled presets ($n_f \approx 372\text{--}396$) and all open-frame presets ($n_f < 250$) run `eigh(K_ff, M_ff)`, introducing $10^{-4}\text{--}10^{-3}\,\text{Hz}$ jitter (e.g., 34.3% FD error on `brace_rear_1` mass sensitivity with `eigh` vs **0.0000%** with shift-invert `eigsh`, and non-monotonic negative $\Delta f_Y$ when stiffening `bed_mount_spring` on Bedslinger). | For dense solves, call inverted `mu, V = eigh(M_ff, K_ff)` (Cholesky on well-conditioned $K_{ff}$) and take largest `k_scan` eigenvalues $\lambda = 1/\mu$; or lower the `eigsh` threshold in `sensitivity.py` to match `modal.py` and still invert the dense fallback. |
| **MODERATE ISSUE** | `src/printsim/fea/sensitivity.py:825-834, 1039-1042` | **Panel bay sensitivity folds CBUSH springs into shell $E$ & uses $t^1$ instead of $t^3$ for bending ROI:** Adding bay CBUSH springs into `bay_acc` inflates reported panel `du_rel_x_dlnE_m` by **+193.6%** and `stiffness_sens_hz` by **+171.9%** relative to shell-only $E$-FD. Computing `net_b = stiff + mass` for `hz_per_100g` ignores cubic bending $D_b \propto t^3$ and scales springs with thickness (**+322.6%** error vs true thickness FD `+1.307 Hz` vs `+5.522 Hz`). | Separate bay fastener spring sensitivities into a distinct `fastener_sens_hz` (or spring member entries) rather than adding into `stiffness_sens_hz`; compute panel thickness derivative explicitly as $\frac{\partial f}{\partial \ln t} = \frac{\boldsymbol{\phi}^T(K_{\text{mem}} + K_{\text{shear}} + 3K_{\text{bend}})\boldsymbol{\phi}}{8\pi^2 f} + \frac{\partial f}{\partial \ln m}$ (for monolithic shells). |
| **MODERATE ISSUE** | `src/printsim/fea/sensitivity.py:125-149, 645` | **Repeated-root guard is excitation projection, not Dailey (1989) eigenvalue splitting:** `_dailey_subspace_vector` projects $\Phi_R$ onto a single fixed vector (`fx_exc + fy_exc` for `phi_fund`) instead of solving the $r\times r$ reduced eigenproblem $D_p = \Phi_R^T(\partial K/\partial p - \lambda_0 \partial M/\partial p)\Phi_R$ per parameter $p$. On symmetric square frames ($f_X = f_Y$), this overpredicts $\partial f_{\text{fund}}/\partial\ln E$ on X-only members where the true lower branch derivative is $\min\operatorname{eig}(D_p)/(8\pi^2 f_0) = 0$. | When `len(cluster) > 1`, compute the $r\times r$ matrix $D_p = \Phi_R^T K_e \Phi_R$ (or $-\lambda_0 \Phi_R^T M_e \Phi_R$) per element and use $\min\operatorname{eig}(D_p)$ for `f_fund` directional sensitivity (or document explicitly as excitation-aligned subspace projection). |
| **MODERATE ISSUE** | `src/printsim/fea/modal.py:485-488`<br>`src/printsim/fea/sensitivity.py:487-490` | **Bedslinger `fy_exc` applies unphysical $2\,\text{N}$ dipole:** Setting `fy_exc[cog_y] = +1.0` and `fy_exc[bed_y] = -1.0` applies $+1\,\text{N}$ to the stationary toolhead and $-1\,\text{N}$ to the bed ($\|\hat{\mathbf{f}}_y\|_1 = 2$), exciting portal fore-aft modes that a Y-bed acceleration does not directly load and doubling Y compliance scale relative to X. | For `cartesian_bedslinger`, set only `fy_exc[6 * bed_idx + 1] = -1.0` (unit bed inertial excitation; remove the $+1.0$ on `cog_idx` for Y). |
| **MODERATE ISSUE** | `src/printsim/fea/modal.py:576-577, 614-618, 781-795` | **Subsystem KE shares vulnerable to consistent-mass cross-term negativity & mismatch with `ke_panel`:** $\boldsymbol{\phi}_{S_\alpha}^T(M_{ff}\boldsymbol{\phi})_{S_\alpha}$ can be negative when consistent beam off-diagonals bridge two roles out of phase, causing silent clipping to `0.0`; `_classify_mode_diagnosis` uses un-renormalized shares while `subsystem_share_pct` renormalizes them. | Accumulate element-level kinetic energies $T_e = \boldsymbol{\phi}_e^T M_e \boldsymbol{\phi}_e \ge 0$ by member/shell role plus $T_i^{\text{lump}} = \boldsymbol{\phi}_i^T M_i^{\text{lump}}\boldsymbol{\phi}_i \ge 0$ by node role; pass normalized shares to `_classify_mode_diagnosis`. |
| **MODERATE ISSUE** | `src/printsim/fea/modal.py:900-924` | **Relative-only `print_impact_score` normalization:** Dividing `c_k / c_max` over only the top `k_ret` modes guarantees `score = 100.0` ("critical") for any printer with a mode $\le 140\,\text{Hz}$ and $A_{\text{noz}} \ge 0.30$ regardless of absolute compliance (confirmed on Delta Racer and Bedslinger). | Anchor `c_norm` against a physical absolute compliance reference (e.g., $c_{\text{ref}} = \frac{10\,\mu\text{m}}{10\,\text{N}} / (60\,\text{Hz})^2$ or target dynamic compliance at rated acceleration) rather than purely `c_max` of 6 modes. |
| **MODERATE ISSUE** | `src/printsim/fea/sensitivity.py:952-954, 978, 1080-1085` | **Mismatched negative normalization on `"modal"` heat-score lens:** Structural members/bays normalize negative `heat_score_modal` by `max_neg_xy_net` (excluding lumped masses), whereas lumped masses normalize by `max_mass_pen` (including lumped masses), skewing visual comparison on the modal lens. | Compute `max_neg_modal = max(max_neg_xy_net, max_mass_pen)` and use that single shared denominator for negative `heat_score_modal` across members, bays, and lumped masses. |
| **MODERATE ISSUE** | `src/printsim/fea/klipper_shaper.py:105-158` | **Inconsistent impulse normalization in `get_shaper_pulses`:** `ZV` ($\sum A=1.729$), `MZV` ($0.802$), `EI` ($0.748$), `2HUMP_EI` ($0.651$), and `3HUMP_EI` ($0.569$) return unnormalized amplitudes at $\zeta=0.1$, whereas `NONE`, `ZVDD`, `SMOOTH_ZV`, and `SMOOTH_EI` return $\sum A = 1.0$. Internal callers divide by `sum(A)`, but external callers assuming unit DC gain get wrong scaling. | Divide `amps /= float(np.sum(amps))` before returning in `get_shaper_pulses` (this leaves `estimate_shaper`, `get_shaper_smoothing`, and `apply_shaper_to_trajectory` bit-for-bit unchanged). |
| **MODERATE ISSUE** | `src/printsim/fea/klipper_shaper.py:508, 589-591` | **Single damping evaluation & narrow frequency window vs Klipper `shaper_calibrate.py`:** `_fit_single_shaper` tests only `test_damping_ratio = 0.10` instead of Klipper's worst-case envelope $\max_{\zeta \in \{0.075,0.10,0.15\}} V(f;\zeta)$ (over-favoring `ZV`), and restricts candidate $f_s$ to $[0.7 f_{\text{pk}}, 1.35 f_{\text{pk}}]$ (blocking `EI`/`2HUMP_EI` from straddling widely separated bimodal peaks). | Take `np.maximum.reduce([estimate_shaper(shaper, freqs_hz, z) for z in (0.075, 0.10, 0.15)])` and expand `[lo_f, hi_f]` to cover all modes above `vibr_threshold`. |
| **MINOR / LIMITATION** | `src/printsim/fea/modal.py:699-702, 990-994` | **Modal truncation residual in Rayleigh ratio $\rho_a$ & $1/f^6$ residue weighting:** Truncating $\sum C_k$ at `k_scan` introduces $0.5\text{--}0.7\%$ error vs exact static compliance (potentially larger if local nozzle modes exceed mode `k_scan`); `w_k = R_k/f_k^2` applies an extra $1/f_k^2$ on top of $R_k \propto 1/f_k^2$. | Optionally compute exact static `sum_comp = c^T K^{-1} f_exc` via the already-built `splu(K_ff)` factorization to eliminate modal truncation error in $\rho_a$. |
| **MINOR / LIMITATION** | `src/printsim/fea/sensitivity.py:691-693, 775-777, 815-817` | **Sign assumption in `comp_raw`:** Uses `max(0.0, -du / abs(u_rel))` instead of `max(0.0, -du / u_rel)`, which would invert sign if a configuration ever produced $u_{\text{rel}} < 0$. | Replace `-du / max(abs(u_rel), 1e-12)` with `-(math.copysign(1.0, u_rel) * du) / max(abs(u_rel), 1e-12)`. |
| **MINOR / LIMITATION** | `src/printsim/fea/sensitivity.py:179-193, 860, 1002` | **Virtual massless links vs springs & fixed-bed moving flag:** Virtual massless beams (`toolhead_mount`, `nozzle_stem`) divide pure stiffness sensitivity by arbitrary `m_ref` to report nonzero `hz_per_100g`, whereas massless springs report `0.0`; `role == "bed"` is tagged `is_moving_mem = True` even for fixed-bed CoreXY/Trident; anisotropic beams with independent `shear_modulus_pa` compute $\partial/\partial\ln(E,G)$ rather than $\partial/\partial\ln E$. | Gate `is_moving_mem` for `role == "bed"` on `spec.architecture == "cartesian_bedslinger"`, and document `m_ref` / uniform stiffness scaling conventions. |
| **MINOR / LIMITATION** | `src/printsim/fea/rcsa.py:290-326, 354, 436` | **RCSA geometric & single-plane simplifications:** Evaluates only a single dominant axis plane (`dom_mode.dominant_axis`), uses only Z drop (ignoring Y overhang `cog_offset_mm[1]`, `nozzle_offset_mm[1]`), uses hardcoded cross-sections (`I_mount=4.5e-9`, `d_blk=0.011`) and empirical J1/J2 joint stiffness laws, and clamps `whip_ratio = max(1.0, peak_noz / peak_sen)` across potentially different peak frequencies. | Extend `analyze_toolhead_rcsa` to evaluate X and Y planes independently using effective lever arm $\sqrt{\Delta y^2 + \Delta z^2}$ and report both peak-ratio and same-frequency transfer ratio $|H_{12,0}(f_{\text{sen}})/H_{0,0}(f_{\text{sen}})|$. |
| **MINOR / LIMITATION** | `src/printsim/fea/klipper_shaper.py:168-198` | **`SMOOTH_ZV`/`SMOOTH_EI` duration vs Kalico smooth shapers:** Hann-window convolution ($\tau = 0.75 T_d$) yields total durations $1.1875 T_d$ (`SMOOTH_ZV`) and $1.6875 T_d$ (`SMOOTH_EI`), longer than Kalico's compact-support smoothers ($T_s \approx 1.0 T_d$). Also `estimate_shaper` line 310 assumes `times_arr` is sorted (`times_arr[-1]`). | Use `float(np.max(times_arr))` in `estimate_shaper` and document `_build_smooth_shaper` as a Hann-convolved approximation (or port Kalico's exact `smoother_defs.py` polynomials). |
| **VERIFIED** | `src/printsim/fea/modal.py:661-703`<br>`src/printsim/fea/sensitivity.py:649-929` | **Exact Static Adjoint, Fox–Kapoor, & Modal Receptance:** Modal compliance units exact ($\text{m/N}$); static adjoint & Fox–Kapoor derivatives match $\pm 0.1\%$ central FD to **0.0000%–0.03%** across beams, combined shell bays, `bed_mount_spring`, and lumped masses; all 3 Euler sum identities hold to **0.0000%**. | — |
| **VERIFIED** | `src/printsim/fea/rcsa.py:57-215, 439-468`<br>`src/printsim/fea/klipper_shaper.py:87-484` | **Exact Timoshenko Receptance, LM-FBS, & Klipper Math:** Timoshenko static cantilever limit matches analytical $\frac{L^3}{3EI} + \frac{L}{\kappa GA}$ to **0.00005%** with $0.00$ reciprocity error; LM-FBS exact; all Klipper shaper formulas, $V(f_n)=0$ (`ZV, MZV, 2HUMP_EI, ZVDD, SMOOTH_ZV`), $S_{90}/S_{180}$, `find_shaper_max_accel`, and Lorentzian PSD damping estimator (**0.75%** error) verified. | — |
