# 03 — Kinematics, Belt/Stepper Drives, Component Thermal-Structural Models, Coupled LPV Mechatronics & G-code/Ballbar Simulator

**Audit scope (read-only):**
`src/printsim/kinematics/solvers.py`, `src/printsim/kinematics/collision.py`, `src/printsim/fea/components.py`, `src/printsim/fea/mechatronics.py`, `src/printsim/gcode/ballbar.py`, `src/printsim/gcode/simulator.py`, `src/printsim/gcode/parser.py`. Also the helpers they call: `StepperSpec` / `BeltSpec` / `BED_MOUNT_CATALOG` in `schema/printer.py` and `MaterialSpec.modulus_retention_at_temp` in `schema/materials.py`.

**Verification script:** `brain/…/scratch/verify_mech.py`, run with `uv run python`. Its key outputs are quoted in §6.

---

## 1. Kinematics (`kinematics/solvers.py`)

### 1.1 Forward/inverse kinematics

| Architecture | IK | FK | Status |
|---|---|---|---|
| CoreXY | $A = x + y,\; B = x - y,\; C = z$ | $x = \tfrac12(A+B),\; y = \tfrac12(A-B)$ | Exact (round-trip error 5.7e-14 mm) |
| Cartesian | identity | identity | Exact |
| Delta (towers at 90°/210°/330°, radius $R$, arm $L$) | $z_i = z + \sqrt{L^2 - (x - t_{x,i})^2 - (y - t_{y,i})^2}$ | trilateration of 3 spheres of radius $L$, lower root | Exact (round-trip error 1.7e-13 mm) |

The delta FK uses the standard three-sphere trilateration. All three radii are equal to $L$, so $x' = d/2$ and $y' = \dfrac{i^2 + j^2}{2j} - \dfrac{i}{j}\,x'$, with $z' = \pm\sqrt{L^2 - x'^2 - y'^2}$. The code takes the lower root because the effector hangs below the carriages.

### 1.2 CoreXY force & velocity scaling
Virtual work with the belt coordinates $A, B$ gives $F_x = F_A + F_B$ and $F_y = F_A - F_B$, so

$$F_A = \tfrac12(F_x + F_y),\qquad F_B = \tfrac12(F_x - F_y),\qquad \dot A = \dot x + \dot y .$$

* **Pure X or pure Y move:** each motor carries $0.5\,F$ at speed $v$.
* **45° diagonal move:** one motor carries $F/\sqrt2 = 0.707\,F$ at speed $\sqrt2\,v$, and the other is idle.

The code correctly uses the $\sqrt2$ factor for motor RPM (`solvers.py:332`). For torque, however, it uses `share = 0.5` (`solvers.py:351`), which is the axis-aligned case. That mixes the worst-case speed with the best-case force (see finding M1). AWD shares the load over 4 motors (`share = 0.25`), which is correct.

### 1.3 Stepper electromechanics (`schema/printer.py:595-648`)

* **Rotor teeth / "pole pairs":** $N_r = \text{steps\_per\_rev}/4$. That gives 50 for a 1.8° motor and 100 for a 0.9° motor. This is correct for 2-phase hybrid steppers (Acarnley 2002, §2.2).
* **Static torque–angle curve:** $T = -T_h \sin(N_r\theta)$, so the electromagnetic stiffness is
  $$k_\theta = \frac{\partial T}{\partial \theta}\Big|_0 = N_r\,T_h(I)\,\eta_V(\omega).$$
  Units: dimensionless × N·m/rad = N·m/rad. Verified: at equal $T_h$, the 0.9° motor gives exactly 2× the static stiffness (25 → 50 N·m/rad). At 600 rpm the ratio falls to only 1.43× because the electrical frequency $\omega_e = N_r\,\omega_m$ is doubled. That is the physically correct trend.
* **Linear stiffness at the belt:** $k_{em} = k_\theta / r_p^2$ (N/m), where $r_p = z\,p / 2\pi$.
* **Voltage roll-off:**
  $$\eta_V = \left[1 + \left(\frac{IR + k_e\omega_m}{V}\right)^2 + \left(\frac{\omega_m N_r L I}{V - IR}\right)^2\right]^{-1/2}.$$
  This is a heuristic phasor-magnitude model, and every term is in volts.
* **Copper loss:** $P = 2 I_{rms}^2 R$. This is correct for sine/cosine microstepping.
* **Torque demand:** $T_{req} = s\,F\,r_p + J\,a/r_p$, compared against the available torque $0.72\,T_h\,\eta_V(0.7\,\text{rpm}_{max})$.

### 1.4 Belt drive train (series springs)

$$\frac1{k_{drive}} = \frac1{k_{cord}} + \frac1{k_{mesh}} + \frac1{k_{idler}} + \frac1{n\,k_\theta/r_p^2} + \frac1{k_{bracket}}$$

* **Cord with crimp and Ernst (1965) catenary sag:**
  $$\frac1{EA_{eff}} = \frac1{EA\,(1 - 0.32\,e^{-T_0/15})} + \frac{(\mu g L)^2}{12\,T_0^3}.$$
  Units check: N⁻¹ = (N/m · m)² / N³. ✓
* **Gerbert (1972) tooth-mesh ladder:** the chain recursion gives $\cosh\lambda = 1 + \dfrac{k_t w\,p}{2\,EA}$ and
  $$K = k_t w\,\frac{\tanh(\lambda z_e)}{\tanh\lambda}.$$
  This equals the continuum shear-lag limit $\sqrt{k' EA}\,\tanh(\beta L)$ for $\lambda \ll 1$ (here $\lambda \approx 0.08$). ✓
* **Idler shaft:** $48EI/L^3$ for double shear and $3EI/L^3$ for single shear, in series with a polymer root term. The hub load is $F = 2T_0\sin(\theta/2)$.
* **Bearings:** Harris/Palmgren $\delta_r = 4.4\times10^{-4}\,Q_{max}^{2/3}/D^{1/3}$ mm, with $Q_{max} = 5F_r/Z$ (Stribeck) and tangent stiffness $k = 1.5\,F/\delta$. ✓
* **Slip check:** slack-side tension $T_0 - F/2$ is compared with a threshold. Algebraically this is a ratio test, $T_t/T_s > e^{\mu\theta}(1 + 0.45 z_e) - 1$. In other words, Euler–Eytelwein with an empirical tooth-engagement multiplier.
* **VFA band:** $v_{VFA} = p \cdot f_{drive}$, i.e. the speed at which the tooth-passing frequency $v/p$ equals the drive resonance $\sqrt{k_{drive}/m}/2\pi$. Units: mm × Hz = mm/s. ✓ The band is ±15%.

---

## 2. Component thermal-structural models (`fea/components.py`)

| Model | Governing equation | Units check | Status |
|---|---|---|---|
| Hertz ball–raceway (Harris & Kotzalas 2006) | $\delta = \left(\dfrac{9Q^2}{16 R_e E^{*2}}\right)^{1/3} F_2$, so $k_n = \dfrac{3}{2}\dfrac{Q^{1/3}}{c}$, with $E^* = \dfrac{E}{2(1-\nu^2)}$ and $R_e = \sqrt{R_xR_y} = \sqrt{0.5d \cdot 10.5d} = d\sqrt{5.25}$ for conformity $f = 0.525$ on a straight raceway | m = (N²/(m·Pa²))^{1/3} ✓ | ✓ (the $R_e$ derivation checks out exactly) |
| Carriage moments | $k_{rx} = k\,r_y^2/1.5$, and $k_{ry} = k\,L_1^2/12$ for uniform spring density along $L_1$ | N·m/rad ✓ | ✓ |
| Newmark (1951) / Eurocode 5 γ-method | $\gamma = \left[1 + \dfrac{\pi^2\,EA^*\,s}{L^2 k_s}\right]^{-1}$, with $EA^* = (1/EA_1 + 1/EA_2)^{-1}$ | dimensionless ✓ | ✓ (two-member Girhammar variant) |
| Timoshenko (1925) bimetal, generalized | $\kappa = \dfrac{\Delta\alpha\,\Delta T\,c}{c^2 + (\sum EI)(1/E_1A_1 + 1/E_2A_2)}$, and bow $= \gamma\,\kappa L^2/8$ | 1/m ✓ | ✓ |
| Rail + backer, 3 layers | 2×2 compatibility system for $(P_r, P_k)$, then $\kappa = (P_r c_r - P_k c_k)/\sum EI$ (`components.py:447-457`) | ✓ | ✓ (the matrix is symmetric and the signs are consistent with a backer on the opposite face) |
| Bed through-thickness gradient | $\Delta T = q t/k$, $\kappa = \alpha\Delta T/t$, so $w = \dfrac{L^2 \alpha q}{8k}$ with $q = h_{eff}(T_{bed} - T_{ch})$ | m ✓ (thickness cancels) | ✓ |
| Taco elastica | $\Delta = \dfrac{\pi^2 w^2}{4L}$ (half-sine), so $w = \dfrac{2}{\pi}\sqrt{L\Delta}$ | m ✓ | Equation ✓; the restraint coupling has a units problem (M4) |
| Z drift | $\delta_z = (\alpha_{frame} - \alpha_{drive})\,H\,\Delta T$ | m ✓ | ✓ |
| Mahieux–Reifsnider $E(T)$ (`materials.py:51-73`) | $E/E_{25} = \dfrac{(1 - e_r)\exp[-(T_K/T_{ref})^m] + e_r}{(\cdot)_{25°C}}$ | dimensionless ✓ | ✓ (Kelvin ratio, normalized to 1 at 25 °C) |
| "Iwan" joint damping | lookup table of $\Delta\zeta$ per joint type | — | Limitation: no Iwan element is implemented |

---

## 3. Coupled LPV electromechanical model (`fea/mechatronics.py`)

**Coordinates:** $q = [q_{fx}, q_{fy}, q_{m,1..N}, q_{cx}, q_{cy}]$ (all in m).

**Equations of motion:**
$$M\ddot q + C\dot q + Kq = F_{cmd}\,u,\qquad K = K_f + K_{em} + E^T\operatorname{diag}(k_b)\,E,\qquad E = [-T,\; -I,\; +T].$$

**Belt mapping $T$ (motor-space → Cartesian):**
* CoreXY 2WD: $T = \begin{bmatrix}1 & 1\\ 1 & -1\end{bmatrix}$.
* AWD: each row duplicated.
* Delta: the Jacobian $\partial z_i/\partial(x,y) = -(x - t_{x,i})/\sqrt{L^2 - d_i^2}$.

**Cartesian belt stiffness:**
$$K_{cart} = \begin{bmatrix}k_A + k_B & k_A - k_B\\ k_A - k_B & k_A + k_B\end{bmatrix}.$$
The off-diagonal term $k_A - k_B$ is the physical source of the CoreXY X–Y cross-coupling when the carriage is off-centre. ✓

**State space:**
$$A = \begin{bmatrix}0 & I\\ -M^{-1}K & -M^{-1}C\end{bmatrix}\ [\text{s}^{-2}, \text{s}^{-1}],\qquad B = \begin{bmatrix}0\\ M^{-1}k_{em}T\end{bmatrix}\ [\text{s}^{-2}],\qquad C = \text{carriage position selector},\qquad D = 0.$$

* Units: input $u$ (m) → output (m).
* The DC gain is $-CA^{-1}B = I$ at every grid point and for every architecture (verified, §6). ✓
* Every $\operatorname{Re}\lambda < 0$ for the open-loop stepper and for FOC+notch at the centre and at the corners. ✓

**Open-loop stepper electrical damping (Acarnley 2002, Leenhouts 1991):**
$$b_{rot} = 2(0.055)\sqrt{k_\theta J} + 0.4\,\frac{k_e^2}{R}\,\frac{1 - x^2}{(1 + x^2)^2}\,(1 - \eta_V^2) - 0.14\,N_r T_h\,\tau_e\,\frac{x}{1 + x^2}\,(1 - \eta_V^2),\qquad x = N_r\,\omega_m\,\tau_e.$$
The terms change sign and peak at $x \approx 1$, which reproduces the mid-band negative damping qualitatively. The coefficients 0.4, 0.14 and 0.055 are fitted rather than derived. Units: (N·m/A)²/Ω = N·m·s/rad ✓.

**FOC servo:** modelled as an ideal P–P cascade impedance, $b = m\,\omega_v$ and $k = m\,\omega_v\,\omega_p$. The closed loop then has $\omega_n = \sqrt{\omega_v\omega_p}$ and $\zeta = \tfrac12\sqrt{\omega_v/\omega_p}$.

**Biquad notch:**
$$H(s) = \frac{s^2 + 2\zeta_z\omega_n s + \omega_n^2}{s^2 + 2\zeta_p\omega_n s + \omega_n^2}.$$
The realization $(a, b, c, d)$ at `mechatronics.py:489-492` reproduces $H(s)$ exactly, and the input-cascade formulas (`500-503`) are correct. ✓

**Workspace FRF map:** 5×5 grid over $(u, v) \in [0.05, 0.95]^2$.
* Belt stiffness varies with position in an exact geometric way. The loop lengths are constant ($l_1 + l_2 = 2(w + d)$). ✓
* The structural frequency, by contrast, is modulated by empirical factors (0.22 / 0.14 / 0.08).

---

## 4. Time-domain metrology (`gcode/ballbar.py`)

* **ZOH discretization (Van Loan):** $\exp\!\left(\begin{bmatrix}A & B & B_d\\ 0 & 0 & 0\\ 0 & 0 & 0\end{bmatrix}dt\right) = \begin{bmatrix}A_d & B_d & B_{d,d}\\ 0 & I & 0\\ 0 & 0 & I\end{bmatrix}$. This is exact for piecewise-constant inputs, here with $dt = 0.5$ ms. ✓
* **Deviation-coordinate simulation** (`_run_zoh_lti`): starting from $x = 0$ and adding back $u_0$ is valid because the DC gain is $I$ (verified). ✓
* **90° corner:** the Klipper junction speed for a 90° turn is
  $$v_j^2 = a\,\delta\,\frac{\sin(\theta/2)}{1 - \sin(\theta/2)},\qquad \delta = \frac{\text{SCV}^2(\sqrt2 - 1)}{a},\qquad \theta = 90° \;\Rightarrow\; v_j = \text{SCV}.$$
  The code uses $v_j = \text{SCV}$ (`ballbar.py:258`). ✓
* **Trapezoid peak speed:** $v_{peak} = \min\!\left(v_c,\ \sqrt{\tfrac12(v_0^2 + v_1^2) + a\,d}\right)$. ✓
* **ISO 230-4 circularity:** $G_{xy} = \max e_r - \min e_r$, with $e_r = |\mathbf r| - R$. The reversal spikes are generated from three sources:
  1. Stribeck friction, $F = -[F_c + (F_s - F_c)e^{-(v/v_s)^2}]\tanh(v/\varepsilon)$;
  2. a backlash operator of width equal to the ISO 5753 bearing radial clearance;
  3. LPV $\Delta K(\theta)$ applied as a disturbance force.

---

## 5. Lookahead planner & parser (`gcode/simulator.py`, `gcode/parser.py`)

* **Planner:** backward pass $v_{in,i} \le \sqrt{v_{out,i}^2 + 2a_iL_i}$ followed by a forward pass, both correct. ✓ Segment timing is trapezoidal. ✓
* **Junction speed:** the formula at `simulator.py:188-192` is **not** Klipper's (finding C1).
* **Parser:** handles G0/G1/G90/G91/M82/M83/G92/M204 correctly, converts F from mm/min to mm/s, and detects layers by a +0.02 mm Z increase on extrusion moves.

---

## 6. Verification run (key outputs)

```
FK(IK(p)) max error: corexy 5.7e-14 mm | cartesian 0 | delta 1.7e-13 mm (300 random pts each)
k_theta: 1.8° p=50 → 25.0 N·m/rad ; 0.9° p=100 → 50.0 N·m/rad (static, exactly 2×)
         at 600 rpm: 17.3 vs 24.6 N·m/rad (eta_V 0.69 vs 0.49 — doubled ω_e roll-off)
half-current T_eff/T_hold = 0.625 (linear expectation 0.500)
CoreXY 45° unit force → F_A = 0.707, F_B = 0 (code assumes 0.5)
LPV max Re(λ) [1/s]: voron −7.3, vzbot AWD −11.2, cartesian −5.1, delta −8.6 (open loop);
                     FOC+80 Hz notch −13…−36  → all asymptotically stable; DC gain = I everywhere
Junction speed (SCV = 5): turn 30° code 8.46 / Klipper 17.13 ; 90° 3.22 / 5.00 ;
                          120° 1.97 / 3.22 ; 179.9° 1.12 / 0.10 mm/s
`import printsim.kinematics` alone → ImportError (circular import with printsim.fea.mechatronics)
```

---

## 7. Audit findings table

| ID | Severity | Location | Finding | Concrete fix |
|---|---|---|---|---|
| **C1** | **CRITICAL BUG** | `gcode/simulator.py:183-192` | The junction-speed formula $v = \text{SCV}\sqrt{(1 - \sin\frac\phi2)/\sin\frac\phi2}$ is not Klipper's. It uses the turn angle with sin instead of cos, drops the $(\sqrt2 - 1)$ factor, and a 0.05 floor makes reversal speed nonzero (0.22·SCV). Effects: 90° corners run at 0.64·SCV (contradicting `ballbar.py:258`, which uses SCV); shallow turns are 2× too slow; 180° reversals carry a non-physical velocity jump. This biases `total_time_s`, `avg_print_speed` and ringing heuristics. | Port Klipper `toolhead.py`: `jcos = -dot(d_prev, d)`, `sin_half = sqrt(0.5*(1-jcos))`, `R = sin_half/(1-sin_half)`, `v2 = min(v2_prev, v2, a*δ*R)` with `δ = scv²(√2-1)/a`. Also add the centripetal `0.5*L*tan_half*a` limits. Add a unit test: 90° → SCV, 180° → 0. |
| **M1** | MODERATE ISSUE | `kinematics/solvers.py:347-354` | The CoreXY torque requirement uses the axis-aligned share 0.5, but the worst case for a vector accel limit is the 45° diagonal ($F_A = F/\sqrt2$). Together with the $\sqrt2$ speed already applied at `:332`, the motor margin is overstated by 41% on the force term. The inertia term `alpha_max = a/r_p` (`:345`) should also be $\sqrt2\,a/r_p$. | Use `share = 1/√2` for 2WD and `1/(2√2)` for AWD in the peak case, or evaluate both axis and diagonal cases and take the max. Use `alpha_max = √2·a/r_p` for CoreXY. |
| **M2** | MODERATE ISSUE | `kinematics/solvers.py:223-228` | The AWD cord model uses 4 spans of 0.25(w+d), i.e. a total of 1.0(w+d) versus 2.2(w+d) for 2WD. That gives ~3.6× the 2WD cord stiffness. AWD keeps the same two closed loops; each belt only gains a second motor terminating the long branch. The model inflates AWD drive stiffness, resonance and VFA predictions. It is also inconsistent with the position-dependent AWD model in `mechatronics.py:197-200`. | Derive the spans from the same geometry as `_corexy_branch_details` (carriage→front motor and carriage→rear motor, evaluated at the centre), or simply call `corexy_belt_span_stiffness_matrix(asm, 0.5, 0.5)` so both modules share one model. |
| **M3** | MODERATE ISSUE | `kinematics/solvers.py:10-14` ↔ `fea/mechatronics.py:27-30` | Circular import: `import printsim.kinematics` (or `from printsim.kinematics.solvers import …`) as the first import raises `ImportError`. It only works when `printsim.fea` is imported first. | Move `compute_bracket_stiffness_n_per_m` and `compute_drive_stiffness_details_n_per_m` into `fea/components.py` (or a new `drives.py`), or make the imports in `mechatronics.py` lazy. Add `tests/test_imports.py` that imports each subpackage in a fresh interpreter. |
| **M4** | MODERATE ISSUE | `fea/components.py:501-512`; `schema/printer.py:137-197` (`k_restraint_thermal`) | `k_restraint_thermal` has no unit suffix (values 0/8/12/18/8000), which violates the AGENTS.md unit rule. It is mixed directly with `k_plate_taco` in N/m (~1.2e6 N/m for 6 mm Al at 300 mm). The `arch_fraction` term ≈ 1e-5…7e-3, so the taco amplitude is suppressed 10–250× except for `rigid_4pt`. There is also no Euler buckling threshold check ($P = k_{ax}\Delta \ge P_{cr} = \pi^2 D b/L^2$). | Rename to `k_restraint_thermal_n_per_m` with datasheet-scaled values. Replace the √(fraction) heuristic with a compatibility solve: restraint force $P = k_r(\Delta L_{th} - \Delta L_{plate})$; if $P > P_{cr}$, then $w = (2/\pi)\sqrt{L\,\Delta_{excess}}$. |
| **M5** | MODERATE ISSUE | `gcode/ballbar.py:380-392, 465-467` | An extra first-order "tracking" low-pass is applied to the command: $\tau = L/R + 11$ ms for open-loop, $\omega_p$ for servo. The state-space already contains the stepper load-angle spring $k_{em}$ and the servo loop ($k = m\omega_v\omega_p$), so position lag is double-counted. Klipper's step timing is deterministic, so no 11 ms interpolator lag exists. This produces about −22 µm of artificial radius shrink at 120 mm/s, R = 50 mm ($-v^2\tau^2/2R$). | Remove the filter for `open_loop_stepper` and `foc_servo`, and let the state-space produce the lag. If a current-loop lag is wanted, model it as an electrical state ($L\,di/dt = V - Ri - k_e\omega$) inside $A$. |
| **M6** | MODERATE ISSUE | `gcode/ballbar.py:418-425, 446-477` | ISO 5753 idler-bearing radial clearance is applied as reversal backlash in Cartesian X/Y. In a preloaded belt drive the hub load $2T_0\sin(\theta/2) > 0$ never reverses, so the clearance stays closed and causes no lost motion. For CoreXY, any drive backlash would also act in motor (A/B) space, producing spikes at 45°/135°, not at 0°/90°. The result is spurious quadrant glitches. | Drop the bearing-play backlash, or gate it on $T_0 - F/2 \le 0$, and apply it in motor coordinates through $T$. Put reversal spikes on Stribeck friction (already modelled) plus belt-tooth flank clearance if desired. |
| **M7** | MODERATE ISSUE | `gcode/ballbar.py:301-313` | The ringing comparison windows are asymmetric. The unshaped window starts at `t_corner`, while the shaped window starts at `t_corner + t_shaper_tail`, so the first post-corner transient of the shaped run is never measured. This inflates `ringing_reduction_pct`. | Use the same window for both runs, offset by each run's group delay ($t_{corner} + \bar t_{shaper}$ for shaped, $t_{corner}$ for unshaped), or measure both from `t_corner + t_shaper_tail`. |
| **M8** | MODERATE ISSUE | `fea/mechatronics.py:405-411, 673-684` | The FOC servo is a fixed PD impedance to ground, and the notch is cascaded on the reference input. That makes it a feed-forward prefilter: it cannot change closed-loop poles or resonance damping, unlike a real drive's in-loop torque-command notch. The model also has no integrator, so static servo stiffness is finite. | Model the servo as an explicit controller state-space (PI velocity + P position + notch on the torque command), closed around measured motor position/velocity, and form $A_{cl}$ by feedback interconnection. |
| m1 | MINOR / LIMITATION | `schema/printer.py:618-623` | The `tanh` current saturation overestimates torque below rated current (0.625 vs 0.5 at half current). | Use a linear law below $I_{rated}$ and soft saturation only above it: $T = T_h\,r$ for $r \le 1$, else $T_h[1 + \tanh(\cdot)]$. |
| m2 | MINOR / LIMITATION | `schema/printer.py:631, 625-639` | $k_e = T_h/I_{rated}$ ignores the two-phase holding convention ($T_h \approx \sqrt2\,k_t I$). $\eta_V$ adds back-EMF and $L\omega$ drops as separate magnitudes rather than as a phasor. | Use $k_t = T_h/(\sqrt2 I)$ and $\eta_V = \min(1,\ V/|(R + jN_r\omega L)I + k_e\omega|)$. |
| m3 | MINOR / LIMITATION | `kinematics/solvers.py:187` | The 0.72 holding-to-pull-out factor is uncited. Stiffness $k_\theta$ also ignores load-angle softening $\cos(N_r\theta_L)$. | Cite a datasheet pull-out curve, and add $k_\theta\cos(\arcsin(T_{load}/T_h))$. |
| m4 | MINOR / LIMITATION | `kinematics/solvers.py:397` | Frame belt preload is doubled for AWD, but AWD keeps the same 2 belts. | Use `frame_preload = 2*T0` for both modes. |
| m5 | MINOR / LIMITATION | `kinematics/solvers.py:399-407` | Delta reachability only checks the 4 rectangle corners against arm reach. It ignores the circular bed, carriage-height limits and arm-angle limits. | Sample the circular print disk, and check $z_i \le z_{max}$ and arm angle ≥ ~20°. |
| m6 | MINOR / LIMITATION | `fea/components.py:300-308` | The slip criterion is Euler–Eytelwein × an empirical $(1 + 0.45z_e)$ multiplier. Toothed belts actually fail by tooth jump (ratcheting), not friction slip. | Use the Gates ratcheting-torque tables or a tooth-climb model: $T_s \ge$ the tooth-jump tension for the given $z_e$ and width. |
| m7 | MINOR / LIMITATION | `fea/components.py:250` | The tooth shear stiffness $k_t = 5.5\times10^4$ N/(m·mm) and the Hertz $F_2 = 0.68$ have no datasheet citation. | Cite Johannesson & Distner / Gates 2GT test data, and the Harris & Kotzalas elliptic-integral tables. |
| m8 | MINOR / LIMITATION | `fea/components.py:72-78, 225-227` | The "Iwan" damping is a constant lookup table; no Iwan/Masing hysteresis element exists. | Rename to `joint_damping_delta`, or implement a parallel-Jenkins Iwan element with amplitude-dependent $\zeta$. |
| m9 | MINOR / LIMITATION | `fea/components.py:478-490` | The bed gradient bow uses the longest side, not the diagonal. For a free square plate, the corner-to-centre bow is ~2× larger. | Use $L_d = \sqrt{L_x^2 + L_y^2}$ for the corner bow, or report both. |
| m10 | MINOR / LIMITATION | `fea/mechatronics.py:431-455` | The Acarnley damping coefficients are fitted. The 0.5% LTI floor (`452-453`) means mid-band instability is flagged but never simulated. | Document the fit source, and expose `zeta_raw` in `to_dict`. Optionally add a linear time-varying (LTV) simulation without the floor. |
| m11 | MINOR / LIMITATION | `fea/mechatronics.py:796-801` | The workspace structural frequency is modulated by uncited empirical factors (0.22/0.14/0.08); FEA is not re-run per grid point. | Re-solve the modal FEA at grid positions (toolhead node moved), or fit the factors from FEA sweeps. |
| m12 | MINOR / LIMITATION | `fea/mechatronics.py:576-582` | The frame mode is calibrated with $k = (m_f + m_c)\omega_s^2$ from the FEA shaper frequency. If the FEA toolhead mode already includes drive compliance, the belt is counted twice. | Calibrate $K_f$ from the FEA with belts constrained rigid, or subtract the belt compliance. |
| m13 | MINOR / LIMITATION | `fea/mechatronics.py:668-670` | The output is the absolute carriage DOF. The print error is nozzle relative to bed, and the bed sits on the frame. | Output $q_c - \beta q_f$, with $\beta$ = bed participation in the frame mode. |
| m14 | MINOR / LIMITATION | `gcode/ballbar.py:497-518` | Stribeck friction is explicit with $\tanh(v/1.5\,\text{mm/s})$. The stability ratio $c_{eff}\,dt/m$ = 0.91 for the Voron preset, but it exceeds 2 for light carriages ($m \to 0.05$ kg) or HGH rails, which causes chatter and spurious spikes. | Use semi-implicit friction (clamp the velocity sign change within a step), or substep the friction at dt/10. |
| m15 | MINOR / LIMITATION | `gcode/ballbar.py:408-414` | Y-axis friction reuses the X-rail carriage data (single carriage). | Use the Y rail profile × number of Y carriages. |
| m16 | MINOR / LIMITATION | `gcode/ballbar.py:500-513` | The LPV $\Delta K$ force is applied only to the carriage, with no reaction on the motor/frame DOFs, so momentum is not conserved. | Apply $-E^T\Delta k\,E\,q$ to the full DOF vector, i.e. rebuild $A_d$ per sector (72 cached expm calls). |
| m17 | MINOR / LIMITATION | `gcode/ballbar.py:541-542` | ISO 230-4 $G_{xy}$ is taken about the commanded centre with no least-squares centre fit. | Fit an LSQ circle (Kåsa) before taking max − min. |
| m18 | MINOR / LIMITATION | `gcode/ballbar.py:95-105`, `simulator.py:103-114` | When $v_0 > v_{peak} \ge v_1$ on short segments, the accel branch uses $+a$ while $t_{acc} > 0$, so the trajectory is wrong (rare: only infeasible segments). | Handle the case where deceleration starts immediately, or ensure the lookahead guarantees feasibility. |
| m19 | MINOR / LIMITATION | `gcode/simulator.py:230, 301` | Volumetric flow is averaged over the segment, which underestimates cruise-phase flow. Filament density is hard-coded to PLA (1.24). | Use $\dot V = (E/L)\,A_f\,v_{peak}$, and take the density from `get_material(filament)`. |
| m20 | MINOR / LIMITATION | `gcode/parser.py:81` | G2/G3 arcs are ignored without updating `pos`, so the next move starts from a stale position (fake long moves). `G1X10` without spaces and `G01` are not parsed. Klipper `SET_VELOCITY_LIMIT` and G10/G11 are not handled. | Linearize G2/G3 (I/J/R) into chords; tokenize with the regex `([A-Z])([-+]?\d*\.?\d+)`; normalize `G0x`. |
| m21 | MINOR / LIMITATION | `kinematics/collision.py:49-51` | The bedslinger Y clearance `fd - (by + 40)` ignores the bed sweep envelope (≈ 2·by). Delta uses rectangular checks. | Use `fd - (2*by + margin)` for the swept-bed envelope, and a circular check for delta. |
| m22 | MINOR / LIMITATION | `kinematics/solvers.py:203-299` vs `fea/mechatronics.py:173-304` | There are two independent drive-stiffness models (fixed span fractions vs position-dependent loops), so the reported `drive_stiffness_n_per_mm` differs from the LPV centre value. | Make one model canonical (see M2). |
| V1 | VERIFIED | `kinematics/solvers.py:121-181` | CoreXY/Cartesian/Delta FK∘IK exact to machine precision; delta lower-root selection correct. | — |
| V2 | VERIFIED | `schema/printer.py:613-615, 641-643` | $N_r$ = steps/4 (50 for 1.8°, 100 for 0.9°); $k_\theta = N_rT_h\eta_V$; 0.9° gives 2× static stiffness. | — |
| V3 | VERIFIED | `kinematics/solvers.py:270-282, 332` | $\sqrt2$ diagonal motor speed; $k_{em} = k_\theta/r_p^2$; CoreXY axis motor factor 2 (2WD) / 4 (AWD) from $x = (A + B)/2$. | — |
| V4 | VERIFIED | `fea/components.py:230-297` | Ernst sag modulus, Gerbert ladder (continuum-consistent), $48EI/L^3$ / $3EI/L^3$ shaft stiffness, Harris bearing $\delta_r$, Stribeck $Q_{max} = 5F/Z$. | — |
| V5 | VERIFIED | `fea/components.py:116-147` | Hertz $E^*$, $R_e = d\sqrt{5.25}$ (f = 0.525), tangent stiffness $1.5\,Q^{1/3}/c$, moment stiffnesses. | — |
| V6 | VERIFIED | `fea/components.py:196, 424-457` | Newmark γ dimensionless; generalized Timoshenko curvature; 3-layer backer compatibility matrix symmetric and correct. | — |
| V7 | VERIFIED | `fea/components.py:482-490, 509, 514-523`; `materials.py:51-73` | Bed gradient bow $L^2\alpha q/8k$; elastica $w = (2/\pi)\sqrt{L\Delta}$; Z drift; Mahieux E(T). | — |
| V8 | VERIFIED | `fea/mechatronics.py:223, 253, 640-666, 487-503` | $K_{cart}$ CoreXY form; $E^T\,\mathrm{diag}(k)\,E$ assembly; state-space units; DC gain = I; all eigenvalues in the LHP; biquad notch realization and cascade exact. | — |
| V9 | VERIFIED | `gcode/ballbar.py:124-163, 258` | Van Loan ZOH exact; deviation-coordinate run valid; 90° junction = SCV. | — |
| V10 | VERIFIED | `gcode/simulator.py:195-201` | Backward/forward lookahead passes and trapezoidal timing are correct (given junction caps). | — |

---

## 8. Key assumptions

* Small-displacement linear elastic springs; stiffnesses linearized about the pretension operating point.
* Stepper electromagnetic stiffness evaluated at a single mid-band operating speed (0.35·rpm_max). The model is LPV in position but LTI in speed.
* Frame represented by one modal DOF per axis, calibrated from the FEA shaper frequency.
* Belts are massless springs; transverse string modes appear only in `pluck_freq_hz`.
* Thermal fields are steady-state with uniform chamber temperature; the bed has a linear through-thickness gradient and is otherwise free.

## 9. References

* Acarnley, P. (2002), *Stepping Motors: A Guide to Theory and Practice*, 4th ed., IET.
* Leenhouts, A. (1991), *The Art and Practice of Step Motor Control*.
* Ernst, H.-J. (1965), *Der E-Modul von Seilen*, Der Bauingenieur 40.
* Gerbert, G. (1972), *Force and slip behaviour in V-belt / toothed-belt drives*; Childs (2014), *Mechanical Design Engineering Handbook*.
* Harris, T. & Kotzalas, M. (2006), *Rolling Bearing Analysis*, 5th ed.
* Newmark, Siess & Viest (1951), *Proc. SESA* 9(1); EN 1995-1-1 Annex B (γ-method).
* Timoshenko, S. (1925), *J. Opt. Soc. Am.* 11(3).
* Mahieux, C. & Reifsnider, K. (2001), *Polymer* 42.
* Klipper `klippy/toolhead.py` (junction deviation), `klippy/extras/shaper_defs.py`.
* ISO 230-4:2005, *Circular tests for NC machine tools*; ISO 5753-1 (radial internal clearance).
