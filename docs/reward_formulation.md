# Reward formulation (v4 recipe; identical for PPO, TD3, Dreamer)

Source: `sb3/AUVEnv.py` (`_tracking_reward`, `_update_references`, `_update_surge_reference`, `_update_error_integrals`),
parameters in `sb3/config/config_sac.yaml` → `tracking_reward`. Control step $\Delta t = 0.1$ s.
Flags: `RACE2_PAPER_TERMS=1 RACE2_SURGE_REWARD=horizon RACE2_REF_ALL=1 RACE2_INTEGRAL=1 RACE2_K_SLEW=0`.

## Total reward

$$
r_t = R_\text{track}
- k_\text{rate}\,\frac{1}{m}\sum_{i=1}^{m}\left(a_{t,i}-a_{t-1,i}\right)^2
- \lambda(d_t)\,\frac{1}{n}\sum_{i=1}^{n}u_{t,i}^2
- k_\text{slew}\,\frac{1}{n_s}\sum_{j=1}^{n_s}\left(\frac{\Delta\beta_{t,j}}{\dot\beta_\text{max}\Delta t}\right)^2
- k_\text{oob}\,\mathbb{1}\!\left[z_t\notin[z_\text{lo},z_\text{hi}]\right]
- k_{\dot\psi}\,r_t^2
- k_\text{term}\,\mathbb{1}[\text{terminated}]
$$

- $a_t$: policy action; with the rate servo action (articulated race2) the sum runs over the $m=n$ thruster dimensions only.
- $u_t\in[-1,1]^n$: thruster commands ($n=4$ race2, $n=6$ race_auv); $\beta$: servo angles ($n_s=2$ race2, 0 race_auv).
- $r_t$: yaw rate (rad/s).

## Score function

$$
s(e,\sigma) = \tfrac12\exp\!\left(-\frac{e^2}{\sigma^2}\right) + \tfrac12\left(1-\tanh\frac{|e|}{\kappa\sigma}\right),\qquad \kappa=10,\quad s\in(0,1]
$$

Fine Gaussian for precision near the target, coarse tanh for gradient far from it.

## Tracking term

$$
R_\text{track} = \frac{\sum_{x\in\mathcal D} w_x\, r_x}{\sum_{x\in\mathcal D} w_x},\qquad \mathcal D=\{z,u,\psi,\theta,\phi\}
$$

$$
r_u = \tfrac12\, s\!\left(u_r-u,\ \sigma_u\right) + \tfrac12\, s\!\left(s_\text{err},\ \sigma_s\right),\qquad
r_x = \tfrac12\, s\!\left(\tilde e_x,\ \sigma_x\right) + \tfrac12\, s\!\left(I_x,\ \sigma_{I,x}\right),\quad x\in\{z,\psi,\theta,\phi\}
$$

## Reference model (depth and attitude, `RACE2_REF_ALL`)

For $x\in\{z,\psi,\theta,\phi\}$ with target $x^*$ (critically damped, rate and acceleration limited; angle differences wrapped to $[-\pi,\pi)$):

$$
\ddot x_r = \operatorname{sat}_{a_{\max,x}}\!\left(\omega_{n,x}^2\,(x^*-x_r) - 2\,\omega_{n,x}\,\dot x_r\right),\quad
\dot x_r \leftarrow \operatorname{sat}_{v_{\max,x}}\!\left(\dot x_r + \ddot x_r\Delta t\right),\quad
x_r \leftarrow x_r + \dot x_r\Delta t
$$

Tracking errors are to the reference: $\tilde e_z = z_r - z$, and

- Euler (race2): $\tilde e_x = \operatorname{wrap}(x_r - x)$ for $x\in\{\psi,\theta,\phi\}$;
- rotation (race_auv, `RACE2_ATT_REP=rot`): $[\tilde e_\phi,\tilde e_\theta,\tilde e_\psi]^\top = \operatorname{Log}\!\left(R^\top R_d\right)^\vee$,
  $R_d = R_z(\psi_r)R_y(\theta_r)R_x(\phi_r)$ (rotation vector of the body-frame attitude error; no singularity at $\theta=\pm90^\circ$).

## Horizon surge (`RACE2_SURGE_REWARD=horizon`)

Velocity reference (acceleration limited) and virtual target moving at $u_r$; $s_\text{err}$ = along-track distance to it:

$$
\ddot u_r = \omega_n^2\,(u^*-u_r) - 2\zeta\omega_n\,\dot u_r,\quad
\dot u_r \leftarrow \operatorname{sat}_{a_\max}\!\left(\dot u_r + \ddot u_r\Delta t\right),\quad
u_r \leftarrow u_r + \dot u_r\Delta t
$$

$$
s_\text{err} \leftarrow \operatorname{sat}_{s_\max}\!\left(\left(s_\text{err} + (u_r - u)\Delta t\right)\left(1-\frac{\Delta t}{\tau}\right)\right)
$$

## Error integrals (`RACE2_INTEGRAL`)

Leaky, clamped (anti-windup), same form as $s_\text{err}$:

$$
I_x \leftarrow \operatorname{sat}_{I_{\max,x}}\!\left(\left(I_x + \tilde e_x\Delta t\right)\left(1-\frac{\Delta t}{\tau}\right)\right),\qquad x\in\{z,\psi,\theta,\phi\}
$$

## Adaptive effort weight (`RACE2_PAPER_TERMS`; IROS 2025 Eq. 4 as in mvp_control)

$$
\lambda(d) = \frac{k}{1+e^{-a d}},\qquad
d = \sqrt{\frac{1}{|\mathcal D|}\sum_{x\in\mathcal D}\left(\frac{e_x}{\kappa\,\sigma_x}\right)^2}
$$

$e_x$: error to the raw target (surge: $u_r-u$). $\lambda$ goes from $k/2$ at zero error to $k$ at large error.

## Termination

$$
\text{terminated} = \left[z\notin[z_\text{min},z_\text{max}]\right]\ \lor\ \text{tilted}
$$

- race2 (Euler): $\text{tilted} = |\phi|>\phi_\max \lor |\theta|>\theta_\max$
- race_auv (rot): $\text{tilted} = \arccos\!\left(\left(R\,\mathbf e_3\right)^\top R_d\,\mathbf e_3\right) > \alpha_\max$ (angle between actual and reference body $z$ axes; heading excluded)

## Parameters

| Symbol | Value | Symbol | Value |
|---|---|---|---|
| $w_z,\ \sigma_z$ | 1.0, 0.10 m | $k_\text{rate}$ | 0.05 |
| $w_u,\ \sigma_u$ | 3.0, 0.05 m/s | $k,\ a$ (effort) | 0.04, 3.0 |
| $w_\psi,\ \sigma_\psi$ | 1.0, 0.087 rad | $k_\text{slew}$ | 0 (v4) |
| $w_\theta,\ \sigma_\theta$ | 1.0, 0.087 rad | $k_\text{oob}$, $[z_\text{lo},z_\text{hi}]$ | 1.0, [0.3, 25] m |
| $w_\phi,\ \sigma_\phi$ | 1.0 (0.5 race2 fixed: no roll authority, target 0), 0.087 rad | $k_{\dot\psi}$ | 0.05 |
| $\omega_n,\ \zeta,\ a_\max$ (surge) | 0.5, 1.0, 0.15 m/s² | $k_\text{term}$ | 10 |
| $\sigma_s,\ s_\max$ | 0.3 m, 2.0 m | $[z_\text{min},z_\text{max}]$ | [0, 28] m |
| $\tau$ (all leaks) | 20 s | $\phi_\max,\theta_\max$ (race2) | 1.4 rad |
| $\omega_n, v_\max, a_\max$ depth | 0.5, 0.3 m/s, 0.1 m/s² | $\alpha_\max$ (race_auv) | 1.4 rad |
| $\omega_n, v_\max, a_\max$ yaw | 0.8, 0.35 rad/s, 0.3 rad/s² | $\sigma_{I,z},\ I_{\max,z}$ | 1.0, 4.0 m·s |
| $\omega_n, v_\max, a_\max$ pitch, roll | 1.0, 0.35 rad/s, 0.5 rad/s² | $\sigma_{I},\ I_{\max}$ (angles) | 0.5, 2.0 rad·s |

Attitude targets: race2 pitch ±20°, roll ±12° (articulated only); race_auv roll and pitch ±90°.
