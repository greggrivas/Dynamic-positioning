# control/controller.py
"""
PID + Feedforward controller for a simplified 3-DOF vessel model (surge, sway, yaw).

This module provides:
- PIDGains: Dataclass holding controller gains and limits with detailed explanations.
- PIDFFController: A combined feedforward + PID controller for a vessel modeled in
  surge (x), sway (y) and yaw (psi). The controller computes control efforts (tau)
  that are suitable as generalized forces/torques for an AGX vessel model.

Conceptual model (variables):
- eta = [x, y, psi]  : pose in world frame (position x,y and yaw psi)
- nu  = [u, v, r]    : body-fixed velocities (surge, sway, yaw-rate)
- nudot = [udot, vdot, rdot] : accelerations in body frame (feedforward)
- tau = control generalized force vector [tau_x, tau_y, tau_psi]

Controller law implemented (component-wise):
  tau = tau_ff + Kp * e_p + Kd * e_v + sigma
where
  tau_ff = M * nudot_r + D * nu_r   (simple diagonal inertia + damping feedforward)
  e_p = eta_r - eta_hat            (position/orientation error; psi wrapped)
  e_v = nu_r - nu_hat              (velocity error)
  sigma = integral term (anti-windup applied)

Notes:
- Position errors for x,y are assumed expressed in the world frame (runner must ensure consistent frames).
- Yaw error is angle-wrapped into [-pi, pi] before use.
- Integral anti-windup: integrator increments only when control is not saturated.
- This controller uses diagonal approximations of M and D (vectors of three elements).
"""

import math
from dataclasses import dataclass
from typing import Sequence, Tuple


@dataclass
class PIDGains:
    """
    Holds PID gains and saturation limits used by PIDFFController.

    Attributes
    ----------
    Kp_x, Kp_y, Kp_psi : float
        Proportional gains for surge (x), sway (y) and yaw (psi).
        Units: [N/m] for translation terms if position in meters and tau in Newtons,
               [Nm/rad] for yaw if psi in radians and tau_psi in Newton-meters.
    Kd_x, Kd_y, Kd_psi : float
        Derivative gains for velocity errors. Units consistent with Kp above divided by seconds.
    Ki_x, Ki_y, Ki_psi : float
        Integral gains. Units consistent with Kp * 1/s.
        Default zero: no integral action unless explicitly set.
    tau_max : float
        Symmetric scalar saturation limit applied component-wise to each tau output.
        Typical value must be chosen according to thruster capacity (N or Nm).
    """
    Kp_x: float = 15_000.0  # Proportional gain for surge
    Kp_y: float = 15_000.0  # Proportional gain for sway
    Kp_psi: float = 8_000.0  # Proportional gain for yaw
    Kd_x: float = 25_000.0  # Derivative gain for surge
    Kd_y: float = 25_000.0  # Derivative gain for sway
    Kd_psi: float = 12_000.0  # Derivative gain for yaw
    Ki_x: float = 500.0  # Integral gain for surge
    Ki_y: float = 500.0  # Integral gain for sway
    Ki_psi: float = 200.0  # Integral gain for yaw
    tau_max: float = 80_000.0  # generic scalar saturation (apply to each tau component)


class PIDFFController:
    """
    PID + Feedforward controller for 3-DOF vessel dynamics.

    Initialization
    --------------
    PIDFFController(M_diag, D_diag, gains)

    Parameters
    ----------
    M_diag : Sequence[float]
        Diagonal inertia vector [m_x, m_y, I_z] used for feedforward:
        tau_ff_component = M_diag[i] * nudot_r[i]
    D_diag : Sequence[float]
        Diagonal damping vector [Xu, Yv, Nr] used in feedforward:
        tau_ff_component += D_diag[i] * nu_r[i]
    gains : PIDGains
        Gains and saturation limits.

    Internal state
    --------------
    sigma_x, sigma_y, sigma_psi : floats
        Integral accumulator (anti-windup handling implemented).

    Behavior
    --------
    - Computes tau = tau_ff + Kp*e_pos + Kd*e_vel + sigma
    - Saturates each tau component to [-tau_max, tau_max]
    - Anti-windup: integrator increments only when the pre-saturated control is within limits.
    """

    def __init__(self, M_diag: Sequence[float], D_diag: Sequence[float], gains: PIDGains):
        """
        Construct the controller.

        Parameters
        ----------
        M_diag : Sequence[float]
            Diagonal inertia coefficients [M_x, M_y, M_psi].
        D_diag : Sequence[float]
            Diagonal damping coefficients [D_x, D_y, D_psi].
        gains : PIDGains
            Controller gains and saturation limit object.
        """
        self.M = M_diag  # [m, m, Iz]
        self.D = D_diag  # [Xu, Yv, Nr]
        self.g = gains  # PID gains
        # Integral (sigma) terms for x, y, psi
        self.sigma_x = 0.0
        self.sigma_y = 0.0
        self.sigma_psi = 0.0

    @staticmethod
    def _wrap_pi(a: float) -> float:
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a

    def step(
        self,
        dt: float,
        eta_r: Sequence[float],
        nu_r: Sequence[float],
        nudot_r: Sequence[float],
        eta_hat: Sequence[float],
        nu_hat: Sequence[float],
    ) -> Tuple[float, float, float]:
        """
        Compute one controller update.

        Parameters
        ----------
        dt : float
            Time step (seconds) since last controller call. Used for integrator updates.
        eta_r : Sequence[float]
            Reference/world pose [x_r, y_r, psi_r] (meters, meters, radians).
        nu_r : Sequence[float]
            Reference/body velocities [u_r, v_r, r_r] (m/s, m/s, rad/s).
        nudot_r : Sequence[float]
            Reference body accelerations [udot_r, vdot_r, rdot_r] (m/s^2, m/s^2, rad/s^2).
            Used in feedforward calculation M * nudot_r.
        eta_hat : Sequence[float]
            Estimated/observed pose [x_hat, y_hat, psi_hat].
        nu_hat : Sequence[float]
            Estimated/observed body velocities [u_hat, v_hat, r_hat].

        Returns
        -------
        (tau_x, tau_y, tau_psi) : Tuple[float, float, float]
            Saturated control outputs for surge, sway and yaw. Units consistent with
            system (e.g. Newtons for tau_x/y, Newton-meters for tau_psi).

        Algorithm
        ---------
        1. Compute position error: ex = x_r - x_hat, ey = y_r - y_hat, epsi = wrap(psi_r - psi_hat)
        2. Compute velocity error: eu = u_r - u_hat, ev = v_r - v_hat, er = r_r - r_hat
        3. Compute feedforward: tau_ff = M * nudot_r + D * nu_r (component-wise)
        4. Compute PD + integral pre-sat: tau = tau_ff + Kp * e_p + Kd * e_v + sigma
        5. Saturate each tau component to [-tau_max, tau_max]
        6. Anti-windup: increment integrator only when the pre-sat tau was not clipped.
        """
        # Unpack references and estimates
        xr, yr, psir = eta_r  # Reference pose
        ur, vr, rr = nu_r  # Reference velocities
        udr, vdr, rdr = nudot_r  # Reference accelerations (feedforward)

        xh, yh, psih = eta_hat  # Estimated pose
        uh, vh, rh = nu_hat  # Estimated velocities

        # Position/orientation errors (world frame for x,y; psi wrapped)
        ex = xr - xh  # Surge position error (m)
        ey = yr - yh  # Sway position error (m)
        epsi = self._wrap_pi(psir - psih)  # Yaw error (rad)

        # Velocity errors (body frame)
        eu = ur - uh  # Surge velocity error (m/s)
        ev = vr - vh  # Sway velocity error (m/s)
        er = rr - rh  # Yaw-rate error (rad/s)

        # Feedforward from diagonal inertia and damping:
        tau_ff_x = self.M[0] * udr + self.D[0] * ur
        tau_ff_y = self.M[1] * vdr + self.D[1] * vr
        tau_ff_psi = self.M[2] * rdr + self.D[2] * rr

        # PD + integral action
        tau_x = tau_ff_x + self.g.Kp_x * ex + self.g.Kd_x * eu + self.sigma_x
        tau_y = tau_ff_y + self.g.Kp_y * ey + self.g.Kd_y * ev + self.sigma_y
        tau_psi = (
            tau_ff_psi + self.g.Kp_psi * epsi + self.g.Kd_psi * er + self.sigma_psi
        )

        # Simple symmetric saturation helper
        def sat(v: float, lim: float) -> float:
            return max(-lim, min(lim, v))

        tau_x_s = sat(tau_x, self.g.tau_max)
        tau_y_s = sat(tau_y, self.g.tau_max)
        tau_psi_s = sat(tau_psi, self.g.tau_max)

        # Anti-windup: only integrate when the controller output was not limited
        # (i.e., when pre-sat tau equals saturated tau within numerical tolerance).
        tol = 1e-9
        if abs(tau_x_s - tau_x) < tol:
            self.sigma_x += self.g.Ki_x * ex * dt
        if abs(tau_y_s - tau_y) < tol:
            self.sigma_y += self.g.Ki_y * ey * dt
        if abs(tau_psi_s - tau_psi) < tol:
            self.sigma_psi += self.g.Ki_psi * epsi * dt

        return tau_x_s, tau_y_s, tau_psi_s