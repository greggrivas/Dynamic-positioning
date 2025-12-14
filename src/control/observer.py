# control/observer.py
import math
from dataclasses import dataclass

@dataclass
class ObsGains:
    """
    Observer tuning gains for the SimpleObserver.

    Attributes
    ----------
    L_eta : float
        Position-correction gain. Scales how strongly the observer corrects estimated
        positions (x,y,psi) based on measured pose errors. Units: 1/s (acts like a proportional
        feedback on position error injected into velocity estimates).
    L_nu_xy : float
        Optional separate velocity correction gain for surge/sway (u,v). Not used directly in
        the current implementation but provided for future extension. Units: 1/s.
    L_nu_psi : float
        Optional yaw-rate correction gain for r. Not used directly in the current implementation
        but included for symmetry. Units: 1/s.

    Notes
    -----
    The present SimpleObserver uses L_eta for both position correction and for leaking position
    error into the velocity estimates (a simple Luenberger-like structure). The separate L_nu_*
    fields are retained for clarity and future tuning.
    """
    L_eta: float = 2.0    # position correction gain
    L_nu_xy: float = 0.8  # vel gains for x,y
    L_nu_psi: float = 0.8 # vel gain for yaw

class SimpleObserver:
    """
    Minimal planar Luenberger-style observer for a 3-DOF vessel (x, y, psi) and body velocities (u, v, r).

    Purpose
    -------
    Provide a lightweight state estimator suitable for control loops where only noisy pose
    measurements (x,y,psi) are available at a lower rate than the controller. The observer:
      - Predicts pose by integrating current estimated body velocities.
      - Predicts body accelerations from current estimated velocities and applied generalized forces:
            M * nudot ≈ tau - D * nu  =>  nudot ≈ inv(M) * (tau - D*nu)
      - Uses measured pose to correct both pose and velocities via proportional gains (L gains).
      - Contains a simple angle-wrapping utility for yaw (psi).

    Assumptions & Conventions
    -------------------------
    - Frames:
        * eta = [x, y, psi] is expressed in the world (NED/planar) frame.
        * nu  = [u, v, r] are body-fixed velocities: u forward, v lateral (positive to port),
          r yaw-rate (rad/s).
    - M and D are expected to be diagonal-like sequences [M_x, M_y, M_psi] and [D_x, D_y, D_psi].
      The observer uses component-wise inverses of M.
    - Inputs to step(...) are:
        dt: time step in seconds,
        meas_x, meas_y, meas_psi: measured pose (may be noisy),
        tau_x, tau_y, tau_n: applied generalized forces/torque (used for feedforward acceleration estimate).
    - This observer is intentionally simple and not intended to replace a full Kalman filter
      when measurement noise statistics and process noise are known.

    Public API
    ----------
    - reset(x, y, psi): initialize internal state
    - step(dt, meas_x, meas_y, meas_psi, tau_x, tau_y, tau_n, M, D):
        perform one predict-correct iteration and return estimated pose and body velocities.
    """
    def __init__(self, gains: ObsGains):
        """
        Initialize the observer state and tuning gains.

        Parameters
        ----------
        gains : ObsGains
            Tuning gains container. Only L_eta is actively used for both pose correction
            and bleeding position error into velocity estimates in this implementation.
        """
        self.g = gains
        self.xhat = 0.0; self.yhat = 0.0; self.psihat = 0.0
        self.uhat = 0.0; self.vhat = 0.0; self.rhat = 0.0

    @staticmethod
    def _wrap_pi(a):
        while a > math.pi:  a -= 2*math.pi
        while a < -math.pi: a += 2*math.pi
        return a

    def reset(self, x, y, psi):
        self.xhat = x; self.yhat = y; self.psihat = psi
        self.uhat = 0.0; self.vhat = 0.0; self.rhat = 0.0

    def step(self, dt, meas_x, meas_y, meas_psi, tau_x, tau_y, tau_n, M, D):
        # Position prediction from body velocities:
        # η̇ = R(ψ) ν
        c = math.cos(self.psihat); s = math.sin(self.psihat)
        self.xhat += (c*self.uhat - s*self.vhat) * dt
        self.yhat += (s*self.uhat + c*self.vhat) * dt
        self.psihat = self._wrap_pi(self.psihat + self.rhat * dt)

        # Position error (in world/NED here)
        ex = meas_x - self.xhat
        ey = meas_y - self.yhat
        epsi = self._wrap_pi(meas_psi - self.psihat)

        # Velocity correction toward forces: M ν̇ ≈ τ - D ν  => ν̇ ≈ M^{-1}(τ - Dν)
        # Simple diagonal M,D expected
        invMx = 1.0 / M[0]; invMy = 1.0 / M[1]; invMz = 1.0 / M[2]
        du = (tau_x - D[0]*self.uhat) * invMx
        dv = (tau_y - D[1]*self.vhat) * invMy
        dr = (tau_n - D[2]*self.rhat) * invMz

        # Apply correction using L gains (position error bleeds into velocities)
        self.uhat += (du + self.g.L_eta*ex) * dt
        self.vhat += (dv + self.g.L_eta*ey) * dt
        self.rhat += (dr + self.g.L_eta*epsi) * dt

        # Small direct correction on positions to avoid drift
        self.xhat += self.g.L_eta * ex * dt
        self.yhat += self.g.L_eta * ey * dt
        self.psihat = self._wrap_pi(self.psihat + self.g.L_eta * epsi * dt)

        return (self.xhat, self.yhat, self.psihat), (self.uhat, self.vhat, self.rhat)
