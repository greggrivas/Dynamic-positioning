# control/observer.py
import math
from dataclasses import dataclass

@dataclass
class ObsGains:
    L_eta: float = 1.0    # position correction gain
    L_nu_xy: float = 0.8  # vel gains for x,y
    L_nu_psi: float = 0.8 # vel gain for yaw

class SimpleObserver:
    """
    Minimal Luenberger-style observer in NED/planar frame for (x,y,psi) and (u,v,r).
    Uses measured pose (with noise) and corrects velocities with gains.  :contentReference[oaicite:2]{index=2}
    """
    def __init__(self, gains: ObsGains):
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
