# control/reference.py
import math
from dataclasses import dataclass

def sat(val, vmin, vmax):
    return max(vmin, min(vmax, val))

@dataclass
class PosRefParams:
    omega: float = 0.2      # ω_r,p
    zeta: float  = 1.0      # ζ_r,p
    Ki: float    = 0.05     # integral gain for xi_p
    vmax: float  = 0.5      # m/s cap

@dataclass
class HeadRefParams:
    omega: float = 0.4
    zeta: float  = 1.0
    Ki: float    = 0.2
    rmax: float  = 0.4      # rad/s cap

class ReferenceFilter:
    """
    1D 2nd-order tracking for distance along a line + separate heading filter.
    Equations inspired by your cheat sheet (15)-(20) and (21)-(23).  :contentReference[oaicite:1]{index=1}
    """
    def __init__(self, pos_params: PosRefParams, head_params: HeadRefParams):
        self.pp = pos_params
        self.hp = head_params
        # position channel internal states
        self.pr = 0.0      # distance (m)
        self.vr = 0.0      # speed along line (m/s)
        self.xip = 0.0     # integrator

        # heading channel internal states
        self.psir = 0.0    # heading reference
        self.rr = 0.0      # yaw rate
        self.xipsi = 0.0

    @staticmethod
    def _wrap_pi(a):
        while a > math.pi:  a -= 2*math.pi
        while a < -math.pi: a += 2*math.pi
        return a

    def reset(self, psi_now=0.0):
        self.pr = 0.0; self.vr = 0.0; self.xip = 0.0
        self.psir = psi_now; self.rr = 0.0; self.xipsi = 0.0

    def step(self, dt, pd, psi_d):
        # --- Position channel (distance along line) ---
        # ξ̇p = sat_vmax( Ki (pd - pr) )
        self.xip += self.pp.Ki * (pd - self.pr) * dt
        xi_rate = sat(self.xip, -self.pp.vmax, self.pp.vmax)

        # v̇r = ω^2 (ξp - pr) - 2 ζ ω vr
        self.vr += (self.pp.omega**2 * (xi_rate - self.pr) - 2*self.pp.zeta*self.pp.omega*self.vr) * dt
        # ṗr = vr
        self.pr += self.vr * dt

        # --- Heading channel ---
        # ψ error wrapped
        epsi = self._wrap_pi(psi_d - self.psir)
        self.xipsi += self.hp.Ki * epsi * dt
        xi_psi_rate = sat(self.xipsi, -self.hp.rmax, self.hp.rmax)

        # ṙr = ω^2 (ξψ - ψr) - 2 ζ ω rr
        self.rr += (self.hp.omega**2 * (xi_psi_rate - self.psir) - 2*self.hp.zeta*self.hp.omega*self.rr) * dt
        # ψ̇r = rr
        self.psir = self._wrap_pi(self.psir + self.rr * dt)

        # Return 1D references; conversion to 2D is done by the runner using line angle φ.
        return self.pr, self.vr, self.rr, self.psir
