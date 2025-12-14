# control/reference.py
import math
from dataclasses import dataclass

"""
Reference generator and 2nd-order tracking filters for position along a line and heading.

This module provides:
- PosRefParams, HeadRefParams: simple dataclasses containing nominal filter tuning
  and limits for the positional and heading reference filters.
- ReferenceFilter: a paired 2nd-order tracking filter (one for distance along a line,
  one for heading) that generates smooth references for a vessel guidance / DP controller.

Conceptual overview
-------------------
The position channel tracks a scalar "distance along a line" pr using a second-order
reference model driven by a saturated integrator state ξ_p (xi_p). The heading channel
is a separate second-order model for ψr (psi reference) with its own integrator ξ_ψ.

Equations implemented (informal)
- ξ̇p = Ki_p * (pd - pr)            (integrator; saturated to vmax as a rate limiter)
- v̇r  = ω_p^2 (ξp - pr) - 2 ζ_p ω_p vr
- ṗr  = vr

and for heading
- ξ̇ψ = Ki_ψ * wrap(ψd - ψr)
- ṙr = ω_ψ^2 (ξψ - ψr) - 2 ζ_ψ ω_ψ rr
- ψ̇r = rr

Usage notes
- The filter produces scalar references (pr, vr, rr, ψr). The runner must map pr
  into a 2D target point along a line (using line angle φ) if needed.
- All angles are radians and wrapped into [-π, π] where appropriate.
"""


def sat(val, vmin, vmax):
    return max(vmin, min(vmax, val))

@dataclass
class PosRefParams:
    omega: float = 0.1      # ω_r,p
    zeta: float  = 1.2      # ζ_r,p
    Ki: float    = 0.05     # integral gain for xi_p
    vmax: float  = 0.3      # m/s cap

@dataclass
class HeadRefParams:
    omega: float = 0.2
    zeta: float  = 1.2
    Ki: float    = 0.2
    rmax: float  = 0.2      # rad/s cap

class ReferenceFilter:
    """
    1D 2nd-order tracking filter for distance along a line and a separate heading filter.

    This class implements two decoupled 2nd-order reference systems:
    - Position channel: produces pr (distance), vr (speed along the line) and an
      internal integrator xi_p used to track a desired distance pd.
    - Heading channel: produces psi_r (heading reference) and rr (yaw-rate) using a
      separate integrator xi_psi to track the desired heading psi_d.

    Constructor
    -----------
    ReferenceFilter(pos_params: PosRefParams, head_params: HeadRefParams)

    Methods
    -------
    reset(psi_now: float = 0.0)
        Reset internal states to zeros and set psi reference to psi_now.
    step(dt: float, pd: float, psi_d: float) -> Tuple[float, float, float, float]
        Advance the filter by dt seconds using desired distance pd and desired heading
        psi_d. Returns (pr, vr, rr, psi_r).

    Inputs and units
    ----------------
    - pd : desired scalar distance (m) along the line (1D command).
    - psi_d : desired heading (rad).
    - dt : time step in seconds.
    - pr : produced reference distance in meters.
    - vr : produced reference speed in m/s.
    - rr : produced reference yaw-rate in rad/s.
    - psi_r : produced reference heading in rad.
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
