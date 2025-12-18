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
    omega: float = 0.3      # ω_r,p
    zeta: float  = 1.0      # ζ_r,p
    Ki: float    = 0.1     # integral gain for xi_p
    vmax: float  = 2.0      # m/s cap

@dataclass
class HeadRefParams:
    omega: float = 0.5
    zeta: float  = 1.0
    Ki: float    = 0.2
    rmax: float  = 0.5      # rad/s cap

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
        # heading channel internal states
        self.psir = 0.0    # heading reference
        self.rr = 0.0      # yaw rate

    @staticmethod
    def _wrap_pi(a):
        return math.atan2(math.sin(a), math.cos(a))

    def reset(self, psi_now=0.0):
        self.pr = 0.0
        self.vr = 0.0
        self.psir = psi_now
        self.rr = 0.0

    def step(self, dt, pd, psi_d):
        # --- Position channel ---
        # Error to goal
        ep = pd - self.pr
        
        # Desired velocity (limited)
        vd = sat(self.pp.Ki * ep, -self.pp.vmax, self.pp.vmax)
        
        # 2nd-order dynamics: accelerate toward vd
        # v̈r = ω^2 (vd - vr) - 2ζω v̇r  (simplified as 1st-order on vr toward vd)
        # Or use full 2nd-order on position:
        # p̈r = ω^2 (pd - pr) - 2ζω ṗr, with velocity limiting
        
        # Compute acceleration
        ar = self.pp.omega**2 * (pd - self.pr) - 2 * self.pp.zeta * self.pp.omega * self.vr
        
        # Update velocity with limit
        self.vr += ar * dt
        self.vr = sat(self.vr, -self.pp.vmax, self.pp.vmax)
        
        # Update position
        self.pr += self.vr * dt
        
        # Clamp pr to not overshoot pd
        if pd >= 0 and self.pr > pd:
            self.pr = pd
            self.vr = 0.0
        elif pd < 0 and self.pr < pd:
            self.pr = pd
            self.vr = 0.0

        # --- Heading channel ---
        epsi = self._wrap_pi(psi_d - self.psir)
        
        # Compute angular acceleration
        alpha = self.hp.omega**2 * epsi - 2 * self.hp.zeta * self.hp.omega * self.rr
        
        # Update yaw rate with limit
        self.rr += alpha * dt
        self.rr = sat(self.rr, -self.hp.rmax, self.hp.rmax)
        
        # Update heading
        self.psir = self._wrap_pi(self.psir + self.rr * dt)

        return self.pr, self.vr, self.rr, self.psir
