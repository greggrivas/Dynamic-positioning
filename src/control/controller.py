# control/controller.py
import math
from dataclasses import dataclass

@dataclass
class PIDGains:
    Kp_x: float = 5_000.0
    Kp_y: float = 5_000.0
    Kp_psi: float = 2_000.0
    Kd_x: float = 10_000.0
    Kd_y: float = 10_000.0
    Kd_psi: float = 5_000.0
    Ki_x: float = 0.0
    Ki_y: float = 0.0
    Ki_psi: float = 0.0
    tau_max: float = 80_000.0  # generic saturation

class PIDFFController:
    """
    τ = M ν̇r + D νr + Kp e_p + Kd e_ν + σ  (with simple anti-windup saturation).  :contentReference[oaicite:3]{index=3}
    """
    def __init__(self, M_diag, D_diag, gains: PIDGains):
        self.M = M_diag  # [m, m, Iz]
        self.D = D_diag  # [Xu, Yv, Nr]
        self.g = gains
        self.sigma_x = 0.0; self.sigma_y = 0.0; self.sigma_psi = 0.0

    @staticmethod
    def _wrap_pi(a):
        while a > math.pi:  a -= 2*math.pi
        while a < -math.pi: a += 2*math.pi
        return a

    def step(self, dt, eta_r, nu_r, nudot_r, eta_hat, nu_hat):
        xr, yr, psir = eta_r
        ur, vr, rr = nu_r
        udr, vdr, rdr = nudot_r

        xh, yh, psih = eta_hat
        uh, vh, rh = nu_hat

        # Position error in WORLD frame here (we keep it simple & consistent with runner’s transforms)
        ex = xr - xh
        ey = yr - yh
        epsi = self._wrap_pi(psir - psih)

        # Velocity error (body frame reference is mapped by runner already)
        eu = ur - uh
        ev = vr - vh
        er = rr - rh

        # Feedforward
        tau_ff_x   = self.M[0]*udr + self.D[0]*ur
        tau_ff_y   = self.M[1]*vdr + self.D[1]*vr
        tau_ff_psi = self.M[2]*rdr + self.D[2]*rr

        # PD + I
        tau_x   = tau_ff_x   + self.g.Kp_x*ex + self.g.Kd_x*eu   + self.sigma_x
        tau_y   = tau_ff_y   + self.g.Kp_y*ey + self.g.Kd_y*ev   + self.sigma_y
        tau_psi = tau_ff_psi + self.g.Kp_psi*epsi + self.g.Kd_psi*er + self.sigma_psi

        # Saturate (very coarse vector clamp)
        def sat(v, lim): return max(-lim, min(lim, v))
        tau_x_s   = sat(tau_x,   self.g.tau_max)
        tau_y_s   = sat(tau_y,   self.g.tau_max)
        tau_psi_s = sat(tau_psi, self.g.tau_max)

        # Anti-windup (freeze when saturated in same sign direction)
        if abs(tau_x_s - tau_x) < 1e-9:
            self.sigma_x += self.g.Ki_x * ex * dt
        if abs(tau_y_s - tau_y) < 1e-9:
            self.sigma_y += self.g.Ki_y * ey * dt
        if abs(tau_psi_s - tau_psi) < 1e-9:
            self.sigma_psi += self.g.Ki_psi * epsi * dt

        return tau_x_s, tau_y_s, tau_psi_s
