# control/allocation.py
import numpy as np
from dataclasses import dataclass

@dataclass
class Geometry2Thrusters:
    lx1: float  # x offset of thruster 1 (port), relative to CoM (m). Negative if behind CoM
    ly1: float  # y offset of thruster 1 (port): positive to starboard is +y (port is +)
    lx2: float  # x offset of thruster 2 (starboard)
    ly2: float  # y offset of thruster 2 (starboard is -)
    biasFy: float = 0.0    # small sway bias to keep jets in a preferred sector (optional)

class TwoThrusterAllocator:
    """
    Linear allocator τ = T f, f = [Fx1, Fy1, Fx2, Fy2]^T with pseudo-inverse,
    plus optional simple bias on lateral forces per tips in the cheat sheet.  :contentReference[oaicite:4]{index=4}
    """
    def __init__(self, geom: Geometry2Thrusters, Tmax=60_000.0):
        self.g = geom
        self.Tmax = float(Tmax)

        self.T = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [self.g.ly1, -self.g.lx1, self.g.ly2, -self.g.lx2]
        ], dtype=float)
        self.Tpinv = self.T.T @ np.linalg.inv(self.T @ self.T.T)

    @staticmethod
    def _sat_symmetric(v, vmax):
        return np.clip(v, -vmax, vmax)

    def allocate(self, tau_x, tau_y, tau_n):
        tau = np.array([tau_x, tau_y, tau_n], dtype=float)

        # Base pseudo-inverse
        f = self.Tpinv @ tau  # [Fx1, Fy1, Fx2, Fy2]

        # Optional simple sway biasing (keeps both jets cooperating)
        if self.g.biasFy != 0.0:
            Fy_sum = f[1] + f[3]
            if Fy_sum > 0:
                f[1] = Fy_sum + self.g.biasFy  # port
                f[3] = self.g.biasFy           # starboard
            elif Fy_sum < 0:
                f[1] = self.g.biasFy           # port
                f[3] = Fy_sum - self.g.biasFy  # starboard

        # Saturate thrust magnitudes per-axis (very simple box saturation)
        f = self._sat_symmetric(f, self.Tmax)
        return float(f[0]), float(f[1]), float(f[2]), float(f[3])
