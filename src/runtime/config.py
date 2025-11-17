# runtime/config.py
from dataclasses import dataclass

@dataclass
class VesselParams:
    mass: float = 50_000.0
    Iz:   float = 1.2e7      # rough yaw inertia
    Xu:   float = 40_000.0   # linear damping (surge)
    Yv:   float = 50_000.0   # linear damping (sway)
    Nr:   float = 6.0e6      # yaw damping

@dataclass
class SceneParams:
    wave_height: float = 1.5

@dataclass
class DPRoute:
    # Waypoint pair in world meters (X-east, Y-north convention in AGX plane)
    start_xy: tuple = (0.0, 0.0)
    goal_xy:  tuple = (80.0,  40.0)   # move diagonally
    psi_d:    float = 0.0             # desired final heading (rad)

@dataclass
class GNSSNoise:
    sigma_pos: float = 0.30  # meters
    sigma_psi: float = 0.01  # rad

# Aggregate config (could be YAML in bigger projects)
vessel = VesselParams()
scene  = SceneParams()
route  = DPRoute()
gnss   = GNSSNoise()
