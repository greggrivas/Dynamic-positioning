# runtime/config.py
from dataclasses import dataclass

"""
Configuration parameters for the dynamic positioning simulation.
These parameters define the vessel properties, scene settings, DP route. 
"""

@dataclass
class VesselParams:
    mass: float = 50_000.0
    Iz:   float = 1.2e7      # rough yaw inertia
    Xu:   float = 40_000.0   # linear damping (surge)
    Yv:   float = 50_000.0   # linear damping (sway)
    Nr:   float = 6.0e6      # yaw damping
    
    half_length: float = 10.0
    half_width: float = 3.75
    half_height: float = 0.9
    cm_shift_x: float = -0.2
    
    thruster_z_offset: float = -2.5
    stern_x_offset: float = None  # default to -half_length
    
    thr_port_x: float = -10.0
    thr_port_y: float = +2.0
    thr_star_x: float = -10.0
    thr_star_y: float = -2.0
    
    Tmax_thruster: float = 80_000.0 # N per thruster
    alloc_bias_Fy: float = 0.0      # N, bias in lateral thruster force allocation

@dataclass
class SceneParams:
    wave_height: float = 1.5
    
    # ref filter
    ref_pos_wn: float = 0.25
    ref_pos_zeta: float = 1.0
    ref_pos_Ki: float = 0.08
    ref_pos_vmax: float = 0.8
    ref_head_wn: float = 0.5
    ref_head_zeta: float = 1.0
    ref_head_Ki: float = 0.2
    ref_head_rmax: float = 0.5
    
    # 
    kp_x: float = 15_000.0
    kp_y: float = 15_000.0
    kp_psi: float = 8_000.0
    kd_x: float = 25_000.0
    kd_y: float = 25_000.0
    kd_psi: float = 12_000.0
    ki_x: float = 500.0
    ki_y: float = 500.0
    ki_psi: float = 200.0
    tau_max: float = 100_000.0
    
    # observer gains
    obs_L_eta: float = 0.5
    obs_L_nu_xy: float = 1.0
    obs_L_nu_psi: float = 1.0
    
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
