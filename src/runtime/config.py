from dataclasses import dataclass

@dataclass
class VesselParams:
    mass: float = 50_000.0
    Iz:   float = 2.0e6
    Xu:   float = 40_000.0
    Yv:   float = 50_000.0
    Nr:   float = 500_000.0
    
    half_length: float = 10.0
    half_width: float = 3.75
    half_height: float = 0.9
    cm_shift_x: float = -0.2
    
    thruster_z_offset: float = -2.5
    stern_x_offset: float = None
    
    thr_port_x: float = -10.0
    thr_port_y: float = +2.76
    thr_star_x: float = -10.0
    thr_star_y: float = -2.76
    
    Tmax_thruster: float = 250_000.0  # Increased for rough seas
    alloc_bias_Fy: float = 0.0

@dataclass
class SceneParams:
    wave_height: float = 1.5
    
    # Reference filter
    ref_pos_wn: float = 0.15
    ref_pos_zeta: float = 1.0
    ref_pos_Ki: float = 0.3
    ref_pos_vmax: float = 0.8
    ref_head_wn: float = 0.2
    ref_head_zeta: float = 1.0
    ref_head_Ki: float = 0.2
    ref_head_rmax: float = 0.2
    
    # Gains for PID+FF controller - RETUNED for wave rejection
    kp_x: float = 15_000.0
    kp_y: float = 15_000.0
    kd_x: float = 50_000.0
    kd_y: float = 50_000.0
    ki_x: float = 200.0
    ki_y: float = 200.0
    
    kp_psi: float = 500_000.0
    kd_psi: float = 2_000_000.0
    ki_psi: float = 1_000.0
    
    tau_max: float = 150_000.0
    
    obs_L_eta: float = 0.3
    obs_L_nu_xy: float = 0.5
    obs_L_nu_psi: float = 0.5
    obs_filter_alpha: float = 0.1  # Low-pass filter for wave rejection (lower = more filtering)
    goal_tol: float = 2.0
    
    # Wave-adaptive gain scaling
    wave_gain_scale_low: float = 1.5   # multiplier for calm seas (height < 0.5m)
    wave_gain_scale_high: float = 0.5  # multiplier for rough seas (height > 2.0m)

@dataclass
class DPRoute:
    start_xy: tuple = (0.0, 0.0)
    goal_xy:  tuple = (30.0, 15.0)
    psi_d:    float = None

@dataclass
class GNSSNoise:
    sigma_pos: float = 0.30
    sigma_psi: float = 0.01
    disable_noise: bool = True

vessel = VesselParams()
scene  = SceneParams()
route  = DPRoute()
gnss   = GNSSNoise()