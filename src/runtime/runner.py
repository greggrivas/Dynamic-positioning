# runtime/runner.py
import math
import random
from typing import Tuple
import agx
from agxPythonModules.utils.environment import simulation, application
from agxPythonModules.utils.callbacks import StepEventCallback as Sec

from agx_wrap.world import create_ocean
from modeling.vessel import TwoThrusterVessel
from control.reference import ReferenceFilter, PosRefParams, HeadRefParams
from control.observer import SimpleObserver, ObsGains
from control.controller import PIDFFController, PIDGains
from control.allocation import TwoThrusterAllocator, Geometry2Thrusters
from runtime.config import vessel as VCFG, scene as SCFG, route as RCFG, gnss as NCFG


def _angle_of_line(x0: float, y0: float, x1: float, y1: float) -> float:
    return math.atan2(y1 - y0, x1 - x0)

def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

def _world_to_body(psi: float, vx: float, vy: float) -> Tuple[float, float]:
    c, s = math.cos(psi), math.sin(psi)
    # body: u =  c*vx + s*vy, v = -s*vx + c*vy
    return c * vx + s * vy, -s * vx + c * vy


def build_scene_and_start():
    # 1) Ocean with sinus waves (Hydrodynamics registered inside)
    create_ocean(height=SCFG.wave_height)

    # 2) Vessel with two stern thrusters
    ship = TwoThrusterVessel(
        mass_kg=VCFG.mass,
        half_length=VCFG.half_length,
        half_width=VCFG.half_width,
        half_height=VCFG.half_height,
        cm_shift_x=VCFG.cm_shift_x,
        thruster_z_offset=VCFG.thruster_z_offset,
        stern_x_offset=VCFG.stern_x_offset
    )
    ship.setPosition(agx.Vec3(RCFG.start_xy[0], RCFG.start_xy[1], 2.0))
    simulation().add(ship)

    # 3) DP Stack
    # Reference (2nd-order pos filter along the path + heading filter)
    ref = ReferenceFilter(
        pos_params=PosRefParams(
            omega=SCFG.ref_pos_wn, zeta=SCFG.ref_pos_zeta,
            Ki=SCFG.ref_pos_Ki, vmax=SCFG.ref_pos_vmax
        ),
        head_params=HeadRefParams(
            omega=SCFG.ref_head_wn, zeta=SCFG.ref_head_zeta,
            Ki=SCFG.ref_head_Ki, rmax=SCFG.ref_head_rmax
        )
    )
    ref.reset(psi_now=ship.get_xy_psi()[2])

    # Observer
    obs = SimpleObserver(ObsGains(
        L_eta=getattr(SCFG, "obs_L_eta", 1.0),
        L_nu_xy=getattr(SCFG, "obs_L_nu_xy", 1.0),
        L_nu_psi=getattr(SCFG, "obs_L_nu_psi", 1.0)
    ))
    x0, y0, psi0 = ship.get_xy_psi()
    obs.reset(x0, y0, psi0)

    # Controller M (m, m, Iz), D (Xu, Yv, Nr) 3-DOF scope. :contentReference[oaicite:5]{index=5}
    M = [VCFG.mass, VCFG.mass, VCFG.Iz]
    D = [VCFG.Xu,   VCFG.Yv,   VCFG.Nr]
    ctl = PIDFFController(
        M_diag=M, D_diag=D,
        gains=PIDGains(
            Kp_x=getattr(SCFG, "kp_x", SCFG.kp_x if hasattr(SCFG, "kp_x") else 15000.0),
            Kd_x=getattr(SCFG, "kd_x", SCFG.kd_x if hasattr(SCFG, "kd_x") else 25000.0),
            Ki_x=getattr(SCFG, "ki_x", SCFG.ki_x if hasattr(SCFG, "ki_x") else 500.0),
            Kp_y=getattr(SCFG, "kp_y", SCFG.kp_y if hasattr(SCFG, "kp_y") else 15000.0),
            Kd_y=getattr(SCFG, "kd_y", SCFG.kd_y if hasattr(SCFG, "kd_y") else 25000.0),
            Ki_y=getattr(SCFG, "ki_y", SCFG.ki_y if hasattr(SCFG, "ki_y") else 500.0),
            Kp_psi=getattr(SCFG, "kp_psi", SCFG.kp_psi if hasattr(SCFG, "kp_psi") else 8000.0),
            Kd_psi=getattr(SCFG, "kd_psi", SCFG.kd_psi if hasattr(SCFG, "kd_psi") else 12000.0),
            Ki_psi=getattr(SCFG, "ki_psi", SCFG.ki_psi if hasattr(SCFG, "ki_psi") else 200.0),
            tau_max=getattr(SCFG, "tau_max", 100000.0)
        )
    )


    # Allocator geometry (thruster frames wrt CoM, using the same local pts as vessel)
    geom = Geometry2Thrusters(
        lx1=VCFG.thr_port_x,  ly1=VCFG.thr_port_y,
        lx2=VCFG.thr_star_x,  ly2=VCFG.thr_star_y,
        biasFy=VCFG.alloc_bias_Fy
    )
    alloc = TwoThrusterAllocator(geom, Tmax=VCFG.Tmax_thruster)

    # Line geometry & goal distance
    xA, yA = RCFG.start_xy
    xB, yB = RCFG.goal_xy
    phi    = _angle_of_line(xA, yA, xB, yB)       # travel direction (world)
    L_path = math.hypot(xB - xA, yB - yA)         # meters
    
    
    # Text overlays
    sd = application().getSceneDecorator()
    sd.setText(1, "DP: progress & remaining to goal")
    sd.setText(2, "Thrusters [Fx1,Fy1,Fx2,Fy2] (kN)")
    sd.setText(3, "Commanded τ [X,Y,N] (kN, kN, kNm/1000)")

    mode = {"state": "TRANSIT"}  # switch to HOLD at goal
    goal_tol = getattr(SCFG, "goal_tol", 1.0)

    # 4) DP loop
    def dp_step(_time: float):
        dt = simulation().getTimeStep()

        # Measurements
        x, y, psi = ship.get_xy_psi()
        if getattr(NCFG, "disable_noise", False):
            x_m, y_m, psi_m = x, y, psi
        else:
            x_m   = x   + random.gauss(0.0, getattr(NCFG, "sigma_pos", 0.0))
            y_m   = y   + random.gauss(0.0, getattr(NCFG, "sigma_pos", 0.0))
            psi_m = _wrap_pi(psi + random.gauss(0.0, getattr(NCFG, "sigma_psi", 0.0)))

        # Remaining distance along the line
        dN, dE   = (xB - x), (yB - y)
        rem_alng = max(0.0, math.cos(phi) * dN + math.sin(phi) * dE)
        progress = L_path - rem_alng

        # Goal switch
        if mode["state"] == "TRANSIT" and rem_alng <= goal_tol:
            mode["state"] = "HOLD"

        # Reference update
        if mode["state"] == "TRANSIT":
            pr, vr, rr, psir = ref.step(dt, pd=L_path, psi_d=RCFG.psi_d or phi)
            xr = xA + pr * math.cos(phi)
            yr = yA + pr * math.sin(phi)
        else:
            psi_hold = getattr(RCFG, "psi_hold", RCFG.psi_d or phi)
            pr, vr, rr, psir = ref.step(dt, pd=L_path, psi_d=psi_hold)
            xr, yr = xB, yB

        # Map ref velocity to body frame
        ur_world, vr_world = vr * math.cos(phi), vr * math.sin(phi)
        ur_body, vr_body   = _world_to_body(psi, ur_world, vr_world)

        etar   = (xr, yr, psir)
        nur    = (ur_body, vr_body, rr)
        nudotr = (0.0, 0.0, 0.0)

        # Observer
        (xh, yh, psih), (uh, vh, rh) = obs.step(
            dt, meas_x=x_m, meas_y=y_m, meas_psi=psi_m,
            tau_x=0.0, tau_y=0.0, tau_n=0.0,
            M=M, D=D
        )

        # Controller 
        taux, tauy, taun = ctl.step(
            dt,
            eta_r=etar, nu_r=nur, nudot_r=nudotr,
            eta_hat=(xh, yh, psih), nu_hat=(uh, vh, rh)
        )

        # Allocation
        Fx1, Fy1, Fx2, Fy2 = alloc.allocate(taux, tauy, taun)
        ship.apply_thruster_forces(Fx1, Fy1, Fx2, Fy2)

        # HUD
        sd.setText(1, f"Mode {mode['state']} | progress {progress:6.1f}/{L_path:.1f} m | rem {rem_alng:5.1f} m")
        sd.setText(2, f"[{Fx1/1000:6.1f}, {Fy1/1000:6.1f}, {Fx2/1000:6.1f}, {Fy2/1000:6.1f}] kN")
        sd.setText(3, f"[{taux/1000:6.1f}, {tauy/1000:6.1f}, {taun/1000:6.1f}]")

    Sec.preCallback(lambda t: dp_step(t))

    # Camera framing
    cam = application().getCameraData()
    cam.eye    = agx.Vec3(RCFG.start_xy[0] - 30.0, RCFG.start_xy[1] - 80.0, 45.0)
    cam.center = agx.Vec3(RCFG.start_xy[0], RCFG.start_xy[1], 5.0)
    cam.up     = agx.Vec3(0.0, 0.0, 1.0)
    cam.nearClippingPlane = 0.1
    cam.farClippingPlane  = 5000.0
    application().applyCameraData(cam)

    return