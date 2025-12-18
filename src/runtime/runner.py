import math
import random
import csv, os, atexit
import numpy as np
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

# CSV logger
log_path = os.path.join(os.path.dirname(__file__), "dp_log.csv")
_log_file = open(log_path, "w", newline="")
_log_writer = csv.writer(_log_file)
_log_writer.writerow(["t","x","y","psi","xr","yr","psir","ex","ey","epsi","tau_x","tau_y","tau_psi","Fx1","Fy1","Fx2","Fy2"])

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
    return c * vx + s * vy, -s * vx + c * vy

def build_scene_and_start():
    create_ocean(height=SCFG.wave_height)

    ship = TwoThrusterVessel(
        mass_kg=VCFG.mass,
        half_length=VCFG.half_length,
        half_width=VCFG.half_width,
        half_height=VCFG.half_height,
        cm_shift_x=VCFG.cm_shift_x,
        thruster_z_offset=VCFG.thruster_z_offset,
        stern_x_offset=VCFG.stern_x_offset,
        thr_port_x=VCFG.thr_port_x,
        thr_port_y=VCFG.thr_port_y,
        thr_star_x=VCFG.thr_star_x,
        thr_star_y=VCFG.thr_star_y,
    )
    ship.setPosition(agx.Vec3(RCFG.start_xy[0], RCFG.start_xy[1], 2.0))
    simulation().add(ship)

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

    obs = SimpleObserver(ObsGains(
        L_eta=getattr(SCFG, "obs_L_eta", 1.0),
        L_nu_xy=getattr(SCFG, "obs_L_nu_xy", 1.0),
        L_nu_psi=getattr(SCFG, "obs_L_nu_psi", 1.0)
    ))
    x0, y0, psi0 = ship.get_xy_psi()
    obs.reset(x0, y0, psi0)

    M = [VCFG.mass, VCFG.mass, VCFG.Iz]
    D = [VCFG.Xu,   VCFG.Yv,   VCFG.Nr]
    ctl = PIDFFController(
        M_diag=M, D_diag=D,
        gains=PIDGains(
            Kp_x=SCFG.kp_x, Kd_x=SCFG.kd_x, Ki_x=SCFG.ki_x,
            Kp_y=SCFG.kp_y, Kd_y=SCFG.kd_y, Ki_y=SCFG.ki_y,
            Kp_psi=SCFG.kp_psi, Kd_psi=SCFG.kd_psi, Ki_psi=SCFG.ki_psi,
            tau_max=SCFG.tau_max
        )
    )

    # Allocator geometry from vessel thruster points
    lx1, ly1 = float(ship.thruster_port_local.x()), float(ship.thruster_port_local.y())
    lx2, ly2 = float(ship.thruster_star_local.x()), float(ship.thruster_star_local.y())
    geom = Geometry2Thrusters(lx1=lx1, ly1=ly1, lx2=lx2, ly2=ly2, biasFy=VCFG.alloc_bias_Fy)
    alloc = TwoThrusterAllocator(geom, Tmax=VCFG.Tmax_thruster)

    xA, yA = RCFG.start_xy
    xB, yB = RCFG.goal_xy
    phi    = _angle_of_line(xA, yA, xB, yB)
    L_path = math.hypot(xB - xA, yB - yA)

    sd = application().getSceneDecorator()
    sd.setText(1, "DP: progress & remaining to goal")
    sd.setText(2, "Thrusters [Fx1,Fy1,Fx2,Fy2] (kN)")
    sd.setText(3, "Commanded τ [X,Y,N] (kN, kN, kNm/1000)")

    mode = {"state": "TRANSIT"}
    goal_tol = getattr(SCFG, "goal_tol", 1.0)
    last_tau = (0.0, 0.0, 0.0)
    t_sim = 0.0

    def dp_step(_time: float):
        nonlocal last_tau, t_sim
        dt = simulation().getTimeStep()
        t_sim += dt

        x, y, psi = ship.get_xy_psi()
        if getattr(NCFG, "disable_noise", False):
            x_m, y_m, psi_m = x, y, psi
        else:
            x_m   = x   + random.gauss(0.0, getattr(NCFG, "sigma_pos", 0.0))
            y_m   = y   + random.gauss(0.0, getattr(NCFG, "sigma_pos", 0.0))
            psi_m = _wrap_pi(psi + random.gauss(0.0, getattr(NCFG, "sigma_psi", 0.0)))

        dN, dE   = (xB - x), (yB - y)
        rem_alng = max(0.0, math.cos(phi) * dN + math.sin(phi) * dE)
        progress = L_path - rem_alng

        if mode["state"] == "TRANSIT" and rem_alng <= goal_tol:
            mode["state"] = "HOLD"

        if mode["state"] == "TRANSIT":
            pr, vr, rr, psir = ref.step(dt, pd=L_path, psi_d=RCFG.psi_d or phi)
            xr = xA + pr * math.cos(phi)
            yr = yA + pr * math.sin(phi)
        else:
            psi_hold = getattr(RCFG, "psi_hold", RCFG.psi_d or phi)
            pr, vr, rr, psir = ref.step(dt, pd=L_path, psi_d=psi_hold)
            xr, yr = xB, yB

        ur_world, vr_world = vr * math.cos(phi), vr * math.sin(phi)
        ur_body, vr_body   = _world_to_body(psi, ur_world, vr_world)

        etar   = (xr, yr, psir)
        nur    = (ur_body, vr_body, rr)
        nudotr = (0.0, 0.0, 0.0)

        # Observer uses last applied wrench
        (xh, yh, psih), (uh, vh, rh) = obs.step(
            dt,
            meas_x=x_m, meas_y=y_m, meas_psi=psi_m,
            tau_x=last_tau[0], tau_y=last_tau[1], tau_n=last_tau[2],
            M=M, D=D
        )

        taux, tauy, taun = ctl.step(
            dt,
            eta_r=etar, nu_r=nur, nudot_r=nudotr,
            eta_hat=(xh, yh, psih), nu_hat=(uh, vh, rh)
        )

        Fx1, Fy1, Fx2, Fy2 = alloc.allocate(taux, tauy, taun)
        ship.apply_thruster_forces(Fx1, Fy1, Fx2, Fy2)

        # Effective applied wrench
        f_vec = np.array([Fx1, Fy1, Fx2, Fy2], dtype=float)
        tau_eff = alloc.T @ f_vec
        tau_eff_x, tau_eff_y, tau_eff_n = float(tau_eff[0]), float(tau_eff[1]), float(tau_eff[2])

        if getattr(VCFG, "enable_drag", False):
            du = uh; dv = vh; dr = rh
            Fx_drag = -(VCFG.drag_lin_u * du + VCFG.drag_quad_u * abs(du) * du)
            Fy_drag = -(VCFG.drag_lin_v * dv + VCFG.drag_quad_v * abs(dv) * dv)
            Nz_drag = -(VCFG.drag_lin_r * dr + VCFG.drag_quad_r * abs(dr) * dr)
            ship.apply_body_drag(Fx_drag, Fy_drag, Nz_drag)

        sd.setText(1, f"Mode {mode['state']} | progress {progress:6.1f}/{L_path:.1f} m | rem {rem_alng:5.1f} m")
        sd.setText(2, f"[{Fx1/1000:6.1f}, {Fy1/1000:6.1f}, {Fx2/1000:6.1f}, {Fy2/1000:6.1f}] kN")
        sd.setText(3, f"[{taux/1000:6.1f}, {tauy/1000:6.1f}, {taun/1000:6.1f}]")

        # Log row
        ex = xr - xh
        ey = yr - yh
        epsi = _wrap_pi(psir - psih)
        _log_writer.writerow([t_sim, xh, yh, psih, xr, yr, psir, ex, ey, epsi, taux, tauy, taun, Fx1, Fy1, Fx2, Fy2])

        last_tau = (tau_eff_x, tau_eff_y, tau_eff_n)

    Sec.preCallback(lambda t: dp_step(t))

    def close_log():
        try:
            _log_file.close()
        except:
            pass
    atexit.register(close_log)

    cam = application().getCameraData()
    cam.eye    = agx.Vec3(RCFG.start_xy[0] - 30.0, RCFG.start_xy[1] - 80.0, 45.0)
    cam.center = agx.Vec3(RCFG.start_xy[0], RCFG.start_xy[1], 5.0)
    cam.up     = agx.Vec3(0.0, 0.0, 1.0)
    cam.nearClippingPlane = 0.1
    cam.farClippingPlane  = 5000.0
    application().applyCameraData(cam)