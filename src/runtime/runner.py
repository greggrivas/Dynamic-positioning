# runtime/runner.py
import math
import numpy as np
import agx
import agxRender
from agxPythonModules.utils.environment import simulation, application, root
from agxPythonModules.utils.callbacks import StepEventCallback as Sec

from agx_wrap.world import create_ocean
from modeling.vessel import TwoThrusterVessel
from control.reference import ReferenceFilter, PosRefParams, HeadRefParams
from control.observer import SimpleObserver, ObsGains
from control.controller import PIDFFController, PIDGains
from control.allocation import TwoThrusterAllocator, Geometry2Thrusters
from runtime.config import vessel as VCFG, scene as SCFG, route as RCFG, gnss as NCFG


def _angle_of_line(x0, y0, x1, y1):
    return math.atan2((y1 - y0), (x1 - x0))

def _wrap_pi(a):
    while a > math.pi:  a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def _world_to_body(psi, vx, vy):
    c, s = math.cos(psi), math.sin(psi)
    # body u =  c*vx + s*vy, body v = -s*vx + c*vy
    return c*vx + s*vy, -s*vx + c*vy

def _body_to_world(psi, u, v):
    c, s = math.cos(psi), math.sin(psi)
    return c*u - s*v, s*u + c*v


def build_scene_and_start():
    # 1) World water with sinus waves (Hydrodynamics registered)
    create_ocean(height=SCFG.wave_height)

    # 2) Vessel with two stern thrusters
    ship = TwoThrusterVessel(
        mass_kg=VCFG.mass,
        half_length=10.0,
        half_width=3.75,
        half_height=0.9,
        cm_shift_x=-0.2,
        thruster_z_offset=-(0.9 + 2*0.99),  # below hull bottom
        stern_x_offset=-10.0
    )
    ship.setPosition(agx.Vec3(RCFG.start_xy[0], RCFG.start_xy[1], 2.0))
    simulation().add(ship)

    # 3) DP Stack
    # Reference
    ref = ReferenceFilter(
        pos_params=PosRefParams(omega=0.25, zeta=1.0, Ki=0.08, vmax=0.8),
        head_params=HeadRefParams(omega=0.5, zeta=1.0, Ki=0.2, rmax=0.5)
    )
    ref.reset(psi_now=ship.get_xy_psi()[2])

    # Observer
    obs = SimpleObserver(ObsGains(L_eta=1.0, L_nu_xy=0.8, L_nu_psi=0.8))
    x0, y0, psi0 = ship.get_xy_psi()
    obs.reset(x0, y0, psi0)

    # Controller M (m, m, Iz), D (Xu, Yv, Nr) 3-DOF scope. :contentReference[oaicite:5]{index=5}
    M = [VCFG.mass, VCFG.mass, VCFG.Iz]
    D = [VCFG.Xu,   VCFG.Yv,   VCFG.Nr]
    ctl = PIDFFController(M_diag=M, D_diag=D, gains=PIDGains(
        Kp_x=6e3, Kp_y=6e3, Kp_psi=3e3, Kd_x=12e3, Kd_y=12e3, Kd_psi=6e3, Ki_x=0.0, Ki_y=0.0, Ki_psi=0.0, tau_max=80_000.0
    ))

    # Allocator geometry (thruster frames wrt CoM, using the same local pts as vessel)
    geom = Geometry2Thrusters(
        lx1=-10.0, ly1=+2.0,   # port
        lx2=-10.0, ly2=-2.0,   # starboard
        biasFy=0.0
    )
    alloc = TwoThrusterAllocator(geom, Tmax=60_000.0)

    # Line geometry & goal distance
    xA, yA = RCFG.start_xy
    xB, yB = RCFG.goal_xy
    phi = _angle_of_line(xA, yA, xB, yB)        # direction of travel (world)
    dist_goal = math.hypot(xB - xA, yB - yA)    # meters

    # Text overlays
    sd = application().getSceneDecorator()
    sd.setText(1, "DP Reference: dist -> goal")
    sd.setText(2, "Thrusts [Fx1,Fy1,Fx2,Fy2] (kN)")
    sd.setText(3, "tau [X,Y,N] (kN, kN, kNm/1000)")

    # 4) DP loop
    def dp_step(_time: float):
        dt = simulation().getTimeStep()

        # --- Measurements (GNSS with noise) ---
        x, y, psi = ship.get_xy_psi()
        meas_x = x + np.random.normal(0.0, NCFG.sigma_pos)
        meas_y = y + np.random.normal(0.0, NCFG.sigma_pos)
        meas_psi = _wrap_pi(psi + np.random.normal(0.0, NCFG.sigma_psi))

        # --- Reference update (1D along line + heading) ---
        # Remaining distance along the line:
        dN = (xB - x); dE = (yB - y)
        rem = math.cos(phi)*dN + math.sin(phi)*dE
        rem = max(0.0, rem)  # do not request negative distance
        pr, vr, rr, psir = ref.step(dt, pd=dist_goal, psi_d=RCFG.psi_d)

        # Compose ηr (world) on the line from A toward B:
        xr = xA + pr*math.cos(phi)
        yr = yA + pr*math.sin(phi)
        etar = (xr, yr, psir)

        # Desired velocities/accelerations in world (straight-line)
        # Use simple 1D signals mapped to 2D with φ:
        ur_world, vr_world = (vr*math.cos(phi), vr*math.sin(phi))
        # Convert to BODY references for controller (so controller eν is in body frame):
        ur, vr_body = _world_to_body(psi, ur_world, vr_world)
        # Simple derivative approximations: we just use proportional to vr dynamics for demo
        udr_world, vdr_world = (0.0, 0.0)  # could be refined
        rdr = 0.0
        nudotr = (udr_world, vdr_world, rdr)
        nur = (ur, vr_body, rr)

        # --- Observer ---
        # Pass current control guess (last τ used is unknown here; we feed 0 for simplicity)
        tau_x, tau_y, tau_n = (0.0, 0.0, 0.0)
        (xh, yh, psih), (uh, vh, rh) = obs.step(dt,
                                               meas_x, meas_y, meas_psi,
                                               tau_x, tau_y, tau_n,
                                               M=M, D=D)

        # --- Controller ---
        taux, tauy, taun = ctl.step(dt,
                                    eta_r=etar,
                                    nu_r=nur,
                                    nudot_r=nudotr,
                                    eta_hat=(xh, yh, psih),
                                    nu_hat=(uh, vh, rh))

        # --- Allocation ---
        Fx1, Fy1, Fx2, Fy2 = alloc.allocate(taux, tauy, taun)

        # --- Apply to vessel (per-thruster forces at local positions) ---
        ship.apply_thruster_forces(Fx1, Fy1, Fx2, Fy2)

        # HUD
        sd.setText(1, f"Ref progress: {pr:6.1f}/{dist_goal:.1f} m  |  rem~{rem:5.1f} m")
        sd.setText(2, f"[{Fx1/1000:6.1f}, {Fy1/1000:6.1f}, {Fx2/1000:6.1f}, {Fy2/1000:6.1f}] kN")
        sd.setText(3, f"[{taux/1000:6.1f}, {tauy/1000:6.1f}, {taun/1000:6.1f}]")

    # Register as pre-step (forces applied before integration)
    Sec.preCallback(lambda t: dp_step(t))

    # Focus camera text
    return
