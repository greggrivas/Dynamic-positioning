import math
import agx
import agxCollide
import agxSDK
from agx_wrap.world import colorize_body
from agxPythonModules.utils.environment import simulation


class TwoThrusterVessel(agxSDK.Assembly):
  def __init__(self,
               mass_kg=50_000.0,
               half_length=10.0,
               half_width=3.75,
               half_height=0.9,
               cm_shift_x=-0.2,
               thruster_z_offset=-2.5,
               stern_x_offset=None,
               thr_port_x=None,
               thr_port_y=None,
               thr_star_x=None,
               thr_star_y=None,
               color=(1.0, 1.0, 0.8, 1.0)):
    super().__init__()
    
    self.hull = agx.RigidBody("vessel_hull")
    self.add(self.hull)
    
    # Hull geometry
    base = agxCollide.Geometry(agxCollide.Box(half_length, half_width, half_height))
    side_r = half_height * 1.1
    side_len = (half_length * 2) - 2 * side_r
    
    left_c = agxCollide.Geometry(agxCollide.Cylinder(side_r, side_len))
    left_c.setRotation(agx.Quat(math.pi * 0.5, agx.Vec3.Z_AXIS()))
    left_c.setPosition(0, half_width - side_r, -(half_height + side_r))
    
    right_c = agxCollide.Geometry(agxCollide.Cylinder(side_r, side_len))
    right_c.setRotation(agx.Quat(math.pi * 0.5, agx.Vec3.Z_AXIS()))
    right_c.setPosition(0, side_r - half_width, -(half_height + side_r))
    
    self.hull.add(base)
    self.hull.add(left_c)
    self.hull.add(right_c)
    
    # Mass and CoM
    mp = self.hull.getMassProperties()
    mp.setMass(mass_kg)
    self.hull.getCmFrame().setLocalTranslate(agx.Vec3(cm_shift_x, 0, 0))
    
    # Thruster positions in body frame
    stern_x = -half_length if stern_x_offset is None else stern_x_offset
    if None not in (thr_port_x, thr_port_y, thr_star_x, thr_star_y):
      self.thruster_port_local = agx.Vec3(thr_port_x, thr_port_y, thruster_z_offset)
      self.thruster_star_local = agx.Vec3(thr_star_x, thr_star_y, thruster_z_offset)
    else:
      self.thruster_port_local = agx.Vec3(stern_x, +half_width - side_r, thruster_z_offset)
      self.thruster_star_local = agx.Vec3(stern_x, -half_width + side_r, thruster_z_offset)
    
    colorize_body(self.hull, color)
    simulation().add(self)
    
  def get_world_pose(self):
    T = self.hull.getTransform()
    return T.getTranslate(), self.hull.getRotation() # self.hull.getRotation() could be written as T.getRotation() for consistency
  
  def get_xy_psi(self):
    p, R = self.get_world_pose()
    x, y = float(p.x()), float(p.y())
    fwd = R * agx.Vec3(1, 0, 0)
    yaw = math.atan2(float(fwd.y()), float(fwd.x()))
    return x, y, yaw
  
  def apply_thruster_forces(self, fx1, fy1, fx2, fy2):
    """
    Apply body-frame forces at thruster points in CoM frame.
    """
    # Get CoM offset to convert body-local to CoM-local
    cm_off = self.hull.getCmFrame().getLocalTranslate()
    p1_cm = self.thruster_port_local - cm_off
    p2_cm = self.thruster_star_local - cm_off
    
    # Apply body-frame forces at CoM-relative positions
    self.hull.addForceAtLocalCmPosition(agx.Vec3(fx1, fy1, 0.0), p1_cm)
    self.hull.addForceAtLocalCmPosition(agx.Vec3(fx2, fy2, 0.0), p2_cm)