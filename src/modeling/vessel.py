import math
import agx
import agxCollide
import agxSDK
from agx_wrap.world import colorize_body
from agxPythonModules.utils.environment import simulation


class TwoThrusterVessel(agxSDK.Assembly):
  """
  A simple vessel model with two stern azimuth (left & right)
  thrusters for dynamic positioning demos. This is represented by force application
  points in the body-local frame.
  """
  
  def __init__(self,
               mass_kg = 50_00.0,
               half_length = 10.0,
               half_width = 3.75,
               half_height = 0.9,
               cm_shift_x = -0.2,
               thruster_z_offset = -2.5,
               stern_x_offset = None,
               color = (1.0, 1.0, 0.8, 1.0)):
              
    super().__init__()
    
    # Create main rigid hull 
    self.hull = agx.RigidBody("vessel_hull")
    self.add(self.hull)
    
    # Hull geometry: rectangular base + rounded sides + small top box
    base = agxCollide.Geometry(agxCollide.Box(half_length, half_width, half_height))
    
    side_r = half_height * 1.1
    side_len = (half_length * 2) - 2 * side_r
    left_c = agxCollide.Geometry(agxCollide.Cylinder(side_r, side_len))
    left_c.setRotation(agx.Quat(math.pi * 0.5, agx.Vec3.Z_AXIS()))
    left_c.setPosition(0, half_width - side_r, -(half_height + side_r))
    
    right_c = agxCollide.Geometry(agxCollide.Cylinder(side_r, side_len))
    right_c.setRotation(agx.Quat(math.pi * 0.5, agx.Vec3.Z_AXIS()))
    right_c.setPosition(0, side_r - half_width, -(half_height + side_r))
    
    top = agxCollide.Geometry(agxCollide.Capsule(side_r, side_len))
    top.setPosition(-0.4, 0, 2 * half_height)
    
    self.hull.add(base)
    self.hull.add(left_c)
    self.hull.add(right_c)
    
    # Mass and Center Of Mass
    mp = self.hull.getMassProperties()
    mp.setMass(mass_kg)
    self.hull.getCmFrame().setLocalTranslate(agx.Vec3(cm_shift_x, 0, 0))
    
    # Thruster local positioning (at stern, port/starboard)
    stern_x = -half_length if stern_x_offset is None else stern_x_offset
    self.thruster_port_local = agx.Vec3(stern_x, +half_width - side_r, thruster_z_offset)
    self.thruster_star_local = agx.Vec3(stern_x, -half_width + side_r, thruster_z_offset)
    
    colorize_body(self.hull, color)
    simulation().add(self)
    
  # State getters
  def get_world_pose(self):
    T = self.hull.getTransform()
    p = T.getTranslate()
    R = self.hull.getRotation()
    return p, R
  
  def get_xy_psi(self):
    p, R = self.get_world_pose()
    x, y = float(p.x()), float(p.y())
    # Yaw angle (psi) from body frame (z-up in AGX: yaw around z)
    # Extract yaw from rotation matrix: atan2(R[1,0], R[0,0])
    fwd = R * agx.Vec3(1, 0, 0)
    yaw = math.atan2(float(fwd.y()), float(fwd.x()))
    return x, y, yaw
  
  # Force application
  def _body_to_world_vec2(self, fx, fy):
    # Map body (x,y) to world using current yaw
    _, R = self.get_world_pose()
    v_body = agx.Vec3(fx, fy, 0.0)
    v_world = R * v_body
    return v_world
  
  def apply_thruster_forces(self, fx1, fy1, fx2, fy2):
    """
    Apply body-frame forces for each thruster at its local position.
    Converts forces to world and uses addForceAtLocalPosition. 
    """
    
    Fw1 = self._body_to_world_vec2(fx1, fy1)
    Fw2 = self._body_to_world_vec2(fx2, fy2)
    self.hull.addForceAtLocalPosition(Fw1, self.thruster_port_local)
    self.hull.addForceAtLocalPosition(Fw2, self.thruster_star_local)