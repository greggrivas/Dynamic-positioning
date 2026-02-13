import numpy as np
# AGX Dynamics imports
import agx
import agxPython
import agxCollide
import agxIO
import agxOSG
import agxSDK
import agxTerrain
import agxRender
import agxUtil
import agxModel
import agxSensor
import agxCable
import agxWire
import os

from agxPythonModules.utils.environment import simulation, root, application, init_app
from agxPythonModules.utils.callbacks import StepEventCallback as Sec
from agxPythonModules.utils.callbacks import KeyboardCallback as Input
from agxPythonModules.utils.numpy_utils import wrap_vector_as_numpy_array

import math

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ship_shape_name = os.path.join(_THIS_DIR, "..", "..", "assets", "Gunnerus.obj")


def createTrimesh(filename: str, scale) -> agxCollide.Trimesh:
  """
  Create a trimesh from a specified mesh file
  """
  mesh = agxUtil.createTrimesh(
    filename,
    agxCollide.Trimesh.REMOVE_DUPLICATE_VERTICES,
    agx.Matrix3x3(agx.Vec3(scale)))
  assert mesh
  return mesh

class Ship(agxSDK.Assembly):
  def __init__(self,
                mass_kg=350_000.0,
                cm_shift_x=-0.2,
                thruster_z_offset=-2.5,
                thr_port_x=-10.0,
                thr_port_y=+2.76,
                thr_star_x=-10.0,
                thr_star_y=-2.76,
                shipColor=agxRender.Color.LightYellow(),
                 # Unused but accepted for config compatibility
                half_length=None,
                half_width=None,
                half_height=None,
                stern_x_offset=None,
                color=None):
    super().__init__()
    
    # Create the ship
    ship_material = agx.Material("steel")
    
    ship_shape = createTrimesh(ship_shape_name, 1.0)
    assert (ship_shape)
    ship_geom = agxCollide.Geometry(ship_shape)
    ship_geom.setMaterial(ship_material)
    
    self.ship_body = agx.RigidBody(ship_geom)
    self.ship_body.getMassProperties().setMass(mass_kg) # 350t
    #ship_body.setMotionControl(agx.RigidBody.STATIC)
    
    self.ship_body.setPosition(agx.Vec3(0, 0, 0))
    # Rotate 90 degrees around X and -90 around Z
    self.ship_body.setRotation(agx.EulerAngles(math.radians(90), 0, math.radians(-90)))
    self.add(self.ship_body)
    
    #Observer frame for reference
    self.f_observer = agx.ObserverFrame("ship observer", self.ship_body,
                                        agx.AffineMatrix4x4.translate(agx.Vec3(1, 0, 0)))
    #f_ob = simulation().getObserverFrame("ship observer")
    
    self.add(self.f_observer)
    
    # Set Center of Mass shift by moving the visual geometry relative to the body frame
    self.ship_body.getCmFrame().setLocalTranslate(agx.Vec3(cm_shift_x, 0, 0))
    
    # Thruster positions in body frame
    self.thruster_port_local = agx.Vec3(thr_port_x, thr_port_y, thruster_z_offset)
    self.thruster_star_local = agx.Vec3(thr_star_x, thr_star_y, thruster_z_offset)
    
    # Storing the hull reference
    self.hull = self.ship_body
    
    agxOSG.setDiffuseColor(agxOSG.createVisual(self.ship_body, root()), shipColor)
    print("Ship created | observer frame position: ", self.f_observer.getPosition())
    
  def get_ship_observer(self):
    return self.f_observer.getPosition()
  
  def get_world_pose(self):
    T = self.ship_body.getTransform()
    return T.getTranslate(), self.ship_body.getRotation()
  
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
    
    cm_off = self.ship_body.getCmFrame().getLocalTranslate()
    p1_cm = self.thruster_port_local - cm_off
    p2_cm = self.thruster_star_local - cm_off
    
    self.ship_body.addForceAtLocalCmPosition(agx.Vec3(fx1, fy1, 0.0), p1_cm)
    self.ship_body.addForceAtLocalCmPosition(agx.Vec3(fx2, fy2, 0.0), p2_cm)
  
class surfaceWater():
  def __init__(self, sim, wwc, length, width):
    
    self.water_hf = agxCollide.HeightField(50, 50, length, width, 20)
    self.heights = agx.RealVector(self.water_hf.getResolutionX() * self.water_hf.getResolutionY())
    for i in range(self.water_hf.getResolutionX() * self.water_hf.getResolutionY()):
      self.heights.append(0)
      
    self.waterSurfaceGeometry = agxCollide.Geometry(self.water_hf)
    self.waterSurfaceGeometry.setPosition(0, 0, 0)
    self.waterSurfaceMaterial = agx.Material("waterSurfaceMaterial")
    self.waterSurfaceGeometry.setMaterial(self.waterSurfaceMaterial)
    
    node = self.createWaterVisual(self.waterSurfaceGeometry)
    sim.add(self.waterSurfaceGeometry)
    self.waveFunction(0)
    
  def createWaterVisual(self, geom):
    node = agxOSG.createVisual(geom, root())
    
    agxOSG.setDiffuseColor(node, agx.Vec4f(0, 0.749020, 1, 1))
    agxOSG.setAmbientColor(node, agx.Vec4f(1))
    agxOSG.setSpecularColor(node, agx.Vec4f(1))
    agxOSG.setShininess(node, 120)
    agxOSG.setAlpha(node, 0.7)
    return node
  
  def waveFunction(self, t):
    np_heights = wrap_vector_as_numpy_array(self.heights, float).reshape(
      (self.water_hf.getResolutionX(), self.water_hf.getResolutionY()))
    
    jj = np.stack((np.arange(self.water_hf.getResolutionY()),) * self.water_hf.getResolutionX())
    ii = np.stack((np.arange(self.water_hf.getResolutionX()),) * self.water_hf.getResolutionY(), axis=1)
    waveHeight = 1.5
    
    np_heights[:] = waveHeight * (0.4 * np.sin(1.0 * jj + 0.6 * t) + 0.1 * np.sin(1.2 * ii + 0.6 * jj +1.45 * t))
    self.water_hf.setHeights(self.heights)
    
  def get_geom(self):
    return self.waterSurfaceGeometry


def buildScene1():
    application().getSceneDecorator().setEnableShadows(False)
    application().setEnableDebugRenderer(True)

    my_ship = Ship()
    my_ship.setPosition(0, 0, 3)
    simulation().add(my_ship)

    # Create surface water
    wwc = agxModel.WindAndWaterController()
    surf_water = surfaceWater(simulation(), wwc, 100, 100)
    wwc.addWater(surf_water.get_geom())
    simulation().add(wwc)

    Sec.preCollideCallback(lambda t: surf_water.waveFunction(t))


def buildScene():
    '''
    Entry point when this script is started with agxViewer
    '''
    buildScene1()


# Entry point when this script is started with python executable
init = init_app(name=__name__,
                scenes=[
                    (buildScene, '1')
                ],
                autoStepping=False)

TwoThrusterVessel = Ship  # Alias for compatibility with config parameters