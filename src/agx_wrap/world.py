# agx_wrap/world.py
import math
import numpy as np
import agx
import agxSDK
import agxCollide
import agxOSG
import agxRender
import agxModel

from agxPythonModules.utils.environment import simulation, root, application
from agxPythonModules.utils.callbacks import StepEventCallback as Sec
from agxPythonModules.utils.numpy_utils import wrap_vector_as_numpy_array


def _color(node, rgba):
    agxOSG.setDiffuseColor(node, agx.Vec4f(*rgba))
    agxOSG.setAmbientColor(node, agx.Vec4f(1))
    agxOSG.setSpecularColor(node, agx.Vec4f(1))
    agxOSG.setShininess(node, 120)


def create_ocean(height=1.5, res_xy=(50, 50), size_xy=(250, 250), cell_h=20.0):
    """
    Builds animated heightfield-water + hydrodynamics registration (WindAndWaterController).
    Returns (hf, water_geom, wwc, wave_updater).
    """
    # Heightfield
    rx, ry = res_xy
    sx, sy = size_xy
    hf = agxCollide.HeightField(rx, ry, sx, sy, cell_h)
    water_geom = agxCollide.Geometry(hf)
    water_geom.setMaterial(agx.Material("waterMaterial"))
    node = agxOSG.createVisual(water_geom, root())
    _color(node, (0.0, 0.74902, 1.0, 0.5))

    simulation().add(water_geom)

    # Hydrodynamics controller (water registration)
    wwc = agxModel.WindAndWaterController()
    wwc.addWater(water_geom)
    simulation().add(wwc)

    # Pre-allocate heights vector and wrap as numpy
    heights = agx.RealVector(hf.getResolutionX() * hf.getResolutionY())
    for _ in range(hf.getResolutionX() * hf.getResolutionY()):
        heights.append(0.0)

    np_heights = wrap_vector_as_numpy_array(heights, np.float64).reshape(
        (hf.getResolutionX(), hf.getResolutionY())
    )

    # Build indices once
    jj = np.stack((np.arange(hf.getResolutionY()),) * hf.getResolutionX())
    ii = np.stack((np.arange(hf.getResolutionX()),) * hf.getResolutionY(), axis=1)

    amp = float(height)

    def wave_function(t: float):
        # Two-component traveling wave (tunable):
        np_heights[:] = amp * (
            0.40 * np.sin(1.00 * jj + 0.60 * t) +
            0.10 * np.sin(1.20 * ii + 0.60 * jj + 1.45 * t)
        )
        hf.setHeights(heights)

    # Hook wave as PRE_COLLIDE so contacts see updated surface
    Sec.preCollideCallback(lambda t: wave_function(t))
    wave_function(0.0)

    # Camera (optional nice default)
    cam = application().getCameraData()
    cam.eye = agx.Vec3(31.1, -87.59, 44.283)
    cam.center = agx.Vec3(9.57, 2.147, 6.95)
    cam.up = agx.Vec3(-0.039, 0.375, 0.925)
    cam.nearClippingPlane = 0.1
    cam.farClippingPlane = 5000
    application().applyCameraData(cam)

    return hf, water_geom, wwc


def colorize_body(rb: agx.RigidBody, rgba=(1.0, 1.0, 0.8, 1.0)):
    node = agxOSG.createVisual(rb, root())
    _color(node, rgba)
    
    
