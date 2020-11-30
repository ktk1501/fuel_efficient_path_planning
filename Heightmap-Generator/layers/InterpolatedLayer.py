from .RandomValuesLayer import RandomValuesLayer

from guis.InterpolationGUI import *    
from guis.ClampGUI         import *
from utils								 import *   

# Interpolated layer makes a <resolution> spaced
# array of random values, which it then interpolates 
# over to a bigger array using selected interpolation.
class InterpolatedLayer(RandomValuesLayer):
  def __init__(self, name):
    RandomValuesLayer.__init__(self, name)
    self.interpolate_gui  = InterpolationGUI()
    self.clamp_gui        = ClampGUI()         
    
  def setInterpolationMode(self, mode):
    self.interpolate_gui.setInterpolationMode(mode)
 
  def setResolution(self, resolution):
    self.interpolate_gui.setResolution(resolution)

  def setClampEnabled(self, enabled):
    self.clamp_gui.setEnabled(enabled)

  def setClampTopValue(self, top_value):
    self.clamp_gui.setTopValue(top_value)

  def setClampBottomValue(self, bottom_value):
    self.clamp_gui.setBottomValue(bottom_value)

  def layoutGUI(self, parent, heightmap=None):
    RandomValuesLayer.layoutGUI(self, parent, heightmap)
    self.interpolate_gui.layoutGUI(parent, heightmap)
    self.clamp_gui.layoutGUI(parent, heightmap)
    
  def makeHeights(self, stack, cumulative):
    res = self.interpolate_gui.getResolution()
    initial_array = Array2D(res, res)
    seed = self.seed_gui.getSeed()
    setSeedNumber(seed)
    initial_array.each(self.randomizeArrayElement)    
    
    interoplator = self.interpolate_gui.createInterpolator()
    w = cumulative.getWidth()
    h = cumulative.getHeight()
    interpolated = initial_array.makeInterpolated(w,h, interoplator)
    
    interpolated.each(self.roundArrayValues)
    
    is_clamped = self.clamp_gui.isEnabled()
    if is_clamped:
      interpolated.each(self.clampArrayValues)
      
    #print "Applying layer " + self.name + "| Mode " + self.mode.getName()
    #print interpolated


    return interpolated
   

  def randomizeArrayElement(self, arr2d, x, y, value):
    minimum = self.range_gui.getMinimum()
    maximum = self.range_gui.getMaximum()
    return getNextSeedInt(minimum, maximum)

  def roundArrayValues(self, arr2d, x, y, value):
    return round(value)
  
  def clampArrayValues(self, arr2d, x, y, value):
    top = self.clamp_gui.getTopValue()
    btm = self.clamp_gui.getBottomValue()
    
    if value > top:
      return top
    elif value < btm:
      return btm
    else: return value
    
    
  def getTypeName(self):
    return "Interpolated layer "
    
  def getTypeDescription(self):
    return \
      "Sine interpolated layer makes a <resolution> spaced \
      array of random values, which it then interpolates \
      over to a bigger array using sinusoidal interpolation."

  def copy(self, name):
    return InterpolatedLayer(name)

  def duplicate(self, name):
    layer = InterpolatedLayer(name)
    layer.interpolate_gui = self.interpolate_gui.duplicate()
    layer.clamp_gui       = self.clamp_gui.duplicate()
    return layer
  
  def __str__(self):
    seed = str(self.seed_gui.getSeed())
    minimum_range = str(self.range_gui.getMinimum())
    maximum_range = str(self.range_gui.getMaximum())
    mode = str(self.interpolate_gui.getInterpolationMode())
    resolution = str(self.interpolate_gui.getResolution())
    clamp_enabled = str(self.clamp_gui.isEnabled())
    clamp_min = str(self.clamp_gui.getBottomValue())
    clamp_max = str(self.clamp_gui.getTopValue())
  
    return "InterpolatedLayer"  " " + seed + " " \
      + minimum_range + " " + maximum_range + " " \
      + mode + " " + resolution + " " \
      + clamp_enabled + " " + clamp_min + " " + clamp_max  
