from .Layer import *
from utils  import *
from guis.RangeGUI import *
from guis.SeedingGUI  import *
from guis.ScaleGUI import *

class SimplexLayer(Layer):
    def __init__(self, name):
        Layer.__init__(self, name)
        self.seed_gui    = SeedingGUI()                   
        self.range_gui   = RangeGUI()                     
        self.scale_gui   = ScaleGUI("Scale",1,100)        
        self.clamp_gui   = ClampGUI("Clamping",0, 255)    
        
    def layoutGUI(self, parent, heightmap):
        self.seed_gui.layoutGUI(parent, heightmap)
        self.range_gui.layoutGUI(parent, heightmap)
        self.scale_gui.layoutGUI(parent, heightmap)
        self.clamp_gui.layoutGUI(parent, heightmap)
        
    def setSeed(self, seed):
      self.seed_gui.setSeed(seed)

    def setClampEnabled(self, enabled):
      self.clamp_gui.setEnabled(enabled)

    def setClampTopValue(self, top_value):
      self.clamp_gui.setTopValue(top_value)

    def setClampBottomValue(self, bottom_value):
      self.clamp_gui.setBottomValue(bottom_value)
 
    def setMaximumRange(self, maximum):
      self.range_gui.setMaximum(maximum)

    def setMinimumRange(self, minimum):
      self.range_gui.setMinimum(minimum)
    
    def setScale(self, scale):
      self.scale_gui.setValue(scale)

    def makeHeights(self, stack, cumulative):
        self.noise  = SimplexNoise()
        
        w = cumulative.getWidth()
        h = cumulative.getHeight()
        
        arr2d = Array2D(w,h)
        arr2d.each(self.makeNoisy)
        
        #print "Applying layer " + self.name + "| Mode " + self.mode.getName()
        #print arr2d

        return arr2d
        
    def makeNoisy(self, arr2d, x, y, orig): # The original will be 0 here always
        scale = float(self.scale_gui.getValue())
       
        seed = self.seed_gui.getSeed()
          
        offset = seed * arr2d.getWidth()

        val = self.noise.noise2d(x / scale + offset, y / scale)

        minimum = self.range_gui.getMinimum()
        maximum = self.range_gui.getMaximum()
        
        val = mapRange(val, -1, 1, minimum, maximum)

        is_clamping_enabled = self.clamp_gui.isEnabled()

        if is_clamping_enabled:
          
          max_val = self.clamp_gui.getTopValue();
          min_val = self.clamp_gui.getBottomValue();
          
          if not max_val < min_val: # Clamp only when possible
              val = clamp(val, min_val, max_val)
        
        return val
        
    def getTypeName(self):
        return "Simplex noise layer."
    
    def getTypeDescription(self):
      return "Layer whose values are generated using simplex noise algorithm."
    
    def copy(self, name):
      return SimplexLayer(name)

    def duplicate(self, name):
        layer = SimplexLayer(name)
        layer.seed_gui    = self.seed_gui.duplicate()            
        layer.range_gui   = self.range_gui.duplicate()                 
        layer.scale_gui   = self.scale_gui.duplicate()        
        layer.clamp_gui   = self.clamp_gui.duplicate()    
        return layer        
        
    def __str__(self):
      seed = str(self.seed_gui.getSeed())
      minimum_range = str(self.range_gui.getMinimum())
      maximum_range = str(self.range_gui.getMaximum())
      scale = str(self.scale_gui.getValue())
      clamp_enabled = str(self.clamp_gui.isEnabled())
      clamp_min = str(self.clamp_gui.getBottomValue())
      clamp_max = str(self.clamp_gui.getTopValue())

      return "SimplexLayer"  " " + seed + " " \
        + minimum_range + " " + maximum_range + " "  \
        + scale + " "  \
        + clamp_enabled + " " + clamp_min + " " + clamp_max 
