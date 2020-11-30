from Layer import Layer

from guis.SeedingGUI import SeedingGUI
from guis.RangeGUI   import RangeGUI
from array2d 				 import *
from utils					 import *

# Layer with random values
class RandomValuesLayer(Layer):
  
  def __init__(self, name):
    Layer.__init__(self, name)
    self.seed_gui   = SeedingGUI()
    self.range_gui  = RangeGUI()  
    
  def layoutGUI(self, parent, heightmap=None):
    self.seed_gui.layoutGUI(parent, heightmap)
    self.range_gui.layoutGUI(parent, heightmap)
   
  def setSeed(self, seed):
    self.seed_gui.setSeed(seed)
  
  def setMaximumRange(self, maximum):
    self.range_gui.setMaximum(maximum)

  def setMinimumRange(self, minimum):
    self.range_gui.setMinimum(minimum)

  def makeHeights(self, stack, cumulative):
    w = cumulative.getWidth()
    h = cumulative.getHeight()
    array = Array2D(w, h)
    
    seed = self.seed_gui.getSeed()
    setSeedNumber(seed)
    
    array.each(self.randomize)

    #print "Applying layer " + self.name + "| Mode " +self.mode.getName()
    #print arr2d

    return array
  
  def randomize(self, arr2d, x, y, element):
    minimum = self.range_gui.getMinimum()
    maximum = self.range_gui.getMaximum()
    
    return getNextSeedInt(minimum, maximum)
  
  def getTypeName(self):
    return "Random values layer."
    
  def getTypeDescription(self):
    return \
     "Layer with random heights distributed among the base  \
      value with a spread given by a delta value.  \
      \n\nThe selected seed is the seed of the random numbers \
      used internally by the python random number generator."
    
  def copy(self, name):
    return RandomValuesLayer(name)
  

  def duplicate(self, name):
    layer = RandomValuesLayer(name, self)
    layer.seed_gui  = self.seed_gui.duplicate()
    layer.range_gui = self.range_gui.duplicate() 
    return layer

  def __str__(self):
    seed = str(self.seed_gui.getSeed())
    minimum_range = str(self.range_gui.getMinimum())
    maximum_range = str(self.range_gui.getMaximum())
    
    return "RandomValuesLayer" + " " + seed + " " + minimum_range + " " + maximum_range
