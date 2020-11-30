from array2d import *

from random  import *
from Tkinter import *


# Range GUI allows to set a base value and the delta around the
# base value. These values may be then used for example in randomization.
class RangeGUI():
  
  def __init__(self, label="Range"):
    self.label = label
    self.minimum  = 0  
    self.maximum = 255 
  
  def layoutGUI(self, parent, heightmap):
    # Create the minimum level slider:
    frame = LabelFrame(parent, text=self.label)
    frame.pack(fill=X, padx=5, pady=5)
    
    min_scale = Scale(frame, orient=HORIZONTAL, label="Minimum value:")
    self.minimum_slider = min_scale
    min_scale["command"] = self.updateMinimum
    min_scale["from"] = 0
    min_scale["to"]   = 255
    min_scale.pack(fill=X)
    min_scale.set(self.minimum)
     
    # Create the maximum value slider   
    max_scale = Scale(frame, orient=HORIZONTAL, label="Maximum value:")
    self.maximum_slider = max_scale
    max_scale["command"] = self.updateMaximum
    max_scale.pack(fill=X)
    max_scale["from"] = 0
    max_scale["to"]   = 255
    max_scale.set(self.maximum)

  def updateMinimum(self, value):
    self.minimum = int(value)
  
  def updateMaximum(self, value):
    self.maximum = int(value)
    
  def setMinimum(self, minimum):
    self.minimum = minimum
    

  def setMaximum(self, maximum):
    self.maximum = maximum

  def getMinimum(self):
    return self.minimum
    
  def getMaximum(self):
    return self.maximum
    
  def duplicate(self):
    gui = RangeGUI()
    gui.minimum  = self.minimum
    gui.maximum  = self.maximum
    return gui
