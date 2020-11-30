from __future__ import division

from array2d import *
from random  import *
from math    import *


class Heightmap():
  def __init__(self, w, h):
    self.width  = w
    self.height = h
	  
    
  def getInitialHeights(self):
    return Array2D(self.width, self.height)
  
  
  
  def getWidth(self):
    return self.width
    
  def getHeight(self):
    return self.height
	  

		




