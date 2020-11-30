

from .RandomValuesLayer import RandomValuesLayer
from .InterpolatedLayer import InterpolatedLayer
from .SimplexLayer      import SimplexLayer

########################################
# Layer factory that keeps prototypes of
# layers, which are to be copied upon
# Creation of new layers
########################################

# The layers factory is a registry kind object
# that stores all the possible layer types in an array.        
class LayersFactory():
  def __init__(self):
    prototypes = []
    self.prototypes = prototypes
    prototypes.extend((
      RandomValuesLayer(""),
      InterpolatedLayer(""),
      SimplexLayer("")
    ))
  
  def getPrototypes(self):
    return self.prototypes


# The definition of a stack of layers used while generating
# the heightmap.
class LayerStack():
  def __init__(self):
    self.layers = []
    
  def getNext(self, idx):
    return self.get(idx + 1)    
    
  def get(self, idx):
    try:
      return self.layers[idx]
    except:
      return None
  
  def append(self, layer):
    self.layers.append(layer)
    


