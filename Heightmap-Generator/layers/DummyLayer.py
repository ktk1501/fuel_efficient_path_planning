from .Layer import *

# Null object layer - used at the start of an application.
class DummyLayer(Layer):
  def __init__(self):
    Layer.__init__(self, "NONE")

  def layoutGUI(self, parent, heightmap):
    l = Label(parent, text="No layer selected. \n Please select a layer \nin the left panel.")
    l.pack(expand=YES)
  
  def copy(self, name):
    return 
 
        
  def getTypeName(self):
    return "Dummy layer"