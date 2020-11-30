from array2d import *

from random  import *
from Tkinter import *
 

   
# Interpolation gui, that layers, which use interpolation have.
class InterpolationGUI():
  def __init__(self, label="Interpolation", initial_value = 20):
    self.label = label
    self.mode  = 2
    self.resolution = initial_value
    self.mode_var   = IntVar()
    self.mode_var.set(2)
  
  def getInterpolationMode(self):
    return self.mode_var.get()

  def setInterpolationMode(self, mode):
    self.mode = mode
    self.mode_var.set(mode)

  def getResolution(self):
    return self.resolution
  
  def setResolution(self, resolution):
    self.resolution = resolution

  def layoutGUI(self, parent, heightmap):
    w = heightmap.getWidth()
    h = heightmap.getHeight()
    
    frame = LabelFrame(parent, text=self.label)
    frame.pack(fill=X, padx=5)
    
    res_scale = Scale(frame, label="Resolution:", from_=2, to=(min(w,h)-1), orient=HORIZONTAL)
    self.resolution_scale = res_scale
    res_scale["command"] = self.updateResolution
    res_scale.set(self.resolution)
    res_scale.pack(fill=X)
    
    radio_linear = Radiobutton(frame, text="Linear", variable=self.mode_var, value=1, command=self.updateInterpolationMode)
    radio_linear.pack()
    radio_sinusoidal = Radiobutton(frame, text="Sinusoidal", variable=self.mode_var, value=2, command=self.updateInterpolationMode)
    radio_sinusoidal.pack()
    
    print self.mode
    self.mode_var.set(self.mode)


  def updateInterpolationMode(self):
    self.mode = self.mode_var.get()
    
  def updateResolution(self, value):
    self.resolution = int(value)
  
  

  def createInterpolator(self):
    mode = self.mode_var.get()
    print("Mode" + str(mode))
    if mode == 1: return lerp()
    if mode == 2: return slerp()
    

  def duplicate(self):
    gui = InterpolationGUI(self.label)
    gui.resolution = self.resolution
    return gui

