from array2d import *

from random  import *
from Tkinter import *

    
# A simple gui that displays a scale.
class ScaleGUI():
  def __init__(self, label="Value", min_val=0, max_val=100, initial_value = 10):
    self.label = label
    self.value = IntVar()
    self.value.set(initial_value)
    self.min = min_val
    self.max = max_val
      
  def setValue(self, value):
    self.value.set(value)

  def getValue(self):
    return self.value.get()
      
  def layoutGUI(self, parent, heightmap):
    frame = LabelFrame(parent, text=self.label)
    frame.pack()
    
    self.value_slider = Scale(frame, label=self.label, 
        orient=HORIZONTAL, variable=self.value, from_=self.min, to=self.max
    )
    self.value_slider.pack()
      
  def duplicate(self):
    gui = ScaleGUI(self.label, self.min_val, self.max_val, self.value.get())
    return gui
