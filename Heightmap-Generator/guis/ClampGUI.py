from array2d import *

from random  import *
from Tkinter import *


 
# The clamping gui that most layers have. 
class ClampGUI():
  def __init__(self, title="Clamping", min_val=0, max_val=255):
    self.title = title
    
    self.min = min_val
    self.max = max_val
    
    self.btm_limit_var = IntVar()
    self.top_limit_var = IntVar()
    
    self.btm_limit_var.set(min_val)
    self.top_limit_var.set(max_val)
    
    self.enabled_var = IntVar()

  def setEnabled(self, enabled):
    self.enabled_var.set(enabled)
  
  def isEnabled(self):
    return self.enabled_var.get()

  def setTopValue(self, top_value):
    self.top_limit_var.set(top_value)

  def getTopValue(self):
    return self.top_limit_var.get()

  def setBottomValue(self, bottom_value):
    self.btm_limit_var.set(bottom_value)

  def getBottomValue(self):
    return self.btm_limit_var.get()

  def layoutGUI(self, parent, heightmap):
    frame = LabelFrame(parent, text="Clamping")
    frame.pack(fill=X, padx=5)
      
    self.enabled_checkbox = Checkbutton(frame, 
        text="enabled", offvalue=0, onvalue=1, variable=self.enabled_var)
    self.enabled_checkbox.pack()
      
    
    btm_slider = Scale(frame, label="Bottom limit:", 
      orient=HORIZONTAL, variable=self.btm_limit_var, from_=self.min, to=self.max
    )
    self.btm_slider = btm_slider
    btm_slider.pack(fill=X)

    top_slider = Scale(frame, label="Top limit:", 
      orient=HORIZONTAL, variable=self.top_limit_var, from_=self.min, to=self.max
    )
    self.top_slider = top_slider
    top_slider.pack(fill=X)
    
    
    
  

    
  def duplicate(self):
    gui = ClampGUI()
    minimum = self.getBottomValue()
    maximum = self.getTopValue()
    
    new_top_var = IntVar()
    new_top_var.set(maximum)
    
    new_btm_var = IntVar()
    new_btm_var.set(minimum)

    gui.btm_limit_var = new_btm_var
    gui.top_limit_var = new_top_var
    
    
    return gui
    
