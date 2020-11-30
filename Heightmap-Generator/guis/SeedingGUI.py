from array2d import *

from random  import *
from Tkinter import *

# Layers GUI's 

# Seeding gui allows to select a seed and randomize it.
class SeedingGUI():
  def __init__(self, label="Seed"):
    self.label    = label
    self.seed_var = IntVar()
  
  def layoutGUI(self, parent, heightmap):
    if self.seed_var == None:
      seed = 0
    else:
      seed = int (self.seed_var.get())
     
    frame = LabelFrame(parent, text=self.label)
    frame.pack(padx=5, pady=5, fill=X)
    
    self.seed_var = StringVar()
    self.seed_var.set(seed)
   
    seed_label = Label(frame, text= "Seed:")
    seed_label.pack(side=LEFT)        

    seed_entry = Entry(frame, textvariable=self.seed_var, width=6)    
    self.seed_entry = seed_entry
    seed_entry.pack(side=LEFT)
    
    randomize_button = Button(frame, text="Randomize")
    randomize_button["command"] = self.randomizeSeed
    randomize_button.pack(side=LEFT)
    
  def randomizeSeed(self):
    seed = randint(0, 1000)
    self.seed_var.set(seed)
    
  def getSeed(self):
    return int(self.seed_var.get())

  def setSeed(self, seed):
    self.seed_var.set(seed)

  def duplicate(self):
    gui = SeedingGUI()
    gui.seed_var = StringVar()    
    gui.seed_var.set(self.seed_var.get())
    return gui

