import random

from Tkinter import *
from PIL import Image, ImageTk


# Utility functions and classess.

# Image loader functions
images = {}    
def loadImage(name):
  img = Image.open("images/" + name)
  tk_img = ImageTk.PhotoImage(img)
  images[name] = tk_img # Just keeping the reference to avoid GC (see ImageTk docs)
  return tk_img



# Seeding functions:
randomizer = random.Random()
seeding_states = {}
def setSeedNumber(new_seed):
  global seeding_states
  
  print(new_seed)
  
  if new_seed in seeding_states:
    state = seeding_states[new_seed]
    randomizer.seed(new_seed)
    randomizer.setstate(state)
    print("retreiving seed")
  else:
    randomizer.seed(new_seed)
    state = randomizer.getstate()
    seeding_states[new_seed] = state
    print("new seed")
  

def getNextSeedInt(low, high):
  return randomizer.randint(low, high)
  

def mapRange(input_value, original_range_min, original_range_max, target_range_min, target_range_max):
  delta_original = float(original_range_max) - float(original_range_min)
  delta_target   = float(target_range_max) - float(target_range_min)
  scale = delta_target / delta_original

  delta_input_to_min = input_value - original_range_min
  output = delta_input_to_min * scale + target_range_min
  return output


def clamp(input_value, minimum, maximum):
  if input_value < minimum:
    return minimum
  elif input_value > maximum:
    return maximum
  else:
    return input_value

if __name__ == "__main__":
  '''setSeedNumber(0)
  n = getNextSeedInt(0,10)
  print(n)
  n = getNextSeedInt(0,10)
  print(n)
  n = getNextSeedInt(0,10)
  print(n)
  
  print("----------------")
  
  setSeedNumber(100)
  n = getNextSeedInt(0,10)
  print(n)
  n = getNextSeedInt(0,10)
  print(n)
  n = getNextSeedInt(0,10)
  print(n)

  print("----------------")
  setSeedNumber(0)
  n = getNextSeedInt(0,10)
  print(n)
  n = getNextSeedInt(0,10)
  print(n)
  n = getNextSeedInt(0,10)
  print(n)'''

  val = mapRange(10, 0, 20 , 100, 250)
  print val

  
  
  
  
  
  
  
