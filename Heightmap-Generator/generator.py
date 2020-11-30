from __future__ import division
from dialogs    import *
from utils      import *
from heightmap  import *

from time         import *
from tkMessageBox import *
from Tkinter      import *

import Tkinter, Tkconstants, tkFileDialog
import json

from PIL import Image, ImageTk
import PIL

from layers.DummyLayer import *
from layers.RandomValuesLayer import *
from layers.InterpolatedLayer import *
from layers.SimplexLayer      import *

# The heigtmap generator main entry point class.
class HeightmapGenerator:
  def __init__(self):
    
    # Initialization:
    ## Make the root window, so things will work correctly:
    ## (the Tkinter variables: "IntVar", "StringVar", require
    ## an existing Tk instance to be active)  
    root = Tk()
    self.root = root
    root.title("Heightmap generator")
    root.tk.call('wm', 'iconphoto', root._w, loadImage("app.ico"))  
    root["padx"] = 10
    root["pady"] = 4
    root.geometry("800x800+0+0")
    
    ## Some of the intance variables:
    size = 100
    
    self.heightmap = Heightmap(size,size) # Current heightmap. 
    self.image = PIL.Image.new("RGB", (size, size), (0,0,0))
    self.layers_factory = LayersFactory() 
    self.layers = {} # A dictionary mapping from layer name to a layer
    self.current_layer = DummyLayer()


    # Creating the UI layout.
    ## Create the menu for root window:
    self.createWindowMenu()
      
    ## Create the top frame:
    top_frame = Frame(root)
    top_frame.pack(expand=YES, fill=BOTH)
  
    ## Create left panel:
    self.left = Frame(top_frame)
    self.left.pack(side=LEFT, fill=BOTH, padx=2)
    self.createHeightmapPane()
    self.createLayersPane()
    self.createActionButtons()

    ## Create the layer settings panel:   
    layer_settings = LabelFrame(top_frame, text = "Layer settings")
    self.layer_settings = layer_settings
    layer_settings.pack( fill=BOTH, side=LEFT, ipadx=5, ipady=5, padx=2)
    self.createLayersSettings()
   
    ## Create right panel:
    self.right = Frame(top_frame)
    self.right.pack(side=LEFT, fill=BOTH, expand=YES, padx=2)
    self.createCanvas()
    
    ## Create the status bar panel:
    status_frame = Frame(root)
    status_frame.pack(fill=BOTH)
    self.status = Label(status_frame, text="Status bar.")
    self.status.pack()
    
    ## Trigger the layer change event do display the dummy layer:
    self.changeCurrentLayer(None)

    
  def createWindowMenu(self):
    menu = Menu(self.root)
    self.menu = menu
    self.root.config(menu=menu)
    
    file_menu = Menu(menu)
    menu.add_cascade(menu=file_menu, label="File")
    
    file_menu.add_command(label="New", command=self.newHeightmap)
    file_menu.add_command(label="Save", command=self.save)
    file_menu.add_command(label="Load", command=self.load)
    file_menu.add_command(label="Export as .png", command=self.export)
    file_menu.add_command(label="Exit", command=self.root.destroy)
    
  def createHeightmapPane(self):
    hm_frame = LabelFrame(self.left, text="Heightmap")
    hm_frame.pack(fill=X, ipadx=5, ipady=5)

    hm_size  = self.heightmap.getWidth()
    
    dims_txt = "Size: %d " % (hm_size)
    self.hm_dims = Label(hm_frame, text=dims_txt )
    self.hm_dims.pack()

    change_size_button = Button(hm_frame, text="Change size", command=self.askHeightmapSize)
    change_size_button.pack()
 
  def createLayersPane(self):
    # Create the layers-pane frame:
    layers = LabelFrame(self.left, text="Layers")
    layers.pack(fill=BOTH, expand=YES, ipadx=5, ipady=5)
    
    ## Create layer buttons:
    bw = 15 # Button width.
      
    new_layer = Button(layers, text="New layer")
    new_layer["width"]   = bw
    new_layer["command"] = self.newLayer
    new_layer.pack()
    
    del_layer = Button(layers, text="Delete layer")
    del_layer["width"]   = bw
    del_layer["command"] = self.delLayer
    del_layer.pack()
    
    dup_layer = Button(layers, text="Duplicate layer")
    dup_layer["command"] = self.duplicateLayer
    dup_layer["width"]   = bw
    dup_layer.pack()
    
    ## Create the layers list:
    layer_list = Listbox(layers, selectmode=SINGLE)
    self.layer_list = layer_list
    layer_list.bind("<Double-Button-1>", self.changeCurrentLayer)
    layer_list.pack(expand=Y, fill=Y)
   
    ## Create layers moving buttons:
    ### First the frame that will make them align horizontal:
    up_down_frame = Frame(layers)
    up_down_frame.pack(fill=X)
    
    up_button = Button(up_down_frame, image=loadImage("layer_up.png"))
    up_button["command"] = self.moveLayerUp
    up_button.pack(side=LEFT, expand=YES)
    
    down_button = Button(up_down_frame, image=loadImage("layer_down.png"))
    down_button["command"] = self.moveLayerDown
    down_button.pack(side=LEFT, expand=YES)
    
  def createActionButtons(self):
    actions = LabelFrame(self.left, text="Actions")
    actions.pack(fill=X , ipadx=5, ipady=5)
    generate = Button(
      actions,
      text="Generate heightmap", image=loadImage("generate.png"), 
      compound=LEFT)
    generate["command"] = self.generateHeightmap
    generate.pack()
    
    
  def createLayersSettings(self):
    # Layer name frame:
    ln_frame = Frame(self.layer_settings)
    ln_frame.pack()
    
    ln_label = Label(ln_frame, text="Layer:")
    ln_label.pack(side=LEFT)

    ln_entry = Entry(ln_frame)
    self.layer_name_entry = ln_entry
    ln_entry.pack()
    
    # Layer type text
    layer_type = self.current_layer.getTypeName()
    lt_label = Label(self.layer_settings, text=layer_type)
    lt_label.pack()

    self.layer_type_label = lt_label

    # Layer mode frame:
    lm_frame = Frame(self.layer_settings)
    lm_frame.pack()
    
    self.layer_mode = StringVar(self.layer_settings)
    self.layer_mode.set("Add")
    lm_combo = OptionMenu(lm_frame, self.layer_mode,
      "Add", "Subtract", "Multiply", "Divide", "Mix",
      command=self.setLayerMode
    )
    lm_combo.pack(side=LEFT)
    
    self.ls_container = Frame(self.layer_settings)
    self.ls_container.pack(expand=YES, fill=BOTH)
    

  def createCanvas(self):
    # Create the right-pane frame:
    prev = LabelFrame(self.right, text="Preview")
    prev.pack(side=LEFT, fill=BOTH, expand=YES)
    # Create the preview canvas:
    canvas = Canvas(prev)
    self.canvas = canvas
    canvas["width"] = 500
    canvas["height"] = 500
    canvas["background"] = "black"
    canvas.pack(anchor=CENTER, expand=YES)
    
    canvas.bind("<Motion>", self.displayHeight)
    
    img = loadImage("cliff.png")
    canvas.create_image(0,0, image=img)
  
  def newHeightmap(self):
    '''dialog = HeightmapDialog(self.root)
    width  = dialog.width
    height = dialog.height
    '''

    dialog = TextDialog(self.root, "Heightmap size", "Please enter the heightmap size", 100) 
    size = int(dialog.getValue())
    if dialog.OK:
      self.heightmap = Heightmap(size, size)
      self.image = PIL.Image.new("RGB", (size, size), (0,0,0))
  
      self.hm_dims["text"] = "Size: %d" % (size)
      self.layers = {}
      self.layer_list.delete(0, END)
      
  def newLayer(self):
    selection = self.layer_list.curselection()
    n_selected = len(selection)
    if n_selected > 0:
      idx = selection[0] + 1 # For inserting after current layer.
    else: idx = 0
    default_name = "Layer %d" % self.layer_list.size()
    # Read the selection before we lose focus in the dialog
    dialog = NewLayerDialog(self.root, self.layers_factory, default_name)
    # Here the list box lost focus, so I need 
    # to restore it afterwards the text was entered.
    if dialog.OK:
      new = dialog.getNewLayer()
      self.insertLayer(new, idx)
      self.layer_list.selection_set(idx) # Restoring the selection
  
  def insertLayer(self, layer, position=END):
    layer_name = layer.getName()
    self.layer_list.insert(position, layer_name)
    self.layers[layer_name] = layer
      
  def duplicateLayer(self):
    selection = self.layer_list.curselection()
    n_selected = len(selection)
    if n_selected > 0:
      idx = selection[0]
    else: 
      showwarning("No layer selected to duplicate.")
      return
    
    layer_name = self.layer_list.get(idx)
    layer      = self.layers[layer_name]
    
    orig_name = layer.getName() # Original name
    new_name =  orig_name # New name (to be made)
    num = 2
    while new_name in self.layers:
      new_name = orig_name + str(num)
      num += 1

    
    dialog = TextDialog(self.root, 
      "Enter new name", "Please enter duplicated layer name", new_name
    )
    
    if dialog.OK:
      new_name = dialog.getValue()
      duplicated = layer.duplicate(new_name)
      self.insertLayer(duplicated, idx+1)
    
  
  # Change the current layer the user is working on.
  # Fired by a double button press on the layers listbox.
  def changeCurrentLayer(self, event):
    selection = self.layer_list.curselection()
    if len(selection) > 0:
      idx = selection[0]
      layer_name = self.layer_list.get(idx)
      layer = self.layers[layer_name]
      self.current_layer = layer
    
    self.readLayerName()
    self.readLayerType()
    self.readLayerMode()
    self.updateLayerSettings()

  def readLayerName(self):
    new_name = self.current_layer.getName()
    self.layer_name_entry.delete(0,END)
    self.layer_name_entry.insert(INSERT, new_name)
    
  def readLayerType(self):
    layer_type_name = self.current_layer.getTypeName()
    #self.layer_type_label.delete(0, END)
    #self.layer_type_label.insert(INSERT, layer_type_name)
    self.layer_type_label["text"] = layer_type_name

  def readLayerMode(self):
    l_mode = self.current_layer.getMode()
    mode_name = l_mode.getName()
    
    self.layer_mode.set(mode_name)

    
  def setLayerMode(self, idx):
    new_name = self.layer_mode.get()
    new_mode = LayerMode.fromName(new_name)
    
    self.current_layer.setMode(new_mode)
    
  def updateLayerSettings(self):
    self.ls_container.pack_forget()
    self.ls_container.destroy()

    ls_container = Frame(self.layer_settings)
    self.ls_container = ls_container
    ls_container.pack(expand=YES, fill=BOTH)
    
    self.current_layer.layoutGUI(ls_container, self.heightmap)
      
  def moveLayerDown(self):
    selection =  self.layer_list.curselection()
    if len(selection) == 0: return
    idx_old = self.layer_list.curselection()[0]
    idx_new = idx_old + 1
    
    # Check if we would go out of the bounds of the list:
    if idx_new > self.layer_list.size() - 1: return
    
    val = self.layer_list.get(idx_old)
    self.layer_list.delete(idx_old)
    self.layer_list.insert(idx_new, val)
    self.layer_list.selection_set(idx_new)
    
  def moveLayerUp(self):
    selection =  self.layer_list.curselection()
    if len(selection) == 0: return
    idx_old = self.layer_list.curselection()[0]
    idx_new = idx_old - 1

    # Check if we would go out of the bounds of the list:
    if idx_new < 0: return
    
    val = self.layer_list.get(idx_old)
    self.layer_list.delete(idx_old)
    self.layer_list.insert(idx_new, val)
    self.layer_list.selection_set(idx_new)
      
  def delLayer(self):
    if not self.isLayerSelected(): return
    idx = self.layer_list.curselection()[0]
    self.layer_list.delete(idx)
  
  def start(self):
    self.root.mainloop()

  def isLayerSelected(self):
    sel = self.layer_list.curselection()
    if len(sel) == 0: return False
    else: return True
  
  def askHeightmapSize(self):
    hm_size = self.heightmap.getWidth()
    dialog = TextDialog(
      self.root, 
      "Change heightmap size", 
      "Please enter new heightmap size", str(hm_size))

    if dialog.OK:
      size = int(dialog.getValue())
      self.setHeightmapSize(size)

  def setHeightmapSize(self, size):
    self.heightmap = Heightmap(size, size)
    self.hm_dims["text"] = "Size: " + str(size)
    

  def generateHeightmap(self):
    l_list = self.layer_list.get(0,END)
    n_layers = len(l_list)
    if(n_layers) == 0: 
      showwarning(
        "No layers!",
        "Cannot generate heightmap - no layers defined."
      )
      return
    
    
    self.setStatus("Generating heightmap...")
    
    # Create the layers stack:
    stack = LayerStack()
    for idx in range(0, n_layers):
      l_name = l_list[idx]
      layer = self.layers[l_name]
      layer.setIndex(idx)
      stack.append(layer)
    
    layer = stack.get(0)
    cumulative = self.heightmap.getInitialHeights()
    
    while layer != None:
      self.setStatus("Applying layer: \"%s\" ..." % layer.getName())
      layer.apply(stack, cumulative)
      layer = layer.getNext(stack)
    
    self.setStatus("Heightmap generated. Generating preview...")
    
    #print "Result: "
    #print cumulative

    self.createPreview(cumulative)    
  
  
  # Create preview image and display it in preview area 
  def createPreview(self, cumulative):
    w = self.heightmap.getWidth()
    h = self.heightmap.getHeight()
    
    self.setStatus("Creating image preview...")
    image = PIL.Image.new("RGBA", (w,h))
    
    for x in range(0, w):
      for y in range(0, h):
        height = int(cumulative.get(x,y))
        
        color = (height, height, height)
        image.putpixel((x,y), color)
        
    self.setStatus("Resizing the image to fit the preview bounds..,")
    original_image = image
    self.image = original_image

    image = original_image.resize(
      ( 1000, 1000 ), 
      PIL.Image.NEAREST
    )
      
    photo_img = ImageTk.PhotoImage(image)
    self.canvas.img = photo_img # Again keep that reference
    self.canvas.create_image(0,0, image=photo_img)
    
    self.setStatus("Done.")
    
  def displayHeight(self, event):
    pass
    
  def setStatus(self, txt):
    self.status["text"] = txt
    self.root.update()


  def save(self):
    data = ""
    hm_size = self.heightmap.getWidth()
    data += "Heightmap: " + str(hm_size) + "\n"
    
    for layer_name in self.layers:
      layer = self.layers[layer_name]
      data += layer.name + ": " 
      data += layer.mode.name
      data += " "
      serialized = str(layer)
      data += serialized

      data += "\n"

    filename = tkFileDialog.asksaveasfilename(
      initialdir = "/", 
      title="Save heightmapp",
      filetypes =  (("heightmap files","*.hm"),("all files","*.*"))
    )

    if filename != "":
      if not filename.endswith(".hm"):
        filename = filename + ".hm"
       
      file = open(filename, "w")
      file.write(data)

    pass

  def load(self):
    filename = tkFileDialog.askopenfilename(filetypes=(("Heightmap files", ".hm"),("all files","*.*")))
    if filename == "": 
      return

    file = open(filename, "r")
    contents = file.read()
    
    lines_raw = contents.split("\n")

    heightmap_size = int(lines_raw[0].split(": ")[1])
    self.setHeightmapSize(heightmap_size)

    #print "Heightmap size is: %d" % (heightmap_size)

    for i in range(1, len(lines_raw)):
      layer_raw = lines_raw[i]
      layer_split = layer_raw.split(": ")
      if len(layer_split) == 1: continue
      layer_name = layer_split[0]
      
      layer_description = layer_split[1].split(" ")
      layer_mode = layer_description[0]
      layer_type = layer_description[1]

      layer = None

      if layer_type == "RandomValuesLayer":
        layer = RandomValuesLayer(layer_name)

        seed = int(layer_description[2])
        minimum_range = int(layer_description[3])
        maximum_range = int(layer_description[4])
        
        
        layer.setSeed(seed)
        layer.setMinimumRange(minimum_range)
        layer.setMaximumRange(maximum_range)
        
      
      if layer_type == "InterpolatedLayer":
        layer = InterpolatedLayer(layer_name)
        
        seed = int(layer_description[2])
        minimum_range = int(layer_description[3])
        maximum_range = int(layer_description[4])
        interpolation_mode = int(layer_description[5])
        resolution = int(layer_description[6])
        clamp_enabled = int(layer_description[7])
        clamp_min = int(layer_description[8])
        clamp_max = int(layer_description[9])
        
        layer.setSeed(seed)
        layer.setMinimumRange(minimum_range)
        layer.setMaximumRange(maximum_range)
        layer.setInterpolationMode(interpolation_mode)
        layer.setResolution(resolution)
        layer.setClampEnabled(clamp_enabled)
        layer.setClampBottomValue(clamp_min)
        layer.setClampTopValue(clamp_max)

      if layer_type == "SimplexLayer":
        layer = SimplexLayer(layer_name)
        seed = int(layer_description[2])
        minimum_range = int(layer_description[3])
        maximum_range = int(layer_description[4])
        scale = int(layer_description[5])
        clamp_enabled = int(layer_description[6])
        clamp_min = int(layer_description[7])
        clamp_max = int(layer_description[8])
 
        layer.setSeed(seed)
        layer.setMinimumRange(minimum_range)
        layer.setMaximumRange(maximum_range)
        layer.setScale(scale)
        layer.setClampEnabled(clamp_enabled)
        layer.setClampBottomValue(clamp_min)
        layer.setClampTopValue(clamp_max)

      
      layer.mode = LayerMode.fromName(layer_mode)
      self.insertLayer(layer)
      

  def export(self):

    if self.image == None:
      self.setStatus("Please generate the heightmap before exporting")

    filename = tkFileDialog.asksaveasfilename(
      initialdir = "/", 
      title="Save heightmapp",
      filetypes =  (("png files","*.png"),("all files","*.*"))
    )

    if filename != "":
      if not filename.endswith(".png"):
        filename = filename + ".png"
      self.image.save(filename)
    
        

if __name__ == "__main__":
  app = HeightmapGenerator()
  app.start()
