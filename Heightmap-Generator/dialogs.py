#from layers import *

from layers.Layers import *

from tkMessageBox import showerror
from Tkinter import *

from PIL import Image, ImageTk

class Dialog():
  def __init__(self, parent, title):
    win = Toplevel(parent)
    self.win = win
    win.title(title)
    win["padx"] = 10
    win["pady"] = 10
  
    self.createControls()
  
    win.protocol("WM_DELETE_WINDOW", self.destroy)
    win.focus_set()
    win.grab_set()
    parent.wait_window(win)
  
  def createControls(self):
    pass

  def destroy(self):
    self.win.destroy()

class TextDialog(Dialog):
  def __init__(self, parent, title, message, value):
    self.message = message
    self.value   = value
    self.OK = False
    Dialog.__init__(self, parent, title)

  def createControls(self):
    entry_frame = Frame(self.win)
    entry_frame.pack()
    
    label = Label(entry_frame, text=self.message)
    label.pack()
    entry = Entry(entry_frame)
    self.entry = entry
    entry.insert(INSERT, self.value)
    entry.pack()
    
    buttons_frame = Frame(self.win)
    buttons_frame.pack()
    
    ok_button = Button(buttons_frame, text="OK")
    ok_button["command"] = self.ok
    ok_button.pack(side=LEFT)
    
    cancel_button = Button(buttons_frame, text="CANCEL")
    cancel_button["command"] = self.destroy
    cancel_button.pack(side=LEFT)
    
    
  def ok(self):
    self.value = self.entry.get()
    self.OK = True
    self.win.destroy()
    
  def getValue(self):
    return self.value
    

class NewLayerDialog(Dialog):
  def __init__(self, parent, layers_factory, default_layer_name):
    self.OK = False
    self.layers_factory = layers_factory
    self.layer_name = default_layer_name
    self.new_layer = None # The new layer that will be created.
    Dialog.__init__(self, parent, "New layer")
  
  def createControls(self):
    # The top frame: 
    tf = Frame(self.win)
    tf.pack()
    
    ## The name entry:
    nl = Label(tf, text="Layer name:")
    nl.grid(row=0, column=0)
    ne = Entry(tf)
    self.name_entry = ne
    ne.insert(INSERT, self.layer_name)
    ne.focus_set()
    ne.selection_range(0,END)
    ne.grid(row=0, column=1)
    
    # The middle frame
    bf = Frame(self.win)
    self.middle_frame = bf
    bf.pack(fill=X)

    ## The type radiobuttons
    self.type_idx = IntVar()
    protos = self.layers_factory.getPrototypes()
    for proto_idx in range(0, len(protos)):
      proto = protos[proto_idx]
      self.createTypeRadioButton(proto_idx, proto.getTypeName())
    
    ## The type description text widget:
    type_desc = Message(bf, 
      text="Please select layer type using the radio buttons above.\
            The description of the selected type will display here.")
    self.type_desc = type_desc
    type_desc["borderwidth"] = 2
    type_desc["relief"] = SOLID
    type_desc["background"] = "white"
    type_desc.pack(fill=X)
    
    
    # The bottom frame (with OK and CANCEL buttons):
    bbf = Frame(self.win)
    bbf.pack()
    
    ok_button = Button(bbf, text="OK")
    ok_button["command"] = self.ok
    ok_button.pack(side=LEFT)
    cancel_button = Button(bbf, text="Cancel")
    cancel_button["command"] = self.destroy
    cancel_button.pack(side=LEFT)
    
  def createTypeRadioButton(self, idx, type_name):
    radio = Radiobutton(self.middle_frame, value=idx, 
      text=type_name, var=self.type_idx)
    radio["command"] = self.typeChanged
    radio.pack()
  
  def typeChanged(self):
    idx = self.type_idx.get()
    protos = self.layers_factory.getPrototypes()
    proto = protos[idx]
    self.type_desc["text"] = proto.getTypeDescription()

    
  def ok(self):
    idx = self.type_idx.get()
    
    print(idx)
    
    protos = self.layers_factory.getPrototypes()
    proto = protos[idx]
    name = self.name_entry.get()
    self.new_layer = proto.copy(name)
    self.OK = True
    self.destroy()
    
  def getNewLayer(self):
    return self.new_layer


# Dialog displayed when a new heightmap is created.
class HeightmapDialog(Dialog):
  def __init__(self, parent):
    Dialog.__init__(self, parent, "New heightmap")
    
  def createControls(self):
    self.width  = 0
    self.height = 0
    
    # Info label:
    info = Label(self.win, 
      text="Please enter the heightmap\n width and height:")
    info.grid(row=0, columnspan=2)
    
    # Width label and entry:
    w_label = Label(self.win, text="Width:")
    w_label.grid(row=1, column=0)      
    w_entry = Entry(self.win)
    self.w_entry = w_entry
    w_entry.insert(INSERT,"100")
    w_entry.focus_set()
    w_entry.grid(row=1, column=1, pady=4)
    
    # Height label and entry:
    h_label = Label(self.win, text="Height:")
    h_label.grid(row=2, column=0)
    h_entry = Entry(self.win)
    self.h_entry = h_entry
    h_entry.insert(INSERT, "100")
    h_entry.grid(row=2, column=1)
    
    # Buttons frame:
    fr = Frame(self.win)
    fr.grid(columnspan=2, pady=5)
    
    # Ok button:  
    ok_button = Button(fr, text="OK")
    ok_button["command"] = self.ok
    ok_button.pack(side=RIGHT)
    
    # Cancel button:
    cl_button = Button(fr, text="Cancel")
    cl_button["command"] = self.destroy
    cl_button.pack(side=RIGHT)
    
    
  def ok(self):
    is_error = False
    try:
      self.width  = int(self.w_entry.get())
      self.height = int(self.h_entry.get())
    except ValueError:
      is_error = True
  
    if(self.width < 1) or (self.height < 1):
      is_error = True
  
    if is_error:
      showerror(
        "Invalid values", 
        "The supported width and height values must be positive, \
         non-zero integral values.")
      return
    else:
      self.destroy()

