# Hieghtmap generator.

The heightmap generator allows to create heightmaps/terrain maps for multiple applications: for games, visualization purposes etc. 

## How it works
The heightmap generator produces a 2-dimensional images in grayscale. The image brightness at each point indicates the "height" at the point. Saving this image and importing it in other application allows to use the heigtmap as terrain maps. 
## How to use it
First download and install Python 2.7, then in the console go to the directory you downloaded and unpacked the Heightmap Generator and type:

```sh
$ python generator.py
```


The first thing you will see is the main window. There are few areas to consider:

- The heightmap info pane, where the heightmap dimensions are shown
- The list of added layers with buttons to add/delete/duplicate a layer
- The generate button, which generates the heightmap preview
- The preview pane, where the generated heightmap is shown in grayscale

### Quick start

#### Defining heightmap width and height
To define the heightmap width and height, go to the **File menu** and select **New** option, then input the desired width and height and press **OK**. 

#### Adding a layer 
The heightmap generator uses procedurally generated layers to assemble the image of a heightmap. It applies the layers top to bottom enhancing the image by each layer.
To create a layer click the **New layer** button, name the layer, select the layer type and press **OK**. Each layer type is described it this menu.You can add multiple layers this way. 

#### Editing layers
To edit parameters of each layer - double click the layer in the layer list pane, For each type of layer the parameters vary.
To move the layers up and down, select a layer with a single click in the layer list pane and press the down or up arrow below that list.

#### Layer modes
Each layer has its mode, either add, subtract, multiply or divide. Each of this modes makes the layer act upon the generated heights from the preceeding layers. For example if mode is set to **add** for a layer, when this layer is being applied, it height values, for each point on the heightmap will add to what was generated with previous layers. Analogically for other modes, this values will be either subtracted from, multiplicated or divided. 

#### Generating a heightmap from layers
Click the **Generate** button to see how the generated height map will look like. 


#### Saving, loading and exporting the heightmap
The heightmap files are saved with **.hm** extension. To save the generated heightmap go to the **File** menu and select **Save** option.You can then load the heightmap file using file's menu **load** button.

If you want to export the heightmap as .png file, you need to press the **Dxport as .png** button in the file menu. You will be asked to save the grayscale image in .png format to the disk. Name your file and click **Save**.



 