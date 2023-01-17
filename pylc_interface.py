


from qgis.core import QgsRasterLayer, QgsProject
from qgis.PyQt.QtWidgets import QFileDialog

import sys
import os.path

# Import the code for pylc
this_dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(this_dir, 'pylc_master')
sys.path.append(path)
import pylc

#PYLC SPECIFIC FUNCTIONS

def modelMenu(dlg):
    """Creates menu for PyLC models"""

    mod_dict = {"Greyscale 1":"pylc_2-1_deeplab_ch1_schema_a.pth","Greyscale 2":"pylc_2-2_deeplab_ch1_schema_a.pth",
                "Greyscale 3":"pylc_2-3_deeplab_ch1_schema_a.pth","Greyscale 4":"pylc_2-4_deeplab_ch1_schema_a.pth",
                "Greyscale 5":"pylc_2-5_deeplab_ch1_schema_a.pth","Colour 1":"pylc_2-1_deeplab_ch3_schema_a.pth",
                "Colour 2":"pylc_2-2_deeplab_ch3_schema_a.pth","Colour 3":"pylc_2-3_deeplab_ch3_schema_a.pth","Colour 5":
                "pylc_2-5_deeplab_ch3_schema_a.pth"}
    dlg.Model_comboBox.clear()
    dlg.Model_comboBox.addItems(key for key in mod_dict) # Add model names to combo box
    return mod_dict

def pylcArgs(dlg, mod_dict):
    """Gets user input and sets up PyLC arguments"""

    # Get user input parameters
    dir_path = __file__
    dir_path = dir_path[:-18]
    model_file = mod_dict[dlg.Model_comboBox.currentText()] # accesses model file name from model dictionary
    model_path = os.path.normpath(dir_path + "\\pylc_master\\data\\models\\"+model_file)

    img_path = os.path.normpath(dlg.InputImg_lineEdit.text())
    out_path = os.path.normpath(dlg.OutputImg_lineEdit.text())
    scale_val = float(dlg.Scale_lineEdit.text())
        
    # Set up model arguments
    args = {'schema':None, 
            'model':model_path, 
            'img':img_path, 
            'mask':None, 
            'scale':scale_val, 
            'save_logits':None, 
            'aggregate_metrics':None,
            'output_dir':out_path}

    # Check for optional model arguments
    mask_path = dlg.InputMsk_lineEdit.text()
    if mask_path:
        args['mask'] = mask_path
    
    return args

def runPylc(dlg, mod_dict):
    """Runs pylc and displays outputs"""

    pylc_args = pylcArgs(dlg, mod_dict) # get pylc args
    #pylc.main(pylc_args) # run pylc
    
    # Display outputs
    showImg(dlg.InputImg_lineEdit.text(),"Original Image",dlg.Img_mapCanvas) # show input image

    outputDir = dlg.OutputImg_lineEdit.text()
    maskName = os.path.basename(dlg.InputImg_lineEdit.text()).rsplit('.', 1)[0]
    maskExt = os.path.basename(dlg.InputImg_lineEdit.text()).rsplit('.', 1)[1]
    scale_val = dlg.Scale_lineEdit.text()
    outputMsk = os.path.join(outputDir,maskName+"_"+maskExt+"_scale_"+scale_val+".png")

    showImg(outputMsk,"PyLC Mask",dlg.Mask_mapCanvas) # show output mask

    enableTools(dlg)

# GENERIC TOOLS FOLLOW

def setScaleBoxVal(dlg, val):
    """Changes scale box value based on slider"""

    val = float(val / 10)
    val = str(val)
    dlg.Scale_lineEdit.setText(val)

def setScaleSlideVal(dlg, val):
    """Changes scale slider value based on text box"""

    val = float(val)
    val = int(val*10)
    dlg.Scale_slider.setValue(val)

def getFileFolder(lineEdit):
    """Allows user to select image file or folder"""

    def _selected(name):
        """Changes file mode depending on whether current selection is a file or a folder"""

        if os.path.isdir(name):
            dialog.setFileMode(QFileDialog.Directory)
        else:
            dialog.setFileMode(QFileDialog.ExistingFile)

  
    dialog = QFileDialog()
    dialog.setFileMode(dialog.ExistingFiles)
    dialog.setOption(dialog.DontUseNativeDialog)
    dialog.currentChanged.connect(_selected) #if selection changes from file to folder
        
    dialog.exec_()

    filepath = dialog.selectedFiles()
    lineEdit.setText(filepath[0])

def getFolder(lineEdit):
    """Select folder (usually output directory)"""

    dialog = QFileDialog()
    dialog.setFileMode(QFileDialog.Directory)
    dialog.setOption(dialog.DontUseNativeDialog)

    dialog.exec_()
    filepath = dialog.selectedFiles()
    lineEdit.setText(filepath[0])

def showImg(filepath, name, canvas):
    """Displays provided image in map canvas"""

    img_lyr= QgsRasterLayer(filepath, name)
    # Load mask as layer into mask map canvas
    QgsProject.instance().addMapLayer(img_lyr, False) # add layer to the registry (but don't load into main map)
    canvas.enableAntiAliasing(True)
    canvas.setExtent(img_lyr.extent()) # set extent to the extent of the image layer
    canvas.setLayers([img_lyr])
    canvas.show()

def enableTools(dlg):
    """Enables canvas tools once canvas is populated with mask and image"""

    dlg.View_toolButton.setEnabled(True)
    dlg.Fit_toolButton.setEnabled(True)
    dlg.Pan_toolButton.setEnabled(True)
    dlg.FullScrn_toolButton.setEnabled(True)