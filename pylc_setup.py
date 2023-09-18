# -*- coding: utf-8 -*-
"""
/***************************************************************************
 MLP_IA_Suite
                                 A QGIS plugin
 This plugin contains an end-to-end workflow for analysing terrestrial oblique images 
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2022-12-15
        git sha              : $Format:%H$
        copyright            : (C) 2022 by Claire Wright | Mountain Legacy Project
        email                : claire.wright.mi@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

from .interface_tools import addImg

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
    dir_path = os.path.dirname(__file__)
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

    # # Check for optional model arguments (decided to get rid of this for V1)
    # mask_path = dlg.InputMsk_lineEdit.text()
    # if mask_path:
    #     args['mask'] = mask_path
    
    return args

def enableTools(dlg):
    """Enables canvas tools once canvas is populated with mask and image"""

    dlg.View_toolButton.setEnabled(True)
    dlg.Fit_toolButton.setEnabled(True)
    dlg.Pan_toolButton.setEnabled(True)
    dlg.FullScrn_toolButton.setEnabled(True)

def runPylc(dlg, mod_dict):
    """Runs pylc and displays outputs"""

    pylc_args = pylcArgs(dlg, mod_dict) # get pylc args
    pylc.main(pylc_args) # run pylc
    
    # Display output
    outputDir = dlg.OutputImg_lineEdit.text()
    maskName = os.path.basename(dlg.InputImg_lineEdit.text()).rsplit('.', 1)[0]
    maskExt = os.path.basename(dlg.InputImg_lineEdit.text()).rsplit('.', 1)[1]
    scale_val = dlg.Scale_lineEdit.text()
    outputMsk = os.path.join(outputDir,maskName+"_"+maskExt+"_scale_"+scale_val+".png")

    addImg(dlg.InputImg_lineEdit.text(),"Original Image",dlg.Img_mapCanvas) # show input image in side-by-side
    addImg(outputMsk,"PyLC Mask",dlg.Mask_mapCanvas) # show output mask in side-by-side
    addImg(outputMsk,"PyLC Mask",dlg.Full_mapCanvas) # show output mask in fullview
    addImg(dlg.InputImg_lineEdit.text(),"Original Image",dlg.Full_mapCanvas) # show input image in full view

    enableTools(dlg)
