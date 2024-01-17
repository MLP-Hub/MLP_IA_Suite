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

from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox

from .interface_tools import addImg, errorMessage
from .refresh import messageBox

import cv2
import tempfile
import sys
import os.path

# Import the code for pylc
this_dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(this_dir, 'pylc_master')
sys.path.append(path)
import pylc

#PYLC SPECIFIC FUNCTIONS

def pylcArgs(dlg):
    """Gets user input and sets up PyLC arguments"""

    # Get user input parameters
    if dlg.InputModel_lineEdit.text() is "":
        errorMessage("Input model cannot be empty")
        return
    else:
        model_path = os.path.normpath(dlg.InputModel_lineEdit.text())

    if dlg.InputImg_lineEdit.text() is "":
        errorMessage("Input image cannot be empty")
        return
    else:
        img_path = os.path.normpath(dlg.InputImg_lineEdit.text())

    dlg.PyLC_path = os.path.join(tempfile.mkdtemp(), 'tempMask.png')
    if os.path.isfile(dlg.PyLC_path):
        # check if the temporary file already exists
        os.remove(dlg.PyLC_path)
    
    try:
        scale_val = float(dlg.Scale_lineEdit.text())
    except ValueError:
        errorMessage("Scale must be between 0.1 and 1.0")
        return
    if scale_val < 0.1 or scale_val > 1.0:
        errorMessage("Scale must be between 0.1 and 1.0")
        return
        
    # Set up model arguments
    args = {'schema':None, 
            'model':model_path, 
            'img':img_path, 
            'mask':None, 
            'scale':scale_val, 
            'save_logits':None, 
            'aggregate_metrics':None,
            'mask_path':dlg.PyLC_path}

    # Check for optional model arguments (decided to get rid of this for V1)
    
    return args

def enableTools(dlg):
    """Enables canvas tools once canvas is populated with mask and image"""

    dlg.SideBySide_pushButton.setEnabled(True)
    dlg.SingleView_pushButton.setEnabled(True)
    dlg.Fit_toolButton.setEnabled(True)
    dlg.Pan_toolButton.setEnabled(True)

def runPylc(dlg):
    """Runs pylc and displays outputs"""

    # check if mask exists in tempfile but was not saved
    if dlg.refresh_dict["PyLC"]["Mask"] is None and dlg.PyLC_path is not None:
        ret = messageBox("PyLC mask")
        if ret == QMessageBox.No:
            return

    pylc_args = pylcArgs(dlg) # get pylc args
    if pylc_args is None:
        return # exit if there was an error getting the arguments
    pylc.main(pylc_args) # run pylc
    
    # Display output
    scale_val = dlg.Scale_lineEdit.text()

    # Resize reference image if PyLC was scaled
    scale = float(scale_val)
    if scale != 1.0:
        ref_img = cv2.imread(dlg.InputImg_lineEdit.text()) # load reference image
        height, width = ref_img.shape[:2]
        dim = (int(scale * width), int(scale * height))
        resized_img = cv2.resize(ref_img, dim, interpolation=cv2.INTER_AREA) # for downscaling an image
        img_path = os.path.join(tempfile.mkdtemp(), 'resizedImg.tiff')
        if os.path.isfile(img_path):
            # check if the temporary file already exists
            os.remove(img_path)
        cv2.imwrite(img_path, resized_img)

    else:
        img_path = dlg.InputImg_lineEdit.text()
             
    addImg(img_path,"Original Image",dlg.Img_mapCanvas, True) # show input image in side-by-side
    addImg(dlg.PyLC_path,"PyLC Mask",dlg.Mask_mapCanvas, True) # show output mask in side-by-side
    addImg(dlg.PyLC_path,"PyLC Mask",dlg.Full_mapCanvas, False) # show output mask in fullview
    addImg(dlg.InputImg_lineEdit.text(),"Original Image",dlg.Full_mapCanvas, False) # show input image in full view

    enableTools(dlg)

def saveMask(dlg):
    """Saves mask to file"""

    mask = cv2.imread(dlg.PyLC_path)
    mask_path = None

    # open save dialog and save aligned image
    dialog = QFileDialog()
    dialog.setOption(dialog.DontUseNativeDialog)
    dialog.setNameFilter("PNG format (*.png)")
    dialog.setDefaultSuffix("png")
    dialog.setAcceptMode(QFileDialog.AcceptSave)

    if dialog.exec_():
        mask_path = dialog.selectedFiles()[0]

        cv2.imwrite(mask_path, mask)
        dlg.refresh_dict["PyLC"]["Mask"]=mask_path
