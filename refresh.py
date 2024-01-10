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

"""This module contains functions for refreshing each tool page in the plugin"""

from qgis.PyQt.QtWidgets import QMessageBox

from qgis.core import QgsProject

from .interface_tools import sideBySide

def messageBox(item):
    """Shows message box asking about unsaved items"""

    msgBox = QMessageBox()
    msgBox.setText("{} is not saved.".format(item))
    msgBox.setInformativeText("Refresh anyway?")
    msgBox.setStandardButtons(QMessageBox.Yes)
    msgBox.addButton(QMessageBox.No)
    #msgBox.setDefaultButton(QMessageBox.No)
    ret = msgBox.exec()

    return ret

def emptyCanvases(canvas_list):
    """Removes all layers from provided canvases and from underlying project"""

    for canvas in canvas_list:
        lyrs = canvas.layers()
        for lyr in lyrs:
            QgsProject.instance().removeMapLayer(lyr)
        canvas.setLayers([])
        canvas.refreshAllLayers()

def refresh_PyLC(dlg, canvas_list):
    """Refresh UI in PyLC tab"""

    # check if mask is saved
    if dlg.refresh_dict["PyLC"]["Mask"] is None and dlg.PyLC_path is not None:
        ret = messageBox("PyLC mask")
        if ret == QMessageBox.No:
            return

    # then empty all canvases
    emptyCanvases(canvas_list)

    # refresh all text boxes
    dlg.InputImg_lineEdit.clear()

    # refresh any other widgets
    dlg.Model_comboBox.setCurrentIndex(0)
    dlg.Scale_slider.setValue(10)
    dlg.Scale_lineEdit.setText("1.0")

    # deactivate tools
    dlg.SideBySide_pushButton.setEnabled(False)
    dlg.SingleView_pushButton.setEnabled(False)
    dlg.Fit_toolButton.setEnabled(False)
    dlg.Pan_toolButton.setEnabled(False)

    # return to side-by-side view
    sideBySide(canvas_list, [dlg.Swipe_toolButton, dlg.Transparency_slider], dlg.SideBySide_pushButton, dlg.SingleView_pushButton)
    dlg.SideBySide_pushButton.hide() # hide side by side view button

    # refresh defaults and save filepaths
    dlg.refresh_dict["PyLC"]["Mask"]=None
    dlg.PyLC_path = None

def checkCamParams(dlg):
    """Check if all camera parameters are empty"""

    if dlg.InputDEM_lineEdit.text() != '':
        return True
    elif dlg.InputRefImg_lineEdit.text() != '':
        return True
    elif dlg.Easting_lineEdit.text() != '':
        return True
    elif dlg.Elev_lineEdit.text() != '':
        return True
    elif dlg.Azi_lineEdit.text() != '':
        return True
    elif dlg.Northing_lineEdit.text() != '':
        return True
    elif dlg.CamHgt_lineEdit.text() != '':
        return True
    elif dlg.horFOV_lineEdit.text() != '':
        return True
    elif dlg.StepSizeM_lineEdit.text() != '1':
        return True
    elif dlg.StepSizeDeg_lineEdit.text() != '1':
        return True
    else:
        return False

def refresh_VP(dlg, canvas_list):
    """Refresh UI in VP tab"""

    # first check if camera parameters and VP are saved
    cam_modified = checkCamParams(dlg)
    if dlg.refresh_dict["VP"]["Cam"] is None and cam_modified:
        ret = messageBox("Camera parameters")
        if ret == QMessageBox.No:
            return
        
    if dlg.refresh_dict["VP"]["VP"] is None and dlg.vp_path is not None:
        ret = messageBox("Virtual photograph")
        if ret == QMessageBox.No:
            return
      
    # then empty all canvases
    emptyCanvases(canvas_list)
        
    # refresh all text boxes
    dlg.InputDEM_lineEdit.clear()
    dlg.InputRefImg_lineEdit.clear()
    dlg.Easting_lineEdit.clear()
    dlg.Elev_lineEdit.clear()
    dlg.Azi_lineEdit.clear()
    dlg.Northing_lineEdit.clear()
    dlg.CamHgt_lineEdit.clear()
    dlg.horFOV_lineEdit.clear()
    dlg.StepSizeM_lineEdit.setText("1")
    dlg.StepSizeDeg_lineEdit.setText("1")

    # deactivate tools
    dlg.SideBySide_pushButton_2.setEnabled(False)
    dlg.SingleView_pushButton_2.setEnabled(False)
    dlg.Fit_toolButton_2.setEnabled(False)
    dlg.Pan_toolButton_2.setEnabled(False)

    # return to side-by-side view
    sideBySide(canvas_list, [dlg.Swipe_toolButton_2, dlg.Transparency_slider_2],dlg.SideBySide_pushButton_2, dlg.SingleView_pushButton_2)
    dlg.SideBySide_pushButton_2.hide() # hide side by side view button

    # refresh defaults and save filepaths
    dlg.vp_path = None
    dlg.lat_init = None # initial lat and long for DEM clipping
    dlg.lon_init = None

    dlg.refresh_dict["VP"]["VP"] = None
    dlg.refresh_dict["VP"]["Cam"] = None

def removeLayers(layerName):
    for layer in QgsProject.instance().mapLayers().values():
        if layer.name()==layerName:
            QgsProject.instance().removeMapLayers( [layer.id()] )   

def refresh_align(dlg, canvas_list):
    """Refresh UI in align tab"""

    # first check if all products are saved
    # CPs, aligned image, aligned mask if available
    if dlg.refresh_dict["Align"]["Img"] is None and dlg.aligned_img_path is not None:
        ret = messageBox("Aligned image")
        if ret == QMessageBox.No:
            return
    
    if dlg.Mask_lineEdit.text() and dlg.refresh_dict["Align"]["Mask"] is None and dlg.aligned_mask_path is not None:
        ret = messageBox("Aligned mask")
        if ret == QMessageBox.No:
            return

    if QgsProject.instance().mapLayersByName("Source CP Layer") or QgsProject.instance().mapLayersByName("Dest CP Layer") and dlg.refresh_dict["Align"]["CPs"] is None:
        ret = messageBox("Control points")
        if ret == QMessageBox.No:
            return
    
    # then empty all canvases
    emptyCanvases(canvas_list)

    # delete temporary CP layers
    removeLayers("Source CP Layer")
    removeLayers("Dest CP Layer")

    # refresh all text boxes
    dlg.SourceImg_lineEdit.clear()
    dlg.DestImg_lineEdit.clear()
    dlg.Mask_lineEdit.clear()

    # refresh any other widgets
    dlg.CP_table.clearContents()
    dlg.CP_table.setRowCount(4)

    # deactivate tools
    dlg.SideBySide_pushButton_3.setEnabled(False)
    dlg.SingleView_pushButton_3.setEnabled(False)
    dlg.Fit_toolButton_3.setEnabled(False)
    dlg.Pan_toolButton_3.setEnabled(False)

    if dlg.Mask_lineEdit.text():
        dlg.Layer_comboBox.setEnabled(False)

    # return to side-by-side view
    sideBySide(canvas_list, [dlg.Swipe_toolButton_3, dlg.Transparency_slider_3],dlg.SideBySide_pushButton_3, dlg.SingleView_pushButton_3)
    dlg.SideBySide_pushButton_3.hide() # hide side by side view button

    # refresh defaults and save filepaths
    dlg.refresh_dict["Align"]["Img"]=None
    dlg.refresh_dict["Align"]["Mask"]=None
    dlg.refresh_dict["Align"]["CPs"]=None

    dlg.aligned_img_path = None
    dlg.aligned_mask_path = None

# def refresh_VS(dlg):
#     """Refresh UI in VS tab"""

#     # first check if VS is saved

#     # then empty all canvases
#     # refresh all text boxes

#     # refresh save filepaths