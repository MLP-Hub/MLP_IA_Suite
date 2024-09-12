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

import sys
import os
from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox
from qgis.core import QgsRasterLayer, QgsProject
from qgis.gui import QgsMapToolPan

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from swipe_tool import mapswipetool

def errorMessage(txt):
    """Displays error message to user"""
    msg = QMessageBox()
    msg.setText(txt)
    msg.setIcon(QMessageBox.Critical)
    msg.exec()

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

def getFolder(lineEdit):
    """Select folder (usually output directory)"""

    dialog = QFileDialog()
    dialog.setFileMode(QFileDialog.Directory)
    dialog.setOption(dialog.DontUseNativeDialog)

    if dialog.exec_():
        filepath = dialog.selectedFiles()[0]
        lineEdit.setText(filepath)

def getFile(lineEdit, filter_string):
    """Select file"""

    dialog = QFileDialog()
    dialog.setFileMode(QFileDialog.ExistingFile)
    dialog.setOption(dialog.DontUseNativeDialog)
    dialog.setNameFilter(filter_string)

    if dialog.exec_():
        filepath = dialog.selectedFiles()[0]
        lineEdit.setText(filepath)

def removeLayer(canvas, img_lyr):
    """Removes provided layer from map canvas"""

    layer_list = canvas.layers()
    layer_list.remove(img_lyr)
    canvas.setLayers(layer_list)

def loadLayer(canvas, img_lyr):
    """Loads provided layer into map canvas"""

    canvas.enableAntiAliasing(True)
    canvas.setExtent(img_lyr.extent()) # set extent to the extent of the image layer
    layer_list = canvas.layers()
    layer_list.append(img_lyr)
    canvas.setLayers(layer_list)

def addImg(filepath, name, canvas, visible):
    """Adds provided image in map canvas"""

    layer_list = canvas.layers()

    lyr_ids = {}
    for lyr in layer_list:
        lyr_ids[lyr.name()] = lyr.id()
    if name in lyr_ids:
        QgsProject.instance().removeMapLayer(lyr_ids[name]) # if the layer already exists, remove it from the canvas

    img_lyr= QgsRasterLayer(filepath, name)
    QgsProject.instance().addMapLayer(img_lyr, False) # add layer to the registry (but don't load into main map)

    if visible:
        loadLayer(canvas, img_lyr)

def updateExtents(canvas, ref_canvas):
    """Updates the extent of a map canvas to match a reference canvas"""

    if canvas.extent() != ref_canvas.extent():
        canvas.setExtent(ref_canvas.extent())
        canvas.refresh()

def panCanvas(dlg, canvas_list, pan_button, swipe_button):
    """Enables/disables pan tool for provided map canvas(es)"""
    dlg.pan_tools = []

    # create pan tools for each canvas
    for canvas in canvas_list:
        dlg.pan_tools.append(QgsMapToolPan(canvas))

    # set map tool for each canvas to the appropriate pan tool
    for i in range(len(canvas_list)):
        if pan_button.isChecked():
            canvas_list[i].setMapTool(dlg.pan_tools[i])
            swipe_button.setChecked(False)
        else:
            canvas_list[i].unsetMapTool(dlg.pan_tools[i])

def swipeTool(dlg, canvas, swipe_button, pan_button):
    """Enables/disables swipe tool"""

    if swipe_button.isChecked():
        dlg.swipeTool = mapswipetool.MapSwipeTool(canvas)
        canvas.setMapTool(dlg.swipeTool)
        pan_button.setChecked(False)
    else:
        canvas.unsetMapTool(dlg.swipeTool)

def zoomToExt(canvas_list):
    """Zooms provided canvas to extent of primary layer"""

    active_layer = canvas_list[1].layer(0)
    if active_layer.name() == "Source CP Layer" or active_layer.name() == "Dest CP Layer":
        active_layer = canvas_list[1].layer(1)

    for canvas in canvas_list:
        if canvas is not None:
            canvas.setExtent(active_layer.extent())
            canvas.refresh()

def transparency(val, canvas):
    """Changes top image transparency based on slider"""

    active_layer = canvas.layer(0)
    opacity = (100 - val)/100.0
    active_layer.renderer().setOpacity(opacity)
    active_layer.triggerRepaint()

def sideBySide(canvas_list, exclusive_tools, ss_view_button, single_view_button):
    """Changes display to side-by-side canvases"""

    # change which canvases are visible
    canvas_list[0].raise_()
    canvas_list[1].raise_()
    canvas_list[2].lower()

    # hide the layers on the main canvas
    lyr_list = canvas_list[2].layers()
    for lyr in lyr_list:
        removeLayer(canvas_list[2],lyr)

    # reset layer transparency
    active_layer = canvas_list[0].layer(0)
    active_layer.renderer().setOpacity(1)
    active_layer.triggerRepaint()
    active_layer = canvas_list[1].layer(0)
    active_layer.renderer().setOpacity(1)
    active_layer.triggerRepaint()


    for tool in exclusive_tools:
        tool.setEnabled(False) # disables any tools exclusive to full view (e.g., swipe)

    # deactivate swipe tool
    if exclusive_tools[0].isChecked:
        exclusive_tools[0].setChecked(False) # changes swipe tool to not be checked
        # canvas_list[2].unsetMapTool(exclusive_tools[0]) # do I need to disable? Would have to pass the tool itself, not the button

    # change which display button is visible
    ss_view_button.hide()
    single_view_button.show()

def singleView(canvas_list, exclusive_tools, ss_view_button, single_view_button):
    """Changes display to single canvas"""

    # change which canvases are visible
    canvas_list[2].raise_()
    canvas_list[0].lower()
    canvas_list[1].lower()

    # make the layers on the main canvas visible
    lyr_list = canvas_list[1].layers()
    lyr_list.extend(canvas_list[0].layers())
    for lyr in lyr_list:
        loadLayer(canvas_list[2],lyr)

    for tool in exclusive_tools:
        tool.setEnabled(True) # enables any tools exclusive to full view (e.g., swipe)

    # change which display button is visible
    single_view_button.hide()
    ss_view_button.show()
