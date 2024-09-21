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
from .interface_tools import addImg, removeLayer, singleView, sideBySide

from qgis.gui import QgsMapToolEmitPoint, QgsMapToolIdentifyFeature
from qgis.core import QgsVectorLayer, QgsField, QgsProject, QgsFeature, QgsGeometry, QgsMarkerSymbol, QgsFontMarkerSymbolLayer, QgsPointXY
from PyQt5.QtCore import Qt, QVariant
from qgis.PyQt.QtWidgets import QTableWidgetItem, QFileDialog

import cv2
import numpy as np
import tempfile
import os

from .interface_tools import errorMessage

def checkForImgs(canvas_list, name_list, button_list):
    """Ensures one image per canvas before enabling tool buttons"""
    
    # check for image layers
    img_count = 0
    for i, canvas in enumerate(canvas_list):
        layer_list = canvas.layers()
        lyr_ids = {}
        for lyr in layer_list:
            lyr_ids[lyr.name()] = lyr.id()
        if name_list[i] in lyr_ids:
            img_count +=1

    # enable all cp buttons
    if img_count == 2:
        for button in button_list:
            button.setEnabled(True)

def createCPLayer(canvas, lyr_name, img_name):
    """Creates a temporary vector layer to receive CPs"""

    img_layer = QgsProject.instance().mapLayersByName(img_name)[0]
    img_crs = img_layer.crs() # get CRS of underlying image so that points are in correct position

    vl = QgsVectorLayer("Point", lyr_name, "memory") # create new vector layer
    vl.setCrs(img_crs)
    pr = vl.dataProvider()
    vl.startEditing() # enter editing mode
    pr.addAttributes( [ QgsField("X", QVariant.Double), QgsField("Y",  QVariant.Double)] ) # add fields for X and Y coords
    vl.commitChanges() # commit changes

    # change layer symbology to cross
    font_style = {}
    font_style['font'] = 'Webdings'
    font_style['chr'] = 'r' # Webdings r corresponds to cross
    font_style['size'] = '6'
    # color depends on source vs destination canvas
    if lyr_name == "Source CP Layer":
        font_style['color'] = 'red'
    else:
        font_style['color'] = 'orange'
    symbol = QgsMarkerSymbol.createSimple({'color': 'black'}) # iniate symbol
    symbol_lyr = QgsFontMarkerSymbolLayer.create(font_style) # create marker
    symbol.changeSymbolLayer(0,symbol_lyr) # swap symbol to cross
    vl.renderer().setSymbol(symbol)

    QgsProject.instance().addMapLayer(vl, False) # add vector layer to the registry (but don't load into main map)

    # add layer to the provided canvas
    layer_list = canvas.layers()
    layer_list.insert(0, vl) # must add to front of layer list to ensure CPs are visible above image
    canvas.setLayers(layer_list)
    canvas.refreshAllLayers()

def addCP(canvas, vl, x, y, table, name, dlg):
    """Adds CP to vector layer and table"""
    
    vl.startEditing() # enter editing mode
    pr = vl.dataProvider()
    fet = QgsFeature()
    fet.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x,y))) # add point with x,y coords
    fet.setAttributes([x,y]) 
    pr.addFeatures([fet])
    vl.commitChanges() # commit changes
    vl.triggerRepaint() # repaint the layer
    canvas.refreshAllLayers()

    # add cp coordinates to table
    # check destination or source layer
    if name == "Source CP Layer":
        col = 0
    else:
        col = 2
    
    cp_list = vl.getFeatures() # get iterator of cps
    row = len(list(cp_list))-1 # tells you which row to fill in data
    if table.rowCount() <= row:
        table.setRowCount(row+1) # add row if number of CPs exceeds table size
    table.setItem(row, col, QTableWidgetItem(str(round(x,2))))
    table.setItem(row, col+1, QTableWidgetItem(str(abs(round(y,2)))))

    dlg.refresh_dict["Align"]["CPs"]=None

def addCPfromClick(dlg, point, canvas, name):
    """Adds CP (from mouse click) to vector layer and table"""

    vl = QgsProject.instance().mapLayersByName(name)[0] # get CP vector layer
    clearSelection([dlg.DestImg_canvas, dlg.SourceImg_canvas]) # remove any highlighted points

    # add cp to vector layer and table
    addCP(canvas, vl, point.x(), point.y(), dlg.CP_table, name, dlg)

    # When done, switch canvases
    canvas.unsetMapTool(dlg.CPtool)
    dlg.CPtool = None
    if name == "Source CP Layer":
        addCPTool(dlg, dlg.DestImg_canvas, "Dest CP Layer", "Destination Image")
    else:
        addCPTool(dlg, dlg.SourceImg_canvas, "Source CP Layer", "Source Image")

def addCPTool(dlg, canvas, name, img_name):
    """Allows user to create new control point"""

    # check if vector layer of CPs exists, otherwise create one
    layer_list = canvas.layers()
    lyr_ids = {}
    for lyr in layer_list:
        lyr_ids[lyr.name()] = lyr.id()
    if name not in lyr_ids:
       createCPLayer(canvas, name, img_name) 

    dlg.CPtool = QgsMapToolEmitPoint(canvas) # create control point tool
    dlg.CPtool.canvasClicked.connect(lambda point: addCPfromClick(dlg, point, canvas, name)) # add cp to canvas when user clicks the image
    dlg.CPtool.setCursor(Qt.CrossCursor) # use crosshair cursor
    canvas.setMapTool(dlg.CPtool) # set the map tool for the canvas

def deleteCP(f, vl_list, table, dlg):
    """Deletes selected feature from layer and cp table"""

    # loop through both layers to find which cp was clicked on,
    # but specifically, to find its position in the table
    flag = False
    for cp_layer in vl_list:
        cp_features = cp_layer.getFeatures()
        row_num = 0
        for ft in cp_features:
            if ft.id() is f.id():
                flag = True # create flag to break outer loop
                break
            else:
                row_num+=1
        if flag:
            break

    table.selectRow(row_num)
    row = table.currentRow()
    if table.rowCount() >= 1:
        table.removeRow(row) # delete cp row from table
    else:
        return # if there are no rows, nothing to delete

    for vl in vl_list:
        vl.startEditing() # enter editing mode
        pr = vl.dataProvider()
        pr.deleteFeatures([f.id()]) # delete selected CP from vector layer
        vl.commitChanges()

        vl.removeSelection() # remove any selection (may have been highlighted from table)

    dlg.refresh_dict["Align"]["CPs"]=None
    
def delCPTool(dlg, canvas_list):
    """Allows user to select control point from map to delete"""

    vl1 = canvas_list[0].layers()[0] # get CP layer for canvas (should always be top layer)
    if isinstance(vl1, QgsVectorLayer):
        dlg.delTool1 = QgsMapToolIdentifyFeature(canvas_list[0])
        dlg.delTool1.setLayer(vl1)
        canvas_list[0].setMapTool(dlg.delTool1)
        dlg.delTool1.featureIdentified.connect(lambda f: deleteCP(f, [vl1,vl2], dlg.CP_table))    # connect delete tools to delete function
    
    vl2 = canvas_list[1].layers()[0] # get CP layer for canvas (should always be top layer)
    
    if isinstance(vl1, QgsVectorLayer):
        dlg.delTool2 = QgsMapToolIdentifyFeature(canvas_list[1])
        dlg.delTool2.setLayer(vl2)
        canvas_list[1].setMapTool(dlg.delTool2)
        dlg.delTool2.featureIdentified.connect(lambda f: deleteCP(f, [vl1,vl2], dlg.CP_table, dlg))    # connect delete tools to delete function

def clearSelection(canvas_list):
    """Clears any selected CPs"""

    for canvas in canvas_list:
        cp_layer = canvas.layers()[0] # get CP layer for canvas (should always be top layer)
        if isinstance(cp_layer, QgsVectorLayer):
            cp_layer.removeSelection() # clear any existing selection

def selectFromTable(table, canvas_list):
    """Allows user to select CP by clicking table row"""
    
    row = table.currentRow()
    for canvas in canvas_list:   
        cp_layer = canvas.layers()[0] # get CP layer for canvas (should always be top layer)
        if isinstance(cp_layer, QgsVectorLayer):
            cp_layer.removeSelection() # clear any existing selection
            cp_features = cp_layer.getFeatures() # get list of features  
            # find cp matching the table row
            i = 0 
            for f in cp_features:
                if i == row:
                    cp_layer.select(f.id())
                    break
                i +=1

def readCPsfromLayer():
    """Reads control point coordinates from vector layers"""
    
    source_pts = []
    dest_pts = []

    try:
        src_layer = QgsProject.instance().mapLayersByName("Source CP Layer")[0] # get CP vector layer
    except IndexError:
        return [], []

    for feat in src_layer.getFeatures():
        pt = [feat.geometry().asPoint().x(), abs(feat.geometry().asPoint().y())]
        source_pts.append(pt)

    try:
        dest_layer = QgsProject.instance().mapLayersByName("Dest CP Layer")[0] # get CP vector layer
    except IndexError:
        return [], []

    for feat in dest_layer.getFeatures():
        pt = [feat.geometry().asPoint().x(), abs(feat.geometry().asPoint().y())]
        dest_pts.append(pt)

    return source_pts, dest_pts

def saveCPs(dlg):
    """Allows user to save set of control points"""

    dialog = QFileDialog()
    dialog.setOption(dialog.DontUseNativeDialog)
    dialog.setNameFilter("Text files (*.txt)")
    dialog.setDefaultSuffix("txt")
    dialog.setAcceptMode(QFileDialog.AcceptSave)

    if dialog.exec_():
        cp_filepath = dialog.selectedFiles()[0]

    cp_file = open(cp_filepath, "w")

    source_pts, dest_pts = readCPsfromLayer()
    for cp in source_pts: 
        for coord in cp:
            cp_file.write('%s ' % (coord))
    cp_file.write('\n') # add new line for destination points
    for cp in dest_pts: 
        for coord in cp:
            cp_file.write('%s ' % (coord))

    cp_file.close()
    
    dlg.refresh_dict["Align"]["CPs"]=cp_filepath

def loadCPs(layer_names, canvases, img_names, table, dlg):
    """Allows user to load a set of saved control points"""
    
    # first, get user specified text file containing previously saved CPs
    dialog = QFileDialog()
    dialog.setFileMode(QFileDialog.ExistingFile)
    dialog.setOption(dialog.DontUseNativeDialog)
    dialog.setNameFilter("Text files (*.txt)")
    if dialog.exec_():
        cp_filepath = dialog.selectedFiles()[0]

    cp_file = open(cp_filepath, "r")

    source_pts_list = cp_file.readline().strip().split() # read first line (source coordinates)
    dest_pts_list = cp_file.readline().strip().split() # read second line (destination coordinates)
    source_pts = []
    dest_pts = []
    pt_count = 0
    # iterate over list of coordinates to create new list with paired [x,y] float coordinates
    while pt_count < len(source_pts_list):
        source_pts.append([float(source_pts_list[pt_count]), float(source_pts_list[pt_count+1])])
        dest_pts.append([float(dest_pts_list[pt_count]), float(dest_pts_list[pt_count+1])])
        pt_count+=2

    cp_file.close()

    cps_list = [source_pts, dest_pts]

    x = 0
    for x in range(0, 2):
        # check if vector layer of CPs exists, otherwise create one
        layer_list = canvases[x].layers()
        lyr_ids = {}
        for lyr in layer_list:
            lyr_ids[lyr.name()] = lyr.id()
        if layer_names[x] not in lyr_ids:
            createCPLayer(canvases[x], layer_names[x], img_names[x]) 

        vl = QgsProject.instance().mapLayersByName(layer_names[x])[0] # get CP vector layer

        for point in cps_list[x]:
            addCP(canvases[x], vl, point[0], -point[1], table, layer_names[x], dlg)

def transformPoints(source_pts, dest_points, matrix, table):
    """Finds coordinates of cps after alignment"""

    i = 0 
    for p in source_pts:

        # trasnform source points based on alignment
        px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
        py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))

        # find difference between new source point location and destination point
        dest_p = dest_points[i]
        dx = abs(dest_p[0] - px)
        dy = abs(dest_p[1] - py)
        rmse = np.sqrt(dx**2+dy**2) # calculate RMSE

        # add values to table
        table.setItem(i, 4, QTableWidgetItem(str(round(dx,7))))
        table.setItem(i, 5, QTableWidgetItem(str(round(dy,7))))
        table.setItem(i, 6, QTableWidgetItem(str(round(rmse,7))))

        i+=1

def enableTools(dlg):
    """Enables canvas tools once canvas is populated with aligned image"""

    dlg.SideBySide_pushButton_3.setEnabled(True)
    dlg.SingleView_pushButton_3.setEnabled(True)
    dlg.Fit_toolButton_3.setEnabled(True)
    dlg.Pan_toolButton_3.setEnabled(True)

    if dlg.Mask_lineEdit.text():
        dlg.Layer_comboBox.setEnabled(True)

def enableButtons(dlg, undo):
    """Disables align and CP buttons after alignment or enables with Undo"""

    dlg.Align_button.setEnabled(undo)
    dlg.addCP_button.setEnabled(undo)
    dlg.delCP_button.setEnabled(undo)
    dlg.loadCP_button.setEnabled(undo)

def switchLayer(comboBox, main_canvas, side_canvas):
    """Switches which image is on top"""

    layer_name = comboBox.currentText() # find selected layer from combo box

    # access canvas layers
    al_img = QgsProject.instance().mapLayersByName('Aligned Image')[0]
    al_mask = QgsProject.instance().mapLayersByName('Aligned Mask')[0]
    ref_img = QgsProject.instance().mapLayersByName('Destination Image')[0]

    # reorder layers
    if layer_name == "Aligned Image":
        main_list = [al_img, ref_img, al_mask]
        side_list = [al_img, al_mask]
    else:
        main_list = [al_mask, ref_img, al_img]
        side_list = [al_mask, al_img]

    # reset layer list in side-by-side and single view
    main_canvas.setLayers(main_list) 
    side_canvas.setLayers(side_list)

def alignImgs(dlg, source_img_path, table):
    """Aligns images using perspective transformation"""
    
    if not os.path.exists(source_img_path):
        return
    img = cv2.imread(source_img_path)
    
    source_pts, dest_pts = readCPsfromLayer()

    if len(source_pts) < 4:
        errorMessage("At least four control points are required for alignment")
        return
    if len(source_pts) != len(dest_pts):
        errorMessage("Number of control points on source and destination images must match")
        return

    source_pts_array = np.float32(source_pts)
    dest_pts_array = np.float32(dest_pts)

    # get best control points based on homography and RANSAC method
    homography, cp_mask = cv2.findHomography(source_pts_array, dest_pts_array, cv2.RANSAC,5.0)
    cp_mask = cp_mask.flatten()

    index = np.nonzero(cp_mask)
    
    source_pts_good = source_pts_array[index]
    dest_pts_good = dest_pts_array[index]
    
    # align image using perspective transform
    dest_img = cv2.imread(dlg.DestImg_lineEdit.text())
    h,w = dest_img.shape[:2] # automatically clip to the destination image
    matrix = cv2.getPerspectiveTransform(source_pts_good, dest_pts_good)
    try:
        aligned_img = cv2.warpPerspective(img, matrix, (w, h))
    except:
        errorMessage("Alignment failed")
        return

    transformPoints(source_pts, dest_pts, matrix, table) # find dx, dy, and RMSE

    # save aligned image to temporary file
    dlg.aligned_img_path = os.path.join(tempfile.mkdtemp(), 'alignedImg.tiff')
    if os.path.isfile(dlg.aligned_img_path):
        # check if the temporary file already exists
        os.remove(dlg.aligned_img_path)

    cv2.imwrite(dlg.aligned_img_path, aligned_img)

    # align mask if provided
    if dlg.Mask_lineEdit.text():
        mask = cv2.imread(dlg.Mask_lineEdit.text())
        img_h, img_w = img.shape[:2]
        mask_h, mask_w = mask.shape[:2]
        if img_h != mask_h or img_w != mask_w:
            errorMessage("Source image and mask must have the same dimensions")
            return
        
        aligned_mask = cv2.warpPerspective(mask, matrix, (w, h), flags = cv2.INTER_NEAREST)

        #check for probability layer (PyLC output)
        mask_path, ext = os.path.splitext(os.path.realpath(dlg.Mask_lineEdit.text()))
        probs_path = os.path.join(mask_path + '.npy')
        if os.path.isfile(probs_path):
            # if probability layer exists, align and save to temporary file
            probs = np.load(probs_path)
            probs = probs.astype(np.float32)
            aligned_probs = cv2.warpPerspective(probs, matrix, (w, h), flags = cv2.INTER_NEAREST) # align probability layer with same 
            # save aligned probabilities to temporary file
            dlg.aligned_probs_path = os.path.join(tempfile.mkdtemp(), 'alignedProbs.npy')
            if os.path.isfile(dlg.aligned_probs_path):
                # check if the temporary file already exists
                os.remove(dlg.aligned_probs_path)
            np.save(dlg.aligned_probs_path, aligned_probs)

        # save aligned mask to temporary file
        dlg.aligned_mask_path = os.path.join(tempfile.mkdtemp(), 'alignedMask.tiff')
        if os.path.isfile(dlg.aligned_mask_path):
            # check if the temporary file already exists
            os.remove(dlg.aligned_mask_path)
    
        cv2.imwrite(dlg.aligned_mask_path, aligned_mask)

        addImg(dlg.aligned_mask_path,"Aligned Mask",dlg.SourceImg_canvas, True) # show aligned mask in side-by-side
        addImg(dlg.aligned_mask_path,"Aligned Mask",dlg.Full_mapCanvas_3, False) # show aligned mask in full view
        
        # add layers to combo box
        dlg.Layer_comboBox.insertItem(0,"Aligned Mask")
        dlg.Layer_comboBox.insertItem(0,"Aligned Image") 

    removeLayer(dlg.SourceImg_canvas, dlg.SourceImg_canvas.layers()[1]) # remove the original image
    removeLayer(dlg.SourceImg_canvas, dlg.SourceImg_canvas.layers()[0]) # remove the source image CPs
    removeLayer(dlg.DestImg_canvas, dlg.DestImg_canvas.layers()[0]) # remove the destination image CPs
    
    addImg(dlg.aligned_img_path,"Aligned Image",dlg.SourceImg_canvas, True) # show aligned image in side-by-side
    addImg(dlg.DestImg_lineEdit.text(),"Destination Image",dlg.Full_mapCanvas_3, False) # show input image in full view
    addImg(dlg.aligned_img_path,"Aligned Image",dlg.Full_mapCanvas_3, False) # show aligned image in full view

    # re-center VP
    dlg.DestImg_canvas.setExtent(dlg.DestImg_canvas.layers()[0].extent())
    dlg.DestImg_canvas.refresh()

    enableTools(dlg)
    enableButtons(dlg, False)

def undoAlign(dlg):
    """Undoes alignment and re-instates control points but keeps info in CP table"""

    # access canvas layers
    source_cps = QgsProject.instance().mapLayersByName('Source CP Layer')[0]
    dest_cps = QgsProject.instance().mapLayersByName('Dest CP Layer')[0]
    source_img = QgsProject.instance().mapLayersByName('Source Image')[0]
    dest_img = QgsProject.instance().mapLayersByName('Destination Image')[0]

    # reset to side-by-side view
    canvas_list_4 = [dlg.SourceImg_canvas, dlg.DestImg_canvas, dlg.Full_mapCanvas_3]
    sideBySide(canvas_list_4, [dlg.Swipe_toolButton_3, dlg.Transparency_slider_3],dlg.SideBySide_pushButton_3, dlg.SingleView_pushButton_3)
    dlg.SingleView_pushButton_3.setEnabled(False)
    dlg.SideBySide_pushButton_3.setEnabled(False)
    dlg.Layer_comboBox.clear()
    dlg.Layer_comboBox.setEnabled(False)

    # reset layer list and canvas extent in side-by-side view
    dlg.Full_mapCanvas_3.setLayers([])
    dlg.SourceImg_canvas.setLayers([source_cps, source_img])
    dlg.DestImg_canvas.setLayers([dest_cps, dest_img])
    dlg.DestImg_canvas.setExtent(dlg.DestImg_canvas.layers()[1].extent())
    dlg.DestImg_canvas.refresh()

    enableButtons(dlg, True)

def saveAlign(dlg):
    """Saves aligned image or mask to specified location"""

    img_to_save = dlg.Layer_comboBox.currentText()

    if img_to_save == "" or img_to_save == "Aligned Image":
        if dlg.aligned_img_path is None:
            errorMessage("No aligned image")
            return
        aligned_img = cv2.imread(dlg.aligned_img_path)
    else:
        if dlg.aligned_mask_path is None:
            errorMessage("No aligned mask")
            return
        aligned_img = cv2.imread(dlg.aligned_mask_path)

    # open save dialog and save aligned image
    dialog = QFileDialog()
    dialog.setOption(dialog.DontUseNativeDialog)
    dialog.setNameFilter("TIFF format (*.tiff *.TIFF)")
    dialog.setDefaultSuffix("tiff")
    dialog.setAcceptMode(QFileDialog.AcceptSave)

    if dialog.exec_():
        align_path = dialog.selectedFiles()[0]
        cv2.imwrite(align_path, aligned_img)
    
        if img_to_save == "" or img_to_save == "Aligned Image":
            dlg.refresh_dict["Align"]["Img"]=align_path
        else:
            dlg.refresh_dict["Align"]["Mask"]=align_path 

        if dlg.aligned_probs_path:
            #check for probability layer (PyLC output)
            aligned_probs = np.load(dlg.aligned_probs_path)
            mask_path, ext = os.path.splitext(align_path)
            probs_path = os.path.join(mask_path + '.npy')
            np.save(probs_path, aligned_probs)