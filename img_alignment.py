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
from .interface_tools import addImg, removeLayer

from qgis.gui import QgsMapToolEmitPoint, QgsMapToolIdentifyFeature
from qgis.core import QgsVectorLayer, QgsField, QgsProject, QgsFeature, QgsGeometry, QgsMarkerSymbol, QgsFontMarkerSymbolLayer
from PyQt5.QtCore import Qt, QVariant
from qgis.PyQt.QtWidgets import QTableWidgetItem, QFileDialog

import cv2
import numpy as np
import tempfile
import os

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

def add_cp(dlg, point, canvas, name):
    """Adds CP (from mouse click) to vector layer"""

    vl = QgsProject.instance().mapLayersByName(name)[0] # get CP vector layer

    # add new cp to vector layer
    vl.startEditing() # enter editing mode
    pr = vl.dataProvider()
    fet = QgsFeature()
    fet.setGeometry(QgsGeometry.fromPointXY(point))
    fet.setAttributes([point.x(),point.y()]) # add point with x,y coords where the user clicked
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
    if dlg.CP_table.rowCount() <= row:
        dlg.CP_table.setRowCount(row+1) # add row if number of CPs exceeds table size
    dlg.CP_table.setItem(row, col, QTableWidgetItem(str(round(point.x(),2))))
    dlg.CP_table.setItem(row, col+1, QTableWidgetItem(str(abs(round(point.y(),2)))))

    # When done, switch canvases
    canvas.unsetMapTool(dlg.CPtool)
    dlg.CPtool = None
    if name == "Source CP Layer":
        newCP(dlg, dlg.DestImg_canvas, "Dest CP Layer", "Destination Image")
    else:
        newCP(dlg, dlg.SourceImg_canvas, "Source CP Layer", "Source Image")

def newCP(dlg, canvas, name, img_name):
    """Allows user to create new control point"""

    # check if vector layer of CPs exists, otherwise create one
    layer_list = canvas.layers()
    lyr_ids = {}
    for lyr in layer_list:
        lyr_ids[lyr.name()] = lyr.id()
    if name not in lyr_ids:
       createCPLayer(canvas, name, img_name) 

    dlg.CPtool = QgsMapToolEmitPoint(canvas) # create control point tool
    dlg.CPtool.canvasClicked.connect(lambda point: add_cp(dlg, point, canvas, name)) # add cp to canvas when user clicks the image
    dlg.CPtool.setCursor(Qt.CrossCursor) # use crosshair cursor
    canvas.setMapTool(dlg.CPtool) # set the map tool for the canvas

def deleteCP(f, vl_list, table):
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
    
def selectCP(dlg, canvas_list):
    """Allows user to select control point from map"""

    # should be able to delete from table or layer
    # active on both canvases - must delete corresponding CP from other canvas
    # check for vector layer type
    vl1 = canvas_list[0].layers()[0] # get CP layer for canvas (should always be top layer)
    dlg.delTool1 = QgsMapToolIdentifyFeature(canvas_list[0])
    dlg.delTool1.setLayer(vl1)
    canvas_list[0].setMapTool(dlg.delTool1)

    vl2 = canvas_list[1].layers()[0] # get CP layer for canvas (should always be top layer)
    dlg.delTool2 = QgsMapToolIdentifyFeature(canvas_list[1])
    dlg.delTool2.setLayer(vl2)
    canvas_list[1].setMapTool(dlg.delTool2)

    dlg.delTool1.featureIdentified.connect(lambda f: deleteCP(f, [vl1,vl2], dlg.CP_table))
    dlg.delTool2.featureIdentified.connect(lambda f: deleteCP(f, [vl1,vl2], dlg.CP_table))

def saveCPs(dlg):
    """Allows user to save set of control points"""

def loadCPs(dlg):
    """Allows user to load a set of saved control points"""

def enableTools(dlg):
    """Enables canvas tools once canvas is populated with aligned image"""

    dlg.SideBySide_pushButton_3.setEnabled(True)
    dlg.SingleView_pushButton_3.setEnabled(True)
    dlg.Fit_toolButton_3.setEnabled(True)
    dlg.Pan_toolButton_3.setEnabled(True)

def alignImgs(dlg, source_img_path, table):
    """Aligns images using perspective transformation"""

    img = cv2.imread(source_img_path)
    (h, w) = img.shape[:2]

    # read control points from table
    source_pts = []
    dest_pts = []
    x = 0
    while x < table.rowCount():
        source_pt = [float(table.item(x, 0).text()),float(table.item(x, 1).text())] 
        dest_pt = [float(table.item(x, 2).text()),float(table.item(x, 3).text())] 
        source_pts.append(source_pt)
        dest_pts.append(dest_pt)
        x+=1

    source_pts_array = np.float32(source_pts)
    dest_pts_array = np.float32(dest_pts)

    # get best control points based on homography and RANSAC method
    homography, mask = cv2.findHomography(source_pts_array, dest_pts_array, cv2.RANSAC,5.0)
    mask = mask.flatten()

    index = np.nonzero(mask)
    
    source_pts_good = source_pts_array[index]
    dest_pts_good = dest_pts_array[index]
    
    # align image using perspective transform
    dest_img = cv2.imread(dlg.DestImg_lineEdit.text())
    h,w = dest_img.shape[:2] # automatically clip to the destination image
    matrix = cv2.getPerspectiveTransform(source_pts_good, dest_pts_good)
    aligned_img = cv2.warpPerspective(img, matrix, (w, h))

    # save aligned image to temporary file
    dlg.aligned_img_path = os.path.join(tempfile.mkdtemp(), 'alignedImg.tiff')
    if os.path.isfile(dlg.aligned_img_path):
        # check if the temporary file already exists
        os.remove(dlg.aligned_img_path)
    
    cv2.imwrite(dlg.aligned_img_path, aligned_img)

    removeLayer(dlg.SourceImg_canvas, dlg.SourceImg_canvas.layers()[1]) # remove the original image
    removeLayer(dlg.SourceImg_canvas, dlg.SourceImg_canvas.layers()[0]) # remove the source image CPs
    removeLayer(dlg.DestImg_canvas, dlg.DestImg_canvas.layers()[0]) # remove the destination image CPs

    addImg(dlg.DestImg_lineEdit.text(),"Destination Image",dlg.Full_mapCanvas_3, False) # show input image in full view
    addImg(dlg.aligned_img_path,"Aligned Image",dlg.SourceImg_canvas, True) # show aligned image in side-by-side
    addImg(dlg.aligned_img_path,"Aligned Image",dlg.Full_mapCanvas_3, False) # show output mask in fullview

    # re-center VP
    dlg.DestImg_canvas.setExtent(dlg.DestImg_canvas.layers()[0].extent())
    dlg.DestImg_canvas.refresh()

    enableTools(dlg)

def automatedAlignment(dlg):
    """Test function for automated image alignment"""
    MIN_MATCH_COUNT = 10

    # source_img = cv2.imread(dlg.SourceImg_lineEdit.text(), cv2.IMREAD_GRAYSCALE) # queryImage
    # dest_img = cv2.imread(dlg.DestImg_lineEdit.text(), cv2.IMREAD_GRAYSCALE) # trainImage
    img1 = cv2.imread(dlg.SourceImg_lineEdit.text()) # queryImage
    img2 = cv2.imread(dlg.DestImg_lineEdit.text()) # trainImage

    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        h,w = img2.shape[:2]

        aligned_img = cv2.warpPerspective(img1, matrix, (w, h))
        
        # save aligned image to temporary file
        img_path = os.path.join(tempfile.mkdtemp(), 'alignedImg.tiff')
        if os.path.isfile(img_path):
            # check if the temporary file already exists
            os.remove(img_path)
    
        cv2.imwrite(img_path, aligned_img)
        
        removeLayer(dlg.SourceImg_canvas, dlg.SourceImg_canvas.layers()[0]) # remove the original image
        addImg(dlg.DestImg_lineEdit.text(),"Destination Image",dlg.Full_mapCanvas_3, False) # show input image in full view
        addImg(img_path,"Aligned Image",dlg.SourceImg_canvas, True) # show aligned image in side-by-side
        addImg(img_path,"Aligned Image",dlg.Full_mapCanvas_3, False) # show output mask in fullview

        # re-center VP
        dlg.DestImg_canvas.setExtent(dlg.DestImg_canvas.layers()[0].extent())
        dlg.DestImg_canvas.refresh()

        enableTools(dlg)

    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )

def saveAlign(dlg):
    """Saves aligned image to specified location"""

    aligned_img = cv2.imread(dlg.aligned_img_path)
    align_path = None

    # open save dialog and save aligned image
    dialog = QFileDialog()
    dialog.setOption(dialog.DontUseNativeDialog)
    dialog.setNameFilter("TIFF format (*.tiff *.TIFF)")
    dialog.setDefaultSuffix("tiff")
    dialog.setAcceptMode(QFileDialog.AcceptSave)

    if dialog.exec_():
        align_path = dialog.selectedFiles()[0]

    cv2.imwrite(align_path, aligned_img)
