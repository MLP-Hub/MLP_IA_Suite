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

from qgis.core import QgsRasterLayer, QgsProcessing, QgsProject, QgsRasterFileWriter, QgsRasterPipe
from qgis.PyQt.QtWidgets import QFileDialog, QProgressDialog, QGraphicsScene, QGraphicsPixmapItem, QMessageBox
from qgis.PyQt.QtGui import QImage, QPixmap
from qgis.PyQt.QtCore import Qt
from qgis import processing

import skimage
import numpy as np
import math
import scipy
import cv2
import tempfile
import os

from .interface_tools import errorMessage, loadLayer
from .vp_creation import reprojectDEM
from .refresh import messageBox

def initCamParams(dlg):
    """Read initial camera parameters from text file"""

    cam_filepath = dlg.CamParam_lineEdit.text() # get filepath to text file with camera parameters

    cam_file = open(cam_filepath, "r")

    # read camera parameters from file
    cam_params = {}

    try:
        for line in cam_file:
            (key, val) = line.split(":")
            cam_params[key] = val
            if cam_params[key] != 'None\n':
                cam_params[key] = float(val)
            else:
                cam_params[key]=None # set elevation to None if not provided
    except ValueError:
        errorMessage("Incorrect value or formatting in input camera parameters.")
        return

    cam_file.close()

    return cam_params

def camXY(DEM_layer, lat, lon):
    """Determines pixel coordinates of camera position"""

    # get DEM extents and spatial resolution
    ex = DEM_layer.extent() 
    pixelSizeX = DEM_layer.rasterUnitsPerPixelX()
    pixelSizeY = DEM_layer.rasterUnitsPerPixelY()
    
    # get top left coordinate
    ymax = ex.yMaximum()
    xmin = ex.xMinimum()

    # get camera position in pixel coordinates
    cam_x = int((lat - xmin)/pixelSizeX)
    cam_y = int((ymax - lon)/pixelSizeY)

    return cam_x, cam_y, pixelSizeX, pixelSizeY


def closest_argmin(A, B):
    """Finds closest matches between two arrays"""

    L = B.size
    sidx_B = B.argsort()
    sorted_B = B[sidx_B]
    sorted_idx = np.searchsorted(sorted_B, A)
    sorted_idx[sorted_idx==L] = L-1
    mask = (sorted_idx > 0) & \
    ((np.abs(A - sorted_B[sorted_idx-1]) < np.abs(A - sorted_B[sorted_idx])) )

    return sorted_idx-mask

def drawViewshed(dlg):
    """Get real world coordinates for each pixel in the mask"""

    # read the camera parameters from provided text file
    cam_params = initCamParams(dlg) 
    if cam_params is None:
        return
    
    # read DEM from user input
    DEM_path = os.path.realpath(dlg.DEM_lineEdit.text())
    DEM_layer = QgsRasterLayer(DEM_path, "VS_DEM")
    if DEM_layer is None:
        errorMessage('Invalid DEM file')
        return
    
    # Check that DEM has appropriate CRS
    source_crs = DEM_layer.crs() # get current CRS
    if source_crs.mapUnits() != 0:
        # if CRS is not in meters
        try:
            DEM_path, DEM_layer = reprojectDEM(DEM_layer)
        except TypeError:
            return
        
    DEM_img = skimage.io.imread(DEM_path) # read DEM into image array
    
    # check camera is on DEM
    ex = DEM_layer.extent()
    if ex.yMaximum() < float(cam_params['lon']) or ex.yMinimum() > float(cam_params['lon']) or ex.xMaximum() < float(cam_params['lat']) or ex.xMinimum() > float(cam_params['lat']):
        errorMessage("Camera off DEM")
        return
    
    # read mask from user input
    mask_path = os.path.realpath(dlg.AlignMask_lineEdit.text())
    mask = cv2.imread(mask_path)

    # check for probability layer and read it in
    mask_name, ext = os.path.splitext(os.path.realpath(dlg.AlignMask_lineEdit.text()))
    probs_path = os.path.join(mask_name + '.npy')
    probs = None
    if os.path.isfile(probs_path):
        probs = np.load(probs_path)

    img_h, img_w, *_ = mask.shape # get height and width of mask
    cam_x, cam_y, pixelSizeX, pixelSizeY = camXY(DEM_layer, cam_params["lat"], cam_params["lon"]) # find pixel coordinates of camera position and raster resolution
    v_fov = cam_params["h_fov"]*img_h/img_w # determine vertical field of view from horizontal field of view and picture size

    # get elevation from DEM if not provided by user
    if cam_params["elev"] is None:
        cam_params["elev"] = DEM_img[cam_y, cam_x]
    
    # create blank viewshed (same size as DEM)
    dem_h, dem_w, *_ = DEM_img.shape
    vs = np.ones((dem_h,dem_w, 3),dtype=np.uint8)
    # create probability layer if using
    probs_lyr = None
    if probs is not None:
        probs_lyr = np.ones((dem_h,dem_w),dtype=np.float32)

    a = cam_params["azi"] - cam_params["h_fov"]/2 # starting ray angle is azimuth minus half of horizontal FOV
    
    img_v_angles = np.linspace(-v_fov/2, v_fov/2, img_h) # create list of image angles
    img_v_angles = np.tan(np.radians(img_v_angles)) # find ratio (opp/adj)

    xmins, xmaxs, ymins, ymaxs = [], [], [], [] # initiate variables to find visible extent for clipping later

    progressDlg = QProgressDialog("Creating viewshed...","Cancel", 0, img_w)
    progressDlg.setWindowModality(Qt.WindowModal)
    progressDlg.setValue(0)
    progressDlg.forceShow()
    progressDlg.show()  

    for img_x in range(0, img_w):
        # First, find all visible pixels on DEM and match to image coordinates based on angles
        # then, load mask colors (landcover) into empty array at cooresponding positions

        progressDlg.setValue(img_x)

        # create ray start position (remove first 100 m)
        ray_start_y = cam_y - (100*math.cos(np.radians(a))/pixelSizeY)
        ray_start_x = cam_x + (100*math.sin(np.radians(a))/pixelSizeX)
        
        # create ray end position
        ray_end_y = cam_y - (7000*math.cos(np.radians(a))/pixelSizeY)
        ray_end_x = cam_x + (7000*math.sin(np.radians(a))/pixelSizeX)

        xs = np.linspace(ray_start_x, ray_end_x, round(100000/pixelSizeX))
        ys = np.linspace(ray_start_y, ray_end_y, round(100000/pixelSizeY))

        elevs = scipy.ndimage.map_coordinates(DEM_img, np.vstack((ys,xs)), order = 1) - cam_params["elev"]-cam_params["hgt"]

        opp = (xs - cam_x)*pixelSizeX
        adj = (cam_y - ys)*pixelSizeY

        dem_dist = np.sqrt(np.add(opp**2, adj**2))
        vert_angles = np.divide(elevs,dem_dist) # find ratio (opp/adj)

        dem_angles_inc = np.fmax.accumulate(vert_angles) # checks for only increasing DEM angles
        unique_angles, unique_angles_indx = np.unique(dem_angles_inc, return_index=True) # keep only unique increasing angles and their index

        xs_visible = xs[unique_angles_indx].astype(int) # keep only visible x-coordinates
        ys_visible = ys[unique_angles_indx].astype(int) # keep only visible y-coordinates

        # add max and min extents
        xmins.append(min(xs_visible))
        xmaxs.append(max(xs_visible))
        ymins.append(min(ys_visible))
        ymaxs.append(max(ys_visible))

        # fund matching angles
        angle_matches_index = closest_argmin(unique_angles, img_v_angles)

        mask_col = mask[:,img_x] # get column from mask
        mask_col = np.flip(mask_col, axis=0) # reverse order of pixels
        mask_col = mask_col[angle_matches_index] # keep only the visible pixels

        vs[ys_visible, xs_visible] = mask_col

        # if probabilities exist, also fill into probs layer
        if probs_lyr is not None:
            probs_col = probs[:,img_x]
            probs_col = np.flip(probs_col, axis=0)
            probs_col = probs_col[angle_matches_index]
            probs_lyr[ys_visible, xs_visible] = probs_col
            
        a = a+(cam_params["h_fov"]/img_w) # update ray angle

    # find visible extents to clip VS later   
    xmin = ex.xMinimum() + min(xmins)*pixelSizeX
    xmax = ex.xMinimum() + max(xmaxs)*pixelSizeX
    ymax = ex.yMaximum() - min(ymins)*pixelSizeY
    ymin = ex.yMaximum() - max(ymaxs)*pixelSizeY
    vis_ex = [xmin, xmax, ymin, ymax]

    return vs, DEM_layer, vis_ex, probs_lyr

def singleBand(vs):
    """Converts viewshed to singleband layer based on PyLC classes"""

    # convert to singleband
    vs_sb = np.sum(vs, 2)

    # convert to correct legend (1 to 8 for classes, 0 is ND)
    vs_sb[vs_sb == 3] = 0
    vs_sb[vs_sb == 207] = 1
    vs_sb[vs_sb == 420] = 2
    vs_sb[vs_sb == 376] = 3
    vs_sb[vs_sb == 227] = 4
    vs_sb[vs_sb == 255] = 5
    vs_sb[vs_sb == 413] = 6
    vs_sb[vs_sb == 259] = 7
    vs_sb[vs_sb == 489] = 8

    vs_sb_int = vs_sb.astype(int)
    
    return(vs_sb_int)

def clipVSLayer(vs_layer, vis_ex):
    """Clip viewshed to visible area"""

    extents = ", ".join(str(e) for e in vis_ex)

    parameters = {'INPUT':vs_layer,
                    'PROJWIN':extents,
                    'OVERCRS':None,
                    'NODATA':None,
                    'OPTIONS':None,
                    'DATA_TYPE':None,
                    'EXTRA':None,
                    'OUTPUT':QgsProcessing.TEMPORARY_OUTPUT
            }

    try:
        clipped_vs = processing.run("gdal:cliprasterbyextent", parameters)
    
        clipVS_path=clipped_vs['OUTPUT']
        clipVS_layer = QgsRasterLayer(clipVS_path, "Clipped VS") 
    except:
        errorMessage("Viewshed failed")
        return
    
    return clipVS_path, clipVS_layer

def createVSLayer(vs_path, DEM_lyr, vis_ex):
    """Converts viewshed image to useable layer"""

    ext = DEM_lyr.extent() 
    ext_list = ["-a_ullr",str(ext.xMinimum()),str(ext.yMaximum()),str(ext.xMaximum()), str(ext.yMinimum())]
    ullr = " ".join(ext_list)

    PARAMS = { 'COPY_SUBDATASETS' : False, 
              'DATA_TYPE' : 0, 
              'EXTRA' : ullr, 
              'INPUT' : vs_path, 
              'NODATA' : "0", 
              'OPTIONS' : '', 
              'OUTPUT' : QgsProcessing.TEMPORARY_OUTPUT, 
              'TARGET_CRS' : None }

    vs_ref=processing.run("gdal:translate", PARAMS)

    vs_ref_path=vs_ref['OUTPUT']
    vs_layer = QgsRasterLayer(vs_ref_path, "Viewshed")

    vs_clip_path, vs_clip_layer = clipVSLayer(vs_layer, vis_ex)

    return vs_clip_path, vs_clip_layer  

def showMask(dlg):
    """Adds aligned mask to QGraphics View"""

    scene = QGraphicsScene()
    dlg.Mask_graphic.setScene(scene)
    mask_path = os.path.realpath(dlg.AlignMask_lineEdit.text())
    aligned_mask = QImage(mask_path)
    w,h = aligned_mask.width(), aligned_mask.height() # get mask dimensions

    pic = QGraphicsPixmapItem()
    pic.setPixmap(QPixmap.fromImage(aligned_mask))
    scene.addItem(pic)

    dlg.Mask_graphic.fitInView(0,0,w,h, Qt.KeepAspectRatio)

def enableTools(dlg):
    """Enables canvas tools once canvas is populated with mask and image"""

    dlg.Transparency_slider_4.setEnabled(True)
    dlg.Swipe_toolButton_4.setEnabled(True)
    dlg.Fit_toolButton_4.setEnabled(True)
    dlg.Pan_toolButton_4.setEnabled(True)

def createVS(dlg):
    """Creates and displays viewshed"""

    # first check if temp VS exists unsaved
    if dlg.refresh_dict["VS"]["VS"] is None and dlg.vs_path is not None:
        ret = messageBox("Viewshed")
        if ret == QMessageBox.No:
            return
    try:
        vs, DEM_layer, vis_ex, probs_lyr = drawViewshed(dlg) # create viewshed using ray tracing
    except TypeError:
        errorMessage("Viewshed creation failed.")
        return
    
    if vs is None:
        # break if create VS did not work
        errorMessage("Viewshed creation failed.")
        return
    
    if dlg.image_checkBox.isChecked():
        vs = singleBand(vs) # if PyLC mask provided, convert to singleband
    
    # save vs to temp path
    dlg.vs_path = os.path.join(tempfile.mkdtemp(), 'tempVS.tiff')
    if os.path.isfile(dlg.vs_path):
        # check if the temporary file already exists
        os.remove(dlg.vs_path)
    
    cv2.imwrite(dlg.vs_path, vs) # write viewshed to image

    dlg.vs_path, vs_ref_layer = createVSLayer(dlg.vs_path, DEM_layer, vis_ex) # convert viewshed from image to referenced raster layer, update path to VS layer

    # if probabilities exist, convert to image and save to temp path
    if probs_lyr is not None:
        probs_img = (1-probs_lyr)*255
        dlg.probs_lyr_path = os.path.join(tempfile.mkdtemp(), 'tempProbs.tiff')
        if os.path.isfile(dlg.probs_lyr_path):
            # check if the temporary file already exists
            os.remove(dlg.probs_lyr_path)
    
        cv2.imwrite(dlg.probs_lyr_path, probs_img) # write viewshed to image
        dlg.probs_lyr_path, probs_ref_layer = createVSLayer(dlg.probs_lyr_path, DEM_layer, vis_ex) # convert viewshed from image to referenced raster layer, update path to VS layer


    QgsProject.instance().addMapLayer(vs_ref_layer, False) # add layer to the registry (but don't load into main map)
    QgsProject.instance().addMapLayer(DEM_layer, False) # add layer to the registry (but don't load into main map)

    if dlg.image_checkBox.isChecked():
        # if using PyLC mask, add style
        dir_path = os.path.dirname(__file__)
        dir_path = os.path.normpath(dir_path)
        style_path = os.path.join(dir_path, "sb_PyLC_style.qml") # path to file containing layer style

        vs_ref_layer.loadNamedStyle(style_path)

    #remove any existing layers, then add VS and DEM to map
    layer_list = dlg.VS_mapCanvas.layers()
    for lyr in layer_list:
        QgsProject.instance().removeMapLayer(lyr.id()) # if the layer already exists, remove it from the canvas
    loadLayer(dlg.VS_mapCanvas, vs_ref_layer)
    loadLayer(dlg.VS_mapCanvas, DEM_layer)
    
    showMask(dlg) # show input mask in side-by-side

    enableTools(dlg)


def writeRaster(layer, save_path, crs):
    """Write raster to file"""
    file_writer = QgsRasterFileWriter(save_path)
    pipe = QgsRasterPipe()
    provider = layer.dataProvider()
    ctc=QgsProject.instance().transformContext()

    if not pipe.set(provider.clone()):
        errorMessage("Cannot set pipe provider")

    file_writer.writeRaster(
        pipe,
        provider.xSize(),
        provider.ySize(),
        provider.extent(),
        crs, ctc)

def saveVS(dlg):
    """Saves viewshed to specified location"""

    vs = QgsRasterLayer(dlg.vs_path)
    save_vs_path = None

    # open save dialog
    dialog = QFileDialog()
    dialog.setOption(dialog.DontUseNativeDialog)
    dialog.setNameFilter("TIFF format (*.tiff *.TIFF)")
    dialog.setDefaultSuffix("tiff")
    dialog.setAcceptMode(QFileDialog.AcceptSave)

    if dialog.exec_():
        save_vs_path = dialog.selectedFiles()[0]

        # get CRS from DEM
        DEM_path = os.path.realpath(dlg.DEM_lineEdit.text())
        DEM_layer = QgsRasterLayer(DEM_path, "DEM")
        dest_crs = DEM_layer.crs()

        writeRaster(vs, save_vs_path, dest_crs)

        if dlg.probs_lyr_path is not None:
            probs = QgsRasterLayer(dlg.probs_lyr_path)
            save_path, ext = os.path.splitext(save_vs_path)
            save_probs_path = os.path.join(save_path + '_probs.tiff')
            writeRaster(probs, save_probs_path, dest_crs)

        dlg.refresh_dict["VS"]["VS"]=save_vs_path