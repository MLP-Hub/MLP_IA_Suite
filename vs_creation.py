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

from qgis.core import QgsRasterLayer, QgsProcessing, QgsProject, QgsRasterTransparency
from qgis.PyQt.QtWidgets import QFileDialog, QProgressDialog, QGraphicsScene, QGraphicsPixmapItem
from qgis.PyQt.QtGui import QImage, QPixmap
from qgis.PyQt.QtCore import Qt
from qgis import processing

from .interface_tools import addImg, loadLayer

import skimage
import numpy as np
import math
import scipy
import cv2
import tempfile
import os

def initCamParams(dlg):
    """Read initial camera parameters from text file"""

    cam_filepath = dlg.CamParam_lineEdit.text() # get filepath to text file with camera parameters

    cam_file = open(cam_filepath, "r")

    # read camera parameters from file
    cam_params = {}

    for line in cam_file:
        (key, val) = line.split(":")
        cam_params[key] = val
        if cam_params[key] != 'None\n':
            cam_params[key] = float(val)
        else:
            cam_params[key]=None # set elevation to None if not provided

    return cam_params

def camXY(DEM_layer, lat, lon):
    """Determines pixel coordinates of camera position"""

    # get hillshade extents and spatial resolution
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

    # read DEM from user input
    DEM_path = os.path.realpath(dlg.DEM_lineEdit.text())
    DEM_layer = QgsRasterLayer(DEM_path, "DEM")
    DEM_img = skimage.io.imread(DEM_path) # read clipped DEM into image array

    # read mask from user input
    mask_path = os.path.realpath(dlg.AlignMask_lineEdit.text())
    mask = cv2.imread(mask_path)

    # read the camera parameters from provided text file
    cam_params = initCamParams(dlg) 

    img_h, img_w, *_ = mask.shape # get height and width of mask
    cam_x, cam_y, pixelSizeX, pixelSizeY = camXY(DEM_layer, cam_params["lat"], cam_params["lon"]) # find pixel coordinates of camera position and raster resolution
    v_fov = cam_params["h_fov"]*img_h/img_w # determine vertical field of view from horizontal field of view and picture size

    # get elevation from DEM if not provided by user
    if cam_params["elev"] is None:
        cam_params["elev"] = DEM_img[cam_y, cam_x]
    
    # create blank viewshed (same size as DEM)
    dem_h, dem_w, *_ = DEM_img.shape
    vs = np.zeros((dem_h,dem_w, 3),dtype=np.uint8) 

    a = cam_params["azi"] - cam_params["h_fov"]/2 # starting ray angle is azimuth minus half of horizontal FOV
    
    img_v_angles = np.linspace(-v_fov/2, v_fov/2, img_h) # create list of image angles
    img_v_angles = np.tan(np.radians(img_v_angles)) # find ratio (opp/adj)

    for img_x in range(0, img_w):
        # First, find all visible pixels on DEM and match to image coordinates based on angles
        # then, load mask colors (landcover) into empty array at cooresponding positions

        # create ray start position (remove first 100 m)
        ray_start_y = cam_y - (100*math.cos(np.radians(a))/pixelSizeY)
        ray_start_x = cam_x + (100*math.sin(np.radians(a))/pixelSizeX)
        
        # create ray end position
        ray_end_y = cam_y - (25000*math.cos(np.radians(a))/pixelSizeY)
        ray_end_x = cam_x + (25000*math.sin(np.radians(a))/pixelSizeX)

        xs = np.linspace(ray_start_x, ray_end_x, round(25000/pixelSizeX))
        ys = np.linspace(ray_start_y, ray_end_y, round(25000/pixelSizeY))

        elevs = scipy.ndimage.map_coordinates(DEM_img, np.vstack((ys,xs)), order = 1) - cam_params["elev"]+cam_params["hgt"]

        opp = (xs - cam_x)*pixelSizeX
        adj = (cam_y - ys)*pixelSizeY

        dem_dist = np.sqrt(np.add(opp**2, adj**2))
        vert_angles = np.divide(elevs,dem_dist) # find ratio (opp/adj)

        dem_angles_inc = np.fmax.accumulate(vert_angles) # checks for only increasing DEM angles
        unique_angles, unique_angles_indx = np.unique(dem_angles_inc, return_index=True) # keep only unique increasing angles and their index

        xs_visible = xs[unique_angles_indx].astype(int) # keep only visible x-coordinates
        ys_visible = ys[unique_angles_indx].astype(int) # keep only visible y-coordinates

        angle_matches_index = closest_argmin(unique_angles, img_v_angles)
        mask_col = mask[:,img_x] # get column from mask
        mask_col = np.flip(mask_col, axis=0) # reverse order of pixels
        mask_col = mask_col[angle_matches_index] # keep only the non-sky pixels

        vs[ys_visible, xs_visible] = mask_col
            
        a = a+(cam_params["h_fov"]/img_w) # update ray angle

    return vs, DEM_layer

def createVSLayer(vs_path, DEM_lyr):
    """Converts viewshed image to useable layer"""

    ext = DEM_lyr.extent() 
    ext_list = ["-a_ullr",str(ext.xMinimum()),str(ext.yMaximum()),str(ext.xMaximum()), str(ext.yMinimum())]
    ullr = " ".join(ext_list)


    PARAMS = { 'COPY_SUBDATASETS' : False, 
              'DATA_TYPE' : 0, 
              'EXTRA' : ullr, 
              'INPUT' : vs_path, 
              'NODATA' : None, 
              'OPTIONS' : '', 
              'OUTPUT' : QgsProcessing.TEMPORARY_OUTPUT, 
              'TARGET_CRS' : None }

    vs_ref=processing.run("gdal:translate", PARAMS)

    vs_ref_path=vs_ref['OUTPUT']
    vs_layer = QgsRasterLayer(vs_ref_path, "Viewshed")

    return vs_ref_path, vs_layer  

def setVSTransparency(vs_layer):
    """Sets black (no data) areas to  transparent"""
    
    raster_transparency  = vs_layer.renderer().rasterTransparency()
    pixel = QgsRasterTransparency.TransparentThreeValuePixel()
    pixel.red, pixel.green, pixel.blue, pixel.percentTransparent = 0, 0, 0, 100
    raster_transparency.setTransparentThreeValuePixelList([pixel])
    vs_layer.triggerRepaint()


def showMask(dlg):
    """Adds aligned mask to QGraphics View"""

    scene = QGraphicsScene()
    dlg.Mask_graphic.setScene(scene)
    mask_path = os.path.realpath(dlg.AlignMask_lineEdit.text())
    aligned_mask = QImage(mask_path)
    w,h = aligned_mask.width(), aligned_mask.height() # get mask dimensions
    #rect = aligned_mask.rect()

    pic = QGraphicsPixmapItem()
    pic.setPixmap(QPixmap.fromImage(aligned_mask))
    #scene.setSceneRect(0,0,w,h)
    scene.addItem(pic)

    dlg.Mask_graphic.fitInView(0,0,w,h, Qt.KeepAspectRatio)

def enableTools(dlg):
    """Enables canvas tools once canvas is populated with mask and image"""

    dlg.Transparency_slider_4.setEnabled(True)
    dlg.Swipe_toolButton_4.setEnabled(True)
    dlg.Fit_toolButton_4.setEnabled(True)
    dlg.Pan_toolButton_4.setEnabled(True)

def displayVS(dlg):
    """Creates and displays viewshed"""

    vs, DEM_layer = drawViewshed(dlg) # create viewshed using ray tracing
    
    if vs is None:
        # break if create VP did not work
        return
    
    # save vs to temp path
    dlg.vs_path = os.path.join(tempfile.mkdtemp(), 'tempVS.tiff')
    if os.path.isfile(dlg.vs_path):
        # check if the temporary file already exists
        os.remove(dlg.vs_path)
    
    cv2.imwrite(dlg.vs_path, vs) # write viewshed to image

    dlg.vs_path, vs_ref_layer = createVSLayer(dlg.vs_path, DEM_layer) # convert viewshed from image to referenced raster layer, update path to VS layer

    QgsProject.instance().addMapLayer(vs_ref_layer, False) # add layer to the registry (but don't load into main map)
    QgsProject.instance().addMapLayer(DEM_layer, False) # add layer to the registry (but don't load into main map)
    setVSTransparency(vs_ref_layer) # set black to transparent
    
    loadLayer(dlg.VS_mapCanvas, vs_ref_layer)
    loadLayer(dlg.VS_mapCanvas, DEM_layer)
    
    showMask(dlg) # show input image in side-by-side

    enableTools(dlg)

def saveVS(dlg):
    """Saves viewshed to specified location"""

    vs = cv2.imread(dlg.vs_path)
    save_vs_path = None

    # open save dialog and save aligned image
    dialog = QFileDialog()
    dialog.setOption(dialog.DontUseNativeDialog)
    dialog.setNameFilter("TIFF format (*.tiff *.TIFF)")
    dialog.setDefaultSuffix("tiff")
    dialog.setAcceptMode(QFileDialog.AcceptSave)

    if dialog.exec_():
        save_vs_path = dialog.selectedFiles()[0]

    cv2.imwrite(save_vs_path, vs)
