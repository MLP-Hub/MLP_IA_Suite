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

from qgis.PyQt.QtWidgets import QFileDialog
from qgis.core import QgsRasterLayer, QgsProcessing
from qgis import processing

import os.path
import tempfile
import skimage
import cv2
import numpy as np
import math


def readCamParams(dlg):
    """Reads user input camera parameters as dictionary"""

    cam_params = {}

    # Get camera parameters from user input
    cam_params["lat"] = float(dlg.Easting_lineEdit.text())
    cam_params["lon"] = float(dlg.Northing_lineEdit.text())
    cam_params["azi"] = float(dlg.Azi_lineEdit.text())
    cam_params["h_fov"] = float(dlg.horFOV_lineEdit.text())
    cam_params["hgt"] = float(dlg.CamHgt_lineEdit.text())
    
    if dlg.Elev_lineEdit.text() == "":
        cam_params["elev"] = None
    else:
        cam_params["elev"] = float(dlg.Elev_lineEdit.text()) # if user specified elevation, read it

    return cam_params
        

def loadCamParam(dlg):
    """This function loads camera parameters from a text file"""
        
    dialog = QFileDialog()
    dialog.setFileMode(QFileDialog.ExistingFile)
    dialog.setOption(dialog.DontUseNativeDialog)
    dialog.setNameFilter("Text files (*.txt)")
    dialog.exec_()
    cam_filepath = dialog.selectedFiles()[0]

    cam_file = open(cam_filepath, "r")

    cam_params = {}

    for line in cam_file:
       (key, val) = line.split(":")
       cam_params[key] = val

    # Read camera parameters into corresponding fields on interface
    dlg.Easting_lineEdit.setText(cam_params["lat"])
    dlg.Northing_lineEdit.setText(cam_params["lon"])
    dlg.Azi_lineEdit.setText(cam_params["azi"])
    dlg.horFOV_lineEdit.setText(cam_params["h_fov"])
    dlg.CamHgt_lineEdit.setText(cam_params["hgt"])

    cam_params["elev"]=cam_params["elev"].strip() # remove newline character from elevation field

    if cam_params["elev"] == "None":  
        dlg.Elev_lineEdit.setText("")
    else:
        dlg.Elev_lineEdit.setText(cam_params["elev"])


def saveCamParam(dlg):
    """This function saves current camera parameters to a text file"""

    dialog = QFileDialog()
    dialog.setOption(dialog.DontUseNativeDialog)
    dialog.setAcceptMode(QFileDialog.AcceptSave)
    dialog.setNameFilter("Text files (*.txt)")
    dialog.exec_()
    cam_filepath = dialog.selectedFiles()[0]
    cam_file = open(cam_filepath, "w")

    cam_params = readCamParams(dlg)
    for key, value in cam_params.items(): 
        cam_file.write('%s:%s\n' % (key, value))

    cam_file.close()

def moveCam(dlg, dir):
    """Moves camera in space relative to azimuth"""

    cam_x = float(dlg.Easting_lineEdit.text())
    cam_y = float(dlg.Northing_lineEdit.text())
    azi = float(dlg.Azi_lineEdit.text())
    step_size_m = float(dlg.StepSizeM_lineEdit.text())

    if dir == "Forward":
        cam_y = cam_y + np.sin(np.radians(azi))*step_size_m
        cam_x = cam_x + np.cos(np.radians(azi))*step_size_m
    elif dir == "Backward":
        cam_y = cam_y - np.sin(np.radians(azi))*step_size_m
        cam_x = cam_x - np.cos(np.radians(azi))*step_size_m
    elif dir == "Left":
        cam_y = cam_y + np.sin(np.radians(azi+90))*step_size_m
        cam_x = cam_x + np.cos(np.radians(azi+90))*step_size_m
    else:
        cam_y = cam_y + np.sin(np.radians(azi-90))*step_size_m
        cam_x = cam_x + np.cos(np.radians(azi-90))*step_size_m

    dlg.Easting_lineEdit.setText(str(round(cam_x,2)))
    dlg.Northing_lineEdit.setText(str(round(cam_y,2)))
        

def camHeight(dlg, dir):
    """Changes camera height"""

    cam_hgt = float(dlg.CamHgt_lineEdit.text())
    step_size_m = float(dlg.StepSizeM_lineEdit.text())

    if dir == "Up":
        cam_hgt = cam_hgt + step_size_m
    else:
        cam_hgt = cam_hgt - step_size_m

    dlg.CamHgt_lineEdit.setText(str(cam_hgt))

def rotateCam(dlg, dir):
    """Changes camera rotation"""

    azi = float(dlg.Azi_lineEdit.text())
    step_size_deg = float(dlg.StepSizeDeg_lineEdit.text())

    if dir == "Clckwise":
        azi = azi + step_size_deg
    else:
        azi = azi - step_size_deg

    dlg.Azi_lineEdit.setText(str(azi))

def getImgDimensions(dlg):
    """Extracts width and height of reference image"""

    img_path = dlg.InputRefImg_lineEdit.text()
    ref_img = cv2.imread(img_path) # read the reference image
    h, w, *_ = ref_img.shape # get image dimensions
    
    return h, w

def createHillshade(dlg, clipped_DEM):
    """Converts DEM to hillshade"""

    # Convert DEM to hillshade and save to temporary file
    parameters = {'INPUT': clipped_DEM, 
                    'BAND': 1,
                    'COMPUTE_EDGES': False,
                    'ZEVENBERGEN': False,
                    'Z_FACTOR': 1.0,
                    'SCALE': 1.0,
                    'AZIMUTH': 315,
                    'COMBINED': False,
                    'ALTITUDE': 45,
                    'MULTIDIRECTIONAL': False,
                    'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT}
    
    hillshade=processing.run("gdal:hillshade", parameters)
    dlg.hillshade_path=hillshade['OUTPUT']
    dlg.hillshade_layer = QgsRasterLayer(dlg.hillshade_path, "Hillshade")   

    # Set hillshade coordinate reference system to NAD83 UTM Zone 12N
    crs = dlg.hillshade_layer.crs()
    crs.createFromId(26912)
    dlg.hillshade_layer.setCrs(crs)

def clipDEM(dlg, cam_x, cam_y):
    """Clips DEM based on camera parameters"""

    # Read DEM from user input
    DEM_path = os.path.realpath(dlg.InputDEM_lineEdit.text())
    DEM_layer = QgsRasterLayer(DEM_path, "DEM")

    # Set DEM coordinate reference system to NAD83 UTM Zone 12N
    crs = DEM_layer.crs()
    crs.createFromId(26912)
    DEM_layer.setCrs(crs)

    # Get clipping extents
    ex = DEM_layer.extent() 
    dem_extents = np.array([ex.xMinimum(), ex.xMaximum(), ex.yMinimum(), ex.yMaximum()]) # get full DEM extents
    clip_extents = np.array([cam_x - 25500, cam_x + 25500, cam_y - 25500, cam_y + 25500]) # get clipped extents
    out = np.max(np.vstack((clip_extents, dem_extents)), axis=0) # ensure clipped extents are still on DEM
    extents = ", ".join(str(e) for e in out)

    parameters = {'INPUT':DEM_layer,
                    'PROJWIN':extents,
                    'OVERCRS':None,
                    'NODATA':None,
                    'OPTIONS':None,
                    'DATA_TYPE':None,
                    'EXTRA':None,
                    'OUTPUT':QgsProcessing.TEMPORARY_OUTPUT
            }

    clipped_DEM = processing.run("gdal:cliprasterbyextent", parameters)
    
    clipDEM_path=clipped_DEM['OUTPUT']
    clipDEM_layer = QgsRasterLayer(clipDEM_path, "Clipped DEM")   
    createHillshade(dlg, clipDEM_layer)

    return clipDEM_path

    
def camXY(dlg, lat, lon):
    """Determines pixel coordinates of camera position"""

    # get hillshade extents and spatial resolution
    ex = dlg.hillshade_layer.extent() 
    pixelSizeX = dlg.hillshade_layer.rasterUnitsPerPixelX()
    pixelSizeY = dlg.hillshade_layer.rasterUnitsPerPixelY()

    # if pixelSizeX != pixelSizeY --> error
    
    # get top left coordinate
    ymax = ex.yMaximum()
    xmin = ex.xMinimum() 

    # get camera position in pixel coordinates
    cam_x = int((lat - xmin)/pixelSizeX)
    cam_y = int((ymax - lon)/pixelSizeY)

    return cam_x, cam_y, pixelSizeX, pixelSizeY

def createVP(dlg):
    """Creates virtual photograph""" 
    
    cam_params = readCamParams(dlg) # read camera parameters

    # generate new clipped DEM and hillshade on first round or when camera moves more than 500 m
    if dlg.lat_init is None or (cam_params['lat'] > (dlg.lat_init + 500)) or (cam_params['lat'] > (dlg.lon_init + 500)):
        dlg.DEM_path = clipDEM(dlg, cam_params["lat"], cam_params["lon"]) # generate clipped DEM
        dlg.lat_init = cam_params['lat'] # replace with new camera position
        dlg.lon_init = cam_params['lon'] 
    
    DEM_img = skimage.io.imread(dlg.DEM_path) # read clipped DEM into image array
    HS_img = skimage.io.imread(dlg.hillshade_path) # read clipped hillshade into image array

    img_h, img_w = getImgDimensions(dlg)
    cam_x, cam_y, pixelSizeX, pixelSizeY = camXY(dlg, cam_params["lat"], cam_params["lon"]) # find pixel coordinates of camera position and raster resolution

    v_fov = cam_params["h_fov"]*img_h/img_w # determine vertical field of view from horizontal field of view and picture size
    
    if cam_params["elev"] is None:
        cam_params["elev"] = DEM_img[cam_y, cam_x] # read elevation from DEM if not provided by user
        dlg.Elev_lineEdit.setText(str(cam_params["elev"]))

    img = np.zeros((img_h,img_w),dtype=np.uint8) # create blank image

    a = cam_params["azi"] - cam_params["h_fov"]/2 # starting ray angle is azimuth minus half of horizontal FOV
    
    pic_angles = np.linspace(-v_fov/2, v_fov/2, img_h) # create list of image angles
    pic_angles = np.tan(np.radians(pic_angles)) # find ratio (opp/adj)

    # create ray start position (remove first 100 m)
    ray_start_y = int(cam_y - (100*math.cos(np.radians(a))/pixelSizeX))
    ray_start_x = int(cam_x + (100*math.sin(np.radians(a))/pixelSizeY))

    for img_x in range(0,img_w):
            
        # create ray end position
        ray_end_y = int(cam_y - (25000*math.cos(np.radians(a))/pixelSizeX))
        ray_end_x = int(cam_x + (25000*math.sin(np.radians(a))/pixelSizeY))
 
        rr, cc = skimage.draw.line(ray_start_y, ray_start_x, ray_end_y, ray_end_x) # create a ray

        val = DEM_img[rr, cc] - (cam_params["elev"]+cam_params["hgt"]) # get array of elevations
        ll = np.sqrt((abs(rr-cam_y)*pixelSizeY)**2 + (abs(cc-cam_x)*pixelSizeX)**2) # create a list of angles of view for the DEM
        dem_angles = np.divide(val,ll) # find ratio (opp/adj)

        dem_angles_inc = np.maximum.accumulate(dem_angles) # checks for only increasing DEM angles
        unique_angles, unique_angles_indx = np.unique(dem_angles_inc, return_index=True) # keep only unique increasing angles and their index
            
        # find rows and columns of increasing unique angles
        rr_new = rr[unique_angles_indx]
        cc_new = cc[unique_angles_indx]

        greyscale_vals = HS_img[rr_new, cc_new] # extract greyscale values at specified rows and columns

        nonsky_pixels = pic_angles[pic_angles < max(dem_angles_inc)] # truncate picture array to remove sky pixels

        yinterp = np.interp(nonsky_pixels, unique_angles, greyscale_vals) # interpolate any missing greyscale values

        img_column = np.concatenate((yinterp, np.ones(img_h - len(yinterp)) * 255)) # create column for image and fill sky pixels white

        img[:, img_x] = np.flip(img_column, axis=0) # flip the column
            
        a = a+(cam_params["h_fov"]/img_w) # update ray angle

    return img

def enableTools(dlg):
    """Enables canvas tools once canvas is populated with mask and image"""

    dlg.View_toolButton_2.setEnabled(True)
    dlg.Fit_toolButton_2.setEnabled(True)
    dlg.Pan_toolButton_2.setEnabled(True)
    dlg.FullScrn_toolButton_2.setEnabled(True)

def displaySaveVP(dlg, save):
    """Displays and saves virtual photo"""

    if save:
        # open save dialog and save vp
        dialog = QFileDialog()
        dialog.setOption(dialog.DontUseNativeDialog)
        vp_path = dialog.getSaveFileName(filter = "TIFF format (*.tiff *.TIFF)")[0]
    else:
        # save vp to temp path
        vp_path = os.path.join(tempfile.mkdtemp(), 'tempVP.tiff')

    vp = createVP(dlg) # creates virtual photo
    cv2.imwrite(vp_path, vp)

    addImg(dlg.InputRefImg_lineEdit.text(),"Original Image",dlg.Img_mapCanvas_2) # show input image in side-by-side
    addImg(vp_path,"Virtual Photo",dlg.VP_mapCanvas) # show output mask in side-by-side
    addImg(vp_path,"Virtual Photo",dlg.Full_mapCanvas_2) # show output mask in fullview
    addImg(dlg.InputRefImg_lineEdit.text(),"Original Image",dlg.Full_mapCanvas_2) # show input image in full view

    dlg.VP_mapCanvas.refresh()
    dlg.Img_mapCanvas_2.refresh()
    dlg.Full_mapCanvas_2.refresh()
    
    enableTools(dlg)
