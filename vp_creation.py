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

from .interface_tools import addImg, errorMessage, sideBySide
from .refresh import messageBox

from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox, QProgressDialog
from qgis.PyQt.QtCore import Qt
from qgis.core import QgsRasterLayer, QgsProcessing
from qgis.gui import QgsProjectionSelectionDialog 
from qgis import processing

import os.path
import tempfile
import skimage
import cv2
import scipy
import numpy as np
import math

def resetCamPath(dlg):
    """Resets path to camera parameters if they are changed"""

    dlg.cam_path = None

def resetCamPos(dlg):
    """Resets the initial lat/lon of the camera when DEM reloaded"""

    dlg.lat_init = None
    dlg.lon_init = None

def readCamParams(dlg):
    """Reads user input camera parameters as dictionary"""

    # Read camera parameters as text
    cam_params = {}
    cam_params["lat"] = dlg.Easting_lineEdit.text()
    cam_params["lon"] = dlg.Northing_lineEdit.text()
    cam_params["azi"] = dlg.Azi_lineEdit.text()
    cam_params["h_fov"] = dlg.horFOV_lineEdit.text()
    cam_params["hgt"] = dlg.CamHgt_lineEdit.text()

    # Convert camera parameters to float (checks if left blank)
    msgFields = ["Easting", "Northing", "Azimuth", "Field of view", "Camera height"]
    i = 0

    for param, val in cam_params.items():
        try:
            cam_params[param] = float(val)
        except ValueError:
            errorMessage("Error: {} cannot be blank".format(msgFields[i]))
            return
        i+=1
    
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
    if dialog.exec_():
        cam_filepath = dialog.selectedFiles()[0]

    cam_file = open(cam_filepath, "r")

    cam_params = {}

    for line in cam_file:
       (key, val) = line.split(":")
       cam_params[key] = val

    cam_file.close()

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

    dlg.cam_path = cam_filepath # change path to camera parameters

def saveCamParam(dlg):
    """This function saves current camera parameters to a text file"""

    dialog = QFileDialog()
    dialog.setOption(dialog.DontUseNativeDialog)
    dialog.setAcceptMode(QFileDialog.AcceptSave)
    dialog.setNameFilter("Text files (*.txt)")
    dialog.setDefaultSuffix("txt")

    if dialog.exec_():
        cam_filepath = dialog.selectedFiles()[0]
    
    cam_file = open(cam_filepath, "w")

    cam_params = readCamParams(dlg)
    for key, value in cam_params.items(): 
        cam_file.write('%s:%s\n' % (key, value))

    cam_file.close()
    dlg.refresh_dict["VP"]["Cam"]=cam_filepath

def moveCam(dlg, dir):
    """Moves camera in space relative to azimuth"""

    params = {'cam_x':dlg.Easting_lineEdit.text(), 'cam_y':dlg.Northing_lineEdit.text(), 'azi':dlg.Azi_lineEdit.text(), 'step_size_m':dlg.StepSizeM_lineEdit.text()}
    msgFields = ['Easting', 'Northing', 'Azimuth', 'Step Size (m)']
    i = 0

    for param, val in params.items():
        try:
            params[param] = float(val)
        except ValueError:
            errorMessage("Error: {} cannot be blank".format(msgFields[i]))
            return
        i+=1
    
    cam_x = params['cam_x']
    cam_y = params['cam_y']
    azi = params['azi']
    step_size_m = params['step_size_m']

    if dir == "Forward":
        cam_y = cam_y + np.cos(np.radians(azi))*step_size_m
        cam_x = cam_x + np.sin(np.radians(azi))*step_size_m
    elif dir == "Backward":
        cam_y = cam_y - np.cos(np.radians(azi))*step_size_m
        cam_x = cam_x - np.sin(np.radians(azi))*step_size_m
    elif dir == "Left":
        cam_y = cam_y + np.sin(np.radians(azi+90))*step_size_m
        cam_x = cam_x + np.cos(np.radians(azi+90))*step_size_m
    else:
        cam_y = cam_y + np.sin(np.radians(azi-90))*step_size_m
        cam_x = cam_x + np.cos(np.radians(azi-90))*step_size_m

    dlg.Easting_lineEdit.setText(str(round(cam_x,3)))
    dlg.Northing_lineEdit.setText(str(round(cam_y,3)))
        
def camHeight(dlg, dir):
    """Changes camera height"""

    cam_hgt = float(dlg.CamHgt_lineEdit.text())
    try:
        step_size_m = float(dlg.StepSizeM_lineEdit.text())
    except ValueError:
        errorMessage("Step size (m) cannot be blank")
        return

    if dir == "Up":
        cam_hgt = cam_hgt + step_size_m
    else:
        cam_hgt = cam_hgt - step_size_m

    dlg.CamHgt_lineEdit.setText(str(cam_hgt))

def rotateCam(dlg, dir):
    """Changes camera rotation"""

    azi = float(dlg.Azi_lineEdit.text())
    try:
        step_size_deg = float(dlg.StepSizeDeg_lineEdit.text())
    except ValueError:
        errorMessage("Step size (°) cannot be blank")
        return

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
    
def createHillshade(clipped_DEM):
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

    try:
        hillshade=processing.run("gdal:hillshade", parameters)

        hillshade_path=hillshade['OUTPUT']
        hillshade_layer = QgsRasterLayer(hillshade_path, "Hillshade")
    except:
        errorMessage("Hillshade creation failed")
        return
    return hillshade_path, hillshade_layer   

def clipDEM(DEM_layer, cam_x, cam_y):
    """Clips DEM based on camera parameters"""

    # Get clipping extents (clipped either to 25 km or the extent of the DEM if it is smaller)
    ex = DEM_layer.extent() 
    min_x = max([ex.xMinimum(),(cam_x - 25200)])
    max_x = min([ex.xMaximum(),(cam_x + 25200)])
    min_y = max([ex.yMinimum(),(cam_y - 25200)])
    max_y = min([ex.yMaximum(),(cam_y + 25200)])

    out = [min_x, max_x, min_y, max_y]
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

    try:
        clipped_DEM = processing.run("gdal:cliprasterbyextent", parameters)
    
        clipDEM_path=clipped_DEM['OUTPUT']
        clipDEM_layer = QgsRasterLayer(clipDEM_path, "Clipped DEM") 
    except:
        errorMessage("DEM Clip failed")
        return

    return clipDEM_path, clipDEM_layer

def reprojectDEM(DEM_layer):
    """Ensures DEM has appropriate CRS"""
    
    # Ensure DEM is in projected CRS with unit meters
    source_crs = DEM_layer.crs() # get current CRS
    dir_path = os.path.dirname(__file__)
    dir_path = os.path.normpath(dir_path)
    crs_catalog_path = os.path.join(dir_path, "crs_list.txt") # path to file containing list of appropriate CRS
  
    crs_dlg = QgsProjectionSelectionDialog() # open CRS selection dialog
    crs_file = open(crs_catalog_path, "r")
    crs_data = crs_file.read()
    crs_list = crs_data.split("\n") # list of acceptable CRS
    crs_file.close()
    crs_dlg.setOgcWmsCrsFilter(crs_list)
    crs_dlg.exec()
    dest_crs = crs_dlg.crs() # destination CRS selected by use

    # Reproject DEM
    parameters = { 'INPUT':DEM_layer, 
                        'SOURCE_CRS':source_crs,
                        'TARGET_CRS':dest_crs,
                        'RESAMPLING':1,
                        'NODATA':None,
                        'TARGET_RESOLUTION':None,
                        'OPTIONS':'',
                        'DATA_TYPE':6,
                        'TARGET_EXTENT':None,
                        'TARGET_EXTENT_CRS':dest_crs,
                        'MULTITHREADING':False,
                        'EXTRA':'',
                        'OUTPUT':QgsProcessing.TEMPORARY_OUTPUT
            }

    try:    
        proj_DEM = processing.run("gdal:warpreproject", parameters)
        projDEM_path=proj_DEM['OUTPUT']
        projDEM_layer = QgsRasterLayer(projDEM_path, "Projected DEM")

    except:
        errorMessage("DEM reprojection failed")
        return

    return projDEM_path, projDEM_layer
   
def camXY(hillshade_layer, lat, lon):
    """Determines pixel coordinates of camera position"""

    # get hillshade extents and spatial resolution
    ex = hillshade_layer.extent() 
    pixelSizeX = hillshade_layer.rasterUnitsPerPixelX()
    pixelSizeY = hillshade_layer.rasterUnitsPerPixelY()
    
    # get top left coordinate
    ymax = ex.yMaximum()
    xmin = ex.xMinimum()

    # get camera position in pixel coordinates
    cam_x = int((lat - xmin)/pixelSizeX)
    cam_y = int((ymax - lon)/pixelSizeY)

    return cam_x, cam_y, pixelSizeX, pixelSizeY

def createVP(dlg):
    """Creates virtual photograph""" 

    # checks if VP exists but hasn't been saved
    if dlg.refresh_dict["VP"]["VP"] is None and dlg.vp_path is not None:
        ret = messageBox("Virtual photograph")
        if ret == QMessageBox.No:
            return
    
    cam_params = readCamParams(dlg) # read camera parameters
    if cam_params is None:
        return # exit VP creation if the camera parameters are incorrect

    # Read DEM from user input
    DEM_path = os.path.realpath(dlg.InputDEM_lineEdit.text())
    DEM_layer = QgsRasterLayer(DEM_path, "VP_DEM")
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

    # check camera is on DEM
    ex = DEM_layer.extent()
    if ex.yMaximum() < float(cam_params['lon']) or ex.yMinimum() > float(cam_params['lon']) or ex.xMaximum() < float(cam_params['lat']) or ex.xMinimum() > float(cam_params['lat']):
        errorMessage("Camera off DEM")
        return

    # generate new clipped DEM and hillshade on first round or when camera moves more than 200 m
    og_lat, og_lon = cam_params['lat'], cam_params['lon']
    
    if dlg.lat_init is None or (abs(dlg.lat_init - og_lat) > 200) or (abs(dlg.lon_init - og_lon) > 200):
        
        msg = QMessageBox()
        msg.setText("Clipping DEM and processing hillshade. This may take several minutes. QGIS may show as not responding. Do not click anything.")
        msg.setIcon(QMessageBox.Warning)
        msg.exec()   
        try:
            dlg.clip_DEM_path, dlg.clip_DEM_layer = clipDEM(DEM_layer, cam_params['lat'], cam_params['lon'])
        except TypeError:
            return
        try:
            dlg.hillshade_path, dlg.hs_layer = createHillshade(dlg.clip_DEM_layer)
        except TypeError:
            return
        dlg.lat_init = cam_params['lat'] # replace with new camera position
        dlg.lon_init = cam_params['lon'] 
    
    DEM_img = skimage.io.imread(dlg.clip_DEM_path) # read clipped DEM into image array
    HS_img = skimage.io.imread(dlg.hillshade_path) # read clipped hillshade into image array

    img_h, img_w = getImgDimensions(dlg)
    cam_x, cam_y, pixelSizeX, pixelSizeY = camXY(dlg.hs_layer, cam_params["lat"], cam_params["lon"]) # find pixel coordinates of camera position and raster resolution

    v_fov = cam_params["h_fov"]*img_h/img_w # determine vertical field of view from horizontal field of view and picture size
    
    if cam_params["elev"] is None:
        cam_params["elev"] = DEM_img[cam_y, cam_x] # read elevation from DEM if not provided by user

    img = np.zeros((img_h,img_w),dtype=np.uint8) # create blank image

    a = cam_params["azi"] - cam_params["h_fov"]/2 # starting ray angle is azimuth minus half of horizontal FOV
    
    img_v_angles = np.linspace(-v_fov/2, v_fov/2, img_h) # create list of image angles
    img_v_angles = np.tan(np.radians(img_v_angles)) # find ratio (opp/adj)

    progressDlg = QProgressDialog("Creating virtual photograph...","Cancel", 0, img_w)
    progressDlg.setWindowModality(Qt.WindowModal)
    progressDlg.setValue(0)
    progressDlg.forceShow()
    progressDlg.show()  

    for img_x in range(0, img_w):
        # First, find all visible pixles
        # get the horizontal and vertical angles to the camera position
        # and find the corresponding greyscale values
        
        progressDlg.setValue(img_x)

        # create ray start position (remove first 100 m)
        ray_start_y = cam_y - (100*math.cos(np.radians(a))/pixelSizeY)
        ray_start_x = cam_x + (100*math.sin(np.radians(a))/pixelSizeX)
        
        # create ray end position
        ray_end_y = cam_y - (25000*math.cos(np.radians(a))/pixelSizeY)
        ray_end_x = cam_x + (25000*math.sin(np.radians(a))/pixelSizeX)

        xs = np.linspace(ray_start_x, ray_end_x, round(100000/pixelSizeX))
        ys = np.linspace(ray_start_y, ray_end_y, round(100000/pixelSizeY))

        elevs = scipy.ndimage.map_coordinates(DEM_img, np.vstack((ys,xs)), order = 1) - cam_params["elev"]-cam_params["hgt"]
        greys = scipy.ndimage.map_coordinates(HS_img, np.vstack((ys,xs)), order = 1)

        opp = (xs - cam_x)*pixelSizeX
        adj = (cam_y - ys)*pixelSizeY

        dem_dist = np.sqrt(np.add(opp**2, adj**2))
        vert_angles = np.arctan(np.divide(elevs,dem_dist)) # find ratio (opp/adj)

        dem_angles_inc = np.fmax.accumulate(vert_angles) # checks for only increasing DEM angles
        unique_angles, unique_angles_indx = np.unique(dem_angles_inc, return_index=True) # keep only unique increasing angles and their index

        greys_visible = greys[unique_angles_indx]

        nonsky_pixels = img_v_angles[img_v_angles < max(dem_angles_inc)] # truncate picture array to remove sky pixels
        yinterp = np.interp(nonsky_pixels, unique_angles, greys_visible) # interpolate any missing greyscale values
        
        img_column = np.concatenate((yinterp, np.ones(img_h - len(yinterp)) * 255)) # create column for image and fill sky pixels white
        img[:, img_x] = np.flip(img_column, axis=0) # flip the column
            
        a = a+(cam_params["h_fov"]/img_w) # update ray angle

    return img

def enableTools(dlg):
    """Enables canvas tools once canvas is populated with mask and image"""

    dlg.SideBySide_pushButton_2.setEnabled(True)
    dlg.SingleView_pushButton_2.setEnabled(True)
    dlg.Fit_toolButton_2.setEnabled(True)
    dlg.Pan_toolButton_2.setEnabled(True)

def displayVP(dlg):
    """Creates and displays virtual photo"""

    if dlg.SideBySide_pushButton_2.isEnabled():
        canvas_list_2 = [dlg.Img_mapCanvas_2, dlg.VP_mapCanvas, dlg.Full_mapCanvas_2]
        sideBySide(canvas_list_2, [dlg.Swipe_toolButton_2, dlg.Transparency_slider_2],dlg.SideBySide_pushButton_2, dlg.SingleView_pushButton_2)
    
    vp = createVP(dlg) # create virtual photo
    
    if vp is None:
        # break if create VP did not work
        errorMessage('Virtual photograph failed')
        return
    
    # save vp to temp path
    dlg.vp_path = os.path.join(tempfile.mkdtemp(), 'tempVP.tiff')
    if os.path.isfile(dlg.vp_path):
        # check if the temporary file already exists
        os.remove(dlg.vp_path)
    
    cv2.imwrite(dlg.vp_path, vp)

    addImg(dlg.InputRefImg_lineEdit.text(),"Original Image",dlg.Img_mapCanvas_2, True) # show input image in side-by-side
    addImg(dlg.vp_path,"Virtual Photo",dlg.VP_mapCanvas, True) # show output mask in side-by-side
    addImg(dlg.vp_path,"Virtual Photo",dlg.Full_mapCanvas_2, False) # show output mask in fullview
    addImg(dlg.InputRefImg_lineEdit.text(),"Original Image",dlg.Full_mapCanvas_2, False) # show input image in full view
    
    enableTools(dlg)

def saveVP(dlg):
    """Saves virtual photograph to specified location"""

    # first check that camera parameters have been saved
    if dlg.refresh_dict["VP"]["Cam"] is None:
        msg = QMessageBox()
        msg.setText("Save latest camera parameters first.")
        msg.exec()

        saveCamParam(dlg) # opens camera parameter save dialog

    vp = cv2.imread(dlg.vp_path)
    save_vp_path = None

    # open save dialog
    dialog = QFileDialog()
    dialog.setOption(dialog.DontUseNativeDialog)
    dialog.setNameFilter("TIFF format (*.tiff *.TIFF)")
    dialog.setDefaultSuffix("tiff")
    dialog.setAcceptMode(QFileDialog.AcceptSave)

    if dialog.exec_():
        save_vp_path = dialog.selectedFiles()[0]

        cv2.imwrite(save_vp_path, vp)
        dlg.refresh_dict["VP"]["VP"]=save_vp_path
