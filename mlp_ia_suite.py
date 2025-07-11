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
from ast import Lambda
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication
from qgis.PyQt.QtGui import QIcon, QDoubleValidator
from qgis.PyQt.QtWidgets import QAction

# Initialize Qt resources from file resources.py
from .resources import *

# Import the code for the dialog
from .mlp_ia_suite_dialog import MLP_IA_SuiteDialog
from .pylc_setup import runPylc, saveMask
from .vp_creation import displayVP, loadCamParam, saveCamParam, moveCam, camHeight, rotateCam, saveVP, resetCamPos, resetCamPath
from .img_alignment import addCPTool, delCPTool, saveCPs, loadCPs, checkForImgs, alignImgs, saveAlign, selectFromTable, switchLayer, undoAlign
from .vs_creation import createVS, saveVS
from .mosaic import addLayer, removeLayer, moveLayerUp, moveLayerDown, mosaicRasters, saveMosaic
from .interface_tools import setScaleBoxVal, setScaleSlideVal, getFile, updateExtents, panCanvas, zoomToExt, singleView, sideBySide, swipeTool, transparency, addImg
from .refresh import refresh_PyLC, refresh_VP, refresh_align, refresh_VS

import os.path

class MLP_IA_Suite:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'MLP_IA_Suite_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&MLP Image Analysis Suite')

        # Check if plugin was started the first time in current QGIS session
        # Must be set in initGui() to survive plugin reloads
        self.first_start = None

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('MLP_IA_Suite', message)


    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            # Adds plugin icon to Plugins toolbar
            self.iface.addToolBarIcon(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/mlp_ia_suite/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'MLP Image Analysis Suite'),
            callback=self.run,
            parent=self.iface.mainWindow())

        # will be set False in run()
        self.first_start = True


    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&MLP Image Analysis Suite'),
                action)
            self.iface.removeToolBarIcon(action)

    
    def run(self):
        """Run method that performs all the real work"""

        # Create the dialog with elements (after translation) and keep reference
        # Only create GUI ONCE in callback, so that it will only load when the plugin is started
        if self.first_start == True:
            self.first_start = False
            self.dlg = MLP_IA_SuiteDialog()

        # Create a dictionary with all the elements that should be saved before refreshing
        self.dlg.refresh_dict = {"PyLC":{"Mask":None},"VP":{"Cam":None,"VP":None},"Align":{"CPs":None, "Img":None, "Mask":None}, "VS":{"VS":None}}
        
        # PYLC TAB 
        self.dlg.PyLC_path = None # initiate variable to hold path to mask (for temp file)
        self.dlg.pylc_run = False # initiate variable to check whether PyLC ran already

        # Set up image scale slider
        self.dlg.Scale_slider.valueChanged['int'].connect(lambda: setScaleBoxVal(self.dlg, self.dlg.Scale_slider.value()))
        self.dlg.Scale_lineEdit.textChanged.connect(lambda: setScaleSlideVal(self.dlg, self.dlg.Scale_lineEdit.text()))

        # Get file inputs
        self.dlg.InputImg_button.clicked.connect(lambda: getFile(self.dlg.InputImg_lineEdit, "Images (*.jpeg *.jpg *.png *.tif *.TIF *.tiff *.TIFF)"))
        self.dlg.InputModel_button.clicked.connect(lambda: getFile(self.dlg.InputModel_lineEdit, "*.pth"))

        # Run PyLC and display outputs
        self.dlg.Run_pushButton.clicked.connect(lambda: runPylc(self.dlg))
        self.dlg.PyLC_Save_pushButton.clicked.connect(lambda: saveMask(self.dlg))

        # Connect tools to appropriate functions (PyLC tab)
        self.dlg.SideBySide_pushButton.hide() # hide side by side view button to start
        canvas_list_1 = [self.dlg.Img_mapCanvas, self.dlg.Mask_mapCanvas, self.dlg.Full_mapCanvas]
        self.dlg.SideBySide_pushButton.clicked.connect(lambda: sideBySide(canvas_list_1, [self.dlg.Swipe_toolButton, self.dlg.Transparency_slider], self.dlg.SideBySide_pushButton, self.dlg.SingleView_pushButton))
        self.dlg.SingleView_pushButton.clicked.connect(lambda: singleView(canvas_list_1, [self.dlg.Swipe_toolButton, self.dlg.Transparency_slider], self.dlg.SideBySide_pushButton, self.dlg.SingleView_pushButton))
        self.dlg.Pan_toolButton.clicked.connect(lambda: panCanvas(self.dlg, canvas_list_1, self.dlg.Pan_toolButton, self.dlg.Swipe_toolButton))
        self.dlg.Fit_toolButton.clicked.connect(lambda: zoomToExt(canvas_list_1)) # Zoom to mask extent
        self.dlg.Swipe_toolButton.clicked.connect(lambda: swipeTool(self.dlg, self.dlg.Full_mapCanvas, self.dlg.Swipe_toolButton, self.dlg.Pan_toolButton))
        self.dlg.Transparency_slider.valueChanged['int'].connect(lambda: transparency(self.dlg.Transparency_slider.value(), self.dlg.Full_mapCanvas))

        self.dlg.PyLC_refresh.clicked.connect(lambda: refresh_PyLC(self.dlg, canvas_list_1))

        # VP TAB

        self.dlg.vp_path = None # initiate variable to hold path to VP (for temp file)

        # Set up line edits (to take appropriate data type)
        numValidator = QDoubleValidator(bottom = 0, notation=QDoubleValidator.StandardNotation) # only allow positive float
        self.dlg.Easting_lineEdit.setValidator(numValidator)
        self.dlg.Northing_lineEdit.setValidator(numValidator)
        self.dlg.Azi_lineEdit.setValidator(numValidator)
        self.dlg.horFOV_lineEdit.setValidator(numValidator)
        self.dlg.CamHgt_lineEdit.setValidator(numValidator)
        self.dlg.Elev_lineEdit.setValidator(numValidator)
        self.dlg.StepSizeM_lineEdit.setValidator(numValidator)
        self.dlg.StepSizeDeg_lineEdit.setValidator(numValidator)

        # reset path to camera parameters if any of them were changed
        self.dlg.Easting_lineEdit.textChanged.connect(lambda: resetCamPath(self.dlg))
        self.dlg.Northing_lineEdit.textChanged.connect(lambda: resetCamPath(self.dlg))
        self.dlg.Azi_lineEdit.textChanged.connect(lambda: resetCamPath(self.dlg))
        self.dlg.horFOV_lineEdit.textChanged.connect(lambda: resetCamPath(self.dlg))
        self.dlg.CamHgt_lineEdit.textChanged.connect(lambda: resetCamPath(self.dlg))
        self.dlg.Elev_lineEdit.textChanged.connect(lambda: resetCamPath(self.dlg))
        self.dlg.StepSizeM_lineEdit.textChanged.connect(lambda: resetCamPath(self.dlg))
        self.dlg.StepSizeDeg_lineEdit.textChanged.connect(lambda: resetCamPath(self.dlg))
        
        # Get file/folder inputs
        self.dlg.InputDEM_button.clicked.connect(lambda: getFile(self.dlg.InputDEM_lineEdit, "TIF format (*.tif *.TIF);;TIFF format (*.tiff *.TIFF)"))
        self.dlg.InputDEM_button.clicked.connect(lambda: resetCamPos(self.dlg))
        self.dlg.InputRefImg_button.clicked.connect(lambda: getFile(self.dlg.InputRefImg_lineEdit, "Images (*.jpeg *.jpg *.png *.tif *.TIF *.tiff *.TIFF)"))
        
        # Generate hillshade and VP
        self.dlg.lat_init = None # initial lat and long for DEM clipping
        self.dlg.lon_init = None
        self.dlg.GenerateVP_button.clicked.connect(lambda: displayVP(self.dlg))
        self.dlg.SaveVP_button.clicked.connect(lambda: saveVP(self.dlg))

        # Load or save camera parameters
        self.dlg.LoadCamParam_button.clicked.connect(lambda: loadCamParam(self.dlg))
        self.dlg.SaveCamParam_button.clicked.connect(lambda: saveCamParam(self.dlg))

        # Connect the camera position buttons to the appropriate functions
        self.dlg.Forward_button.clicked.connect(lambda: moveCam(self.dlg, "Forward"))
        self.dlg.Backward_button.clicked.connect(lambda: moveCam(self.dlg, "Backward"))
        self.dlg.Left_button.clicked.connect(lambda: moveCam(self.dlg, "Left"))
        self.dlg.Right_button.clicked.connect(lambda: moveCam(self.dlg, "Right"))
        self.dlg.Up_button.clicked.connect(lambda: camHeight(self.dlg, "Up"))
        self.dlg.Down_button.clicked.connect(lambda: camHeight(self.dlg, "Down"))
        self.dlg.Clckwis_button.clicked.connect(lambda: rotateCam(self.dlg, "Clckwise"))
        self.dlg.CountClckwis_button.clicked.connect(lambda: rotateCam(self.dlg, "CntrClckwise"))

        # Connect tools to appropriate functions (VP tab)
        self.dlg.SideBySide_pushButton_2.hide() # hide side by side view button to start
        canvas_list_2 = [self.dlg.Img_mapCanvas_2, self.dlg.VP_mapCanvas, self.dlg.Full_mapCanvas_2]
        self.dlg.SideBySide_pushButton_2.clicked.connect(lambda: sideBySide(canvas_list_2, [self.dlg.Swipe_toolButton_2, self.dlg.Transparency_slider_2],self.dlg.SideBySide_pushButton_2, self.dlg.SingleView_pushButton_2))
        self.dlg.SingleView_pushButton_2.clicked.connect(lambda: singleView(canvas_list_2, [self.dlg.Swipe_toolButton_2, self.dlg.Transparency_slider_2],self.dlg.SideBySide_pushButton_2, self.dlg.SingleView_pushButton_2))
        self.dlg.Pan_toolButton_2.clicked.connect(lambda: panCanvas(self.dlg, canvas_list_2, self.dlg.Pan_toolButton_2, self.dlg.Swipe_toolButton_2))
        self.dlg.Fit_toolButton_2.clicked.connect(lambda: zoomToExt(canvas_list_2)) # Zoom to mask extent
        self.dlg.Swipe_toolButton_2.clicked.connect(lambda: swipeTool(self.dlg, self.dlg.Full_mapCanvas_2, self.dlg.Swipe_toolButton_2, self.dlg.Pan_toolButton_2))
        self.dlg.Transparency_slider_2.valueChanged['int'].connect(lambda: transparency(self.dlg.Transparency_slider_2.value(), self.dlg.Full_mapCanvas_2))

        self.dlg.VP_refresh.clicked.connect(lambda: refresh_VP(self.dlg, canvas_list_2))

        # Image Alignment Tab

        self.dlg.aligned_img_path = None # initiate variable to hold path to aligned image (for temp file)
        self.dlg.aligned_mask_path = None # initiate variable to hold path to aligned mask
        self.dlg.CPtool = None # initiate variable for control point tool
        self.dlg.aligned_probs_path = None
        
        # Get file/folder inputs and display images
        self.dlg.SourceImg_button.clicked.connect(lambda: getFile(self.dlg.SourceImg_lineEdit, "Images (*.jpeg *.jpg *.png *.tif *.TIF *.tiff *.TIFF)"))
        self.dlg.DestImg_button.clicked.connect(lambda: getFile(self.dlg.DestImg_lineEdit, "Images (*.jpeg *.jpg *.png *.tif *.TIF *.tiff *.TIFF)"))
        self.dlg.Mask_button.clicked.connect(lambda: getFile(self.dlg.Mask_lineEdit, "Images (*.jpeg *.jpg *.png *.tif *.TIF *.tiff *.TIFF)"))
        self.dlg.SourceImg_button.clicked.connect(lambda: addImg(self.dlg.SourceImg_lineEdit.text(), "Source Image", self.dlg.SourceImg_canvas, True))
        self.dlg.DestImg_button.clicked.connect(lambda: addImg(self.dlg.DestImg_lineEdit.text(), "Destination Image", self.dlg.DestImg_canvas, True))
        
        # Check for both images --> enable control point tools if two images loaded
        canvas_list_3 = [self.dlg.SourceImg_canvas, self.dlg.DestImg_canvas]
        name_list = ["Source Image", "Destination Image"]
        button_list = [self.dlg.addCP_button, self.dlg.delCP_button, self.dlg.loadCP_button, self.dlg.saveCP_button]
        self.dlg.SourceImg_button.clicked.connect(lambda: checkForImgs(canvas_list_3, name_list, button_list))
        self.dlg.DestImg_button.clicked.connect(lambda: checkForImgs(canvas_list_3, name_list, button_list))
        
        self.dlg.addCP_button.clicked.connect(lambda: addCPTool(self.dlg, self.dlg.SourceImg_canvas, "Source CP Layer", "Source Image"))
        self.dlg.delCP_button.clicked.connect(lambda: delCPTool(self.dlg, canvas_list_3))
        self.dlg.saveCP_button.clicked.connect(lambda: saveCPs(self.dlg))
        self.dlg.loadCP_button.clicked.connect(lambda: loadCPs(["Source CP Layer","Dest CP Layer"], canvas_list_3, name_list, self.dlg.CP_table, self.dlg))
        self.dlg.Align_button.clicked.connect(lambda: alignImgs(self.dlg, self.dlg.SourceImg_lineEdit.text(), self.dlg.CP_table))
        self.dlg.UndoAlign_button.clicked.connect(lambda: undoAlign(self.dlg))
        self.dlg.SaveAlign_button.clicked.connect(lambda: saveAlign(self.dlg))

        self.dlg.CP_table.itemSelectionChanged.connect(lambda: selectFromTable(self.dlg.CP_table, canvas_list_3)) # Select point on map when row selected in table

        # Connect tools to appropriate functions (Alignment tab)
        self.dlg.SideBySide_pushButton_3.hide() # hide side by side view button to start
        canvas_list_4 = [self.dlg.DestImg_canvas,self.dlg.SourceImg_canvas, self.dlg.Full_mapCanvas_3]
        self.dlg.SideBySide_pushButton_3.clicked.connect(lambda: sideBySide(canvas_list_4, [self.dlg.Swipe_toolButton_3, self.dlg.Transparency_slider_3],self.dlg.SideBySide_pushButton_3, self.dlg.SingleView_pushButton_3))
        self.dlg.SingleView_pushButton_3.clicked.connect(lambda: singleView(canvas_list_4, [self.dlg.Swipe_toolButton_3, self.dlg.Transparency_slider_3],self.dlg.SideBySide_pushButton_3, self.dlg.SingleView_pushButton_3))
        self.dlg.Pan_toolButton_3.clicked.connect(lambda: panCanvas(self.dlg, canvas_list_4, self.dlg.Pan_toolButton_3, self.dlg.Swipe_toolButton_3))
        self.dlg.Fit_toolButton_3.clicked.connect(lambda: zoomToExt(canvas_list_4)) # Zoom to mask extent
        self.dlg.Swipe_toolButton_3.clicked.connect(lambda: swipeTool(self.dlg, self.dlg.Full_mapCanvas_3, self.dlg.Swipe_toolButton_3, self.dlg.Pan_toolButton_3))
        self.dlg.Transparency_slider_3.valueChanged['int'].connect(lambda: transparency(self.dlg.Transparency_slider_3.value(), self.dlg.Full_mapCanvas_3))
        self.dlg.Layer_comboBox.activated.connect(lambda: switchLayer(self.dlg.Layer_comboBox, self.dlg.Full_mapCanvas_3, self.dlg.SourceImg_canvas, self.dlg))

        self.dlg.Align_refresh.clicked.connect(lambda: refresh_align(self.dlg, canvas_list_4))

        # VS TAB

        self.dlg.vs_path = None # initiate variable to hold path to VP (for temp file)
        self.dlg.probs_lyr_path = None # initiate variable to hold path to probabilities layer
        
        # Get file/folder inputs
        self.dlg.DEM_button.clicked.connect(lambda: getFile(self.dlg.DEM_lineEdit, "TIF format (*.tif *.TIF *.tiff *.TIFF)"))
        self.dlg.AlignMask_button.clicked.connect(lambda: getFile(self.dlg.AlignMask_lineEdit, "Images (*.jpeg *.jpg *.png *.tif *.TIF *.tiff *.TIFF)"))
        self.dlg.CamParam_button.clicked.connect(lambda: getFile(self.dlg.CamParam_lineEdit, "Text files (*.txt)"))

        # Generate VS
        self.dlg.VS_Run_pushButton.clicked.connect(lambda: createVS(self.dlg))
        self.dlg.VS_Save_pushButton.clicked.connect(lambda: saveVS(self.dlg))

        # Connect tools to appropriate functions (VS tab)
        canvas_list_5 = [self.dlg.VS_mapCanvas]
        self.dlg.Pan_toolButton_4.clicked.connect(lambda: panCanvas(self.dlg, canvas_list_5, self.dlg.Pan_toolButton_4, self.dlg.Swipe_toolButton_4))
        self.dlg.Fit_toolButton_4.clicked.connect(lambda: zoomToExt([None, self.dlg.VS_mapCanvas])) # Zoom to viewshed extent
        self.dlg.Swipe_toolButton_4.clicked.connect(lambda: swipeTool(self.dlg, self.dlg.VS_mapCanvas, self.dlg.Swipe_toolButton_4, self.dlg.Pan_toolButton_4))
        self.dlg.Transparency_slider_4.valueChanged['int'].connect(lambda: transparency(self.dlg.Transparency_slider_4.value(), self.dlg.VS_mapCanvas))

        self.dlg.VS_refresh.clicked.connect(lambda: refresh_VS(self.dlg, canvas_list_5))

        # Mosaic Tab

        self.dlg.mosaic_path = None

        self.dlg.addLayer_button.clicked.connect(lambda: addLayer("TIF format (*.tif *.TIF *.tiff *.TIFF)", self.dlg.layers_listWidget))
        self.dlg.removerLayer_button.clicked.connect(lambda: removeLayer(self.dlg.layers_listWidget))
        self.dlg.moveLayerUp_button.clicked.connect(lambda: moveLayerUp(self.dlg.layers_listWidget))
        self.dlg.moveLayerDown_button.clicked.connect(lambda: moveLayerDown(self.dlg.layers_listWidget))
        self.dlg.mosaicRun_pushButton.clicked.connect(lambda: mosaicRasters(self.dlg.layers_listWidget, self.dlg.ranking_checkBox, self.dlg))
        self.dlg.mosaicSave_pushButton.clicked.connect(lambda: saveMosaic(self.dlg))
        

        # SHOW THE DIALOG
        self.dlg.show()

        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result:
            # Do something useful here - delete the line containing pass and
            # substitute with your code.
            pass

