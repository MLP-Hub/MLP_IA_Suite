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
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QFileDialog

from qgis.core import QgsRasterLayer

# Initialize Qt resources from file resources.py
from .resources import *

# Import the code for the dialog
from .mlp_ia_suite_dialog import MLP_IA_SuiteDialog
from .pylc_setup import modelMenu, runPylc
from .vp_creation import createHillshade, getHS, displaySaveVP, loadCamParam, saveCamParam
from .interface_tools import setScaleBoxVal, setScaleSlideVal, getFileFolder, getFolder, getFile, updateExtents, panCanvas, zoomToExt, changeView, changeTools, swipeTool, transparency


import sys
import os.path
import subprocess



# Install requirements for pylc.py
#MAKE SURE TO CHECK IF ALREADY INSTALLED AND WARN USERS ABOUT DEPENDENCIES
#bat_path = os.path.join(this_dir, 'pylc_master\pylc_env.bat')
#bat_path="C:\Thesis\QGIS_Plugin\pylc_env.bat"
#subprocess.call([r"C:\Thesis\QGIS_Plugin\pylc_env.bat"]) #THIS SEEMS TO BE WORKING NOW


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
        
        # PYLC TAB 
        self.dlg.Full_mapCanvas.hide() # hide the full map canvas (default side-by-side view)
        mod_dict = modelMenu(self.dlg) # set up model menu

        # Set up image scale slider
        self.dlg.Scale_slider.valueChanged['int'].connect(lambda: setScaleBoxVal(self.dlg, self.dlg.Scale_slider.value()))
        self.dlg.Scale_lineEdit.textChanged.connect(lambda: setScaleSlideVal(self.dlg, self.dlg.Scale_lineEdit.text()))

        # Get file/folder inputs
        self.dlg.InputImg_button.clicked.connect(lambda: getFileFolder(self.dlg.InputImg_lineEdit))
        self.dlg.OutputImg_button.clicked.connect(lambda: getFolder(self.dlg.OutputImg_lineEdit))

        # Run PyLC and display outputs
        self.dlg.Run_pushButton.clicked.connect(lambda: runPylc(self.dlg, mod_dict))
        
        # Link the extent of the image to the extent of the mask and v.v.
        self.dlg.Mask_mapCanvas.extentsChanged.connect(lambda: updateExtents(self.dlg.Img_mapCanvas, self.dlg.Mask_mapCanvas))
        self.dlg.Img_mapCanvas.extentsChanged.connect(lambda: updateExtents(self.dlg.Mask_mapCanvas, self.dlg.Img_mapCanvas))

        # Connect tools to appropriate functions (PyLC tab)
        canvas_list = [self.dlg.Img_mapCanvas, self.dlg.Mask_mapCanvas, self.dlg.Full_mapCanvas]
        self.dlg.View_toolButton.clicked.connect(lambda: changeView(self.dlg.Full_mapCanvas, [self.dlg.Swipe_toolButton, self.dlg.Transparency_slider]))
        self.dlg.Pan_toolButton.clicked.connect(lambda: panCanvas(self.dlg, canvas_list, self.dlg.Pan_toolButton))
        self.dlg.Pan_toolButton.clicked.connect(lambda: changeTools([self.dlg.Swipe_toolButton], self.dlg.Pan_toolButton))
        self.dlg.Fit_toolButton.clicked.connect(lambda: zoomToExt(canvas_list)) # Zoom to mask extent
        self.dlg.Swipe_toolButton.clicked.connect(lambda: swipeTool(self.dlg, self.dlg.Full_mapCanvas, self.dlg.Swipe_toolButton))
        self.dlg.Swipe_toolButton.clicked.connect(lambda: changeTools([self.dlg.Pan_toolButton], self.dlg.Swipe_toolButton))
        self.dlg.Transparency_slider.valueChanged['int'].connect(lambda: transparency(self.dlg.Transparency_slider.value(), self.dlg.Full_mapCanvas))
        #self.dlg.FullScrn_toolButton.clicked.connect(self.fullScrn)

        # VP TAB
        self.dlg.Full_mapCanvas_2.hide()
        
        # Get file/folder inputs
        self.dlg.InputDEM_button.clicked.connect(lambda: getFile(self.dlg.InputDEM_lineEdit, "TIF format (*.tif *.TIF);;TIFF format (*.tiff *.TIFF)"))
        #self.dlg.InputDEM_button.clicked.connect(lambda: createDEM(self.dlg))
        self.dlg.InputRefImg_button.clicked.connect(lambda: getFile(self.dlg.InputRefImg_lineEdit, "JPEG format (*.jpeg);;JPG format (*.jpg);;PNG format (*.png);;TIF format (*.tif *.TIF);;TIFF format (*.tiff *.TIFF)"))
        self.dlg.Load_hillshd_button.clicked.connect(lambda: getHS(self.dlg))
        
        # Generate hillshade and VP
        #self.dlg.GenerateHS_button.clicked.connect(lambda: createHillshade(self.dlg)) # generate hillshade from DEM
        self.dlg.GenerateVP_button.clicked.connect(lambda: displaySaveVP(self.dlg, False))
        self.dlg.SaveVP_button.clicked.connect(lambda: displaySaveVP(self.dlg, True))

        self.dlg.LoadCamParam_button.clicked.connect(lambda: loadCamParam(self.dlg))
        self.dlg.SaveCamParam_button.clicked.connect(lambda: saveCamParam(self.dlg))


        # Link the extent of the image to the extent of the mask and v.v.
        self.dlg.VP_mapCanvas.extentsChanged.connect(lambda: updateExtents(self.dlg.Img_mapCanvas_2, self.dlg.VP_mapCanvas))
        self.dlg.Img_mapCanvas_2.extentsChanged.connect(lambda: updateExtents(self.dlg.VP_mapCanvas, self.dlg.Img_mapCanvas_2))

        # Connect tools to appropriate functions (VP tab)
        canvas_list = [self.dlg.Img_mapCanvas_2, self.dlg.VP_mapCanvas, self.dlg.Full_mapCanvas_2]
        self.dlg.View_toolButton_2.clicked.connect(lambda: changeView(self.dlg.Full_mapCanvas_2, [self.dlg.Swipe_toolButton_2, self.dlg.Transparency_slider_2]))
        self.dlg.Pan_toolButton_2.clicked.connect(lambda: panCanvas(self.dlg, canvas_list, self.dlg.Pan_toolButton_2))
        self.dlg.Pan_toolButton_2.clicked.connect(lambda: changeTools([self.dlg.Swipe_toolButton_2], self.dlg.Pan_toolButton_2))
        self.dlg.Fit_toolButton_2.clicked.connect(lambda: zoomToExt(canvas_list)) # Zoom to mask extent
        self.dlg.Swipe_toolButton_2.clicked.connect(lambda: swipeTool(self.dlg, self.dlg.Full_mapCanvas_2, self.dlg.Swipe_toolButton_2))
        self.dlg.Swipe_toolButton_2.clicked.connect(lambda: changeTools([self.dlg.Pan_toolButton_2], self.dlg.Swipe_toolButton_2))
        self.dlg.Transparency_slider_2.valueChanged['int'].connect(lambda: transparency(self.dlg.Transparency_slider_2.value(), self.dlg.Full_mapCanvas_2))
        


        # SHOW THE DIALOG
        self.dlg.show()

        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result:
            # Do something useful here - delete the line containing pass and
            # substitute with your code.
            pass

