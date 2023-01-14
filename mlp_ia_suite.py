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
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication, Qt
from qgis.PyQt.QtGui import QIcon, QPixmap
from qgis.PyQt.QtWidgets import QAction, QFileDialog
from qgis.core import QgsRasterLayer, QgsProject, QgsProcessing 
from qgis.gui import QgsMapToolPan

# Initialize Qt resources from file resources.py
from .resources import *
# Import the code for the dialog
from .mlp_ia_suite_dialog import MLP_IA_SuiteDialog
from .swipe_tool import mapswipetool
from .pylc_interface import showImag

import sys
import os.path
import subprocess

# Import the code for pylc
this_dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(this_dir, 'pylc_master')
sys.path.append(path)
import pylc



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

    def setScaleBoxVal(self, val):
        """Changes scale box value based on slider"""

        val = float(val / 10)
        val = str(val)
        self.dlg.Scale_lineEdit.setText(val)

    def setScaleSlideVal(self, val):
        """Changes scale slider value based on text box"""

        val = float(val)
        val = int(val*10)
        self.dlg.Scale_slider.setValue(val)

            
    def getImgFile(self):
        """Allows user to select image file or folder"""

        def _selected(name):
            """Changes file mode depending on whether current selection is a file or a folder"""

            if os.path.isdir(name):
                dialog.setFileMode(QFileDialog.Directory)
            else:
                dialog.setFileMode(QFileDialog.ExistingFile)

  
        dialog = QFileDialog()
        dialog.setFileMode(dialog.ExistingFiles)
        dialog.setOption(dialog.DontUseNativeDialog)
        dialog.currentChanged.connect(_selected) #if selection changes from file to folder
        
        dialog.exec_()

        filepath = dialog.selectedFiles()
        self.dlg.InputImg_lineEdit.setText(filepath[0])
    
        
    def selectOutDir(self):
        """Select output directory"""

        dirpath=QFileDialog.getExistingDirectory(self.dlg,"Select output directory")
        self.dlg.OutputImg_lineEdit.setText(dirpath)

    def getMskFile(self):
        """Allows user to select mask file or folder"""

        def _selected(name):
            """Changes file mode depending on whether current selection is a file or a folder"""

            if os.path.isdir(name):
                dialog.setFileMode(QFileDialog.Directory)
            else:
                dialog.setFileMode(QFileDialog.ExistingFile)

  
        dialog = QFileDialog()
        dialog.setFileMode(dialog.ExistingFiles)
        dialog.setOption(dialog.DontUseNativeDialog)
        dialog.currentChanged.connect(_selected) # if selection changes from file to folder
        
        dialog.exec_()

        filepath = dialog.selectedFiles()
        self.dlg.InputMsk_lineEdit.setText(filepath[0])

    def showMask(self):
        """Displays classified mask in PyLC tool tab"""

        # Load mask as layer into mask map canvas
        mask_pth = self.outputFile
        self.mask_lyr= QgsRasterLayer(mask_pth, "Mask")
        QgsProject.instance().addMapLayer(self.mask_lyr, False) # add layer to the registry (but don't load into main map)
        self.dlg.Mask_mapCanvas.enableAntiAliasing(True)
        self.dlg.Mask_mapCanvas.setExtent(self.mask_lyr.extent()) # set extent to the extent of the mask layer
        self.dlg.Mask_mapCanvas.setLayers([self.mask_lyr])
        self.dlg.Mask_mapCanvas.show()

    def showImg(self):
        """Displays image in PyLC tool tab"""

        img_pth = self.dlg.InputImg_lineEdit.text()
        self.img_lyr= QgsRasterLayer(img_pth, "Image")
        # Load mask as layer into mask map canvas
        QgsProject.instance().addMapLayer(self.img_lyr, False) # add layer to the registry (but don't load into main map)
        self.dlg.Img_mapCanvas.enableAntiAliasing(True)
        self.dlg.Img_mapCanvas.setExtent(self.img_lyr.extent()) # set extent to the extent of the image layer
        self.dlg.Img_mapCanvas.setLayers([self.img_lyr])
        self.dlg.Img_mapCanvas.show()

    def enableTools(self):
        """Enables canvas tools once canvas is populated with mask and image"""

        self.dlg.View_toolButton.setEnabled(True)
        self.dlg.Fit_toolButton.setEnabled(True)
        self.dlg.Pan_toolButton.setEnabled(True)
        self.dlg.FullScrn_toolButton.setEnabled(True)
        
        
    def updateImg(self):
        """Updates the extent of the image to match the extent of the mask"""

        if self.dlg.Mask_mapCanvas.extent() != self.dlg.Img_mapCanvas.extent():
            self.dlg.Img_mapCanvas.setExtent(self.dlg.Mask_mapCanvas.extent())
            self.dlg.Img_mapCanvas.refresh()

    def updateMsk(self):
        """Updates the extent of the image to match the extent of the mask"""

        if self.dlg.Mask_mapCanvas.extent() != self.dlg.Img_mapCanvas.extent():
            self.dlg.Mask_mapCanvas.setExtent(self.dlg.Img_mapCanvas.extent())
            self.dlg.Mask_mapCanvas.refresh()

    def panCanvas(self):
        """Enables/disables pan tool"""

        self.dlg.Swipe_toolButton.setChecked(False) # tools are exclusive, so turn off swipe button

        # create pan tools
        self.toolPanSwipe = QgsMapToolPan(self.dlg.Swipe_mapCanvas)
        self.toolPanImg = QgsMapToolPan(self.dlg.Img_mapCanvas)
        self.toolPanMask = QgsMapToolPan(self.dlg.Mask_mapCanvas)

        # enable or disable pan tools depending on whether the tool button is checked
        if self.dlg.Pan_toolButton.isChecked():
            if self.dlg.Swipe_mapCanvas.isVisible():    
                self.dlg.Swipe_mapCanvas.setMapTool(self.toolPanSwipe)
            else:
                self.dlg.Mask_mapCanvas.setMapTool(self.toolPanMask)
                self.dlg.Img_mapCanvas.setMapTool(self.toolPanImg)
        else:
            if self.dlg.Swipe_mapCanvas.isVisible():
                self.dlg.Swipe_mapCanvas.unsetMapTool(self.toolPanSwipe)
            else:
                self.dlg.Mask_mapCanvas.unsetMapTool(self.toolPanMask)
                self.dlg.Img_mapCanvas.unsetMapTool(self.toolPanImg)

    def zoomToExt(self):
        """Zooms to extent of mask and image layer"""

        if self.dlg.Swipe_mapCanvas.isVisible():
            self.dlg.Swipe_mapCanvas.setExtent(self.img_lyr.extent())
            self.dlg.Swipe_mapCanvas.refresh()
        else:
            self.dlg.Mask_mapCanvas.setExtent(self.mask_lyr.extent())
            self.dlg.Img_mapCanvas.setExtent(self.img_lyr.extent())
            self.dlg.Img_mapCanvas.refresh()
            self.dlg.Mask_mapCanvas.refresh()
        
    
    def changeView(self):
        """Changes display from side-by-side to one window or v-v"""
        if not self.dlg.Swipe_mapCanvas.isVisible(): 
            
            # hide the side-by-side view canvases
            self.dlg.Mask_mapCanvas.hide()
            self.dlg.Img_mapCanvas.hide()

            # unset pan tool (if set)
            if self.dlg.Pan_toolButton.isChecked():
                self.dlg.Mask_mapCanvas.unsetMapTool(self.toolPanMask)
                self.dlg.Img_mapCanvas.unsetMapTool(self.toolPanImg)

            # show the swipe view canvas
            self.dlg.Swipe_mapCanvas.setExtent(self.img_lyr.extent())
            self.dlg.Swipe_mapCanvas.setLayers([self.mask_lyr, self.img_lyr])
            self.dlg.Swipe_mapCanvas.show()
            self.dlg.Swipe_toolButton.setEnabled(True) # enable the swipe tool
        
        else:
            self.dlg.Swipe_mapCanvas.hide() # hide the swipe view canvas
            # show the side-by-side view canvases
            self.dlg.Mask_mapCanvas.show()
            self.dlg.Mask_mapCanvas.setExtent(self.mask_lyr.extent())
            self.dlg.Img_mapCanvas.show()
            self.dlg.Img_mapCanvas.setExtent(self.img_lyr.extent())

            # unset tools
            self.dlg.Swipe_toolButton.setChecked(False) # uncheck the swipe button
            self.dlg.Swipe_toolButton.setEnabled(False) # disable the swipe tool
            if self.dlg.Pan_toolButton.isChecked():
                self.dlg.Swipe_mapCanvas.unsetMapTool(self.toolPanSwipe)
            
        self.dlg.Pan_toolButton.setChecked(False) # uncheck pan button

    
    def swipeTool(self):
        """Enables/disables swipe tool"""

        self.dlg.Pan_toolButton.setChecked(False) # uncheck pan button
        if self.dlg.Swipe_toolButton.isChecked():
            swipeTool = mapswipetool.MapSwipeTool(self.dlg.Swipe_mapCanvas)
            self.dlg.Swipe_mapCanvas.setMapTool(swipeTool)
        else:
            self.dlg.Swipe_mapCanvas.unsetMapTool(swipeTool)

    def fullScrn(self):
        """Opens image viewer in full screen"""
        self.dlg.showMaximized()

    def run(self):
        """Run method that performs all the real work"""


        # Create the dialog with elements (after translation) and keep reference
        # Only create GUI ONCE in callback, so that it will only load when the plugin is started
        if self.first_start == True:
            self.first_start = False
            self.dlg = MLP_IA_SuiteDialog()
        
        # GET USER INPUT
        # Set up model combo box
        mod_dict = {"Greyscale 1":"pylc_2-1_deeplab_ch1_schema_a.pth","Greyscale 2":"pylc_2-2_deeplab_ch1_schema_a.pth",
                "Greyscale 3":"pylc_2-3_deeplab_ch1_schema_a.pth","Greyscale 4":"pylc_2-4_deeplab_ch1_schema_a.pth",
                "Greyscale 5":"pylc_2-5_deeplab_ch1_schema_a.pth","Colour 1":"pylc_2-1_deeplab_ch3_schema_a.pth",
                "Colour 2":"pylc_2-2_deeplab_ch3_schema_a.pth","Colour 3":"pylc_2-3_deeplab_ch3_schema_a.pth","Colour 5":
                "pylc_2-5_deeplab_ch3_schema_a.pth"}
        self.dlg.Model_comboBox.clear()
        self.dlg.Model_comboBox.addItems(key for key in mod_dict) # Add model names to combo box

        # Get image scale
        self.dlg.Scale_slider.valueChanged['int'].connect(self.setScaleBoxVal)
        self.dlg.Scale_lineEdit.textChanged.connect(self.setScaleSlideVal)

        # Get file/folder inputs
        self.dlg.InputImg_button.clicked.connect(self.getImgFile)
        self.dlg.OutputImg_button.clicked.connect(self.selectOutDir)
        self.dlg.InputMsk_button.clicked.connect(self.getMskFile)

        # EXTRACT MODEL PARAMETERS
        # Get model filepath
        dir_path = __file__
        dir_path = dir_path[:-9]
        model_file = mod_dict[self.dlg.Model_comboBox.currentText()] #accesses model file name from model dictionary
        model_path = os.path.normpath(dir_path + "\pylc-master\data\models\\"+model_file)

        img_path = self.dlg.InputImg_lineEdit.text()
        out_path = self.dlg.OutputImg_lineEdit.text()
        scale_val = float(self.dlg.Scale_lineEdit.text())
        
        # Set up model arguments (with defaults)
        args = {'schema':None, 
                'model':model_path, 
                'img':img_path, 
                'mask':None, 
                'scale':scale_val, 
                'save_logits':None, 
                'aggregate_metrics':None,
                'output_dir':out_path}

        # Check for optional model arguments
        mask_path = self.dlg.InputMsk_lineEdit.text()
        if mask_path:
            args['mask'] = mask_path

        # RUN PYLC
        #self.dlg.Run_pushButton.clicked.connect(pylc.main(args)) # Run PyLC and display outputs

        # DISPLAY OUPUTS
        #output file path will be: outputDir\imageFileName_imageExtension_scale_1.0.png
        self.outputFile = "C:\WLNP\pylc-master\data\outputs\pylc_2-1_deeplab_ch3_schema_a\masks\image1_jpg_scale_1.0.png" # temp for testing

        #change this to run when PyLC is done --> maybe result = pylc.main(args) and return true from pylc.py when it finishes?
        self.dlg.Run_pushButton.clicked.connect(self.showMask)
        #self.dlg.Run_pushButton.clicked.connect(self.showImg)
        self.dlg.Run_pushButton.clicked.connect(lambda: showImag(self.dlg,"Image"))
        self.dlg.Run_pushButton.clicked.connect(self.enableTools)

        # Link the extent of the image to the extent of the mask
        self.dlg.Mask_mapCanvas.extentsChanged.connect(self.updateImg)
        self.dlg.Img_mapCanvas.extentsChanged.connect(self.updateMsk)

        # Connect tools to appropriate functions
        self.dlg.View_toolButton.clicked.connect(self.changeView)
        self.dlg.Pan_toolButton.clicked.connect(self.panCanvas)
        self.dlg.Fit_toolButton.clicked.connect(self.zoomToExt) # Zoom to mask extent
        self.dlg.Swipe_toolButton.clicked.connect(self.swipeTool)
        #self.dlg.FullScrn_toolButton.clicked.connect(self.fullScrn)

        # SHOW THE DIALOG
        self.dlg.Swipe_mapCanvas.hide()
        self.dlg.show()

        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result:
            # Do something useful here - delete the line containing pass and
            # substitute with your code.
            pass

