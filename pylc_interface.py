


from qgis.core import QgsRasterLayer, QgsProject


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

def showImg(dialog, name):
    """Displays image in PyLC tool tab"""

    img_pth = dlg.InputImg_lineEdit.text()
    img_lyr= QgsRasterLayer(img_pth, name)
    # Load mask as layer into mask map canvas
    QgsProject.instance().addMapLayer(img_lyr, False) # add layer to the registry (but don't load into main map)
    dlg.Img_mapCanvas.enableAntiAliasing(True)
    dlg.Img_mapCanvas.setExtent(img_lyr.extent()) # set extent to the extent of the image layer
    dlg.Img_mapCanvas.setLayers([img_lyr])
    dlg.Img_mapCanvas.show()

