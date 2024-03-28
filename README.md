# Mountain Image Analysis Suite (MIAS)
### QGIS plugin for analysing ground-based oblique images 

Reference: PUT PAPER HERE

## Overview
This plugin contains four tools for analysing oblique terrestrial photographs. The final product is a classified and spatially referenced viewshed representing the landscape shown in the photograph. MIAS harnesses the PyLC (Python Landscape Classifie) available indipendently [here](https://github.com/scrose/pylc). The training dataset is sampled from the [Mountain Legacy Project](https://mountainlegacy.ca) repeat photography collection hosted at the [University of Victoria](https://www.uvic.ca).

## Requirements
QGIS 3.28.1
Python 3.9

numpy >=1.18.5
opencv >=3.4.1
torch >=1.6.0
scikit-image >= VERSION NUMBER

## Installation

### Windows Users
Install the latest version of QGIS through the OSGEO installer from URL. INSTRUCTIONS

### MacOS Users
Install the latest version of QGIS from URL.

### Dependencies
MIAS relies on some Python packages that do not come installed with QGIS and has conflicts with the existing versions of opencv and numpy. From QGIS, open the Python console and type the following commands:
```
import pip
pip.main(['uninstall','-y','opencv-contrib-python'])
pip.main(['install','opencv-python'])
pip.main(['install','--upgrage','numpy'])
pip.main(['install','torch'])
pip.main(['install','scikit-image'])
```

## Usage
A video tutorial for MIAS is available at URL.
A written tutorial and corresponding test data are found in the Example_1 folder.


