# Mountain Image Analysis Suite (MIAS)
__QGIS plugin for analysing ground-based oblique images__ 

[![QGIS 3.28](https://img.shields.io/badge/QGIS-3.28.1-blue.svg)](https://www.qgis.org/en/site/forusers/download.html)
[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![GNU License](https://img.shields.io/badge/License-GNU-green.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12587817.svg)](https://doi.org/10.5281/zenodo.12587817)

Reference: Wright, C., Bone, C., Mathews, D., Tricker, J., Wright, B., and Higgs, E. (2024) Mountain Image Analysis Suite (MIAS): A new plugin for converting oblique images to landcover maps in QGIS. *Transactions in GIS*.[doi.org/10.1111/tgis.13229](doi.org/10.1111/tgis.13229)

## Overview
This plugin contains four tools for analysing ground-based oblique images. The final product is a classified and spatially referenced viewshed representing the landscape shown in the photograph. MIAS harnesses PyLC (Python Landscape Classifier) available independently [here](https://github.com/MLP-Hub/pylc_mlp). The training dataset is sampled from the [Mountain Legacy Project](https://mountainlegacy.ca) repeat photography collection hosted at the [University of Victoria](https://www.uvic.ca).

## Requirements (QGIS 3.28.1) 

- [numpy](https://numpy.org/) >=1.18.5
- [opencv](https://opencv.org/) >=3.4.1
- [torch](https://pytorch.org/) >=1.6.0
- [scikit-image](https://scikit-image.org/) >=0.19.3

## Installation

### Windows Users
Install the latest version of QGIS through the OSGeo4W installer from [here](https://qgis.org/en/site/forusers/alldownloads.html#osgeo4w-installer). From the OSGeo4W installer, select Express install. Choose QGIS LTR, GDAL, and GRASS GIS from the option menu. 

### MacOS Users
Install the latest version of QGIS [here](https://qgis.org/en/site/forusers/download.html).

### Dependencies
MIAS relies on some Python packages that do not come installed with QGIS and has conflicts with the existing versions of opencv and numpy. From the Plugins menu on QGIS, open the Python console and type the following commands:
```python
import pip
pip.main(['uninstall','-y','opencv-contrib-python'])
pip.main(['install','opencv-python'])
pip.main(['install','--upgrage','numpy'])
pip.main(['install','torch'])
pip.main(['install','scikit-image'])
```

### Installing the Plugin
From the Code menu (green button) on the [GitHub page](https://github.com/MLP-Hub/MLP_IA_Suite), select Download ZIP.  
From the Plugins menu in QGIS, choose Manage and Install Plugins, then Install from ZIP. Upload the ZIP file that you just downloaded from GitHub.

## Usage
A video tutorial for MIAS is available [here](https://youtu.be/y0ASg9S3Up8).
Example data and written instructions can be downloaded [here](https://doi.org/10.5683/SP3/PR368H).


