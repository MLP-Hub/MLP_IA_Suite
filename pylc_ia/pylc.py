"""
(c) 2020 Spencer Rose, MIT Licence
Python Landscape Classification Tool (PyLC)
An evaluation of deep learning semantic segmentation for
land cover classification of oblique ground-based photography
MSc. Thesis 2020.
<http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: Application
File: pylc.py
"""
import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pylc_ia.test import test_model


def main(args):
    """
    Main application handler
    """
    # Get parsed input arguments
    
    test_model(args)
