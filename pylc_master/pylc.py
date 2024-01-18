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

# Import the code for pylc
from mlp_ia_suite.pylc_master.test import tester
from mlp_ia_suite.pylc_master.config import defaults


def main(args):
    """
    Main application handler
    """
    # Get parsed input arguments

    #GET INPUT ARGUMENTS FROM QGIS -> make sure to set defaults

    # ensure data directories exist in project root
    # HOW MANY OF THESE DO I NEED?
    dirs = [defaults.root, defaults.db_dir, defaults.save_dir, defaults.model_dir, defaults.output_dir]
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    # execute processing function
    tester(args)


if __name__ == "__main__":
    main()
