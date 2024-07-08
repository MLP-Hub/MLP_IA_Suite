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

# Import the code for pylc
import sys
import os

this_dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(this_dir, 'pylc_master')
sys.path.append(path)
import test


def main(args):
    """
    Main application handler
    """
    # execute processing function
    test.tester(args)


if __name__ == "__main__":
    main()
