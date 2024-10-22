"""
(c) 2020 Spencer Rose, MIT Licence
Python Landscape Classification Tool (PyLC)
 Reference: An evaluation of deep learning semantic segmentation
 for land cover classification of oblique ground-based photography,
 MSc. Thesis 2020.
 <http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: Evaluator Class
File: evaluate.py
"""


import os, sys
import numpy as np
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import defaults, Parameters

from interface_tools import errorMessage

class Evaluator:
    """
    Handles model test/evaluation functionality.

    Parameters
    ------
    params: Parameters
        Updated parameters.
    """

    def __init__(self, params=None):

        # initialize parameters, metrics
        self.meta = Parameters(params) if params is not None else defaults

        # Model results
        self.fid = None
        self.logits = None
        self.mask_pred = None
        self.probs_pred = None
        self.results = []

        # data buffers
        self.y_true = None
        self.y_pred = None
        self.labels = []

        # multi-image data buffers for aggregate evaluation
        self.aggregate = False
        self.y_true_aggregate = []
        self.y_pred_aggregate = []

        # Make output and mask directories for results
        self.model_path = None
        self.output_dir = os.path.join(defaults.output_dir, self.meta.id)

    def load(self, mask_pred, probs_pred, meta, mask_true_path=None, scale=None):
        """
        Initialize predicted/ground truth image masks for
        evaluation metrics.

        Parameters:
        -----------
        mask_pred_logits: torch.tensor
            Unnormalized model logits for predicted segmentation [NCHW]
        meta: dict
            Reconstruction metadata.
        mask_true_path: str
            File path to ground-truth mask [CHW]
        """

        # store metadata
        self.meta = meta
        # file identifier (include current scale)
        self.fid = self.meta.extract['fid']

        # reconstruct unnormalized model outputs into mask data array
        self.mask_pred = mask_pred
        self.probs_pred = probs_pred

        return self

    def update(self, meta):
        """
        Update local metadata

        """
        self.meta = meta
        return self
    

    def reset(self):
        """
        Resets evaluator buffers.
        """
        self.logits = None
        self.mask_pred = None
        self.probs_pred = None
        self.results = []
        self.meta = {}
        self.y_true = None
        self.y_pred = None

    def save_image(self, args):
        """
        Reconstructs segmentation prediction as mask image.
        Output mask image saved to file (RGB -> BGR conversion)
        Note that the default color format in OpenCV is often
        referred to as RGB but it is actually BGR (the bytes are
        reversed).

        Returns
        -------
        mask_data: np.array
            Output mask data.
        """

        # Build mask file path
        mask_file = args["mask_path"]

        if self.mask_pred is None:
            errorMessage("Mask has not been reconstructed. Image save cancelled.")

            # Reconstruct seg-mask from predicted tiles and write to file
        cv2.imwrite(mask_file, cv2.cvtColor(self.mask_pred, cv2.COLOR_RGB2BGR))
        return mask_file

    def save_probs(self, args):
        """
        Saves the probabilities for the most probable class at each pixel to file.

        Returns
        -------
        probs_file: np.array
            Output probability data.
        """
        
        mask_file = args["mask_path"]
        mask_name, ext = os.path.splitext(os.path.realpath(mask_file))
        probs_file = os.path.join(mask_name + '.npy')

        if self.probs_pred is None:
            errorMessage("Probabilities have not been reconstructed. Image save cancelled.")
            return

        np.save(probs_file, self.probs_pred)
        return probs_file
