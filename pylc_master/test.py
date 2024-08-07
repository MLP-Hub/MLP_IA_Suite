"""
(c) 2020 Spencer Rose, MIT Licence
Python Landscape Classification Tool (PyLC)
An evaluation of deep learning semantic segmentation for
land cover classification of oblique ground-based photography
MSc. Thesis 2020.
<http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: Model Test
File: test.py
"""
import torch
import utils.tools as utils
from config import defaults, Parameters
from utils.extract import Extractor
from utils.evaluate import Evaluator
from models.model import Model

from qgis.PyQt.QtWidgets import QProgressDialog
from qgis.PyQt.QtCore import Qt


def test_model(args):
    """
    Apply model to input image(s) to generate segmentation maps.

    Parameters
    ----------
    args: dict
        User-defined options.
    """
    # trained model path
    model_path = args['model']

    # load parameters
    params = Parameters(args)
    if params is None:
        return

    # Load model for testing/evaluation
    model = Model(args).load(model_path)
    if model is None:
        return
    model.net.eval()

    # get test file(s) - returns list of filenames
    files = utils.collate(args['img'], args['mask'])
    if files is None:
        return

    # initialize extractor, evaluator
    extractor = Extractor(model.meta)
    evaluator = Evaluator(model.meta)

    for f_idx, fpair in enumerate(files):

        # get image and associated mask data (if provided)
        if type(fpair) == dict and 'img' in fpair and 'mask' in fpair:
            img_file = fpair.get('img')
            mask_file = fpair.get('mask')
        else:
            img_file = fpair
            mask_file = None

        # extract image tiles (image is resized and cropped to fit tile size)
        img_tiles = extractor.load(img_file)
        if img_tiles is None:
            return
        img_tiles = img_tiles.extract(
            fit=True,
            stride=defaults.tile_size // 2,
            scale=params.scale
        )
        if img_tiles is None:
            return
        img_tiles = img_tiles.get_data()

        # get data loader
        img_loader, n_batches = img_tiles.loader(
            batch_size=8,
            drop_last=False
        )

        # apply model to input tiles
        with torch.no_grad():
            # get model outputs
            model_outputs = []
            progressDlg = QProgressDialog("Running classification...","Cancel", 0, n_batches)
            progressDlg.setWindowModality(Qt.WindowModal)
            progressDlg.setValue(0)
            progressDlg.forceShow()
            progressDlg.show()

            for i, (tile, _) in enumerate(img_loader):
                progressDlg.setValue(i)
                logits = model.test(tile)
                model_outputs += logits
                model.iter += 1

        # load results into evaluator
        results = utils.reconstruct(model_outputs, extractor.get_meta())
        # - save full-sized predicted mask image to file
        if mask_file:
            evaluator.load(
                results,
                extractor.get_meta(),
                mask_true_path=mask_file,
                scale=params.scale
            ).save_image(args)

            # Evaluate prediction against ground-truth
            # - skip if only global/aggregated requested

        else:
            evaluator.load(results, extractor.get_meta()).save_image(args)

        # save unnormalized models outputs (i.e. raw logits) to file (if requested)
        # if args['save_logits']:
            # evaluator.save_logits(model_outputs)

        # Reset evaluator
        evaluator.reset()

    # Compute global metrics
    if args['aggregate_metrics']:
        evaluator.evaluate(aggregate=True)
