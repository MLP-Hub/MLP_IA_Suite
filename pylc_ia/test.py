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

    # Load model for testing/evaluation
    model = Model().load(model_path)
    model.print_settings()
    model.net.eval()

    # get test file(s) - returns list of filenames
    files = utils.load_files(args['img'], ['.tif', '.tiff', '.jpg', '.jpeg', '.TIFF', '.JPEG', '.JPG', '.TIF'])
    # initialize extractor, evaluator
    extractor = Extractor(model.meta)
    evaluator = Evaluator(model.meta)

    for img_file in files:

        # extract image tiles (image is resized and cropped to fit tile size)
        img_tiles = extractor.load(img_file).extract(
            fit=True,
            stride=params.tile_size // 2,
            scale=params.scale
        ).imgs

        # apply model to input tiles
        with torch.no_grad():
            # get model outputs
            model_outputs = []
            
            progressDlg = QProgressDialog("Running classification...","Cancel", 0, len(img_tiles))
            progressDlg.setWindowModality(Qt.WindowModal)
            progressDlg.setValue(0)
            progressDlg.forceShow()
            progressDlg.show()
            
            for i, tile in enumerate(img_tiles):
                progressDlg.setValue(i)
                logits = model.test(torch.Tensor(tile))
                model_outputs += logits
                model.iter += 1

        # load results into evaluator
        results, probs = utils.reconstruct(model_outputs, extractor.get_meta())
        
        # - save full-sized predicted mask image to file
        evaluator.load(results, probs, extractor.get_meta()).save_image(args)

        if args['save_probs']:
            evaluator.save_probs(args)

        # Reset evaluator
        evaluator.reset()


