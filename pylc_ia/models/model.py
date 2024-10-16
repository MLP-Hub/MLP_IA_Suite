"""
(c) 2020 Spencer Rose, MIT Licence
Python Landscape Classification Tool (PyLC)
 Reference: An evaluation of deep learning semantic segmentation
 for land cover classification of oblique ground-based photography,
 MSc. Thesis 2020.
 <http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: Base Model Class
File: model.py
"""
import os
import torch
import torch.utils.data
from torch import nn
import numpy as np
from models.architectures.deeplab import DeepLab
from config import defaults
from utils.tools import get_fname


class Model:
    """
    Abstract model for Pytorch network configuration.
    Uses Pytorch Model class as superclass
    """

    def __init__(self):

        super(Model, self).__init__()

        # initialize local metadata
        self.meta = defaults
        self.device = torch.device(self.meta.device)

        # build network
        self.net = None
        self.model_path = None

        # initialize global iteration counter
        self.iter = 0

        # initialize network parameters
        self.crit = None
        self.loss = None
        self.epoch = 0
        self.optim = None
        self.sched = None
        self.crop_target = False

        # initialize checkpoint
        self.checkpoint = None
        self.resume_checkpoint = False

        # layer activation functions
        self.activations = nn.ModuleDict({
            'relu': torch.nn.ReLU(inplace=True),
            'lrelu': nn.LeakyReLU(negative_slope=0.01, inplace=True),
            'prelu': nn.PReLU(),
            'selu': torch.nn.SELU(inplace=True)
        })

        # layer normalizers
        self.normalizers = {
            'batch': torch.nn.BatchNorm2d,
            'instance': torch.nn.InstanceNorm2d,
            'layer': torch.nn.LayerNorm,
            'syncbatch': torch.nn.SyncBatchNorm
        }

    def load(self, model_path):
        """
        Loads models PyLC model for evaluation.

        Parameters
        ----------
        model_path: str
            Path to PyLC models model.
        """

        if not model_path:
            print("\nModel path is empty. Use \'--model\' option to specify path.")
            exit(1)
        else:
            print('\nLoading model:\n\t{}'.format(model_path))

        if os.path.exists(model_path):
            self.model_path = model_path
            model_data = None

            # load model data
            try:
                model_data = torch.load(self.model_path, map_location=self.device)
            except Exception as err:
                print('An error occurred loading model:\n\t{}.'.format(model_path))
                print(err)
                exit()

            # build model from metadata
            assert 'meta' in model_data, '\nLoaded model missing metadata attribute.'

            # build model from metadata
            self.meta.update(model_data["meta"])
            self.meta.pretrained = False
            self.build()

            # load model state
            self.net.load_state_dict(model_data["model"])

        else:
            print('Model file does not exist.')
            exit()

        return self

    def build(self):
        """
        Builds neural network model from configuration settings.
        """

        # create model identifier if none exists
        # format: pylc_<architecture>_<channel_label>_<schema_id>
        self.gen_id()
        
        # DeeplabV3+
        if self.meta.arch == 'deeplab':
            self.net = DeepLab(
                activ_func=self.activations[self.meta.activ_type],
                normalizer=self.normalizers[self.meta.norm_type],
                backbone=self.meta.backbone,
                n_classes=self.meta.n_classes,
                in_channels=self.meta.ch,
                pretrained=self.meta.pretrained
            )
            self.net = self.net.to(self.device)
        # Unknown model requested
        else:
            print('Model {} not available.'.format(self.meta.arch))
            exit(1)

        # Enable CUDA
        if torch.cuda.is_available():
            print("\n --- CUDA enabled.")

        # Parallelize model on multiple GPUs (disabled)
        if torch.cuda.device_count() > 1:
            print("\t{} GPUs in use.".format(torch.cuda.device_count()))
            # self.net = torch.nn.DataParallel(self.net)

        # Check multiprocessing enabled
        if torch.utils.data.get_worker_info():
            print('\tPooled data loading: {} workers enabled.'.format(
                torch.utils.data.get_worker_info().num_workers))

        return self

    def test(self, x):

        """model test forward"""

        # normalize
        x = self.normalize_image(x, default=self.meta.normalize_default)
        x = x.to(self.device).float()

        # stack single-channel input tensors (Deeplab)
        if self.meta.ch == 1 and self.meta.arch == 'deeplab':
            x = torch.cat((x, x, x), 1)

        # run forward pass
        with torch.no_grad():
            y_hat = self.net.forward(x)
            return [y_hat]

    def get_meta(self):
        """Get model metadata."""
        return self.meta


    def normalize_image(self, img, default=False):
        """
        Normalize input image data [NCWH]
            - uses precomputed mean/std of pixel intensities

        Parameters
        ----------
        img: np.array
            Input image.
        default: bool
            Use default pixel mean/std deviation values.
        """
        # grayscale
        if img.shape[1] == 1:
            if default:
                return torch.tensor(
                    (img.numpy().astype('float32') - defaults.px_grayscale_mean) / defaults.px_grayscale_std)
            mean = np.mean(self.meta.px_mean)
            std = np.mean(self.meta.px_std)
            return torch.tensor((img.numpy().astype('float32') - mean) / std) / 255
        # colour
        else:
            if default:
                px_mean = torch.tensor(defaults.px_rgb_mean)
                px_std = torch.tensor(defaults.px_rgb_std)
            else:
                px_mean = torch.tensor(self.meta.px_mean)
                px_std = torch.tensor(self.meta.px_std)
            return ((img - px_mean[None, :, None, None]) /
                    px_std[None, :, None, None]) / 255

    def print_settings(self):
        """
        Prints model configuration settings to screen.
        """
        hline = '_' * 40
        print("\nModel Configuration")
        print(hline)
        print('{:30s} {}'.format('ID', self.meta.id))
        if self.model_path is not None:
            print('{:30s} {}'.format('Model File', os.path.basename(self.model_path)))
        print('{:30s} {}'.format('Architecture', self.meta.arch))
        # show encoder backbone for Deeplab
        if self.meta.arch == 'deeplab':
            print('   - {:25s} {}'.format('Backbone', self.meta.backbone))
            print('   - {:25s} {}'.format('Pretrained model', self.meta.pretrained))
        print('{:30s} {}'.format('Input channels', self.meta.ch))
        print('{:30s} {}'.format('Output channels', self.meta.n_classes))
        print('{:30s} {}{}'.format('Px mean', self.meta.px_mean,
                                   '*' if self.meta.normalize_default else ''))
        print('{:30s} {}{}'.format('Px std-dev', self.meta.px_std,
                                   '*' if self.meta.normalize_default else ''))
        print('{:30s} {}'.format('Batch size', self.meta.batch_size))
        print('{:30s} {}'.format('Activation function', self.meta.activ_type))
        print('{:30s} {}'.format('Optimizer', self.meta.optim_type))
        print('{:30s} {}'.format('Scheduler', self.meta.sched_type))
        print('{:30s} {}'.format('Learning rate (default)', self.meta.lr))
        print('{:30s} {}'.format('Resume checkpoint', self.resume_checkpoint))
        print()
        # use default pixel normalization (if requested)
        if self.meta.normalize_default:
            print('* Normalized default settings')



    def gen_id(self):
        """
        Generate model identifier from metadata and assign to id.
        """
        if self.model_path is None:
            # format: pylc_<architecture>_ch<channels>_<schema_id>
            self.meta.id = 'pylc_' + self.meta.arch + '_ch' + \
                           str(self.meta.ch) + '_' + \
                           self.meta.schema_name
        else:
            self.meta.id = get_fname(self.model_path)
