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
from models.architectures.unet import UNet
from models.architectures.res_unet import ResUNet
from models.architectures.deeplab import DeepLab
from models.modules.loss import MultiLoss, RunningLoss
from models.modules.checkpoint import Checkpoint
from numpy import random
from config import defaults, Parameters
from utils.tools import get_fname

from interface_tools import errorMessage


class Model:
    """
    Abstract model for Pytorch network configuration.
    Uses Pytorch Model class as superclass
    """

    def __init__(self, args):

        super(Model, self).__init__()

        # initialize local metadata
        self.meta = Parameters(args)
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
            errorMessage("\nModel path is empty. Use \'--model\' option to specify path.")
            return

        if os.path.exists(model_path):
            self.model_path = model_path
            model_data = None

            # load model data
            try:
                model_data = torch.load(self.model_path, map_location=self.device)
            except Exception as err:
                errorMessage('An error occurred loading model:\n\t{}.'.format(model_path))
                errorMessage(err)
                return

            # build model from metadata
            assert 'meta' in model_data, '\nLoaded model missing metadata attribute.'

            # build model from metadata
            self.meta.update(model_data["meta"])
            self.meta.pretrained = False
            self.build()

            # load model state
            self.net.load_state_dict(model_data["model"])

        else:
            errorMessage('Model file does not exist.')
            return

        return self

    def build(self):
        """
        Builds neural network model from configuration settings.
        """

        # create model identifier if none exists
        # format: pylc_<architecture>_<channel_label>_<schema_id>
        self.gen_id()

        # initialize checkpoint
        self.checkpoint = Checkpoint(
            self.meta.id,
            self.meta.save_dir
        )

        # UNet
        if self.meta.arch == 'unet':
            self.net = UNet(
                in_channels=self.meta.ch,
                n_classes=self.meta.n_classes,
                up_mode=self.meta.up_mode,
                activ_func=self.activations[self.meta.activ_type],
                normalizer=self.normalizers[self.meta.norm_type],
                dropout=self.meta.dropout
            )
            self.net = self.net.to(self.device)
            self.crop_target = self.meta.crop_target

        # Alternate Residual UNet
        elif self.meta.arch == 'resunet':
            self.net = ResUNet(
                in_channels=self.meta.ch,
                n_classes=self.meta.n_classes,
                up_mode=self.meta.up_mode,
                activ_func=self.activations[self.meta.activ_type],
                batch_norm=True,
                dropout=self.meta.dropout
            )
            self.net = self.net.to(self.device)
            self.crop_target = self.meta.crop_target

        # DeeplabV3+
        elif self.meta.arch == 'deeplab':
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
            errorMessage('Model {} not available.'.format(self.meta.arch))
            return

        # initialize network loss calculators, etc.
        self.crit = MultiLoss(
            loss_weights={
                'weighted': self.meta.weighted,
                'weights': self.meta.weights,
                'ce': self.meta.ce_weight,
                'dice': self.meta.dice_weight,
                'focal': self.meta.focal_weight
            },
            schema={
                'n_classes': self.meta.n_classes,
                'class_codes': self.meta.class_codes,
                'class_labels': self.meta.class_labels
            }
        )
        self.loss = RunningLoss(
            self.meta.id,
            save_dir=self.meta.save_dir,
            resume=self.meta.resume_checkpoint
        )

        # initialize optimizer and optimizer scheduler
        self.optim = self.init_optim()
        self.sched = self.init_sched()
        if self.optim is None or self.sched is None:
            return

        return self

    def resume(self):
        """
        Check for existing checkpoint. If exists, resume from
        previous training. If not, delete the checkpoint.
        """
        if self.resume_checkpoint:
            checkpoint_data = self.checkpoint.load()
            if checkpoint_data is not None:
                self.epoch = checkpoint_data['epoch']
                self.iter = checkpoint_data['iter']
                self.meta = checkpoint_data["meta"]
                self.net.load_state_dict(checkpoint_data["model"])
                self.optim.load_state_dict(checkpoint_data["optim"])
        else:
            self.checkpoint.reset()

    def init_optim(self):
        """select optimizer"""
        if self.meta.optim_type == 'adam':
            return torch.optim.AdamW(
                self.net.parameters(),
                lr=self.meta.lr,
                weight_decay=self.meta.weight_decay
            )
        elif self.meta.optim_type == 'sgd':
            return torch.optim.SGD(
                self.net.parameters(),
                lr=self.meta.lr,
                momentum=self.meta.momentum
            )
        else:
            errorMessage('Optimizer is not defined.')
            return

    def init_sched(self):
        """(Optional) Scheduled learning rate step"""
        if self.meta.sched_type == 'step_lr':
            return torch.optim.lr_scheduler.StepLR(
                self.optim,
                step_size=1,
                gamma=self.meta.gamma
            )
        elif self.meta.sched_type == 'cyclic_lr':
            return torch.optim.lr_scheduler.CyclicLR(
                self.optim,
                self.meta.lr_min,
                self.meta.lr_max,
                step_size_up=2000
            )
        elif self.meta.sched_type == 'anneal':
            return
            # steps_per_epoch = int(self.meta.clip * 29000 // self.meta.batch_size)
            # return torch.optim.lr_scheduler.CosineAnnealingLR(
            #     self.optim
            # )

        else:
            errorMessage('Optimizer scheduler is not defined.')
            return

    def train(self, x, y):

        """
        Model training step.

        Parameters
        ----------
        x: torch.tensor
            Input training image tensor.
        y: torch.tensor
            Input training mask tensor.
        """

        # apply random vertical flip
        if bool(random.randint(0, 1)):
            x = torch.flip(x, [3])
            y = torch.flip(y, [2])

        # normalize input [NCWH]
        x = self.normalize_image(x)
        x = x.to(self.device)
        y = y.to(self.device)

        # crop target mask to fit output size (e.g. UNet model)
        if self.meta.arch == 'unet':
            y = y[:, self.meta.crop_left:self.meta.crop_right, self.meta.crop_up:self.meta.crop_down]

        # stack single-channel input tensors (deeplab)
        if self.meta.ch == 1 and self.meta.arch == 'deeplab':
            x = torch.cat((x, x, x), 1)

        # forward pass
        y_hat = self.net.forward(x)

        # compute losses
        loss = self.crit.forward(y_hat, y)

        self.loss.intv += [(self.crit.ce.item(), self.crit.dsc.item(), self.crit.fl.item())]

        # zero gradients, compute, step, log losses,
        self.optim.zero_grad()
        loss.backward()

        # in-place normalization of gradients
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)

        self.optim.step()

        if self.iter % self.meta.report == 0:
            self.log()

        # log learning rate
        self.loss.lr += [(self.iter, self.get_lr())]

        self.iter += 1

    def eval(self, x, y):

        """model test/validation step"""

        self.net.eval()

        # normalize
        x = self.normalize_image(x)
        x = x.to(self.device)
        y = y.to(self.device)

        # crop target mask to fit output size (UNet)
        if self.meta.arch == 'unet':
            y = y[:, self.meta.crop_left:self.meta.crop_right, self.meta.crop_up:self.meta.crop_down]

        # stack single-channel input tensors (Deeplab)
        if self.meta.ch == 1 and self.meta.arch == 'deeplab':
            x = torch.cat((x, x, x), 1)

        # run forward pass
        with torch.no_grad():
            y_hat = self.net.forward(x)
            ce = self.crit.ce_loss(y_hat, y).cpu().numpy()
            dice = self.crit.dice_loss(y_hat, y).cpu().numpy()
            focal = self.crit.focal_loss(y_hat, y).cpu().numpy()
            self.loss.intv += [(ce, dice, focal)]

        return [y_hat]

    def test(self, x):

        """model test forward"""

        # normalize
        x = self.normalize_image(x, default=self.meta.normalize_default)
        x = x.to(self.device)

        # stack single-channel input tensors (Deeplab)
        if self.meta.ch == 1 and self.meta.arch == 'deeplab':
            x = torch.cat((x, x, x), 1)

        # run forward pass
        with torch.no_grad():
            y_hat = self.net.forward(x)
            return [y_hat]

    def log(self):
        """log ce/dice losses at defined intervals"""
        self.loss.log(self.iter, self.net.training)
        self.loss.save()

    def save(self):
        """save model checkpoint"""
        self.checkpoint.save(self, is_best=self.loss.is_best)
        self.loss.save()

    def get_lr(self):
        """Get current learning rate."""
        for param_group in self.optim.param_groups:
            return param_group['lr']

    def get_meta(self):
        """Get model metadata."""
        return self.meta

    def update_meta(self, params):
        """
        Update model metadata.

        Parameters
        ----------
        params: Parameters
            Model metadata.
        """
        self.meta.update(params)

        return self

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
