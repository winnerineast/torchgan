import os
import torch
import torchvision
import torch.nn as nn
from operator import itemgetter
from torch.nn import DataParallel
from .trainer import Trainer
from ..models.model import Generator, Discriminator

def parallelize(model, device_ids=None):
    assert isinstance(model, Generator) or isinstance(model, Discriminator)
    dataparallel = DataParallel(model, device_ids=device_ids)
    for prop in model.__dict__.keys():
        # Pytorch stores everything as a hidden variable. So we access only what we have stored
        if prop[0] != '_':
            setattr(dataparallel, prop, getattr(model, prop))
    return dataparallel

class ParallelTrainer(Trainer):
    def __init__(self, models, optimizers, losses_list, metrics_list=None, schedulers=None,
                 ncritic=None, batch_size=128, device_ids=None,
                 sample_size=8, epochs=5, checkpoints="./model/gan", retain_checkpoints=5,
                 recon="./images", log_tensorboard=True, test_noise=None, **kwargs):
        self.device = torch.device("cuda:0") if device_ids is None else device_ids[0]
        self.model_names = []
        for key, val in models.items():
            self.model_names.append(key)
            if "args" in val:
                setattr(self, key, parallelize((val["name"](**val["args"])).to(self.device), device_ids=device_ids))
            else:
                setattr(self, key, parallelize((val["name"]()).to(self.device), device_ids=device_ids))
        self.optimizer_names = []
        for key, val in optimizers.items():
            self.optimizer_names.append(key)
            model = getattr(self, key.split("_", 1)[1])
            if "args" in val:
                setattr(self, key, val["name"](model.parameters(), **val["args"]))
            else:
                setattr(self, key, val["name"](model.parameters()))
        self.schedulers = []
        if schedulers is not None:
            for key, val in schedulers.items():
                opt = getattr(self, key.split("_", 1)[1])
                if "args" in val:
                    self.schedulers.append(val["name"](opt, **val["args"]))
                else:
                    self.schedulers.append(val["name"](opt))
        self.losses = {}
        self.loss_logs = {}
        for loss in losses_list:
            name = type(loss).__name__
            self.loss_logs[name] = []
            self.losses[name] = loss
        if metrics_list is None:
            self.metrics = None
            self.metric_logs = None
        else:
            self.metric_logs = {}
            self.metrics = {}
            for metric in metrics_list:
                name = type(metric).__name__
                self.metric_logs[name] = []
                self.metrics[name] = metric
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.epochs = epochs
        self.checkpoints = checkpoints
        self.retain_checkpoints = retain_checkpoints
        self.recon = recon
        self.test_noise = []
        for model in self.model_names:
            if isinstance(getattr(self, model), Generator):
                self.test_noise.append(torch.randn(self.sample_size, getattr(self, model).encoding_dims,
                                                   device=self.device) if test_noise is None else test_noise)
        # Not needed but we need to store this to avoid errors. Also makes life simpler
        self.noise = torch.randn(1)
        self.real_inputs = torch.randn(1)
        self.labels = torch.randn(1)

        self.loss_information = {
            'generator_losses': 0.0,
            'discriminator_losses': 0.0,
            'generator_iters': 0,
            'discriminator_iters': 0,
        }
        self.ncritic = ncritic
        self.start_epoch = 0
        self.last_retained_checkpoint = 0
        self.log_tensorboard = log_tensorboard
        if self.log_tensorboard:
            self.tensorboard_information = {
                "step": 0,
                "repeat_step": 4,
                "repeats": 1
            }
        self.nrow = 8
        for key, val in kwargs.items():
            if key in self.__dict__:
                warn("Overiding the default value of {} from {} to {}".format(key, getattr(self, key), val))
            setattr(self, key, val)

        os.makedirs(self.checkpoints, exist_ok=True)
        os.makedirs(self.recon, exist_ok=True)
