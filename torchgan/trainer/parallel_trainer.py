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
    def __init__(self, models, losses_list, metrics_list=None, device_ids=None,
                 ncritic=None, batch_size=128, epochs=5, sample_size=8, checkpoints="./model/gan",
                 retain_checkpoints=5, recon="./images", log_dir=None, test_noise=None, nrow=8, **kwargs):
        super(ParallelTrainer, self).__init__({}, losses_list, metrics_list,
                                              torch.device("cuda:0") if device_ids is None else device_ids[0],
                                              ncritic, batch_size, epochs, sample_size, checkpoints,
                                              retain_checkpoints, recon, log_dir, test_noise, nrow, **kwargs)
        self.model_names = []
        self.optimizer_names = []
        self.schedulers = []
        for key, model in models.items():
            self.model_names.append(key)
            if "args" in model:
                setattr(self, key, parallelize((model["name"](**model["args"])).to(self.device), device_ids=device_ids))
            else:
                setattr(self, key, parallelize((model["name"]()).to(self.device), device_ids=device_ids))
            opt = model["optimizer"]
            opt_name = "optimizer_{}".format(key)
            if "var" in opt:
                opt_name = opt["var"]
            self.optimizer_names.append(opt_name)
            model_params = getattr(self, key).parameters()
            if "args" in opt:
                setattr(self, opt_name, (opt["name"](model_params, **opt["args"])))
            else:
                setattr(self, opt_name, (opt["name"](model_params)))
            if "scheduler" in opt:
                sched = opt["scheduler"]
                if "args" in sched:
                    self.schedulers.append(sched["name"](getattr(self, opt_name), **sched["args"]))
                else:
                    self.schedulers.append(sched["name"](getattr(self, opt_name)))
