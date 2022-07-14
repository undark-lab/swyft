import swyft.lightning as sl
import numpy as np
import torch
import pylab as plt
plt.switch_backend("agg")
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
import os
import hydra
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import shutil
from dataclasses import dataclass

class TorchBounds:
    def __init__(self, filename):
        self.bounds = torch.load(filename)
