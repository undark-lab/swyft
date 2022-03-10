import os
import shutil
import numpy as np
import pylab as plt
plt.switch_backend("agg")
import torch
import hydra
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import swyft.lightning as sl


def simulate(cfg):
    # Loading simulator (potentially bounded)
    simulator = hydra.utils.instantiate(cfg.simulation.model)

    # Generate or load training data & generate datamodule
    train_samples = sl.file_cache(lambda:
            simulator.sample(cfg.simulation.store.train_size),
            cfg.simulation.store.path)
    datamodule = sl.SwyftDataModule(store = train_samples, model = None,
            batch_size = cfg.estimation.batch_size, num_workers =
            cfg.estimation.num_workers)

    return datamodule


def analyse(cfg, datamodule):
    # Setting up tensorboard logger, which defines also logdir (contains trained network)
    tbl = pl_loggers.TensorBoardLogger(save_dir = cfg.tensorboard.save_dir,
            name = cfg.tensorboard.name, version = cfg.tensorboard.version,
            default_hp_metric=False)
    logdir = tbl.experiment.get_logdir()  # Directory where all logging information and checkpoints etc are stored

    # Load network and train (or re-load trained network)
    network = hydra.utils.instantiate(cfg.estimation.network, cfg)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stop_callback = EarlyStopping(monitor="val_loss",
            min_delta=cfg.estimation.early_stopping.min_delta,
            patience=cfg.estimation.early_stopping.patience, verbose=False,
            mode="min")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=logdir+"/checkpoint/",
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )
    t = sl.SwyftTrainer(accelerator = cfg.estimation.accelerator, gpus=1,
            max_epochs = cfg.estimation.max_epochs, logger = tbl,
            callbacks=[lr_monitor, early_stop_callback, checkpoint_callback])
    best_checkpoint = logdir+"/checkpoint/best.ckpt"
    if not os.path.isfile(best_checkpoint):
        t.fit(network, datamodule)
        shutil.copy(checkpoint_callback.best_model_path, best_checkpoint)
        t.test(network, datamodule)
    else:
        t.fit(network, datamodule, ckpt_path = best_checkpoint)

    return dict(network=network, trainer=t, datamodule=datamodule, tbl=tbl)


def interpret(cfg, network, trainer, datamodule, tbl):
    hydra.utils.instantiate(cfg.inference.interpreter, cfg, network, trainer, datamodule, tbl)


@hydra.main(config_path=".", config_name="config")
def main(cfg):
    datamodule = simulate(cfg)
    results = analyse(cfg, datamodule)
    interpret(cfg, **results)


if __name__ == "__main__":
    main()
