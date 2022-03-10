import numpy as np
import pylab as plt
plt.switch_backend("agg")
import torch
import swyft.lightning as sl

def interpret(cfg, network, trainer, datamodule, tbl):
    # Loading target observation inference data and perform inference
    observation = torch.load(cfg.inference.obs_path)

    # Generate weighted posterior samples, posterior samples, prior samples
    condition_x = dict(data = observation['data'])
    p = trainer.infer(network, datamodule, condition_x = condition_x)
    post_samples = p.sample(10000)['z']
    prior_samples = p.sample(10000000, replacement = False)['z']
    
    # Generate new bounds (if requested)
    new_bounds = None
    if cfg.inference.bound is not None:
        new_bounds = sl.get_1d_rect_bounds(dict(z = p['z']), th = 1e-6)
        torch.save(new_bounds, cfg.inference.bound)

    # Lots of plots
    for i in range(5):
        fig = plt.figure(dpi=100)
        plt.hist(prior_samples[:,i].numpy(), bins = 30, density= True);
        plt.hist(post_samples[:,i].numpy(), bins = 30, density = True);
        plt.axvline(observation['z'][i])
        tbl.experiment.add_figure("posterior/%i"%i, fig)

    train_samples = datamodule.samples(8)
    for i in range(8):
        fig = plt.figure(dpi=100)
        plt.imshow(np.log10(0.1+train_samples['data'][i].numpy()), cmap = 'inferno')
        tbl.experiment.add_figure("train_data/%i"%i, fig)

    fig = plt.figure(dpi=100)
    plt.imshow(np.log10(0.1+observation['data'].numpy()), cmap = 'inferno')
    tbl.experiment.add_figure("observation", fig)

    if new_bounds:
        fig = plt.figure(dpi=100)
        for i in range(5):
            plt.plot([i, i], [new_bounds['z'].low[i], new_bounds['z'].high[i]])
            plt.scatter([i], [observation['z'][i]], marker='o')
        tbl.experiment.add_figure("bounds", fig)

    tbl.experiment.flush()
    print("logdir:", tbl.experiment.get_logdir())
