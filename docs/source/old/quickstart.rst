Quickstart with *swyft*
=======================
Run this example in a colab notebook here_.

..  _here: https://colab.research.google.com/github/undark-lab/swyft/blob/master/notebooks/Quickstart.ipynb

.. As a quick example, the following code defines a simple "simulator" and noise model and performs inference given a particular draw.
.. ::
..     import numpy as np
..     import pylab as plt
..     import swyft
..     import torch
..     from scipy import stats

..     DEVICE = 'cuda' #your gpu, or 'cpu' if a gpu is not available

..     #a simple simulator
..     def model(params):
..         a = params['a']
..         b = params['b']
..         x=np.array([a,2*(b-a)])
..         return dict(mu=x)

..     #a simple noise model
..     def noise(obs, params, noise = 0.01):
..         x = obs['mu']
..         n = np.random.randn(*x.shape)*noise
..         return dict(x=x + n)

..     #choose the "true" parameters for an inference problem
..     par0 = dict(a=0.55, b=0.45)
..     obs0 = model(par0) # using Asimov data

..     #give priors for model parameters
..     prior = swyft.Prior({"a": ["uniform", 0., 1.], "b": ["uniform",  0., 1.]})

..     #a simple inference
..     s = swyft.NestedRatios(model, prior, noise = noise, obs = obs0, device = DEVICE)
..     #train!
..     s.run(Ninit = 500)

.. The last line, which trains networks that estimate the 1-dimensional marginal posteriors, will output something like:
.. ::
..     Simulate:  14%|█▎        | 67/495 [00:00<00:00, 667.16it/s]

..     NRE ROUND 0

..     Simulate: 100%|██████████| 495/495 [00:00<00:00, 644.85it/s]

..     NRE ROUND 1

..     Simulate: 100%|██████████| 517/517 [00:00<00:00, 643.51it/s]

..     NRE ROUND 2

..     Simulate: 100%|██████████| 498/498 [00:00<00:00, 713.97it/s]

..     NRE ROUND 3

..     Simulate: 100%|██████████| 820/820 [00:01<00:00, 647.67it/s]

..     NRE ROUND 4

..     Simulate: 100%|██████████| 1598/1598 [00:02<00:00, 653.44it/s]

..     NRE ROUND 5

..     Simulate: 100%|██████████| 2745/2745 [00:04<00:00, 672.84it/s]

..     NRE ROUND 6

..     Simulate: 100%|██████████| 5027/5027 [00:07<00:00, 704.09it/s]

..     NRE ROUND 7
..     --> Posterior volume is converged. <--


.. This "zooms in" to the relevant region of parameter space. The resulting marginal posteriors can be plotted:
.. ::
..     #train 2d marginals
..     post = s.gen_2d_marginals(N = 15000)
..     #generate samples at which to evaluate posteriors
..     samples = post(obs0, 1000000);
..     #plot estimated posteriors
..     swyft.corner(samples, ["a", "b"], color='k', figsize = (15,15), truth=par0)

.. .. image:: images/quickstart-2d.png
..    :width: 600

For details on tweaking *swyft*, see the tutorial as a notebook on github_ or colab_.

.. _github: https://github.com/undark-lab/swyft/blob/master/notebooks/Tutorial.ipynb
.. _colab: https://colab.research.google.com/github/undark-lab/swyft/blob/master/notebooks/Tutorial.ipynb
