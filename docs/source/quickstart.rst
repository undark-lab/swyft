Quickstart with *swyft*
=====================
Run this example in a colab notebook here_.

..  _here: https://colab.research.google.com/github/undark-lab/swyft/blob/master/notebooks/Quickstart.ipynb

As a quick example, the following code defines a simple "simulator" and noise model and performs inference given a particular draw.
:: 
    import numpy as np
    import pylab as plt
    import torch
    import swyft
    np.random.seed(25)
    torch.manual_seed(25)
    
    DEVICE = 'cuda:0' #your gpu, or 'cpu' if a gpu is not available
    MAX_EPOCHS = 100 #maximum number of epochs per training round
    EXPECTED_N = 10000 #the average number of samples for the algorithm to see per training round
    
    #a simple simulator
    def simulator(z):
        return np.array([z[0],2*(z[1]-z[0])])
    #a simple noise model
    def noise(x, z = None, noise=0.01):
        n = np.random.randn(*x.shape)*noise
        return x + n
    #choose the "true" parameters for an inference problem
    z0 = np.array([0.55,0.45])
    zdim = len(z0)
    x0 = simulator(z0)  # Using Asimov data
    
    #a simple inference
    points, re = swyft.run(x0, simulator, zdim = 2, noise = noise, device = DEVICE, n_train = 10000,n_rounds=4)
  
The resulting 1-dimensional posteriors can be plotted:
::
    swyft.plot1d(re, x0 = x0, z0 = z0, cmap = 'Greys')
    
.. image:: images/quickstart-1d.png
   :width: 600

The 2-dimensional posterior can be easily trained:
::
    re2 = swyft.RatioEstimator(x0, points, combinations = [[0, 1]], device=DEVICE)
    re2.train(max_epochs=MAX_EPOCHS, batch_size=32, lr_schedule=[1e-3, 3e-4, 1e-4])

Allowing one to generate a classic triangle plot:
::
    swyft.corner(re, re2, x0 = x0, z0 = z0, cmap = 'Greys', dim = 10)
    
.. image:: images/quickstart-2d.png
   :width: 1000

For details on tweaking *swyft*, see the tutorial as a notebook on github_ or colab_.

.. _github: https://github.com/undark-lab/swyft/blob/master/notebooks/Quickstart.ipynb
.. _colab: https://colab.research.google.com/github/undark-lab/swyft/blob/master/notebooks/Tutorial.ipynb