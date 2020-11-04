Quickstart with SWYFT
=====================

General usage
-------------

SWYFT is simple to use.  First, we define a model function which takes an
argument numpy vector $z$ and returns simulated data $x$.::

    def model(z):
        m = fancy_simulator(z)
        n = noise()
        x = m + n
        return x

We then use the model to generate mock observational data.::

    z0 = np.array([0.5, 0.3])
    x0 = model(z0)

The analysis is then started by invoking SWYFT as follows.::

    from swyft import SWYFT

    sw = SWYFT(model, x0, zdim = 3)
    sw.run(nrounds = 3, nsamples = 5000)

This will call the simulator `model` around 15000 times, and sequentially zoom
into the parameter range that is consistent with mock data $x0$.  After
training, the 1-dim marginal posteriors can be plot using::

    from pylab import plot, show

    for i in range(zdim):
        z, p = sw.posterior(i)
        plot(z, p, label = i)
        axvline(z0[i])  # Comparison with true value
    show()

If, e.g., 2-dim posteriors for some of the parameter pairs are desired, they
can be generated simply as well.::

    SWYFT.comb([[0, 1]])
    z, p = SWYFT.posterior([0, 1])


iP3 Data Caching
----------

(IN PROGRESS)

Simulator runs can be automatically re-used.  This is done by specifying the
cache when invoking `SWYFT`::

    from swyft import MemoryCache, DirectoryCache

    ds = DirectoryCache(filename = 'ds.hdf5')

    sw = SWYFT(model, x0, zdim = 3, ds = ds)
    sw.run(nrounds = 3, nsamples = 5000)

This works just as above.  However, if we perform a similar analysis again
(with the same or other mock data), the number of simulator calls is, sometimes
greatly, reduced.::

    # Rerunning does require less sampler runs
    sw = SWYFT(model, x1, zdim = 3, ds = ds)
    sw.run(nrounds = 3, nsamples = 5000)


Custom head networks
--------------------

(IN PROGRESS)

Input data is assumed to be vector-like.  Usually, vectors up to few hundred
values work well out-of-the-box.  For larger sets of data (images, volumetric
data, in general diverse data from various experiments), pre-processing is
required.  This is done using a `Head` network.  SWYFT comes with a range of
Head networks for typical use-cases, but those networks can be also custom made
and user-defined.  For image analysis problems, a simple convolutional neural network can be used. Invoking SWYFT with the head-network is shown below.::

    from swyft import CNN

    sw = SWYFT(model, x0, zdim = 3, head = CNN)
    sw.run(nrounds = 3, nsamples = 5000)
