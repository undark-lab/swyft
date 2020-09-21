Quickstart with SWYFT
=====================

General usage
-------------

SWYFT is simple to use.  First, we define a model function which takes an argument vector z and returns some data x.::

    def model(z):
        g = np.linspace(0, 2*np.pi, 10)
        x = np.sin(g)*z[0] + np.sin(g+0.2)*z[1]
        n = np.random
        return x + n

A simple example for a swyft module looks like this::

    import swyft


Generating mock data
--------------------


Performing fit to mock data
---------------------------


Estimating posteriors
---------------------
