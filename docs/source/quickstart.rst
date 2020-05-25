Quickstart with pyrofit
=======================

Writing a simple pyrofit module
-------------------------------

A simple example for a pyrofit module looks like this::

    import pyro
    import pyro.distributions as dist
    import pyrofit.core as pf
    
    @pf.register
    def linear(a:Yaml, b:Yaml, x:Yaml):
        pyro.sample("y", dist.Normal(a + b*x, 1.0))


The corresponding YAML file looks like this::

    pyrofit:
      module: examples.minimal
      model: linear  # Name of relevant function
    conditioning:
      # Sample site names are (internally and for the purpose of conditioning)
      # prepended with function name
      linear/y: [5., 4., 3., 2., 1.]
    linear:  # Function name
      x: [1., 2., 3., 4., 5.]
      a:
        sample: [dist.Uniform, -10., 10.]
        init: 0.
      b:
        sample: [dist.Uniform, -10., 10.]
        init: 0.
    
    
Generating mock data
--------------------


Performing fit to mock data
---------------------------


Estimating posteriors
---------------------
