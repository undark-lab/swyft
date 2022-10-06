Installation
===============


Installation from github
------------------------

The latest development version of swyft can be installed directly from github.  To do so, run the command:

.. code-block:: bash

  pip install git+https://github.com/undark-lab/swyft.git@lightning

This command also installs required dependencies. We checked that it works well
on top of a fresh conda environment with python>=3.8.

Note: Right now only installation directly from github works for this version
(Swyft based on pytorch-lightning).  :code:`pip install swyft` will install the previous v0.3.x
version.

If in trouble, check out information about how to install `pytorch <https://pytorch.org/get-started/locally/>`_.



Development mode
----------------

If you're interested in contributing to swyft there is another procedure.
First clone the github repo, navigate to the repo in your terminal, from within that directory run the command:

.. code-block:: bash

  pip install git+https://github.com/undark-lab/swyft.git@lightning -e .[dev]

The :code:`-e` flag will install *swyft* in development mode such that your version of the code is used when *swyft* is imported.
The :code:`[dev]` flag installs the extra tools necessary to format and test your contribution.


Compiling documentation
-----------------------

Compiling the docs requires an additional flag. Then the docs may be compiled by navigating to the docs folder.

.. code-block:: bash

  pip install git+https://github.com/undark-lab/swyft.git@lightning -e .[docs]
  cd docs
  make html


.. _pytorch: https://pytorch.org/get-started/locally/
