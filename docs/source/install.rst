Installation
============


Installation from github
------------------------

The latest stable version of swyft can be installed via pip.  To do so, run the command:

.. code-block:: bash

  pip install swyft

This command also installs required dependencies.

The development version can be installed using

.. code-block:: bash

  pip install git+https://github.com/undark-lab/swyft.git@dev

If in trouble, check out information about how to install `pytorch <https://pytorch.org/get-started/locally/>`_.



Development mode
----------------

If you're interested in contributing to swyft there is another procedure.
First clone the github repo, navigate to the repo in your terminal, from within that directory run the command:

.. code-block:: bash

  pip install git+https://github.com/undark-lab/swyft.git@dev -e .[dev]

The :code:`-e` flag will install *swyft* in development mode such that your version of the code is used when *swyft* is imported.
The :code:`[dev]` flag installs the extra tools necessary to format and test your contribution.


Compiling documentation
-----------------------

Compiling the docs (which you find on swyft.readthedocs.io) requires an
additional flag. Then the docs may be compiled by navigating to the docs
folder.

.. code-block:: bash

  pip install git+https://github.com/undark-lab/swyft.git -e .[docs]
  cd docs
  make html
