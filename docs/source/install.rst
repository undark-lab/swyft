Installation
===============


pip
--------
**After installing** `pytorch <https://pytorch.org/get-started/locally/>`_, please run the command:
::
  pip install swyft
::


github
---------
If you want the lastest version, it is also possible to install *swyft* from github.
To do so, run the command:
::
  pip install git+https://github.com/undark-lab/swyft.git
::

develop
---------
If you're interested in contributing to swyft there is another procedure. 
First clone the github repo, navigate to the repo in your terminal, from within that directory run the command:
::
  pip install -e .[dev]
::

The :code:`-e` flag will install *swyft* in development mode such that your version of the code is used when *swyft* is imported.
The :code:`[dev]` flag installs the extra tools necessary to format and test your contribution.

.. _pytorch: https://pytorch.org/get-started/locally/
