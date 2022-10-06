(this document has not been updated to swyft-lightning yet)

# Community Guidelines and Contributing
[![codecov](https://codecov.io/gh/undark-lab/swyft/branch/master/graph/badge.svg?token=E253LRJWWE)](https://codecov.io/gh/undark-lab/swyft)
[![Tests](https://github.com/undark-lab/swyft/actions/workflows/tests.yml/badge.svg)](https://github.com/undark-lab/swyft/actions)
[![Syntax](https://github.com/undark-lab/swyft/actions/workflows/syntax.yml/badge.svg)](https://github.com/undark-lab/swyft/actions)
[![Documentation Status](https://readthedocs.org/projects/swyft/badge/?version=latest)](https://swyft.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

If you'd like to contribute to *swyft* or report a bug, please take a moment to read the guide on contributing to [open source software](https://opensource.guide/how-to-contribute/).

If you are a *swyft* user, we welcome a report of your experience. Simple bugs and minor changes could be directly added as [github issues](https://github.com/undark-lab/swyft/issues). Larger issues regarding your particular simulator or computational environment may require [more discussion](https://github.com/undark-lab/swyft/discussions).

## Report issues, bugs, or problems with *swyft*

Please be patient as *swyft* is research software but it is being actively developed. Small issues or bugs directly related to *swyft* code justify an issue. If the problem is specific to your simulator, then we will need a lot more detail about the simulator and its setup.

Please reference the version of *swyft* you are using in any issues / bug reports. Check out [issues on GitHub](https://github.com/undark-lab/swyft/issues).

## Contribute to *swyft*

We'd be happy to include your contribution! We operate by introducing issues and solving them with a corresponding pull request to address the issue.

### Setup Envrionment

Unless your contribution is purely documentation based, you will need to setup a development version of *swyft*. Create the environment, including pre-commit hooks.

```bash
git clone https://github.com/undark-lab/swyft.git
cd swyft
pip install -e .[dev]
pre-commit install
```

The :code:`-e` flag will install *swyft* in development mode such that your version of the code is used when *swyft* is imported.
The :code:`[dev]` flag installs the extra tools necessary to format and test your contribution.
`pre-commit` will enforce [black](https://github.com/psf/black),
[isort](https://github.com/timothycrosley/isort),
and a few other code format rules before every commit.

### Testing
Any code that ends up in the master branch must be tested. It is simple to test the software by running the following command from the root directory:

```bash
pytest tests/
```

If you're introducing new functionality, we ask that you create tests so that our code coverage percentage may remain high. The best tests are comprehensive, using `pytest.mark.parameterize` to cover various cases where the functionality may be called. New tests should be fast computationally, i.e., training a neural network should not be part of the test suite.

### Code Standards

#### Docstrings
- Please use [Google Style](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstrings and comments.
- When we create a class, we put the docstring in the class rather than the `__init__` function, where appropriate.
- Use type annotations rather than putting the type in the docstring.
- When there is a default argument, do not repeat the default argument within the docstring since it is already visible when the user calls help on the function. The exception to this rule is when the default is `None`, then it may require an explanation which may include a default argument.

#### Types
Contributed code should have type hints. Some relevant types are defined within `swyft/types.py`, try to use them whenever possible.

#### Naming
- Integers which count the quantity of something should be proceded with an `n_*`, i.e. `n_parameters`.
- Although brevity is appreciated...
- Long names are generally more useful than unclear / single-use shortened versions. If you introduce a shortened version, please make sure it is consistent throughout your code and the existing codebase.
- When introducing a new variable, consider whether  it already has a name... see the table below.

For quick / common naming reference:

| Python Variable Name | Mathematical Object                           |
|----------------------|-----------------------------------------------|
| `v`                  | parameter vector in natural prior coordinates |
| `u`                  | parameter vector mapped to the hypercube      |
| `marginal_index`     | `tuple` of integers, often a key in a dict    |
| `marginal_indices`   | `tuple` of `marginal_index`                   |
| `observation`        | `dict` maps to simulated / observational data |
| `*_o`                | Either `v` or `observation` of interest       |

#### Converting between arrays and tensors
Please use the functions `array_to_tensor` and `tensor_to_array` when converting between arbitrary array data and pytorch tensors. This is to maintain consistency with default conversion types.

## Documentation

We have a [readthedocs](https://swyft.readthedocs.io/en/latest/) site and use the [sphinx_rtd_theme](https://github.com/readthedocs/sphinx_rtd_theme). The details of the configuration can be found in the [docs/source/conf.py](https://github.com/undark-lab/swyft/blob/master/docs/source/conf.py) file.

### Compiling documentation

To compile your own version of the documentation, run the following commands:

```bash
cd docs
make html
```

This will produce an html version of the documentation which can easily be viewed on a web browser by pointing to `swyft/docs/build/html/index.html`.
