## Tell us about your experience!

If you are a `swyft` user, please tell us about your use experience.
In particular, please provide details about your simulator, the setup that you used, and importantly the version of `swyft`.

Bug reports, feature requests, etc. are available on [issues on GitHub](https://github.com/undark-lab/swyft/issues).

## Would you like to contribute code?

We try to use pull requests when we change `swyft`.
Ideally those pull requests are answering a git issue of some kind.
Do you have a change you'd like to see made? We recommend first making an issue then addressing it with a pull request.

### Setup Envrionment

Create the environment, including pre-commit hooks.

```bash
git clone https://github.com/undark-lab/swyft.git
pip install -e .[dev]
pre-commit install
```

pre-commit will enforce **[black](https://github.com/psf/black)**,
**[isort](https://github.com/timothycrosley/isort)**,
and a few other code format rules before every commit.
It is a good idea to run `pytest` before making commits you intend to pull into the master branch.

### Linting
We also highly recommend the use of the **[flake8](https://flake8.pycqa.org/en/latest/)** linter, although we do not have a CLI for it right now.

## Standards

### Docstrings
Please use [Google Style](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstrings and comments.  
When we create a class, we put the docstring in the class rather than the `__init__` function, where appropriate.  
Use type annotations rather than putting the type in the docstring.  
When there is a default argument, do not write that default in the argument documentation line. The exception is when the default is `None`, then it needs an explanation.  

### Naming
Consider the naming conventions when you introduce or modify functions. The relevant types are defined within `swyft/types.py`.  
For quick reference, use `pnames` for parameters names; `v` for the actual parameter vectors; `u` for the hypercube projection; `PoI` for "parameters of interest" in tuple form, e.g. `(0,)` or `(0, 1)`; `marginals` for a dictionary mapping of `PoI` to another value, e.g. weights.

### Converting between arrays and tensors
Please use the functions `array_to_tensor` and `tensor_to_array` when converting between arbitrary array data and pytorch tensors.


## Online documentation

We have a **[readthedocs](https://swyft.readthedocs.io/en/latest/)** site.
Please follow conventions in order to let that site compile.
