## Tell us about your experience!

If you are a `swyft` user, please tell us about your use experience.
In particular, please provide details about your simulator, the setup that you used, and importantly the version of `swyft`.

Bug reports, feature requests, etc. are available on [issues on GitHub](https://github.com/mackelab/sbi/issues).

To report bugs and suggest features (including better documentation), please equally
head over to [issues on GitHub](https://github.com/undark-lab/swyft/issues).


## Would you like to contribute code?

Although our git history doesn't show it at this point, we try to use pull requests when we change `swyft`.
Ideally those pull requests are answering a git issue of some kind.

### Setup Envrionment

Create the envrionment, including pre-commit hooks.

```bash
git clone https://github.com/undark-lab/swyft.git
pip install -e .[dev]
pre-commit install
```

pre-commit will enforce **[black](https://github.com/psf/black)**,
**[isort](https://github.com/timothycrosley/isort)**,
and a few other code format rules before every commit.
It is a good idea to run `pytest` before making commits you intend to pull into the master branch.

Please use [Google Style](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstrings and comments.
We also highly recommend the use of the **[flake8](https://flake8.pycqa.org/en/latest/)** linter, although we do not have a CLI for it right now.

## Online documentation

We have a **[readthedocs](https://swyft.readthedocs.io/en/latest/)** site.
Please follow conventions in order to let that site compile.
