from setuptools import setup

setup(
    name='swyft',
    description="",
    version='0.1',
    packages=['swyft'],
    install_requires=[
        'tqdm',
        'torch',
        'sklearn',
    ],
    zip_safe=False,
)
