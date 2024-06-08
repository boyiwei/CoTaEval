from setuptools import setup

setup(
    name='dataportraits',
    version='0.0.1',
    egg_base="_build/eggs",
    packages=['dataportraits'],
    package_dir={'':'src'},
    install_requires=[
        'redis',
        'numpy',
        'tqdm'
    ],
)
