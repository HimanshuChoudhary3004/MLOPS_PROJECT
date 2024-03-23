import io
import os
from pathlib import Path

from setuptools import setup, find_packages

# Metadata for the package

NAME = "Loan_prediction_model"
DESCRIPTION = "Loan Prediction"
URL = "https://github.com/HimanshuChoudhary3004/MLOPS_PROJECT.git"
EMAIL = "HCHOUDHARY525@gmail.com"
AUTHOR = "Himanshu Choudhary"
REQUIRES_PYTHON = '>=3.7.0'


#pwd = os.path.abspath(os.Path.dirname(__file__))
pwd = os.path.abspath(os.path.dirname(__file__))

# get the list of dependencies to be installed

def list_requires(fname='requirements.txt'):
    with io.open(os.path.join(pwd,fname) , encoding ='utf-8') as f:
        return f.read().splitlines()


try:
    with io.open(os.path.join(pwd,'README.md'), encoding ='utf-8') as f:
        long_description ='\n' + f.read()

except FileNotFoundError:
    long_description = DESCRIPTION


# load the package's __version.py file as module
# load the package's __version.py file as module
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / 'prediction_model'

about = {}

with open(PACKAGE_DIR / 'VERSION') as f:
    _version = f.read().strip()
    about['__version__'] = _version


setup(
    name=NAME,
    version=about['__version__'],
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    REQUIRES_PYTHON=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    package_data = {'prediction_model':['_VERSION']},
    install_requires=list_requires(),
    extras_require = {},
    include_package_data = True,
    license ='MIT',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python ::",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "License :: OSI Approved :: MIT License",
    ],
)



