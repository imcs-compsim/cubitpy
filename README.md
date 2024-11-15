[![build-test](https://github.com/imcs-compsim/cubitpy/actions/workflows/.github/workflows/build-test.yml/badge.svg)](https://github.com/imcs-compsim/cubitpy/actions/workflows/.github/workflows/build-test.yml)

# CubitPy

Utility functions and 4C related functionality for the Cubit and Coreform python interface,
Especially for the creation of input files for 4C.

## Usage

A tutorial can be found in the `/tutorial` directory.

## Contributing

If you are interested in contributing to CubitPy, we welcome your collaboration.
For general questions, feature request and bug reports please open an [issue](https://github.com/imcs-compsim/cubitpy/issues).

If you contribute actual code, fork the repository and make the changes in a feature branch.
Depending on the topic and amount of changes you also might want to open an [issue](https://github.com/imcs-compsim/cubitpy/issues).
To merge your changes into the CubitPy repository, create a pull request to the `main` branch.
A few things to keep in mind:
- It is highly encouraged to add tests covering the functionality of your changes, see the test suite in `tests/`.
- CubitPy uses `black` to format python code.
  Make sure to apply `black` to the changed source files.
- Feel free to add yourself to the [CONTRIBUTORS](CONTRIBUTORS) file.

## Installation

CubitPy is developed with `python3.12`.
Other versions of Python might lead to issues.
It is recommended to use virtual environments with `python`.
On Debian systems the following packages have to be installed
```bash
sudo apt-get install python3-venv python3-dev
```

Now a virtual environment can be created (chose an appropriate directory for this, e.g., `/home/user/opt`)
```bash
python3 -m venv cubitpy-env
```

The created virtual environment can be loaded with
```bash
source cubitpy-env/bin/activate
```

From now on we assume that the virtual environment is loaded.
To install `cubitpy` go to the repository root directory
```bash
cd path_to_cubitpy
```

And install `cubitpy` via `pip`
```bash
pip install .
```

If you intend to actively develop `cubitpy`, install it in *editable mode*
```bash
pip install -e .
```

To run CubitPy it is required to set an environment variable with the path to the Cubit directory. This should be the "root" directory for the installation.
```bash
export CUBIT_ROOT=path_to_cubit_root_directory
```

To check if everything worked as expected, run the tests from within the `tests` directory
```bash
cd path_to_cubitpy/tests
pytest -q testing.py
```

If you intend to actively develop CubitPy, please make sure to install the `pre-commit` hook within the python environment to follow our style guides:
```bash
pre-commit install
```
