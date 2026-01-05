# CubitPy

Utility functions and 4C related functionality for the Cubit and Coreform python interface,
Especially for the creation of input files for 4C.

[![CubitPy test suite](https://github.com/imcs-compsim/cubitpy/actions/workflows/cubitpy_test_suite_with_secrets.yml/badge.svg)](https://github.com/imcs-compsim/cubitpy/actions/workflows/cubitpy_test_suite_with_secrets.yml)

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

CubitPy is developed with `python3.13`.
Other versions of Python might lead to issues.
It is recommended to use a python environment container such as `conda` or `venv`.
- `conda`:
  A [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment can be created and loaded with
  ```bash
  conda create -n cubitpy python=3.13
  conda activate cubitpy
  ```
- `venv`: Chose an appropriate directory for this, e.g., `/home/user/opt`.
  A virtual environment can be setup with
  - On Debian systems the following packages have to be installed:
    ```bash
    sudo apt-get install python3-venv python3-dev
    ```
  - Create and load the environment
    ```bash
    cd <path-to-env-folder>
    python -m venv cubitpy-env
    source cubitpy-env/bin/activate
    ```

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
pip install -e ".[dev]"
```

To run CubitPy it is required to set an environment variable with the path to the Cubit directory. This should be the "root" directory for the installation.
```bash
export CUBIT_ROOT=path_to_cubit_root_directory
```

If you are using a Cubit version that still runs on python2, you need to specify a path to a valid python2.7 executable
```bash
export CUBITPY_PYTHON2=path_to_python2.7
```
Note: for IMCS workstations this path is `"/imcs/public/compsim/opt/Python-2.7.18/python"`.

To check if everything worked as expected, run the test suite (from the root directory)
```bash
cd path_to_cubitpy/tests
pytest
```

If you intend to actively develop CubitPy, please make sure to install the `pre-commit` hook within the python environment to follow our style guides:
```bash
pre-commit install
```

## Debugging in VS Code and PyCharm

When debugging, IDEs may try to attach to subprocesses, which can cause issues. The instructions below show how to disable this behavior in VS Code and PyCharm.

### VS Code

To prevent the debugger from attaching to subprocesses, add the following to your `.vscode/launch.json`:

```jsonc
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Launch current file",
      "type": "debugpy",
      "request": "launch",
      "program": "${file}", // Launches the currently open file
      "console": "integratedTerminal", // or "externalTerminal" if needed
      "subProcess":false, // Disables debugging for subprocesses and libraries
    }
  ]
}
```

### PyCharm

To disable subprocess debugging:

1. Go to **File > Settings**
2. Navigate to:
   **Build, Execution, Deployment > Python Debugger**
3. Uncheck:
   **Attach to subprocess automatically while debugging**
