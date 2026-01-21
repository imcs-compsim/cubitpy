# The MIT License (MIT)
#
# Copyright (c) 2018-2026 CubitPy Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""This module defines a global object that manages all kind of stuff regarding
cubitpy."""

import getpass
import glob
import os
import shutil
import warnings
from pathlib import Path
from sys import platform

import yaml

from cubitpy.cubitpy_types import (
    BoundaryConditionType,
    CubitItems,
    ElementType,
    FiniteElementObject,
    GeometryType,
)


def get_path(environment_variable, test_function, *, throw_error=True):
    """Check if he environment variable is set and the path exits."""
    if environment_variable in os.environ.keys():
        if test_function(os.environ[environment_variable]):
            return os.environ[environment_variable]

    # No valid path found or given.
    if throw_error:
        raise ValueError("Path for {} not found!".format(environment_variable))
    else:
        return None


class CubitOptions(object):
    """Object for types in cubitpy."""

    _config = None

    def __init__(self):
        # Temporary directory for cubitpy.
        self.temp_dir = os.path.join(
            "/tmp/cubitpy_{}".format(getpass.getuser()),  # nosec
            "pid_{}".format(os.getpid()),
        )
        self.temp_log = os.path.join(self.temp_dir, "cubitpy.log")

        # Check if temp path exits, if not create it.
        os.makedirs(self.temp_dir, exist_ok=True)

        # Geometry types.
        self.geometry = GeometryType

        # Finite element types.
        self.finite_element_object = FiniteElementObject

        # Element shape types.
        self.element_type = ElementType

        # Cubit internal items.
        self.cubit_items = CubitItems

        # Boundary condition type.
        self.bc_type = BoundaryConditionType

        # Tolerance for geometry.
        self.eps_pos = 1e-10

    @staticmethod
    def get_cubit_config_filepath():
        """Return path to remote config if it exists, else None."""
        return get_path("CUBITPY_CONFIG_PATH", os.path.isfile, throw_error=False)

    @classmethod
    def validate_cubit_config(cls):
        """Validate the already loaded config dict and raise helpful errors."""

        config = cls._config
        if config is None:
            raise RuntimeError(
                "Config not loaded yet. Call cupy.get_cubit_config(...) first."
            )

        TEMPLATE = (
            "\n\nCorrect YAML structure:\n"
            "----------------------------------------\n"
            'cubitpy_mode: "remote"  # or "local"\n'
            "\n"
            "remote_config:\n"
            '  user: "<username>"\n'
            '  host: "<hostname_or_ip>"\n'
            '  cubit_path: "<remote_cubit_install_path>"\n'
            "\n"
            "local_config:\n"
            '  cubit_path: "<local_cubit_install_path>"\n'
            "----------------------------------------\n"
            "- If mode = 'remote': remote_config MUST exist and contain user, host, cubit_path.\n"
            "- If mode = 'local' : local_config MUST exist and contain cubit_path.\n"
            "- The unused section may be omitted.\n"
            "----------------------------------------\n"
        )

        def fail(msg: str):
            """Helper to raise a RuntimeError with template."""
            raise RuntimeError(msg + TEMPLATE)

        # Check mode
        if "cubitpy_mode" not in config:
            fail("Missing required key: 'cubitpy_mode'.")

        mode = config["cubitpy_mode"]
        if mode not in ("remote", "local"):
            fail(f"Invalid cubitpy_mode '{mode}'. Expected 'remote' or 'local'.")

        if mode == "remote":
            if "remote_config" not in config:
                fail("cubitpy_mode='remote' requires a 'remote_config' section.")

            remote_config = config["remote_config"]
            required = ["user", "host", "cubit_path"]
            missing = [
                k for k in required if k not in remote_config or not remote_config[k]
            ]
            if missing:
                fail("remote_config is missing required fields: " + ", ".join(missing))

        if mode == "local":
            if "local_config" not in config:
                fail("cubitpy_mode='local' requires a 'local_config' section.")

            local_config = config["local_config"]
            if "cubit_path" not in local_config or not local_config["cubit_path"]:
                fail("local_config must contain a non-empty 'cubit_path'.")

            local_cubit_path = local_config["cubit_path"]
            if not Path(local_cubit_path).expanduser().exists():
                raise FileNotFoundError(
                    f"local_config.cubit_path '{local_cubit_path}' does not exist."
                )

    @classmethod
    def load_cubit_config(cls, config_path: Path | None = None):
        """Read the CubitPy YAML config."""

        if config_path is None:
            config_path = cls.get_cubit_config_filepath()

        if not config_path:
            warnings.warn(
                "CubitPy configuration file not found.Using default config: local",
                DeprecationWarning,
            )
            root_path = get_path("CUBIT_ROOT", os.path.isdir, throw_error=True)

            default_config = {
                "cubitpy_mode": "local",
                "local_config": {"cubit_path": root_path},
                "remote_config": {},
            }

            cubit_config_dict = default_config
        else:
            try:
                with open(config_path, "r") as f:
                    cubit_config_dict = yaml.safe_load(f)
            except Exception as e:
                raise ImportError(f"Failed to read YAML at '{config_path}': {e}")

            if not isinstance(cubit_config_dict, dict):
                raise ImportError("YAML top level must be a mapping (dict).")

        cls._config = cubit_config_dict
        cls.validate_cubit_config()

    @classmethod
    def get_cubit_exe_path(cls, **kwargs):
        """Get Path to cubit executable."""
        cubit_root = cls._config["local_config"]["cubit_path"]
        if platform == "linux" or platform == "linux2":
            if cupy.is_coreform():
                return os.path.join(cubit_root, "bin", "coreform_cubit")
            else:
                return os.path.join(cubit_root, "cubit")
        elif platform == "darwin":
            if cupy.is_coreform():
                cubit_exe_name = cubit_root.split("/")[-1].split(".app")[0]
                return os.path.join(cubit_root, "Contents/MacOS", cubit_exe_name)
            else:
                return os.path.join(cubit_root, "Contents/MacOS/Cubit")
        else:
            raise ValueError("Got unexpected platform")

    @classmethod
    def get_cubit_lib_path(cls, **kwargs):
        """Get Path to cubit lib directory."""
        cubit_root = cls._config["local_config"]["cubit_path"]
        if platform == "linux" or platform == "linux2":
            return os.path.join(cubit_root, "bin")
        elif platform == "darwin":
            if cls.is_coreform():
                return os.path.join(cubit_root, "Contents/lib")
            else:
                return os.path.join(cubit_root, "Contents/MacOS")
        else:
            raise ValueError("Got unexpected platform")

    @classmethod
    def get_cubit_interpreter(cls):
        """Get the path to the python interpreter to be used for CubitPy."""
        cubit_root = cls._config["local_config"]["cubit_path"]
        if cls.is_coreform():
            pattern = "**/python3"
            full_pattern = os.path.join(cubit_root, pattern)
            python3_matches = glob.glob(full_pattern, recursive=True)
            python3_files = [path for path in python3_matches if os.path.isfile(path)]
            if not len(python3_files) == 1:
                raise ValueError(
                    "Could not find the path to the cubit python interpreter"
                )
            cubit_python_interpreter = python3_files[0]
            return cubit_python_interpreter
        else:
            python2_path_env = get_path(
                "CUBITPY_PYTHON2", os.path.isfile, throw_error=False
            )
            if python2_path_env is not None:
                return python2_path_env

            if shutil.which("python2.7") is not None:
                return "python2.7"

            raise ValueError(
                "Could not find a python2 interpreter. "
                "You can specify this by setting the environment variable "
                "CUBITPY_PYTHON2 to the path of your python2 interpreter."
            )

    @classmethod
    def is_coreform(cls):
        """Return if the given path is a path to cubit coreform."""
        cubit_root = cls._config["local_config"]["cubit_path"]
        if "15.2" in cubit_root and not cls.is_remote():
            return False
        else:
            return True

    @classmethod
    def is_remote(cls) -> bool:
        """Return True if cubit is running remotely based on the loaded
        config."""
        if cls._config is None:
            raise RuntimeError(
                "Config not loaded yet. Call load_cubit_config() first use of is_remote."
            )
        return cls._config.get("cubitpy_mode") == "remote"


# Global object with options for cubitpy.
cupy = CubitOptions()
