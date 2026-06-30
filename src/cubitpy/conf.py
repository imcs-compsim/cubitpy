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
import sys
from pathlib import Path, PureWindowsPath
from typing import Any, Dict

import yaml

from cubitpy.cubitpy_types import (
    BoundaryConditionType,
    CubitItems,
    ElementType,
    FiniteElementObject,
    GeometryType,
)


class CubitPyWarning(UserWarning):
    """Warning emitted by CubitPy."""


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

    def __init__(self):
        # This dict holds the configuration by the user.
        self._config: Dict[str, Any] | None = None

        # Try to load the default config.
        self.load_cubit_config(allow_missing=True)

        # Define the host platform.
        self._host_platform = sys.platform
        if self._host_platform.startswith("linux"):
            self._host_platform = "linux"
        elif self._host_platform.startswith("darwin"):
            self._host_platform = "macos"
        elif self._host_platform.startswith("win32"):
            self._host_platform = "windows"
        else:
            raise ValueError(
                "Got unexpected host platform: {}".format(self._host_platform)
            )

        # Temporary directory for cubitpy.
        self.temp_dir = os.path.join(
            "/tmp/cubitpy_{}".format(getpass.getuser()),  # nosec
            "pid_{}".format(os.getpid()),
        )

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

    @property
    def on_cubit_error(self) -> str:
        """How to react when a cubit command reports an error.

        One of "raise" (default, raise a RuntimeError), "warn" (emit a
        CubitPyWarning and continue) or "ignore" (silently continue),
        set via the optional top-level 'on_cubit_error' key in the
        config file.
        """
        return self._get_config().get("on_cubit_error", "raise")

    def _get_config(self) -> Dict[str, Any]:
        """Get the config dict or raise an error if not loaded."""
        if self._config is None:
            raise RuntimeError(
                "Config not loaded yet. Call CubitOptions.load_cubit_config(...) first."
            )
        return self._config

    def validate_cubit_config(self) -> None:
        """Validate the already loaded config dict and raise helpful errors."""

        config = self._get_config()

        TEMPLATE = (
            "\n\nCorrect YAML structure:\n"
            "----------------------------------------\n"
            'cubitpy_mode: "remote"  # or "local"\n'
            "\n"
            "remote_config:\n"
            '  user: "<username>"\n'
            '  host: "<hostname_or_ip>"\n'
            '  platform: "<remote_platform>"  # "linux" or "windows"\n'
            '  cubit_path: "<remote_cubit_install_path>"\n'
            "\n"
            "local_config:\n"
            '  cubit_path: "<local_cubit_install_path>"\n'
            "\n"
            'on_cubit_error: "raise"  # optional: "raise" (default), "warn" or "ignore"\n'
            "----------------------------------------\n"
            "- If mode = 'remote': remote_config MUST exist and contain user, host, platform, cubit_path.\n"
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

        if config.get("on_cubit_error", "raise") not in ("raise", "warn", "ignore"):
            fail(
                f"Invalid on_cubit_error '{config['on_cubit_error']}'. "
                "Expected 'raise', 'warn' or 'ignore'."
            )

        if mode == "remote":
            if "remote_config" not in config:
                fail("cubitpy_mode='remote' requires a 'remote_config' section.")

            remote_config = config["remote_config"]
            required = ["user", "host", "platform", "cubit_path"]
            missing = [
                k for k in required if k not in remote_config or not remote_config[k]
            ]
            if missing:
                fail("remote_config is missing required fields: " + ", ".join(missing))

            user = remote_config["user"]
            host = remote_config["host"]
            if not isinstance(user, str) or not isinstance(host, str):
                fail("remote_config 'user' and 'host' must be strings.")
            if any(c.isspace() or c in "@:/\\" for c in user + host):
                fail(
                    "remote_config 'user' and 'host' must not contain whitespace "
                    "or any of '@ : / \\'."
                )
            platform = remote_config["platform"]
            if platform not in ("windows", "linux"):
                fail("remote_config 'platform' must be either 'windows' or 'linux'.")

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

    def load_cubit_config(
        self, config_path: Path | None = None, allow_missing: bool = False
    ) -> None:
        """Read the CubitPy YAML config."""

        if config_path is None:
            config_path = get_path(
                "CUBITPY_CONFIG_PATH", os.path.isfile, throw_error=False
            )

        if config_path is None:
            if allow_missing:
                self._config = None
                return
            raise ValueError("CubitPy configuration file could not be loaded.")

        try:
            with open(config_path, "r") as f:
                cubit_config_dict = yaml.safe_load(f)
        except Exception as e:
            raise ImportError(f"Failed to read YAML at '{config_path}': {e}")

        if not isinstance(cubit_config_dict, dict):
            raise ImportError("YAML top level must be a mapping (dict).")

        self._config = cubit_config_dict
        self.validate_cubit_config()

    def get_cubit_exe_path(self) -> Path | PureWindowsPath:
        """Get Path to cubit executable."""
        config = self._get_config()
        cubit_path: Path | PureWindowsPath
        if self.is_remote():
            cubit_platform = config["remote_config"]["platform"]
            cubit_path = Path(config["remote_config"]["cubit_path"])
        else:
            cubit_platform = self._host_platform
            cubit_path = Path(config["local_config"]["cubit_path"])

        if cubit_platform == "linux":
            if cupy.is_coreform():
                return cubit_path / "bin" / "coreform_cubit"
            else:
                return cubit_path / "cubit"
        elif cubit_platform == "macos":
            if cupy.is_coreform():
                cubit_exe_name = cubit_path.stem
                return cubit_path / "Contents" / "MacOS" / cubit_exe_name
            else:
                return cubit_path / "Contents" / "MacOS" / "Cubit"
        elif cubit_platform == "windows":
            cubit_path = PureWindowsPath(cubit_path)
            return cubit_path / "bin" / "coreform_cubit.exe"
        else:
            raise ValueError("Got unexpected platform")

    def get_cubit_lib_path(self) -> Path | PureWindowsPath:
        """Get Path to cubit lib directory."""
        config = self._get_config()
        cubit_path: Path | PureWindowsPath
        if self.is_remote():
            cubit_platform = config["remote_config"]["platform"]
            cubit_path = Path(config["remote_config"]["cubit_path"])
        else:
            cubit_platform = self._host_platform
            cubit_path = Path(config["local_config"]["cubit_path"])

        if cubit_platform == "linux":
            return cubit_path / "bin"
        elif cubit_platform == "macos":
            if self.is_coreform():
                return cubit_path / "Contents" / "lib"
            else:
                return cubit_path / "Contents" / "MacOS"
        elif cubit_platform == "windows":
            cubit_path = PureWindowsPath(cubit_path)
            return cubit_path / "bin"
        else:
            raise ValueError("Got unexpected platform")

    def get_cubit_python_interpreter(self) -> str:
        """Get the path to the python interpreter to be used for CubitPy."""
        config = self._get_config()
        cubit_path: Path | PureWindowsPath
        if self.is_remote():
            cubit_platform = config["remote_config"]["platform"]
            cubit_path = Path(config["remote_config"]["cubit_path"])
        else:
            cubit_platform = self._host_platform
            cubit_path = Path(config["local_config"]["cubit_path"])

        if cubit_platform == "linux" or cubit_platform == "macos":
            if self.is_coreform():
                pattern = "**/python3"
                full_pattern = os.path.join(cubit_path, pattern)
                python3_matches = glob.glob(full_pattern, recursive=True)
                python3_files = [
                    path for path in python3_matches if os.path.isfile(path)
                ]
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

        elif cubit_platform == "windows":
            cubit_path = PureWindowsPath(cubit_path)
            return str(cubit_path / "bin" / "python3" / "python.exe")
        else:
            raise ValueError("Got unexpected platform")

    def is_coreform(self) -> bool:
        """Return if the given path is a path to cubit coreform."""
        config = self._get_config()
        if self.is_remote():
            cubit_path = Path(config["remote_config"]["cubit_path"])
        else:
            cubit_path = Path(config["local_config"]["cubit_path"])
        if "15.2" in str(cubit_path):
            return False
        else:
            return True

    def is_remote(self) -> bool:
        """Return True if cubitpy is running in remote mode."""
        return self._get_config()["cubitpy_mode"] == "remote"

    def _require_remote(self) -> Dict[str, Any]:
        """Return the remote_config section of the config file."""
        config = self._get_config()
        if config.get("cubitpy_mode") != "remote":
            raise RuntimeError(
                "Remote config required but cubitpy_mode is not 'remote'."
            )
        return config["remote_config"]

    def get_remote_user(self) -> str:
        """Return the remote user from config."""
        return self._require_remote()["user"]

    def get_remote_host(self) -> str:
        """Return the remote host from config."""
        return self._require_remote()["host"]

    def get_remote_platform(self) -> str:
        """Return the remote platform from config."""
        return self._require_remote()["platform"]


# Global object with options for cubitpy.
cupy = CubitOptions()
