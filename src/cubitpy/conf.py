# The MIT License (MIT)
#
# Copyright (c) 2018-2025 CubitPy Authors
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
from sys import platform

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
    def get_cubit_root_path(**kwargs):
        """Get Path to cubit root directory."""
        return get_path("CUBIT_ROOT", os.path.isdir, **kwargs)

    @classmethod
    def get_cubit_exe_path(cls, **kwargs):
        """Get Path to cubit executable."""
        cubit_root = cls.get_cubit_root_path(**kwargs)
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
        cubit_root = cls.get_cubit_root_path(**kwargs)
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
        cubit_root = cls.get_cubit_root_path()
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
            if shutil.which("python2.7") is not None:
                return "python2.7"

            imcs_python2_path = "/imcs/public/compsim/opt/Python-2.7.18/python"
            if os.path.exists(imcs_python2_path):
                return imcs_python2_path

            raise ValueError("Could not find a python2 interpreter")

    @classmethod
    def is_coreform(cls):
        """Return if the given path is a path to cubit coreform."""
        cubit_root = cls.get_cubit_root_path()
        if "15.2" in cubit_root:
            return False
        else:
            return True


# Global object with options for cubitpy.
cupy = CubitOptions()
