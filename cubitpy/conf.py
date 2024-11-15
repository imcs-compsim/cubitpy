# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# CubitPy: Utility functions and 4C related functionality for the Cubit and
#     Coreform python interface
#
# MIT License
#
# Copyright (c) 2018-2024
#     Ivo Steinbrecher
#     Institute for Mathematics and Computer-Based Simulation
#     Universitaet der Bundeswehr Muenchen
#     https://www.unibw.de/imcs-en
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------
"""
This module defines a global object that manages all kind of stuff regarding
cubitpy.
"""

# Python imports.
import os
import getpass
from sys import platform

# Cubitpy imports.
from .cubitpy_types import (
    FiniteElementObject,
    GeometryType,
    ElementType,
    CubitItems,
    BoundaryConditionType,
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
            "/tmp/cubitpy_{}".format(getpass.getuser()), "pid_{}".format(os.getpid())
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
        return get_path("CUBIT_ROOT", os.path.isdir, **kwargs)

    @classmethod
    def get_cubit_exe_path(cls, **kwargs):
        cubit_root = cls.get_cubit_root_path(**kwargs)
        if platform == "linux" or platform == "linux2":
            return os.path.join(cubit_root, "cubit")
        elif platform == "darwin":
            return os.path.join(cubit_root, "Contents/MacOS/Cubit")
        else:
            raise ValueError("Got unexpected platform")

    @classmethod
    def get_cubit_lib_path(cls, **kwargs):
        cubit_root = cls.get_cubit_root_path(**kwargs)
        if platform == "linux" or platform == "linux2":
            return os.path.join(cubit_root, "bin")
        elif platform == "darwin":
            return os.path.join(cubit_root, "Contents/MacOS")
        else:
            raise ValueError("Got unexpected platform")


# Global object with options for cubitpy.
cupy = CubitOptions()
