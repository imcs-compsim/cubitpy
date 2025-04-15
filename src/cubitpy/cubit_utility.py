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
"""Utility functions for the use of cubitpy."""

from cubitpy.conf import cupy


def get_surface_center(surf):
    """Get a 3D point that has the local coordinated on the surface of (0,0),
    with the parameter space being ([-1,1],[-1,1])."""

    if not surf.get_geometry_type() == cupy.geometry.surface:
        raise TypeError("Did not expect {}".format(type(surf)))

    range_u = surf.get_param_range_U()
    u = 0.5 * (range_u[1] + range_u[0])
    range_v = surf.get_param_range_V()
    v = 0.5 * (range_v[1] + range_v[0])
    return surf.position_from_u_v(u, v)


def import_fluent_geometry(cubit, file, feature_angle=135):
    """Import fluent mesh geometry in cubit from file with according
    feature_angle."""

    cubit.cmd(
        'import fluent mesh geometry  "{}" feature_angle {} '.format(
            file, feature_angle
        )
    )
