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
"""Utility functions for the use of cubitpy."""

from cubitpy.conf import cupy
from cubitpy.cubit_group import CubitGroup
from cubitpy.cubit_wrapper.cubit_wrapper_host import CubitObject


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


def formatter(*args, geometry_type=None) -> str:
    """Formatter function for arguments passed to the cubit command line.

    This function can format arguments of type CubitObject, CubitGroup,
    int, or iterable (list, set, tuple) of these types. It ensures that
    all arguments have the same geometry type and returns a formatted
    string suitable for cubit commands.

    Args:
        *args: Variable length argument list containing CubitObject,
            CubitGroup, int, or iterable (list, set, tuple) of these types.
            If only explicit IDs are provided as integer arguments, the geometry
            type must be provided via the `geometry_type` argument.
        geometry_type: Optional; if provided, all arguments must match this
            geometry type.

    Returns:
        Formatted string for cubit command. Ordering of the items in the output
        string is the same as the ordering of the arguments passed to the function.
    """

    geometry_types = set()
    if geometry_type is not None:
        geometry_types.add(geometry_type)
    item_list = []
    for index, argument in enumerate(args):
        if isinstance(argument, (list, tuple, set)):
            if not len(args) == 1:
                raise ValueError(
                    f"The argument at position {index} is iterable, this only works "
                    f"if there is a single argument, got {len(args)} arguments."
                )
            return formatter(*argument, geometry_type=geometry_type)
        elif isinstance(argument, CubitGroup):
            geometry_type = argument.get_geometry_type()
            geometry_types.add(geometry_type)
            for item_id in argument.get_item_ids()[geometry_type]:
                item_list.append(item_id)

        elif isinstance(argument, CubitObject):
            # First check if the object is a body. In that case, we try to get the underlying
            # geometry (surface or volume).
            object_type = argument.get_object_type()
            if object_type == "body":
                if argument.is_sheet_body():
                    geometry_type = cupy.geometry.surface
                    objects = argument.surfaces()
                else:
                    geometry_type = cupy.geometry.volume
                    objects = argument.volumes()

                if not len(objects) == 1:
                    raise ValueError(
                        f"Expected exactly one {geometry_type} in the body, but got {len(objects)}"
                    )

                geometry_types.add(geometry_type)
                item_list.append(objects[0].id())
            elif isinstance(object_type, cupy.geometry):
                geometry_types.add(object_type)
                item_list.append(argument.id())
            else:
                raise TypeError(
                    f"Did not get a valid geometry from the CubitObject, got {object_type}"
                )
        else:
            if isinstance(argument, int):
                item_list.append(argument)
            else:
                raise TypeError(
                    f"Expected CubitObject, CubitGroup or int, but got {type(argument)}"
                )

    if len(geometry_types) == 0:
        raise ValueError(
            "No geometry types were found in the arguments, either check the "
            "arguments or explicitly provide a `geometry_type`."
        )
    elif not len(geometry_types) == 1:
        raise ValueError(
            f"All arguments must have the same geometry type, got {geometry_types}"
        )

    if len(item_list) == 0:
        raise ValueError("No item ids were found in the arguments.")

    geometry_type = geometry_types.pop()
    return "{} {}".format(
        geometry_type.get_cubit_string(), " ".join(map(str, item_list))
    )
