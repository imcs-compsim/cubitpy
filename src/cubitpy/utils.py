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


def get_geometry_type(item):
    """Return the type of this item."""

    # At the moment we need to import this here to avoid circular imports
    from cubitpy.cubit_group import CubitGroup

    if isinstance(item, CubitGroup):
        return item.get_geometry_type()
    else:
        if item.isinstance("cubitpy_vertex"):
            return cupy.geometry.vertex
        elif item.isinstance("cubitpy_curve"):
            return cupy.geometry.curve
        elif item.isinstance("cubitpy_surface"):
            return cupy.geometry.surface
        elif item.isinstance("cubitpy_volume"):
            return cupy.geometry.volume

    # Default value -> not a valid geometry
    raise TypeError("The item is not a valid geometry!")


def get_node_ids(cubit, item):
    """Return a list with the node IDs (index 1) of this object.

    This is done by creating a temporary node set that this geometry is
    added to. It is not possible to get the node list directly from
    cubit.
    """

    # Get a node set ID that is not yet taken
    node_set_ids = [0]
    node_set_ids.extend(cubit.get_nodeset_id_list())
    temp_node_set_id = max(node_set_ids) + 1

    # Add a temporary node set with this geometry
    cubit.cmd(
        "nodeset {} {} {}".format(
            temp_node_set_id, get_geometry_type(item).get_cubit_string(), item.id()
        )
    )

    # Get the nodes in the created node set
    node_ids = cubit.get_nodeset_nodes_inclusive(temp_node_set_id)

    # Delete the temp node set and return the node list
    cubit.cmd("delete nodeset {}".format(temp_node_set_id))
    return node_ids
