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
Implements functions that create basic meshes in cubit.
"""

# Python imports.
import numpy as np

# Import cubitpy stuff.
from . import cupy


def create_brick(
    cubit,
    h_x,
    h_y,
    h_z,
    *,
    element_type=None,
    mesh_interval=None,
    mesh_factor=None,
    mesh=True,
    **kwargs,
):
    """
    Create a cube in cubit.

    Args
    ----
    cubit: CubitPy
        CubitPy object.
    h_x, h_y, h_z: float
        size of the cube in x, y, and z direction.
    element_type: cubit.ElementType
        Type of the created element (HEX8, HEX20, HEX27, ...)
    mesh_interval: [int, int, int]
        Number of elements in each direction. This option is mutually
        exclusive with mesh_factor.
    mesh_interval: int
        Meshing factor in cubit. 10 is the largest. This option is mutually
        exclusive with mesh_factor.
    mesh: bool
        If the cube will be meshed or not.
    kwargs:
        Are passed to the Cubit.add_element_type function.
    """

    # Check if default value has to be set for element_type.
    if element_type is None:
        element_type = cupy.element_type.hex8

    # Check the input parameters.
    if h_x < 0 or h_y < 0 or h_z < 0:
        raise ValueError("Only positive lengths are possible!")
    if mesh_interval is not None and mesh_factor is not None:
        raise ValueError(
            "The keywords mesh_interval and mesh_factor are mutually exclusive!"
        )

    # Create the block in cubit.
    solid = cubit.brick(h_x, h_y, h_z)
    volume_id = solid.volumes()[0].id()

    # Set the element type.
    cubit.add_element_type(solid.volumes()[0], el_type=element_type, **kwargs)

    # Set mesh properties.
    if mesh_interval is not None:
        # Get the lines in x, y and z direction.
        dir_curves = [[] for _i in range(3)]
        for curve in solid.curves():
            # Get the tangent on the line.
            tan = curve.tangent([0, 0, 0])
            for direction in range(3):
                # Project the tangent on the basis vector and check if it is
                # larger than 0.
                if np.abs(tan[direction]) > cupy.eps_pos:
                    dir_curves[direction].append(curve)
                    continue

        # Set the number of elements in x, y and z direction.
        for direction in range(3):
            string = ""
            for curve in dir_curves[direction]:
                string += " {}".format(curve.id())
            cubit.cmd(
                "curve {} interval {} scheme equal".format(
                    string, mesh_interval[direction]
                )
            )

    if mesh_factor is not None:
        # Set a cubit factor for the mesh size.
        cubit.cmd("volume {} size auto factor {}".format(volume_id, mesh_factor))

    # Mesh the created block.
    if mesh:
        cubit.cmd("mesh volume {}".format(volume_id))

    return solid


def extrude_mesh_normal_to_surface(
    cubit,
    surfaces,
    thickness,
    n_layer=2,
    offset=[0, 0, 0],
    extrude_dir="outside",
    average_normals=False,
    tol_coord=1e-10,
    tol_normal=1e-10,
):
    """
    Extrude multiple meshed surfaces in normal direction of the surfaces.

    Args
    ----
    cubit: CubitPy
        Cubit object.
    surfaces: [CubitSurface]
        List of cubit surfaces that should be extruded. Each surface must be
        meshed.
    thickness: float
        Thickness of the extruded layer.
    n_layer: int
        Number of layers.
    offset: [x, y, z]
        Constant translational offset to be applied to the mesh.
    extrude_dir: 'outside', 'inside', 'symmetric'
        Direction of the extrusion.
    feature_angle: float
        Feature angle of the created volume.
    average_normals: bool
        Averages the different normals of the same coordinate evaluted at multiple surfaces.
        May lead to unexpected results.
    tol_coord: double
        Tolerance for the norm of the difference between node coordinates with the same ID
    tol_normal: double
        Tolerance for the norm of the difference between node normals evaluated at different surfaces
    ----
    return: [CubitVolume]
        Return a volume created from the combined elements created in this
        function.
    """

    # Calculate the offset depending on the extrude direction. The algorithm
    # always extrudes outside.
    if extrude_dir == "outside":
        extrude_offset = 0.0
    elif extrude_dir == "inside":
        extrude_offset = -thickness
    elif extrude_dir == "symmetric":
        extrude_offset = -0.5 * thickness
    else:
        raise ValueError("Got wrong extrude_type!")

    # Get a dictionary of all nodes on the surfaces, their positions and their
    # normals.
    quads = []
    node_id_pos_normal_map = {}
    for surface in surfaces:
        # Get all quad elements on the surface.
        surface_quads = cubit.get_surface_quads(surface.id())
        quads.extend(surface_quads)
        surface_nodes = []
        for quad in surface_quads:
            # Get all nodes on this face element.
            surface_nodes.extend(cubit.get_connectivity("quad", quad))
        # Remove double entries in node list.
        surface_nodes = list(set(surface_nodes))
        if len(surface_nodes) == 0:
            raise ValueError("Each surface must be meshed!")

        # Get normals and positions of the nodes.
        for node_id in surface_nodes:
            my_coordinates = np.array(cubit.get_nodal_coordinates(node_id))
            my_normal = np.array(surface.normal_at(my_coordinates))
            if node_id in node_id_pos_normal_map.keys():
                # Check that the normal and position are equal.
                other_coordinates = node_id_pos_normal_map[node_id][0]
                other_normal = node_id_pos_normal_map[node_id][1]

                # Check if coordinates match.
                if np.linalg.norm(my_coordinates - other_coordinates) < tol_coord:

                    # Check if normals do not match.
                    if (np.linalg.norm(my_normal - other_normal)) > tol_normal:

                        # Add normal for average calculation
                        if average_normals:
                            node_id_pos_normal_map[node_id][1] += my_normal
                        else:
                            raise ValueError(
                                f"Normals of node with ID {node_id} do not match!"
                            )
                else:
                    raise ValueError(
                        f"Coordinates of node with ID {node_id} do not match!"
                    )

            else:
                node_id_pos_normal_map[node_id] = [my_coordinates, my_normal]

    # Get a sorted list of the nodes on the surfaces.
    node_ids = list(node_id_pos_normal_map.keys())
    node_ids.sort()

    if average_normals:
        # Simply average all previously added normals.
        for value in node_id_pos_normal_map.values():
            value[1] *= 1.0 / np.linalg.norm(value[1])

    # Create the new nodal coordinates.
    n_nodes = len(node_ids)
    node_id_map = {}
    new_nodes = np.zeros([(n_layer + 1) * n_nodes, 3])
    for i_node, node_id in enumerate(node_ids):
        node_id_map[node_id] = i_node
        position = node_id_pos_normal_map[node_id][0]
        normal = node_id_pos_normal_map[node_id][1]
        for i_layer in range(n_layer + 1):
            new_nodes[n_nodes * i_layer + i_node, :] = (
                offset
                + position
                + normal * (extrude_offset + thickness * i_layer / n_layer)
            )
    mi = cubit.MeshImport()
    new_node_id = mi.add_nodes(
        3, (n_layer + 1) * n_nodes, new_nodes.reshape([(n_layer + 1) * n_nodes * 3])
    )
    if not new_node_id == 0:
        raise ValueError("Should not happen!")

    # Get hex topology.
    n_quads = len(quads)
    element_topology = np.zeros([n_quads * n_layer, 8])
    for i_quad, quad in enumerate(quads):
        quad_nodes = cubit.get_connectivity("quad", quad)
        quad_new_node_ids = [node_id_map[node] for node in quad_nodes]

        for i_layer in range(n_layer):
            element_topology[i_layer * n_quads + i_quad, :4] = np.add(
                quad_new_node_ids, i_layer * n_nodes + 1
            )
            element_topology[i_layer * n_quads + i_quad, 4:] = np.add(
                quad_new_node_ids, (i_layer + 1) * n_nodes + 1
            )

    # Create the elements.
    n_elements_old = cubit.get_hex_count()
    topology_list = list(map(int, element_topology.reshape(n_quads * n_layer * 8)))
    mi.add_elements(cubit.HEX, n_quads * n_layer, topology_list)

    # Create a volume from the created elements and return a reference to that
    # volume.
    ball_hex_ids = range(n_elements_old + 1, n_elements_old + n_quads * n_layer + 1)
    cubit.cmd(
        "create mesh geometry hex {} feature_angle 135.0".format(
            " ".join(map(str, ball_hex_ids))
        )
    )
    last_id = cubit.get_entities(cupy.geometry.volume)[-1]
    return cubit.volume(last_id)
