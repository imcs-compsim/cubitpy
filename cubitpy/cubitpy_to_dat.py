# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# CubitPy: Cubit utility functions and a cubit wrapper for python3
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
Implements a function that converts a cubit session to a dat file that can be used
with 4C.
"""


# Python modules.
import os
import numpy as np
import netCDF4

# Cubitpy modules.
from .conf import cupy


def add_node_sets(dat_lines, cubit, exo):
    """Add the node sets contained in the cubit session/exo file to the dat_lines"""

    # If there are no node sets we can return immediately
    if len(cubit.node_sets) == 0:
        return

    # Get names of the node sets
    names = []
    for string_list in exo.variables["ns_names"]:
        string = ""
        for char in string_list:
            if isinstance(char, np.bytes_):
                string += char.decode("UTF-8")
        names.append(string)

    # Sort the sets into their geometry type
    node_sets = {
        cupy.geometry.vertex: [],
        cupy.geometry.curve: [],
        cupy.geometry.surface: [],
        cupy.geometry.volume: [],
    }
    boundary_condition_map = {}
    node_set_keys = [key for key in exo.variables.keys() if "node_ns" in key]
    for i_set, key in enumerate(node_set_keys):
        bc_section, bc_description, geometry_type = cubit.node_sets[i_set]
        node_sets[geometry_type].append(exo.variables[key][:])
        bc_key = (bc_section, geometry_type)
        if bc_key not in boundary_condition_map.keys():
            boundary_condition_map[bc_key] = []
        boundary_condition_map[bc_key].append(
            [len(node_sets[geometry_type]), bc_description, names[i_set]]
        )

    # Write the boundary conditions
    for (bc_section, geo), item in boundary_condition_map.items():
        dat_lines.append("-" * 40 + bc_section)
        if geo is cupy.geometry.vertex:
            geo_string = "DPOINT"
        elif geo is cupy.geometry.curve:
            geo_string = "DLINE"
        elif geo is cupy.geometry.surface:
            geo_string = "DSURF"
        elif geo is cupy.geometry.volume:
            geo_string = "DVOL"
        dat_lines.append(f"{geo_string} {len(item)}")
        for set_id, bc_description, name in item:
            if not name == "":
                dat_lines.append(f"// {name}")
            dat_lines.append(f"E {set_id} - {bc_description}")

    name_geometry_tuple = [
        [
            cupy.geometry.vertex,
            "-----------------------------------------------DNODE-NODE TOPOLOGY",
            "DNODE",
        ],
        [
            cupy.geometry.curve,
            "-----------------------------------------------DLINE-NODE TOPOLOGY",
            "DLINE",
        ],
        [
            cupy.geometry.surface,
            "-----------------------------------------------DSURF-NODE TOPOLOGY",
            "DSURFACE",
        ],
        [
            cupy.geometry.volume,
            "-----------------------------------------------DVOL-NODE TOPOLOGY",
            "DVOL",
        ],
    ]
    for geo, section_name, set_label in name_geometry_tuple:
        dat_lines.append(section_name)
        for i_set, node_set in enumerate(node_sets[geo]):
            node_set.sort()
            for i_node in node_set:
                dat_lines.append(f"NODE {i_node:6d} {set_label} {i_set+1}")


def get_element_connectivity_string(connectivity):
    """Return the connectivity string for an element

    For hex27 we need a different ordering than the one we get from cubit"""

    if len(connectivity) == 27:
        # hex27
        ordering = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            21,
            25,
            24,
            26,
            23,
            22,
            20,
        ]
        return " ".join([f"{item:d}" for item in connectivity[ordering]])
    else:
        # all other elements
        return " ".join([f"{item:d}" for item in connectivity])


def cubit_to_dat(cubit):
    """Convert a CubitPy session to a dat file that can be read with 4C"""

    # Create exodus file
    os.makedirs(cupy.temp_dir, exist_ok=True)
    exo_path = os.path.join(cupy.temp_dir, "cubitpy.exo")
    cubit.export_exo(exo_path)
    exo = netCDF4.Dataset(exo_path)

    dat_lines = []

    # Add the header
    for line in cubit.head.split("\n"):
        dat_lines.append(line.strip())

    # Add the node sets
    add_node_sets(dat_lines, cubit, exo)

    # Add the nodal data
    dat_lines.append(
        "-------------------------------------------------------NODE COORDS"
    )
    if "coordz" in exo.variables:
        coordinates = np.array(
            [exo.variables["coord" + dim][:] for dim in ["x", "y", "z"]],
        ).transpose()
    else:
        temp = [exo.variables["coord" + dim][:] for dim in ["x", "y"]]
        temp.append([0 for i in range(len(temp[0]))])
        coordinates = np.array(temp).transpose()
    for i, coordinate in enumerate(coordinates):
        dat_lines.append(
            f"NODE {i+1:9d} COORD {coordinate[0]: .16e} {coordinate[1]: .16e} {coordinate[2]: .16e}"
        )

    # Add the element connectivity
    current_section = None
    connectivity_keys = [key for key in exo.variables.keys() if "connect" in key]
    connectivity_keys.sort()
    i_element = 0
    for i_block, key in enumerate(connectivity_keys):
        ele_type, block_string = cubit.blocks[i_block]
        block_section = ele_type.get_four_c_section()
        if not block_section == current_section:
            current_section = block_section
            dat_lines.append(
                f"------------------------------------------------{current_section} ELEMENTS"
            )
        for connectivity in exo.variables[key][:]:
            connectivity_string = get_element_connectivity_string(connectivity)
            dat_lines.append(
                f"{i_element+1:9d} {ele_type.get_four_c_name()} {ele_type.get_four_c_type()} {connectivity_string} {block_string}"
            )
            i_element += 1

    return dat_lines
