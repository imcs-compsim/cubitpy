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
"""Implements a function that converts a cubit session to a dat file that can
be used with 4C."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import netCDF4
import numpy as np
from fourcipp.fourc_input import FourCInput

from cubitpy.conf import GeometryType, cupy

if TYPE_CHECKING:
    from cubitpy.cubitpy import CubitPy


def get_exo_info(exo, entry_type) -> tuple[dict, dict]:
    """Build mappings between Exodus IDs and Cubit IDs for blocks or
    nodesets."""

    if entry_type == "block":
        exo_identifier = "eb"
    elif entry_type == "nodeset":
        exo_identifier = "ns"
    else:
        raise ValueError(f"Invalid entry type: {entry_type}")

    # List of explicitly given names
    names = []
    for line in exo.variables[exo_identifier + "_names"]:
        name: str | None = ""
        for char in line:
            if isinstance(char, np.bytes_):
                name += char.decode("UTF-8")
        if name == "":
            name = None
        names.append(name)

    # Get information of all entries of the given type
    cubit_id_to_info = {}
    exo_id_to_info = {}
    for exo_id, cubit_id in enumerate(
        exo.variables[exo_identifier + "_prop1"][:].tolist()
    ):
        info = {"cubit_id": cubit_id, "exo_id": exo_id, "name": names[exo_id]}
        cubit_id_to_info[cubit_id] = info.copy()
        exo_id_to_info[exo_id] = info.copy()

    return cubit_id_to_info, exo_id_to_info


def add_node_sets_external_geometry(cubit: CubitPy, input_file: FourCInput) -> None:
    """Add a reference to the node sets contained in the cubit session/exo file
    to the yaml file."""

    # If there are no node sets we can return immediately
    if len(cubit.node_sets) == 0:
        return

    # To align with the ordering for mesh data in the input file, we sort the
    # node sets according to their node set id.
    node_set_keys_sorted = sorted(cubit.node_sets.keys())

    # Write the node set information to the input file.
    for node_set_id in node_set_keys_sorted:
        node_set_info = cubit.node_sets[node_set_id]
        # Only add the boundary condition to the input file if a bc_section is
        # given - we can also add node sets without a boundary condition.
        bc_section = node_set_info.bc_section
        if bc_section is not None:
            # We modify the bc_description, thus we create a copy to avoid
            # modifying the original CubitPy data.
            bc_description = node_set_info.bc_description.copy()
            bc_description["E"] = node_set_id

            if bc_section not in input_file.inlined.keys():
                input_file[bc_section] = []

            # when working with external .exo meshes, we simply have to specify that
            # the id of the node set is in the exo file.
            bc_description["ENTITY_TYPE"] = "node_set_id"

            input_file[bc_section].append(bc_description)


def add_node_sets_input_file(cubit: CubitPy, exo, input_file: FourCInput) -> None:
    """Add the node sets contained in the cubit session/exo file to the yaml
    file."""

    # If there are no node sets we can return immediately
    if len(cubit.node_sets) == 0:
        return

    # Get a mapping between the node set IDs and the node set names and keys in the exo file.
    _, exo_id_to_info = get_exo_info(exo, "nodeset")

    # Sort the sets into their geometry type
    node_sets: dict[GeometryType, list] = {
        cupy.geometry.vertex: [],
        cupy.geometry.curve: [],
        cupy.geometry.surface: [],
        cupy.geometry.volume: [],
    }
    for exo_id in range(len(exo.variables["ns_prop1"])):
        cubit_id = exo_id_to_info[exo_id]["cubit_id"]
        node_set_info = cubit.node_sets[cubit_id]
        node_sets[node_set_info.geometry_type].append(
            exo.variables[f"node_ns{exo_id + 1}"][:]
        )

        # We modify the bc_description, thus we create a copy to avoid
        # modifying the original CubitPy data.
        bc_description = node_set_info.bc_description.copy()
        bc_description["E"] = len(node_sets[node_set_info.geometry_type])

        # Only add the boundary condition to the input file if a bc_section is
        # given - we can also add node sets without a boundary condition.
        bc_section = node_set_info.bc_section
        if bc_section is not None:
            if bc_section not in input_file.inlined.keys():
                input_file[bc_section] = []
            input_file[bc_section].append(bc_description)

    # When the mesh is supposed to be contained in the .yaml file, we have
    # to write the topology information of the node sets
    name_geometry_tuple = [
        [cupy.geometry.vertex, "DNODE-NODE TOPOLOGY", "DNODE"],
        [cupy.geometry.curve, "DLINE-NODE TOPOLOGY", "DLINE"],
        [cupy.geometry.surface, "DSURF-NODE TOPOLOGY", "DSURFACE"],
        [cupy.geometry.volume, "DVOL-NODE TOPOLOGY", "DVOL"],
    ]
    for geo, section_name, set_label in name_geometry_tuple:
        if len(node_sets[geo]) > 0:
            input_file[section_name] = []
            for i_set, node_set in enumerate(node_sets[geo]):
                node_set.sort()
                for i_node in node_set:
                    input_file[section_name].append(
                        {
                            "type": "NODE",
                            "node_id": i_node,
                            "d_type": set_label,
                            "d_id": i_set + 1,
                        }
                    )


def add_exodus_geometry_section(
    cubit: CubitPy, input_file: FourCInput, exo_file_name: str
) -> None:
    """Add the problem specific geometry section to the input file required to
    directly read the mesh from an exodus file.

    This section contains information about all element blocks as well as the
    path to the exo file that contains the mesh.

    Args
    ----
    cubit: CubitPy
        The python object for managing the current Cubit session (exclusively
        used in a read-only fashion).
    input_file: FourCInput
        The input file dictionary that will be modified to include the geometry
        section.
    exo_file_name: str
        Name of the exodus file that contains the mesh, it is assumed that this
        file is in the same directory as the yaml input file.
    """

    # Iterate over all blocks and add them to the input file
    element_blocks: dict[str, dict] = {}
    for cur_block_id, cur_block_data in cubit.blocks.items():
        # retrieve the name of the geometry section that this block belongs to
        cur_geometry_section_key = cur_block_data[0].get_four_c_section() + " GEOMETRY"
        if cur_geometry_section_key not in element_blocks:
            element_blocks[cur_geometry_section_key] = {
                "FILE": exo_file_name,
                "SHOW_INFO": "detailed_summary",
                "ELEMENT_BLOCKS": [],
            }
        # retrieve the fourc name (e.g., SOLID/FLUID/...) and the cubit name
        # (e.g., HEX8/TET4/...) for the element
        four_c_element_name = cur_block_data[0].get_four_c_name()
        _, cubit_element_name = cur_block_data[0].get_cubit_names()
        # add block id, fourc element name and element data string to the element block dictionary
        element_block_dict = {
            "ID": cur_block_id,
            four_c_element_name: {cubit_element_name: cur_block_data[1]},
        }
        # append the dictionary with the element block information to the element block list
        element_block_list = element_blocks[cur_geometry_section_key]["ELEMENT_BLOCKS"]
        element_block_list.append(element_block_dict)

    # Add the data to the input file - this will add a deep copy.
    input_file.combine_sections(element_blocks)


def get_element_connectivity_list(connectivity):
    """Return the connectivity list for an element.

    For hex27 we need a different ordering than the one we get from
    cubit.
    """

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
        return [connectivity[i] for i in ordering]
    else:
        # all other elements
        return connectivity.tolist()


def get_input_file_with_mesh(cubit: CubitPy) -> FourCInput:
    """Return a copy of cubit.fourc_input with mesh data (nodes and elements)
    added."""

    # Create exodus file
    os.makedirs(cupy.temp_dir, exist_ok=True)
    exo_path = Path(cupy.temp_dir) / "cubitpy.exo"
    cubit.export_exo(exo_path)
    exo = netCDF4.Dataset(exo_path)

    # create a deep copy of the input_file
    input_file = cubit.fourc_input.copy()
    # Add the node sets
    add_node_sets_input_file(cubit, exo, input_file)

    # Add the nodal data
    input_file["NODE COORDS"] = []
    if "coordz" in exo.variables:
        coordinates = np.array(
            [exo.variables["coord" + dim][:] for dim in ["x", "y", "z"]],
        ).transpose()
    else:
        temp = [exo.variables["coord" + dim][:] for dim in ["x", "y"]]
        temp.append([0 for i in range(len(temp[0]))])
        coordinates = np.array(temp).transpose()
    for i, coordinate in enumerate(coordinates):
        input_file["NODE COORDS"].append(
            {
                "COORD": [coordinate[0], coordinate[1], coordinate[2]],
                "data": {"type": "NODE"},
                "id": i + 1,
            }
        )

    # Add the element connectivity
    _, exo_id_to_info = get_exo_info(exo, "block")
    i_element = 0
    for exo_id in range(len(exo.variables["eb_prop1"])):
        cubit_id = exo_id_to_info[exo_id]["cubit_id"]
        ele_type, block_dict = cubit.blocks[cubit_id]
        block_section = f"{ele_type.get_four_c_section()} ELEMENTS"
        if block_section not in input_file.sections.keys():
            input_file[block_section] = []
        for connectivity in exo.variables[f"connect{exo_id + 1}"][:]:
            input_file[block_section].append(
                {
                    "id": i_element + 1,
                    "cell": {
                        "connectivity": get_element_connectivity_list(connectivity),
                        "type": ele_type.get_four_c_type(),
                    },
                    "data": {
                        "type": ele_type.get_four_c_name(),
                        **block_dict,
                    },
                }
            )
            i_element += 1
    return input_file
