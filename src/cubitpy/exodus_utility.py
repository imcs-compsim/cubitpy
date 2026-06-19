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
"""Utility functions interacting with the exodus file format."""

from pathlib import Path

import netCDF4
import numpy as np


def get_exo_info(exo, entry_type) -> tuple[dict, dict]:
    """Build mappings between Exodus IDs and Cubit IDs for blocks or
    nodesets."""

    if entry_type == "block":
        exo_identifier = "eb"
    elif entry_type == "nodeset":
        exo_identifier = "ns"
    else:
        raise ValueError(f"Invalid entry type: {entry_type}")

    if exo_identifier + "_names" not in exo.variables.keys():
        return {}, {}

    # List of explicitly given names
    names = []
    for line in exo.variables[exo_identifier + "_names"]:
        name: str | None = str(netCDF4.chartostring(line))
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


def convert_exodus_to_dict(exo_path: Path) -> dict:
    """Load an exodus file from disk and convert it to a dictionary containing
    all the relevant information.

    This function can be used in testing to compare exodus files with
    each other.
    """
    exo_data = {}
    with netCDF4.Dataset(exo_path) as exo:
        exo_data["coordinates"] = (
            np.array(
                [exo.variables["coord" + dim][:] for dim in ["x", "y", "z"]],
            )
            .transpose()
            .tolist()
        )

        _, exo_block_id_to_info = get_exo_info(exo, "block")
        exo_data["exo_block_id_to_info"] = exo_block_id_to_info
        for exo_id in exo_block_id_to_info.keys():
            connectivity_name = f"connect{exo_id + 1}"
            connectivity = np.array(exo.variables[connectivity_name][:]).tolist()
            exo_data[connectivity_name] = connectivity

        _, exo_node_set_id_to_info = get_exo_info(exo, "nodeset")
        exo_data["exo_node_set_id_to_info"] = exo_node_set_id_to_info
        for exo_id in exo_node_set_id_to_info.keys():
            node_set_name = f"node_ns{exo_id + 1}"
            unique_node_set_ids = np.unique(np.array(exo.variables[node_set_name][:]))
            # The correct data type here would be `set`. However, this can not be
            # serialized by FourCIPP, so we return an ordered list of the unique IDs here.
            exo_data[node_set_name] = unique_node_set_ids.tolist()
    return exo_data
