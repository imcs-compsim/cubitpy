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
"""Implements a class that helps create meshes with cubit."""

import os
import subprocess  # nosec B404
import time
import warnings

import netCDF4
from fourcipp.fourc_input import FourCInput

from cubitpy.conf import cupy
from cubitpy.cubit_group import CubitGroup
from cubitpy.cubit_to_fourc_input import (
    add_exodus_geometry_section,
    add_node_sets,
    get_input_file_with_mesh,
)
from cubitpy.cubit_wrapper.cubit_wrapper_host import CubitConnect


class CubitPy(object):
    """A wrapper class with additional functionality for cubit."""

    def __init__(self, *, cubit_exe=None, **kwargs):
        """Initialize CubitPy.

        Args
        ----
        cubit_exe: str
            Path to the cubit executable

        kwargs:
            Arguments passed on to the creation of the python wrapper
        """

        # Set paths
        if cubit_exe is None:
            cubit_exe = cupy.get_cubit_exe_path()
        self.cubit_exe = cubit_exe

        # Set the "real" cubit object
        self.cubit = CubitConnect(**kwargs).cubit

        # Reset cubit
        self.cubit.cmd("reset")
        self.cubit.cmd("set geometry engine acis")

        # Set lists and counters for blocks and sets
        self._default_cubit_variables()

        self.fourc_input = FourCInput()

    def _default_cubit_variables(self):
        """Set the default values for the lists and counters used in cubit."""
        self.blocks = []
        self.node_sets = []
        self.fourc_input = FourCInput()

    def __getattr__(self, key, *args, **kwargs):
        """All calls to methods and attributes that are not in this object get
        passed to cubit."""
        return self.cubit.__getattribute__(key, *args, **kwargs)

    def _name_created_set(self, set_type, set_id, name, item):
        """Create a node set or block and name it. This is an own method
        because it can be used for both types of set in cubit. If the added
        item is a group, no explicit name should be given and the group name
        should be used.

        Args
        ----
        set_type: str
            Type of the set to be added. Can be one of the following:
              - 'nodeset'
              - 'block'
        set_id: int
            Id of the item to rename.
        name: str
            An explicitly given name.
        item: CubitObject, CubitGroup
            The item that was added to the set.
        """

        # Check if the item is a group and if it has a name.
        if isinstance(item, CubitGroup) and item.name is not None:
            group_name = item.get_name(set_type)
        else:
            group_name = None

        # If two names are given, a warning is displayed as this is not the
        # intended case.
        rename_name = None
        if name is not None and group_name is not None:
            warnings.warn(
                'A {} is added for the group "{}" and an explicit name of "{}" is given. This might be unintended, as usually if a group is given, we expect to use the name of the group. In the current case we will use the given name.'.format(
                    set_type, item.name, name
                )
            )
            rename_name = name
        elif group_name is not None:
            rename_name = group_name
        elif name is not None:
            rename_name = name

        # Rename the item.
        if rename_name is not None:
            self.cubit.cmd('{} {} name "{}"'.format(set_type, set_id, rename_name))

    def add_element_type(
        self, item, el_type, *, name=None, material=None, bc_description=None
    ):
        """Add a block to cubit that contains the geometry in item. Also set
        the element type of block.

        Args
        ----
        item: CubitObject, CubitGroup
            Geometry to set the element type for.
        el_type: cubit.ElementType
            Cubit element type.
        name: str
            Name of the block.
        material: dict
            Material string of the block, will be the first part of the BC
            description.
        bc_description: dict
            Will be written after the material string. If this is not set, the
            default values for the given element type will be used.
        """

        # default values
        if material is None:
            material = {"MAT": 1}
        if bc_description is None:
            bc_description = {}

        # Check that all blocks in cubit are created with this function.
        n_blocks = len(self.blocks)
        if not len(self.cubit.get_block_id_list()) == n_blocks:
            raise ValueError(
                "The block counter is {1}, but the number of blocks in cubit is {0}, all blocks should be created with this function!".format(
                    len(self.cubit.get_block_id_list()), n_blocks
                )
            )

        # Get element type of item.
        geometry_type = item.get_geometry_type()

        self.cubit.cmd("create block {}".format(n_blocks + 1))

        if not isinstance(item, CubitGroup):
            cubit_scheme, cubit_element_type = el_type.get_cubit_names()

            # Set the meshing scheme for this element type.
            self.cubit.cmd(
                "{} {} scheme {}".format(
                    geometry_type.get_cubit_string(), item.id(), cubit_scheme
                )
            )

            self.cubit.cmd(
                "block {} {} {}".format(
                    n_blocks + 1, geometry_type.get_cubit_string(), item.id()
                )
            )
            self.cubit.cmd(
                "block {} element type {}".format(n_blocks + 1, cubit_element_type)
            )
        else:
            item.add_to_block(n_blocks + 1, el_type)

        self._name_created_set("block", n_blocks + 1, name, item)

        # If the user does not give a bc_description, load the default one.
        if not bc_description:
            bc_description = el_type.get_default_four_c_description()

        # Add data that will be written to bc file.
        self.blocks.append([el_type, material | bc_description])

    def reset_blocks(self):
        """This method deletes all blocks in Cubit and resets the counter in
        this object."""

        # Reset the block list of this object.
        self.blocks = []

        # Delete all blocks.
        for block_id in self.get_block_id_list():
            self.cmd("delete Block {}".format(block_id))

    def add_node_set(
        self,
        item,
        *,
        name=None,
        bc_type=None,
        bc_description=None,
        bc_section=None,
        geometry_type=None,
    ):
        """Add a node set to cubit. This node set can have a boundary
        condition.

        Args
        ----
        item: CubitObject, CubitGroup
            Geometry whose nodes will be put into the node set.
        name: str
            Name of the node set.
        bc_type: cubit.bc_type
            Type of boundary (dirichlet or neumann).
        bc_section: str
            Name of the section in the input file. Mutually exclusive with
            bc_type.
        bc_description: dict
            Definition of the boundary condition.
        geometry_type: cupy.geometry
            Directly set the geometry type, instead of obtaining it from the
            given item.
        """

        # Check that all node sets in cubit are created with this function.
        n_node_sets = len(self.node_sets)
        if not len(self.cubit.get_nodeset_id_list()) == n_node_sets:
            raise ValueError(
                "The node set counter is {1}, but the number of node sets in cubit is {0}, all node sets should be created with this function!".format(
                    len(self.cubit.get_nodeset_id_list()), n_node_sets
                )
            )

        # Get element type of item if it was not explicitly given.
        if geometry_type is None:
            geometry_type = item.get_geometry_type()

        self.cubit.cmd("create nodeset {}".format(n_node_sets + 1))
        if not isinstance(item, CubitGroup):
            # Add the geometries to the node set in cubit.
            self.cubit.cmd(
                "nodeset {} {} {}".format(
                    n_node_sets + 1, geometry_type.get_cubit_string(), item.id()
                )
            )
        else:
            # Add the group to the node set in cubit.
            item.add_to_nodeset(n_node_sets + 1)

        self._name_created_set("nodeset", n_node_sets + 1, name, item)

        # Add data that will be written to bc file.
        if (
            (bc_section is None and bc_type is None)
            or bc_section is not None
            and bc_type is not None
        ):
            raise ValueError(
                'One of the two arguments "bc_section" and '
                + '"bc_type" has to be set!'
            )
        if bc_section is None:
            bc_section = bc_type.get_dat_bc_section_header(geometry_type)
        if bc_description is None:
            bc_description = {}
        self.node_sets.append([bc_section, bc_description, geometry_type])

    def get_ids(self, geometry_type):
        """Get a list with all available ids of a certain geometry type."""
        return self.get_entities(geometry_type.get_cubit_string())

    def get_items(self, geometry_type, item_ids=None):
        """Get a list with all available cubit objects of a certain geometry
        type."""

        if geometry_type == cupy.geometry.vertex:
            funct = self.vertex
        elif geometry_type == cupy.geometry.curve:
            funct = self.curve
        elif geometry_type == cupy.geometry.surface:
            funct = self.surface
        elif geometry_type == cupy.geometry.volume:
            funct = self.volume
        else:
            raise ValueError("Got unexpected geometry type!")

        if item_ids is None:
            item_ids = self.get_ids(geometry_type)
        return [funct(index) for index in item_ids]

    def set_line_interval(self, item, n_el):
        """Set the number of elements along a line.

        Args
        ----
        item: cubit.curve
            The line that will be seeded into the intervals.
        n_el: int
            Number of intervals along line.
        """

        # Check if item is line.
        if not item.get_geometry_type() == cupy.geometry.curve:
            raise TypeError("Expected line, got {}".format(type(item)))
        self.cubit.cmd("curve {} interval {} scheme equal".format(item.id(), n_el))

    def export_cub(self, path):
        """Export the cubit input."""
        if cupy.is_coreform():
            self.cubit.cmd(f'save cub5 "{path}" overwrite journal')
        else:
            self.cubit.cmd('save as "{}" overwrite'.format(path))

    def export_exo(self, path):
        """Export the mesh."""
        self.cubit.cmd('export mesh "{}" dimension 3 overwrite'.format(path))

    def dump(self, yaml_path, mesh_in_exo=False):
        """Create the yaml file and save it in under provided yaml_path.

        Args
        ----
        yaml_path: str
            Path where the input file will be saved
        mesh_in_exo: bool
            If True, the mesh will be exported in exodus format and the input file
            will contain a reference to the exodus file. If False, the mesh will
            be exported in the 4C format and the input file will contain the mesh
            directly in the yaml file.
            Default is False.
        """

        # Check if output path exists
        dat_dir = os.path.dirname(os.path.abspath(yaml_path))
        if not os.path.exists(dat_dir):
            raise ValueError("Path {} does not exist!".format(dat_dir))

        if mesh_in_exo:
            # Determine the path stem: Strip the '(.4C).yaml' suffix
            # (if the filename does not contain '.4C' the second call to
            # 'removesuffix' won't alter the string at all)
            path_stem = yaml_path.removesuffix(".yaml").removesuffix(".4C")
            # Export the mesh in exodus format
            exo_path = path_stem + ".exo"
            self.export_exo(exo_path)
            # parse the exodus file
            exo = netCDF4.Dataset(exo_path)
            # create a deep copy of the input_file
            input_file = self.fourc_input.copy()
            # Add the node sets
            add_node_sets(self, exo, input_file, write_topology_information=False)
            # Add the problem geometry section
            rel_exo_path = os.path.relpath(exo_path, start=dat_dir)
            add_exodus_geometry_section(self, input_file, rel_exo_path)
        else:
            input_file = get_input_file_with_mesh(self)
        # Export the input file in YAML format
        input_file.dump(yaml_path)

    def group(self, **kwargs):
        """Reference a group in cubit.

        Depending on the passed keyword arguments the group is created
        or just references an existing group.
        """
        return CubitGroup(self, **kwargs)

    def reset(self):
        """Reset all objects in cubit and the created BCs and blocks and node
        sets."""

        self.cubit.reset()
        self._default_cubit_variables()

    def display_in_cubit(self, labels=[], delay=0.5, testing=False):
        """Save the state to a cubit file and open cubit with that file.
        Additionally labels can be displayed in cubit to simplify the mesh
        creation process.

        Attention - displays for stls not the same as an export_exo (TODO: maybe
        use import instead of open).

        Args
        ----
        labels: [GeometryType, FiniteElementObject]
            What kind of labels should be shown in cubit.
        delay: float
            Time (in seconds) to wait after sending the write command until the
            new cubit session is opened.
        testing: bool
            If this is true, cubit will not be opened, instead the created
            journal and command will re returned.
        """

        # Export the cubit state. After the export, we wait, to ensure that the
        # write operation finished, and the state file can be opened cleanly
        # (in some cases the creation of the state file takes to long and in
        # the subsequent parts of this code we open a file that is not yet
        # fully written to disk).
        # TODO: find a way to do this without the wait command, but to check if
        # the file is readable.
        os.makedirs(cupy.temp_dir, exist_ok=True)
        if cupy.is_coreform():
            state_path = os.path.join(cupy.temp_dir, "state.cub5")
        else:
            state_path = os.path.join(cupy.temp_dir, "state.cub")
        self.export_cub(state_path)
        time.sleep(delay)

        # Write file that opens the state in cubit.
        journal_path = os.path.join(cupy.temp_dir, "open_state.jou")
        with open(journal_path, "w") as journal:
            journal.write('open "{}"\n'.format(state_path))

            # Get the cubit names of the desired display items.
            cubit_names = [label.get_cubit_string() for label in labels]

            # Label items in cubit, per default all labels are deactivated.
            cubit_labels = [
                "volume",
                "surface",
                "curve",
                "vertex",
                "hex",
                "tet",
                "face",
                "tri",
                "edge",
                "node",
            ]
            for item in cubit_labels:
                if item in cubit_names:
                    on_off = "On"
                else:
                    on_off = "Off"
                journal.write("label {} {}\n".format(item, on_off))
            journal.write("display\n")

        # Get the command and arguments to open cubit with.
        cubit_command = [
            self.cubit_exe,
            "-nojournal",
            "-information",
            "Off",
            "-input",
            "open_state.jou",
        ]

        if not testing:
            # Open the state in cubit.
            subprocess.call(
                cubit_command,  # nosec B603
                cwd=cupy.temp_dir,
            )
        else:
            return journal_path
