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
"""This file contains a class to represent groups in cubit."""

from .conf import cupy
from .cubit_wrapper.cubit_wrapper_host import CubitObject


class CubitGroup(object):
    """This object helps to represent groups in cubit."""

    def __init__(
        self,
        cubit,
        *,
        name=None,
        add_value=None,
        group_from_id=None,
        group_from_name=None,
    ):
        """Set up the group object.

        Args
        ----
        cubit: CubitPy
            Link to a cubitpy object.
        name: str
            Name of the group in cubit.
        add_value: str
            If this argument is given, it is added to the group.
        group_from_id: int
            Id of an existing group in cubit. If this parameter is given, the
            other optional parameters have to be empty.
        group_from_name: str
            Name of an existing group in cubit. If this parameter is given, the
            other optional parameters have to be empty.
        """

        self.name = name
        self.cubit = cubit

        self.n_node_sets = 0
        self.n_blocks = 0

        if group_from_id is None and group_from_name is None:
            # Create a new group.
            self._id = cubit.create_new_group()

            # Rename it if a name was given.
            if self.name is not None:
                # Check that the name does not already exist.
                if not cubit.get_id_from_name(self.name) == 0:
                    raise ValueError(
                        'The given group name "{}" already exists!'.format(self.name)
                    )
                cubit.cmd("group {} rename '{}'".format(self._id, self.name))

            if add_value is not None:
                self.add(add_value)
        elif group_from_id is not None:
            if (
                add_value is not None
                or self.name is not None
                or group_from_name is not None
            ):
                raise ValueError(
                    'A group can not be initiated with a "group_from_id" and "add_value" or "name" or "group_from_id".'
                )
            self._id = group_from_id
            self.name = self.cubit.cubit.get_entity_name("group", self._id)
        elif group_from_name is not None:
            if (
                add_value is not None
                or self.name is not None
                or group_from_id is not None
            ):
                raise ValueError(
                    'A group can not be initiated with a "group_from_id" and "add_value" or "name" or "group_from_id".'
                )
            self._id = cubit.get_id_from_name(group_from_name)
            self.name = group_from_name
            if self._id == 0:
                raise NameError(
                    'No group with the name "{}" could be found'.format(group_from_name)
                )
        else:
            raise NotImplementedError("This case is not implemented")

    def add(self, add_value):
        """Add items to the group. This can be done in three different ways:

        Args
        ----
        add_value:
            str:
                A string that will be executed in cubit to add items.
            CubitObject.
                Add this object to the group.
            [CubitObject]:
                Add this list of cubit objects.
        """

        if isinstance(add_value, str):
            self.cubit.cmd("group {} {}".format(self._id, add_value))
        elif isinstance(add_value, CubitObject):
            self.cubit.add_entity_to_group(
                self._id,
                add_value.id(),
                add_value.get_geometry_type().get_cubit_string(),
            )
        elif isinstance(add_value, list):
            for item in add_value:
                self.add(item)
        else:
            raise TypeError("Got wrong type {}!".format(type(add_value)))

    def get_geometry_type(self):
        """Return the geometry type of this group.

        This only works if the group contains a single type of geometry
        of finite element objects.
        """

        group_items = self.get_item_ids()
        group_keys = [key for key in group_items if len(group_items[key]) > 0]
        if not len(group_keys) == 1:
            raise TypeError("Got wrong types in get_geometry_type")

        key = group_keys[0]
        if key in cupy.geometry:
            return key
        elif key == cupy.finite_element_object.node:
            return cupy.geometry.vertex
        elif (
            key == cupy.finite_element_object.triangle
            or key == cupy.finite_element_object.face
        ):
            return cupy.geometry.surface
        elif (
            key == cupy.finite_element_object.tet
            or key == cupy.finite_element_object.hex
            or key == cupy.finite_element_object.wedge
        ):
            return cupy.geometry.volume
        else:
            raise TypeError("Error in get_geometry_type")

    def get_geometry_objects(self, item_type):
        """Get a list of all items in the group for the given geometry_type."""
        return self.cubit.get_items(
            item_type, item_ids=self.get_item_ids_from_type(item_type)
        )

    def get_item_ids_from_type(self, item_type):
        """Get the IDs of a certain type of item in this group."""

        # Geometry items.
        if item_type == cupy.geometry.vertex:
            return self.cubit.get_group_vertices(self._id)
        elif item_type == cupy.geometry.curve:
            return self.cubit.get_group_curves(self._id)
        elif item_type == cupy.geometry.surface:
            return self.cubit.get_group_surfaces(self._id)
        elif item_type == cupy.geometry.volume:
            return self.cubit.get_group_volumes(self._id)

        # Finite element items.
        elif item_type == cupy.finite_element_object.node:
            return self.cubit.get_group_nodes(self._id)
        elif item_type == cupy.finite_element_object.edge:
            return self.cubit.get_group_edges(self._id)
        elif item_type == cupy.finite_element_object.face:
            return self.cubit.get_group_quads(self._id)
        elif item_type == cupy.finite_element_object.triangle:
            return self.cubit.get_group_tris(self._id)
        elif item_type == cupy.finite_element_object.tet:
            return self.cubit.get_group_tets(self._id)
        elif item_type == cupy.finite_element_object.hex:
            return self.cubit.get_group_hexes(self._id)
        elif item_type == cupy.finite_element_object.wedge:
            return self.cubit.get_group_wedges(self._id)

        # Cubit items.
        elif item_type == cupy.cubit_items.group:
            return self.cubit.get_group_groups(self._id)

        else:
            raise TypeError("Wrong item type.")

    def _get_item_ids(self, group_items):
        """Add all items in this group to group_items.

        Also add all items of contained subgroups.
        """

        # Add entries from subgroups.
        sub_groups = [
            self.cubit.group(group_id=group_id)
            for group_id in self.get_item_ids_from_type(cupy.cubit_items.group)
        ]
        for sub_group in sub_groups:
            sub_group._get_item_ids(group_items)

        # Add entries from this group.
        for geometry_type in cupy.geometry:
            group_items[geometry_type].extend(
                self.get_item_ids_from_type(geometry_type)
            )
        for fe_object in cupy.finite_element_object:
            group_items[fe_object].extend(self.get_item_ids_from_type(fe_object))

    def get_item_ids(self):
        """Get a dictionary with the IDs of all entries in this group, this
        also includes items in subgroups."""

        # Initialize the empty dictionary.
        group_items = {}
        for geometry_type in cupy.geometry:
            group_items[geometry_type] = []
        for fe_object in cupy.finite_element_object:
            group_items[fe_object] = []

        # Get all entries.
        self._get_item_ids(group_items)

        # Remove double entries.
        for key in group_items.keys():
            group_items[key] = list(set(group_items[key]))
        return group_items

    def add_to_block(self, block_id, el_type):
        """Add the items of this group to a block in cubit. If there are
        explicit elements in this group, they are also added to the block.

        Args
        ----
        block_id: int
            Number of the block which the geometry and or elements should be
            added to.
        el_type: ElementType
            Type of the finite elements.
        """

        self_geometry = self.get_geometry_type()
        if (
            self_geometry != cupy.geometry.surface
            and self_geometry != cupy.geometry.volume
        ):
            raise NotImplementedError("This case is not implemented")

        group_items = self.get_item_ids()
        cubit_scheme, cubit_element_type = el_type.get_cubit_names()

        # Set element type and meshing scheme for each geometry item in the group.
        for i in group_items[self_geometry]:
            self.cubit.cmd(
                "{} {} scheme {}".format(
                    self_geometry.get_cubit_string(), i, cubit_scheme
                )
            )
            self.cubit.cmd(
                "block {} element type {}".format(block_id, cubit_element_type)
            )

        self.cubit.cmd(f"block {block_id} add group {self._id}")

    def add_to_nodeset(self, nodeset_id):
        """Add the nodes from this geometry to a node set.

        Args
        ----
        nodeset_id: int
            Number of the node set which the geometry and or nodes should be
            added to.
        """

        # Add this group. This will add all geometry and directly contained
        # nodes in this group. If there are elements in this group, there will
        # be a warning. This warning is explicitly deactivated here.
        self.cubit.cmd("set warning off")
        self.cubit.cmd("nodeset {} group {}".format(nodeset_id, self._id))
        self.cubit.cmd("set warning on")

        # Add all nodes that are part of faces and elements to the node set.
        group_items = self.get_item_ids()
        nodes = []
        for mesh_item in cupy.finite_element_object:
            for i in group_items[mesh_item]:
                nodes.extend(
                    self.cubit.get_connectivity(mesh_item.get_cubit_string(), i)
                )

        # Add all nodes to the node set.
        for i_node in nodes:
            self.cubit.cmd("nodeset {} node {}".format(nodeset_id, i_node))

    def get_name(self, set_type):
        """Return the name for this set to be used in cubit.

        Args
        ----
        set_type: str
            Type of the set this group is added to. Can be one of the
            following:
              - 'nodeset'
              - 'block'
        """

        return_string = None
        if set_type == "nodeset":
            if self.n_node_sets == 0:
                return_string = self.name
            else:
                return_string = "{}_{}".format(self.name, self.n_node_sets)
            self.n_node_sets += 1
        elif set_type == "block":
            if self.n_blocks > 0:
                raise ValueError(
                    "Only one block can be created from a single group object."
                )
            return_string = self.name
            self.n_blocks += 1
        else:
            raise ValueError('Got unexpected set_type "{}"'.format(set_type))
        return return_string

    def id(self):
        """Return the string with all ids of the types in this object."""
        id_list = self.get_item_ids_from_type(self.get_geometry_type())
        return " ".join(map(str, id_list))

    def __str__(self, *args, **kwargs):
        """The string representation of a group is its name."""
        return self.name
