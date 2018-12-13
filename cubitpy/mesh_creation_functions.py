# -*- coding: utf-8 -*-
"""
Implements functions that create basic meshes in cubit.
"""


# Import cubitpy stuff.
from . import cupy


def create_brick(cubit, h_x, h_y, h_z, *, element_type=None, mesh_size=None,
        name=None, mesh=True, material='MAT 1 KINEM nonlinear'):
    """
    Create a cube in cubit.

    Args
        ----
        cubit: CubitPy
            CubitPy object.
        h_x, h_y, h_z: float
            size of the cube in x, y, and z direction.
        element_type: str
            Type of the created element (HEX8, HEX20, HEX27, TETRA4, TETRA10)
        mesh_size: list
            mesh_size[0] == 'factor':
                A global factor will be set for the meshsize. The factor will
                be taken from mesh_size[1].
            mesh_size[0] == 'interval':
                Intervals in each direction will be set. The intervals are
                taken from mesh_size[1:]
        name: str
            Name that will be given to this block in the baci bc file.
        mesh: bool
            If the cube will be meshed or not.
        material: str
            Material string in baci.
    """

    # Check if default value has to be set for element_type.
    if element_type is None:
        element_type = cupy.element_type.hex8

    # Check the input parameters.
    if h_x < 0 or h_y < 0 or h_z < 0:
        raise ValueError('Only positive lengths are possible!')

    # Get the element type parameters.
    cubit_scheme, cubit_element_type, baci_element_type, baci_dat_string = \
        element_type.get_element_type_strings()

    # Create the block in cubit.
    solid = cubit.brick(h_x, h_y, h_z)
    volume_id = solid.volumes()[0].id()

    # Set the size and type of the elements.
    cubit.cmd('volume {} scheme {}'.format(volume_id, cubit_scheme))

    # Set mesh properties.
    if mesh_size is not None:
        if mesh_size[0] == 'factor':
            # Set a cubit factor for the mesh size. 10 is the largest.
            cubit.cmd('volume {} size auto factor {}'.format(
                volume_id, mesh_size[1]))
        elif mesh_size[0] == 'interval':
            # Set the number of elements in x, y and z direction.
            cubit.cmd('curve {} {} {} {} interval {} scheme equal'.format(
                solid.curves()[1].id(), solid.curves()[3].id(),
                solid.curves()[5].id(), solid.curves()[7].id(),
                mesh_size[1]))
            cubit.cmd('curve {} {} {} {} interval {} scheme equal'.format(
                solid.curves()[0].id(), solid.curves()[2].id(),
                solid.curves()[4].id(), solid.curves()[6].id(),
                mesh_size[2]))
            cubit.cmd('curve {} {} {} {} interval {} scheme equal'.format(
                solid.curves()[8].id(), solid.curves()[9].id(),
                solid.curves()[10].id(), solid.curves()[11].id(),
                mesh_size[3]))

    # Set the element type.
    cubit.add_element_type(solid.volumes()[0], cubit_element_type, name=name,
        bc=['STRUCTURE', '{} {}'.format(material, baci_dat_string),
            baci_element_type
            ])

    # Mesh the created block.
    if mesh:
        cubit.cmd('mesh volume {}'.format(volume_id))

    return solid