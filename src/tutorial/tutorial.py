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
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
"""This file contains a tutorial for CubitPy, following the Cubit Step-By-Step
tutorial from the cubit documentation.

Most basic functionality is covered by this tutorial. For more
information have a closer look at the test cases, as they cover all
relevant functionality.
"""

import numpy as np

from cubitpy import CubitPy, cupy
from cubitpy.cubit_utility import get_surface_center


def cubit_step_by_step_tutorial_cli(
    input_file_path, *, display=True, cubit=None, size=1.0
):
    """This tutorial follows the Cubit step-by-step tutorial, which can be
    found in the cubit documentation."""

    # Geometry parameters
    brick_size = 10.0
    hole_radius = 3.0

    # The first step is to setup a CubitPy object. This object is derived from the
    # cubit python interface and provides all functionality of the direct cubit
    # python interface and also adds some additional functionality to make the
    # creation of 4C input files easier.
    if None == cubit:
        cubit = CubitPy()

    # Once the CubitPy object is initialized, we can create our first brick
    # object. To do so we use the cubit python interface function `brick`.
    # This function returns a python object referencing the brick.
    brick = cubit.brick(brick_size, brick_size, brick_size)

    # Alternatively the brick could be created via the command line interface (CLI)
    # (not all cubit commands are available through python functions).
    # The `cubit.cmd` method allows to pass any cubit command as string.
    #
    # cubit.cmd(f"brick x {brick_size}")

    # The cube can be shown on your display using the following command
    # (You may have to click on the coordinate icon or refresh the display
    # for the item to show)
    if display:
        cubit.display_in_cubit()

    # You can also show the cube with ID labels on the geometry. The cupy object is
    # a container holding all of cubits enums, such as the geometry type.
    if display:
        cubit.display_in_cubit(labels=[cupy.geometry.curve, cupy.geometry.surface])

    # Now you must create the cylinder which will be used to cut the hole
    # from the brick. This is accomplished with the CLI.
    # (Alternatively we could also use the cubit python command `cubit.cylinder`)
    cubit.cmd(f"create cylinder height {brick_size*1.2} radius {hole_radius}")

    # Executing a command via the CLI does not return the geometry object. To get the
    # cylinder object we can write
    cylinder = cubit.volume(cubit.get_last_id(cupy.geometry.volume))

    # At this point you will see both a brick and a cylinder appear
    # in the CUBIT display window.
    if display:
        cubit.display_in_cubit()

    # Now, the cylinder can be subtracted from the brick to form the hole in the block.
    # Note that the after the subtraction, the cylinder volume is removed and the brick
    # object now points to the result of the subtraction.
    # For the subtraction, we need the ids of the respective volumes. There are two ways
    # to obtain the IDs:
    #   - Open Cubit and manually write down the volume IDs (in this case 1 and 2)
    #         cubit.cmd("subtract 2 from 1")
    #     This has the drawback that if something in the numbering changes, i.e., due to a
    #     Cubit internal change or if more items are inserted before the current ones, that
    #     the code has to be changed.
    #   - The more sustainable way is to get the ID from the cubit objects directly, which can
    #     be done with the `id()` method:
    cubit.cmd(f"subtract volume {cylinder.id()} from volume {brick.id()}")

    # Now we see, that the cylinder is removed from the brick
    if display:
        cubit.display_in_cubit()

    # We now start with the meshing. For more details about this have a look at the
    # Cubit tutorial.

    # First setting the intervals for the brick and defining the mesh
    # size for the volume. we begin by setting an overall volume size interval.
    # Since the brick is 10 units in length on a side, this specifies that each
    # straight curve is to receive approximately 10 mesh elements.
    cubit.cmd(f"volume {brick.id()} size {size}")

    # We will use a sweep mesh, so we will first mesh one surface of the body. We want to
    # first mesh the top surface. This surface can be selected by looking for the only
    # surface with a normal vector in positive z-direction.
    for surface in brick.surfaces():
        # We use the following helper function to get the center of the surface
        center = get_surface_center(surface)

        # Get the normal at the center (also works for the surface with the hole in it)
        normal = surface.normal_at(center)

        # Check if the normal is in positive z-direction
        if np.dot([0, 0, 1], normal) > (1 - 1e-10):
            mesh_surface = surface

    # In order to better resolve the hole in the middle of the surface,
    # we set a smaller size for the curve bounding this hole. We can either get the
    # curve ID from the GUI, or select them in python.
    # Here we select them in python. To do so, we loop over all curves of the surface and
    # check if the curve lies on the surface of the cylinder and on the surface we want.
    for curve in mesh_surface.curves():
        # Position in the middle of the curve
        curve_pos = curve.position_from_fraction(0.5)

        # Radius w.r.t. the cylinder axis
        radius = np.linalg.norm(curve_pos[:2])

        if np.abs(radius - hole_radius) < 1e-10:
            # The curve lies on the cylinder radius, now set the meshing interval
            cubit.cmd(f"curve {curve.id()} interval size {size/2}")

    # Now we can mesh the surface
    cubit.cmd(f"mesh surface {mesh_surface.id()}")

    # We can now see the meshed surface in the CUBIT display window.
    if display:
        cubit.display_in_cubit()

    # The volume mesh can now be generated. Again, the first step is to specify
    # the type of meshing scheme to be used and the second step is to issue the
    # order to mesh. In certain cases, the scheme can be determined by CUBIT
    # automatically. For sweepable volumes, the automatic scheme detection
    # algorithm also identifies the source and target surfaces of the
    # sweep automatically.
    cubit.cmd(f"volume {brick.id()} scheme auto")
    cubit.cmd(f"mesh volume {brick.id()}")

    # We can now see the meshed volume in the CUBIT display window.
    if display:
        cubit.display_in_cubit()

    # We can now define the boundary conditions, from supports to loading.
    # We will place a load on the top side of the cube, and fix the bottom.
    # It is worth noting that using coordinates to select the geometry
    # is preferred, as the element ID may change with different
    # versions / runs of Cubit (as mentioned earlier).
    # The `cubit.add_node_set` method tracks the defined node sets, which allows
    # an automated creation of the BCs in the input file (no need for a `.bc` file)
    cubit.add_node_set(
        cubit.group(add_value="add surface with y_coord < -4.99"),
        name="fix",
        bc_type=cupy.bc_type.dirichlet,
        bc_description="NUMDOF 3 ONOFF 1 1 1 VAL 0 0 0 FUNCT 0 0 0",
    )
    cubit.add_node_set(
        cubit.group(add_value="add surface with y_coord > 4.99 "),
        name="load",
        bc_type=cupy.bc_type.neumann,
        bc_description="NUMDOF 3 ONOFF 0 1 0 VAL 0 0.1 0 FUNCT 0 1 0",
    )

    # Finally we have to set the element blocks.
    cubit.add_element_type(brick.volumes()[0], el_type=cupy.element_type.hex8)

    # We can view the created mesh along with the node sets we defined
    # earlier with the boundary conditions.
    if display:
        cubit.display_in_cubit()

    # Set the head string.
    cubit.head = """
    ------------------------------------------------------------------PROBLEM TYPE
    PROBLEMTYP                      Structure
    ----------------------------------------------------------------------------IO
    OUTPUT_BIN                      no
    STRUCT_DISP                     yes
    FILESTEPS                       1000
    VERBOSITY                       Standard
    STRUCT_STRAIN                   gl
    STRUCT_STRESS                   cauchy
    OUTPUT_SPRING                   Yes
    WRITE_INITIAL_STATE             yes
    ---------------------------------------------------------IO/RUNTIME VTK OUTPUT
    OUTPUT_DATA_FORMAT              binary
    INTERVAL_STEPS                  1
    EVERY_ITERATION                 no
    -----------------------------------------------IO/RUNTIME VTK OUTPUT/STRUCTURE
    OUTPUT_STRUCTURE                yes
    DISPLACEMENT                    yes
    ELEMENT_OWNER                   yes
    STRESS_STRAIN                   yes
    ----------------------------------------------------------------------SOLVER 1
    NAME                            Structure_Solver
    SOLVER                          Superlu
    ------------------------------------------------------------STRUCTURAL DYNAMIC
    INT_STRATEGY                    Standard
    DYNAMICTYP                      Statics
    PRESTRESSTOLDISP                1e-10
    TIMESTEP                        0.5
    NUMSTEP                         20
    MAXTIME                         10
    TOLDISP                         1e-10
    TOLRES                          1e-10
    LINEAR_SOLVER                   1
    NLNSOL                          fullnewton
    MAXITER                         200
    -----------------------------------------------------------STRUCT NOX/Printing
    Outer Iteration                 = Yes
    Inner Iteration                 = No
    Outer Iteration StatusTest      = No
    ---------------------------------------------------------------------MATERIALS
     MAT 1 MAT_Struct_StVenantKirchhoff  YOUNG 1.0E1  NUE 0.3 DENS 0
    --------------------------------------------------------------------------FUNCT1
    SYMBOLIC_FUNCTION_OF_SPACE_TIME t
    """

    # Write the input file.
    cubit.create_dat(input_file_path)


if __name__ == "__main__":
    """Execution part of script."""

    cubit_step_by_step_tutorial_cli("cubit_tutorial.dat", display=False)
