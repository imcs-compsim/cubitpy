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
"""Implements functions that create geometries in cubit."""

from cubitpy.conf import cupy


def create_spline_interpolation_curve(cubit, vertices, *, delete_points=True):
    """Interpolate multiple vertices with a Cubit internal spline.

    Args
    ----
    cubit: Cubit
        Link to the main cubit object.
    vertices: list(points in R3)
        Points to be interpolated by the spline curve
    delete_points: bool
        If the created vertices should be kept or should be deleted.
    """

    # Create the vertices
    vertices = [cubit.create_vertex(*vertex) for vertex in vertices]
    vertices_ids = [str(vertex.id()) for vertex in vertices]
    cubit.cmd(
        "create curve spline vertex {} {}".format(
            " ".join(vertices_ids), ("delete" if delete_points else "")
        )
    )
    return cubit.curve(cubit.get_last_id(cupy.geometry.curve))


def create_parametric_curve(
    cubit,
    f,
    interval,
    n_segments=10,
    delete_points=True,
    function_args=[],
    function_kwargs={},
):
    """Create a parametric curve in space.

    Args
    ----
    cubit: Cubit
        Link to the main cubit object.
    f: function(t)
        Parametric function of a single parameter t. Maps the parameter to a
        point in R3.
    interval: [t_start, t_end]
        Start and end values for the parameter coordinate.
    n_segments: int
        Number of segments for the interval.#
    delete_points: bool
        If the created vertices should be kept or should be deleted.
    function_args: list
        Additional arguments for the function.
    function_kwargs: dir
        Additional keyword arguments for the function.
    """

    # Create the vertices along the curve.
    parameter_points = [
        interval[0] + i * (interval[1] - interval[0]) / float(n_segments)
        for i in range(n_segments + 1)
    ]
    vertices = [f(t, *function_args, **function_kwargs) for t in parameter_points]
    return create_spline_interpolation_curve(
        cubit, vertices, delete_points=delete_points
    )


def create_parametric_surface(
    cubit,
    f,
    interval,
    n_segments=[10, 10],
    delete_curves=True,
    delete_points=True,
    function_args=[],
    function_kwargs={},
):
    """Create a parametric surface in space.

    Args
    ----
    cubit: Cubit
        Link to the main cubit object.
    f: function(u, v)
        Parametric function of two surface parameters u and v. Maps a single
        set of parameters (u, v) to a point in R3.
    interval: [[u_start, u_end], [v_start, v_end]]
        Start and end values for the parameter coordinate.
    n_segments: [int, int]
        Number of segments for the interval in u and v.
    delete_curves: bool
        If the created curves should be kept or should be deleted.
    delete_points: bool
        If the created vertices should be kept or should be deleted.
    function_args: list
        Additional arguments for the function.
    function_kwargs: dir
        Additional keyword arguments for the function.
    """

    # Loop over the parameter coordinates dimension.
    curves = [[], []]
    for dim in range(2):
        # Get the constant values for the other parameter coordinate in this
        # direction.
        other_dim = 1 - dim
        parameter_points = [
            interval[other_dim][0]
            + i
            * (interval[other_dim][1] - interval[other_dim][0])
            / float(n_segments[other_dim])
            for i in range(n_segments[other_dim] + 1)
        ]

        # Create all curves along this parameter coordinate.
        for point in parameter_points:

            def f_temp(t):
                """Temporary function that is evaluated at a constant value of
                one of the two parameter coordinates."""
                if dim == 0:
                    return f(t, point, *function_args, **function_kwargs)
                else:
                    return f(point, t, *function_args, **function_kwargs)

            curves[dim].append(
                create_parametric_curve(
                    cubit,
                    f_temp,
                    interval[dim],
                    n_segments=n_segments[dim],
                    delete_points=delete_points,
                )
            )

    # Create the surface.
    curve_u_ids = " ".join([str(curve.id()) for curve in curves[0]])
    curve_v_ids = " ".join([str(curve.id()) for curve in curves[1]])
    cubit.cmd(
        "create surface net U curve {} V curve {} noheal".format(
            curve_u_ids, curve_v_ids
        )
    )

    if delete_curves:
        for id_string in [curve_u_ids, curve_v_ids]:
            cubit.cmd("delete curve {}".format(id_string))

    return cubit.surface(cubit.get_last_id(cupy.geometry.surface))


def create_surface_by_vertices(cubit, vertices):
    """Create a surface by the bounding vertices.

    Args
    ----
    cubit: Cubit
        Link to the main cubit object.
    vertices: list(Cubit.Vertex)
        A list of cubit vertices that make up the surface. Be aware that
        the ordering matters.
    """
    vertex_str = " ".join([str(vertex.id()) for vertex in vertices])
    cubit.cmd(f"create surface vertex {vertex_str}")
    last_id = cubit.get_last_id("surface")
    return cubit.surface(last_id)


def create_brick_by_corner_points(cubit, corner_points):
    """Create a brick by its corner points.

    Args
    ----
    cubit: Cubit
        Link to the main cubit object.
    corner_points: list(points in R3)
        A list or array of points in 3D that make up the brick. The ordering
        is expected to be the same as for a hex8 finite element.
    """

    vertices = [
        cubit.create_vertex(float(vertex[0]), float(vertex[1]), float(vertex[2]))
        for vertex in corner_points
    ]
    surface_vertex_ids = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [1, 2, 6, 5],
        [2, 6, 7, 3],
        [0, 3, 7, 4],
        [0, 1, 5, 4],
    ]
    surfaces = [
        create_surface_by_vertices(cubit, [vertices[id] for id in ids])
        for ids in surface_vertex_ids
    ]
    surface_str = " ".join([str(surface.id()) for surface in surfaces])
    cubit.cmd(f"create volume surface {surface_str} heal")
    last_id = cubit.get_last_id("volume")
    return cubit.volume(last_id)
