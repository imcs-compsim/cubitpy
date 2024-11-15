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
"""This script gets called with the cubit python interpreter and loads the cubit
module. With the package execnet in the host python interpreter a connection is
established between the two different python interpreters and data and commands
can be exchanged. The exchange happens in a serial matter, items are sent to this
script, and results are sent back, until None is sent. If cubit creates a cubit
object it is saved in a dictionary in this script, with the key being the id of
the object. The host interpreter only knows the id of this object and can pass it 
to this script to call a function on it or use it as an argument."""

# Python modules
import sys
import os


# Cubit constants
cubit_vertex = "cubitpy_vertex"
cubit_curve = "cubitpy_curve"
cubit_surface = "cubitpy_surface"
cubit_volume = "cubitpy_volume"


# Default parameters
parameters = {}


def out(string):
    """The print version does over different interpreters, so this function prints
    strings to an active console. Insert the path of your console to get the
    right output.
    To get the current path of your console type: tty
    """

    if "tty" in parameters.keys():
        out_console = parameters["tty"]
    else:
        out_console = "/dev/pts/18"
    escaped_string = "{}".format(string).replace('"', '\\"')
    os.system('echo "{}" > {}'.format(escaped_string, out_console))


def is_cubit_type(obj):
    """Check if the object is of a cubit base."""
    if (
        isinstance(obj, cubit.Body)
        or isinstance(obj, cubit.Vertex)
        or isinstance(obj, cubit.Curve)
        or isinstance(obj, cubit.Surface)
        or isinstance(obj, cubit.Volume)
        or isinstance(obj, cubit.MeshImport)
    ):
        return True
    else:
        return False


# All cubit items that are created are stored in this dictionary. The keys are
# the unique object ids. The items are deleted once they run out of scope in
# the host interpreter.
cubit_objects = {}


# The first call are parameters needed in this script
parameters = channel.receive()
channel.send(None)
if not isinstance(parameters, dict):
    raise TypeError(
        "The first item should be a dictionary. Got {}!\nparameters={}".format(
            type(parameters), parameters
        )
    )

# Add paths to sys and load utility functions and cubit
dir_name = os.path.dirname(parameters["__file__"])
sys.path.append(dir_name)
sys.path.append(parameters["cubit_lib_path"])

from cubit_wrapper_utility import object_to_id, cubit_item_to_id, is_base_type
import cubit


# The second call is the initialization call for cubit
# init = ['init', cubit_path, [args]]
init = channel.receive()
if not init[0] == "init":
    raise ValueError("The second call must be init!")
if not len(init) == 2:
    raise ValueError("Two arguments must be given to init!")
cubit.init(init[1])
cubit_objects[id(cubit)] = cubit
channel.send(object_to_id(cubit))


# Now start an endless loop (until None is sent) and perform the cubit functions
while 1:
    # Get input from the python host.
    receive = channel.receive()

    # If None is sent, break the connection and exit
    if receive is None:
        break

    # The first argument decides that functionality will be performed:
    # 'cubit_object': return an attribute of a cubit object. If the attribute is
    #       callable, it is executed with the given arguments.
    #       [[cubit_object], 'name', ['arguments']]
    # 'iscallable': Check if a name is callable or not
    # 'isinstance': Check if the cubit object is of a cerain instance
    # 'get_self_dir': Return the attributes in a cubit_object
    # 'delete': Delete the cubit pbject from the dictionary

    if cubit_item_to_id(receive[0]) is not None:
        # The first item is an id for a cubit object. Return an attribute of
        # this object.

        # Get object and attribute name
        call_object = cubit_objects[cubit_item_to_id(receive[0])]
        name = receive[1]

        def deserialize_item(item):
            """
            Deserialize the item, also if it contains nested nested lists.
            """
            item_id = cubit_item_to_id(item)
            if item_id is not None:
                return cubit_objects[item_id]
            elif isinstance(item, tuple) or isinstance(item, list):
                arguments = []
                for sub_item in item:
                    arguments.append(deserialize_item(sub_item))
                return arguments
            else:
                return item

        if callable(getattr(call_object, name)):
            # Call the function
            arguments = deserialize_item(receive[2])
            cubit_return = call_object.__getattribute__(name)(*arguments)
        else:
            # Get the attribute value
            cubit_return = call_object.__getattribute__(name)

        # Check what to return
        if is_base_type(cubit_return):
            # The return item is a string, integer or float
            channel.send(cubit_return)

        elif isinstance(cubit_return, tuple):
            # A tuple was returned, loop over each entry and check its type
            return_list = []
            for item in cubit_return:
                if is_base_type(item):
                    return_list.append(item)
                elif is_cubit_type(item):
                    cubit_objects[id(item)] = item
                    return_list.append(object_to_id(item))
                else:
                    raise TypeError(
                        "Expected string, int, float or cubit object! Got {}!".format(
                            item
                        )
                    )
            channel.send(return_list)

        elif is_cubit_type(cubit_return):
            # Store the object locally and return the id
            cubit_objects[id(cubit_return)] = cubit_return
            channel.send(object_to_id(cubit_return))

        else:
            raise TypeError(
                "Expected string, int, float, cubit object or tuple! Got {}!".format(
                    cubit_return
                )
            )

    elif receive[0] == "iscallable":
        cubit_object = cubit_objects[cubit_item_to_id(receive[1])]
        channel.send(callable(getattr(cubit_object, receive[2])))

    elif receive[0] == "isinstance":
        # Compare the second item with a predefined cubit class
        compare_object = cubit_objects[cubit_item_to_id(receive[1])]

        if receive[2] == cubit_vertex:
            channel.send(isinstance(compare_object, cubit.Vertex))
        elif receive[2] == cubit_curve:
            channel.send(isinstance(compare_object, cubit.Curve))
        elif receive[2] == cubit_surface:
            channel.send(isinstance(compare_object, cubit.Surface))
        elif receive[2] == cubit_volume:
            channel.send(isinstance(compare_object, cubit.Volume))
        else:
            raise ValueError(
                "Wrong compare type given! Expected vertex, curve, surface or volume, got{}".format(
                    receive[2]
                )
            )

    elif receive[0] == "get_self_dir":
        # Return a list with all callable methods of this object
        cubit_object = cubit_objects[cubit_item_to_id(receive[1])]
        channel.send(
            [
                [method_name, callable(getattr(cubit_object, method_name))]
                for method_name in dir(cubit_object)
            ]
        )

    elif receive[0] == "delete":
        # Get the id of the object to delete
        cubit_id = cubit_item_to_id(receive[1])
        if cubit_id is None:
            raise TypeError("Expected cubit object! Got {}!".format(item))

        # Delete the object from the dictionary.
        if cubit_id in cubit_objects.keys():
            del cubit_objects[cubit_id]
        else:
            raise ValueError(
                "The id {} is not in the cubit_objects dictionary".format(cubit_id)
            )

        # Return to python host
        channel.send(None)

    else:
        raise ValueError('The case of "{}" is not implemented!'.format(receive[0]))


# Send EOF
channel.send("EOF")
