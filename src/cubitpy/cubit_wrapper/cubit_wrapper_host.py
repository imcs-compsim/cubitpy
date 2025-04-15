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
"""This module creates object that are used to connect between the cubit python
interpreter and the main python interpreter."""

import atexit
import os

import execnet
import numpy as np

from cubitpy.conf import cupy
from cubitpy.cubit_wrapper.cubit_wrapper_utility import cubit_item_to_id, is_base_type


class CubitConnect(object):
    """This class holds a connection to a cubit python interpreter and
    initializes cubit there.

    It is possible to send function calls to that interpreter and
    receive the output.
    """

    def __init__(
        self,
        *,
        cubit_args=None,
        cubit_lib=None,
        interpreter=None,
    ):
        """Initialize the connection between the client (cubit) python
        interpreter and this one. Also load the cubit module in the remote
        interpreter.

        Args
        ----
        cubit_args: [str]
            List of arguments to pass to cubit.init
        cubit_lib: str
            Path to the directory containing the cubit python library.
                - For Linux systems this is the path where the `cubit`
                script is located (`bin` should be a subfolder of this path).
                - For MacOS this is the `.../Cubit.app/Contents/MacOS`) folder.
        interpreter: str
            Python interpreter to be used for running cubit.
        """

        if interpreter is None:
            interpreter = f"popen//python={cupy.get_cubit_interpreter()}"

        if cubit_lib is None:
            cubit_lib = cupy.get_cubit_lib_path()

        # Set up the client python interpreter
        self.gw = execnet.makegateway(interpreter)
        self.gw.reconfigure(py3str_as_py2str=True)

        # Load the main code in the client python interpreter
        client_python_file = os.path.join(
            os.path.dirname(__file__), "cubit_wrapper_client.py"
        )
        with open(client_python_file, "r") as myfile:
            data = myfile.read()

        # Set up the connection channel
        self.channel = self.gw.remote_exec(data)

        # Send parameters to the client interpreter
        parameters = {}
        parameters["__file__"] = __file__
        parameters["cubit_lib_path"] = cubit_lib

        # Arguments for cubit
        if cubit_args is None:
            arguments = [
                "cubit",
                # "-log",  # Write the log to a file
                # "dev/null",
                "-information",  # Do not output information of cubit
                "Off",
                "-nojournal",  # Do write a journal file
                "-noecho",  # Do not output commands used in cubit
            ]
        else:
            arguments = ["cubit"] + cubit_args

        # Check if a log file was given in the cubit arguments
        for arg in arguments:
            if arg.startswith("-log="):
                log_given = True
                break
        else:
            log_given = False

        self.log_check = False

        if not log_given:
            # Write the log to a temporary file and check the contents after each call to cubit
            arguments.extend(["-log", cupy.temp_log])
            parameters["tty"] = cupy.temp_log
            self.log_check = True

        # Send the parameters to the client interpreter
        self.send_and_return(parameters)

        # Initialize cubit in the client and create the linking object here
        cubit_id = self.send_and_return(["init", arguments])
        self.cubit = CubitObjectMain(self, cubit_id)

        def cleanup_execnet_gateway():
            """We need to register a function called at interpreter shutdown
            that ensures that the execnet connection is closed first,
            otherwise, we get a runtime error during shutdown."""
            self.cubit.cubit_connect.gw.exit()

        atexit.register(cleanup_execnet_gateway)

    def send_and_return(self, argument_list):
        """Send arguments to the python client and collect the return values.

        Args
        ----
        argument_list: list
            First item is either a string with the action, or a cubit item id.
            In the second case a method will be called on the item, with the
            arguments stored in the second entry in argument_list.
        """

        # If the channel is already finalized we get a runtime error here. This happens in cases
        # where we delete items after the connection has been closed. We catch this error here.
        try:
            self.channel.send(argument_list)
            return self.channel.receive()
        except:
            return None

    def get_attribute(self, cubit_object, name):
        """Return the attribute 'name' of cubit_object. If the attribute is
        callable a function is returned, otherwise the attribute value is
        returned.

        Args
        ----
        cubit_object: CubitObject
            The object on which the method is called.
        name: str
            Name of the method.
        """

        def function(*args):
            """This function gets returned from the parent method."""

            def serialize_item(item):
                """Serialize an item, also nested lists."""

                if (
                    isinstance(item, tuple)
                    or isinstance(item, list)
                    or isinstance(item, np.ndarray)
                ):
                    arguments = []
                    for sub_item in item:
                        arguments.append(serialize_item(sub_item))
                    return arguments
                elif isinstance(item, CubitObject):
                    return item.cubit_id
                elif isinstance(item, float):
                    return float(item)
                elif isinstance(item, int):
                    return int(item)
                elif isinstance(item, cupy.geometry):
                    return item.get_cubit_string()
                else:
                    return item

            if self.log_check:
                # Check if the log file is empty. If it is not, empty it.
                if os.stat(cupy.temp_log).st_size != 0:
                    with open(cupy.temp_log, "w"):
                        pass

            # Check if there are cubit objects in the arguments
            arguments = serialize_item(args)

            # Call the method on the cubit object
            cubit_return = self.send_and_return(
                [cubit_object.cubit_id, name, arguments]
            )

            if self.log_check:
                # Print the content of the log file
                with open(cupy.temp_log, "r") as log_file:
                    print(log_file.read(), end="")

            # Check if the return value is a cubit object
            if cubit_item_to_id(cubit_return) is not None:
                return CubitObject(self, cubit_return)
            elif isinstance(cubit_return, list):
                # If the return value is a list, check if any entry of the list
                # is a cubit object
                return_list = []
                for item in cubit_return:
                    if cubit_item_to_id(item) is not None:
                        return_list.append(CubitObject(self, item))
                    elif is_base_type(item):
                        return_list.append(item)
                    else:
                        raise TypeError(
                            "Expected cubit object, or base_type, "
                            + "got {}!".format(item)
                        )
                return return_list
            elif is_base_type(cubit_return):
                return cubit_return
            else:
                raise TypeError(
                    "Expected cubit object, or base_type, "
                    + "got {}!".format(cubit_return)
                )

        # Depending on the type of attribute, return the attribute value or a
        # callable function
        if self.send_and_return(["iscallable", cubit_object.cubit_id, name]):
            return function
        else:
            return function()


class CubitObject(object):
    """This class holds a link to a cubit object in the client.

    Methods that are called on this class will 'really' be called in the
    client.
    """

    def __init__(self, cubit_connect, cubit_data_list):
        """Initialize the object.

        Args
        ----
        cubit_connect: CubitConnect
            A link to the cubit_connect object that will be used to call
            methods.
        cubit_data_list: []
            A list of strings that contains info about the cubit object.
            The first item is the id of this object in th client.
        """

        # Check formatting of cubit_id
        if cubit_item_to_id(cubit_data_list) is None:
            raise TypeError("Wrong type {}".format(cubit_data_list))

        self.cubit_connect = cubit_connect
        self.cubit_id = cubit_data_list

    def __getattribute__(self, name, *args, **kwargs):
        """This function gets called for each attribute in this object. First
        it is checked if the attribute exists in the host (basic stuff), if not
        the attribute is called on the client.

        For now if an attribute is sent to the client, it is assumed
        that it is a method.
        """

        # Check if the attribute exists in this interpreter
        try:
            return object.__getattribute__(self, name, *args, **kwargs)
        except AttributeError:
            return self.cubit_connect.get_attribute(self, name)

    def __del__(self):
        """When this object is deleted, the object in the client can also be
        deleted."""
        self.cubit_connect.send_and_return(["delete", self.cubit_id])

    def __str__(self):
        """Return the string from the client."""
        return '<CubitObject>"' + self.cubit_id[1] + '"'

    def isinstance(self, geom_type):
        """Check if this object is of geom_type.

        Args
        ----
        geom_type: str
            Name of the geometry to compare (vertex, curve, surface, volume).
        """

        # Compare in client python interpreter.
        return self.cubit_connect.send_and_return(
            ["isinstance", self.cubit_id, geom_type]
        )

    def get_self_dir(self):
        """Return a list of all cubit child items of this object.

        Also return a flag if the child item is callable or not.
        """
        return self.cubit_connect.send_and_return(["get_self_dir", self.cubit_id])

    def get_methods(self):
        """Return a list of all callable cubit methods for this object."""
        return [method for method, callable in self.get_self_dir() if callable]

    def get_attributes(self):
        """Return a list of all non callable cubit methods for this object."""
        return [method for method, callable in self.get_self_dir() if not callable]

    def get_geometry_type(self):
        """Return the type of this item."""

        if self.isinstance("cubitpy_vertex"):
            return cupy.geometry.vertex
        elif self.isinstance("cubitpy_curve"):
            return cupy.geometry.curve
        elif self.isinstance("cubitpy_surface"):
            return cupy.geometry.surface
        elif self.isinstance("cubitpy_volume"):
            return cupy.geometry.volume

        # Default value -> not a valid geometry
        raise TypeError("The item is not a valid geometry!")

    def get_node_ids(self):
        """Return a list with the node IDs (index 1) of this object.

        This is done by creating a temporary node set that this geometry
        is added to. It is not possible to get the node list directly
        from cubit.
        """

        # Get a node set ID that is not yet taken
        node_set_ids = [0]
        node_set_ids.extend(self.cubit_connect.cubit.get_nodeset_id_list())
        temp_node_set_id = max(node_set_ids) + 1

        # Add a temporary node set with this geometry
        self.cubit_connect.cubit.cmd(
            "nodeset {} {} {}".format(
                temp_node_set_id, self.get_geometry_type().get_cubit_string(), self.id()
            )
        )

        # Get the nodes in the created node set
        node_ids = self.cubit_connect.cubit.get_nodeset_nodes_inclusive(
            temp_node_set_id
        )

        # Delete the temp node set and return the node list
        self.cubit_connect.cubit.cmd("delete nodeset {}".format(temp_node_set_id))
        return node_ids


class CubitObjectMain(CubitObject):
    """The main cubit object will be of this type, it can not delete itself."""

    def __del__(self):
        """Overwrite the default, because we don't want to delete any objects
        on the client if this main object is deleted."""
        pass
