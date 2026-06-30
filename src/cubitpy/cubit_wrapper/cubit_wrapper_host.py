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
"""This module creates object that are used to connect between the cubit python
interpreter and the main python interpreter."""

import atexit
import warnings
from pathlib import Path

import execnet
import numpy as np

from cubitpy.conf import CubitPyWarning, GeometryType, cupy
from cubitpy.cubit_wrapper.cubit_wrapper_utility import cubit_item_to_id, is_base_type


class CubitConnect(object):
    """This class holds a connection to a cubit python interpreter and
    initializes cubit there.

    It is possible to send function calls to that interpreter and
    receive the output.
    """

    def __init__(self):
        """Initialize the connection between the client (cubit) python
        interpreter and this one."""

        # Set up the gateway to the client python interpreter
        if cupy.is_remote():
            interpreter = f"ssh={cupy.get_remote_user()}@{cupy.get_remote_host()}//python={cupy.get_cubit_python_interpreter()}"
        else:
            interpreter = f"popen//python={cupy.get_cubit_python_interpreter()}"

        # Get the path where the cubit python libs are located.
        cubit_lib = cupy.get_cubit_lib_path()

        # Set up the client python interpreter
        self.gw = execnet.makegateway(interpreter)
        self.gw.reconfigure(py3str_as_py2str=True)

        # Get the code to be executed in the client interpreter. This code also has to
        # contain the utility functions.
        path_client_utils = Path(__file__).parent / "cubit_wrapper_utility.py"
        path_client_code = Path(__file__).parent / "cubit_wrapper_client.py"
        client_code = (
            path_client_utils.read_text() + "\n" + path_client_code.read_text()
        )

        # Set up the connection channel
        self.channel = self.gw.remote_exec(client_code)

        # Arguments for cubit
        init_arguments = [
            "cubit",
            "-information",  # Do not output information of cubit
            "Off",
            "-nojournal",  # Do write a journal file
            "-noecho",  # Do not output commands used in cubit
        ]

        # Parameters for initialization of the client interpreter.
        parameters = {
            "additional_sys_paths": [str(cubit_lib)],
            "is_remote": cupy.is_remote(),
            "init_arguments": init_arguments,
        }

        # Initialize cubit in the client and create the linking object here
        cubit_id = self.send_and_return(parameters)
        if cubit_id is None:
            raise RuntimeError(
                "Could not initialize cubit in the client! "
                "Likely due to a missing license."
            )
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

        def get_log_string(log_lines: list[str], name: str) -> str:
            """Get a string from the log lines."""
            text = f"The command\n    {argument_list}\nraised the following {name}:"
            return "\n".join([text] + log_lines)

        try:
            self.channel.send(argument_list)
            return_value = self.channel.receive()
            if isinstance(return_value, dict):
                if len(return_value["messages"]) > 0:
                    warnings.warn(
                        get_log_string(return_value["messages"], "message(s)"),
                        category=CubitPyWarning,
                        stacklevel=3,
                    )
                if len(return_value["errors"]) > 0:
                    error_string = get_log_string(return_value["errors"], "error(s)")
                    if cupy.on_cubit_error == "raise":
                        raise RuntimeError(error_string)
                    elif cupy.on_cubit_error == "warn":
                        warnings.warn(
                            error_string,
                            category=CubitPyWarning,
                            stacklevel=3,
                        )
                    # "ignore": silently continue
                return return_value["return_value"]
            else:
                return return_value
        except OSError as e:
            if "cannot send" in str(e):
                # If the channel is already finalized we get this error here. This
                # happens in cases where we delete items after the connection has been
                # closed.
                return None
            raise

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

                if isinstance(item, tuple) or isinstance(item, list):
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
                elif isinstance(item, np.ndarray):
                    return item.tolist()
                else:
                    return item

            # Check if there are cubit objects in the arguments
            arguments = serialize_item(args)

            # Call the method on the cubit object
            cubit_return = self.send_and_return(
                [cubit_object.cubit_id, name, arguments]
            )

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

    def __init__(self, cubit_connect: CubitConnect, cubit_data_list: list):
        """Initialize the object.

        Args:
            cubit_connect:
                A link to the cubit_connect object that will be used to call
                methods.
            cubit_data_list:
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

    def __eq__(self, other):
        """Compare two cubit objects based on their type and IDs."""
        if not isinstance(other, CubitObject):
            return False
        if self.get_object_type() == other.get_object_type():
            if self.id() == other.id():
                return True
        return False

    def __hash__(self):
        """Return a hash based on the same properties used for equality.

        This allows CubitObject instances to be used in sets and as
        dictionary keys while remaining consistent with __eq__.
        """
        return hash((self.get_object_type(), self.id()))

    def __del__(self):
        """When this object is deleted, the object in the client can also be
        deleted."""
        self.cubit_connect.send_and_return(["delete", self.cubit_id])

    def __str__(self):
        """Return the string from the client."""
        return '<CubitObject>"' + self.cubit_id[1] + '"'

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

    def get_object_type(self):
        """Return the type of this object."""
        string_representation = self.cubit_connect.send_and_return(
            ["get_object_type", self.cubit_id]
        )
        if string_representation is None:
            raise TypeError("Could not get object type for {}".format(self.cubit_id))
        mapping = {
            "cubitpy_vertex": cupy.geometry.vertex,
            "cubitpy_curve": cupy.geometry.curve,
            "cubitpy_surface": cupy.geometry.surface,
            "cubitpy_volume": cupy.geometry.volume,
            "cubitpy_body": "body",
        }
        if string_representation not in mapping:
            raise TypeError(
                "Unknown object type {} for {}".format(
                    string_representation, self.cubit_id
                )
            )
        return mapping[string_representation]

    def get_geometry_type(self) -> GeometryType:
        """Return the type of this item."""

        object_type = self.get_object_type()
        if object_type == "body":
            raise TypeError("The item is a body, not a pure geometry!")
        else:
            return object_type

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
