# -*- coding: utf-8 -*-
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
"""
This script is used to test that all headers in the repository are correct.

This file is adapted from LaTeX2AI (https://github.com/stoani89/LaTeX2AI).
"""

# Import python modules.
import os
import subprocess


def get_repository_dir():
    """
    Get the root directory of this repository.
    """

    script_path = os.path.realpath(__file__)
    root_dir = os.path.dirname(os.path.dirname(script_path))
    return root_dir


def get_license_text():
    """
    Return the license text as a string.
    """

    license_path = os.path.join(get_repository_dir(), "LICENSE")
    with open(license_path) as license_file:
        return license_file.read().strip()


def get_all_source_files():
    """
    Get all source files that should be checked for license headers.
    """

    # Get the files in the git repository.
    repo_dir = get_repository_dir()
    process = subprocess.Popen(
        ["git", "ls-files"], stdout=subprocess.PIPE, cwd=repo_dir
    )
    out, _err = process.communicate()
    files = out.decode("UTF-8").strip().split("\n")

    source_line_endings = [".py", ".pyx"]
    source_ending_types = {".py": "py", ".pyx": "py"}
    source_files = {"py": []}
    for file in files:
        extension = os.path.splitext(file)[1]
        if extension not in source_line_endings:
            pass
        else:
            source_files[source_ending_types[extension]].append(
                os.path.join(repo_dir, file)
            )
    return source_files


def license_to_source(license_text, source_type):
    """
    Convert the license text to a text that can be written to source code.
    """

    header = []
    start_line = "-" * 77
    if source_type == "py":
        header = ["# -*- coding: utf-8 -*-"]
        comment = "#"
    else:
        raise ValueError("Wrong extension!")

    source = []
    source.append(comment + " " + start_line)
    for line in license_text.split("\n"):
        if len(line) > 0:
            source.append(comment + " " + line)
        else:
            source.append(comment + line)
    source.append(comment + " " + start_line)
    return source, header


def check_license():
    """
    Check the license for all source files.
    """

    license_text = get_license_text()
    source_files = get_all_source_files()

    skip_list = []
    wrong_headers = []

    for key in source_files:
        source, header = license_to_source(license_text, key)
        license = "\n".join(source)
        license_plus_header = "\n".join(header + source)
        for file in source_files[key]:
            for skip in skip_list:
                if file.endswith(skip):
                    break
            else:
                with open(file) as source_file:
                    source_text = source_file.read()
                    if not source_text.startswith(
                        license_plus_header
                    ) and not source_text.startswith(license):
                        wrong_headers.append(file)

    return wrong_headers


def test_headers():
    """Check if all headers are correct."""

    wrong_headers = check_license()
    wrong_headers_string = "Wrong headers in: " + ", ".join(wrong_headers)
    assert len(wrong_headers) == 0, wrong_headers_string
