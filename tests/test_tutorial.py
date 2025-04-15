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
"""This script is used to test the tutorial."""

import os
import unittest

from test_cubitpy import compare, testing_temp

from cubitpy import CubitPy
from tutorial.tutorial import cubit_step_by_step_tutorial_cli


class TestTutorial(unittest.TestCase):
    """This class tests the tutorials in the repository."""

    def test_tutorial(self):
        """Test that the tutorial works."""
        cubit = CubitPy()
        tutorial_file = os.path.join(testing_temp, "tutorial.dat")
        cubit_step_by_step_tutorial_cli(
            tutorial_file, display=False, cubit=cubit, size=5.0
        )
        compare(cubit, name="test_cubit_tutorial")


if __name__ == "__main__":
    # Execution part of script.
    unittest.main()
