#   Copyright 2022 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import os

from codecs import open

from setuptools import find_packages, setup


def read_version():
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, "pymc_extras", "version.txt")) as f:
        version = f.read().strip()
    return version


if __name__ == "__main__":
    setup(
        name="pymc-extras",
        version=read_version(),
        packages=find_packages(exclude=["tests*"]),
    )
