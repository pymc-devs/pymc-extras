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
import logging

from importlib.metadata import version

from pymc_extras import gp, statespace, utils
from pymc_extras.distributions import *
from pymc_extras.inference import find_MAP, fit, fit_laplace, fit_pathfinder
from pymc_extras.model.marginal.marginal_model import (
    MarginalModel,
    marginalize,
    recover_marginals,
)
from pymc_extras.model.model_api import as_model

_log = logging.getLogger("pmx")

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)


__version__ = version("pymc-extras")
