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

"""``ppf_bounds_cont`` vendored verbatim from pymc-devs/pytensor-distributions.

https://github.com/pymc-devs/pytensor-distributions/blob/13ab4708a5fce7b4f73bd35014c73ca7a8667d6a/pytensor_distributions/helper.py
"""

import pytensor.tensor as pt


def ppf_bounds_cont(x_val, q, lower, upper):
    """
    Apply bounds checking for the inverse CDF of continuous distributions.

    Parameters
    ----------
    x_val : tensor
        The computed PPF value
    q : tensor
        Probability value (quantile) between 0 and 1
    lower : float
        Lower bound of the distribution support
    upper : float
        Upper bound of the distribution support

    Returns
    -------
    tensor
        PPF value with proper bounds: NaN for q outside [0,1],
        lower bound for q=0, upper bound for q=1, otherwise x_val
    """
    return pt.switch(
        pt.or_(pt.lt(q, 0), pt.gt(q, 1)),
        pt.nan,
        pt.switch(pt.eq(q, 0), lower, pt.switch(pt.eq(q, 1), upper, x_val)),
    )
