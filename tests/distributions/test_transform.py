#   Copyright 2025 The PyMC Developers
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
import numpy as np
import pymc as pm

from pymc_extras.distributions.transforms import PartialOrder


class TestPartialOrder:
    adj_mats = np.array(
        [
            # 0 < {1, 2} < 3
            [[0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]],
            # 1 < 0 < 3 < 2
            [[0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]],
        ]
    )

    valid_values = np.array([[0, 2, 1, 3], [1, 0, 3, 2]], dtype=float)

    # Test that forward and backward are inverses of eachother
    # And that it works when extra dimensions are added in data
    def test_forward_backward_dimensionality(self):
        po = PartialOrder(self.adj_mats)
        po0 = PartialOrder(self.adj_mats[0])
        vv = self.valid_values
        vv0 = self.valid_values[0]

        testsets = [
            (vv, po),
            (po.initvals(), po),
            (vv0, po0),
            (po0.initvals(), po0),
            (np.tile(vv0, (2, 1)), po0),
            (np.tile(vv0, (2, 3, 2, 1)), po0),
            (np.tile(vv, (2, 3, 2, 1, 1)), po),
        ]

        for vv, po in testsets:
            fw = po.forward(vv)
            bw = po.backward(fw)
            np.testing.assert_allclose(bw.eval(), vv)

    def test_sample_model(self):
        po = PartialOrder(self.adj_mats)
        with pm.Model() as model:
            x = pm.Normal(
                "x",
                size=(3, 2, 4),
                transform=po,
                initval=po.initvals(shape=(3, 2, 4), lower=-1, upper=1),
            )
            idata = pm.sample()

        # Check that the order constraints are satisfied
        # Move chain, draw and "3" dimensions to the back
        xvs = idata.posterior.x.values.transpose(3, 4, 0, 1, 2)
        x0 = xvs[0]  # 0 < {1, 2} < 3
        assert (
            (x0[0] < x0[1]).all()
            and (x0[0] < x0[2]).all()
            and (x0[1] < x0[3]).all()
            and (x0[2] < x0[3]).all()
        )
        x1 = xvs[1]  # 1 < 0 < 3 < 2
        assert (x1[1] < x1[0]).all() and (x1[0] < x1[3]).all() and (x1[3] < x1[2]).all()
