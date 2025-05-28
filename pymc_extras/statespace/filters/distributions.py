import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from pymc import intX
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.distribution import Continuous, SymbolicRandomVariable
from pymc.distributions.shape_utils import get_support_shape_1d
from pymc.logprob.abstract import _logprob
from pytensor.graph.basic import Node

floatX = pytensor.config.floatX
COV_ZERO_TOL = 0

lgss_shape_message = (
    "The LinearGaussianStateSpace distribution needs shape information to be constructed. "
    "Ensure that all input matrices have shape information specified."
)


def make_signature(sequence_names):
    states = "s"
    obs = "p"
    exog = "r"
    time = "t"
    state_and_obs = "n"

    matrix_to_shape = {
        "x0": (states,),
        "P0": (states, states),
        "c": (states,),
        "d": (obs,),
        "T": (states, states),
        "Z": (obs, states),
        "R": (states, exog),
        "H": (obs, obs),
        "Q": (exog, exog),
    }

    for matrix in sequence_names:
        base_shape = matrix_to_shape[matrix]
        matrix_to_shape[matrix] = (time, *base_shape)

    signature = ",".join(["(" + ",".join(shapes) + ")" for shapes in matrix_to_shape.values()])

    return f"{signature},[rng]->[rng],({time},{state_and_obs})"


class LinearGaussianStateSpaceRV(SymbolicRandomVariable):
    default_output = 1
    _print_name = ("LinearGuassianStateSpace", "\\operatorname{LinearGuassianStateSpace}")

    def update(self, node: Node):
        return {node.inputs[-1]: node.outputs[0]}


class _LinearGaussianStateSpace(Continuous):
    def __new__(
        cls,
        name,
        a0,
        P0,
        c,
        d,
        T,
        Z,
        R,
        H,
        Q,
        steps=None,
        sequence_names=None,
        append_x0=True,
        method="svd",
        **kwargs,
    ):
        # Ignore dims in support shape because they are just passed along to the "observed" and "latent" distributions
        # created by LinearGaussianStateSpace. This "combined" distribution shouldn't ever be directly used.
        steps = get_support_shape_1d(
            support_shape=steps,
            shape=None,
            dims=None,
            observed=kwargs.get("observed", None),
            support_shape_offset=0,
        )

        return super().__new__(
            cls,
            name,
            a0,
            P0,
            c,
            d,
            T,
            Z,
            R,
            H,
            Q,
            steps=steps,
            sequence_names=sequence_names,
            append_x0=append_x0,
            method=method,
            **kwargs,
        )

    @classmethod
    def dist(
        cls,
        a0,
        P0,
        c,
        d,
        T,
        Z,
        R,
        H,
        Q,
        steps=None,
        sequence_names=None,
        append_x0=True,
        method="svd",
        **kwargs,
    ):
        steps = get_support_shape_1d(
            support_shape=steps, shape=kwargs.get("shape", None), support_shape_offset=0
        )

        if steps is None:
            raise ValueError("Must specify steps or shape parameter")

        steps = pt.as_tensor_variable(intX(steps), ndim=0)

        return super().dist(
            [a0, P0, c, d, T, Z, R, H, Q, steps],
            sequence_names=sequence_names,
            append_x0=append_x0,
            method=method,
            **kwargs,
        )

    @classmethod
    def rv_op(
        cls,
        a0,
        P0,
        c,
        d,
        T,
        Z,
        R,
        H,
        Q,
        steps,
        size=None,
        sequence_names=None,
        append_x0=True,
        method="svd",
    ):
        if sequence_names is None:
            sequence_names = []

        a0_, P0_, c_, d_, T_, Z_, R_, H_, Q_ = map(
            lambda x: x.type(), [a0, P0, c, d, T, Z, R, H, Q]
        )

        c_.name = "c"
        d_.name = "d"
        T_.name = "T"
        Z_.name = "Z"
        R_.name = "R"
        H_.name = "H"
        Q_.name = "Q"

        sequences = [
            x
            for x, name in zip([c_, d_, T_, Z_, R_, H_, Q_], ["c", "d", "T", "Z", "R", "H", "Q"])
            if name in sequence_names
        ]
        non_sequences = [x for x in [c_, d_, T_, Z_, R_, H_, Q_] if x not in sequences]

        rng = pytensor.shared(np.random.default_rng())

        def sort_args(args):
            sorted_args = []

            # Inside the scan, outputs_info variables get a time step appended to their name
            # e.g. x -> x[t]. Remove this so we can identify variables by name.
            arg_names = [x.name.replace("[t]", "") for x in args]

            # c, d ,T, Z, R, H, Q is the "canonical" ordering
            for name in ["c", "d", "T", "Z", "R", "H", "Q"]:
                idx = arg_names.index(name)
                sorted_args.append(args[idx])

            return sorted_args

        n_seq = len(sequence_names)

        def step_fn(*args):
            seqs, state, non_seqs = args[:n_seq], args[n_seq], args[n_seq + 1 :]
            non_seqs, rng = non_seqs[:-1], non_seqs[-1]

            c, d, T, Z, R, H, Q = sort_args(seqs + non_seqs)
            k = T.shape[0]
            a = state[:k]

            middle_rng, a_innovation = pm.MvNormal.dist(
                mu=0, cov=Q, rng=rng, method=method
            ).owner.outputs
            next_rng, y_innovation = pm.MvNormal.dist(
                mu=0, cov=H, rng=middle_rng, method=method
            ).owner.outputs

            a_mu = c + T @ a
            a_next = a_mu + R @ a_innovation

            y_mu = d + Z @ a_next
            y_next = y_mu + y_innovation

            next_state = pt.concatenate([a_next, y_next], axis=0)

            return next_state, {rng: next_rng}

        Z_init = Z_ if Z_ in non_sequences else Z_[0]
        H_init = H_ if H_ in non_sequences else H_[0]

        init_x_ = pm.MvNormal.dist(a0_, P0_, rng=rng, method=method)
        init_y_ = pm.MvNormal.dist(Z_init @ init_x_, H_init, rng=rng, method=method)

        init_dist_ = pt.concatenate([init_x_, init_y_], axis=0)

        statespace, updates = pytensor.scan(
            step_fn,
            outputs_info=[init_dist_],
            sequences=None if len(sequences) == 0 else sequences,
            non_sequences=[*non_sequences, rng],
            n_steps=steps,
            strict=True,
        )

        if append_x0:
            statespace_ = pt.concatenate([init_dist_[None], statespace], axis=0)
            statespace_ = pt.specify_shape(statespace_, (steps + 1, None))
        else:
            statespace_ = statespace
            statespace_ = pt.specify_shape(statespace_, (steps, None))

        (ss_rng,) = tuple(updates.values())
        linear_gaussian_ss_op = LinearGaussianStateSpaceRV(
            inputs=[a0_, P0_, c_, d_, T_, Z_, R_, H_, Q_, steps, rng],
            outputs=[ss_rng, statespace_],
            extended_signature=make_signature(sequence_names),
        )

        linear_gaussian_ss = linear_gaussian_ss_op(a0, P0, c, d, T, Z, R, H, Q, steps, rng)
        return linear_gaussian_ss


class LinearGaussianStateSpace(Continuous):
    """
    Linear Gaussian Statespace distribution

    """

    def __new__(
        cls,
        name,
        a0,
        P0,
        c,
        d,
        T,
        Z,
        R,
        H,
        Q,
        *,
        steps,
        k_endog=None,
        sequence_names=None,
        append_x0=True,
        method="svd",
        **kwargs,
    ):
        dims = kwargs.pop("dims", None)
        latent_dims = None
        obs_dims = None
        if dims is not None:
            if len(dims) != 3:
                ValueError(
                    "LinearGaussianStateSpace expects 3 dims: time, all_states, and observed_states"
                )
            time_dim, state_dim, obs_dim = dims
            latent_dims = [time_dim, state_dim]
            obs_dims = [time_dim, obs_dim]

        latent_obs_combined = _LinearGaussianStateSpace(
            f"{name}_combined",
            a0,
            P0,
            c,
            d,
            T,
            Z,
            R,
            H,
            Q,
            steps=steps,
            sequence_names=sequence_names,
            append_x0=append_x0,
            method=method,
            **kwargs,
        )
        latent_obs_combined = pt.specify_shape(latent_obs_combined, (steps + int(append_x0), None))
        if k_endog is None:
            k_endog = cls._get_k_endog(H)
        latent_slice = slice(None, -k_endog)
        obs_slice = slice(-k_endog, None)

        latent_states = latent_obs_combined[..., latent_slice]
        obs_states = latent_obs_combined[..., obs_slice]

        latent_states = pm.Deterministic(f"{name}_latent", latent_states, dims=latent_dims)
        obs_states = pm.Deterministic(f"{name}_observed", obs_states, dims=obs_dims)

        return latent_states, obs_states

    @classmethod
    def dist(cls, a0, P0, c, d, T, Z, R, H, Q, *, steps=None, **kwargs):
        latent_obs_combined = _LinearGaussianStateSpace.dist(
            a0, P0, c, d, T, Z, R, H, Q, steps=steps, **kwargs
        )
        k_states = T.type.shape[0]

        latent_states = latent_obs_combined[..., :k_states]
        obs_states = latent_obs_combined[..., k_states:]

        return latent_states, obs_states

    @classmethod
    def _get_k_states(cls, T):
        k_states = T.type.shape[0]
        if k_states is None:
            raise ValueError(lgss_shape_message)
        return k_states

    @classmethod
    def _get_k_endog(cls, H):
        k_endog = H.type.shape[0]
        if k_endog is None:
            raise ValueError(lgss_shape_message)

        return k_endog


class KalmanFilterRV(SymbolicRandomVariable):
    default_output = 1
    _print_name = ("KalmanFilter", "\\operatorname{KalmanFilter}")
    extended_signature = "(t,s),(t,s,s),(t),[rng]->[rng],(t,s)"

    def update(self, node: Node):
        return {node.inputs[-1]: node.outputs[0]}


class SequenceMvNormal(Continuous):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    @classmethod
    def dist(cls, mus, covs, logp, method="svd", **kwargs):
        return super().dist([mus, covs, logp], method=method, **kwargs)

    @classmethod
    def rv_op(cls, mus, covs, logp, method="svd", size=None):
        # Batch dimensions (if any) will be on the far left, but scan requires time to be there instead
        if mus.ndim > 2:
            mus = pt.moveaxis(mus, -2, 0)
        if covs.ndim > 3:
            covs = pt.moveaxis(covs, -3, 0)

        mus_, covs_ = mus.type(), covs.type()

        logp_ = logp.type()
        rng = pytensor.shared(np.random.default_rng())

        def step(mu, cov, rng):
            new_rng, mvn = pm.MvNormal.dist(mu=mu, cov=cov, rng=rng, method=method).owner.outputs
            return mvn, {rng: new_rng}

        mvn_seq, updates = pytensor.scan(
            step, sequences=[mus_, covs_], non_sequences=[rng], strict=True, n_steps=mus_.shape[0]
        )
        mvn_seq = pt.specify_shape(mvn_seq, mus.type.shape)

        # Move time axis back to position -2 so batches are on the left
        if mvn_seq.ndim > 2:
            mvn_seq = pt.moveaxis(mvn_seq, 0, -2)

        (seq_mvn_rng,) = tuple(updates.values())

        mvn_seq_op = KalmanFilterRV(
            inputs=[mus_, covs_, logp_, rng], outputs=[seq_mvn_rng, mvn_seq], ndim_supp=2
        )

        mvn_seq = mvn_seq_op(mus, covs, logp, rng)
        return mvn_seq


@_logprob.register(KalmanFilterRV)
def sequence_mvnormal_logp(op, values, mus, covs, logp, rng, **kwargs):
    return check_parameters(
        logp,
        pt.eq(values[0].shape[0], mus.shape[0]),
        pt.eq(covs.shape[0], mus.shape[0]),
        msg="Observed data and parameters must have the same number of timesteps (dimension 0)",
    )
