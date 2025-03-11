import pymc as pm
from pymc.step_methods.metropolis import Metropolis
from pymc.step_methods.slicer import Slice
from pymc.blocking import DictToArrayBijection

import collections
import itertools
import functools as f
import numpy as np
import random
import re


class Jump(pm.step_methods.arraystep.BlockedStep):
    """
    Placeholder class for jumps

    The following methods must be overriden by the inheriting class
    - get_vars_from_src_dest
    - get_deltas_updater
    - get_subspace_updater
    - get_log_acceptance_fraction_calculator
    """

    # TODO figure out how to generate and gather stats properly
    # generates_stats = True
    # stats_dtypes = [
    #     {
    #         "accept": np.float64
    #         ,"accepted": bool
    #         ,"diverged": bool
    #         , "tune": bool
    #     }
    # ]

    @classmethod
    def get_vars_from_src_dest(cls, delta_parameters, configurations_to_subspaces, src, dest):
        "get the pymc RVs this jump will affect"

        # Get deltas being changed
        vars = {d for i,d in enumerate(delta_parameters) if src[i] != dest[i]}

        # Get the new ks to be set (only the destination subspace matters)
        vars = vars.union(set(configurations_to_subspaces[dest]))

        return list(vars)

    def get_deltas_updater(self):
        """Return method operating on PointType to update the label parameters"""
        raise (NotImplementedError())

    def get_subspace_updater(self):
        """Return method operating on PointType to update the subspace continous parameters (point proposal)"""
        raise (NotImplementedError())

    def get_log_acceptance_fraction_calculator(self):
        """Returns log(acceptance fraction) function f(new_point, point)"""
        raise (NotImplementedError())

    @classmethod
    def get_forward_transform(cls, rv):
        """Returns a function which performs the forward value_var transform on a scalar"""

        def f(x):
            return rv.tag.value_var.tag.transform.forward(x, *rv.owner.inputs).eval()

        return f

    @classmethod
    def get_backward_transform(cls, rv):
        """Returns a function which performs the backward value_var transform on a scalar"""

        def f(x):
            return rv.tag.value_var.tag.transform.backward(x, *rv.owner.inputs).eval()

        return f

    def transform_log_jac_det(self, name, value):
        rv = self.model.named_vars[name]
        return rv.tag.value_var.tag.transform.log_jac_det(
            value, *rv.owner.inputs
        ).eval()

    @classmethod
    def get_value_var_from_rv(self, rv):
        return rv.tag.value_var

    def __new__(cls, delta_parameters, configurations_to_subspaces, src, dest, *args, **kwargs):

        # These have to be appended here because BlockedStep does some stuff with it in __new__
        kwargs["vars"] = cls.get_vars_from_src_dest(delta_parameters, configurations_to_subspaces, src, dest)

        return super().__new__(cls, *args, **kwargs)

    def __init__(self, delta_parameters, configurations_to_subspaces, src, dest, p_src_dest, p_dest_src, *args, **kwargs):
        """
        Creates a very simple Jump
        between src and dest configurations with p_src_dest, p_dest_src being the probabilities that this was selected to begin with
        """
        self.src = src
        self.dest = dest
        self.delta_parameters = delta_parameters
        self.configurations_to_subspaces = configurations_to_subspaces

        # Precompute the prior bias in having selected this move type
        self.p_src_dest = p_src_dest
        self.p_dest_src = p_dest_src
        self.diff_logj = np.log(self.p_dest_src) - np.log(self.p_src_dest)

        self.model = pm.modelcontext(None)

        # Generate the dlogp function
        self.rvs = self.get_vars_from_src_dest(delta_parameters, configurations_to_subspaces, src, dest)
        self.vars = pm.inputvars(
            [self.model.rvs_to_values.get(var, var) for var in self.rvs]
        )
        self.logp = self.model.compile_logp()

        # Precompute delta flips
        self.delta_updater = self.get_deltas_updater()

        # Precompute setting ks
        self.set_new_ks = self.get_subspace_updater()

        # Retrieve log_acceptance_fraction_calculator
        self.calculate_log_acceptance_fraction = (
            self.get_log_acceptance_fraction_calculator()
        )

    def step(self, point):
        """
        point is a dictionary of {str(value_var): array(x)}
        the transformation on the varlue variables is reversed for some of my computations
        and then applied again
        """
        new_point = {x: y for x, y in point.items()}

        # stats = {}

        # discrete parameter set
        self.delta_updater(new_point)

        # Continous parameter set
        extra_data = self.set_new_ks(new_point)

        # The general RJMCMC kernel will have to provide this object with it
        if extra_data is None:
            log_acceptance_fraction = self.calculate_log_acceptance_fraction(
                new_point, point
            )
        else:
            log_acceptance_fraction = self.calculate_log_acceptance_fraction(
                new_point, point, **extra_data
            )

        if not np.isfinite(log_acceptance_fraction):
            # raise(ValueError())
            print("diverged TODO put this in stats")

        # Check for acceptance
        # stats['diverged'] = not np.isfinite(log_acceptance_fraction)
        # stats['accept'] = log_acceptance_fraction

        # If np.isfinite fails then we just consider we're out of bounds (diverged)
        if np.isfinite(log_acceptance_fraction) and np.log(np.random.random()) < min(
            0, log_acceptance_fraction
        ):
            # stats['accepted'] = True
            # return new_point, [stats]
            return new_point
        else:
            # stats['accepted'] = False
            # return point, [stats]
            return point


class RJMCMC:
    """
    Largely based on the structure of CompoundStep
    """

    def __init__(
        self,
        delta_variables,
        configurations_to_subpaces,
        jumps,
        jump_probabilities,
        p_jump=0.2,
        tune=True,
        n_tune=0,
    ):
        """
        Arguments:
            - delta_variables: [delta_1, ..., delta_m] ordered collection of marker variables
            - configurations_to_subspaces: {(0/1, ...): {theta_1, ..., theta_n}} mapping between configuration numbers and model variables
                                            ex: (1, 0) -> delta_1 = 1, delta_2 = 0 (The order is as specified in the delta_variables parameter)
            - jumps: {(x_1, x_2, ...): {(y_1, y_2, ...): step_function }} double dictionary refering to the step function that maps between the configuration spaces
            - jump_probabilities: {(x_1, ...): {(y_1, ...): p_x_y}} must satify the condition sum(probas[x] over y) == 1 for all x

            - p_jump: probability to select a jump over staying in the same space
            - n_tune: number of samples to tune for. The sampler will then
        """
        self.delta_variables = delta_variables
        self.configurations_to_subspaces = configurations_to_subpaces
        self.jumps = jumps
        self.jump_probabilities = jump_probabilities
        self.p_jump = p_jump

        # Create the intra subspace stepper functions
        default_intraspace_sampler = pm.NUTS
        # default_intraspace_sampler = pm.step_methods.Slice
        # default_intraspace_sampler = pm.step_methods.Metropolis
        self.intraspace_steppers = collections.OrderedDict(
            {
                config: default_intraspace_sampler(list(subspace))
                for config, subspace in configurations_to_subpaces.items()
            }
        )

        # We need to refer to all the steppers that in the collections
        self.methods = [x for y in self.jumps.values() for x in y.values()] + [
            x for x in self.intraspace_steppers.values()
        ]

        # We tune each of the intraspace steppers for
        # TODO figure out how to get all the necessary data and communicate with pm.sample
        # Or do we just need to write out own pm.sample
        self.tune = tune
        self.n_tune = n_tune  # Number of tuning steps to take in total
        self.tuning_stepper_iterator = self.generate_method_tuning_iterator()

        # Determine if we generate states (from CompoundStep)
        # TODO figure out the best way to handle the eclectic stat types since different jumps are called at each sample (and not each time in order like in CompoundStep)
        self.generates_stats = any(method.generates_stats for method in self.methods)
        self.stats_dtypes = []
        for method in self.methods:
            if method.generates_stats:
                self.stats_dtypes.extend(method.stats_dtypes)

    def generate_method_tuning_iterator(self):
        """Return iterator from which to sample the next stepper method during the tuning phase"""
        return itertools.chain.from_iterable(
            itertools.repeat(x, int(self.n_tune / len(self.intraspace_steppers)))
            for x in self.intraspace_steppers.values()
        )

    def step(self, point):

        # TODO figure out how this thing should be tuned
        jumping_probability = self.p_jump
        # We need to randomely select a move type and (intra or inter)
        # Then proceed to simply stepping in that space
        # Since spending a little bit more time in each model is probably better than zig zagging
        # we'll just have a bias in that direction

        # Figure out the value of the current configuration
        current_config = tuple(int(point[str(x)]) for x in self.delta_variables)

        # As long as we're tuning we just allow the subspace steppers to run and do their thing
        if self.tune:
            try:
                next_method = next(self.tuning_stepper_iterator)
            except StopIteration:
                self.stop_tuning()
                print("Done tuning by running out of iterator")
            else:
                method = next_method
                # print('tuning {}'.format(method))

        if not self.tune:
            if np.random.random() < jumping_probability:
                # jump
                # randomly select a new space to jump to
                # TODO precompute these arrays somewhere so we don't have to do so much looping
                choices = list(self.jumps[current_config].keys())
                choice_weights = [
                    self.jump_probabilities[current_config][destination]
                    for destination in choices
                ]

                next_space = random.choices(choices, weights=choice_weights)[0]
                method = self.jumps[current_config][next_space]
            else:
                # stay in current subspace
                method = self.intraspace_steppers[current_config]

        if self.generates_stats and method.generates_stats:
            point, state = method.step(point)
            return point, state
        else:
            point = method.step(point)
            return point, []

    def warnings(self):
        """From CompoundStep"""
        warns = []
        for method in self.methods:
            if hasattr(method, "warnings"):
                warns.extend(method.warnings())
        return warns

    def stop_tuning(self):
        """From CompoundStep"""
        self.tune = False
        for method in self.methods:
            method.stop_tuning()

    def reset_tuning(self):
        """From CompoundStep"""
        self.tune = True
        self.tuning_stepper_iterator = self.generate_method_tuning_iterator()
        for method in self.methods:
            if hasattr(method, "reset_tuning"):
                method.reset_tuning()

    @property
    def vars(self):
        # TODO check if this needs to be properly ordered or something for some sort of guarantee
        return list({var for method in self.methods for var in method.vars})
