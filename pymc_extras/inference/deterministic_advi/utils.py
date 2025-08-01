from time import time


def opt_callback_fun(theta, val_and_grad_fun, hvp_fun):
    # Callback for jax_advi. It records only what will be needed
    # to estimate the KL later on.

    opt_callback_fun.opt_sequence.append(
        {
            "val_and_grad_calls": val_and_grad_fun.calls,
            "hvp_calls": hvp_fun.calls,
            "theta": theta,
            "time": time(),
        }
    )


opt_callback_fun.opt_sequence = list()
