import numpy as np
import pymc as pm

from rich.console import Console

from pymc_extras.printing import model_table


def get_text(table) -> str:
    console = Console(width=80)
    with console.capture() as capture:
        console.print(table)
    return capture.get()


def test_model_table():
    with pm.Model(coords={"trial": range(6), "subject": range(20)}) as model:
        x_data = pm.Data("x_data", np.random.normal(size=(6, 20)), dims=("trial", "subject"))
        y_data = pm.Data("y_data", np.random.normal(size=(6, 20)), dims=("trial", "subject"))

        mu = pm.Normal("mu", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)
        global_intercept = pm.Normal("global_intercept", mu=0, sigma=1)
        intercept_subject = pm.Normal("intercept_subject", mu=0, sigma=1, shape=(20, 1))
        beta_subject = pm.Normal("beta_subject", mu=mu, sigma=sigma, dims="subject")

        mu_trial = pm.Deterministic(
            "mu_trial",
            global_intercept.squeeze() + intercept_subject + beta_subject * x_data,
            dims=["trial", "subject"],
        )
        noise = pm.Exponential("noise", lam=1)
        y = pm.Normal("y", mu=mu_trial, sigma=noise, observed=y_data, dims=("trial", "subject"))

        pm.Potential("beta_subject_penalty", -pm.math.abs(beta_subject), dims="subject")

    table_txt = get_text(model_table(model))
    expected = """               Variable  Expression                      Dimensions
────────────────────────────────────────────────────────────────────────────────
               x_data =  Data                            trial[6] × subject[20]
               y_data =  Data                            trial[6] × subject[20]

                   mu ~  Normal(0, 1)
                sigma ~  HalfNormal(0, 1)
     global_intercept ~  Normal(0, 1)
    intercept_subject ~  Normal(0, 1)                    [20, 1]
         beta_subject ~  Normal(mu, sigma)               subject[20]
                noise ~  Exponential(f())
                                                         Parameter count = 44

             mu_trial =  f(intercept_subject,            trial[6] × subject[20]
                         beta_subject,
                         global_intercept)

 beta_subject_penalty =  Potential(f(beta_subject))      subject[20]

                    y ~  Normal(mu_trial, noise)         trial[6] × subject[20]
"""
    assert [s.strip() for s in table_txt.splitlines()] == [s.strip() for s in expected.splitlines()]

    table_txt = get_text(model_table(model, split_groups=False))
    expected = """               Variable  Expression                      Dimensions
────────────────────────────────────────────────────────────────────────────────
               x_data =  Data                            trial[6] × subject[20]
               y_data =  Data                            trial[6] × subject[20]
                   mu ~  Normal(0, 1)
                sigma ~  HalfNormal(0, 1)
     global_intercept ~  Normal(0, 1)
    intercept_subject ~  Normal(0, 1)                    [20, 1]
         beta_subject ~  Normal(mu, sigma)               subject[20]
             mu_trial =  f(intercept_subject,            trial[6] × subject[20]
                         beta_subject,
                         global_intercept)
                noise ~  Exponential(f())
                    y ~  Normal(mu_trial, noise)         trial[6] × subject[20]
 beta_subject_penalty =  Potential(f(beta_subject))      subject[20]
                                                         Parameter count = 44
"""
    assert [s.strip() for s in table_txt.splitlines()] == [s.strip() for s in expected.splitlines()]

    table_txt = get_text(
        model_table(model, split_groups=False, truncate_deterministic=30, parameter_count=False)
    )
    expected = """               Variable  Expression                  Dimensions
────────────────────────────────────────────────────────────────────────────
               x_data =  Data                        trial[6] × subject[20]
               y_data =  Data                        trial[6] × subject[20]
                   mu ~  Normal(0, 1)
                sigma ~  HalfNormal(0, 1)
     global_intercept ~  Normal(0, 1)
    intercept_subject ~  Normal(0, 1)                [20, 1]
         beta_subject ~  Normal(mu, sigma)           subject[20]
             mu_trial =  f(intercept_subject, ...)   trial[6] × subject[20]
                noise ~  Exponential(f())
                    y ~  Normal(mu_trial, noise)     trial[6] × subject[20]
 beta_subject_penalty =  Potential(f(beta_subject))  subject[20]
"""
    assert [s.strip() for s in table_txt.splitlines()] == [s.strip() for s in expected.splitlines()]
