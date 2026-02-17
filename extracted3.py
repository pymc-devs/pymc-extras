"""Standalone PyMC MaxDiff demo without salk_toolkit dependencies."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pymc import math as pmmath
from pymc.distributions import transforms as pm_transforms

try:  # Optional but nice to have for quick approximations
    import pymc_extras as pmx  # type: ignore
except ImportError:  # pragma: no cover
    pmx = None

ROOT = Path(__file__).resolve().parents[1]
PYTENSOR_CACHE = Path("/tmp/pytensor_cache")
PYTENSOR_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("PYTENSOR_FLAGS", f"compiledir={PYTENSOR_CACHE}")

pd.set_option("display.max_rows", 50)

TOPICS = [
    "Sama erakonna toetamine, keda toetan siseriiklikult",
    "Umbusalduse avaldamine Eesti valitsusele",
    "Kiire rahuläbirääkimiste alustamine Vene-Ukraina sõja lõpetamiseks",
    "Taastuvenergiale ülemineku kiirendamine",
    "Põlevkivienergiaga jätkamiseks erandi tegemine Euroopa kliimapoliitikas",
    "Euroopa kaitsetööstuse võimekuse tõstmine",
    "Euroopa energiatööstuse tootmisvõimsuse suurendamine",
    "Loodussäästlike majandusreformide ja kliimapoliitikaga jätkamine",
    "Vene-vastaste sanktsioonide tugevdamine",
    "Vaesemate piirkondade majandusliku mahajäämuse leevendamine",
    "Euroopa sõjalise toetuse suurendamine Ukrainale",
    "Euroopa välispiiride kaitse tugevdamine",
    "Sisserände piiramine Euroopasse",
    "Sunnimeetmete tugevdamine õigusriigi tagamiseks liikmesriikides (nt Ungari, Slovakkia puhul)",
    "Euroopa Liidu bürokraatia vähendamine",
    "Eesti põllumeeste toetuste tõstmine Euroopa Liidu keskmise tasemeni",
    "Miinimumpalga tõstmine 50% keskmisest palgast kõigis liikmesriikides",
    "Euroopa majandusliku konkurentsivõime suurendamine",
]

MODEL_DESC = {
    "name": "maxdiff-min",
    "data": {"file": "synthetic_maxdiff", "filter": "citizen and age >= 18"},
    "sequence": [
        {
            "inp_cols": ["unit", "nationality", "gender", "age_group", "education"],
            "name": "maxdiff",
        }
    ],
}


def default_priors() -> Dict[str, Callable[..., object]]:
    """Return tiny convenience wrappers for baseline priors."""

    def scale_prior(_name: str, fixed: bool = False):
        if fixed:
            return 1.0

        def inner(name: str, **kwargs):
            return pm.HalfNormal(name, sigma=1.0, **kwargs)

        return inner

    return {
        "baseline": lambda name, **kwargs: pm.StudentT(name, nu=5, sigma=3.5, **kwargs),
        "mu": lambda name, **kwargs: pm.Normal(name, mu=0.0, sigma=1.0, **kwargs),
        "scale": scale_prior,
    }


def _random_set(rng: np.random.Generator, k: int = 5) -> List[str]:
    return rng.choice(TOPICS, size=k, replace=False).tolist()


def _build_synthetic_dataframe(num_rows: int = 500, seed: int = 13) -> pd.DataFrame:
    """Create a toy MaxDiff frame with best-of-five answers."""

    rng = np.random.default_rng(seed)
    units = ["Harjumaa", "Lääne-Virumaa", "Tartumaa", "Tallinn", "Pärnumaa", "Saaremaa"]
    nationalities = ["Estonian", "Other"]
    genders = ["Male", "Female"]
    age_groups = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]
    educations = ["Basic education", "Secondary education", "Higher education"]

    rows = []
    for _ in range(num_rows):
        row = {
            "unit": rng.choice(units),
            "nationality": rng.choice(nationalities),
            "gender": rng.choice(genders),
            "age_group": rng.choice(age_groups),
            "education": rng.choice(educations),
            "citizen": bool(rng.random() > 0.1),
            "age": int(rng.integers(18, 80)),
        }
        for i in range(1, 11):
            choice_set = _random_set(rng)
            row[f"Q2_{i}best"] = rng.choice(choice_set)
            row[f"Q2_{i}set"] = choice_set
        rows.append(row)

    df = pd.DataFrame(rows)

    for col, cats in {
        "unit": units,
        "nationality": nationalities,
        "gender": genders,
        "age_group": age_groups,
        "education": educations,
    }.items():
        df[col] = pd.Categorical(df[col], categories=cats)

    for i in range(1, 11):
        df[f"Q2_{i}best"] = pd.Categorical(df[f"Q2_{i}best"], categories=TOPICS)

    return df


def read_and_process_data(desc: Dict[str, str]) -> pd.DataFrame:
    df = _build_synthetic_dataframe()
    flt = desc.get("filter")
    if flt:
        df = df.query(flt, engine="python")
    return df.reset_index(drop=True)


def load_maxdiff_inputs() -> tuple[pd.DataFrame, Dict[str, Dict]]:
    return read_and_process_data(MODEL_DESC["data"]), MODEL_DESC


def build_maxdiff_model() -> pm.Model:
    df, model_desc = load_maxdiff_inputs()
    step = model_desc["sequence"][0]
    inp_cols = step["inp_cols"]
    model_name = step["name"]

    use_df = df.dropna(subset=inp_cols + ["Q2_1best"]).reset_index(drop=True)
    priors = default_priors()

    with pm.Model() as model:
        obs_dim = f"{model_name}_obs_idx"
        obs_idx = np.arange(len(use_df))
        model.add_coord(obs_dim, obs_idx)

        cat_ids = {}
        for col in inp_cols:
            cats = list(use_df[col].cat.categories)
            model.add_coord(col, cats)
            cat_ids[col] = pm.Data(
                f"{model_name}_{col}_id",
                use_df[col].cat.codes.to_numpy(dtype=int),
                dims=(obs_dim,),
            )

        model.add_coord("topics", TOPICS)

        intercept = priors["baseline"](
            "intercept_topics",
            dims="topics",
            transform=pm_transforms.ZeroSumTransform([-1]),
        )
        mu = intercept[None, :]  # type: ignore[index]
        for col, idx_data in cat_ids.items():
            effect = priors["mu"](
                f"α_{col}_topics",
                dims=(col, "topics"),
                transform=pm_transforms.ZeroSumTransform([-1]),
            )
            mu = mu + effect[idx_data]  # type: ignore[index]

        obs_mask = ~use_df["Q2_1best"].isna()
        mask_idx = np.where(obs_mask)[0]
        best_idx = use_df.loc[obs_mask, "Q2_1best"].cat.codes.to_numpy(dtype=int)
        obs_dim_best = f"{model_name}_obs_best"
        model.add_coord(obs_dim_best, mask_idx)

        pm.Categorical(
            f"y_{model_name}_best",
            p=pmmath.softmax(mu[mask_idx], axis=1),
            observed=best_idx,
            dims=obs_dim_best,
        )
        pm.Deterministic("mu_logits", mu, dims=(obs_dim, "topics"))

    return model


def fit_model(model: pm.Model):
    if pmx is not None:
        try:
            with model:
                return pmx.fit(
                    method="pathfinder",
                    inference_backend="pymc",
                    jitter=2,
                    num_paths=4,
                    num_draws=1,
                )
        except Exception as exc:  # pragma: no cover
            print(f"Pathfinder failed ({exc}); falling back to NUTS sampling.")

    with model:
        return pm.sample(draws=200, tune=200, chains=2, target_accept=0.9, progressbar=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the MaxDiff PyMC model.")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run inference after constructing the model (default: build only).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = build_maxdiff_model()
    print("PyMC model constructed successfully.")
    if args.sample:
        idata = fit_model(model)
        engine = "Pathfinder" if pmx else "NUTS"
        print(f"{engine} inference finished.")
        return idata
    return model


if __name__ == "__main__":
    main()
