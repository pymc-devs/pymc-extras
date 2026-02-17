"""
Plain PyMC implementation of the MaxDiff ordinal-ranking model.

The helpers below are adapted from `salk_internal_package.glm` and
`salk_internal_package.latent_models` so this module can be executed
without importing `salk_internal_package` directly.
"""

from __future__ import annotations

import warnings
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import os

ROOT = Path(__file__).resolve().parents[1]
PYTENSOR_CACHE = Path("/tmp/pytensor_cache")
PYTENSOR_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("PYTENSOR_FLAGS", f"compiledir={PYTENSOR_CACHE}")

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

try:
    import pymc_extras as pmx
except ImportError:  # pragma: no cover - optional dependency
    pmx = None

pd.set_option("display.max_rows", 50)
warnings.filterwarnings("ignore", "Column .* not found")

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
    "data": {
        "file": "synthetic_maxdiff",
        "filter": "citizen and age >= 18",
    },
    "population": {
        "draws": 100,
        "file": "synthetic_population",
        "group_size_column": "N",
        "samples": 10,
    },
    "sequence": [
        {
            "inp_cols": ["unit", "nationality", "gender", "age_group", "education"],
            "name": "maxdiff",
            "res_cols": [
                {
                    "name": "maxdiff",
                    "structure": [
                        [["Q2_1best"], ["Q2_1set"]],
                        [["Q2_2best"], ["Q2_2set"]],
                        [["Q2_3best"], ["Q2_3set"]],
                        [["Q2_4best"], ["Q2_4set"]],
                        [["Q2_5best"], ["Q2_5set"]],
                        [["Q2_6best"], ["Q2_6set"]],
                        [["Q2_7best"], ["Q2_7set"]],
                        [["Q2_8best"], ["Q2_8set"]],
                        [["Q2_9best"], ["Q2_9set"]],
                        [["Q2_10best"], ["Q2_10set"]],
                        [["Q2_1set"], ["Q2_1worst"]],
                        [["Q2_2set"], ["Q2_2worst"]],
                        [["Q2_3set"], ["Q2_3worst"]],
                        [["Q2_4set"], ["Q2_4worst"]],
                        [["Q2_5set"], ["Q2_5worst"]],
                        [["Q2_6set"], ["Q2_6worst"]],
                        [["Q2_7set"], ["Q2_7worst"]],
                        [["Q2_8set"], ["Q2_8worst"]],
                        [["Q2_9set"], ["Q2_9worst"]],
                        [["Q2_10set"], ["Q2_10worst"]],
                    ],
                }
            ],
            "sampler": {"sampler": "pathfinder", "target_accept": 0.95},
        }
    ],
    "settings": {},
}


# ---------------------------------------------------------------------------
# Helpers inspired by salk_internal_package.latent_models
# ---------------------------------------------------------------------------

def is_latent(rc):
    return isinstance(rc, dict)


def res_col_id(rc):
    if isinstance(rc, str):
        return rc
    if is_latent(rc):
        return rc["name"]
    raise TypeError(f"Unknown res_col type for {rc}")


def latent_res_cats(rc, prefix=True):
    cats = rc["res_cats"]
    if prefix:
        pref = rc.get("prefix") or ""
        return [pref + str(v) for v in cats]
    return cats


def prepare_ordinal_ranking(ordinal_ranking, df):
    ordinal_ranking = ordinal_ranking.copy()
    if df is None:
        raise ValueError("Cannot infer ordinal_ranking categories without dataframe")

    vals: set[str] = set()
    for seg in ordinal_ranking["structure"]:
        (l1, l2) = (seg["a"], seg["b"]) if isinstance(seg, dict) else seg
        for col in (l1 or []) + (l2 or []):
            fel = df[col].dropna().iloc[0]
            if isinstance(fel, (list, np.ndarray)):
                vals |= set(df[col].dropna().explode().dropna())
            else:
                vals |= set(df[col].dropna().unique())

    ordinal_ranking["res_cats"] = sorted(list(vals))
    ordinal_ranking.setdefault("prefix", f"{ordinal_ranking['name']}_")
    return ordinal_ranking


def factorize_w_codes(series: pd.Series, categories: List[str]) -> np.ndarray:
    dtype = pd.CategoricalDtype(categories=categories, ordered=True)
    return series.astype(dtype).cat.codes.to_numpy(dtype=int)


def factorize_w_codes_lst(series: pd.Series, categories: List[str]) -> np.ndarray:
    rep = series.dropna().iloc[0]
    if isinstance(rep, (list, np.ndarray)):
        maxlen = max(len(l) for l in series.dropna())
        mat = np.array(
            [
                (list(l) + [None] * (maxlen - len(l)))
                if isinstance(l, Iterable)
                else [None] * maxlen
                for l in series
            ],
            dtype=object,
        )
        return np.stack([factorize_w_codes(pd.Series(mat[:, i]), categories) for i in range(mat.shape[1])], axis=1)
    return factorize_w_codes(series, categories)[:, None]


def shift_rows(arr: np.ndarray, shifts: np.ndarray, fill=0):
    max_pos_shift = max(0, np.max(shifts))
    max_neg_shift = max(0, -np.min(shifts))
    pad_width = ((0, 0), (max_pos_shift, max_neg_shift))
    padded = np.pad(arr, pad_width, mode="constant", constant_values=fill)
    starts = max_pos_shift - shifts
    rows = np.arange(arr.shape[0])[:, None]
    columns = starts[:, None] + np.arange(arr.shape[1])
    return padded[rows, columns]


def po_matrices_from_segment(seg, df, categories, defaults=None):
    defaults = defaults or {}
    if isinstance(seg, dict):
        l1, l2, opts = seg["a"], seg["b"], seg
    else:
        l1, l2, opts = seg[0], seg[1], {}
    opts = {**defaults, **opts}

    full = np.tile(np.arange(len(categories), dtype=int), (len(df), 1))
    empty = np.zeros((len(df), 0), dtype=int)

    a = (
        np.concatenate([factorize_w_codes_lst(df[c], categories) for c in l1], axis=1)
        if l1
        else (full if l1 is None else empty)
    )
    b = (
        np.concatenate([factorize_w_codes_lst(df[c], categories) for c in l2], axis=1)
        if l2
        else (full if l2 is None else empty)
    )

    if opts.get("local_b"):
        local_cats = df[l1[0]].dtype.categories
        b = np.tile([categories.index(c) for c in local_cats], (len(df), 1))

    reverse = (a.shape[1] > b.shape[1] and b.shape[1] > 0)
    if reverse:
        a, b = b, a

    matches = (a[:, :, None] == b[:, None, :]).any(-2)
    b[np.where(matches)] = -1

    tie_offsets = None
    if opts.get("ties"):
        tie_offsets = np.array(
            [
                list(tl) + [0] * (a.shape[1] - len(tl)) if isinstance(tl, Iterable) else [0] * a.shape[1]
                for tl in df[opts["ties"]].tolist()
            ]
        )

    dn = len(categories) + 1
    po_am = np.zeros((len(df), dn, dn), dtype=bool)
    mask = np.ones(a.shape[0], dtype=bool)
    first_n = opts.get("first_n") or a.shape[1]

    for i in range(a.shape[1]):
        non_na = a[:, i] != -1
        if i >= first_n:
            if tie_offsets is None:
                break
            mask &= tie_offsets[:, i - 1] > 0
            non_na &= mask
        if not non_na.any():
            continue

        cur = a[non_na, i]
        cmp_b = b[non_na, :]
        if tie_offsets is not None:
            avs = shift_rows(a[non_na, i + 1 :], -tie_offsets[non_na, i], -1)
        elif opts.get("ordered"):
            avs = a[non_na, i + 1 :]
        else:
            avs = None
        if avs is not None:
            cmp_b = np.concatenate([avs, cmp_b], axis=1)

        nna_idx = np.arange(len(df))[non_na]
        for j in range(cmp_b.shape[1]):
            po_am[nna_idx, cmp_b[:, j], cur] = True

    if reverse:
        po_am = po_am.transpose(0, 2, 1)
    return po_am[:, :-1, :-1]


def transitive_closure(am):
    tc = am.astype(bool)
    for k in range(tc.shape[-1]):
        tc |= np.logical_and(tc[..., :, k, None], tc[..., None, k, :])
    return tc


def transitive_reduction(am, am_closed=False):
    tc = transitive_closure(am) if not am_closed else am.astype(bool)
    return tc * (1 - np.matmul(tc, tc))


def merge_partial_orders(am1, am2, mode="first"):
    if mode == "first":
        res = am1 | (am2 & ~np.swapaxes(am1, -1, -2))
    elif mode == "clean":
        res = am1 | am2
    else:
        raise ValueError(f"Unknown PO merge mode {mode}")

    tc = transitive_closure(res)
    conflicts = tc & np.swapaxes(tc, -1, -2)
    tc = tc & ~conflicts
    return tc


def observe_po_gaussian(po_am, mu, var, cov, observe_name=None):
    cmps = np.argwhere(po_am)
    diffs = mu[cmps[:, 0], cmps[:, 2]] - mu[cmps[:, 0], cmps[:, 1]]
    cvs = var[cmps[:, 1]] + var[cmps[:, 2]]
    if cov is not None:
        cvs -= 2 * cov[cmps[:, 1], cmps[:, 2]]
    zscores = diffs / np.sqrt(cvs)
    eps = 1e-4
    probs = (0.5 + eps) + (0.5 - eps) * pm.math.erf(zscores / np.sqrt(2))
    if observe_name:
        pm.Potential(observe_name, pt.log(probs))
    inds = cmps[:, 0]
    logp = pt.zeros(po_am.shape[0])
    logc = pt.zeros(po_am.shape[0], dtype="int32")
    logp = pt.inc_subtensor(logp[inds], pt.log(probs))
    logc = pt.inc_subtensor(logc[inds], 1)
    return logp, logc


def observe_po_thurstone(po_am, mu, var, cov, initvals, observe_name=None):
    if observe_name:
        raise NotImplementedError("Thurstone model not supported in this standalone script")
    return pt.zeros(po_am.shape[0]), pt.zeros(po_am.shape[0], dtype="int32")


def observe_po_multinomial(po_am, mu, observe_name=None):
    odds = pt.exp(mu)
    po_rev = po_am.transpose(0, 2, 1)
    nz = np.where(po_rev.any(axis=-1))
    cur_odds = odds[nz]
    lower_odds = (po_rev[nz] * odds[nz[0]]).sum(axis=-1)
    probs = cur_odds / (cur_odds + lower_odds)
    if observe_name:
        pm.Potential(observe_name, pt.log(probs))
    inds = nz[0]
    logp = pt.zeros(po_am.shape[0])
    logc = pt.zeros(po_am.shape[0], dtype="int32")
    logp = pt.inc_subtensor(logp[inds], pt.log(probs))
    logc = pt.inc_subtensor(logc[inds], 1)
    return logp, logc


def observe_model(model_type, po_am, mu, mdl_data, initvals, oname=None):
    if model_type.startswith("Multinomial"):
        return observe_po_multinomial(po_am, mu, observe_name=oname)
    if model_type.startswith("Gaussian"):
        return observe_po_gaussian(po_am, mu, mdl_data.get("var"), mdl_data.get("cov"), observe_name=oname)
    if model_type.startswith("Thurstone"):
        return observe_po_thurstone(po_am, mu, mdl_data.get("var"), mdl_data.get("cov"), initvals, observe_name=oname)
    raise ValueError(f"Unknown ordinal-ranking model type {model_type}")


def model_ordinal_ranking_setup_shared(model, df, or_cols, ocdim_map, zs_set):
    or_shared = {}
    if len(or_cols) == 0:
        return or_shared

    for ordinal_ranking in or_cols:
        pcats = latent_res_cats(ordinal_ranking, prefix=True)
        name = res_col_id(ordinal_ranking)
        model.add_coord(name, pcats)
        model.add_coord(f"obs_idx_map_{name}", list(df.index))
        ocdim_map[name] = name
        zs_set.add(name)

    or_shared["use_intermediate"] = pm.Data("use_intermediate_ordinal_ranking_inputs", 0.0)
    return or_shared


def model_ordinal_ranking_output(
    model,
    df,
    o_mu,
    o_mu0,
    or_shared,
    cur_ordinal_ranking,
    priors,
    ci_powers,
    m_vals_z,
    mname,
    initvals,
    seq_cond,
):
    ordinal_ranking = {**or_shared, **cur_ordinal_ranking}
    mdm = ordinal_ranking["structure"]
    ordered = ordinal_ranking.get("ordered", False)
    model_type = ordinal_ranking.get("model", "Multinomial")
    name = res_col_id(ordinal_ranking)

    oidx = list(df.index)
    oidx_dim = f"obs_idx_map_{name}"
    model.add_coord(oidx_dim, oidx)

    mdcats = latent_res_cats(ordinal_ranking, prefix=False)
    obs_map = pm.Data(f"map_oidx_{name}", np.array(oidx, dtype=int), dims=(oidx_dim,))
    df = df.loc[oidx, :]

    logsum = pt.zeros(model.dim_lengths[f"{mname}_obs_idx"], dtype="float")
    ls_counts = pt.ones(model.dim_lengths[f"{mname}_obs_idx"], dtype="float") * 1e-5

    for ti, seg in enumerate(mdm):
        po_am = po_matrices_from_segment(seg, df, mdcats, {"ordered": ordered})
        for mu_i, current_mu in enumerate([o_mu, o_mu0]):

            oname = f"ordinal_ranking_probs_{name}_{ti}" if mu_i == 0 else None
            logp, logc = observe_model(model_type, po_am, current_mu, {}, initvals, oname)
            if mu_i == 0:
                logsum, ls_counts = logsum + logp, ls_counts + logc
            else:
                logsum -= logp

    res = o_mu
    res = pm.Deterministic(f"y_{name}", res, dims=(f"{mname}_obs_idx", name))

    if seq_cond:
        pcats = latent_res_cats(ordinal_ranking, prefix=True)
        mean, std = 0.0, priors["scale"]("default", fixed=True)
        for i, cc in enumerate(pcats):
            model.add_coord(cc, ci_powers[cc])
            inp_res = pm.Data(f"{mname}_{cc}_id", np.zeros(len(df)), dims=(f"{mname}_obs_idx"))
            use_int = or_shared["use_intermediate"]
            rvs = inp_res * use_int + res[:, i] * (1.0 - use_int)
            z_scores = 0.5 * (rvs - mean) / std
            m_vals_z[cc] = pt.stack([z_scores**pow for pow in ci_powers[cc]], axis=0)

    o_logsum = pt.zeros(model.dim_lengths[f"{mname}_obs_idx"], dtype="float")
    o_logsum = pt.set_subtensor(o_logsum[obs_map], logsum / ls_counts)
    pm.Deterministic(f"logp_{name}", o_logsum, dims=(f"{mname}_obs_idx"))

    return f"Ordinal Ranking {model_type} {len(mdcats)}"


# ---------------------------------------------------------------------------
# Simplified GLM pieces adapted from salk_internal_package.glm
# ---------------------------------------------------------------------------

def default_priors():
    def scale_prior(_name, fixed=False):
        if fixed:
            return 1.0

        def inner(name, **kwargs):
            return pm.HalfNormal(name, sigma=1.0, **kwargs)

        return inner

    return {
        "baseline": lambda name, **kwargs: pm.StudentT(name, nu=5, sigma=3.5, **kwargs),
        "mu": lambda name, **kwargs: pm.Normal(name, mu=0.0, sigma=1.0, **kwargs),
        "scale": scale_prior,
        "pool": lambda name, **kwargs: pm.Exponential(name, 1.0, **kwargs),
        "detailed": {},
    }


# ---------------------------------------------------------------------------
# Data loading and model construction
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Data loading and model construction
# ---------------------------------------------------------------------------


def _random_set(rng: np.random.Generator, k: int = 5) -> List[str]:
    return rng.choice(TOPICS, size=k, replace=False).tolist()


def _build_synthetic_dataframe(num_rows: int = 500, seed: int = 13) -> pd.DataFrame:
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
            "N": 1,
        }
        for i in range(1, 11):
            choice_set = _random_set(rng)
            best_topic = rng.choice(choice_set)
            remaining = [t for t in choice_set if t != best_topic]
            worst_topic = rng.choice(remaining) if remaining else best_topic
            row[f"Q2_{i}best"] = best_topic
            row[f"Q2_{i}set"] = choice_set.copy()
            row[f"Q2_{i}worst"] = worst_topic
        rows.append(row)

    df = pd.DataFrame(rows)

    category_map = {
        "unit": units,
        "nationality": nationalities,
        "gender": genders,
        "age_group": age_groups,
        "education": educations,
    }
    for col, cats in category_map.items():
        df[col] = pd.Categorical(df[col], categories=cats)

    for i in range(1, 11):
        for suffix in ("best", "worst"):
            col = f"Q2_{i}{suffix}"
            df[col] = pd.Categorical(df[col], categories=TOPICS)

    return df


def read_and_process_data(desc: Dict, return_meta: bool = True, add_original_inds: bool = True):
    df = _build_synthetic_dataframe()
    flt = desc.get("filter")
    if flt:
        df = df.query(flt, engine="python")

    if add_original_inds:
        df = df.reset_index(names="orig_index")
    else:
        df = df.reset_index(drop=True)

    meta = {"columns": list(df.columns)}
    return (df, meta) if return_meta else df


def load_maxdiff_inputs():
    df, meta = read_and_process_data(MODEL_DESC["data"], return_meta=True, add_original_inds=True)
    return df, meta, MODEL_DESC


def build_maxdiff_model():
    df, _meta, model_desc = load_maxdiff_inputs()
    step = model_desc["sequence"][0]
    ordinal_desc = prepare_ordinal_ranking(step["res_cols"][0], df)
    inp_cols = step["inp_cols"]
    model_name = step["name"]

    use_df = df.dropna(subset=inp_cols).reset_index(drop=True)
    priors = default_priors()

    with pm.Model() as model:
        obs_dim = f"{model_name}_obs_idx"
        obs_idx = np.arange(len(use_df))
        model.add_coord(obs_dim, obs_idx)

        cat_ids = {}
        for col in inp_cols:
            use_df[col] = use_df[col].astype("category")
            cats = list(use_df[col].cat.categories)
            model.add_coord(col, cats)
            cat_ids[col] = pm.Data(f"{model_name}_{col}_id", use_df[col].cat.codes.to_numpy(dtype=int), dims=(obs_dim))

        ocdim = res_col_id(ordinal_desc)
        topics = latent_res_cats(ordinal_desc, prefix=True)
        model.add_coord(ocdim, topics)

        intercept = priors["baseline"](
            f"intercept_{ocdim}",
            dims=ocdim,
            transform=pm.distributions.transforms.ZeroSumTransform([-1]),
        )
        mu0 = intercept[None, :]

        mu = pt.zeros((len(use_df), len(topics))) + mu0
        for col, idx_data in cat_ids.items():
            effect = priors["mu"](
                f"α_{col}_{ocdim}",
                dims=(col, ocdim),
                transform=pm.distributions.transforms.ZeroSumTransform([-1]),
            )
            mu = mu + effect[idx_data]

        ocdim_map: Dict[str, str] = {}
        zs_set: set[str] = set()
        or_shared = model_ordinal_ranking_setup_shared(model, use_df, [ordinal_desc], ocdim_map, zs_set)
        ci_powers = defaultdict(lambda: [0.0])
        m_vals_z: Dict[str, pt.TensorVariable] = {}
        initvals: Dict[str, np.ndarray] = {}
        model_ordinal_ranking_output(
            model,
            use_df,
            mu,
            mu0,
            or_shared,
            ordinal_desc,
            priors,
            ci_powers,
            m_vals_z,
            model_name,
            initvals,
            seq_cond=False,
        )

    return model


def fit_model(model):
    if pmx is not None:
        try:
            with model:
                return pmx.fit(
                    method="pathfinder",
                    inference_backend="pymc",
                    jitter=2,
                    num_paths=4,
                    num_draws=10,
                )
        except Exception as exc:  # pragma: no cover - best-effort fallback
            warnings.warn(f"Pathfinder failed ({exc}); falling back to NUTS sampling.")

    with model:
        return pm.sample(draws=200, tune=200, chains=2, target_accept=0.9, progressbar=False)


def parse_args():
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
        where = "Pathfinder" if pmx else "NUTS"
        print(f"{where} inference finished.")
        return idata
    return model


if __name__ == "__main__":
    main()