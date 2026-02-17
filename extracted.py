import numpy as np, pandas as pd, os
import pymc as pm
import pytensor.tensor as pt
from collections import defaultdict

age_groups = ['1', '2', '5', '4', '1', '6', '2', '3', '0', '2', '5', '5', '3', '1', '1', '4', '2', '1', '1', '1', '0', '4', '0', '5', '2', '4', '4', '0', '1', '6', '3', '2', '2', '2', '1', '1', '5', '2', '5', '1', '1', '1', '6', '3', '3', '3', '4', '6', '1', '1']
incs = ['C', 'D', 'B', 'F', 'A', 'B', 'D', 'F', 'B', 'D', "Don't wish to say", 'B', "Don't wish to say", 'C', 'B', 'A', 'B', 'C', 'B', 'D', "Don't know", 'C', "Don't wish to say", 'B', 'B', 'B', "Don't wish to say", "Don't wish to say", 'C', 'C', "Don't know", 'F', 'F', 'A', 'B', 'C', 'B', 'C', 'B', 'F', 'C', "Don't know", 'F', 'F', 'C', 'C', 'A', 'B', 'A', 'C']
ordered_outputs = {
    'model': 'Stereotype',
    'nonordered': ["Don't know", 'Hard to say', "Don't wish to say"],
    'n_fitted_dm': 3
}
df = pd.DataFrame({'age_group': age_groups, "income": incs})
tcats = pd.CategoricalDtype(categories=["Don't know", 'A', 'B', 'C', "Don't wish to say", 'D', 'F'], ordered=True)
df.income = df.income.astype(tcats)
df.age_group = df.age_group.astype("category")

inds = list(range(50))
model = pm.Model()
name = 'small_model'
def model_input_effect(cc, oc, ocdim):
    r"""
    Return Deterministic node <- prior_slope + scale*offset
    Also adds offset node.
    """
    prior_slope, scale = (0.0, None)
    zs_axes = [0,1]
    offset = pm.StudentT(f"α_{cc}_offset_{oc}", nu=5, sigma=1, dims=(cc, ocdim), transform=pm.distributions.transforms.ZeroSumTransform(zs_axes))
    scale = 3 * pt.ones_like(offset)
    effect = pm.Deterministic(f"α_{cc}_slope_{oc}", prior_slope + scale*offset, dims=(cc,ocdim) ) # mat - with dims no_cats_inp(agegroup) x income(val)
    return effect

def model_categorical_output(model:pm.Model, obs_map, oc, ocdim, oidx_dim, cats, odf, mu, ordered_options):
    r"""
    Creates 
    y_obs - Data 
    test_income - Deterministic
    y_income - Multinomial
    phi_delta - Dirichlet 
    stereotype_intercept - StudentT
    -------
    Returns:
    om_name : str
    lpv : TensorVariable
    yobs : SharedVariable | TensorConstant
        only for testing/outlier detection purposes?
    """

    # Ns = cat_shared['Ns'][obs_map]
    Ns = pm.Data(f"small_model_N", np.ones(len(df),dtype='int'), dims="small_model_obs_idx")
    occats = ["Don't know", 'A', 'B', 'C', "Don't wish to say", 'D', 'F']

    n_ovs = ((ordered_options['n_fitted_dm']-1) 
            if ordered_options['model'] in ['Mixture'] else 1)
    n_unordered = (len(cats[ocdim]) - n_ovs)
    n_ordered = len(occats) - n_unordered # (Total cats) - (unordered cats)
    om_name = f"Ordered {ordered_options['model']} {n_ordered}"
    if n_unordered>0: om_name += f' + {cats[ocdim][n_ovs:]}'
    # phi_delta = pm.Dirichlet(f'phi_diffs_{ oc }', [1.0]*(n_ordered-1) )
    # phi = pt.concatenate([[0], pt.cumsum(phi_delta)])
    s_mu = pm.StudentT(f"stereotype_intercept_{oc}", nu=5, sigma=3.5, size=n_ordered, transform=pm.distributions.transforms.ZeroSumTransform([-1]))
    log_odds = s_mu[None,:] #+ phi[None,:]*mu[:,0,None] 
    probs = pm.math.softmax(log_odds, axis=-1)

    # if log_odds is None: 
    #     log_odds = pt.log(probs)
    #     log_odds += np.log(n_ordered) # Add log(n_ordered) to make the log_odds centered at 0

    # Empty tensor for full odds
    fodds = pt.empty( (model.dim_lengths[oidx_dim],len(occats)), dtype='float')

    # Unordered log_odds
    uoinds = [ occats.index(c) for c in cats[ocdim][n_ovs:] ]
    pmu = mu[:,-n_unordered:]
    fodds = pt.set_subtensor(fodds[:,uoinds], pmu)

    # Ordered log_odds
    oinds = [ i for i in range(len(occats)) if i not in uoinds ]
    fodds = pt.set_subtensor(fodds[:,oinds], log_odds)

    # Turn to probabilities via softmax
    fprobs = pm.math.softmax(fodds, axis=-1)
    # from pytensor.printing import debugprint
    # print(debugprint(fprobs))

    # Create the observation model
    ov = pd.get_dummies(odf[oc]).loc[:,list(occats)]
    yobs = pm.Data(f"obs_{ oc }", ov.to_numpy().astype('int'), dims=(oidx_dim, oc+'_outp'))
    pm.Deterministic(f'test_{oc}', fprobs, dims=(oidx_dim, oc+'_outp')) # This is for testing
    # pm.Multinomial(f"y_{ oc }", p=fprobs, n=Ns, observed=yobs, dims=(oidx_dim, oc+'_outp'))
    return om_name, yobs

with model:
    cats = model.coords
    mutable = { f"{name}_obs_idx": inds }
    model.add_coord("small_model_obs_idx",inds)
        
    m_ids = {}
    needed_input_cols = {'age_group', 'income'}
 
    ccats = list(df['age_group'].dtype.categories)
    cidx = df['age_group'].astype('object').replace(dict(zip(ccats,range(len(ccats))))).fillna(-1).to_numpy(dtype='int')
    model.add_coord('age_group',ccats)
    m_ids['age_group'] = pm.Data(f"small_model_age_group_id", cidx, dims="small_model_obs_idx")

    ccats = list(df['income'].dtype.categories)
    cidx = df['income'].astype('object').replace(dict(zip(ccats,range(len(ccats))))).fillna(-1).to_numpy(dtype='int')
    model.add_coord('income',ccats)
    m_ids['income'] = pm.Data(f"small_model_income_id", cidx, dims="small_model_obs_idx")

    zs_set = {'income'}
    uvals = set(df['income'].unique())
    ccats = ["Don't know", 'A', 'B', 'C', "Don't wish to say", 'D', 'F']
    model.add_coord('income_outp',ccats)
    model.add_coord("income_inp", ['order_val',"Don't know", "Don't wish to say"])

    mu0 = pt.zeros((model.dim_lengths[f"{name}_obs_idx"],model.dim_lengths["income_inp"]))
    mu0 += pm.StudentT("intercept_income", nu=5, sigma=3.5, dims="income_inp", transform=pm.distributions.transforms.ZeroSumTransform([-1]))[None,:]
    mu_list = [mu0]
    for cc in ['age_group','income']:
        eff = model_input_effect(cc, 'income', "income_inp")
        mu_list += [eff[m_ids[cc]]]

    stack = pt.stack(mu_list,axis=0)
    mu = pt.sum(stack,axis=0)

    # Create a map dimension of rows that have a value for this output
    oidx_dim = "obs_idx_map_income"
    mutable[oidx_dim] = list(df.index)
    model.add_coord(oidx_dim, mutable[oidx_dim])#, mutable = True)
    obs_map = pm.Data("map_oidx_income",np.array(mutable[oidx_dim]).astype('int'),dims=(oidx_dim,))
    omu, odf = mu[obs_map,...], df.loc[mutable[oidx_dim],:]
    
    om_name, yobs = model_categorical_output(model, obs_map, "income", "income_inp", oidx_dim, cats, odf, omu, ordered_options=ordered_outputs)

with model:
    import pymc_extras as pmx
    pmx.fit(
        method="pathfinder",
        inference_backend="pymc",
        jitter=2,
        num_paths=4,
        num_draws=1
    )