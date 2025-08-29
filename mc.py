"""Monte Carlo marginalization and MAP selection across discrete groups."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .kde import kde_bounded_sample, count_kde_modes

def monte_carlo_integral(lithology, ibeta, Vp_obs, Vs_obs, EC_obs, P_GPa, T_K,
                         objective_function, num_samples=100_000):
    # log10-uniform priors over ranges (expressed in log10)
    phi_bounds     = (-5, -0.1)
    alpha_bounds   = (-5, -0.01)
    alpha_EC_bounds= (-5, -0.1)

    phi   = 10 ** np.random.uniform(*phi_bounds,     num_samples)
    alpha = 10 ** np.random.uniform(*alpha_bounds,   num_samples)
    a_ec  = 10 ** np.random.uniform(*alpha_EC_bounds,num_samples)

    logp = []
    for p, a, e in zip(phi, alpha, a_ec):
        err = objective_function([lithology, ibeta, p, a, e],
                                 Vp_obs, Vs_obs, EC_obs, P_GPa, T_K)[0]
        logp.append(-err)
    logp = np.asarray(logp)
    m = logp.max()
    return lithology, ibeta, m + np.log(np.mean(np.exp(logp - m)))

def estimate_marginal_over_discrete(
    Vp_obs, Vs_obs, EC_obs, P_GPa, T_K, objective_function, n_per_dim=10
):
    # build small log10 grid to scan discrete combos
    log_phi = np.linspace(-5, -0.01, n_per_dim)
    log_alp = np.linspace(-5, -0.01, n_per_dim)
    log_aec = np.linspace(-5, -0.01, n_per_dim)
    grid = np.array(np.meshgrid(log_phi, log_alp, log_aec)).reshape(3, -1).T

    rows = []
    for lphi, lalp, laec in grid:
        phi, alpha, aec = 10 ** lphi, 10 ** lalp, 10 ** laec
        logL = []
        best = None
        for lith in range(1, 79):
            for ibeta in (1, 2):
                obj, *_ = objective_function([lith, ibeta, phi, alpha, aec],
                                             Vp_obs, Vs_obs, EC_obs, P_GPa, T_K)
                lp = -obj
                logL.append(lp)
                if (best is None) or (lp > best[2]):
                    best = (lith, ibeta, lp)
        logL = np.asarray(logL)
        log_marg = logsumexp(logL) - np.log(len(logL))
        rows.append([phi, alpha, aec, log_marg, best[0], best[1], best[2]])

    df = pd.DataFrame(rows, columns=[
        "phi", "alpha", "alpha_EC",
        "log_marginal_likelihood",
        "lithology", "ibeta", "best_logp"
    ])
    Z = logsumexp(df["log_marginal_likelihood"].values)
    df["posterior_prob"] = np.exp(df["log_marginal_likelihood"] - Z)
    return df.sort_values(by="posterior_prob", ascending=False).reset_index(drop=True)

def monte_carlo_map_from_discrete_groups(
    df_marginal, Vp_obs, Vs_obs, EC_obs, P_GPa, T_K,
    objective_function, output_dir, n_samples_per_group=100_000
):
    os.makedirs(output_dir, exist_ok=True)

    # sharpen to pick viable (lith, ibeta)
    logp = df_marginal["best_logp"].values
    T = 1.0
    df = df_marginal.copy()
    df["sharp_softmax"] = np.exp(logp / T)
    df["sharp_softmax"] /= df["sharp_softmax"].sum()
    grouped = df.groupby(["lithology", "ibeta"])["sharp_softmax"].sum().reset_index()
    df_sel = grouped[grouped["sharp_softmax"] > 1e-4].reset_index(drop=True)

    # global KDE proposal (optional); here re-use all df_marginal
    MC_global = kde_bounded_sample(df_marginal, n_samples=200_000, bounds_log10=(-5, -0.01), n_keep=100_000)

    results = []
    tau = 0.01
    for (lith, ibeta), _g in df_sel.groupby(["lithology", "ibeta"]):
        best = (np.inf, None, None, None, None, None, None)
        log_probs = []
        obj_records = []

        for _, row in MC_global.iterrows():
            phi, alpha, aec = row["phi"], row["alpha"], row["alpha_EC"]
            obj, Vp_m, Vs_m, Sig_m = objective_function(
                [lith, ibeta, phi, alpha, aec], Vp_obs, Vs_obs, EC_obs, P_GPa, T_K
            )
            log_probs.append(-obj)
            obj_records.append([phi, alpha, aec, obj])
            if obj < best[0]:
                best = (obj, phi, alpha, aec, Vp_m, Vs_m, Sig_m)

        log_probs = np.asarray(log_probs)
        m = log_probs.max()
        log_marg = m + np.log(np.mean(np.exp(log_probs - m)))

        # logistic proxy
        c_gl, k_gl = 0.01, 10
        posterior_logistic = 1 / (1 + (-log_probs) / c_gl) ** k_gl
        log_marg_new = np.log(posterior_logistic.sum() + 1e-12)

        good_mask = log_probs > -tau
        mode_count = count_kde_modes(MC_global.iloc[good_mask]) if good_mask.any() else 0

        pd.DataFrame(obj_records, columns=["phi", "alpha", "alpha_EC", "obj"]).to_csv(
            os.path.join(output_dir, f"obj_lith{lith}_ibeta{ibeta}.csv"), index=False
        )

        results.append([
            lith, ibeta,
            best[1], best[2], best[3],
            best[4], best[5], best[6],
            best[0], log_marg, log_marg_new,
            good_mask.mean(), mode_count,
            MC_global["phi"].min(), MC_global["phi"].max(),
            MC_global["alpha"].min(), MC_global["alpha"].max(),
            MC_global["alpha_EC"].min(), MC_global["alpha_EC"].max(),
        ])

    df_best = pd.DataFrame(results, columns=[
        "lithology", "ibeta",
        "phi", "alpha", "alpha_EC",
        "Vp_model", "Vs_model", "Sigma_model",
        "best_obj", "log_marginal_likelihood", "log_marginal_new",
        "posterior_support_volume", "mode_count",
        "phi_min", "phi_max",
        "alpha_min", "alpha_max",
        "alpha_EC_min", "alpha_EC_max",
    ])

    Z1 = logsumexp(df_best["log_marginal_likelihood"].values)
    df_best["posterior_prob"] = np.exp(df_best["log_marginal_likelihood"] - Z1)
    Z2 = logsumexp(df_best["log_marginal_new"].values)
    df_best["posterior_prob_log"] = np.exp(df_best["log_marginal_new"] - Z2)
    return df_best.sort_values(by="posterior_prob_log", ascending=False).reset_index(drop=True)
