"""KDE-based posterior sampling and diagnostics."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
from scipy.integrate import simpson

def kde_bounded_sample(df_posterior, n_samples=50_000, bounds_log10=(-5, -0.01), n_keep=10_000):
    """Fit KDE in log10-space (phi, alpha, alpha_EC) with weights=df['posterior_prob'] and resample within bounds."""
    log_samples = np.log10(df_posterior[["phi", "alpha", "alpha_EC"]].values.T)
    w = df_posterior["posterior_prob"].to_numpy()
    w = w / w.sum()

    kde = gaussian_kde(log_samples, weights=w)

    accepted = []
    lo, hi = bounds_log10
    while len(accepted) < n_keep:
        samples_log = kde.resample(n_samples).T
        mask = np.all((samples_log >= lo) & (samples_log <= hi), axis=1)
        accepted.extend(samples_log[mask])
        if len(accepted) > n_keep:
            accepted = accepted[:n_keep]
    accepted = np.asarray(accepted)
    return pd.DataFrame(10 ** accepted, columns=["phi", "alpha", "alpha_EC"])

def count_kde_modes(samples_df, bandwidth=0.15, grid_size=100):
    """Project to 1D by PCA and count peaks on KernelDensity score along that axis."""
    log_samples = np.log10(samples_df[["phi", "alpha", "alpha_EC"]].values)
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian").fit(log_samples)
    pca = PCA(n_components=1)
    proj = pca.fit_transform(log_samples)
    grid = np.linspace(proj.min(), proj.max(), grid_size).reshape(-1, 1)
    # inverse-transform the 1D grid back to 3D for scoring
    back = pca.inverse_transform(grid)
    log_dens = kde.score_samples(back)
    peaks, _ = find_peaks(log_dens)
    return len(peaks)

def kde_posterior_volume(samples_df, posterior_weights=None, bandwidth=0.2, n_grid=100):
    """Crude 1D integral of KDE along the first PCA axis as a proxy of 'high-density volume'."""
    log_samples = np.log10(samples_df[["phi", "alpha", "alpha_EC"]].values)
    if len(log_samples) < 2:
        return np.nan
    if posterior_weights is None:
        posterior_weights = np.ones(log_samples.shape[0]) / log_samples.shape[0]

    kde = gaussian_kde(log_samples.T, weights=posterior_weights, bw_method=bandwidth)
    pca = PCA(n_components=1)
    proj = pca.fit_transform(log_samples).flatten()
    grid = np.linspace(proj.min(), proj.max(), n_grid).reshape(-1, 1)
    back = pca.inverse_transform(grid)

    log_vals = kde(back.T)
    log_vals -= np.max(log_vals)
    dens = np.exp(log_vals)
    dens = np.nan_to_num(dens, nan=0.0, posinf=0.0, neginf=0.0)
    return simpson(dens, x=grid.flatten())
