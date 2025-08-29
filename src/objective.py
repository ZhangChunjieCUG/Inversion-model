"""
Objective function and forward-model wrapper.
NOTE: This module calls your NON-PUBLIC forward model:
- vcalc_HI1
- get_lithology_info, calc_dry_v_all6, read_dens_fluid, interpolate, calc_VpVsEC11
"""

from __future__ import annotations
import numpy as np

# ---- forward-model imports (non-public) ----
try:
    from vcalc_HI1 import vcalc_HI1
    from Clucate_VP_VS_EC import (
        get_lithology_info, calc_dry_v_all6, read_dens_fluid, interpolate, calc_VpVsEC11
    )
except Exception as e:
    _FORWARD_IMPORT_ERROR = e
    vcalc_HI1 = None
    get_lithology_info = calc_dry_v_all6 = read_dens_fluid = interpolate = calc_VpVsEC11 = None

# ---- global constants used by objective ----
# composition / solution parameters used by your forward model
S_NACL_WT = 5.0   # NaCl wt% in aqueous fluid
W_H2O_WT  = 5.0   # H2O wt% in silicate melt
# Example bulk composition (used to build C_melt)
SIO2, TIO2, AL2O3, FEO, FE2O3, MNO, MGO, CAO, NA2O, K2O, P2O5 = (
    51.381, 0.761, 18.240, 9.666, 0.0, 0.176, 7.157, 10.401, 1.914, 0.119, 0.086
)
C_MELT = [SIO2, TIO2, AL2O3, FEO, FE2O3, MNO, MGO, CAO, NA2O, K2O, P2O5, S_NACL_WT]

# density/κ data (you can pass these from caller if you prefer)
def load_dens_fluid_table(path: str):
    if read_dens_fluid is None:
        raise ImportError(
            f"Forward model not available: {repr(_FORWARD_IMPORT_ERROR)}.\n"
            "Please provide your own forward-model package or stub."
        )
    return read_dens_fluid(path)


def objective_function(
    params, Vp_obs, Vs_obs, EC_obs, p_GPa, T_K,
    dens_table=None
):
    """
    Calculate misfit between observations (Vp, Vs, sigma) and forward-model predictions.
    params = [rock_type (int), ibeta (int), phi, alpha, alpha_ec]
    Returns: total_error, Vp_model, Vs_model, EC_model
    """
    if get_lithology_info is None:
        raise ImportError(
            f"Forward model not available: {repr(_FORWARD_IMPORT_ERROR)}.\n"
            "You must provide your own forward-model implementation."
        )

    rock_type, ibeta, phi, alpha, alpha_ec = int(params[0]), int(params[1]), params[2], params[3], params[4]

    # forward-model precomputations
    lithology_name, lithology_type, solid_density, _ = get_lithology_info(rock_type)
    vp0, vs0, _ = calc_dry_v_all6(rock_type, p_GPa, T_K)
    Ks = solid_density * (vp0 ** 2 - 4.0 * vs0 ** 2 / 3.0) * 1e-3

    # fluid / melt branch
    if ibeta == 1 or (ibeta == 2 and p_GPa <= 0.01):
        if dens_table is None:
            raise ValueError("Please pass dens_table=(pres,temp,conc,rho,kappa) from read_dens_fluid().")
        pres, temp, conc, rho, kappa = dens_table
        rh, kap = interpolate(p_GPa, T_K, S_NACL_WT, pres, temp, conc, rho, kappa)
        beta = Ks * kap
        ro = rh / solid_density
        vf0 = np.sqrt(1000.0 / (kap * rh))
    else:
        rho_m, Km = vcalc_HI1(p_GPa, T_K, C_MELT)
        beta = Ks / Km
        ro = rho_m / solid_density
        vf0 = np.sqrt(Km / rho_m)

    # main forward evaluation
    ec0, Vp_model, Vs_model, EC_model, ecf = calc_VpVsEC11(
        rock_type, ibeta, phi, alpha, alpha_ec, beta, ro, p_GPa, T_K,
        S_NACL_WT, W_H2O_WT, SIO2, NA2O, vp0, vs0, vf0
    )

    # observation error model
    sigma_vp = 0.03 * Vp_obs
    sigma_vs = 0.03 * Vs_obs
    sigma_ec = 0.5
    log_sigma_obs = np.log10(EC_obs)
    log_sigma_model = np.log10(EC_model)

    error_vp = (Vp_obs - Vp_model) ** 2 / (2 * sigma_vp ** 2)
    error_vs = (Vs_obs - Vs_model) ** 2 / (2 * sigma_vs ** 2)
    nu = 3
    error_sigma = 0.5 * (nu + 1) * np.log(1 + ((log_sigma_obs - log_sigma_model) ** 2) / (nu * sigma_ec ** 2))

    total_error = error_vp + error_vs + error_sigma
    return total_error, Vp_model, Vs_model, EC_model
