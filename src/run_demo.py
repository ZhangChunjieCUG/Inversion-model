import os
import numpy as np
import pandas as pd

from src import (
    objective_function, load_dens_fluid_table,
    estimate_marginal_over_discrete, monte_carlo_map_from_discrete_groups,
    create_results_folder
)

# prepare forward-model and your data
dens_file = "package_20210417_forCJ/SEFMO_20210417/dens_fluid.dat"
dens_table = load_dens_fluid_table(dens_file)
Vp_obs, Vs_obs, EC_obs = 6.12, 3.49, 0.0046
P_GPa, T_K = 0.20, 465.0

# magrnilzation discrete 
df_grid = estimate_marginal_over_discrete(
    Vp_obs, Vs_obs, EC_obs, P_GPa, T_K,
    lambda pars, *args: objective_function(pars, *args, dens_table=dens_table),
    n_per_dim=10
)

outdir = create_results_folder("results_demo")
df_best = monte_carlo_map_from_discrete_groups(
    df_marginal=df_grid,
    Vp_obs=Vp_obs, Vs_obs=Vs_obs, EC_obs=EC_obs,
    P_GPa=P_GPa, T_K=T_K,
    objective_function=lambda pars, *a: objective_function(pars, *a, dens_table=dens_table),
    output_dir=os.path.join(outdir, "mc_groups"),
    n_samples_per_group=100000
)
print(df_best.head())

