# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm

from .adjustment_model import em_latent_normal


def leave_one_out_validation(
    domain_estimates: pd.DataFrame,
    diff_est_col: str,
    diff_sd_col: str,
    proxy_est_col: str,
    proxy_sd_col: str,
    em_params: dict[str, float | bool] | None = None,
) -> pd.DataFrame:
    """
    Performs hold-one-out validation for the adjustment methodology by iteratively removing each domain from the input DataFrame,
    re-estimating the model parameters without that domain, and computing a corrected proxy standard deviation for each domain.
    The corrected values are added as a new column to the DataFrame.
    """
    if em_params is None:
        em_params = {
            "max_iter": 50000,
            "tol": 10 ** (-9),
            "verbose": True,
            "constrain_mean_0": False,
        }

    max_iter = int(em_params["max_iter"])
    tol = float(em_params["tol"])
    verbose = bool(em_params["verbose"])
    constrain_mean_0 = bool(em_params["constrain_mean_0"])

    new_total_proxy_sd = []
    new_corrected_means = []
    for i, row in domain_estimates.iterrows():
        domain_estimates_loo = domain_estimates.drop(index=i)
        Y = domain_estimates_loo[diff_est_col]
        sigma = domain_estimates_loo[diff_sd_col]
        rho, gamma2, _history = em_latent_normal(
            Y,
            sigma,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            constrain_mean_0=constrain_mean_0,
        )
        new_total_proxy_sd.append(np.sqrt(gamma2 + row[proxy_sd_col] ** 2))
        new_corrected_means.append(row[proxy_est_col] + rho)
    domain_estimates[proxy_est_col + "_corrected"] = np.array(new_corrected_means)
    domain_estimates[proxy_sd_col + "_corrected"] = np.array(new_total_proxy_sd)
    return domain_estimates


def coverage_calibration_curve(
    mean_primary: NDArray[np.float64],
    sigma_primary: NDArray[np.float64],
    mean_proxy: NDArray[np.float64],
    sigma_proxy: NDArray[np.float64],
    mean_proxy_corr: NDArray[np.float64],
    sigma_proxy_corr: NDArray[np.float64],
    alpha_seq: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if alpha_seq is None:
        alpha_seq = np.linspace(1, 0.005, 200)

    base_coverage = []
    corr_coverage = []
    for alpha in alpha_seq:
        z_thresh = norm.ppf(1 - alpha / 2)

        # Naive coverage
        crossover_1 = (mean_proxy + z_thresh * sigma_proxy) >= (
            mean_primary - z_thresh * sigma_primary
        )
        crossover_2 = (mean_proxy - z_thresh * sigma_proxy) <= (
            mean_primary + z_thresh * sigma_primary
        )
        cover_intervals = crossover_1 & crossover_2

        # Corrected coverage
        crossover_corr_1 = (mean_proxy_corr + z_thresh * sigma_proxy_corr) >= (
            mean_primary - z_thresh * sigma_primary
        )
        crossover_corr_2 = (mean_proxy_corr - z_thresh * sigma_proxy_corr) <= (
            mean_primary + z_thresh * sigma_primary
        )
        cover_corr_intervals = crossover_corr_1 & crossover_corr_2

        base_coverage.append(np.mean(cover_intervals))
        corr_coverage.append(np.mean(cover_corr_intervals))
    return np.array(base_coverage), np.array(corr_coverage), alpha_seq
