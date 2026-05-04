# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
Estimate-Level Adjustment Module.

This module provides optimization-based methodology for adjusting confidence
intervals to account for unobserved heterogeneity across multiple estimates.

The core model assumes:
    Y_i = delta_i + epsilon_i
    epsilon_i ~ N(0, sigma_i^2)  (known measurement error)
    delta_i ~ N(rho, gamma^2)    (latent true values)

Where:
    - Y_i: Observed estimate for unit i
    - sigma_i: Known standard error of Y_i
    - rho: Mean bias (systematic component)
    - gamma^2: Variance of random biases (heterogeneity)
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize


@dataclass
class FitResult:
    """Result from fitting the latent normal model."""

    rho: float  # Estimated mean bias
    gamma_squared: float  # Estimated variance of biases
    log_likelihood: float  # Log-likelihood at optimum
    success: bool  # Whether optimization succeeded
    message: str  # Optimization message


def negative_log_likelihood(
    params: tuple[float, float],
    Y: npt.NDArray[np.floating[Any]],
    sigma_sq: npt.NDArray[np.floating[Any]],
    constrain_mean_0: bool = False,
) -> float:
    """
    Compute negative log-likelihood for the latent normal model.

    The marginal distribution of Y_i is:
        Y_i ~ N(rho, sigma_i^2 + gamma^2)

    Args:
        params: (rho, gamma_squared) or (gamma_squared,) if constrain_mean_0
        Y: Observed data
        sigma_sq: Known variances (sigma^2)
        constrain_mean_0: If True, rho is fixed at 0

    Returns:
        Negative log-likelihood value
    """
    if constrain_mean_0:
        rho = 0.0
        gamma_sq = params[0]
    else:
        rho, gamma_sq = params

    # Ensure gamma_sq is positive
    if gamma_sq < 0:
        return 1e10

    # Total variance for each observation
    total_var = sigma_sq + gamma_sq

    # Log-likelihood: sum of log N(Y_i; rho, total_var_i)
    log_lik = -0.5 * np.sum(np.log(2 * np.pi * total_var))
    log_lik -= 0.5 * np.sum((Y - rho) ** 2 / total_var)

    return -log_lik


def fit_latent_normal(
    Y: npt.NDArray[np.floating[Any]],
    sigma: npt.NDArray[np.floating[Any]],
    constrain_mean_0: bool = False,
    method: str = "MoM",
) -> FitResult:
    """
    Fit the latent normal model using method of moments (default) or MLE.

    Model:
        Y_i = delta_i + epsilon_i
        epsilon_i ~ N(0, sigma_i^2)  (known)
        delta_i ~ N(rho, gamma^2)    (latent)

    The marginal distribution is:
        Y_i ~ N(rho, sigma_i^2 + gamma^2)

    Method of Moments (equal-weighted):
        rho = (1/K) * sum(Y_k)  (simple average)
        gamma^2 = (1/K) * sum((Y_k - Y_bar)^2) - (1/K) * sum(sigma_k^2)
                = Var(Y) - mean(sigma^2)

    Args:
        Y: Observed data of shape (n,)
        sigma: Known standard deviations for each Y_i, shape (n,)
        constrain_mean_0: If True, constrain rho to be zero
        method: Estimation method - "MoM" (default) or "MLE"

    Returns:
        FitResult with estimated parameters
    """
    # Convert to numpy arrays and handle missing values
    Y_arr = np.asarray(Y, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)

    valid_mask = ~(np.isnan(Y_arr) | np.isnan(sigma_arr))
    Y_arr = Y_arr[valid_mask]
    sigma_arr = sigma_arr[valid_mask]

    if len(Y_arr) == 0:
        raise ValueError("No valid observations after removing NaN values")

    K = len(Y_arr)
    if K < 2:
        raise ValueError("Need at least 2 observations to estimate parameters")

    sigma_sq = sigma_arr**2

    if method == "MoM":
        # Method of Moments (equal-weighted estimator)
        # rho = (1/K) * sum(Y_k)  (simple average)
        if constrain_mean_0:
            rho = 0.0
        else:
            rho = float(np.mean(Y_arr))

        # gamma^2 = (1/K) * sum((Y_k - Y_bar)^2) - (1/K) * sum(sigma_k^2)
        #         = Var(Y) - mean(sigma^2)
        sample_var = float(np.var(Y_arr, ddof=0))  # (1/K) * sum((Y_k - Y_bar)^2)
        mean_sigma_sq = float(np.mean(sigma_sq))  # (1/K) * sum(sigma_k^2)

        gamma_squared_raw = sample_var - mean_sigma_sq
        gamma_squared = max(0.0, gamma_squared_raw)

        return FitResult(
            rho=rho,
            gamma_squared=gamma_squared,
            log_likelihood=np.nan,  # Not computed for MoM
            success=True,
            message="Method of moments estimation",
        )

    elif method == "MLE" or method == "L-BFGS-B":
        # Maximum likelihood estimation
        # Initial values using method of moments
        rho_init = 0.0 if constrain_mean_0 else float(np.mean(Y_arr))
        gamma_sq_init = max(
            float(np.var(Y_arr, ddof=1) - np.mean(sigma_sq)),
            1e-6,
        )

        # Set up optimization
        if constrain_mean_0:
            x0 = [gamma_sq_init]
            bounds = [(1e-10, None)]
        else:
            x0 = [rho_init, gamma_sq_init]
            bounds = [(None, None), (1e-10, None)]

        # Optimize
        result = minimize(
            negative_log_likelihood,
            x0=x0,
            args=(Y_arr, sigma_sq, constrain_mean_0),
            method="L-BFGS-B",
            bounds=bounds,
        )

        if constrain_mean_0:
            rho = 0.0
            gamma_squared = float(result.x[0])
        else:
            rho = float(result.x[0])
            gamma_squared = float(result.x[1])

        return FitResult(
            rho=rho,
            gamma_squared=gamma_squared,
            log_likelihood=-result.fun,
            success=result.success,
            message=str(result.message),
        )

    else:
        raise ValueError(f"Unknown method: {method}. Use 'MoM' or 'MLE'.")
