# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
import numpy as np
from numpy.typing import NDArray


def em_latent_normal(
    Y: NDArray[np.float64],
    sigma: NDArray[np.float64],
    max_iter: int = 10000,
    tol: float = 1e-12,
    verbose: bool = False,
    constrain_mean_0: bool = False,
    gamma_squared_initial_boundary: float = 10**-5,
) -> tuple[float, float, list[tuple[float, float]]]:
    """
    EM algorithm for latent normal model:
        Y_i = delta_i + epsilon_i
        epsilon_i ~ N(0, sigma_i^2) (known)
        delta_i ~ N(rho, theta^2) (latent)
    Args:
        Y: array-like, observed data (n,) - accepts list or np.ndarray
        sigma: array-like, known std deviations for each Y_i (n,) - accepts list or np.ndarray
        max_iter: maximum number of EM iterations
        tol: convergence tolerance for parameter updates
        verbose: print progress if True
        constrain_mean_0: constrains the mean (rho) to be zero
        gamma_squared_initial_boundary: initial value for gamma2, to avoid negative/zero values on initialization.
    Returns:
        rho: estimated mean correction
        gamma2: estimated dialtion factor (variance of the random biases)
        history: list of (rho, gamma2) for each iteration
    """
    # Convert inputs to numpy arrays if they aren't already
    Y_arr = np.asarray(Y, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)

    missing_idx = (np.isnan(Y_arr)) | (np.isnan(sigma_arr))

    Y_arr = Y_arr[~missing_idx]
    sigma_arr = sigma_arr[~missing_idx]
    # Validate that Y and sigma have the same length
    if Y_arr.shape != sigma_arr.shape:
        raise ValueError(
            f"Y and sigma must have the same length. Got Y.shape={Y_arr.shape}, sigma.shape={sigma_arr.shape}"
        )

    # Initialize parameters
    rho = np.mean(Y_arr)
    if constrain_mean_0:
        rho = 0.0
    gamma2: float = max(
        float(np.mean((Y_arr - rho) ** 2) - np.mean(sigma_arr**2)),
        gamma_squared_initial_boundary,
    )  # avoid negative/zero

    history = [(rho, gamma2)]

    for it in range(max_iter):
        # E-step: compute posterior mean and variance for each delta_i
        sigma_squared: NDArray[np.float64] = np.square(sigma_arr)
        inv_sigma_squared: NDArray[np.float64] = np.reciprocal(sigma_squared)
        inv_gamma2: float = 1.0 / gamma2
        precision_sum: NDArray[np.float64] = (
            np.full_like(sigma_squared, inv_gamma2) + inv_sigma_squared
        )
        v: NDArray[np.float64] = np.divide(1.0, precision_sum)  # shape (n,)
        weighted_obs: NDArray[np.float64] = np.multiply(Y_arr, inv_sigma_squared)
        rho_contribution: NDArray[np.float64] = np.full_like(Y_arr, rho * inv_gamma2)
        mean_numerator: NDArray[np.float64] = np.add(rho_contribution, weighted_obs)
        m: NDArray[np.float64] = np.multiply(v, mean_numerator)  # shape (n,)

        # M-step: update rho and gamma2
        rho_new = np.mean(m)
        if constrain_mean_0:
            rho_new = 0.0
        gamma2_new = np.mean(v + (m - rho_new) ** 2)

        history.append((rho_new, gamma2_new))

        # Check convergence
        if np.abs(rho_new - rho) < tol and np.abs(gamma2_new - gamma2) < tol:
            if verbose:
                print(f"Converged at iteration {it + 1}")
            break

        rho, gamma2 = rho_new, gamma2_new

    return rho, gamma2, history
