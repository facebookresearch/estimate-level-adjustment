# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import unittest

import numpy as np
from estimate_level_adjustment.adjustment_model import em_latent_normal
from numpy.typing import NDArray


class EmLatentNormalTest(unittest.TestCase):
    def test_all_valid_values(self) -> None:
        # Setup: Create valid input arrays without nulls
        Y: NDArray[np.float64] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sigma: NDArray[np.float64] = np.array([0.1, 0.2, 0.15, 0.25, 0.3])

        # Execute: Run the EM algorithm with all valid values
        rho, gamma2, history = em_latent_normal(Y, sigma, max_iter=100, tol=1e-9)

        # Assert: Check that the function returns valid results
        self.assertIsInstance(rho, float)
        self.assertIsInstance(gamma2, float)
        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)
        # Check that gamma2 is non-negative
        self.assertGreaterEqual(gamma2, 0.0)

    def test_with_nulls_in_y_and_sigma(self) -> None:
        # Setup: Create arrays with NaN values in both Y and sigma
        Y: NDArray[np.float64] = np.array([1.0, np.nan, 3.0, 4.0, np.nan])
        sigma: NDArray[np.float64] = np.array([0.1, 0.2, np.nan, 0.25, 0.3])

        # Execute: Run the EM algorithm with nulls - function should filter them out
        rho, gamma2, history = em_latent_normal(Y, sigma, max_iter=100, tol=1e-9)

        # Assert: Check that the function returns valid results despite nulls
        self.assertIsInstance(rho, float)
        self.assertIsInstance(gamma2, float)
        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)
        # Check that gamma2 is non-negative
        self.assertGreaterEqual(gamma2, 0.0)

    def test_with_negative_sigma_values(self) -> None:
        # Setup: Create arrays with negative sigma values
        Y: NDArray[np.float64] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sigma: NDArray[np.float64] = np.array([0.1, -0.2, 0.15, -0.25, 0.3])

        # Execute: Run the EM algorithm with negative sigma values
        # Note: The function doesn't explicitly validate negative sigmas,
        # but they will produce valid mathematical results (sigma is squared internally)
        rho, gamma2, history = em_latent_normal(Y, sigma, max_iter=100, tol=1e-9)

        # Assert: Check that the function completes and returns results
        self.assertIsInstance(rho, float)
        self.assertIsInstance(gamma2, float)
        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)
        self.assertGreaterEqual(gamma2, 0.0)

    def test_all_null_values(self) -> None:
        # Setup: Create arrays where all values are NaN
        Y: NDArray[np.float64] = np.array([np.nan, np.nan, np.nan])
        sigma: NDArray[np.float64] = np.array([np.nan, np.nan, np.nan])

        # Execute: The function should handle this edge case
        # After filtering, arrays will be empty, which should cause computation with empty arrays
        # We expect NaN results since mean of empty arrays is NaN
        rho, gamma2, history = em_latent_normal(Y, sigma, max_iter=100, tol=1e-9)

        # Assert: With all null values, results will be NaN
        self.assertTrue(np.isnan(rho))
        self.assertTrue(np.isnan(gamma2))
        self.assertIsInstance(history, list)
        # The function will run all iterations since convergence check with NaN always fails
        self.assertGreater(len(history), 1)
