# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import numpy as np
import numpy.typing as npt
import pandas as pd

from .stats_utils import Estimate, get_ci


class CovShiftPPIEstimator:
    """
    Covariate Shift Prediction-Powered Inference Estimator.

    This class provides methods to estimate mean values using proxy predictions
    with optional importance weighting for covariate shift correction.
    """

    _proxy_outcome: npt.NDArray[np.floating]
    _primary_outcome: npt.NDArray[np.floating]
    _importance_weight: npt.NDArray[np.floating]
    _n_primary: int
    _n_proxy: int
    _n_overlap: int

    def __init__(
        self,
        df: pd.DataFrame,
        primary_outcome_column: str = "primary_outcome",
        proxy_outcome_column: str = "proxy_outcome",
        importance_weight_column: str = "importance_weight",
        domain_column: str | None = None,
        target_domain_value: object = None,
    ) -> None:
        """
        Initializes the CovShiftPPIEstimator with data from a DataFrame.

        Args:
            df: DataFrame containing the data.
            primary_outcome_column: Column name for primary/true outcome variable.
            proxy_outcome_column: Column name for proxy/predicted outcome variable.
            importance_weight_column: Column name for importance weights.
            domain_column: Optional column name identifying the domain.
            target_domain_value: If provided with domain_column, filter to this domain.
        """
        if domain_column is not None and target_domain_value is not None:
            df = df[df[domain_column] == target_domain_value]

        proxy_outcome = df[proxy_outcome_column].to_numpy()
        primary_outcome = df[primary_outcome_column].to_numpy()
        importance_weight = df[importance_weight_column].to_numpy()

        self._proxy_outcome = proxy_outcome
        self._primary_outcome = primary_outcome
        importance_weight = importance_weight / np.sum(importance_weight)
        self._importance_weight = importance_weight  # normalized
        self._n_primary = int(np.count_nonzero(~np.isnan(primary_outcome)))
        self._n_proxy = int(np.count_nonzero(~np.isnan(proxy_outcome)))
        self._n_overlap = int(
            np.count_nonzero(~np.isnan(primary_outcome) & ~np.isnan(proxy_outcome))
        )

    def compute_proxy_mean_estimate(
        self, weighted: bool = False, confidence_level: float = 0.95
    ) -> Estimate:
        """
        Compute the proxy mean estimate with confidence intervals.

        This method estimates the mean using only the proxy predictions,
        optionally weighted by importance weights for covariate shift correction.

        Args:
            weighted: If True, use importance-weighted estimation.
            confidence_level: Confidence level for intervals (default 0.95 for 95% CI).

        Returns:
            Estimate object containing point estimate, standard error, bounds, and confidence level.
        """
        if weighted:
            proxy_point_estimate = np.sum(self._importance_weight * self._proxy_outcome)
            proxy_variance = np.sum(
                self._importance_weight
                * (self._proxy_outcome - proxy_point_estimate) ** 2
            )
            proxy_standard_error = np.sqrt(proxy_variance / self._n_proxy)
        else:
            proxy_point_estimate = np.mean(self._proxy_outcome)
            proxy_standard_error = np.std(self._proxy_outcome, ddof=1) / np.sqrt(
                self._n_proxy
            )

        lower_bound, upper_bound = get_ci(
            delta=float(proxy_point_estimate),
            se=float(proxy_standard_error),
            conf=int(confidence_level * 100),
        )

        return Estimate(
            confidence_interval_level=confidence_level,
            estimate_val=float(proxy_point_estimate),
            lower_bound=float(lower_bound),
            upper_bound=float(upper_bound),
            sample_size=self._n_proxy,
            standard_error=float(proxy_standard_error),
        )

    def compute_sample_covariance(self, weighted: bool = False) -> float:
        """
        Calculate the sample covariance between primary_outcome and proxy_outcome.

        This method computes the covariance using only the overlapping units
        where both primary_outcome and proxy_outcome are observed (non-missing).

        Args:
            weighted: If True, use importance-weighted covariance estimation.

        Returns:
            The sample covariance between primary_outcome and proxy_outcome.

        Raises:
            ValueError: If there are fewer than 2 overlapping observations.
        """
        overlap_mask = ~np.isnan(self._primary_outcome) & ~np.isnan(self._proxy_outcome)
        primary_outcome_overlap = self._primary_outcome[overlap_mask]
        proxy_outcome_overlap = self._proxy_outcome[overlap_mask]
        importance_weight_overlap = self._importance_weight[overlap_mask]

        if self._n_overlap < 2:
            raise ValueError(
                f"Need at least 2 overlapping observations to compute covariance, "
                f"but only {self._n_overlap} found."
            )

        if weighted:
            importance_weight_normalized = importance_weight_overlap / np.sum(
                importance_weight_overlap
            )
            mean_primary = np.sum(
                importance_weight_normalized * primary_outcome_overlap
            )
            mean_proxy = np.sum(importance_weight_normalized * proxy_outcome_overlap)
            covariance = np.sum(
                importance_weight_normalized
                * (primary_outcome_overlap - mean_primary)
                * (proxy_outcome_overlap - mean_proxy)
            )
            covariance = covariance * self._n_overlap / (self._n_overlap - 1)
        else:
            covariance = np.cov(primary_outcome_overlap, proxy_outcome_overlap, ddof=1)[
                0, 1
            ]

        return float(covariance)

    def compute_primary_mean_estimate(
        self, weighted: bool = False, confidence_level: float = 0.95
    ) -> Estimate:
        """
        Compute the primary mean estimate with confidence intervals.

        This method estimates the mean using only the primary outcomes,
        optionally weighted by importance weights for covariate shift correction.

        Args:
            weighted: If True, use importance-weighted estimation.
            confidence_level: Confidence level for intervals (default 0.95 for 95% CI).

        Returns:
            Estimate object containing point estimate, standard error, bounds, and confidence level.
        """
        if weighted:
            point_estimate = np.sum(self._importance_weight * self._primary_outcome)
            variance = np.sum(
                self._importance_weight * (self._primary_outcome - point_estimate) ** 2
            )
            standard_error = np.sqrt(variance / self._n_primary)
        else:
            point_estimate = np.mean(self._primary_outcome)
            standard_error = np.std(self._primary_outcome, ddof=1) / np.sqrt(
                self._n_primary
            )

        lower_bound, upper_bound = get_ci(
            delta=float(point_estimate),
            se=float(standard_error),
            conf=int(confidence_level * 100),
        )

        return Estimate(
            confidence_interval_level=confidence_level,
            estimate_val=float(point_estimate),
            lower_bound=float(lower_bound),
            upper_bound=float(upper_bound),
            sample_size=self._n_primary,
            standard_error=float(standard_error),
        )

    def compute_ppi_mean_estimate(
        self, weighted: bool = False, confidence_level: float = 0.95
    ) -> Estimate:
        """
        Compute the Prediction-Powered Inference (PPI) mean estimate.

        This method uses both the primary outcomes and proxy predictions to
        compute a debiased estimate. The PPI estimator corrects for prediction
        bias by using the labeled data to estimate and subtract the systematic
        error in the proxy predictions.

        The PPI estimator is: theta_ppi = mean(proxy_outcome) + mean(primary_outcome - proxy_outcome)
        which simplifies to: theta_ppi = mean(primary_outcome) when labels are available.

        For the weighted version, importance weights are applied to handle
        covariate shift between the labeled and target distributions.

        Args:
            weighted: If True, use importance-weighted estimation.
            confidence_level: Confidence level for intervals (default 0.95 for 95% CI).

        Returns:
            Estimate object containing point estimate, standard error, bounds, and confidence level.
        """
        rectifier = self._primary_outcome - self._proxy_outcome

        if weighted:
            proxy_mean = np.sum(self._importance_weight * self._proxy_outcome)
            rectifier_mean = np.sum(self._importance_weight * rectifier)
            point_estimate = proxy_mean + rectifier_mean

            variance_proxy = np.sum(
                self._importance_weight * (self._proxy_outcome - proxy_mean) ** 2
            )
            variance_rectifier = np.sum(
                self._importance_weight * (rectifier - rectifier_mean) ** 2
            )
            covariance = np.sum(
                self._importance_weight
                * (self._proxy_outcome - proxy_mean)
                * (rectifier - rectifier_mean)
            )

            total_variance = variance_proxy + variance_rectifier + 2 * covariance
            standard_error = np.sqrt(total_variance / self._n_primary)
        else:
            proxy_mean = np.mean(self._proxy_outcome)
            rectifier_mean = np.mean(rectifier)
            point_estimate = proxy_mean + rectifier_mean

            variance_proxy = np.var(self._proxy_outcome, ddof=1)
            variance_rectifier = np.var(rectifier, ddof=1)
            covariance = np.cov(self._proxy_outcome, rectifier, ddof=1)[0, 1]

            total_variance = variance_proxy + variance_rectifier + 2 * covariance
            standard_error = np.sqrt(total_variance / self._n_primary)

        lower_bound, upper_bound = get_ci(
            delta=float(point_estimate),
            se=float(standard_error),
            conf=int(confidence_level * 100),
        )

        return Estimate(
            confidence_interval_level=confidence_level,
            estimate_val=float(point_estimate),
            lower_bound=float(lower_bound),
            upper_bound=float(upper_bound),
            sample_size=self._n_primary,
            standard_error=float(standard_error),
        )
