# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import norm

from .adjustment import fit_latent_normal, FitResult
from .cov_shift_estimator import CovShiftPPIEstimator
from .stats_utils import Estimate

# TODO: rename the domain_estimate_stats to hold_one_domain_out_estimate_stats


@dataclass
class DomainEstimateStats:
    """Per-domain estimate statistics for latent model fitting."""

    domain: Any
    n_samples: int
    # Basic estimates
    proxy_mean: float
    proxy_se: float
    primary_mean: float
    primary_se: float
    # Difference (primary - proxy) estimate and variance
    est_diff: float
    est_diff_var: float
    # PPI Estimates
    ppi_mean: float | None = None
    ppi_se: float | None = None
    # PPI rectifier components (when using cross-domain weights)
    rectifier: float | None = None
    rectifier_var: float | None = None
    # PPI error: primary - ppi_mean
    ppi_error: float | None = None
    ppi_error_var: float | None = None


@dataclass
class AdjustedEstimate:
    """Extended estimate with adjustment information."""

    confidence_interval_level: float
    estimate_val: float
    lower_bound: float
    upper_bound: float
    standard_error: float
    sample_size: int

    # Adjustment parameters
    gamma_squared: float
    rho: float


@dataclass
class BootstrapEstimate:
    """Estimate with bootstrap confidence intervals."""

    estimate_val: float
    lower_bound: float
    upper_bound: float
    standard_error: float
    sample_size: int
    confidence_interval_level: float
    bootstrap_lower_bound: float
    bootstrap_upper_bound: float
    n_bootstrap: int


@dataclass
class DomainBootstrapEstimate:
    """
    Estimate with domain-bootstrap confidence intervals.

    This uses estimate-level resampling (resampling domain-level estimate_diff_k values)
    rather than observation-level resampling, and accounts for metric uncertainty.
    """

    estimate_val: float
    lower_bound: float
    upper_bound: float
    standard_error: float
    sample_size: int
    confidence_interval_level: float
    domain_bootstrap_lower_bound: float
    domain_bootstrap_upper_bound: float
    domain_bootstrap_point_estimate: float
    n_bootstrap: int
    rho_mean: float  # Mean of bootstrap rho estimates
    gamma_squared_mean: float  # Mean of bootstrap gamma^2 estimates


class AdjustedCovShiftPPIEstimator:
    """
    Covariate Shift PPI Estimator with Estimate-Level Adjustment.

    Computes domain-level PPI estimate errors and learns a latent model
    to account for cross-domain heterogeneity.

    The latent model assumes:
        estimate_diff_k ~ N(rho, sigma_k^2 + gamma^2)

    Where:
        estimate_diff_k: Domain-level estimate (difference or PPI error)
        sigma_k: Standard error of domain estimate
        rho: Mean bias
        gamma^2: Cross-domain heterogeneity variance
    """

    _df: pd.DataFrame
    _domain_column: str
    _target_domain_value: Any
    _primary_outcome_column: str
    _proxy_outcome_column: str
    _importance_weight_column: str
    _cross_domain_weight_pattern: str | None
    _base_estimator: CovShiftPPIEstimator | None

    def __init__(
        self,
        df: pd.DataFrame,
        primary_outcome_column: str = "primary_outcome",
        proxy_outcome_column: str = "proxy_outcome",
        importance_weight_column: str = "importance_weight",
        domain_column: str = "domain",
        target_domain_value: Any = None,
        cross_domain_weight_pattern: str | None = None,
    ) -> None:
        """
        Initialize the AdjustedCovShiftPPIEstimator.

        Args:
            df: DataFrame containing the data.
            primary_outcome_column: Column name for primary/true outcome.
            proxy_outcome_column: Column name for proxy/predicted outcome.
            importance_weight_column: Column name for importance weights.
            domain_column: Column name identifying the domain.
            target_domain_value: Value identifying the target domain.
            cross_domain_weight_pattern: Pattern for cross-domain weight columns.
                Use {target_domain} as placeholder. Example: "weight_to_domain_{target_domain}"
        """
        self._df = df
        self._domain_column = domain_column
        self._target_domain_value = target_domain_value
        self._primary_outcome_column = primary_outcome_column
        self._proxy_outcome_column = proxy_outcome_column
        self._importance_weight_column = importance_weight_column
        self._cross_domain_weight_pattern = cross_domain_weight_pattern
        self._base_estimator = None

    def _get_sorted_domains(self) -> list[Any]:
        """Get sorted unique domain values."""
        return sorted(self._df[self._domain_column].unique())

    def _get_base_estimator(self) -> CovShiftPPIEstimator:
        """Get or create the base CovShiftPPIEstimator (cached for efficiency)."""
        if self._base_estimator is None:
            self._base_estimator = CovShiftPPIEstimator(
                df=self._df,
                primary_outcome_column=self._primary_outcome_column,
                proxy_outcome_column=self._proxy_outcome_column,
                importance_weight_column=self._importance_weight_column,
                domain_column=self._domain_column,
                target_domain_value=self._target_domain_value,
            )
        return self._base_estimator

    def _get_valid_values(
        self,
        proxy: npt.NDArray[np.floating[Any]],
        primary: npt.NDArray[np.floating[Any]],
    ) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
        """Filter out NaN values from proxy and primary arrays."""
        valid_mask = ~(np.isnan(proxy) | np.isnan(primary))
        return proxy[valid_mask], primary[valid_mask]

    def _precompute_cross_domain_data(
        self,
        domains_no_target: list[Any],
        domain_col: npt.NDArray[Any],
        df: pd.DataFrame,
        weighted: bool,
    ) -> tuple[dict[Any, npt.NDArray[np.bool_]], dict[Any, npt.NDArray[np.float64]]]:
        """Pre-compute source masks and weight columns for cross-domain PPI."""
        cross_domain_weight_pattern = self._cross_domain_weight_pattern
        source_masks: dict[Any, npt.NDArray[np.bool_]] = {}
        weight_cols: dict[Any, npt.NDArray[np.float64]] = {}

        if cross_domain_weight_pattern is None:
            return source_masks, weight_cols

        for domain in domains_no_target:
            if self._target_domain_value is not None:
                source_masks[domain] = (domain_col != domain) & (
                    domain_col != self._target_domain_value
                )
            else:
                source_masks[domain] = domain_col != domain

        if weighted:
            missing_weight_cols: list[str] = []
            for domain in domains_no_target:
                weight_col_name = cross_domain_weight_pattern.format(
                    target_domain=domain
                )
                if weight_col_name in df.columns:
                    weight_cols[domain] = df[weight_col_name].values.astype(np.float64)
                else:
                    missing_weight_cols.append(weight_col_name)
            if len(weight_cols) == 0:
                raise ValueError(
                    f"No cross-domain weight columns found in DataFrame. "
                    f"Expected columns matching pattern "
                    f"'{cross_domain_weight_pattern}' "
                    f"(e.g., {missing_weight_cols}). "
                    f"Available columns: {list(df.columns)}"
                )

        return source_masks, weight_cols

    @staticmethod
    def _compute_domain_ppi_rectifier(
        rect_proxy_raw: npt.NDArray[np.floating[Any]],
        rect_primary_raw: npt.NDArray[np.floating[Any]],
        weights_raw: npt.NDArray[np.float64] | None,
        proxy_mean: float,
        var_proxy: float,
        primary_mean: float,
        var_primary: float,
        cov_primary_proxy: float,
    ) -> tuple[
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
    ]:
        """
        Compute PPI rectifier from other source domains' data.

        Returns (ppi_mean, ppi_var, rectifier, rectifier_var, ppi_error,
        ppi_error_var), or all Nones if insufficient valid data.
        """
        rect_valid_mask = ~(np.isnan(rect_proxy_raw) | np.isnan(rect_primary_raw))
        rect_proxy = rect_proxy_raw[rect_valid_mask]
        rect_primary = rect_primary_raw[rect_valid_mask]
        n_rect = len(rect_primary)

        if n_rect == 0:
            return None, None, None, None, None, None

        if weights_raw is not None:
            weights = weights_raw[rect_valid_mask]
        else:
            weights = np.ones(n_rect)

        weights = weights / np.sum(weights)

        d_values = rect_primary - rect_proxy
        rectifier_mean = float(np.sum(weights * d_values))

        # Weighted variance using effective sample size (Kish formula)
        var_weighted = np.sum(weights * (d_values - rectifier_mean) ** 2)
        n_eff = 1.0 / np.sum(weights**2)
        var_rectifier = var_weighted / (n_eff - 1)

        ppi_mean = proxy_mean + rectifier_mean
        ppi_var = var_proxy + var_rectifier
        ppi_error = primary_mean - ppi_mean
        ppi_error_var = var_primary + ppi_var - 2 * cov_primary_proxy

        return (
            ppi_mean,
            ppi_var,
            rectifier_mean,
            var_rectifier,
            ppi_error,
            ppi_error_var,
        )

    # NOTE: This should be able to be computed a single time here regardless of the type of estimate. It's only used for the PPI estimate, but it's computed for the proxy and primary estimates as well.
    def compute_hold_one_domain_out_estimate_stats(
        self,
        weighted: bool = True,
    ) -> list[DomainEstimateStats]:
        """
        Compute per-domain estimate statistics for all source domains.

        For each domain (excluding target), computes:
        - Basic stats: proxy_mean, primary_mean, and their SEs
        - Difference estimate: primary - proxy and its variance
        - PPI rectifier: weighted (primary - proxy) from OTHER source domains
        - PPI error: primary - (proxy + rectifier)
        - The weighted PPI error must be weighted to the specific target domain, hence the use of _cross_domain_weight_pattern

        Handles missing values in primary/proxy columns by excluding them.

        Args:
            weighted: If True, use importance weights for rectifier computation.

        Returns:
            List of DomainEstimateStats for each source domain.
        """
        domains = self._get_sorted_domains()

        # Source domains = all domains except target
        if self._target_domain_value is not None:
            domains_no_target = [d for d in domains if d != self._target_domain_value]
        else:
            domains_no_target = domains

        # Pre-extract columns as numpy arrays (faster than repeated DataFrame access)
        df = self._df
        domain_col = df[self._domain_column].values
        proxy_col = df[self._proxy_outcome_column].values.astype(np.float64)
        primary_col = df[self._primary_outcome_column].values.astype(np.float64)

        cross_domain_weight_pattern = self._cross_domain_weight_pattern
        source_masks, weight_cols = self._precompute_cross_domain_data(
            domains_no_target, domain_col, df, weighted
        )

        domain_stats = []

        for domain in domains_no_target:
            # Use numpy boolean indexing instead of DataFrame filtering
            domain_mask = domain_col == domain
            proxy_raw = proxy_col[domain_mask]
            primary_raw = primary_col[domain_mask]

            # Get primary and proxy values, handling missing values
            proxy, primary = self._get_valid_values(proxy_raw, primary_raw)
            n = len(proxy)

            if n < 2:
                continue

            # Basic statistics
            proxy_mean = float(np.mean(proxy))
            primary_mean = float(np.mean(primary))

            # Difference estimate (primary - proxy) and its variance
            # Compute variance and covariance in fewer passes
            est_diff = primary_mean - proxy_mean
            proxy_centered = proxy - proxy_mean
            primary_centered = primary - primary_mean
            var_primary = float(np.sum(primary_centered**2) / (n - 1) / n)
            var_proxy = float(np.sum(proxy_centered**2) / (n - 1) / n)
            proxy_se = float(np.sqrt(var_proxy))
            primary_se = float(np.sqrt(var_primary))
            cov_primary_proxy = float(
                np.sum(primary_centered * proxy_centered) / (n - 1) / n
            )
            est_diff_var = var_primary + var_proxy - 2 * cov_primary_proxy

            # Initialize PPI components
            # Default PPI to proxy values (no rectifier case)
            ppi_mean: float | None = None
            ppi_var: float | None = None
            rectifier: float | None = None
            rectifier_var: float | None = None
            ppi_error: float | None = None
            ppi_error_var: float | None = None

            if cross_domain_weight_pattern is not None:
                rect_mask = source_masks[domain]
                raw_weights = (
                    weight_cols[domain][rect_mask] if domain in weight_cols else None
                )
                (
                    ppi_mean,
                    ppi_var,
                    rectifier,
                    rectifier_var,
                    ppi_error,
                    ppi_error_var,
                ) = self._compute_domain_ppi_rectifier(
                    rect_proxy_raw=proxy_col[rect_mask],
                    rect_primary_raw=primary_col[rect_mask],
                    weights_raw=raw_weights,
                    proxy_mean=proxy_mean,
                    var_proxy=var_proxy,
                    primary_mean=primary_mean,
                    var_primary=var_primary,
                    cov_primary_proxy=cov_primary_proxy,
                )

            domain_stats.append(
                DomainEstimateStats(
                    domain=domain,
                    n_samples=n,
                    proxy_mean=proxy_mean,
                    proxy_se=proxy_se,
                    primary_mean=primary_mean,
                    primary_se=primary_se,
                    est_diff=est_diff,
                    est_diff_var=max(est_diff_var, 0),
                    ppi_mean=ppi_mean,
                    ppi_se=float(np.sqrt(ppi_var)) if ppi_var is not None else None,
                    rectifier=rectifier,
                    rectifier_var=rectifier_var,
                    ppi_error=ppi_error,
                    ppi_error_var=ppi_error_var,
                )
            )

        return domain_stats

    def estimate_latent_parameters(
        self,
        use_ppi_errors: bool = True,
        weighted: bool = True,
        method: str = "MoM",
    ) -> FitResult:
        """
        Estimate latent model parameters (rho, gamma^2) from domain-level estimates.

        Fits the latent normal model:
            estimate_diff_k ~ N(rho, sigma_k^2 + gamma^2)

        Where estimate_diff_k is either:
        - PPI error (primary - rectified_proxy) if use_ppi_errors=True
        - Simple difference (primary - proxy) if use_ppi_errors=False

        Args:
            use_ppi_errors: If True, use PPI errors; otherwise use simple differences.
            weighted: If True, use importance weights for rectifier computation.
            method: Estimation method - "MoM" (default) or "MLE".

        Returns:
            FitResult with estimated rho and gamma_squared.
        """
        domain_stats = self.compute_hold_one_domain_out_estimate_stats(
            weighted=weighted
        )

        if weighted and self._cross_domain_weight_pattern is None:
            raise ValueError(
                "Weighted estimation requires specifying cross-domain weights via "
                "`cross_domain_weight_pattern`. Provide a pattern like "
                "'weight_to_domain_{target_domain}' or set weighted=False."
            )

        if use_ppi_errors:
            # Use PPI errors (primary - rectified_proxy)
            valid_stats = [s for s in domain_stats if s.ppi_error is not None]
            estimate_diff_k = np.array([s.ppi_error for s in valid_stats])
            sigma_k = np.sqrt(np.array([s.ppi_error_var for s in valid_stats]))
        else:
            # Use simple differences (primary - proxy)
            estimate_diff_k = np.array([s.est_diff for s in domain_stats])
            sigma_k = np.sqrt(np.array([s.est_diff_var for s in domain_stats]))

        return fit_latent_normal(
            Y=estimate_diff_k,
            sigma=sigma_k,
            constrain_mean_0=False,
            method=method,
        )

    def compute_ppi_mean_estimate(
        self,
        weighted: bool = False,
        confidence_level: float = 0.95,
    ) -> Estimate:
        """Compute PPI mean estimate (delegates to base estimator)."""
        return self._get_base_estimator().compute_ppi_mean_estimate(
            weighted=weighted,
            confidence_level=confidence_level,
        )

    def compute_proxy_mean_estimate(
        self,
        confidence_level: float = 0.95,
    ) -> Estimate:
        """Compute proxy mean estimate (delegates to base estimator)."""
        return self._get_base_estimator().compute_proxy_mean_estimate(
            weighted=False,
            confidence_level=confidence_level,
        )

    def compute_primary_mean_estimate(
        self,
        confidence_level: float = 0.95,
    ) -> Estimate:
        """Compute primary mean estimate (delegates to base estimator)."""
        return self._get_base_estimator().compute_primary_mean_estimate(
            confidence_level=confidence_level,
        )

    def compute_adjusted_ppi_estimate(
        self,
        weighted: bool = True,
        confidence_level: float = 0.95,
        method: str = "MoM",
    ) -> AdjustedEstimate:
        """
        Compute PPI estimate with adjusted confidence intervals.

        The adjusted SE accounts for cross-domain heterogeneity:
            adjusted_se = sqrt(base_se^2 + gamma^2)

        Args:
            weighted: If True, use importance weights.
            confidence_level: Confidence level for intervals.
            method: Latent model estimation method ("MoM" or "MLE").

        Returns:
            AdjustedEstimate with original and adjusted bounds.
        """
        base_estimate = self.compute_ppi_mean_estimate(
            weighted=weighted,
            confidence_level=confidence_level,
        )

        fit_result = self.estimate_latent_parameters(
            use_ppi_errors=True,
            weighted=weighted,
            method=method,
        )
        gamma_squared = fit_result.gamma_squared
        rho = fit_result.rho

        total_variance = base_estimate.standard_error**2 + gamma_squared
        adjusted_est = base_estimate.estimate_val + rho
        adjusted_se = float(np.sqrt(total_variance))

        z = norm.ppf(1 - (1 - confidence_level) / 2)
        adjusted_lower = adjusted_est - z * adjusted_se
        adjusted_upper = adjusted_est + z * adjusted_se

        return AdjustedEstimate(
            confidence_interval_level=confidence_level,
            estimate_val=adjusted_est,
            lower_bound=adjusted_lower,
            upper_bound=adjusted_upper,
            standard_error=adjusted_se,
            sample_size=base_estimate.sample_size,
            # Adjustment parameters
            gamma_squared=gamma_squared,
            rho=rho,
        )

    def compute_adjusted_proxy_estimate(
        self,
        confidence_level: float = 0.95,
        method: str = "MoM",
    ) -> AdjustedEstimate:
        """
        Compute proxy estimate with adjusted confidence intervals.

        The adjustment adds gamma^2 from cross-domain rectifier heterogeneity
        plus the rectifier variance for the target domain.

        Args:
            confidence_level: Confidence level for intervals.
            method: Latent model estimation method ("MoM" or "MLE").

        Returns:
            AdjustedEstimate with original and adjusted bounds.
        """
        base_estimate = self.compute_proxy_mean_estimate(
            confidence_level=confidence_level,
        )

        fit_result = self.estimate_latent_parameters(
            use_ppi_errors=False,
            weighted=False,
            method=method,
        )
        gamma_squared = fit_result.gamma_squared
        rho = fit_result.rho

        total_variance = base_estimate.standard_error**2 + gamma_squared
        adjusted_est = base_estimate.estimate_val + fit_result.rho
        adjusted_se = float(np.sqrt(total_variance))

        z = norm.ppf(1 - (1 - confidence_level) / 2)
        adjusted_lower = adjusted_est - z * adjusted_se
        adjusted_upper = adjusted_est + z * adjusted_se

        return AdjustedEstimate(
            confidence_interval_level=confidence_level,
            estimate_val=adjusted_est,
            lower_bound=adjusted_lower,
            upper_bound=adjusted_upper,
            standard_error=adjusted_se,
            sample_size=base_estimate.sample_size,
            # Adjustment parameters
            gamma_squared=gamma_squared,
            rho=rho,
        )

    def _bootstrap_resample_df(
        self,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """
        Create a bootstrap resample of the DataFrame by domain.

        Resamples rows within each domain with replacement.

        Args:
            rng: NumPy random generator for reproducibility.

        Returns:
            Bootstrap resampled DataFrame.
        """
        resampled_dfs = []
        for domain in self._get_sorted_domains():
            domain_df = self._df[self._df[self._domain_column] == domain]
            n = len(domain_df)
            if n > 0:
                indices = rng.choice(n, size=n, replace=True)
                resampled_dfs.append(domain_df.iloc[indices])
        return pd.concat(resampled_dfs, ignore_index=True)

    def compute_bootstrap_proxy_estimate(
        self,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
        seed: int | None = None,
    ) -> BootstrapEstimate:
        """
        Compute proxy estimate with bootstrap confidence intervals.

        Uses percentile bootstrap by resampling within domains and
        recomputing the rectified proxy estimate.

        Args:
            confidence_level: Confidence level for intervals.
            n_bootstrap: Number of bootstrap replications.
            seed: Random seed for reproducibility.

        Returns:
            BootstrapEstimate with bootstrap CI bounds.
        """
        rng = np.random.default_rng(seed)

        # Get original estimate
        base_estimate = self.compute_proxy_mean_estimate(
            confidence_level=confidence_level,
        )

        # Compute bootstrap distribution of estimates
        bootstrap_estimates = []
        for _ in range(n_bootstrap):
            resampled_df = self._bootstrap_resample_df(rng)
            bootstrap_estimator = CovShiftPPIEstimator(
                df=resampled_df,
                primary_outcome_column=self._primary_outcome_column,
                proxy_outcome_column=self._proxy_outcome_column,
                importance_weight_column=self._importance_weight_column,
                domain_column=self._domain_column,
                target_domain_value=self._target_domain_value,
            )
            try:
                boot_est = bootstrap_estimator.compute_proxy_mean_estimate(
                    weighted=False,
                    confidence_level=confidence_level,
                )
                bootstrap_estimates.append(boot_est.estimate_val)
            except (ValueError, ZeroDivisionError):
                # Skip failed bootstrap samples
                continue

        if len(bootstrap_estimates) < 10:
            # Not enough valid bootstrap samples, fall back to normal CI
            return BootstrapEstimate(
                estimate_val=base_estimate.estimate_val,
                lower_bound=base_estimate.lower_bound,
                upper_bound=base_estimate.upper_bound,
                standard_error=base_estimate.standard_error,
                sample_size=base_estimate.sample_size,
                confidence_interval_level=confidence_level,
                bootstrap_lower_bound=base_estimate.lower_bound,
                bootstrap_upper_bound=base_estimate.upper_bound,
                n_bootstrap=len(bootstrap_estimates),
            )

        # Compute percentile CI
        alpha = 1 - confidence_level
        bootstrap_arr = np.array(bootstrap_estimates)
        lower_pct = np.percentile(bootstrap_arr, 100 * alpha / 2)
        upper_pct = np.percentile(bootstrap_arr, 100 * (1 - alpha / 2))

        return BootstrapEstimate(
            estimate_val=base_estimate.estimate_val,
            lower_bound=base_estimate.lower_bound,
            upper_bound=base_estimate.upper_bound,
            standard_error=base_estimate.standard_error,
            sample_size=base_estimate.sample_size,
            confidence_interval_level=confidence_level,
            bootstrap_lower_bound=float(lower_pct),
            bootstrap_upper_bound=float(upper_pct),
            n_bootstrap=len(bootstrap_estimates),
        )

    def compute_bootstrap_ppi_estimate(
        self,
        weighted: bool = False,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
        seed: int | None = None,
    ) -> BootstrapEstimate:
        """
        Compute PPI estimate with bootstrap confidence intervals.

        Uses percentile bootstrap by resampling within domains and
        recomputing the PPI estimate.

        Args:
            weighted: If True, use importance weights.
            confidence_level: Confidence level for intervals.
            n_bootstrap: Number of bootstrap replications.
            seed: Random seed for reproducibility.

        Returns:
            BootstrapEstimate with bootstrap CI bounds.
        """
        rng = np.random.default_rng(seed)

        # Get original estimate
        base_estimate = self.compute_ppi_mean_estimate(
            weighted=weighted,
            confidence_level=confidence_level,
        )

        # Compute bootstrap distribution of estimates
        bootstrap_estimates = []
        for _ in range(n_bootstrap):
            resampled_df = self._bootstrap_resample_df(rng)
            bootstrap_estimator = CovShiftPPIEstimator(
                df=resampled_df,
                primary_outcome_column=self._primary_outcome_column,
                proxy_outcome_column=self._proxy_outcome_column,
                importance_weight_column=self._importance_weight_column,
                domain_column=self._domain_column,
                target_domain_value=self._target_domain_value,
            )
            try:
                boot_est = bootstrap_estimator.compute_ppi_mean_estimate(
                    weighted=weighted,
                    confidence_level=confidence_level,
                )
                bootstrap_estimates.append(boot_est.estimate_val)
            except (ValueError, ZeroDivisionError):
                # Skip failed bootstrap samples
                continue

        if len(bootstrap_estimates) < 10:
            # Not enough valid bootstrap samples, fall back to normal CI
            return BootstrapEstimate(
                estimate_val=base_estimate.estimate_val,
                lower_bound=base_estimate.lower_bound,
                upper_bound=base_estimate.upper_bound,
                standard_error=base_estimate.standard_error,
                sample_size=base_estimate.sample_size,
                confidence_interval_level=confidence_level,
                bootstrap_lower_bound=base_estimate.lower_bound,
                bootstrap_upper_bound=base_estimate.upper_bound,
                n_bootstrap=len(bootstrap_estimates),
            )

        # Compute percentile CI
        alpha = 1 - confidence_level
        bootstrap_arr = np.array(bootstrap_estimates)
        lower_pct = np.percentile(bootstrap_arr, 100 * alpha / 2)
        upper_pct = np.percentile(bootstrap_arr, 100 * (1 - alpha / 2))

        return BootstrapEstimate(
            estimate_val=base_estimate.estimate_val,
            lower_bound=base_estimate.lower_bound,
            upper_bound=base_estimate.upper_bound,
            standard_error=base_estimate.standard_error,
            sample_size=base_estimate.sample_size,
            confidence_interval_level=confidence_level,
            bootstrap_lower_bound=float(lower_pct),
            bootstrap_upper_bound=float(upper_pct),
            n_bootstrap=len(bootstrap_estimates),
        )

    def compute_domain_bootstrap_ppi_estimate(
        self,
        weighted: bool = True,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
        seed: int | None = None,
    ) -> DomainBootstrapEstimate:
        """
        Compute PPI estimate with domain-bootstrap confidence intervals.

        This implements the domain-bootstrap algorithm that resamples at the
        ESTIMATE LEVEL (domain-level d_k values) and fits the latent model
        for each bootstrap sample.

        Algorithm:
            1. Compute d_k and σ_k for each source domain
            2. For each bootstrap iteration b:
               a. Resample K-1 indices with replacement from source domains
               b. Fit latent model to get ρ̂^(b) and γ̂²^(b)
               c. Draw θ_K^(b) ~ N(ppi_K + ρ̂^(b), σ²_ppi,K + γ̂²^(b))
            3. Return quantiles of bootstrap samples as confidence interval

        Args:
            weighted: If True, use importance weights for rectifier computation.
            use_ppi_errors: If True, use PPI errors (primary - rectified_proxy)
                which account for the rectifier from other domains. If False,
                use simple proxy-primary differences.
            confidence_level: Confidence level for intervals.
            n_bootstrap: Number of bootstrap draws B.
            seed: Random seed for reproducibility.

        Returns:
            DomainBootstrapEstimate with domain-bootstrap CI bounds.
        """
        rng = np.random.default_rng(seed)

        # Get base PPI estimate for the target domain
        base_estimate = self.compute_ppi_mean_estimate(
            weighted=weighted,
            confidence_level=confidence_level,
        )

        # Compute domain-level statistics for source domains
        domain_stats = self.compute_hold_one_domain_out_estimate_stats(
            weighted=weighted
        )

        if len(domain_stats) < 2:
            raise ValueError("Need at least 2 source domains for domain-bootstrap")

        valid_stats = [
            s
            for s in domain_stats
            if s.ppi_error is not None and s.ppi_error_var is not None
        ]
        if len(valid_stats) < 2:
            raise ValueError(
                "Need at least 2 domains with valid PPI errors for "
                "domain-bootstrap with use_ppi_errors=True"
            )
        d_k = np.array([s.ppi_error for s in valid_stats])
        sigma_k = np.sqrt(np.array([s.ppi_error_var for s in valid_stats]))

        # Target domain PPI estimate and variance
        target_ppi_mean = base_estimate.estimate_val
        target_ppi_var = base_estimate.standard_error**2

        K_minus_1 = len(d_k)

        # Bootstrap: resample domains and fit latent model each iteration
        rho_estimates = np.zeros(n_bootstrap)
        gamma_sq_estimates = np.zeros(n_bootstrap)
        theta_K_samples = np.zeros(n_bootstrap)

        for b in range(n_bootstrap):
            # Resample domain indices with replacement
            boot_indices = rng.choice(K_minus_1, size=K_minus_1, replace=True)
            d_k_boot = d_k[boot_indices]
            sigma_k_boot = sigma_k[boot_indices]

            # Fit latent model to resampled data
            fit_result = fit_latent_normal(
                Y=d_k_boot,
                sigma=sigma_k_boot,
                constrain_mean_0=False,
                method="MoM",
            )
            rho_estimates[b] = fit_result.rho
            gamma_sq_estimates[b] = fit_result.gamma_squared

            # Draw θ_K^(b) ~ N(ppi_K + ρ̂^(b), σ²_ppi,K + γ̂²^(b))
            mean_b = target_ppi_mean + fit_result.rho
            var_b = target_ppi_var + fit_result.gamma_squared
            theta_K_samples[b] = rng.normal(mean_b, np.sqrt(var_b))

        # Compute empirical quantiles
        alpha = 1 - confidence_level
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2

        domain_bootstrap_lower = float(np.quantile(theta_K_samples, lower_quantile))
        domain_bootstrap_upper = float(np.quantile(theta_K_samples, upper_quantile))
        domain_bootstrap_point = float(np.median(theta_K_samples))

        return DomainBootstrapEstimate(
            estimate_val=base_estimate.estimate_val,
            lower_bound=base_estimate.lower_bound,
            upper_bound=base_estimate.upper_bound,
            standard_error=base_estimate.standard_error,
            sample_size=base_estimate.sample_size,
            confidence_interval_level=confidence_level,
            domain_bootstrap_lower_bound=domain_bootstrap_lower,
            domain_bootstrap_upper_bound=domain_bootstrap_upper,
            domain_bootstrap_point_estimate=domain_bootstrap_point,
            n_bootstrap=n_bootstrap,
            rho_mean=float(np.mean(rho_estimates)),
            gamma_squared_mean=float(np.mean(gamma_sq_estimates)),
        )

    def compute_domain_bootstrap_proxy_estimate(
        self,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
        seed: int | None = None,
    ) -> DomainBootstrapEstimate:
        """
        Compute proxy estimate with domain-bootstrap confidence intervals.

        This implements the domain-bootstrap algorithm for the proxy-only case,
        resampling at the ESTIMATE LEVEL (domain-level d_k values) and fitting
        the latent model for each bootstrap sample.

        Algorithm:
            1. Compute d_k and σ_k for each source domain
            2. For each bootstrap iteration b:
               a. Resample K-1 indices with replacement from source domains
               b. Fit latent model to get ρ̂^(b) and γ̂²^(b)
               c. Draw θ_K^(b) ~ N(proxy_K + ρ̂^(b), σ²_proxy,K + γ̂²^(b))
            3. Return quantiles of bootstrap samples as confidence interval

        Args:
            confidence_level: Confidence level for intervals.
            n_bootstrap: Number of bootstrap draws B.
            seed: Random seed for reproducibility.
            use_ppi_errors: If True, use PPI errors; otherwise use simple differences.
            weighted: If True, use importance weights for rectifier computation.

        Returns:
            DomainBootstrapEstimate with domain-bootstrap CI bounds.
        """
        rng = np.random.default_rng(seed)

        # Get base proxy estimate for the target domain
        base_estimate = self.compute_proxy_mean_estimate(
            confidence_level=confidence_level,
        )

        # Compute domain-level statistics for source domains
        domain_stats = self.compute_hold_one_domain_out_estimate_stats(weighted=False)

        if len(domain_stats) < 2:
            raise ValueError("Need at least 2 source domains for domain-bootstrap")

        # Use simple differences (primary - proxy)
        d_k = np.array([s.est_diff for s in domain_stats])
        sigma_k = np.sqrt(np.array([s.est_diff_var for s in domain_stats]))

        # Target domain proxy estimate and variance
        target_df = self._df[self._df[self._domain_column] == self._target_domain_value]
        target_proxy = target_df[self._proxy_outcome_column].values.astype(np.float64)
        target_proxy = target_proxy[~np.isnan(target_proxy)]
        n_target = len(target_proxy)

        if n_target < 2:
            raise ValueError("Target domain needs at least 2 valid samples")

        target_proxy_mean = float(np.mean(target_proxy))
        target_proxy_var = float(np.var(target_proxy, ddof=1) / n_target)

        K_minus_1 = len(d_k)

        # Bootstrap: resample domains and fit latent model each iteration
        rho_estimates = np.zeros(n_bootstrap)
        gamma_sq_estimates = np.zeros(n_bootstrap)
        theta_K_samples = np.zeros(n_bootstrap)

        for b in range(n_bootstrap):
            # Resample domain indices with replacement
            boot_indices = rng.choice(K_minus_1, size=K_minus_1, replace=True)
            d_k_boot = d_k[boot_indices]
            sigma_k_boot = sigma_k[boot_indices]

            # Fit latent model to resampled data
            fit_result = fit_latent_normal(
                Y=d_k_boot,
                sigma=sigma_k_boot,
                constrain_mean_0=False,
                method="MoM",
            )
            rho_estimates[b] = fit_result.rho
            gamma_sq_estimates[b] = fit_result.gamma_squared

            # Draw θ_K^(b) ~ N(proxy_K + ρ̂^(b), σ²_proxy,K + γ̂²^(b))
            mean_b = target_proxy_mean + fit_result.rho
            var_b = target_proxy_var + fit_result.gamma_squared
            theta_K_samples[b] = rng.normal(mean_b, np.sqrt(var_b))

        # Compute empirical quantiles
        alpha = 1 - confidence_level
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2

        domain_bootstrap_lower = float(np.quantile(theta_K_samples, lower_quantile))
        domain_bootstrap_upper = float(np.quantile(theta_K_samples, upper_quantile))
        domain_bootstrap_point = float(np.median(theta_K_samples))

        return DomainBootstrapEstimate(
            estimate_val=base_estimate.estimate_val,
            lower_bound=base_estimate.lower_bound,
            upper_bound=base_estimate.upper_bound,
            standard_error=base_estimate.standard_error,
            sample_size=base_estimate.sample_size,
            confidence_interval_level=confidence_level,
            domain_bootstrap_lower_bound=domain_bootstrap_lower,
            domain_bootstrap_upper_bound=domain_bootstrap_upper,
            domain_bootstrap_point_estimate=domain_bootstrap_point,
            n_bootstrap=n_bootstrap,
            rho_mean=float(np.mean(rho_estimates)),
            gamma_squared_mean=float(np.mean(gamma_sq_estimates)),
        )
