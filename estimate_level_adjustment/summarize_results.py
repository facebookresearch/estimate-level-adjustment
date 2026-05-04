# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .utils import leave_one_out_validation


# Define a function to compute the correlation coefficient
def compute_coefficient_mean(
    df: pd.DataFrame, domains: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Compute the necessary statistics between primary and proxy metrics.

    Parameters:
    - df: DataFrame containing primary_metric and proxy_metric columns
    - domains (optional): List of domain columns to group by

    Returns:
    - DataFrame with domain, primary_mean, proxy_mean, primary_var, proxy_var, primary_proxy_covar
    if domains is None, returns a DataFrame with the overall correlation coefficient
    """
    if domains is None:
        domains = ["domain"]
        df["domain"] = "overall"
    p = len(domains)
    stats = (
        df.groupby(domains, as_index=False)
        .agg(
            {
                "primary_metric": ["count", "mean", "var"],
                "proxy_metric": ["mean", "var"],
            }
        )
        .droplevel(1, axis=1)
    )
    stats.columns = domains + [
        "count",
        "primary_mean",
        "primary_var",
        "proxy_mean",
        "proxy_var",
    ]

    stats_cov = (
        df.groupby(domains)[["primary_metric", "proxy_metric"]].cov().reset_index()
    )
    stats_cov = stats_cov[stats_cov[f"level_{p}"] == "primary_metric"]
    del stats_cov[f"level_{p}"]
    del stats_cov["primary_metric"]
    stats_cov.columns = domains + ["primary_proxy_covar"]
    stats = pd.merge(stats, stats_cov, on=domains)
    stats["primary_var"] = stats["primary_var"] / stats["count"]
    stats["proxy_var"] = stats["proxy_var"] / stats["count"]
    stats["primary_proxy_covar"] = stats["primary_proxy_covar"] / stats["count"]
    return stats


# Define a function to compute the average treatment effect
def compute_average_treatment_effect(
    df: pd.DataFrame,
    condition_column: str = "condition",
    domains: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute the average treatment effect (ATE) for primary and proxy outcomes,
    along with their covariance.

    Parameters:
    - df: DataFrame containing primary_metric, proxy_metric, and condition columns
    - condition_column: The column indicating treatment/control condition (default: "condition")
    - domains (optional): The list of domain column to group by for heterogeneous treatment effects

    Returns:
    - DataFrame with ATE statistics including:
      - primary_ate, proxy_ate: ATEs for each outcome
      - primary_ate_var, proxy_ate_var: Variances of the ATEs
      - ate_covar: Covariance between primary and proxy ATEs
    """
    if domains is None:
        domains = ["domain"]
        df["domain"] = "overall"
    p = len(domains)
    # Separate treatment and control groups
    treatment_df = df[df[condition_column] != "control"]
    control_df = df[df[condition_column] == "control"]

    # Compute means for treatment group
    treatment_stats = (
        treatment_df.groupby(domains, as_index=False)
        .agg(
            {
                "primary_metric": ["count", "mean", "var"],
                "proxy_metric": ["mean", "var"],
            }
        )
        .droplevel(1, axis=1)
    )
    treatment_stats.columns = domains + [
        "n_treatment",
        "primary_mean_t",
        "primary_var_t",
        "proxy_mean_t",
        "proxy_var_t",
    ]

    # Compute covariance for treatment group
    treatment_cov = (
        treatment_df.groupby(domains)[["primary_metric", "proxy_metric"]]
        .cov()
        .reset_index()
    )
    treatment_cov = treatment_cov[treatment_cov[f"level_{p}"] == "primary_metric"]
    del treatment_cov[f"level_{p}"]
    del treatment_cov["primary_metric"]
    treatment_cov.columns = domains + ["primary_proxy_cov_t"]
    treatment_stats = pd.merge(treatment_stats, treatment_cov, on=domains)

    # Compute means for control group
    control_stats = (
        control_df.groupby(domains, as_index=False)
        .agg(
            {
                "primary_metric": ["count", "mean", "var"],
                "proxy_metric": ["mean", "var"],
            }
        )
        .droplevel(1, axis=1)
    )
    control_stats.columns = domains + [
        "n_control",
        "primary_mean_c",
        "primary_var_c",
        "proxy_mean_c",
        "proxy_var_c",
    ]

    # Compute covariance for control group
    control_cov = (
        control_df.groupby(domains)[["primary_metric", "proxy_metric"]]
        .cov()
        .reset_index()
    )
    control_cov = control_cov[control_cov[f"level_{p}"] == "primary_metric"]
    del control_cov[f"level_{p}"]
    del control_cov["primary_metric"]
    control_cov.columns = domains + ["primary_proxy_cov_c"]
    control_stats = pd.merge(control_stats, control_cov, on=domains)

    # Merge treatment and control statistics
    stats = pd.merge(treatment_stats, control_stats, on=domains, how="outer")

    # Compute ATEs: E[Y|T=1] - E[Y|T=0]
    stats["primary_mean"] = stats["primary_mean_t"] - stats["primary_mean_c"]
    stats["proxy_mean"] = stats["proxy_mean_t"] - stats["proxy_mean_c"]

    # Compute variances of ATEs: Var(Y_t)/n_t + Var(Y_c)/n_c
    stats["primary_var"] = (
        stats["primary_var_t"] / stats["n_treatment"]
        + stats["primary_var_c"] / stats["n_control"]
    )
    stats["proxy_var"] = (
        stats["proxy_var_t"] / stats["n_treatment"]
        + stats["proxy_var_c"] / stats["n_control"]
    )

    # Compute covariance between primary and proxy ATEs
    # Cov(ATE_primary, ATE_proxy) = Cov(Y1_t, Y2_t)/n_t + Cov(Y1_c, Y2_c)/n_c
    stats["primary_proxy_covar"] = (
        stats["primary_proxy_cov_t"] / stats["n_treatment"]
        + stats["primary_proxy_cov_c"] / stats["n_control"]
    )

    # Select final columns
    result = stats[
        domains
        + [
            "n_treatment",
            "n_control",
            "primary_mean",
            "proxy_mean",
            "primary_var",
            "proxy_var",
            "primary_proxy_covar",
        ]
    ]

    return result


def clean_domain_stats(stats: pd.DataFrame, minimum_count: int = 10) -> pd.DataFrame:
    """
    Clean the domain statistics DataFrame.

    Parameters:
    - stats: DataFrame containing domain statistics

    Returns:
    - Cleaned DataFrame with domain, primary_mean, proxy_mean, primary_var, proxy_var, primary_proxy_covar
    """
    stats = stats.dropna()  # Drop rows with NaN values
    stats = stats[
        stats["count"] >= minimum_count
    ]  # Filter rows with count less than minimum_count

    return stats


def adapt_stats_to_coverage(
    stats: pd.DataFrame,
) -> Tuple[
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    Adapt domain statistics for coverage analysis by computing corrected proxy
    estimates using leave-one-out validation.

    Parameters:
    - stats: DataFrame with columns: primary_mean, primary_var, proxy_mean, proxy_var, primary_proxy_covar

    Returns:
    - Tuple of (mean_primary, sigma_primary, mean_proxy, sigma_proxy, mean_proxy_corr, sigma_proxy_corr)
    """
    stats["primary_sd"] = np.sqrt(stats["primary_var"])
    stats["proxy_sd"] = np.sqrt(stats["proxy_var"])
    stats["diff_est"] = stats["primary_mean"] - stats["proxy_mean"]
    stats["diff_sd"] = np.sqrt(
        stats["primary_var"] + stats["proxy_var"] - 2 * stats["primary_proxy_covar"]
    )
    stats = leave_one_out_validation(
        stats, "diff_est", "diff_sd", "proxy_mean", "proxy_sd"
    )
    sigma_primary = stats["primary_sd"]
    mean_primary = stats["primary_mean"]

    sigma_proxy = stats["proxy_sd"]
    mean_proxy = stats["proxy_mean"]

    sigma_proxy_corr = stats["proxy_sd_corrected"]
    mean_proxy_corr = stats["proxy_mean_corrected"]
    return (
        mean_primary,
        sigma_primary,
        mean_proxy,
        sigma_proxy,
        mean_proxy_corr,
        sigma_proxy_corr,
    )


def plot_coverage_curve(
    base_coverage: NDArray[np.floating],
    corr_coverage: NDArray[np.floating],
    alpha_seq: NDArray[np.floating],
) -> None:
    """Plot coverage curve comparing base and corrected proxy methods.

    Parameters:
    - base_coverage: Array of base coverage values
    - corr_coverage: Array of corrected coverage values
    - alpha_seq: Array of alpha (error rate) values
    """
    plt.style.use("default")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(alpha_seq, base_coverage, color="red", label="Proxy", linewidth=2)
    ax.plot(
        alpha_seq,
        corr_coverage,
        color="red",
        linestyle="--",
        label="Proxy with Adjustments",
        linewidth=2,
    )
    ax.plot(
        alpha_seq,
        1.0 - alpha_seq,
        color="black",
        linestyle="--",
        label="Sufficient overlap rate",
        linewidth=1.5,
        alpha=0.7,
    )

    ax.set_xlabel(
        "Alpha (Error Rate)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel(
        "Empirical Overlap Rate",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_title(
        "Empirical Overlap Rate vs Alpha",
        fontsize=16,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.tick_params(axis="both", labelsize=10)

    fig.tight_layout()
    plt.show()


def plot_coverage_curves_side_by_side(
    base_coverage_1: NDArray[np.floating],
    corr_coverage_1: NDArray[np.floating],
    alpha_seq_1: NDArray[np.floating],
    base_coverage_2: NDArray[np.floating],
    corr_coverage_2: NDArray[np.floating],
    alpha_seq_2: NDArray[np.floating],
    title_1: str = "Coverage Curve 1",
    title_2: str = "Coverage Curve 2",
) -> None:
    """Display two coverage curves side by side for comparison.

    Parameters:
    - base_coverage_1: Array of base coverage values for left plot
    - corr_coverage_1: Array of corrected coverage values for left plot
    - alpha_seq_1: Array of alpha (error rate) values for left plot
    - base_coverage_2: Array of base coverage values for right plot
    - corr_coverage_2: Array of corrected coverage values for right plot
    - alpha_seq_2: Array of alpha (error rate) values for right plot
    - title_1: Title for the left plot (default: "Coverage Curve 1")
    - title_2: Title for the right plot (default: "Coverage Curve 2")
    """
    plt.style.use("default")

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Left plot
    ax1 = axes[0]
    ax1.plot(alpha_seq_1, base_coverage_1, color="red", label="Proxy", linewidth=2)
    ax1.plot(
        alpha_seq_1,
        corr_coverage_1,
        color="red",
        linestyle="--",
        label="Proxy with Adjustments",
        linewidth=2,
    )
    ax1.plot(
        alpha_seq_1,
        1.0 - alpha_seq_1,
        color="black",
        linestyle="--",
        label="Sufficient overlap rate",
        linewidth=1.5,
        alpha=0.7,
    )
    ax1.set_xlabel("Alpha (Error Rate)", fontsize=12)
    ax1.set_ylabel("Empirical Overlap Rate", fontsize=12)
    ax1.set_title(title_1, fontsize=14, fontweight="bold")
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)
    ax1.tick_params(axis="both", labelsize=10)

    # Right plot
    ax2 = axes[1]
    ax2.plot(alpha_seq_2, base_coverage_2, color="red", label="Proxy", linewidth=2)
    ax2.plot(
        alpha_seq_2,
        corr_coverage_2,
        color="red",
        linestyle="--",
        label="Proxy with Adjustments",
        linewidth=2,
    )
    ax2.plot(
        alpha_seq_2,
        1.0 - alpha_seq_2,
        color="black",
        linestyle="--",
        label="Sufficient overlap rate",
        linewidth=1.5,
        alpha=0.7,
    )
    ax2.set_xlabel("Alpha (Error Rate)", fontsize=12)
    ax2.set_ylabel("Empirical Overlap Rate", fontsize=12)
    ax2.set_title(title_2, fontsize=14, fontweight="bold")
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)
    ax2.tick_params(axis="both", labelsize=10)

    fig.tight_layout()
    plt.show()


def plot_domain_level_metrics(stats: pd.DataFrame) -> None:
    """Plot domain level metrics with confidence intervals.

    Parameters:
    - stats: DataFrame containing domain statistics with columns:
        primary_mean, primary_sd, proxy_mean, proxy_sd, proxy_sd_corrected
    """
    plt.style.use("default")

    num_stddev = 1.96
    offset = 0.1

    n_experiments = len(stats)
    y_positions = np.arange(n_experiments)

    primary_means = stats["primary_mean"].values
    primary_errors = num_stddev * stats["primary_sd"].values

    proxy_means = stats["proxy_mean"].values
    proxy_errors = num_stddev * stats["proxy_sd"].values

    proxy_corr_means = stats["proxy_mean_corrected"].values
    proxy_corr_errors = num_stddev * stats["proxy_sd_corrected"].values

    fig, ax = plt.subplots(figsize=(10, 16))

    ax.errorbar(
        y=y_positions - offset,
        x=primary_means,
        xerr=primary_errors,
        fmt="o",
        color="blue",
        label="Primary",
    )
    ax.errorbar(
        y=y_positions + offset,
        x=proxy_means,
        xerr=proxy_errors,
        fmt="o",
        color="red",
        label="Proxy",
    )
    ax.errorbar(
        y=y_positions,
        x=proxy_corr_means,
        xerr=proxy_corr_errors,
        fmt="s",
        color="grey",
        label="Proxy (Corrected)",
    )

    ax.set_yticks(range(n_experiments))
    ax.set_ylabel(
        "Experiments",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel(
        "Mean Values",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_title(
        "Mean Values with Confidence Intervals", fontsize=16, fontweight="bold"
    )
    ax.legend(loc="best", fontsize=10)
    ax.tick_params(axis="both", labelsize=10)

    fig.tight_layout()
    plt.show()


def plot_coverage_and_domain_metrics_side_by_side(
    base_coverage: NDArray[np.floating],
    corr_coverage: NDArray[np.floating],
    alpha_seq: NDArray[np.floating],
    stats: pd.DataFrame,
) -> None:
    """
    Display coverage curve and domain level metrics plots side by side.

    Parameters:
    - base_coverage: Array of base coverage values
    - corr_coverage: Array of corrected coverage values
    - alpha_seq: Array of alpha (error rate) values
    - stats: DataFrame containing domain statistics with columns:
        primary_mean, primary_sd, proxy_mean, proxy_sd, proxy_sd_corrected
    """
    plt.style.use("default")

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Left plot: Coverage curve
    ax1 = axes[0]
    ax1.plot(alpha_seq, base_coverage, color="red", label="Proxy", linewidth=2)
    ax1.plot(
        alpha_seq,
        corr_coverage,
        color="red",
        linestyle="-.",
        label="Proxy with Adjustments",
        linewidth=2,
    )
    ax1.plot(
        alpha_seq,
        1.0 - alpha_seq,
        color="black",
        linestyle="--",
        label="Sufficient overlap rate",
        linewidth=1.5,
        alpha=0.7,
    )
    ax1.set_xlabel("Alpha (Error Rate)", fontsize=12)
    ax1.set_ylabel("Empirical Overlap Rate", fontsize=12)
    ax1.set_title(
        "Empirical Overlap Rate vs Alpha",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)
    ax1.tick_params(axis="both", labelsize=10)

    # Right plot: Domain level metrics
    ax2 = axes[1]
    num_stddev = 1.96
    offset = 0.1

    n_experiments = len(stats)
    y_positions = np.arange(n_experiments)

    primary_means = stats["primary_mean"].values
    primary_errors = num_stddev * stats["primary_sd"].values

    proxy_means = stats["proxy_mean"].values
    proxy_errors = num_stddev * stats["proxy_sd"].values

    proxy_corr_means = stats["proxy_mean_corrected"].values
    proxy_corr_errors = num_stddev * stats["proxy_sd_corrected"].values

    ax2.errorbar(
        y=y_positions - offset,
        x=primary_means,
        xerr=primary_errors,
        fmt="o",
        color="blue",
        label="Primary",
    )
    ax2.errorbar(
        y=y_positions + offset,
        x=proxy_means,
        xerr=proxy_errors,
        fmt="o",
        color="red",
        label="Proxy",
    )
    ax2.errorbar(
        y=y_positions,
        x=proxy_corr_means,
        xerr=proxy_corr_errors,
        fmt="s",
        color="grey",
        label="Proxy (Corrected)",
    )

    ax2.set_yticks(range(n_experiments))
    ax2.set_ylabel("Experiments", fontsize=12)
    ax2.set_xlabel("Mean Values", fontsize=12)
    ax2.set_title(
        "Mean Values with Confidence Intervals", fontsize=14, fontweight="bold"
    )
    ax2.legend(loc="best", fontsize=10)
    ax2.tick_params(axis="both", labelsize=10)

    fig.tight_layout()
    plt.show()
