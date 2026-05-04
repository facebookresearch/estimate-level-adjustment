# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from dataclasses import dataclass
from typing import Tuple

import scipy.stats as st


def alpha_to_z(alpha: float) -> float:
    z = st.norm.ppf(1 - (alpha * 0.5))
    return z


def conf_to_z(conf: int) -> float:
    alpha = (100 - conf) / 100
    return alpha_to_z(alpha)


def get_ci(delta: float, se: float, conf: int = 90) -> Tuple[float, float]:
    ci = (delta - conf_to_z(conf) * se, delta + conf_to_z(conf) * se)
    return ci


@dataclass
class Estimate:
    confidence_interval_level: float
    estimate_val: float
    lower_bound: float
    upper_bound: float
    sample_size: int
    standard_error: float

    @property
    def ci_width(self) -> float:
        """Width of the confidence interval."""
        return self.upper_bound - self.lower_bound

    @property
    def relative_error(self) -> float:
        """Relative standard error (coefficient of variation)."""
        if self.estimate_val == 0:
            return float("inf")
        return abs(self.standard_error / self.estimate_val)

    def __repr__(self) -> str:
        return (
            f"Estimate({self.estimate_val:.4f} "
            f"[{self.lower_bound:.4f}, {self.upper_bound:.4f}], "
            f"n={self.sample_size:,})"
        )

    def _repr_html_(self) -> str:
        ci_pct = self.confidence_interval_level * 100
        return f"""
        <div style="border:1px solid #6c757d; padding:10px; border-radius:5px; margin:5px 0;">
            <h4 style="margin:0 0 8px 0;">Estimate</h4>
            <table style="border-collapse:collapse;">
                <tr><td style="padding-right:10px;"><b>Value:</b></td><td>{self.estimate_val:.4f}</td></tr>
                <tr><td style="padding-right:10px;"><b>{ci_pct:.0f}% CI:</b></td><td>[{self.lower_bound:.4f}, {self.upper_bound:.4f}]</td></tr>
                <tr><td style="padding-right:10px;"><b>Std Error:</b></td><td>{self.standard_error:.4f}</td></tr>
                <tr><td style="padding-right:10px;"><b>Sample Size:</b></td><td>{self.sample_size:,}</td></tr>
            </table>
        </div>
        """
