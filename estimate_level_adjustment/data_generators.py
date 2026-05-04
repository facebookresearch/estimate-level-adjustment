# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class CovShiftDataGenerator:
    """
    Data generator for covariate shift and concept drift simulations.

    Generates synthetic data with controllable covariate shift across domains
    and optional concept drift, following the simulation setup for evaluating
    confidence interval methods under distribution shift.
    """

    _n_samples_per_domain: int
    _n_domains: int
    _concept_drift_degree: float
    _primary_lambda: float
    _primary_phi: float
    _proxy_lambda: float
    _proxy_phi: float
    _n_covariates: int
    _target_domain_means: NDArray[np.float64]
    _domain_means: NDArray[np.float64]
    _domain_deltas: NDArray[np.float64]

    def __init__(
        self,
        n_samples_per_domain: int,
        n_domains: int,
        concept_drift_degree: float,
        primary_lambda: float = 0.5,
        primary_phi: float = 0.0,
        proxy_lambda: float = 0.5,
        proxy_phi: float = 0.0,
        n_covariates: int = 4,
        target_domain_means: NDArray[np.float64] | None = None,
        random_seed: int | None = None,
    ) -> None:
        """
        Initialize the covariate shift data generator.

        Args:
            n_samples_per_domain: Sample size per domain
            n_domains: Number of domains (target domain is n_domains)
            concept_drift_degree: Degree of concept drift (0 = no drift, >0 = increasing drift)
            primary_lambda: Primary outcome model parameter (default: 0.5)
            primary_phi: Primary outcome model intercept parameter (default: 0.0)
            proxy_lambda: Proxy model parameter (default: 0.5)
            proxy_phi: Proxy model intercept parameter (default: 0.0)
            n_covariates: Number of covariates (default: 4)
            target_domain_means: Target domain mean vector. If None, uses alternating pattern
                  [1, -1, 1, -1, ...] (length n_covariates) normalized to unit vector.
                  If provided, length must equal n_covariates (warning raised if not).
            random_seed: Random seed for reproducibility
        """
        self._n_samples_per_domain = n_samples_per_domain
        self._n_domains = n_domains
        self._concept_drift_degree = concept_drift_degree
        self._primary_lambda = primary_lambda
        self._primary_phi = primary_phi
        self._proxy_lambda = proxy_lambda
        self._proxy_phi = proxy_phi
        self._n_covariates = n_covariates

        if random_seed is not None:
            np.random.seed(random_seed)

        # Set target domain mean with validation
        if target_domain_means is None:
            # Use alternating pattern: [1, -1, 1, -1, ...] normalized
            self._target_domain_means = np.array(
                [1.0 if i % 2 == 0 else -1.0 for i in range(self._n_covariates)],
                dtype=np.float64,
            )
            self._target_domain_means = self._target_domain_means / np.linalg.norm(
                self._target_domain_means
            )
        else:
            # Validate length matches n_covariates
            if len(target_domain_means) != self._n_covariates:
                import warnings

                warnings.warn(
                    f"Length of target_domain_mean ({len(target_domain_means)}) does not match "
                    f"n_covariates ({self._n_covariates}). This may cause unexpected behavior.",
                    UserWarning,
                    stacklevel=2,
                )
            self._target_domain_means = target_domain_means

        # Generate domain-specific parameters
        self._domain_means = self._generate_domain_means()
        self._domain_deltas = self._generate_deltas()

    def _generate_domain_means(self) -> NDArray[np.float64]:
        """
        Generate domain-specific covariate means.

        For domains 1 to n_domains-1: Sample from uniform ball with ||mu_s||_2 <= 1
        For target domain n_domains: target_domain_mean we specify this mean exactly.

        Returns:
            Array of shape (n_domains, n_covariates) with domain means
        """
        domain_means = np.zeros((self._n_domains, self._n_covariates))

        # Generate means for domains 1 to n_domains-1 from uniform ball
        for s in range(self._n_domains - 1):
            # Sample from unit ball using rejection sampling
            while True:
                candidate = np.random.uniform(-1, 1, size=self._n_covariates)
                if np.linalg.norm(candidate) <= 1:
                    domain_means[s] = candidate
                    break
        domain_means[self._n_domains - 1] = self._target_domain_means

        return domain_means

    def _generate_deltas(self) -> NDArray[np.float64]:
        """
        Generate domain-specific threshold shifts for concept drift.

        When concept_drift_degree = 0: delta_s = 0 (no concept drift, pure covariate shift)
        When concept_drift_degree > 0: delta_s ~ N(0, concept_drift_degree/2) (concept drift)

        Returns:
            Array of shape (n_domains,) with delta values
        """
        if self._concept_drift_degree == 0:
            return np.zeros(self._n_domains)
        return np.random.normal(
            0, np.sqrt(self._concept_drift_degree / 2), size=self._n_domains
        )

    def _compute_primary_probability(
        self, covariates: NDArray[np.float64], domain: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute P(Y=1|X,S) using the logistic response model.
        Y is the primary outcome, X are the covariates, and S is the domain.
        P(Y=1|X,S=s) = (1 + exp(-primary_lambda * (sum(I(X_p >= delta_s)) - n_covariates/2) - primary_phi))^-1

        Args:
            covariates: Covariate matrix of shape (n, n_covariates)
            domain: Domain index (0 to n_domains-1)

        Returns:
            Tuple of (probabilities, sum_indicators), each of shape (n,)
        """
        delta: float = float(self._domain_deltas[domain])
        indicators: NDArray[np.float64] = (covariates >= delta).astype(np.float64)
        sum_indicators: NDArray[np.float64] = np.sum(indicators, axis=1)  # type: ignore[assignment]

        logit: NDArray[np.float64] = np.asarray(
            self._primary_lambda * (sum_indicators - self._n_covariates / 2)
            - self._primary_phi
        )
        prob: NDArray[np.float64] = np.asarray(1 / (1 + np.exp(-logit)))

        return prob, sum_indicators

    def _compute_proxy_outcome(
        self, covariates: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute proxy outcome proxy_outcome = f(X).

        f(x) = (1/pi) * arctan(proxy_lambda * (sum(I(x_p >= 0)) - n_covariates/2) - proxy_phi) + 1/2

        Args:
            covariates: Covariate matrix of shape (n_s, n_covariates)

        Returns:
            Tuple of (proxy_outcomes, sum_indicators), each of shape (n,)
        """
        indicators: NDArray[np.float64] = (covariates >= 0).astype(np.float64)
        sum_indicators: NDArray[np.float64] = np.sum(indicators, axis=1)  # type: ignore[assignment]

        proxy_outcome: NDArray[np.float64] = np.asarray(
            (1 / np.pi)
            * np.arctan(
                self._proxy_lambda * (sum_indicators - self._n_covariates / 2)
                - self._proxy_phi
            )
            + 0.5
        )

        return proxy_outcome, sum_indicators

    def _compute_importance_weights(
        self, covariates: NDArray[np.float64], domain: int
    ) -> NDArray[np.float64]:
        """
        Compute importance weights w_s(x) for reweighting to target domain.

        - un-normalized
        w_s(x) = p_target(x) / p_s(x) = exp(0.5|x - mu_s|^2 - 0.5|x - mu_target|^2)

        Args:
            covariates: Covariate matrix of shape (n, n_covariates)
            domain: Source domain index (0 to n_domains-1)

        Returns:
            Array of importance weights of shape (n,)
        """
        domain_mean = self._domain_means[domain]
        target_mean = self._target_domain_means

        dist_squared_to_domain = np.sum((covariates - domain_mean) ** 2, axis=1)
        dist_squared_to_target = np.sum((covariates - target_mean) ** 2, axis=1)

        importance_weights = np.exp(
            0.5 * dist_squared_to_domain - 0.5 * dist_squared_to_target
        )

        return importance_weights

    def generate_data(self) -> pd.DataFrame:
        """
        Generate complete dataset for all domains.

        For each domain s:
        - Generate X_i | S_i = s ~ N(mu_s, I_n_covariates)
        - Generate primary_outcome_i ~ Bernoulli(P(primary_outcome=1|X_i, S_i=s))
        - Compute proxy_outcome = f(X_i)
        - Compute importance weights w_s(X_i)

        Returns:
            DataFrame with columns:
                - domain: Domain identifier (1 to n_domains)
                - covariate_1, ..., covariate_n_covariates: Covariate values
                - primary_outcome: Binary outcome
                - primary_outcome_probability: Probability of primary_outcome=1
                - primary_sum_indicator: Sum of indicators for primary_outcome (sum(I(x_p >= delta_s)))
                - proxy_outcome: Proxy outcome
                - proxy_sum_indicator: Sum of indicators for proxy_outcome (sum(I(x_p >= 0)))
                - delta: Domain-specific delta value
                - importance_weight: Importance weight for domain adaptation
        """
        data_list = []

        for s in range(self._n_domains):
            # Generate covariates from N(mu_s, I_n_covariates)
            covariates = np.random.multivariate_normal(
                mean=self._domain_means[s],
                cov=np.eye(self._n_covariates),
                size=self._n_samples_per_domain,
            )

            # Generate primary_outcome and get sum indicators
            probs, primary_sum_indicators = self._compute_primary_probability(
                covariates, s
            )
            primary_outcome = np.random.binomial(1, probs)

            # Compute proxy outcome and get proxy sum indicators
            proxy_outcome, proxy_sum_indicators = self._compute_proxy_outcome(
                covariates
            )

            # Compute importance weights
            importance_weights = self._compute_importance_weights(covariates, s)
            importance_weights = importance_weights / np.sum(
                importance_weights
            )  # normalize weights

            # Get delta value for this domain
            delta_value = self._domain_deltas[s]

            # Create domain data
            domain_data = {
                "domain": np.full(
                    self._n_samples_per_domain, s + 1
                ),  # 1-indexed domains
                **{
                    f"covariate_{p + 1}": covariates[:, p]
                    for p in range(self._n_covariates)
                },
                "primary_outcome": primary_outcome,
                "primary_outcome_probability": probs,
                "primary_sum_indicator": primary_sum_indicators,
                "proxy_outcome": proxy_outcome,
                "proxy_sum_indicator": proxy_sum_indicators,
                "delta": np.full(self._n_samples_per_domain, delta_value),
                "importance_weight": importance_weights,
            }

            data_list.append(pd.DataFrame(domain_data))

        return pd.concat(data_list, ignore_index=True)

    def calculate_target_population_prevalence(
        self, n_samples: int = 1_000_000
    ) -> float:
        """
        Approximate true target prevalence E[primary_outcome | S = n_domains] via Monte Carlo.

        Samples n_samples observations from the target domain and computes
        the empirical mean, which approximates the true population prevalence.

        Args:
            n_samples: Number of Monte Carlo samples (default: 1,000,000)

        Returns:
            Approximated population prevalence rate
        """
        # Generate covariates from target domain N(target_domain_mean, I_n_covariates)
        covariates = np.random.multivariate_normal(
            mean=self._target_domain_means,
            cov=np.eye(self._n_covariates),
            size=n_samples,
        )

        # Compute primary outcome probabilities (unpack tuple)
        probs, _ = self._compute_primary_probability(covariates, self._n_domains - 1)

        # Return mean probability (expected value)
        return float(np.mean(probs))
