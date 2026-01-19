"""Multi-factor Gaussian copula model for credit risk."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import norm

from .portfolio import Obligor, Portfolio


class MultiFactorModel:
    """Multi-factor model for correlated credit defaults.

    Implements the asset return model:
        R_i = Σ_k (β_ik × F_k) + √(1 - Σ_k β_ik²) × ε_i

    Where:
        - F_k: Systematic factor k (standard normal)
        - β_ik: Factor loading of obligor i on factor k
        - ε_i: Idiosyncratic shock (standard normal)

    Default occurs when R_i < Φ⁻¹(PD_i)
    """

    def __init__(self, factor_names: Optional[List[str]] = None):
        """Initialize the multi-factor model.

        Args:
            factor_names: List of factor names. If None, uses default factors.
        """
        if factor_names is None:
            self.factor_names = [
                "global",
                "financials",
                "technology",
                "healthcare",
                "energy",
                "consumer",
                "industrials",
                "us",
                "europe",
                "asia"
            ]
        else:
            self.factor_names = list(factor_names)

        self._factor_correlation: Optional[np.ndarray] = None
        self._factor_cholesky: Optional[np.ndarray] = None

    @property
    def num_factors(self) -> int:
        """Number of factors in the model."""
        return len(self.factor_names)

    def set_factor_correlation(self, correlation_matrix: np.ndarray) -> None:
        """Set the factor correlation matrix.

        Args:
            correlation_matrix: Symmetric positive semi-definite matrix
                               of shape (num_factors, num_factors)
        """
        n = self.num_factors
        if correlation_matrix.shape != (n, n):
            raise ValueError(
                f"Correlation matrix must be {n}x{n}, got {correlation_matrix.shape}"
            )

        if not np.allclose(correlation_matrix, correlation_matrix.T):
            raise ValueError("Correlation matrix must be symmetric")

        diag = np.diag(correlation_matrix)
        if not np.allclose(diag, 1.0):
            raise ValueError("Diagonal elements must be 1")

        if np.any(correlation_matrix < -1) or np.any(correlation_matrix > 1):
            raise ValueError("Correlations must be between -1 and 1")

        eigvals = np.linalg.eigvalsh(correlation_matrix)
        if np.any(eigvals < -1e-10):
            raise ValueError("Correlation matrix must be positive semi-definite")

        self._factor_correlation = correlation_matrix.copy()
        self._factor_cholesky = np.linalg.cholesky(
            correlation_matrix + np.eye(n) * 1e-10
        )

    def set_default_correlation(self, inter_sector: float = 0.3,
                                 intra_sector: float = 0.5,
                                 global_correlation: float = 0.2) -> None:
        """Set a default correlation structure.

        Creates a block correlation structure:
        - Global factor correlated with all other factors
        - Sector factors correlated within groups
        - Geographic factors correlated within groups

        Args:
            inter_sector: Correlation between different sectors
            intra_sector: Correlation within same category
            global_correlation: Correlation of global factor with others
        """
        n = self.num_factors
        corr = np.eye(n)

        sector_indices = []
        geo_indices = []
        for i, name in enumerate(self.factor_names):
            if name == "global":
                continue
            elif name in ["us", "europe", "asia"]:
                geo_indices.append(i)
            else:
                sector_indices.append(i)

        global_idx = self.factor_names.index("global") if "global" in self.factor_names else None

        if global_idx is not None:
            for i in range(n):
                if i != global_idx:
                    corr[global_idx, i] = global_correlation
                    corr[i, global_idx] = global_correlation

        for i in sector_indices:
            for j in sector_indices:
                if i != j:
                    corr[i, j] = inter_sector

        for i in geo_indices:
            for j in geo_indices:
                if i != j:
                    corr[i, j] = intra_sector

        self.set_factor_correlation(corr)

    def get_factor_index(self, factor_name: str) -> int:
        """Get the index of a factor by name."""
        try:
            return self.factor_names.index(factor_name)
        except ValueError:
            raise KeyError(f"Factor '{factor_name}' not found in model")

    def get_loading_vector(self, obligor: Obligor) -> np.ndarray:
        """Get the factor loading vector for an obligor.

        Args:
            obligor: The obligor to get loadings for

        Returns:
            Array of shape (num_factors,) with factor loadings
        """
        loadings = np.zeros(self.num_factors)
        for factor_name, loading in obligor.factor_loadings.items():
            if factor_name in self.factor_names:
                idx = self.get_factor_index(factor_name)
                loadings[idx] = loading
        return loadings

    def calculate_asset_correlation(self, obligor1: Obligor,
                                    obligor2: Obligor) -> float:
        """Calculate asset correlation between two obligors.

        The asset correlation is:
            ρ_ij = β_i' × Σ × β_j

        Where Σ is the factor correlation matrix.

        Args:
            obligor1: First obligor
            obligor2: Second obligor

        Returns:
            Asset correlation coefficient
        """
        if self._factor_correlation is None:
            raise ValueError("Factor correlation matrix not set")

        beta_i = self.get_loading_vector(obligor1)
        beta_j = self.get_loading_vector(obligor2)

        correlation = beta_i @ self._factor_correlation @ beta_j
        return float(correlation)

    def calculate_default_threshold(self, obligor: Obligor) -> float:
        """Calculate the default threshold for an obligor.

        Default occurs when R_i < Φ⁻¹(PD_i)

        Args:
            obligor: The obligor

        Returns:
            Default threshold (inverse normal CDF of PD)
        """
        if obligor.pd <= 0:
            return -np.inf
        elif obligor.pd >= 1:
            return np.inf
        return norm.ppf(obligor.pd)

    def generate_factor_scenarios(self, num_scenarios: int,
                                  random_state: Optional[int] = None) -> np.ndarray:
        """Generate correlated factor scenarios.

        Args:
            num_scenarios: Number of scenarios to generate
            random_state: Random seed for reproducibility

        Returns:
            Array of shape (num_scenarios, num_factors) with factor values
        """
        if self._factor_cholesky is None:
            raise ValueError("Factor correlation not set. Call set_factor_correlation first.")

        rng = np.random.default_rng(random_state)
        independent_factors = rng.standard_normal((num_scenarios, self.num_factors))
        correlated_factors = independent_factors @ self._factor_cholesky.T

        return correlated_factors

    def simulate_asset_returns(self, portfolio: Portfolio,
                               factor_scenarios: np.ndarray,
                               random_state: Optional[int] = None) -> np.ndarray:
        """Simulate asset returns for all obligors across scenarios.

        Args:
            portfolio: The credit portfolio
            factor_scenarios: Factor values of shape (num_scenarios, num_factors)
            random_state: Random seed for idiosyncratic shocks

        Returns:
            Array of shape (num_scenarios, num_obligors) with asset returns
        """
        num_scenarios = factor_scenarios.shape[0]
        num_obligors = len(portfolio)
        rng = np.random.default_rng(random_state)

        loading_matrix = np.zeros((num_obligors, self.num_factors))
        idio_weights = np.zeros(num_obligors)

        for i, obligor in enumerate(portfolio.obligors):
            loading_matrix[i] = self.get_loading_vector(obligor)
            idio_weights[i] = obligor.idiosyncratic_weight

        systematic_returns = factor_scenarios @ loading_matrix.T
        idiosyncratic_shocks = rng.standard_normal((num_scenarios, num_obligors))
        asset_returns = systematic_returns + idiosyncratic_shocks * idio_weights

        return asset_returns

    def get_portfolio_correlation_matrix(self, portfolio: Portfolio) -> np.ndarray:
        """Calculate the asset correlation matrix for a portfolio.

        Args:
            portfolio: The credit portfolio

        Returns:
            Correlation matrix of shape (num_obligors, num_obligors)
        """
        n = len(portfolio)
        corr_matrix = np.eye(n)

        obligors = portfolio.obligors
        for i in range(n):
            for j in range(i + 1, n):
                corr = self.calculate_asset_correlation(obligors[i], obligors[j])
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

        return corr_matrix
