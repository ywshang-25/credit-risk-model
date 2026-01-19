"""LGD (Loss Given Default) distribution models.

Supports multiple approaches for simulating LGD:
- Constant: Fixed LGD value (deterministic)
- Beta: Parametric Beta distribution
- Empirical: Non-parametric empirical CDF from historical data
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d


class LGDDistribution(ABC):
    """Abstract base class for LGD distributions."""

    @abstractmethod
    def sample(self, n_samples: int, systematic_factor: Optional[np.ndarray] = None,
               random_state: Optional[int] = None) -> np.ndarray:
        """Sample LGD values from the distribution.

        Args:
            n_samples: Number of samples to draw
            systematic_factor: Optional systematic factor values for correlation
                              (array of shape (n_samples,))
            random_state: Random seed for reproducibility

        Returns:
            Array of LGD values of shape (n_samples,)
        """
        pass

    @abstractmethod
    def mean(self) -> float:
        """Return the mean (expected) LGD."""
        pass

    @abstractmethod
    def std(self) -> float:
        """Return the standard deviation of LGD."""
        pass


class ConstantLGD(LGDDistribution):
    """Constant (deterministic) LGD - no randomness.

    Use this for backward compatibility or when LGD uncertainty is not modeled.
    """

    def __init__(self, value: float):
        """Initialize constant LGD.

        Args:
            value: The fixed LGD value (between 0 and 1)
        """
        if not 0 <= value <= 1:
            raise ValueError(f"LGD must be between 0 and 1, got {value}")
        self._value = value

    def sample(self, n_samples: int, systematic_factor: Optional[np.ndarray] = None,
               random_state: Optional[int] = None) -> np.ndarray:
        """Return constant LGD values."""
        return np.full(n_samples, self._value)

    def mean(self) -> float:
        return self._value

    def std(self) -> float:
        return 0.0

    def __repr__(self) -> str:
        return f"ConstantLGD(value={self._value:.4f})"


class BetaLGD(LGDDistribution):
    """Beta-distributed LGD with optional systematic factor correlation.

    The Beta distribution is ideal for LGD as it's bounded between 0 and 1.
    Can incorporate correlation with systematic factors to capture the
    empirical observation that LGD increases during stress periods.

    Model:
        Base LGD ~ Beta(alpha, beta) scaled to [floor, cap]
        If factor_sensitivity > 0:
            LGD_adjusted = LGD_base + factor_sensitivity * systematic_factor * std(LGD)
    """

    def __init__(self, mean: float, std: float,
                 factor_sensitivity: float = 0.0,
                 floor: float = 0.0, cap: float = 1.0):
        """Initialize Beta LGD distribution.

        Args:
            mean: Mean LGD value (between floor and cap)
            std: Standard deviation of LGD
            factor_sensitivity: Sensitivity to systematic factors (0 = no correlation)
            floor: Minimum LGD value
            cap: Maximum LGD value
        """
        if not 0 <= floor < cap <= 1:
            raise ValueError(f"Must have 0 <= floor < cap <= 1")
        if not floor <= mean <= cap:
            raise ValueError(f"Mean must be between floor and cap")
        if std < 0:
            raise ValueError(f"Std must be non-negative")

        self._mean = mean
        self._std = std
        self._factor_sensitivity = factor_sensitivity
        self._floor = floor
        self._cap = cap

        # Convert to Beta parameters for the [0, 1] range
        # Then scale to [floor, cap]
        range_size = cap - floor
        if range_size > 0 and std > 0:
            # Normalize to [0, 1]
            normalized_mean = (mean - floor) / range_size
            normalized_std = std / range_size

            # Clamp normalized values to valid Beta range
            normalized_mean = np.clip(normalized_mean, 0.01, 0.99)
            max_std = np.sqrt(normalized_mean * (1 - normalized_mean))
            normalized_std = min(normalized_std, max_std * 0.99)

            # Calculate Beta parameters using method of moments
            variance = normalized_std ** 2
            common = normalized_mean * (1 - normalized_mean) / variance - 1
            self._alpha = normalized_mean * common
            self._beta = (1 - normalized_mean) * common

            # Ensure valid parameters
            self._alpha = max(0.1, self._alpha)
            self._beta = max(0.1, self._beta)
        else:
            # Degenerate case - constant
            self._alpha = None
            self._beta = None

    def sample(self, n_samples: int, systematic_factor: Optional[np.ndarray] = None,
               random_state: Optional[int] = None) -> np.ndarray:
        """Sample LGD values from Beta distribution."""
        rng = np.random.default_rng(random_state)

        if self._alpha is None or self._beta is None:
            # Constant case
            return np.full(n_samples, self._mean)

        # Sample from Beta and scale to [floor, cap]
        base_samples = rng.beta(self._alpha, self._beta, size=n_samples)
        lgd_values = self._floor + base_samples * (self._cap - self._floor)

        # Apply systematic factor adjustment if provided
        if systematic_factor is not None and self._factor_sensitivity != 0:
            # Higher systematic factor (worse economy) -> higher LGD
            adjustment = self._factor_sensitivity * systematic_factor * self._std
            lgd_values = lgd_values + adjustment

        # Clip to valid range
        lgd_values = np.clip(lgd_values, self._floor, self._cap)

        return lgd_values

    def mean(self) -> float:
        return self._mean

    def std(self) -> float:
        return self._std

    def __repr__(self) -> str:
        return (f"BetaLGD(mean={self._mean:.4f}, std={self._std:.4f}, "
                f"factor_sensitivity={self._factor_sensitivity:.2f})")


class EmpiricalLGD(LGDDistribution):
    """Empirical LGD distribution calibrated from historical data.

    Uses the empirical CDF from historical LGD observations.
    Supports optional systematic factor correlation through conditional quantiles.

    Model:
        1. Build empirical CDF from historical data
        2. Sample uniform random variables
        3. Apply inverse CDF to get LGD samples
        4. Optionally adjust based on systematic factor
    """

    def __init__(self, historical_lgd: Union[List[float], np.ndarray],
                 factor_sensitivity: float = 0.0,
                 floor: Optional[float] = None,
                 cap: Optional[float] = None,
                 interpolation: str = 'linear'):
        """Initialize Empirical LGD distribution.

        Args:
            historical_lgd: Array of historical LGD observations
            factor_sensitivity: Sensitivity to systematic factors
            floor: Optional minimum LGD (defaults to min of historical data)
            cap: Optional maximum LGD (defaults to max of historical data)
            interpolation: Interpolation method ('linear', 'nearest', 'cubic')
        """
        historical_lgd = np.asarray(historical_lgd)
        if len(historical_lgd) < 2:
            raise ValueError("Need at least 2 historical observations")

        self._historical_lgd = np.sort(historical_lgd)
        self._factor_sensitivity = factor_sensitivity

        # Set floor and cap
        self._floor = floor if floor is not None else float(np.min(historical_lgd))
        self._cap = cap if cap is not None else float(np.max(historical_lgd))

        # Build empirical CDF
        n = len(self._historical_lgd)
        self._ecdf_y = np.arange(1, n + 1) / n  # Cumulative probabilities

        # Build inverse CDF (quantile function) using interpolation
        # Add boundary points for extrapolation
        lgd_extended = np.concatenate([[self._floor], self._historical_lgd, [self._cap]])
        cdf_extended = np.concatenate([[0.0], self._ecdf_y, [1.0]])

        self._inverse_cdf = interp1d(
            cdf_extended, lgd_extended,
            kind=interpolation,
            bounds_error=False,
            fill_value=(self._floor, self._cap)
        )

        # Cache statistics
        self._mean_val = float(np.mean(historical_lgd))
        self._std_val = float(np.std(historical_lgd))

    def sample(self, n_samples: int, systematic_factor: Optional[np.ndarray] = None,
               random_state: Optional[int] = None) -> np.ndarray:
        """Sample LGD values using inverse CDF method."""
        rng = np.random.default_rng(random_state)

        # Sample uniform random variables
        u = rng.uniform(0, 1, size=n_samples)

        # Apply inverse CDF
        lgd_values = self._inverse_cdf(u)

        # Apply systematic factor adjustment if provided
        if systematic_factor is not None and self._factor_sensitivity != 0:
            adjustment = self._factor_sensitivity * systematic_factor * self._std_val
            lgd_values = lgd_values + adjustment

        # Clip to valid range
        lgd_values = np.clip(lgd_values, self._floor, self._cap)

        return lgd_values

    def mean(self) -> float:
        return self._mean_val

    def std(self) -> float:
        return self._std_val

    @property
    def n_observations(self) -> int:
        """Number of historical observations used."""
        return len(self._historical_lgd)

    def percentile(self, q: float) -> float:
        """Get the q-th percentile of the distribution.

        Args:
            q: Percentile (between 0 and 100)

        Returns:
            LGD value at the given percentile
        """
        return float(self._inverse_cdf(q / 100.0))

    def __repr__(self) -> str:
        return (f"EmpiricalLGD(n_obs={self.n_observations}, mean={self._mean_val:.4f}, "
                f"std={self._std_val:.4f}, factor_sensitivity={self._factor_sensitivity:.2f})")


def create_lgd_distribution(lgd_type: str, **kwargs) -> LGDDistribution:
    """Factory function to create LGD distributions.

    Args:
        lgd_type: Type of distribution ('constant', 'beta', 'empirical')
        **kwargs: Arguments passed to the distribution constructor

    Returns:
        LGDDistribution instance

    Examples:
        >>> create_lgd_distribution('constant', value=0.45)
        >>> create_lgd_distribution('beta', mean=0.45, std=0.1)
        >>> create_lgd_distribution('empirical', historical_lgd=[0.3, 0.4, 0.5, 0.6])
    """
    lgd_type = lgd_type.lower()

    if lgd_type == 'constant':
        return ConstantLGD(**kwargs)
    elif lgd_type == 'beta':
        return BetaLGD(**kwargs)
    elif lgd_type == 'empirical':
        return EmpiricalLGD(**kwargs)
    else:
        raise ValueError(f"Unknown LGD distribution type: {lgd_type}. "
                        f"Choose from: 'constant', 'beta', 'empirical'")
