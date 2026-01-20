"""Monte Carlo simulation engine for credit risk."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from .portfolio import Portfolio, Obligor
from .model import MultiFactorModel


@dataclass
class SimulationResult:
    """Results from a Monte Carlo simulation.

    Attributes:
        scenario_losses: Array of portfolio losses per scenario
        default_indicators: Boolean array of defaults (scenarios x obligors)
        obligor_losses: Array of losses per obligor per scenario
        factor_scenarios: The factor scenarios used
        num_scenarios: Number of scenarios simulated
        num_defaults_per_scenario: Number of defaults in each scenario
    """
    scenario_losses: np.ndarray
    default_indicators: np.ndarray
    obligor_losses: np.ndarray
    factor_scenarios: np.ndarray
    num_scenarios: int
    num_defaults_per_scenario: np.ndarray

    @property
    def expected_loss(self) -> float:
        """Average loss across all scenarios."""
        return float(np.mean(self.scenario_losses))

    @property
    def loss_std(self) -> float:
        """Standard deviation of losses."""
        return float(np.std(self.scenario_losses))

    @property
    def var(self, confidence: float = 0.99) -> float:
        """Value at Risk at specified confidence level."""
        return float(np.percentile(self.scenario_losses, confidence * 100))

    def get_var(self, confidence: float = 0.99) -> float:
        """Value at Risk at specified confidence level."""
        return float(np.percentile(self.scenario_losses, confidence * 100))

    @property
    def expected_shortfall(self, confidence: float = 0.99) -> float:
        """Expected Shortfall (CVaR) at specified confidence level."""
        var = self.var
        return float(np.mean(self.scenario_losses[self.scenario_losses >= var]))

    def get_expected_shortfall(self, confidence: float = 0.99) -> float:
        """Expected Shortfall (CVaR) at specified confidence level."""
        var = self.get_var(confidence)
        tail_losses = self.scenario_losses[self.scenario_losses >= var]
        if len(tail_losses) == 0:
            return var
        return float(np.mean(tail_losses))

    @property
    def default_rate(self) -> float:
        """Average default rate across scenarios."""
        return float(np.mean(self.default_indicators))

    def get_obligor_default_rate(self, obligor_idx: int) -> float:
        """Get the simulated default rate for a specific obligor."""
        return float(np.mean(self.default_indicators[:, obligor_idx]))

    def get_obligor_expected_loss(self, obligor_idx: int) -> float:
        """Get the expected loss for a specific obligor."""
        return float(np.mean(self.obligor_losses[:, obligor_idx]))


class MonteCarloEngine:
    """Monte Carlo simulation engine for credit portfolio risk.

    Generates scenarios, simulates defaults, and calculates losses.
    Supports parallel processing for large simulations.
    """

    def __init__(self, model: MultiFactorModel, random_state: Optional[int] = None):
        """Initialize the simulation engine.

        Args:
            model: The multi-factor model to use
            random_state: Random seed for reproducibility
        """
        self.model = model
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def simulate(self, portfolio: Portfolio, num_scenarios: int,
                 batch_size: Optional[int] = None) -> SimulationResult:
        """Run Monte Carlo simulation.

        Args:
            portfolio: The credit portfolio to simulate
            num_scenarios: Number of scenarios to generate
            batch_size: Optional batch size for memory efficiency

        Returns:
            SimulationResult with all simulation outputs
        """
        if batch_size is None or batch_size >= num_scenarios:
            return self._simulate_batch(portfolio, num_scenarios)

        all_losses = []
        all_defaults = []
        all_obligor_losses = []
        all_factors = []

        remaining = num_scenarios
        while remaining > 0:
            current_batch = min(batch_size, remaining)
            result = self._simulate_batch(portfolio, current_batch)
            all_losses.append(result.scenario_losses)
            all_defaults.append(result.default_indicators)
            all_obligor_losses.append(result.obligor_losses)
            all_factors.append(result.factor_scenarios)
            remaining -= current_batch

        scenario_losses = np.concatenate(all_losses)
        default_indicators = np.concatenate(all_defaults)
        obligor_losses = np.concatenate(all_obligor_losses)
        factor_scenarios = np.concatenate(all_factors)

        return SimulationResult(
            scenario_losses=scenario_losses,
            default_indicators=default_indicators,
            obligor_losses=obligor_losses,
            factor_scenarios=factor_scenarios,
            num_scenarios=num_scenarios,
            num_defaults_per_scenario=np.sum(default_indicators, axis=1)
        )

    def _simulate_batch(self, portfolio: Portfolio,
                        num_scenarios: int) -> SimulationResult:
        """Simulate a single batch of scenarios.

        Args:
            portfolio: The credit portfolio
            num_scenarios: Number of scenarios in this batch

        Returns:
            SimulationResult for this batch
        """
        seed1 = self._rng.integers(0, 2**31)
        seed2 = self._rng.integers(0, 2**31)

        factor_scenarios = self.model.generate_factor_scenarios(
            num_scenarios, random_state=seed1
        )

        asset_returns = self.model.simulate_asset_returns(
            portfolio, factor_scenarios, random_state=seed2
        )

        obligors = portfolio.obligors
        num_obligors = len(obligors)

        default_thresholds = np.array([
            self.model.calculate_default_threshold(o) for o in obligors
        ])

        default_indicators = asset_returns < default_thresholds

        # Sample LGD for each obligor (stochastic or constant)
        seed3 = self._rng.integers(0, 2**31)
        lgd_matrix = self._sample_lgd_matrix(
            obligors, num_scenarios, factor_scenarios, seed3
        )
        ead_array = np.array([o.ead for o in obligors])

        obligor_losses = default_indicators * lgd_matrix * ead_array

        scenario_losses = np.sum(obligor_losses, axis=1)

        return SimulationResult(
            scenario_losses=scenario_losses,
            default_indicators=default_indicators,
            obligor_losses=obligor_losses,
            factor_scenarios=factor_scenarios,
            num_scenarios=num_scenarios,
            num_defaults_per_scenario=np.sum(default_indicators, axis=1)
        )

    def simulate_without_obligor(self, portfolio: Portfolio, obligor_name: str,
                                 num_scenarios: int,
                                 factor_scenarios: Optional[np.ndarray] = None,
                                 random_state: Optional[int] = None) -> SimulationResult:
        """Simulate portfolio loss without a specific obligor.

        Used for calculating incremental risk contribution.

        Args:
            portfolio: The full portfolio
            obligor_name: Name of obligor to exclude
            num_scenarios: Number of scenarios
            factor_scenarios: Pre-generated factor scenarios (for consistency)
            random_state: Random seed for idiosyncratic shocks

        Returns:
            SimulationResult for portfolio without the specified obligor
        """
        reduced_portfolio = portfolio.copy()
        reduced_portfolio.remove_obligor(obligor_name)

        if factor_scenarios is None:
            seed = random_state if random_state else self._rng.integers(0, 2**31)
            factor_scenarios = self.model.generate_factor_scenarios(
                num_scenarios, random_state=seed
            )

        seed2 = random_state if random_state else self._rng.integers(0, 2**31)
        asset_returns = self.model.simulate_asset_returns(
            reduced_portfolio, factor_scenarios, random_state=seed2
        )

        obligors = reduced_portfolio.obligors
        default_thresholds = np.array([
            self.model.calculate_default_threshold(o) for o in obligors
        ])

        default_indicators = asset_returns < default_thresholds

        # Sample LGD for each obligor (stochastic or constant)
        seed3 = random_state if random_state else self._rng.integers(0, 2**31)
        lgd_matrix = self._sample_lgd_matrix(
            obligors, num_scenarios, factor_scenarios, seed3 + 1
        )
        ead_array = np.array([o.ead for o in obligors])

        obligor_losses = default_indicators * lgd_matrix * ead_array
        scenario_losses = np.sum(obligor_losses, axis=1)

        return SimulationResult(
            scenario_losses=scenario_losses,
            default_indicators=default_indicators,
            obligor_losses=obligor_losses,
            factor_scenarios=factor_scenarios,
            num_scenarios=num_scenarios,
            num_defaults_per_scenario=np.sum(default_indicators, axis=1)
        )

    def _sample_lgd_matrix(self, obligors: List, num_scenarios: int,
                           factor_scenarios: np.ndarray,
                           random_state: int) -> np.ndarray:
        """Sample LGD values for all obligors across all scenarios.

        Uses vectorized numpy operations for efficiency with large portfolios.

        Args:
            obligors: List of Obligor objects
            num_scenarios: Number of scenarios
            factor_scenarios: Factor values for systematic correlation
            random_state: Random seed

        Returns:
            LGD matrix of shape (num_scenarios, num_obligors)
        """
        num_obligors = len(obligors)
        lgd_matrix = np.zeros((num_scenarios, num_obligors))

        # Use global factor (first factor) as systematic factor for LGD correlation
        systematic_factor = factor_scenarios[:, 0] if factor_scenarios.shape[1] > 0 else None

        rng = np.random.default_rng(random_state)

        # Classify obligors by distribution type for vectorized processing
        constant_indices = []
        constant_values = []
        beta_indices = []
        beta_params = []  # (alpha, beta, floor, cap, factor_sensitivity, std)
        empirical_obligors = []  # (index, obligor) - need individual processing

        for i, obligor in enumerate(obligors):
            if not obligor.has_stochastic_lgd:
                constant_indices.append(i)
                constant_values.append(obligor.lgd)
            else:
                dist = obligor.lgd_distribution
                dist_type = getattr(dist, 'distribution_type', 'unknown')

                if dist_type == 'constant':
                    params = dist.get_params()
                    constant_indices.append(i)
                    constant_values.append(params['value'])
                elif dist_type == 'beta':
                    params = dist.get_params()
                    beta_indices.append(i)
                    beta_params.append((
                        params['alpha'],
                        params['beta'],
                        params['floor'],
                        params['cap'],
                        params['factor_sensitivity'],
                        params['std']
                    ))
                else:
                    # Empirical or unknown - process individually
                    empirical_obligors.append((i, obligor))

        # Process constant LGD obligors (vectorized)
        if constant_indices:
            constant_indices = np.array(constant_indices)
            constant_values = np.array(constant_values)
            lgd_matrix[:, constant_indices] = constant_values

        # Process Beta LGD obligors (vectorized)
        if beta_indices:
            lgd_matrix = self._sample_beta_lgd_vectorized(
                lgd_matrix, beta_indices, beta_params,
                num_scenarios, systematic_factor, rng
            )

        # Process Empirical LGD obligors (individual - each has unique CDF)
        for idx, obligor in empirical_obligors:
            obligor_seed = rng.integers(0, 2**31)
            lgd_matrix[:, idx] = obligor.sample_lgd(
                num_scenarios,
                systematic_factor=systematic_factor,
                random_state=obligor_seed
            )

        return lgd_matrix

    def _sample_beta_lgd_vectorized(self, lgd_matrix: np.ndarray,
                                     beta_indices: List[int],
                                     beta_params: List[Tuple],
                                     num_scenarios: int,
                                     systematic_factor: Optional[np.ndarray],
                                     rng: np.random.Generator) -> np.ndarray:
        """Vectorized sampling for all Beta LGD distributions.

        Args:
            lgd_matrix: Output matrix to fill
            beta_indices: Indices of obligors with Beta LGD
            beta_params: List of (alpha, beta, floor, cap, factor_sensitivity, std)
            num_scenarios: Number of scenarios
            systematic_factor: Systematic factor values
            rng: Random number generator

        Returns:
            Updated lgd_matrix
        """
        n_beta = len(beta_indices)
        beta_indices = np.array(beta_indices)

        # Extract parameters as arrays
        alphas = np.array([p[0] for p in beta_params])
        betas = np.array([p[1] for p in beta_params])
        floors = np.array([p[2] for p in beta_params])
        caps = np.array([p[3] for p in beta_params])
        factor_sensitivities = np.array([p[4] for p in beta_params])
        stds = np.array([p[5] for p in beta_params])

        # Handle degenerate cases (None alpha/beta means constant)
        valid_mask = (alphas != None) & (betas != None)
        valid_mask = np.array([a is not None and b is not None
                               for a, b in zip(alphas, betas)])

        if not np.any(valid_mask):
            # All are degenerate (constant) - use midpoint of floor/cap
            lgd_matrix[:, beta_indices] = (floors + caps) / 2
            return lgd_matrix

        # Sample all valid Beta distributions at once
        # Shape: (num_scenarios, n_beta)
        valid_indices = np.where(valid_mask)[0]
        valid_alphas = alphas[valid_mask].astype(float)
        valid_betas = betas[valid_mask].astype(float)
        valid_floors = floors[valid_mask]
        valid_caps = caps[valid_mask]
        valid_sensitivities = factor_sensitivities[valid_mask]
        valid_stds = stds[valid_mask]
        valid_obligor_indices = beta_indices[valid_mask]

        # Vectorized beta sampling: shape (num_scenarios, n_valid_beta)
        base_samples = rng.beta(valid_alphas, valid_betas,
                                size=(num_scenarios, len(valid_alphas)))

        # Scale to [floor, cap]
        lgd_values = valid_floors + base_samples * (valid_caps - valid_floors)

        # Apply systematic factor adjustment if provided
        if systematic_factor is not None:
            # systematic_factor: (num_scenarios,)
            # valid_sensitivities, valid_stds: (n_valid_beta,)
            # Broadcast to get adjustment: (num_scenarios, n_valid_beta)
            adjustment = (systematic_factor[:, np.newaxis] *
                         valid_sensitivities * valid_stds)
            lgd_values = lgd_values + adjustment

        # Clip to valid range
        lgd_values = np.clip(lgd_values, valid_floors, valid_caps)

        # Assign to matrix
        lgd_matrix[:, valid_obligor_indices] = lgd_values

        # Handle degenerate (constant) cases
        if not np.all(valid_mask):
            invalid_indices = np.where(~valid_mask)[0]
            invalid_obligor_indices = beta_indices[~valid_mask]
            invalid_floors = floors[~valid_mask]
            invalid_caps = caps[~valid_mask]
            lgd_matrix[:, invalid_obligor_indices] = (invalid_floors + invalid_caps) / 2

        return lgd_matrix


def _simulate_chunk(args: Tuple) -> SimulationResult:
    """Worker function for parallel simulation.

    Args:
        args: Tuple of (model, portfolio, num_scenarios, seed)

    Returns:
        SimulationResult for this chunk
    """
    model, portfolio, num_scenarios, seed = args
    engine = MonteCarloEngine(model, random_state=seed)
    return engine.simulate(portfolio, num_scenarios)


class ParallelMonteCarloEngine:
    """Parallel Monte Carlo engine using multiple processes."""

    def __init__(self, model: MultiFactorModel, num_workers: Optional[int] = None,
                 random_state: Optional[int] = None):
        """Initialize parallel engine.

        Args:
            model: The multi-factor model
            num_workers: Number of parallel workers (None = CPU count)
            random_state: Random seed
        """
        self.model = model
        self.num_workers = num_workers
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def simulate(self, portfolio: Portfolio, num_scenarios: int) -> SimulationResult:
        """Run parallel Monte Carlo simulation.

        Args:
            portfolio: The credit portfolio
            num_scenarios: Total number of scenarios

        Returns:
            Combined SimulationResult
        """
        from multiprocessing import cpu_count
        workers = self.num_workers or cpu_count()
        scenarios_per_worker = num_scenarios // workers
        remainder = num_scenarios % workers

        chunks = []
        for i in range(workers):
            n = scenarios_per_worker + (1 if i < remainder else 0)
            seed = self._rng.integers(0, 2**31)
            chunks.append((self.model, portfolio, n, seed))

        results = []
        for chunk in chunks:
            result = _simulate_chunk(chunk)
            results.append(result)

        scenario_losses = np.concatenate([r.scenario_losses for r in results])
        default_indicators = np.concatenate([r.default_indicators for r in results])
        obligor_losses = np.concatenate([r.obligor_losses for r in results])
        factor_scenarios = np.concatenate([r.factor_scenarios for r in results])

        return SimulationResult(
            scenario_losses=scenario_losses,
            default_indicators=default_indicators,
            obligor_losses=obligor_losses,
            factor_scenarios=factor_scenarios,
            num_scenarios=num_scenarios,
            num_defaults_per_scenario=np.sum(default_indicators, axis=1)
        )
