"""Core library modules for credit risk modeling.

This subpackage contains the core implementation:
- portfolio: Obligor and Portfolio data structures
- model: Multi-factor Gaussian copula model
- simulation: Monte Carlo simulation engine
- risk_metrics: IRC and risk calculations
- lgd_distribution: Stochastic LGD models
"""

from .portfolio import Obligor, Portfolio
from .model import MultiFactorModel
from .simulation import MonteCarloEngine, ParallelMonteCarloEngine, SimulationResult
from .risk_metrics import (
    RiskCalculator,
    IncrementalRiskResult,
    RiskDecomposition,
    create_irc_report,
    create_decomposition_report
)
from .lgd_distribution import (
    LGDDistribution,
    ConstantLGD,
    BetaLGD,
    EmpiricalLGD,
    create_lgd_distribution
)

__all__ = [
    # Portfolio
    "Obligor",
    "Portfolio",
    # Model
    "MultiFactorModel",
    # Simulation
    "MonteCarloEngine",
    "ParallelMonteCarloEngine",
    "SimulationResult",
    # Risk metrics
    "RiskCalculator",
    "IncrementalRiskResult",
    "RiskDecomposition",
    "create_irc_report",
    "create_decomposition_report",
    # LGD distributions
    "LGDDistribution",
    "ConstantLGD",
    "BetaLGD",
    "EmpiricalLGD",
    "create_lgd_distribution",
]
