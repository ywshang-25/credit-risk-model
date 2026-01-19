"""Multi-factor credit risk model for incremental default loss calculation.

This package provides tools for credit portfolio risk analysis using a
multi-factor Gaussian copula model.

Main components:
- portfolio: Obligor and Portfolio data structures
- model: Multi-factor model with factor correlations
- simulation: Monte Carlo simulation engine
- risk_metrics: Incremental Risk Contribution (IRC) calculations
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

__version__ = "1.0.0"

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
]
