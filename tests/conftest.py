"""Pytest fixtures for credit risk model tests."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from credit_risk import (
    Obligor,
    Portfolio,
    MultiFactorModel,
    MonteCarloEngine,
    BetaLGD,
    EmpiricalLGD,
    ConstantLGD,
)


@pytest.fixture
def sample_obligor():
    """Create a simple obligor for testing."""
    return Obligor(
        name="TestObligor",
        pd=0.02,
        lgd=0.45,
        ead=1_000_000,
        factor_loadings={"global": 0.3, "technology": 0.4},
        sector="technology",
        country="us",
        rating="BBB"
    )


@pytest.fixture
def sample_obligor_with_beta_lgd():
    """Create an obligor with Beta LGD distribution."""
    return Obligor(
        name="BetaLGDObligor",
        pd=0.02,
        lgd=0.45,
        ead=1_000_000,
        factor_loadings={"global": 0.3, "financials": 0.4},
        sector="financials",
        country="us",
        rating="A",
        lgd_distribution=BetaLGD(mean=0.45, std=0.1, factor_sensitivity=0.2)
    )


@pytest.fixture
def sample_portfolio(sample_obligor):
    """Create a simple portfolio with one obligor."""
    portfolio = Portfolio(name="TestPortfolio")
    portfolio.add_obligor(sample_obligor)
    return portfolio


@pytest.fixture
def multi_obligor_portfolio():
    """Create a portfolio with multiple obligors across sectors."""
    portfolio = Portfolio(name="MultiObligorPortfolio")

    obligors = [
        Obligor(
            name="Tech_A",
            pd=0.02,
            lgd=0.45,
            ead=10_000_000,
            factor_loadings={"global": 0.3, "technology": 0.4, "us": 0.2},
            sector="technology",
            country="us",
            rating="BBB"
        ),
        Obligor(
            name="Bank_A",
            pd=0.01,
            lgd=0.55,
            ead=15_000_000,
            factor_loadings={"global": 0.4, "financials": 0.35, "us": 0.2},
            sector="financials",
            country="us",
            rating="A"
        ),
        Obligor(
            name="Energy_A",
            pd=0.03,
            lgd=0.60,
            ead=20_000_000,
            factor_loadings={"global": 0.25, "energy": 0.5, "europe": 0.15},
            sector="energy",
            country="europe",
            rating="BB"
        ),
    ]

    for obligor in obligors:
        portfolio.add_obligor(obligor)

    return portfolio


@pytest.fixture
def configured_model():
    """Create a multi-factor model with default correlations."""
    model = MultiFactorModel()
    model.set_default_correlation(
        inter_sector=0.3,
        intra_sector=0.5,
        global_correlation=0.2
    )
    return model


@pytest.fixture
def historical_lgd_data():
    """Generate sample historical LGD data for empirical distribution."""
    np.random.seed(42)
    return np.clip(np.random.beta(2, 3, size=100) * 0.8 + 0.1, 0.1, 0.9)


@pytest.fixture
def monte_carlo_engine(configured_model):
    """Create a Monte Carlo engine with fixed random state."""
    return MonteCarloEngine(configured_model, random_state=42)
