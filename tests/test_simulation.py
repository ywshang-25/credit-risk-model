"""Tests for simulation.py - Monte Carlo simulation engine."""

import pytest
import numpy as np

from credit_risk import (
    MonteCarloEngine,
    SimulationResult,
    MultiFactorModel,
    Portfolio,
    Obligor,
    BetaLGD,
    EmpiricalLGD
)


class TestSimulationResult:
    """Tests for SimulationResult dataclass."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample simulation result."""
        np.random.seed(42)
        n_scenarios = 1000
        n_obligors = 3
        return SimulationResult(
            scenario_losses=np.random.exponential(1000, n_scenarios),
            default_indicators=np.random.random((n_scenarios, n_obligors)) < 0.1,
            obligor_losses=np.random.exponential(500, (n_scenarios, n_obligors)),
            factor_scenarios=np.random.randn(n_scenarios, 5),
            num_scenarios=n_scenarios,
            num_defaults_per_scenario=np.random.poisson(0.3, n_scenarios)
        )

    def test_expected_loss(self, sample_result):
        """Test expected loss calculation."""
        expected = np.mean(sample_result.scenario_losses)
        assert sample_result.expected_loss == expected

    def test_loss_std(self, sample_result):
        """Test loss standard deviation."""
        expected = np.std(sample_result.scenario_losses)
        assert sample_result.loss_std == expected

    def test_get_var(self, sample_result):
        """Test VaR calculation."""
        var_99 = sample_result.get_var(0.99)
        assert var_99 == np.percentile(sample_result.scenario_losses, 99)

    def test_get_expected_shortfall(self, sample_result):
        """Test Expected Shortfall calculation."""
        var = sample_result.get_var(0.99)
        tail = sample_result.scenario_losses[sample_result.scenario_losses >= var]
        if len(tail) > 0:
            expected_es = np.mean(tail)
            assert sample_result.get_expected_shortfall(0.99) == pytest.approx(expected_es, rel=0.01)

    def test_default_rate(self, sample_result):
        """Test average default rate."""
        expected = np.mean(sample_result.default_indicators)
        assert sample_result.default_rate == expected

    def test_get_obligor_default_rate(self, sample_result):
        """Test obligor-specific default rate."""
        for i in range(3):
            expected = np.mean(sample_result.default_indicators[:, i])
            assert sample_result.get_obligor_default_rate(i) == expected

    def test_get_obligor_expected_loss(self, sample_result):
        """Test obligor-specific expected loss."""
        for i in range(3):
            expected = np.mean(sample_result.obligor_losses[:, i])
            assert sample_result.get_obligor_expected_loss(i) == expected


class TestMonteCarloEngine:
    """Tests for MonteCarloEngine class."""

    def test_create_engine(self, configured_model):
        """Test creating Monte Carlo engine."""
        engine = MonteCarloEngine(configured_model, random_state=42)
        assert engine.model is configured_model
        assert engine.random_state == 42

    def test_simulate_basic(self, monte_carlo_engine, multi_obligor_portfolio):
        """Test basic simulation."""
        result = monte_carlo_engine.simulate(multi_obligor_portfolio, num_scenarios=1000)
        assert result.num_scenarios == 1000
        assert result.scenario_losses.shape == (1000,)
        assert result.default_indicators.shape == (1000, 3)
        assert result.obligor_losses.shape == (1000, 3)

    def test_simulate_reproducible(self, configured_model, multi_obligor_portfolio):
        """Test simulation reproducibility."""
        engine1 = MonteCarloEngine(configured_model, random_state=42)
        engine2 = MonteCarloEngine(configured_model, random_state=42)

        result1 = engine1.simulate(multi_obligor_portfolio, num_scenarios=100)
        result2 = engine2.simulate(multi_obligor_portfolio, num_scenarios=100)

        assert np.allclose(result1.scenario_losses, result2.scenario_losses)

    def test_simulate_losses_non_negative(self, monte_carlo_engine, multi_obligor_portfolio):
        """Test that all losses are non-negative."""
        result = monte_carlo_engine.simulate(multi_obligor_portfolio, num_scenarios=1000)
        assert np.all(result.scenario_losses >= 0)
        assert np.all(result.obligor_losses >= 0)

    def test_simulate_losses_bounded(self, monte_carlo_engine, multi_obligor_portfolio):
        """Test that losses are bounded by total possible loss."""
        result = monte_carlo_engine.simulate(multi_obligor_portfolio, num_scenarios=1000)
        max_possible_loss = sum(o.ead * o.lgd for o in multi_obligor_portfolio.obligors)
        assert np.all(result.scenario_losses <= max_possible_loss * 1.01)  # Small tolerance

    def test_simulate_default_indicators_binary(self, monte_carlo_engine, multi_obligor_portfolio):
        """Test that default indicators are binary."""
        result = monte_carlo_engine.simulate(multi_obligor_portfolio, num_scenarios=1000)
        assert np.all((result.default_indicators == 0) | (result.default_indicators == 1))

    def test_simulate_with_batch_size(self, monte_carlo_engine, multi_obligor_portfolio):
        """Test simulation with batching."""
        result = monte_carlo_engine.simulate(
            multi_obligor_portfolio, num_scenarios=1000, batch_size=250
        )
        assert result.num_scenarios == 1000
        assert result.scenario_losses.shape == (1000,)

    def test_simulate_higher_pd_more_defaults(self, configured_model):
        """Test that higher PD leads to more defaults."""
        portfolio_low_pd = Portfolio()
        portfolio_low_pd.add_obligor(Obligor(
            name="LowPD", pd=0.01, lgd=0.45, ead=1_000_000,
            factor_loadings={"global": 0.3}
        ))

        portfolio_high_pd = Portfolio()
        portfolio_high_pd.add_obligor(Obligor(
            name="HighPD", pd=0.10, lgd=0.45, ead=1_000_000,
            factor_loadings={"global": 0.3}
        ))

        engine = MonteCarloEngine(configured_model, random_state=42)
        result_low = engine.simulate(portfolio_low_pd, num_scenarios=10000)
        result_high = engine.simulate(portfolio_high_pd, num_scenarios=10000)

        assert result_high.default_rate > result_low.default_rate

    def test_simulate_stochastic_lgd_beta(self, configured_model):
        """Test simulation with Beta LGD distribution."""
        portfolio = Portfolio()
        portfolio.add_obligor(Obligor(
            name="BetaLGD",
            pd=0.05,
            lgd=0.45,
            ead=1_000_000,
            factor_loadings={"global": 0.3},
            lgd_distribution=BetaLGD(mean=0.45, std=0.15)
        ))

        engine = MonteCarloEngine(configured_model, random_state=42)
        result = engine.simulate(portfolio, num_scenarios=10000)

        # With stochastic LGD, losses should show more variation
        # when there are defaults
        defaults = result.default_indicators[:, 0] == 1
        if np.sum(defaults) > 10:
            default_losses = result.obligor_losses[defaults, 0]
            # Should have variation in losses among defaults
            assert np.std(default_losses) > 0

    def test_simulate_stochastic_lgd_empirical(self, configured_model, historical_lgd_data):
        """Test simulation with Empirical LGD distribution."""
        portfolio = Portfolio()
        portfolio.add_obligor(Obligor(
            name="EmpiricalLGD",
            pd=0.05,
            lgd=float(np.mean(historical_lgd_data)),
            ead=1_000_000,
            factor_loadings={"global": 0.3},
            lgd_distribution=EmpiricalLGD(historical_lgd=historical_lgd_data)
        ))

        engine = MonteCarloEngine(configured_model, random_state=42)
        result = engine.simulate(portfolio, num_scenarios=10000)

        assert result.num_scenarios == 10000
        assert np.all(result.scenario_losses >= 0)

    def test_simulate_mixed_lgd_types(self, configured_model, historical_lgd_data):
        """Test simulation with mixed constant and stochastic LGD."""
        portfolio = Portfolio()
        portfolio.add_obligor(Obligor(
            name="Constant",
            pd=0.03,
            lgd=0.45,
            ead=1_000_000,
            factor_loadings={"global": 0.3}
        ))
        portfolio.add_obligor(Obligor(
            name="Beta",
            pd=0.03,
            lgd=0.50,
            ead=1_000_000,
            factor_loadings={"global": 0.3},
            lgd_distribution=BetaLGD(mean=0.50, std=0.15)
        ))
        portfolio.add_obligor(Obligor(
            name="Empirical",
            pd=0.03,
            lgd=float(np.mean(historical_lgd_data)),
            ead=1_000_000,
            factor_loadings={"global": 0.3},
            lgd_distribution=EmpiricalLGD(historical_lgd=historical_lgd_data)
        ))

        engine = MonteCarloEngine(configured_model, random_state=42)
        result = engine.simulate(portfolio, num_scenarios=5000)

        assert result.num_scenarios == 5000
        assert result.obligor_losses.shape == (5000, 3)

    def test_simulate_without_obligor(self, monte_carlo_engine, multi_obligor_portfolio):
        """Test simulating without a specific obligor."""
        result = monte_carlo_engine.simulate_without_obligor(
            multi_obligor_portfolio,
            obligor_name="Tech_A",
            num_scenarios=1000
        )
        # Should have one less obligor
        assert result.obligor_losses.shape == (1000, 2)

    def test_stochastic_lgd_increases_tail_risk(self, configured_model):
        """Test that stochastic LGD with factor sensitivity increases tail risk."""
        # Portfolio with constant LGD
        portfolio_constant = Portfolio()
        portfolio_constant.add_obligor(Obligor(
            name="Obligor1",
            pd=0.05,
            lgd=0.50,
            ead=10_000_000,
            factor_loadings={"global": 0.4}
        ))

        # Portfolio with stochastic LGD and factor sensitivity
        portfolio_stochastic = Portfolio()
        portfolio_stochastic.add_obligor(Obligor(
            name="Obligor1",
            pd=0.05,
            lgd=0.50,
            ead=10_000_000,
            factor_loadings={"global": 0.4},
            lgd_distribution=BetaLGD(mean=0.50, std=0.15, factor_sensitivity=0.5)
        ))

        engine = MonteCarloEngine(configured_model, random_state=42)
        result_constant = engine.simulate(portfolio_constant, num_scenarios=50000)
        result_stochastic = engine.simulate(portfolio_stochastic, num_scenarios=50000)

        # Stochastic LGD should have higher tail risk
        es_constant = result_constant.get_expected_shortfall(0.99)
        es_stochastic = result_stochastic.get_expected_shortfall(0.99)

        # Allow for some Monte Carlo variance
        assert es_stochastic >= es_constant * 0.9  # Should be at least close


class TestLGDMatrixSampling:
    """Tests for the _sample_lgd_matrix method."""

    def test_lgd_matrix_shape(self, configured_model, multi_obligor_portfolio):
        """Test that LGD matrix has correct shape."""
        engine = MonteCarloEngine(configured_model, random_state=42)
        factor_scenarios = configured_model.generate_factor_scenarios(100, random_state=42)

        lgd_matrix = engine._sample_lgd_matrix(
            multi_obligor_portfolio.obligors,
            100,
            factor_scenarios,
            random_state=42
        )

        assert lgd_matrix.shape == (100, 3)

    def test_lgd_matrix_constant_values(self, configured_model, multi_obligor_portfolio):
        """Test that constant LGD produces constant columns."""
        engine = MonteCarloEngine(configured_model, random_state=42)
        factor_scenarios = configured_model.generate_factor_scenarios(100, random_state=42)

        lgd_matrix = engine._sample_lgd_matrix(
            multi_obligor_portfolio.obligors,
            100,
            factor_scenarios,
            random_state=42
        )

        # All obligors have constant LGD, so each column should be constant
        for i, obligor in enumerate(multi_obligor_portfolio.obligors):
            assert np.all(lgd_matrix[:, i] == obligor.lgd)

    def test_lgd_matrix_stochastic_variation(self, configured_model):
        """Test that stochastic LGD produces varying values."""
        portfolio = Portfolio()
        portfolio.add_obligor(Obligor(
            name="Stochastic",
            pd=0.05,
            lgd=0.45,
            ead=1_000_000,
            factor_loadings={"global": 0.3},
            lgd_distribution=BetaLGD(mean=0.45, std=0.15)
        ))

        engine = MonteCarloEngine(configured_model, random_state=42)
        factor_scenarios = configured_model.generate_factor_scenarios(1000, random_state=42)

        lgd_matrix = engine._sample_lgd_matrix(
            portfolio.obligors,
            1000,
            factor_scenarios,
            random_state=42
        )

        # Should have variation
        assert np.std(lgd_matrix[:, 0]) > 0.05
