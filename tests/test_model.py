"""Tests for model.py - MultiFactorModel class."""

import pytest
import numpy as np
from scipy.stats import norm

from credit_risk import MultiFactorModel, Obligor, Portfolio


class TestMultiFactorModel:
    """Tests for the MultiFactorModel class."""

    def test_create_model_default_factors(self):
        """Test creating model with default factors."""
        model = MultiFactorModel()
        assert model.num_factors == 10
        assert "global" in model.factor_names
        assert "technology" in model.factor_names
        assert "us" in model.factor_names

    def test_create_model_custom_factors(self):
        """Test creating model with custom factors."""
        factors = ["factor1", "factor2", "factor3"]
        model = MultiFactorModel(factor_names=factors)
        assert model.num_factors == 3
        assert model.factor_names == factors

    def test_set_factor_correlation_valid(self, configured_model):
        """Test setting a valid correlation matrix."""
        assert configured_model._factor_correlation is not None
        assert configured_model._factor_cholesky is not None

    def test_set_factor_correlation_invalid_shape(self):
        """Test that invalid shape raises error."""
        model = MultiFactorModel(factor_names=["a", "b", "c"])
        corr = np.eye(2)  # Wrong size
        with pytest.raises(ValueError, match="must be"):
            model.set_factor_correlation(corr)

    def test_set_factor_correlation_not_symmetric(self):
        """Test that non-symmetric matrix raises error."""
        model = MultiFactorModel(factor_names=["a", "b"])
        corr = np.array([[1.0, 0.5], [0.3, 1.0]])  # Not symmetric
        with pytest.raises(ValueError, match="symmetric"):
            model.set_factor_correlation(corr)

    def test_set_factor_correlation_invalid_diagonal(self):
        """Test that non-unit diagonal raises error."""
        model = MultiFactorModel(factor_names=["a", "b"])
        corr = np.array([[0.9, 0.3], [0.3, 1.0]])  # Diagonal not 1
        with pytest.raises(ValueError, match="Diagonal"):
            model.set_factor_correlation(corr)

    def test_set_factor_correlation_out_of_range(self):
        """Test that correlations outside [-1, 1] raise error."""
        model = MultiFactorModel(factor_names=["a", "b"])
        corr = np.array([[1.0, 1.5], [1.5, 1.0]])  # Correlation > 1
        with pytest.raises(ValueError, match="between -1 and 1"):
            model.set_factor_correlation(corr)

    def test_set_default_correlation(self):
        """Test setting default correlation structure."""
        model = MultiFactorModel()
        model.set_default_correlation(
            inter_sector=0.3,
            intra_sector=0.5,
            global_correlation=0.2
        )
        corr = model._factor_correlation
        assert corr is not None
        # Check diagonal is 1
        assert np.allclose(np.diag(corr), 1.0)
        # Check symmetry
        assert np.allclose(corr, corr.T)

    def test_get_factor_index(self, configured_model):
        """Test getting factor index by name."""
        idx = configured_model.get_factor_index("global")
        assert idx == 0
        idx = configured_model.get_factor_index("technology")
        assert idx == 2

    def test_get_factor_index_invalid(self, configured_model):
        """Test that invalid factor name raises error."""
        with pytest.raises(KeyError, match="not found"):
            configured_model.get_factor_index("nonexistent")

    def test_get_loading_vector(self, configured_model, sample_obligor):
        """Test getting loading vector for an obligor."""
        loadings = configured_model.get_loading_vector(sample_obligor)
        assert len(loadings) == configured_model.num_factors
        assert loadings[0] == 0.3  # global
        assert loadings[2] == 0.4  # technology

    def test_calculate_default_threshold(self, configured_model):
        """Test default threshold calculation."""
        obligor = Obligor(name="Test", pd=0.02, lgd=0.45, ead=1_000_000)
        threshold = configured_model.calculate_default_threshold(obligor)
        expected = norm.ppf(0.02)
        assert np.isclose(threshold, expected)

    def test_calculate_default_threshold_edge_cases(self, configured_model):
        """Test default threshold for edge case PDs."""
        # PD = 0 -> threshold = -inf
        obligor_zero = Obligor(name="Zero", pd=0.0, lgd=0.45, ead=1_000_000)
        assert configured_model.calculate_default_threshold(obligor_zero) == -np.inf

        # PD = 1 -> threshold = +inf
        obligor_one = Obligor(name="One", pd=1.0, lgd=0.45, ead=1_000_000)
        assert configured_model.calculate_default_threshold(obligor_one) == np.inf

    def test_calculate_asset_correlation(self, configured_model):
        """Test asset correlation calculation between two obligors."""
        obligor1 = Obligor(
            name="O1", pd=0.02, lgd=0.45, ead=1_000_000,
            factor_loadings={"global": 0.3, "technology": 0.4}
        )
        obligor2 = Obligor(
            name="O2", pd=0.02, lgd=0.45, ead=1_000_000,
            factor_loadings={"global": 0.3, "technology": 0.4}
        )
        # Same loadings -> correlation should be sum of squared loadings (if factors independent)
        corr = configured_model.calculate_asset_correlation(obligor1, obligor2)
        assert 0 <= corr <= 1

    def test_calculate_asset_correlation_same_sector_higher(self, configured_model):
        """Test that same-sector obligors have higher correlation."""
        tech1 = Obligor(
            name="Tech1", pd=0.02, lgd=0.45, ead=1_000_000,
            factor_loadings={"global": 0.3, "technology": 0.5}
        )
        tech2 = Obligor(
            name="Tech2", pd=0.02, lgd=0.45, ead=1_000_000,
            factor_loadings={"global": 0.3, "technology": 0.5}
        )
        bank = Obligor(
            name="Bank", pd=0.02, lgd=0.45, ead=1_000_000,
            factor_loadings={"global": 0.3, "financials": 0.5}
        )

        corr_same = configured_model.calculate_asset_correlation(tech1, tech2)
        corr_diff = configured_model.calculate_asset_correlation(tech1, bank)

        assert corr_same > corr_diff

    def test_generate_factor_scenarios(self, configured_model):
        """Test generating factor scenarios."""
        scenarios = configured_model.generate_factor_scenarios(1000, random_state=42)
        assert scenarios.shape == (1000, configured_model.num_factors)
        # Check approximate standard normal properties
        assert np.abs(np.mean(scenarios)) < 0.1
        assert np.abs(np.std(scenarios) - 1.0) < 0.1

    def test_generate_factor_scenarios_reproducible(self, configured_model):
        """Test that same random state gives same results."""
        scenarios1 = configured_model.generate_factor_scenarios(100, random_state=42)
        scenarios2 = configured_model.generate_factor_scenarios(100, random_state=42)
        assert np.allclose(scenarios1, scenarios2)

    def test_simulate_asset_returns(self, configured_model, multi_obligor_portfolio):
        """Test simulating asset returns."""
        factor_scenarios = configured_model.generate_factor_scenarios(1000, random_state=42)
        returns = configured_model.simulate_asset_returns(
            multi_obligor_portfolio, factor_scenarios, random_state=42
        )
        assert returns.shape == (1000, 3)  # 3 obligors
        # Returns should be approximately standard normal
        assert np.abs(np.mean(returns)) < 0.1
        assert np.abs(np.std(returns) - 1.0) < 0.2

    def test_simulate_asset_returns_reproducible(self, configured_model, multi_obligor_portfolio):
        """Test that same random states give same results."""
        factor_scenarios = configured_model.generate_factor_scenarios(100, random_state=42)
        returns1 = configured_model.simulate_asset_returns(
            multi_obligor_portfolio, factor_scenarios, random_state=123
        )
        returns2 = configured_model.simulate_asset_returns(
            multi_obligor_portfolio, factor_scenarios, random_state=123
        )
        assert np.allclose(returns1, returns2)

    def test_get_portfolio_correlation_matrix(self, configured_model, multi_obligor_portfolio):
        """Test getting portfolio correlation matrix."""
        corr = configured_model.get_portfolio_correlation_matrix(multi_obligor_portfolio)
        assert corr.shape == (3, 3)
        # Diagonal should be 1
        assert np.allclose(np.diag(corr), 1.0)
        # Should be symmetric
        assert np.allclose(corr, corr.T)
        # All correlations should be in valid range
        assert np.all(corr >= -1) and np.all(corr <= 1)

    def test_factor_correlation_not_set_error(self):
        """Test that operations fail if correlation not set."""
        model = MultiFactorModel()
        with pytest.raises(ValueError, match="not set"):
            model.generate_factor_scenarios(100)

        obligor1 = Obligor(name="O1", pd=0.02, lgd=0.45, ead=1_000_000)
        obligor2 = Obligor(name="O2", pd=0.02, lgd=0.45, ead=1_000_000)
        with pytest.raises(ValueError, match="not set"):
            model.calculate_asset_correlation(obligor1, obligor2)
