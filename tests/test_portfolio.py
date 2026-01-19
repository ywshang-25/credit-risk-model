"""Tests for portfolio.py - Obligor and Portfolio classes."""

import pytest
import numpy as np

from credit_risk import Obligor, Portfolio, BetaLGD, EmpiricalLGD


class TestObligor:
    """Tests for the Obligor dataclass."""

    def test_create_obligor_basic(self):
        """Test creating a basic obligor."""
        obligor = Obligor(
            name="Test",
            pd=0.02,
            lgd=0.45,
            ead=1_000_000
        )
        assert obligor.name == "Test"
        assert obligor.pd == 0.02
        assert obligor.lgd == 0.45
        assert obligor.ead == 1_000_000

    def test_create_obligor_with_all_fields(self):
        """Test creating an obligor with all optional fields."""
        obligor = Obligor(
            name="FullObligor",
            pd=0.03,
            lgd=0.50,
            ead=5_000_000,
            factor_loadings={"global": 0.3, "technology": 0.4},
            sector="technology",
            country="us",
            rating="BB"
        )
        assert obligor.sector == "technology"
        assert obligor.country == "us"
        assert obligor.rating == "BB"
        assert obligor.factor_loadings["global"] == 0.3

    def test_obligor_pd_validation(self):
        """Test that PD must be between 0 and 1."""
        with pytest.raises(ValueError, match="PD must be between 0 and 1"):
            Obligor(name="Test", pd=1.5, lgd=0.45, ead=1_000_000)

        with pytest.raises(ValueError, match="PD must be between 0 and 1"):
            Obligor(name="Test", pd=-0.1, lgd=0.45, ead=1_000_000)

    def test_obligor_lgd_validation(self):
        """Test that LGD must be between 0 and 1."""
        with pytest.raises(ValueError, match="LGD must be between 0 and 1"):
            Obligor(name="Test", pd=0.02, lgd=1.5, ead=1_000_000)

    def test_obligor_ead_validation(self):
        """Test that EAD must be non-negative."""
        with pytest.raises(ValueError, match="EAD must be non-negative"):
            Obligor(name="Test", pd=0.02, lgd=0.45, ead=-1_000_000)

    def test_obligor_factor_loading_validation(self):
        """Test that sum of squared factor loadings must be <= 1."""
        with pytest.raises(ValueError, match="Sum of squared factor loadings"):
            Obligor(
                name="Test",
                pd=0.02,
                lgd=0.45,
                ead=1_000_000,
                factor_loadings={"global": 0.8, "tech": 0.8}  # 0.64 + 0.64 > 1
            )

    def test_expected_loss_calculation(self, sample_obligor):
        """Test expected loss calculation."""
        expected = sample_obligor.pd * sample_obligor.lgd * sample_obligor.ead
        assert sample_obligor.expected_loss == expected

    def test_idiosyncratic_weight(self, sample_obligor):
        """Test idiosyncratic weight calculation."""
        loadings = sample_obligor.factor_loadings
        total_sq = sum(b**2 for b in loadings.values())
        expected = np.sqrt(1 - total_sq)
        assert np.isclose(sample_obligor.idiosyncratic_weight, expected)

    def test_has_stochastic_lgd_false(self, sample_obligor):
        """Test has_stochastic_lgd returns False for constant LGD."""
        assert sample_obligor.has_stochastic_lgd is False

    def test_has_stochastic_lgd_true(self, sample_obligor_with_beta_lgd):
        """Test has_stochastic_lgd returns True for stochastic LGD."""
        assert sample_obligor_with_beta_lgd.has_stochastic_lgd is True

    def test_get_lgd_mean_constant(self, sample_obligor):
        """Test get_lgd_mean returns lgd for constant case."""
        assert sample_obligor.get_lgd_mean() == sample_obligor.lgd

    def test_get_lgd_mean_stochastic(self, sample_obligor_with_beta_lgd):
        """Test get_lgd_mean returns distribution mean for stochastic case."""
        assert sample_obligor_with_beta_lgd.get_lgd_mean() == 0.45

    def test_sample_lgd_constant(self, sample_obligor):
        """Test sample_lgd returns constant values."""
        samples = sample_obligor.sample_lgd(100)
        assert len(samples) == 100
        assert np.all(samples == sample_obligor.lgd)

    def test_sample_lgd_stochastic(self, sample_obligor_with_beta_lgd):
        """Test sample_lgd returns variable values for stochastic LGD."""
        samples = sample_obligor_with_beta_lgd.sample_lgd(1000, random_state=42)
        assert len(samples) == 1000
        assert np.std(samples) > 0  # Should have variation
        assert np.all(samples >= 0) and np.all(samples <= 1)


class TestPortfolio:
    """Tests for the Portfolio class."""

    def test_create_empty_portfolio(self):
        """Test creating an empty portfolio."""
        portfolio = Portfolio(name="Empty")
        assert portfolio.name == "Empty"
        assert len(portfolio) == 0

    def test_add_obligor(self, sample_obligor):
        """Test adding an obligor to portfolio."""
        portfolio = Portfolio()
        portfolio.add_obligor(sample_obligor)
        assert len(portfolio) == 1
        assert sample_obligor.name in portfolio

    def test_add_duplicate_obligor_raises(self, sample_obligor):
        """Test that adding duplicate obligor raises error."""
        portfolio = Portfolio()
        portfolio.add_obligor(sample_obligor)
        with pytest.raises(ValueError, match="already exists"):
            portfolio.add_obligor(sample_obligor)

    def test_remove_obligor(self, sample_portfolio, sample_obligor):
        """Test removing an obligor from portfolio."""
        removed = sample_portfolio.remove_obligor(sample_obligor.name)
        assert removed.name == sample_obligor.name
        assert len(sample_portfolio) == 0

    def test_remove_nonexistent_obligor_raises(self, sample_portfolio):
        """Test that removing nonexistent obligor raises error."""
        with pytest.raises(KeyError, match="not found"):
            sample_portfolio.remove_obligor("NonExistent")

    def test_get_obligor(self, sample_portfolio, sample_obligor):
        """Test getting an obligor by name."""
        retrieved = sample_portfolio.get_obligor(sample_obligor.name)
        assert retrieved.name == sample_obligor.name

    def test_total_ead(self, multi_obligor_portfolio):
        """Test total EAD calculation."""
        expected = 10_000_000 + 15_000_000 + 20_000_000
        assert multi_obligor_portfolio.total_ead == expected

    def test_total_expected_loss(self, multi_obligor_portfolio):
        """Test total expected loss calculation."""
        total_el = sum(o.expected_loss for o in multi_obligor_portfolio.obligors)
        assert multi_obligor_portfolio.total_expected_loss == total_el

    def test_get_sectors(self, multi_obligor_portfolio):
        """Test getting unique sectors."""
        sectors = multi_obligor_portfolio.get_sectors()
        assert sectors == {"technology", "financials", "energy"}

    def test_get_countries(self, multi_obligor_portfolio):
        """Test getting unique countries."""
        countries = multi_obligor_portfolio.get_countries()
        assert countries == {"us", "europe"}

    def test_get_ratings(self, multi_obligor_portfolio):
        """Test getting unique ratings."""
        ratings = multi_obligor_portfolio.get_ratings()
        assert ratings == {"BBB", "A", "BB"}

    def test_filter_by_sector(self, multi_obligor_portfolio):
        """Test filtering by sector."""
        tech_obligors = multi_obligor_portfolio.filter_by_sector("technology")
        assert len(tech_obligors) == 1
        assert tech_obligors[0].name == "Tech_A"

    def test_filter_by_country(self, multi_obligor_portfolio):
        """Test filtering by country."""
        us_obligors = multi_obligor_portfolio.filter_by_country("us")
        assert len(us_obligors) == 2

    def test_filter_by_rating(self, multi_obligor_portfolio):
        """Test filtering by rating."""
        a_rated = multi_obligor_portfolio.filter_by_rating("A")
        assert len(a_rated) == 1
        assert a_rated[0].name == "Bank_A"

    def test_portfolio_copy(self, multi_obligor_portfolio):
        """Test portfolio deep copy."""
        copy = multi_obligor_portfolio.copy()
        assert len(copy) == len(multi_obligor_portfolio)
        assert copy.name == f"{multi_obligor_portfolio.name}_copy"

        # Modify original, copy should be unchanged
        multi_obligor_portfolio.remove_obligor("Tech_A")
        assert len(copy) == 3

    def test_portfolio_iteration(self, multi_obligor_portfolio):
        """Test iterating over portfolio."""
        names = [o.name for o in multi_obligor_portfolio]
        assert len(names) == 3

    def test_get_factor_names(self, multi_obligor_portfolio):
        """Test getting all factor names."""
        factors = multi_obligor_portfolio.get_factor_names()
        assert "global" in factors
        assert "technology" in factors
        assert "financials" in factors

    def test_obligor_names_property(self, multi_obligor_portfolio):
        """Test obligor_names property."""
        names = multi_obligor_portfolio.obligor_names
        assert len(names) == 3
        assert "Tech_A" in names

    def test_portfolio_copy_preserves_lgd_distribution(self):
        """Test that portfolio copy preserves LGD distribution."""
        portfolio = Portfolio()
        obligor = Obligor(
            name="Test",
            pd=0.02,
            lgd=0.45,
            ead=1_000_000,
            lgd_distribution=BetaLGD(mean=0.45, std=0.1)
        )
        portfolio.add_obligor(obligor)

        copy = portfolio.copy()
        copied_obligor = copy.get_obligor("Test")
        assert copied_obligor.has_stochastic_lgd
        assert copied_obligor.lgd_distribution is obligor.lgd_distribution
