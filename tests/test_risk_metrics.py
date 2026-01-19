"""Tests for risk_metrics.py - Risk calculations and IRC."""

import pytest
import numpy as np
import pandas as pd

from credit_risk import (
    RiskCalculator,
    IncrementalRiskResult,
    RiskDecomposition,
    create_irc_report,
    create_decomposition_report,
    MultiFactorModel,
    Portfolio,
    Obligor,
    BetaLGD
)


class TestRiskCalculator:
    """Tests for RiskCalculator class."""

    def test_create_risk_calculator(self, configured_model):
        """Test creating a risk calculator."""
        calc = RiskCalculator(configured_model, random_state=42)
        assert calc.model is configured_model

    def test_calculate_portfolio_metrics(self, configured_model, multi_obligor_portfolio):
        """Test calculating portfolio metrics."""
        calc = RiskCalculator(configured_model, random_state=42)
        metrics = calc.calculate_portfolio_metrics(
            multi_obligor_portfolio, num_scenarios=10000, confidence=0.99
        )

        assert 'total_ead' in metrics
        assert 'expected_loss' in metrics
        assert 'var' in metrics
        assert 'expected_shortfall' in metrics
        assert 'unexpected_loss' in metrics
        assert 'num_scenarios' in metrics

        assert metrics['total_ead'] == multi_obligor_portfolio.total_ead
        assert metrics['num_scenarios'] == 10000
        assert metrics['confidence_level'] == 0.99

    def test_portfolio_metrics_values_sensible(self, configured_model, multi_obligor_portfolio):
        """Test that portfolio metrics have sensible values."""
        calc = RiskCalculator(configured_model, random_state=42)
        metrics = calc.calculate_portfolio_metrics(
            multi_obligor_portfolio, num_scenarios=10000
        )

        # Expected loss should be positive
        assert metrics['expected_loss'] > 0
        # VaR should be greater than expected loss
        assert metrics['var'] > metrics['expected_loss']
        # ES should be greater than VaR
        assert metrics['expected_shortfall'] >= metrics['var']
        # Unexpected loss should be positive
        assert metrics['unexpected_loss'] > 0

    def test_calculate_incremental_loss(self, configured_model, multi_obligor_portfolio):
        """Test calculating IRC for a single obligor."""
        calc = RiskCalculator(configured_model, random_state=42)
        result = calc.calculate_incremental_loss(
            multi_obligor_portfolio,
            obligor_name="Tech_A",
            num_scenarios=10000,
            confidence=0.99
        )

        assert isinstance(result, IncrementalRiskResult)
        assert result.obligor_name == "Tech_A"
        assert result.standalone_el > 0

    def test_calculate_all_incremental_losses(self, configured_model, multi_obligor_portfolio):
        """Test calculating IRC for all obligors."""
        calc = RiskCalculator(configured_model, random_state=42)
        results = calc.calculate_all_incremental_losses(
            multi_obligor_portfolio, num_scenarios=10000
        )

        assert len(results) == len(multi_obligor_portfolio)
        for result in results:
            assert isinstance(result, IncrementalRiskResult)

    def test_irc_higher_for_riskier_obligors(self, configured_model):
        """Test that riskier obligors have higher IRC."""
        portfolio = Portfolio()
        portfolio.add_obligor(Obligor(
            name="Safe",
            pd=0.01,
            lgd=0.30,
            ead=10_000_000,
            factor_loadings={"global": 0.3}
        ))
        portfolio.add_obligor(Obligor(
            name="Risky",
            pd=0.10,
            lgd=0.70,
            ead=10_000_000,
            factor_loadings={"global": 0.3}
        ))

        calc = RiskCalculator(configured_model, random_state=42)
        results = calc.calculate_all_incremental_losses(portfolio, num_scenarios=20000)

        safe_irc = next(r for r in results if r.obligor_name == "Safe")
        risky_irc = next(r for r in results if r.obligor_name == "Risky")

        assert risky_irc.irc_expected_loss > safe_irc.irc_expected_loss

    def test_risk_decomposition_by_sector(self, configured_model, multi_obligor_portfolio):
        """Test risk decomposition by sector."""
        calc = RiskCalculator(configured_model, random_state=42)
        decomp = calc.risk_decomposition_by_sector(
            multi_obligor_portfolio, num_scenarios=10000
        )

        assert len(decomp) == 3  # technology, financials, energy
        sectors = {d.category for d in decomp}
        assert sectors == {"technology", "financials", "energy"}

        for d in decomp:
            assert isinstance(d, RiskDecomposition)
            assert d.total_exposure > 0
            assert d.obligor_count >= 1

    def test_risk_decomposition_by_country(self, configured_model, multi_obligor_portfolio):
        """Test risk decomposition by country."""
        calc = RiskCalculator(configured_model, random_state=42)
        decomp = calc.risk_decomposition_by_country(
            multi_obligor_portfolio, num_scenarios=10000
        )

        assert len(decomp) == 2  # us, europe
        countries = {d.category for d in decomp}
        assert countries == {"us", "europe"}

    def test_risk_decomposition_by_rating(self, configured_model, multi_obligor_portfolio):
        """Test risk decomposition by rating."""
        calc = RiskCalculator(configured_model, random_state=42)
        decomp = calc.risk_decomposition_by_rating(
            multi_obligor_portfolio, num_scenarios=10000
        )

        ratings = {d.category for d in decomp}
        assert ratings == {"BBB", "A", "BB"}

    def test_decomposition_exposure_sums_correctly(self, configured_model, multi_obligor_portfolio):
        """Test that decomposition exposures sum to total."""
        calc = RiskCalculator(configured_model, random_state=42)
        decomp = calc.risk_decomposition_by_sector(
            multi_obligor_portfolio, num_scenarios=10000
        )

        total_exposure = sum(d.total_exposure for d in decomp)
        assert total_exposure == pytest.approx(multi_obligor_portfolio.total_ead, rel=0.01)


class TestIncrementalRiskResult:
    """Tests for IncrementalRiskResult dataclass."""

    def test_incremental_risk_result_fields(self):
        """Test that all fields are accessible."""
        result = IncrementalRiskResult(
            obligor_name="Test",
            irc_expected_loss=1000,
            irc_var=5000,
            irc_es=7000,
            standalone_el=1200,
            standalone_var=4000,
            marginal_pd=0.001
        )

        assert result.obligor_name == "Test"
        assert result.irc_expected_loss == 1000
        assert result.irc_var == 5000
        assert result.irc_es == 7000
        assert result.standalone_el == 1200
        assert result.standalone_var == 4000
        assert result.marginal_pd == 0.001


class TestRiskDecomposition:
    """Tests for RiskDecomposition dataclass."""

    def test_risk_decomposition_fields(self):
        """Test that all fields are accessible."""
        decomp = RiskDecomposition(
            category="technology",
            total_exposure=10_000_000,
            expected_loss=100_000,
            var_contribution=500_000,
            es_contribution=700_000,
            obligor_count=5
        )

        assert decomp.category == "technology"
        assert decomp.total_exposure == 10_000_000
        assert decomp.expected_loss == 100_000
        assert decomp.obligor_count == 5


class TestReportFunctions:
    """Tests for report generation functions."""

    def test_create_irc_report(self, configured_model, multi_obligor_portfolio):
        """Test creating IRC report DataFrame."""
        calc = RiskCalculator(configured_model, random_state=42)
        results = calc.calculate_all_incremental_losses(
            multi_obligor_portfolio, num_scenarios=10000
        )

        df = create_irc_report(results, multi_obligor_portfolio)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'Obligor' in df.columns
        assert 'IRC_EL' in df.columns
        assert 'IRC_VaR' in df.columns
        assert 'Sector' in df.columns

    def test_create_irc_report_sorted(self, configured_model, multi_obligor_portfolio):
        """Test that IRC report is sorted by ES contribution."""
        calc = RiskCalculator(configured_model, random_state=42)
        results = calc.calculate_all_incremental_losses(
            multi_obligor_portfolio, num_scenarios=10000
        )

        df = create_irc_report(results, multi_obligor_portfolio)

        # Should be sorted descending by IRC_ES
        es_values = df['IRC_ES'].values
        assert all(es_values[i] >= es_values[i+1] for i in range(len(es_values)-1))

    def test_create_decomposition_report(self, configured_model, multi_obligor_portfolio):
        """Test creating decomposition report DataFrame."""
        calc = RiskCalculator(configured_model, random_state=42)
        decomp = calc.risk_decomposition_by_sector(
            multi_obligor_portfolio, num_scenarios=10000
        )

        df = create_decomposition_report(decomp, 'Sector')

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'Sector' in df.columns
        assert 'Expected_Loss' in df.columns
        assert 'VaR_Contribution' in df.columns

    def test_create_decomposition_report_sorted(self, configured_model, multi_obligor_portfolio):
        """Test that decomposition report is sorted by ES contribution."""
        calc = RiskCalculator(configured_model, random_state=42)
        decomp = calc.risk_decomposition_by_sector(
            multi_obligor_portfolio, num_scenarios=10000
        )

        df = create_decomposition_report(decomp, 'Sector')

        # Should be sorted descending by ES_Contribution
        es_values = df['ES_Contribution'].values
        assert all(es_values[i] >= es_values[i+1] for i in range(len(es_values)-1))


class TestRiskCalculatorWithStochasticLGD:
    """Tests for risk calculations with stochastic LGD."""

    def test_irc_with_stochastic_lgd(self, configured_model):
        """Test IRC calculation with stochastic LGD."""
        portfolio = Portfolio()
        portfolio.add_obligor(Obligor(
            name="BetaLGD",
            pd=0.03,
            lgd=0.50,
            ead=10_000_000,
            factor_loadings={"global": 0.3},
            lgd_distribution=BetaLGD(mean=0.50, std=0.15, factor_sensitivity=0.3)
        ))
        portfolio.add_obligor(Obligor(
            name="ConstantLGD",
            pd=0.03,
            lgd=0.50,
            ead=10_000_000,
            factor_loadings={"global": 0.3}
        ))

        calc = RiskCalculator(configured_model, random_state=42)
        results = calc.calculate_all_incremental_losses(portfolio, num_scenarios=20000)

        assert len(results) == 2
        # Both should have positive IRC
        for r in results:
            assert r.irc_expected_loss > 0

    def test_portfolio_metrics_with_stochastic_lgd(self, configured_model):
        """Test portfolio metrics with stochastic LGD."""
        portfolio = Portfolio()
        portfolio.add_obligor(Obligor(
            name="StochasticObligor",
            pd=0.05,
            lgd=0.50,
            ead=10_000_000,
            factor_loadings={"global": 0.4},
            lgd_distribution=BetaLGD(mean=0.50, std=0.15, factor_sensitivity=0.5)
        ))

        calc = RiskCalculator(configured_model, random_state=42)
        metrics = calc.calculate_portfolio_metrics(portfolio, num_scenarios=20000)

        assert metrics['expected_loss'] > 0
        assert metrics['var'] > metrics['expected_loss']
