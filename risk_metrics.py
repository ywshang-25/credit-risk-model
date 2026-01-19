"""Risk metrics and incremental risk contribution calculations."""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .portfolio import Portfolio, Obligor
from .model import MultiFactorModel
from .simulation import MonteCarloEngine, SimulationResult


@dataclass
class IncrementalRiskResult:
    """Results from incremental risk contribution analysis.

    Attributes:
        obligor_name: Name of the obligor
        irc_expected_loss: Incremental contribution to expected loss
        irc_var: Incremental contribution to VaR
        irc_es: Incremental contribution to Expected Shortfall
        standalone_el: Standalone expected loss of the obligor
        standalone_var: Standalone VaR of the obligor
        marginal_pd: Marginal probability of default contribution
    """
    obligor_name: str
    irc_expected_loss: float
    irc_var: float
    irc_es: float
    standalone_el: float
    standalone_var: float
    marginal_pd: float


@dataclass
class RiskDecomposition:
    """Risk decomposition by factor or category.

    Attributes:
        category: Name of the category (sector, country, rating, factor)
        total_exposure: Total EAD in this category
        expected_loss: Expected loss contribution
        var_contribution: VaR contribution
        es_contribution: ES contribution
        obligor_count: Number of obligors in category
    """
    category: str
    total_exposure: float
    expected_loss: float
    var_contribution: float
    es_contribution: float
    obligor_count: int


class RiskCalculator:
    """Calculate risk metrics and incremental risk contributions."""

    def __init__(self, model: MultiFactorModel, random_state: Optional[int] = None):
        """Initialize risk calculator.

        Args:
            model: The multi-factor model
            random_state: Random seed for reproducibility
        """
        self.model = model
        self.random_state = random_state
        self._engine = MonteCarloEngine(model, random_state)

    def calculate_incremental_loss(self, portfolio: Portfolio, obligor_name: str,
                                   num_scenarios: int = 10000,
                                   confidence: float = 0.99) -> IncrementalRiskResult:
        """Calculate incremental risk contribution for a single obligor.

        IRC_i = E[Loss | obligor i in portfolio] - E[Loss | obligor i removed]

        Args:
            portfolio: The full portfolio
            obligor_name: Name of the obligor to analyze
            num_scenarios: Number of Monte Carlo scenarios
            confidence: Confidence level for VaR/ES

        Returns:
            IncrementalRiskResult with all IRC metrics
        """
        rng = np.random.default_rng(self.random_state)
        seed1 = rng.integers(0, 2**31)
        seed2 = rng.integers(0, 2**31)

        factor_scenarios = self.model.generate_factor_scenarios(
            num_scenarios, random_state=seed1
        )

        engine_full = MonteCarloEngine(self.model, random_state=seed2)
        full_result = engine_full._simulate_batch(portfolio, num_scenarios)

        full_result = self._simulate_with_factors(
            portfolio, factor_scenarios, seed2
        )

        reduced_result = self._simulate_with_factors(
            portfolio, factor_scenarios, seed2, exclude_obligor=obligor_name
        )

        irc_el = full_result.expected_loss - reduced_result.expected_loss
        irc_var = full_result.get_var(confidence) - reduced_result.get_var(confidence)
        irc_es = full_result.get_expected_shortfall(confidence) - reduced_result.get_expected_shortfall(confidence)

        obligor = portfolio.get_obligor(obligor_name)
        standalone_el = obligor.expected_loss

        obligor_idx = portfolio.obligor_names.index(obligor_name)
        standalone_var = np.percentile(
            full_result.obligor_losses[:, obligor_idx], confidence * 100
        )

        marginal_pd = full_result.get_obligor_default_rate(obligor_idx) - obligor.pd

        return IncrementalRiskResult(
            obligor_name=obligor_name,
            irc_expected_loss=irc_el,
            irc_var=irc_var,
            irc_es=irc_es,
            standalone_el=standalone_el,
            standalone_var=standalone_var,
            marginal_pd=marginal_pd
        )

    def _simulate_with_factors(self, portfolio: Portfolio,
                               factor_scenarios: np.ndarray,
                               idio_seed: int,
                               exclude_obligor: Optional[str] = None) -> SimulationResult:
        """Simulate using pre-generated factor scenarios.

        Args:
            portfolio: The portfolio to simulate
            factor_scenarios: Pre-generated factor scenarios
            idio_seed: Seed for idiosyncratic shocks
            exclude_obligor: Optional obligor to exclude

        Returns:
            SimulationResult
        """
        if exclude_obligor:
            sim_portfolio = portfolio.copy()
            sim_portfolio.remove_obligor(exclude_obligor)
        else:
            sim_portfolio = portfolio

        num_scenarios = factor_scenarios.shape[0]

        asset_returns = self.model.simulate_asset_returns(
            sim_portfolio, factor_scenarios, random_state=idio_seed
        )

        obligors = sim_portfolio.obligors
        default_thresholds = np.array([
            self.model.calculate_default_threshold(o) for o in obligors
        ])

        default_indicators = asset_returns < default_thresholds

        lgd_array = np.array([o.lgd for o in obligors])
        ead_array = np.array([o.ead for o in obligors])

        obligor_losses = default_indicators * lgd_array * ead_array
        scenario_losses = np.sum(obligor_losses, axis=1)

        return SimulationResult(
            scenario_losses=scenario_losses,
            default_indicators=default_indicators,
            obligor_losses=obligor_losses,
            factor_scenarios=factor_scenarios,
            num_scenarios=num_scenarios,
            num_defaults_per_scenario=np.sum(default_indicators, axis=1)
        )

    def calculate_all_incremental_losses(self, portfolio: Portfolio,
                                         num_scenarios: int = 10000,
                                         confidence: float = 0.99) -> List[IncrementalRiskResult]:
        """Calculate IRC for all obligors in the portfolio.

        Args:
            portfolio: The portfolio to analyze
            num_scenarios: Number of Monte Carlo scenarios
            confidence: Confidence level for VaR/ES

        Returns:
            List of IncrementalRiskResult for each obligor
        """
        results = []
        rng = np.random.default_rng(self.random_state)

        seed1 = rng.integers(0, 2**31)
        seed2 = rng.integers(0, 2**31)

        factor_scenarios = self.model.generate_factor_scenarios(
            num_scenarios, random_state=seed1
        )

        full_result = self._simulate_with_factors(portfolio, factor_scenarios, seed2)

        for obligor_name in portfolio.obligor_names:
            reduced_result = self._simulate_with_factors(
                portfolio, factor_scenarios, seed2, exclude_obligor=obligor_name
            )

            irc_el = full_result.expected_loss - reduced_result.expected_loss
            irc_var = full_result.get_var(confidence) - reduced_result.get_var(confidence)
            irc_es = (full_result.get_expected_shortfall(confidence) -
                      reduced_result.get_expected_shortfall(confidence))

            obligor = portfolio.get_obligor(obligor_name)
            obligor_idx = portfolio.obligor_names.index(obligor_name)

            standalone_var = np.percentile(
                full_result.obligor_losses[:, obligor_idx], confidence * 100
            )

            marginal_pd = full_result.get_obligor_default_rate(obligor_idx) - obligor.pd

            results.append(IncrementalRiskResult(
                obligor_name=obligor_name,
                irc_expected_loss=irc_el,
                irc_var=irc_var,
                irc_es=irc_es,
                standalone_el=obligor.expected_loss,
                standalone_var=standalone_var,
                marginal_pd=marginal_pd
            ))

        return results

    def risk_decomposition_by_sector(self, portfolio: Portfolio,
                                     num_scenarios: int = 10000,
                                     confidence: float = 0.99) -> List[RiskDecomposition]:
        """Decompose portfolio risk by sector.

        Args:
            portfolio: The portfolio to analyze
            num_scenarios: Number of Monte Carlo scenarios
            confidence: Confidence level

        Returns:
            List of RiskDecomposition by sector
        """
        return self._risk_decomposition(
            portfolio, 'sector', num_scenarios, confidence
        )

    def risk_decomposition_by_country(self, portfolio: Portfolio,
                                      num_scenarios: int = 10000,
                                      confidence: float = 0.99) -> List[RiskDecomposition]:
        """Decompose portfolio risk by country.

        Args:
            portfolio: The portfolio to analyze
            num_scenarios: Number of Monte Carlo scenarios
            confidence: Confidence level

        Returns:
            List of RiskDecomposition by country
        """
        return self._risk_decomposition(
            portfolio, 'country', num_scenarios, confidence
        )

    def risk_decomposition_by_rating(self, portfolio: Portfolio,
                                     num_scenarios: int = 10000,
                                     confidence: float = 0.99) -> List[RiskDecomposition]:
        """Decompose portfolio risk by rating.

        Args:
            portfolio: The portfolio to analyze
            num_scenarios: Number of Monte Carlo scenarios
            confidence: Confidence level

        Returns:
            List of RiskDecomposition by rating
        """
        return self._risk_decomposition(
            portfolio, 'rating', num_scenarios, confidence
        )

    def _risk_decomposition(self, portfolio: Portfolio, category_type: str,
                            num_scenarios: int,
                            confidence: float) -> List[RiskDecomposition]:
        """Internal method for risk decomposition by any category.

        Args:
            portfolio: The portfolio
            category_type: 'sector', 'country', or 'rating'
            num_scenarios: Number of scenarios
            confidence: Confidence level

        Returns:
            List of RiskDecomposition
        """
        rng = np.random.default_rng(self.random_state)
        seed1 = rng.integers(0, 2**31)
        seed2 = rng.integers(0, 2**31)

        factor_scenarios = self.model.generate_factor_scenarios(
            num_scenarios, random_state=seed1
        )

        full_result = self._simulate_with_factors(portfolio, factor_scenarios, seed2)

        if category_type == 'sector':
            categories = portfolio.get_sectors()
            get_category = lambda o: o.sector
        elif category_type == 'country':
            categories = portfolio.get_countries()
            get_category = lambda o: o.country
        elif category_type == 'rating':
            categories = portfolio.get_ratings()
            get_category = lambda o: o.rating
        else:
            raise ValueError(f"Unknown category type: {category_type}")

        results = []
        obligor_names = portfolio.obligor_names
        obligors = portfolio.obligors

        for category in categories:
            if category is None:
                continue

            category_mask = np.array([
                get_category(o) == category for o in obligors
            ])

            category_losses = np.sum(
                full_result.obligor_losses[:, category_mask], axis=1
            )

            total_exposure = sum(
                o.ead for o in obligors if get_category(o) == category
            )

            expected_loss = np.mean(category_losses)
            var_contribution = np.percentile(category_losses, confidence * 100)

            tail_losses = category_losses[
                full_result.scenario_losses >= full_result.get_var(confidence)
            ]
            es_contribution = np.mean(tail_losses) if len(tail_losses) > 0 else var_contribution

            obligor_count = sum(1 for o in obligors if get_category(o) == category)

            results.append(RiskDecomposition(
                category=category,
                total_exposure=total_exposure,
                expected_loss=expected_loss,
                var_contribution=var_contribution,
                es_contribution=es_contribution,
                obligor_count=obligor_count
            ))

        return results

    def calculate_portfolio_metrics(self, portfolio: Portfolio,
                                    num_scenarios: int = 10000,
                                    confidence: float = 0.99) -> Dict:
        """Calculate comprehensive portfolio risk metrics.

        Args:
            portfolio: The portfolio to analyze
            num_scenarios: Number of Monte Carlo scenarios
            confidence: Confidence level

        Returns:
            Dictionary with all portfolio metrics
        """
        result = self._engine.simulate(portfolio, num_scenarios)

        return {
            'total_ead': portfolio.total_ead,
            'expected_loss': result.expected_loss,
            'expected_loss_rate': result.expected_loss / portfolio.total_ead,
            'loss_volatility': result.loss_std,
            'var': result.get_var(confidence),
            'var_rate': result.get_var(confidence) / portfolio.total_ead,
            'expected_shortfall': result.get_expected_shortfall(confidence),
            'es_rate': result.get_expected_shortfall(confidence) / portfolio.total_ead,
            'unexpected_loss': result.get_var(confidence) - result.expected_loss,
            'average_default_rate': result.default_rate,
            'max_loss': np.max(result.scenario_losses),
            'num_scenarios': num_scenarios,
            'confidence_level': confidence
        }


def create_irc_report(irc_results: List[IncrementalRiskResult],
                      portfolio: Portfolio) -> pd.DataFrame:
    """Create a DataFrame report of IRC results.

    Args:
        irc_results: List of IncrementalRiskResult
        portfolio: The portfolio (for additional obligor info)

    Returns:
        DataFrame with IRC analysis
    """
    data = []
    for result in irc_results:
        obligor = portfolio.get_obligor(result.obligor_name)
        data.append({
            'Obligor': result.obligor_name,
            'Sector': obligor.sector,
            'Country': obligor.country,
            'Rating': obligor.rating,
            'EAD': obligor.ead,
            'PD': obligor.pd,
            'LGD': obligor.lgd,
            'Standalone_EL': result.standalone_el,
            'IRC_EL': result.irc_expected_loss,
            'IRC_VaR': result.irc_var,
            'IRC_ES': result.irc_es,
            'Diversification_Benefit': result.standalone_el - result.irc_expected_loss
        })

    df = pd.DataFrame(data)
    df = df.sort_values('IRC_ES', ascending=False)
    return df


def create_decomposition_report(decompositions: List[RiskDecomposition],
                                category_name: str) -> pd.DataFrame:
    """Create a DataFrame report of risk decomposition.

    Args:
        decompositions: List of RiskDecomposition
        category_name: Name of the category (for column naming)

    Returns:
        DataFrame with decomposition analysis
    """
    data = []
    for decomp in decompositions:
        data.append({
            category_name: decomp.category,
            'Obligor_Count': decomp.obligor_count,
            'Total_Exposure': decomp.total_exposure,
            'Expected_Loss': decomp.expected_loss,
            'VaR_Contribution': decomp.var_contribution,
            'ES_Contribution': decomp.es_contribution,
            'EL_Rate': decomp.expected_loss / decomp.total_exposure if decomp.total_exposure > 0 else 0
        })

    df = pd.DataFrame(data)
    df = df.sort_values('ES_Contribution', ascending=False)
    return df
