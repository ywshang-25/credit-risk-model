#!/usr/bin/env python3
"""Example usage of the multi-factor credit risk model.

This script demonstrates:
1. Creating a multi-sector portfolio
2. Setting up the multi-factor model
3. Running Monte Carlo simulation
4. Calculating Incremental Risk Contributions (IRC)
5. Risk decomposition by sector, country, and rating
6. Stochastic LGD using Beta and Empirical distributions
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from credit_risk import (
    Obligor,
    Portfolio,
    MultiFactorModel,
    MonteCarloEngine,
    RiskCalculator,
    create_irc_report,
    create_decomposition_report,
    BetaLGD,
    EmpiricalLGD,
    create_lgd_distribution
)


def create_sample_portfolio() -> Portfolio:
    """Create a sample diversified credit portfolio."""
    portfolio = Portfolio(name="Sample Portfolio")

    obligors_data = [
        {"name": "TechCorp_A", "pd": 0.02, "lgd": 0.45, "ead": 10_000_000,
         "sector": "technology", "country": "us", "rating": "BBB",
         "loadings": {"global": 0.3, "technology": 0.4, "us": 0.2}},

        {"name": "TechCorp_B", "pd": 0.015, "lgd": 0.40, "ead": 8_000_000,
         "sector": "technology", "country": "us", "rating": "A",
         "loadings": {"global": 0.25, "technology": 0.45, "us": 0.15}},

        {"name": "TechCorp_EU", "pd": 0.025, "lgd": 0.50, "ead": 6_000_000,
         "sector": "technology", "country": "europe", "rating": "BB",
         "loadings": {"global": 0.3, "technology": 0.35, "europe": 0.25}},

        {"name": "Bank_US", "pd": 0.01, "lgd": 0.55, "ead": 15_000_000,
         "sector": "financials", "country": "us", "rating": "A",
         "loadings": {"global": 0.4, "financials": 0.35, "us": 0.2}},

        {"name": "Bank_EU", "pd": 0.012, "lgd": 0.50, "ead": 12_000_000,
         "sector": "financials", "country": "europe", "rating": "A",
         "loadings": {"global": 0.35, "financials": 0.40, "europe": 0.15}},

        {"name": "Insurance_Asia", "pd": 0.008, "lgd": 0.45, "ead": 9_000_000,
         "sector": "financials", "country": "asia", "rating": "AA",
         "loadings": {"global": 0.3, "financials": 0.30, "asia": 0.25}},

        {"name": "Energy_US", "pd": 0.035, "lgd": 0.60, "ead": 20_000_000,
         "sector": "energy", "country": "us", "rating": "BB",
         "loadings": {"global": 0.25, "energy": 0.50, "us": 0.15}},

        {"name": "Energy_EU", "pd": 0.03, "lgd": 0.55, "ead": 14_000_000,
         "sector": "energy", "country": "europe", "rating": "BBB",
         "loadings": {"global": 0.3, "energy": 0.45, "europe": 0.2}},

        {"name": "Healthcare_US", "pd": 0.018, "lgd": 0.40, "ead": 7_000_000,
         "sector": "healthcare", "country": "us", "rating": "BBB",
         "loadings": {"global": 0.2, "healthcare": 0.35, "us": 0.25}},

        {"name": "Healthcare_Asia", "pd": 0.022, "lgd": 0.45, "ead": 5_000_000,
         "sector": "healthcare", "country": "asia", "rating": "BB",
         "loadings": {"global": 0.25, "healthcare": 0.30, "asia": 0.30}},

        {"name": "Consumer_US", "pd": 0.025, "lgd": 0.50, "ead": 11_000_000,
         "sector": "consumer", "country": "us", "rating": "BB",
         "loadings": {"global": 0.35, "consumer": 0.35, "us": 0.20}},

        {"name": "Consumer_EU", "pd": 0.02, "lgd": 0.45, "ead": 8_500_000,
         "sector": "consumer", "country": "europe", "rating": "BBB",
         "loadings": {"global": 0.30, "consumer": 0.40, "europe": 0.15}},

        {"name": "Industrial_US", "pd": 0.028, "lgd": 0.55, "ead": 13_000_000,
         "sector": "industrials", "country": "us", "rating": "BB",
         "loadings": {"global": 0.30, "industrials": 0.40, "us": 0.20}},

        {"name": "Industrial_Asia", "pd": 0.032, "lgd": 0.50, "ead": 10_000_000,
         "sector": "industrials", "country": "asia", "rating": "B",
         "loadings": {"global": 0.25, "industrials": 0.45, "asia": 0.20}},

        {"name": "HighRisk_Startup", "pd": 0.08, "lgd": 0.70, "ead": 3_000_000,
         "sector": "technology", "country": "us", "rating": "CCC",
         "loadings": {"global": 0.20, "technology": 0.50, "us": 0.15}},
    ]

    for data in obligors_data:
        obligor = Obligor(
            name=data["name"],
            pd=data["pd"],
            lgd=data["lgd"],
            ead=data["ead"],
            factor_loadings=data["loadings"],
            sector=data["sector"],
            country=data["country"],
            rating=data["rating"]
        )
        portfolio.add_obligor(obligor)

    return portfolio


def main():
    """Run the example."""
    print("=" * 70)
    print("MULTI-FACTOR CREDIT RISK MODEL - EXAMPLE")
    print("=" * 70)

    np.random.seed(42)

    print("\n1. Creating sample portfolio...")
    portfolio = create_sample_portfolio()
    print(f"   Portfolio: {portfolio.name}")
    print(f"   Number of obligors: {len(portfolio)}")
    print(f"   Total EAD: ${portfolio.total_ead:,.0f}")
    print(f"   Total Expected Loss: ${portfolio.total_expected_loss:,.0f}")
    print(f"   Sectors: {portfolio.get_sectors()}")
    print(f"   Countries: {portfolio.get_countries()}")
    print(f"   Ratings: {portfolio.get_ratings()}")

    print("\n2. Setting up multi-factor model...")
    model = MultiFactorModel()
    model.set_default_correlation(
        inter_sector=0.3,
        intra_sector=0.5,
        global_correlation=0.2
    )
    print(f"   Factors: {model.factor_names}")
    print(f"   Number of factors: {model.num_factors}")

    print("\n3. Running Monte Carlo simulation...")
    num_scenarios = 50000
    engine = MonteCarloEngine(model, random_state=42)
    result = engine.simulate(portfolio, num_scenarios)
    print(f"   Scenarios simulated: {num_scenarios:,}")
    print(f"   Expected Loss: ${result.expected_loss:,.0f}")
    print(f"   Loss Std Dev: ${result.loss_std:,.0f}")
    print(f"   VaR (99%): ${result.get_var(0.99):,.0f}")
    print(f"   Expected Shortfall (99%): ${result.get_expected_shortfall(0.99):,.0f}")
    print(f"   Average Default Rate: {result.default_rate:.4f}")

    print("\n4. Calculating comprehensive portfolio metrics...")
    risk_calc = RiskCalculator(model, random_state=42)
    metrics = risk_calc.calculate_portfolio_metrics(portfolio, num_scenarios=50000)

    print(f"   Total EAD: ${metrics['total_ead']:,.0f}")
    print(f"   Expected Loss: ${metrics['expected_loss']:,.0f} ({metrics['expected_loss_rate']:.2%})")
    print(f"   Loss Volatility: ${metrics['loss_volatility']:,.0f}")
    print(f"   VaR (99%): ${metrics['var']:,.0f} ({metrics['var_rate']:.2%})")
    print(f"   Expected Shortfall: ${metrics['expected_shortfall']:,.0f} ({metrics['es_rate']:.2%})")
    print(f"   Unexpected Loss: ${metrics['unexpected_loss']:,.0f}")
    print(f"   Maximum Loss: ${metrics['max_loss']:,.0f}")

    print("\n5. Calculating Incremental Risk Contributions (IRC)...")
    irc_results = risk_calc.calculate_all_incremental_losses(
        portfolio, num_scenarios=50000, confidence=0.99
    )

    irc_df = create_irc_report(irc_results, portfolio)
    print("\n   IRC Results (sorted by ES contribution):")
    print("-" * 70)
    display_cols = ['Obligor', 'Sector', 'Rating', 'EAD', 'IRC_EL', 'IRC_VaR', 'IRC_ES']
    print(irc_df[display_cols].to_string(index=False))

    total_irc_el = irc_df['IRC_EL'].sum()
    total_standalone_el = irc_df['Standalone_EL'].sum()
    print(f"\n   Total IRC Expected Loss: ${total_irc_el:,.0f}")
    print(f"   Sum of Standalone EL: ${total_standalone_el:,.0f}")
    print(f"   Diversification Benefit: ${total_standalone_el - total_irc_el:,.0f}")

    print("\n6. Risk Decomposition by Sector...")
    sector_decomp = risk_calc.risk_decomposition_by_sector(
        portfolio, num_scenarios=50000
    )
    sector_df = create_decomposition_report(sector_decomp, 'Sector')
    print(sector_df.to_string(index=False))

    print("\n7. Risk Decomposition by Country...")
    country_decomp = risk_calc.risk_decomposition_by_country(
        portfolio, num_scenarios=50000
    )
    country_df = create_decomposition_report(country_decomp, 'Country')
    print(country_df.to_string(index=False))

    print("\n8. Risk Decomposition by Rating...")
    rating_decomp = risk_calc.risk_decomposition_by_rating(
        portfolio, num_scenarios=50000
    )
    rating_df = create_decomposition_report(rating_decomp, 'Rating')
    print(rating_df.to_string(index=False))

    print("\n9. Verification checks...")
    tech_a = portfolio.get_obligor("TechCorp_A")
    bank_us = portfolio.get_obligor("Bank_US")
    corr = model.calculate_asset_correlation(tech_a, bank_us)
    print(f"   Asset correlation (TechCorp_A vs Bank_US): {corr:.4f}")

    tech_b = portfolio.get_obligor("TechCorp_B")
    corr_same_sector = model.calculate_asset_correlation(tech_a, tech_b)
    print(f"   Asset correlation (TechCorp_A vs TechCorp_B - same sector): {corr_same_sector:.4f}")

    print("\n   Checking IRC properties:")
    high_pd_obligors = [r for r in irc_results if portfolio.get_obligor(r.obligor_name).pd > 0.03]
    low_pd_obligors = [r for r in irc_results if portfolio.get_obligor(r.obligor_name).pd < 0.015]

    avg_high_pd_irc = np.mean([r.irc_es for r in high_pd_obligors]) if high_pd_obligors else 0
    avg_low_pd_irc = np.mean([r.irc_es for r in low_pd_obligors]) if low_pd_obligors else 0
    print(f"   Avg IRC-ES for high PD obligors (>3%): ${avg_high_pd_irc:,.0f}")
    print(f"   Avg IRC-ES for low PD obligors (<1.5%): ${avg_low_pd_irc:,.0f}")

    if avg_high_pd_irc > avg_low_pd_irc:
        print("   [PASS] Higher PD obligors have higher average IRC")
    else:
        print("   [NOTE] IRC depends on multiple factors including EAD and correlations")

    print("\n" + "=" * 70)
    print("STOCHASTIC LGD DEMONSTRATION")
    print("=" * 70)

    print("\n10. Creating portfolio with stochastic LGD...")

    # Create sample historical LGD data for empirical distribution
    np.random.seed(123)
    historical_lgd_data = np.clip(np.random.beta(2, 3, size=100) * 0.8 + 0.1, 0.1, 0.9)

    # Create a portfolio with mixed LGD types
    stochastic_portfolio = Portfolio(name="Stochastic LGD Portfolio")

    # Obligor with Beta-distributed LGD
    stochastic_portfolio.add_obligor(Obligor(
        name="Bank_Beta_LGD",
        pd=0.02,
        lgd=0.45,  # Mean LGD (used for expected_loss calculation)
        ead=15_000_000,
        factor_loadings={"global": 0.35, "financials": 0.40, "us": 0.15},
        sector="financials",
        country="us",
        rating="A",
        lgd_distribution=BetaLGD(
            mean=0.45,
            std=0.15,
            factor_sensitivity=0.3  # LGD increases when economy is stressed
        )
    ))

    # Obligor with Empirical LGD from historical data
    stochastic_portfolio.add_obligor(Obligor(
        name="Energy_Empirical_LGD",
        pd=0.03,
        lgd=float(np.mean(historical_lgd_data)),
        ead=20_000_000,
        factor_loadings={"global": 0.30, "energy": 0.45, "us": 0.15},
        sector="energy",
        country="us",
        rating="BB",
        lgd_distribution=EmpiricalLGD(
            historical_lgd=historical_lgd_data,
            factor_sensitivity=0.2
        )
    ))

    # Obligor with constant LGD (traditional approach)
    stochastic_portfolio.add_obligor(Obligor(
        name="Tech_Constant_LGD",
        pd=0.015,
        lgd=0.40,
        ead=10_000_000,
        factor_loadings={"global": 0.25, "technology": 0.45, "us": 0.20},
        sector="technology",
        country="us",
        rating="A"
        # No lgd_distribution -> uses constant lgd
    ))

    print(f"   Portfolio: {stochastic_portfolio.name}")
    print(f"   Number of obligors: {len(stochastic_portfolio)}")
    for obligor in stochastic_portfolio.obligors:
        lgd_type = type(obligor.lgd_distribution).__name__ if obligor.lgd_distribution else "Constant"
        print(f"   - {obligor.name}: LGD type = {lgd_type}, mean = {obligor.get_lgd_mean():.2%}")

    print("\n11. Comparing constant vs stochastic LGD simulation...")

    # Run simulation with stochastic LGD
    engine_stochastic = MonteCarloEngine(model, random_state=42)
    result_stochastic = engine_stochastic.simulate(stochastic_portfolio, num_scenarios=50000)

    print(f"\n   Stochastic LGD Results:")
    print(f"   Expected Loss: ${result_stochastic.expected_loss:,.0f}")
    print(f"   Loss Std Dev: ${result_stochastic.loss_std:,.0f}")
    print(f"   VaR (99%): ${result_stochastic.get_var(0.99):,.0f}")
    print(f"   Expected Shortfall (99%): ${result_stochastic.get_expected_shortfall(0.99):,.0f}")

    # Create equivalent portfolio with constant LGD for comparison
    constant_portfolio = Portfolio(name="Constant LGD Portfolio")
    for obligor in stochastic_portfolio.obligors:
        constant_portfolio.add_obligor(Obligor(
            name=obligor.name,
            pd=obligor.pd,
            lgd=obligor.get_lgd_mean(),  # Use mean as constant
            ead=obligor.ead,
            factor_loadings=obligor.factor_loadings.copy(),
            sector=obligor.sector,
            country=obligor.country,
            rating=obligor.rating
            # No lgd_distribution
        ))

    engine_constant = MonteCarloEngine(model, random_state=42)
    result_constant = engine_constant.simulate(constant_portfolio, num_scenarios=50000)

    print(f"\n   Constant LGD Results (same mean LGD):")
    print(f"   Expected Loss: ${result_constant.expected_loss:,.0f}")
    print(f"   Loss Std Dev: ${result_constant.loss_std:,.0f}")
    print(f"   VaR (99%): ${result_constant.get_var(0.99):,.0f}")
    print(f"   Expected Shortfall (99%): ${result_constant.get_expected_shortfall(0.99):,.0f}")

    print(f"\n   Impact of Stochastic LGD:")
    var_diff = result_stochastic.get_var(0.99) - result_constant.get_var(0.99)
    es_diff = result_stochastic.get_expected_shortfall(0.99) - result_constant.get_expected_shortfall(0.99)
    print(f"   VaR increase: ${var_diff:,.0f} ({var_diff/result_constant.get_var(0.99)*100:.1f}%)")
    print(f"   ES increase: ${es_diff:,.0f} ({es_diff/result_constant.get_expected_shortfall(0.99)*100:.1f}%)")
    print("   [Note: Stochastic LGD typically increases tail risk due to LGD-PD correlation]")

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
