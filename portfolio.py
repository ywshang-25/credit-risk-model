"""Portfolio and obligor data structures for credit risk modeling."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class Obligor:
    """Represents a single obligor (borrower) in the portfolio.

    Attributes:
        name: Unique identifier for the obligor
        pd: Probability of default (annualized)
        lgd: Loss given default (recovery = 1 - lgd)
        ead: Exposure at default
        factor_loadings: Dict mapping factor names to loadings (betas)
        sector: Industry sector classification
        country: Geographic region/country
        rating: Credit rating (e.g., 'AAA', 'BB', 'CCC')
    """
    name: str
    pd: float
    lgd: float
    ead: float
    factor_loadings: Dict[str, float] = field(default_factory=dict)
    sector: Optional[str] = None
    country: Optional[str] = None
    rating: Optional[str] = None

    def __post_init__(self):
        if not 0 <= self.pd <= 1:
            raise ValueError(f"PD must be between 0 and 1, got {self.pd}")
        if not 0 <= self.lgd <= 1:
            raise ValueError(f"LGD must be between 0 and 1, got {self.lgd}")
        if self.ead < 0:
            raise ValueError(f"EAD must be non-negative, got {self.ead}")

        total_loading_sq = sum(b**2 for b in self.factor_loadings.values())
        if total_loading_sq > 1:
            raise ValueError(
                f"Sum of squared factor loadings must be <= 1, got {total_loading_sq}"
            )

    @property
    def expected_loss(self) -> float:
        """Calculate expected loss for this obligor."""
        return self.pd * self.lgd * self.ead

    @property
    def idiosyncratic_weight(self) -> float:
        """Calculate the idiosyncratic (firm-specific) weight."""
        total_loading_sq = sum(b**2 for b in self.factor_loadings.values())
        return np.sqrt(1 - total_loading_sq)


class Portfolio:
    """Collection of obligors forming a credit portfolio.

    Provides methods for adding/removing obligors and aggregating
    portfolio-level statistics.
    """

    def __init__(self, name: str = "Portfolio"):
        self.name = name
        self._obligors: Dict[str, Obligor] = {}

    def add_obligor(self, obligor: Obligor) -> None:
        """Add an obligor to the portfolio."""
        if obligor.name in self._obligors:
            raise ValueError(f"Obligor '{obligor.name}' already exists in portfolio")
        self._obligors[obligor.name] = obligor

    def remove_obligor(self, name: str) -> Obligor:
        """Remove and return an obligor from the portfolio."""
        if name not in self._obligors:
            raise KeyError(f"Obligor '{name}' not found in portfolio")
        return self._obligors.pop(name)

    def get_obligor(self, name: str) -> Obligor:
        """Get an obligor by name."""
        if name not in self._obligors:
            raise KeyError(f"Obligor '{name}' not found in portfolio")
        return self._obligors[name]

    @property
    def obligors(self) -> List[Obligor]:
        """Return list of all obligors."""
        return list(self._obligors.values())

    @property
    def obligor_names(self) -> List[str]:
        """Return list of all obligor names."""
        return list(self._obligors.keys())

    def __len__(self) -> int:
        return len(self._obligors)

    def __iter__(self):
        return iter(self._obligors.values())

    def __contains__(self, name: str) -> bool:
        return name in self._obligors

    @property
    def total_ead(self) -> float:
        """Total exposure at default across all obligors."""
        return sum(o.ead for o in self._obligors.values())

    @property
    def total_expected_loss(self) -> float:
        """Total expected loss across all obligors."""
        return sum(o.expected_loss for o in self._obligors.values())

    def get_factor_names(self) -> set:
        """Get all unique factor names used across obligors."""
        factors = set()
        for obligor in self._obligors.values():
            factors.update(obligor.factor_loadings.keys())
        return factors

    def get_sectors(self) -> set:
        """Get all unique sectors in the portfolio."""
        return {o.sector for o in self._obligors.values() if o.sector is not None}

    def get_countries(self) -> set:
        """Get all unique countries in the portfolio."""
        return {o.country for o in self._obligors.values() if o.country is not None}

    def get_ratings(self) -> set:
        """Get all unique ratings in the portfolio."""
        return {o.rating for o in self._obligors.values() if o.rating is not None}

    def filter_by_sector(self, sector: str) -> List[Obligor]:
        """Return obligors belonging to a specific sector."""
        return [o for o in self._obligors.values() if o.sector == sector]

    def filter_by_country(self, country: str) -> List[Obligor]:
        """Return obligors belonging to a specific country."""
        return [o for o in self._obligors.values() if o.country == country]

    def filter_by_rating(self, rating: str) -> List[Obligor]:
        """Return obligors with a specific rating."""
        return [o for o in self._obligors.values() if o.rating == rating]

    def copy(self) -> "Portfolio":
        """Create a deep copy of the portfolio."""
        new_portfolio = Portfolio(name=f"{self.name}_copy")
        for obligor in self._obligors.values():
            new_obligor = Obligor(
                name=obligor.name,
                pd=obligor.pd,
                lgd=obligor.lgd,
                ead=obligor.ead,
                factor_loadings=obligor.factor_loadings.copy(),
                sector=obligor.sector,
                country=obligor.country,
                rating=obligor.rating
            )
            new_portfolio.add_obligor(new_obligor)
        return new_portfolio
