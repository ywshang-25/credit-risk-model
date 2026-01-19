"""Tests for lgd_distribution.py - LGD distribution classes."""

import pytest
import numpy as np

from credit_risk import (
    ConstantLGD,
    BetaLGD,
    EmpiricalLGD,
    create_lgd_distribution
)


class TestConstantLGD:
    """Tests for ConstantLGD class."""

    def test_create_constant_lgd(self):
        """Test creating a constant LGD."""
        lgd = ConstantLGD(value=0.45)
        assert lgd.mean() == 0.45
        assert lgd.std() == 0.0

    def test_constant_lgd_validation(self):
        """Test that invalid values raise errors."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            ConstantLGD(value=1.5)
        with pytest.raises(ValueError, match="between 0 and 1"):
            ConstantLGD(value=-0.1)

    def test_constant_lgd_sample(self):
        """Test sampling from constant LGD."""
        lgd = ConstantLGD(value=0.45)
        samples = lgd.sample(100)
        assert len(samples) == 100
        assert np.all(samples == 0.45)

    def test_constant_lgd_ignores_systematic_factor(self):
        """Test that systematic factor doesn't affect constant LGD."""
        lgd = ConstantLGD(value=0.45)
        factor = np.array([1.0, 2.0, -1.0])
        samples = lgd.sample(3, systematic_factor=factor)
        assert np.all(samples == 0.45)

    def test_constant_lgd_repr(self):
        """Test string representation."""
        lgd = ConstantLGD(value=0.45)
        assert "0.4500" in repr(lgd)


class TestBetaLGD:
    """Tests for BetaLGD class."""

    def test_create_beta_lgd(self):
        """Test creating a Beta LGD distribution."""
        lgd = BetaLGD(mean=0.45, std=0.15)
        assert lgd.mean() == 0.45
        assert lgd.std() == 0.15

    def test_beta_lgd_validation_floor_cap(self):
        """Test floor/cap validation."""
        with pytest.raises(ValueError, match="floor < cap"):
            BetaLGD(mean=0.45, std=0.15, floor=0.8, cap=0.2)

    def test_beta_lgd_validation_mean_range(self):
        """Test that mean must be between floor and cap."""
        with pytest.raises(ValueError, match="Mean must be between"):
            BetaLGD(mean=0.1, std=0.1, floor=0.3, cap=0.8)

    def test_beta_lgd_validation_std(self):
        """Test that std must be non-negative."""
        with pytest.raises(ValueError, match="non-negative"):
            BetaLGD(mean=0.45, std=-0.1)

    def test_beta_lgd_sample_bounds(self):
        """Test that samples are within bounds."""
        lgd = BetaLGD(mean=0.45, std=0.15, floor=0.1, cap=0.9)
        samples = lgd.sample(10000, random_state=42)
        assert np.all(samples >= 0.1)
        assert np.all(samples <= 0.9)

    def test_beta_lgd_sample_mean(self):
        """Test that sample mean is close to specified mean."""
        lgd = BetaLGD(mean=0.45, std=0.10)
        samples = lgd.sample(50000, random_state=42)
        assert np.abs(np.mean(samples) - 0.45) < 0.02

    def test_beta_lgd_sample_variation(self):
        """Test that samples have variation."""
        lgd = BetaLGD(mean=0.45, std=0.15)
        samples = lgd.sample(1000, random_state=42)
        assert np.std(samples) > 0.05

    def test_beta_lgd_reproducible(self):
        """Test reproducibility with same random state."""
        lgd = BetaLGD(mean=0.45, std=0.15)
        samples1 = lgd.sample(100, random_state=42)
        samples2 = lgd.sample(100, random_state=42)
        assert np.allclose(samples1, samples2)

    def test_beta_lgd_factor_sensitivity(self):
        """Test that factor sensitivity affects samples."""
        lgd = BetaLGD(mean=0.45, std=0.15, factor_sensitivity=0.5)
        # Positive factor (stress) should increase LGD
        factor_positive = np.full(1000, 2.0)
        # Negative factor (good economy) should decrease LGD
        factor_negative = np.full(1000, -2.0)

        samples_positive = lgd.sample(1000, systematic_factor=factor_positive, random_state=42)
        samples_negative = lgd.sample(1000, systematic_factor=factor_negative, random_state=42)

        assert np.mean(samples_positive) > np.mean(samples_negative)

    def test_beta_lgd_zero_std(self):
        """Test Beta LGD with zero std behaves like constant."""
        lgd = BetaLGD(mean=0.45, std=0.0)
        samples = lgd.sample(100)
        assert np.all(samples == 0.45)

    def test_beta_lgd_repr(self):
        """Test string representation."""
        lgd = BetaLGD(mean=0.45, std=0.15, factor_sensitivity=0.3)
        repr_str = repr(lgd)
        assert "0.4500" in repr_str
        assert "0.1500" in repr_str
        assert "0.30" in repr_str


class TestEmpiricalLGD:
    """Tests for EmpiricalLGD class."""

    def test_create_empirical_lgd(self, historical_lgd_data):
        """Test creating an Empirical LGD distribution."""
        lgd = EmpiricalLGD(historical_lgd=historical_lgd_data)
        assert lgd.n_observations == 100
        assert lgd.mean() == pytest.approx(np.mean(historical_lgd_data), rel=0.01)

    def test_empirical_lgd_validation_min_observations(self):
        """Test that at least 2 observations are required."""
        with pytest.raises(ValueError, match="at least 2"):
            EmpiricalLGD(historical_lgd=[0.5])

    def test_empirical_lgd_sample_bounds(self, historical_lgd_data):
        """Test that samples are within data range."""
        lgd = EmpiricalLGD(historical_lgd=historical_lgd_data)
        samples = lgd.sample(10000, random_state=42)
        assert np.all(samples >= np.min(historical_lgd_data))
        assert np.all(samples <= np.max(historical_lgd_data))

    def test_empirical_lgd_sample_distribution(self, historical_lgd_data):
        """Test that samples follow the empirical distribution."""
        lgd = EmpiricalLGD(historical_lgd=historical_lgd_data)
        samples = lgd.sample(50000, random_state=42)
        # Mean should be close to historical mean
        assert np.abs(np.mean(samples) - np.mean(historical_lgd_data)) < 0.02

    def test_empirical_lgd_reproducible(self, historical_lgd_data):
        """Test reproducibility with same random state."""
        lgd = EmpiricalLGD(historical_lgd=historical_lgd_data)
        samples1 = lgd.sample(100, random_state=42)
        samples2 = lgd.sample(100, random_state=42)
        assert np.allclose(samples1, samples2)

    def test_empirical_lgd_factor_sensitivity(self, historical_lgd_data):
        """Test that factor sensitivity affects samples."""
        lgd = EmpiricalLGD(historical_lgd=historical_lgd_data, factor_sensitivity=0.5)
        factor_positive = np.full(1000, 2.0)
        factor_negative = np.full(1000, -2.0)

        samples_positive = lgd.sample(1000, systematic_factor=factor_positive, random_state=42)
        samples_negative = lgd.sample(1000, systematic_factor=factor_negative, random_state=42)

        assert np.mean(samples_positive) > np.mean(samples_negative)

    def test_empirical_lgd_custom_floor_cap(self, historical_lgd_data):
        """Test custom floor and cap."""
        lgd = EmpiricalLGD(historical_lgd=historical_lgd_data, floor=0.2, cap=0.8)
        samples = lgd.sample(10000, random_state=42)
        assert np.all(samples >= 0.2)
        assert np.all(samples <= 0.8)

    def test_empirical_lgd_percentile(self, historical_lgd_data):
        """Test percentile method."""
        lgd = EmpiricalLGD(historical_lgd=historical_lgd_data)
        p50 = lgd.percentile(50)
        assert np.abs(p50 - np.median(historical_lgd_data)) < 0.05

    def test_empirical_lgd_repr(self, historical_lgd_data):
        """Test string representation."""
        lgd = EmpiricalLGD(historical_lgd=historical_lgd_data, factor_sensitivity=0.2)
        repr_str = repr(lgd)
        assert "n_obs=100" in repr_str
        assert "0.20" in repr_str


class TestCreateLGDDistribution:
    """Tests for the factory function."""

    def test_create_constant(self):
        """Test creating constant LGD via factory."""
        lgd = create_lgd_distribution('constant', value=0.45)
        assert isinstance(lgd, ConstantLGD)
        assert lgd.mean() == 0.45

    def test_create_beta(self):
        """Test creating beta LGD via factory."""
        lgd = create_lgd_distribution('beta', mean=0.45, std=0.15)
        assert isinstance(lgd, BetaLGD)
        assert lgd.mean() == 0.45

    def test_create_empirical(self, historical_lgd_data):
        """Test creating empirical LGD via factory."""
        lgd = create_lgd_distribution('empirical', historical_lgd=historical_lgd_data)
        assert isinstance(lgd, EmpiricalLGD)

    def test_create_case_insensitive(self):
        """Test that type is case insensitive."""
        lgd1 = create_lgd_distribution('CONSTANT', value=0.45)
        lgd2 = create_lgd_distribution('Constant', value=0.45)
        lgd3 = create_lgd_distribution('constant', value=0.45)
        assert all(isinstance(l, ConstantLGD) for l in [lgd1, lgd2, lgd3])

    def test_create_unknown_type(self):
        """Test that unknown type raises error."""
        with pytest.raises(ValueError, match="Unknown LGD distribution type"):
            create_lgd_distribution('unknown', value=0.45)
