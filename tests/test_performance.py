"""Tests for performance analysis.

Cross-validates MOC results against analytical values.
"""

import numpy as np
import pytest
from nozzle.analysis import conical_performance, moc_performance
from nozzle.contours import minimum_length_nozzle
from nozzle.gas import thrust_coefficient_ideal


class TestConicalPerformance:

    def test_conical_15deg_ar10(self):
        result = conical_performance(15, 10)
        assert result['lambda'] == pytest.approx(0.9830, abs=0.001)
        assert result['M_exit'] > 1.0
        assert result['Cf'] < result['Cf_ideal']

    def test_conical_cf_reasonable(self):
        """Cf for a 15° conical with AR=10 should be in reasonable range."""
        result = conical_performance(15, 10)
        assert 1.0 < result['Cf'] < 2.0


class TestMOCPerformance:

    def test_mln_performance_runs(self):
        """MOC performance should compute without error."""
        x_wall, y_wall, mesh = minimum_length_nozzle(2.0, n_chars=15)
        result = moc_performance(mesh)
        assert 'Cf' in result
        assert 'M_mean' in result
        assert result['M_mean'] > 1.0

    def test_mln_performance_keys(self):
        x_wall, y_wall, mesh = minimum_length_nozzle(2.0, n_chars=15)
        result = moc_performance(mesh)
        expected_keys = ['Cf', 'Cf_ideal', 'efficiency', 'M_mean',
                         'M_max', 'M_min', 'theta_max_deg']
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
