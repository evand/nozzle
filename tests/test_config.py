"""Tests for nozzle config loading — exit_pressure_ratio and related options."""

import pytest
from nozzle.config import build_nozzle_spec
from nozzle.gas import pressure_ratio


class TestExitPressureRatio:
    """Test exit_pressure_ratio as a 4th exit condition option."""

    def test_pressure_ratio_gives_correct_mach(self):
        """P/P0 = 0.1278 → M ≈ 2.0 for γ=1.4."""
        cfg = {'type': 'conical', 'gamma': 1.4, 'exit_pressure_ratio': 0.1278}
        spec = build_nozzle_spec(cfg)
        assert abs(spec['M_exit'] - 2.0) < 0.01

    def test_pressure_ratio_sets_area_ratio(self):
        """exit_pressure_ratio should also set area_ratio via M_exit."""
        cfg = {'type': 'conical', 'gamma': 1.4, 'exit_pressure_ratio': 0.1278}
        spec = build_nozzle_spec(cfg)
        # M≈2.0 → A/A* ≈ 1.6875
        assert abs(spec['area_ratio'] - 1.6875) < 0.01

    def test_m_exit_overrides_pressure_ratio(self):
        """M_exit takes priority over exit_pressure_ratio."""
        cfg = {'type': 'conical', 'gamma': 1.4,
               'M_exit': 3.0, 'exit_pressure_ratio': 0.1278}
        spec = build_nozzle_spec(cfg)
        assert abs(spec['M_exit'] - 3.0) < 0.001

    def test_area_ratio_overrides_pressure_ratio(self):
        """area_ratio takes priority over exit_pressure_ratio."""
        cfg = {'type': 'conical', 'gamma': 1.4,
               'area_ratio': 10.0, 'exit_pressure_ratio': 0.1278}
        spec = build_nozzle_spec(cfg)
        assert abs(spec['area_ratio'] - 10.0) < 0.001

    def test_invalid_pressure_ratio_zero(self):
        """P/P0 = 0 should raise ValueError."""
        cfg = {'type': 'conical', 'exit_pressure_ratio': 0.0}
        with pytest.raises(ValueError, match="between 0 and 1"):
            build_nozzle_spec(cfg)

    def test_invalid_pressure_ratio_one(self):
        """P/P0 = 1 should raise ValueError."""
        cfg = {'type': 'conical', 'exit_pressure_ratio': 1.0}
        with pytest.raises(ValueError, match="between 0 and 1"):
            build_nozzle_spec(cfg)

    def test_invalid_pressure_ratio_negative(self):
        """P/P0 < 0 should raise ValueError."""
        cfg = {'type': 'conical', 'exit_pressure_ratio': -0.5}
        with pytest.raises(ValueError, match="between 0 and 1"):
            build_nozzle_spec(cfg)

    def test_roundtrip_pressure_ratio(self):
        """Verify roundtrip: M → P/P0 → M recovers original."""
        M_target = 2.5
        gamma = 1.4
        p_ratio = float(pressure_ratio(M_target, gamma))
        cfg = {'type': 'conical', 'gamma': gamma,
               'exit_pressure_ratio': p_ratio}
        spec = build_nozzle_spec(cfg)
        assert abs(spec['M_exit'] - M_target) < 1e-6
