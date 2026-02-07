"""Tests for CSV contour loading."""

import numpy as np
import pytest
import tempfile
from pathlib import Path
from nozzle.contours import load_contour_csv, conical_nozzle


class TestLoadContourCSV:

    def _write_csv(self, tmpdir, x, y, filename="contour.csv"):
        path = tmpdir / filename
        np.savetxt(path, np.column_stack([x, y]), delimiter=',',
                   header='x, y')
        return path

    def test_load_basic(self, tmp_path):
        """Load a simple CSV contour."""
        x = np.array([0, 1, 2, 3, 4])
        y = np.array([1.0, 1.2, 1.5, 1.8, 2.0])
        path = self._write_csv(tmp_path, x, y)
        x_out, y_out = load_contour_csv(path)
        assert len(x_out) == 5
        assert len(y_out) == 5
        assert x_out[0] == pytest.approx(0.0, abs=0.01)
        assert y_out[0] == pytest.approx(1.0, abs=0.01)

    def test_throat_at_zero(self, tmp_path):
        """After loading, throat should be at x=0."""
        x = np.array([10, 11, 12, 13, 14])
        y = np.array([5.0, 4.5, 4.0, 4.5, 5.0])
        path = self._write_csv(tmp_path, x, y)
        x_out, y_out = load_contour_csv(path)
        # Throat is at x=12 (min y), so x_out should start at -2
        assert x_out[0] == pytest.approx(-2.0 / 4.0, abs=0.01)

    def test_normalization(self, tmp_path):
        """Y should be normalized to 1.0 at the throat."""
        x = np.array([0, 1, 2, 3])
        y = np.array([5.0, 4.0, 5.0, 6.0])
        path = self._write_csv(tmp_path, x, y)
        x_out, y_out = load_contour_csv(path)
        # Min y = 4.0, so y should be normalized by 4.0
        assert min(y_out) == pytest.approx(1.0, abs=0.01)

    def test_explicit_r_throat(self, tmp_path):
        """Explicit r_throat should override auto-detection."""
        x = np.array([0, 1, 2, 3])
        y = np.array([2.0, 2.0, 3.0, 4.0])
        path = self._write_csv(tmp_path, x, y)
        x_out, y_out = load_contour_csv(path, r_throat=2.0)
        assert y_out[0] == pytest.approx(1.0, abs=0.01)

    def test_roundtrip_conical(self, tmp_path):
        """Load a saved conical nozzle and check it round-trips."""
        x, y = conical_nozzle(15, 10)
        path = self._write_csv(tmp_path, x, y)
        x_out, y_out = load_contour_csv(path, r_throat=1.0, x_throat=0.0)
        np.testing.assert_allclose(x_out, x, atol=1e-10)
        np.testing.assert_allclose(y_out, y, atol=1e-10)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_contour_csv("/nonexistent/path.csv")

    def test_too_few_points(self, tmp_path):
        path = tmp_path / "short.csv"
        np.savetxt(path, np.array([[0, 1]]), delimiter=',')
        with pytest.raises(ValueError, match="at least 2"):
            load_contour_csv(path)

    def test_sorted_output(self, tmp_path):
        """Output should be sorted by x even if input is not."""
        x = np.array([3, 1, 2, 0])
        y = np.array([2.0, 1.2, 1.5, 1.0])
        path = self._write_csv(tmp_path, x, y)
        x_out, y_out = load_contour_csv(path)
        assert np.all(np.diff(x_out) >= 0)
