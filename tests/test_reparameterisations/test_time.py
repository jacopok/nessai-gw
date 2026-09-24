"""Tests for :class:`DetectorCenterTimeReparameterisation`."""

import numpy as np
import pytest
from nessai.livepoint import dict_to_live_points, empty_structured_array

from nessai_gw._geometry import geocenter_time_delay, greenwich_mean_sidereal_time
from nessai_gw.group_mixture import ET_EMR_VERTEX
from nessai_gw.reparameterisations import DetectorCenterTimeReparameterisation

REFERENCE_TIME = 1187008882.4
SCALE = 5e-3


class TestDetectorCenterTime:
    prior_bounds = {"geocent_time": [REFERENCE_TIME - 0.1, REFERENCE_TIME + 0.1]}

    def _reparam(self, scale=SCALE):
        return DetectorCenterTimeReparameterisation(
            parameters="geocent_time",
            prior_bounds=self.prior_bounds,
            vertex=ET_EMR_VERTEX,
            reference_time=REFERENCE_TIME,
            scale=scale,
        )

    def test_init(self):
        reparam = self._reparam()
        assert reparam.prime_parameters == ["t_det"]
        assert reparam.requires == ["ra", "dec"]
        assert reparam.scale == SCALE

    def test_requires_vertex_and_reference_time(self):
        with pytest.raises(ValueError, match="requires `vertex`"):
            DetectorCenterTimeReparameterisation(
                parameters="geocent_time", prior_bounds=self.prior_bounds
            )

    def test_must_act_on_geocent_time(self):
        with pytest.raises(RuntimeError, match="must act on"):
            DetectorCenterTimeReparameterisation(
                parameters="ra",
                prior_bounds={"ra": [0.0, 2 * np.pi]},
                vertex=ET_EMR_VERTEX,
                reference_time=REFERENCE_TIME,
            )

    def _points(self, n, rng):
        ra = rng.uniform(0, 2 * np.pi, n)
        dec = np.arcsin(rng.uniform(-1, 1, n))
        gt = REFERENCE_TIME + rng.uniform(-0.05, 0.05, n)
        return ra, dec, gt

    def test_reparameterise_values(self):
        reparam = self._reparam()
        rng = np.random.default_rng(1)
        n = 25
        ra, dec, gt = self._points(n, rng)
        x = dict_to_live_points(
            {"geocent_time": gt, "ra": ra, "dec": dec}
        )
        x_prime = empty_structured_array(n, names=["t_det", "ra", "dec"])
        x_prime["ra"] = ra
        x_prime["dec"] = dec
        log_j = np.zeros(n)
        _, x_prime, log_j = reparam.reparameterise(x, x_prime, log_j)

        gmst = greenwich_mean_sidereal_time(REFERENCE_TIME)
        delay = geocenter_time_delay(ET_EMR_VERTEX, gmst, ra, dec)
        # epoch subtracted first: gt + delay would round onto the ~0.24 us
        # float64 grid of GPS times
        want = ((gt - REFERENCE_TIME) + delay) / SCALE
        np.testing.assert_allclose(x_prime["t_det"], want, atol=1e-9)
        np.testing.assert_allclose(log_j, -np.log(SCALE), atol=1e-12)

    @pytest.mark.integration_test
    def test_invertible(self):
        reparam = self._reparam()
        rng = np.random.default_rng(2)
        n = 50
        ra, dec, gt = self._points(n, rng)
        x = dict_to_live_points({"geocent_time": gt, "ra": ra, "dec": dec})
        names = ["t_det", "ra", "dec"]
        x_prime = empty_structured_array(n, names=names)
        x_prime["ra"] = ra
        x_prime["dec"] = dec
        log_j = np.zeros(n)

        x_f, x_prime_f, log_j_f = reparam.reparameterise(
            x.copy(), x_prime.copy(), log_j.copy()
        )
        np.testing.assert_array_equal(x_f["geocent_time"], gt)

        x_in = x_f.copy()
        x_in["geocent_time"] = np.nan
        x_i, _, log_j_i = reparam.inverse_reparameterise(
            x_in, x_prime_f.copy(), log_j_f.copy()
        )
        np.testing.assert_allclose(x_i["geocent_time"], gt, rtol=1e-12)
        np.testing.assert_allclose(log_j_i, 0.0, atol=1e-10)

    def test_reflection_shifts_geocent_time_by_delay_change(self):
        """When a group reflection moves (ra, dec), holding t_det fixed
        reconstructs geocent_time shifted by delay(n) - delay(n')."""
        reparam = self._reparam()
        rng = np.random.default_rng(3)
        n = 30
        ra, dec, gt = self._points(n, rng)
        gmst = greenwich_mean_sidereal_time(REFERENCE_TIME)

        # forward to t_det
        x = dict_to_live_points({"geocent_time": gt, "ra": ra, "dec": dec})
        x_prime = empty_structured_array(n, names=["t_det", "ra", "dec"])
        x_prime["ra"] = ra
        x_prime["dec"] = dec
        _, x_prime, _ = reparam.reparameterise(x, x_prime, np.zeros(n))

        # a "reflection" of the sky direction (arbitrary here): dec -> -dec
        ra2, dec2 = ra, -dec
        x_in = empty_structured_array(
            n, names=["geocent_time", "ra", "dec"]
        )
        x_in["ra"] = ra2
        x_in["dec"] = dec2
        x_in["geocent_time"] = np.nan
        x_i, _, _ = reparam.inverse_reparameterise(
            x_in, x_prime.copy(), np.zeros(n)
        )

        d0 = geocenter_time_delay(ET_EMR_VERTEX, gmst, ra, dec)
        d1 = geocenter_time_delay(ET_EMR_VERTEX, gmst, ra2, dec2)
        np.testing.assert_allclose(
            x_i["geocent_time"], gt + d0 - d1, rtol=1e-12
        )
