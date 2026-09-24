"""Timing reparameterisations for single-site triangular detectors."""

import inspect

import numpy as np
from nessai.reparameterisations import Reparameterisation

from .. import nessai_logger
from .._geometry import geocenter_time_delay, greenwich_mean_sidereal_time

logger = nessai_logger.getChild(__name__)


class DetectorCenterTimeReparameterisation(Reparameterisation):
    """Reparameterise ``geocent_time`` to the arrival time at the detector centre.

    For a single triangular detector (motivating case: one Einstein Telescope
    site) the frozen-limit extrinsic degeneracy fixes the arrival time at the
    detector centre, ``t_det``, not bilby's geocentric ``geocent_time``.  With
    ``n`` the unit vector towards the source and ``r_det`` the detector
    ``vertex``,

        t_det = geocent_time + delay(n),   delay(n) = -(n . r_det) / c

    (``delay`` is bilby's ``time_delay_from_geocenter``, i.e. ``t_det -
    geocent_time``).  The flow-facing coordinate is the affine-rescaled
    ``t_det`` ::

        t_det_prime = (geocent_time + delay(ra, dec) - reference_time) / scale

    Because ``t_det`` is invariant under the triangular-detector group,
    :class:`nessai_gw.group_mixture.PrimeSpaceTriangularGroupAction` passes this
    prime coordinate through unchanged -- the group need not act on time at all.
    See Tissino et al. 2026, arXiv:2606.04918
    (https://arxiv.org/abs/2606.04918).

    ``ra`` and ``dec`` are held fixed by this map (the group action is what
    transforms them), so ``d t_det / d geocent_time = 1`` and the Jacobian is the
    constant ``|d t_det_prime / d geocent_time| = 1 / scale`` -- exactly the
    ``scaleandshift`` entry this replaces.

    Requires ``ra`` and ``dec`` on both the forward and the inverse pass.

    Parameters
    ----------
    parameters : Union[str, List[str]]
        Name of the parameter; must be ``geocent_time``.
    prior_bounds : Union[list, dict], optional
        Prior bounds for ``geocent_time``.  Unused -- the coordinate is always
        rescaled by the fixed ``scale`` / ``reference_time``.
    vertex : array_like
        Earth-fixed detector position in metres, shape ``(3,)`` (the triangle
        centroid; see :func:`nessai_gw.group_mixture.detector_vertex`).
    reference_time : float
        Geocentric GPS time of the event; its GMST rotates between the
        equatorial and Earth-fixed frames and it is the constant subtracted in
        prime space.
    scale : float, optional
        Prime-space scale in seconds (default ``5e-3``, matching
        :data:`nessai_gw.group_mixture._GEOCENT_SCALE`).
    prior : optional
        Accepted for registry compatibility and ignored.
    """

    one_to_one = False

    def __init__(
        self,
        parameters=None,
        prior_bounds=None,
        vertex=None,
        reference_time=None,
        scale=5e-3,
        prior=None,
        rng=None,
        **kwargs,
    ):
        parent_params = inspect.signature(
            Reparameterisation.__init__
        ).parameters
        call = dict(parameters=parameters, prior_bounds=prior_bounds)
        if "rng" in parent_params:
            call["rng"] = rng
        for key, value in kwargs.items():
            if key in parent_params:
                call[key] = value
        super().__init__(**call)

        if self.parameters != ["geocent_time"]:
            raise RuntimeError(
                "DetectorCenterTimeReparameterisation must act on "
                f"'geocent_time'; got {self.parameters}"
            )
        if vertex is None or reference_time is None:
            raise ValueError(
                "DetectorCenterTimeReparameterisation requires `vertex` and "
                "`reference_time`."
            )
        self._vertex = np.asarray(vertex, dtype=float)
        if self._vertex.shape != (3,):
            raise ValueError(
                f"`vertex` must be shape (3,), got {self._vertex.shape}."
            )
        self._reference_time = float(reference_time)
        self._gmst = greenwich_mean_sidereal_time(self._reference_time)
        self.scale = float(scale)

        self.prime_parameters = ["t_det"]
        if hasattr(self, "output_parameters"):
            self.output_parameters = ["t_det"]
        self.requires = ["ra", "dec"]
        if hasattr(self, "inverse_input_parameters"):
            self.inverse_input_parameters = list(
                dict.fromkeys(
                    list(self.inverse_input_parameters or []) + ["ra", "dec"]
                )
            )
        self._log_j = -float(np.log(self.scale))

    def _delay(self, x):
        return geocenter_time_delay(
            self._vertex, self._gmst, x["ra"], x["dec"]
        )

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        # Subtract the GPS epoch *before* adding the delay: at ~1e9 s a float64
        # only resolves ~0.24 us, so ``geocent_time + delay`` would round
        # ``t_det`` onto that grid -- a comb of only ~20 teeth across the
        # timing posterior of an SNR ~ 700 signal.
        t_det = (x["geocent_time"] - self._reference_time) + self._delay(x)
        x_prime[self.prime_parameters[0]] = t_det / self.scale
        return x, x_prime, log_j + self._log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        t_det = x_prime[self.prime_parameters[0]] * self.scale
        x["geocent_time"] = self._reference_time + (t_det - self._delay(x))
        return x, x_prime, log_j - self._log_j
