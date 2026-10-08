"""Timing reparameterisations for single-site triangular detectors."""

import inspect

import numpy as np
from nessai.reparameterisations import Reparameterisation

from .. import nessai_logger
from .._geometry import geocenter_time_delay, greenwich_mean_sidereal_time
from .mass import doppler_factor

logger = nessai_logger.getChild(__name__)

#: G M_sun / c^3 in seconds.
_SOLAR_MASS_TIME = 4.925490947641267e-06


def pn_time_to_merger(chirp_mass, mass_ratio, chi_1, chi_2, frequency):
    """Time to merger (s) from the moment the 22 mode has ``frequency`` (Hz).

    The 2PN TaylorT2 expression with the aligned-spin spin-orbit term (the
    spin-spin term is left out: at this order it shifts the time by a
    smooth, nearly constant amount, which is all the coordinate needs),

    .. math::
        \\tau = \\frac{5 M}{256 \\eta v^8}\\Big[1 + \\Big(\\frac{743}{252}
        + \\frac{11}{3}\\eta\\Big) v^2 - \\frac{8}{5}(4\\pi - \\beta) v^3
        + \\Big(\\frac{3058673}{508032} + \\frac{5429}{504}\\eta
        + \\frac{617}{72}\\eta^2\\Big) v^4\\Big],

    with :math:`v = (\\pi M f)^{1/3}` and
    :math:`\\beta = \\sum_i \\chi_i (113 m_i^2 / M^2 + 75 \\eta) / 12`.
    ``mass_ratio`` is ``m_2 / m_1 <= 1`` (bilby's), ``chi_1`` belongs to the
    heavier mass, and the masses are those in the frame the signal is
    observed in.
    """
    q = np.asarray(mass_ratio, dtype=float)
    eta = q / (1.0 + q) ** 2
    total = np.asarray(chirp_mass, dtype=float) * eta**-0.6
    m1, m2 = 1.0 / (1.0 + q), q / (1.0 + q)
    beta = (chi_1 * (113.0 * m1**2 + 75.0 * eta)
            + chi_2 * (113.0 * m2**2 + 75.0 * eta)) / 12.0
    v2 = (np.pi * total * _SOLAR_MASS_TIME * frequency) ** (2.0 / 3.0)
    v = np.sqrt(v2)
    return 5.0 * total * _SOLAR_MASS_TIME / (256.0 * eta * v2**4) * (
        1.0
        + (743.0 / 252.0 + 11.0 / 3.0 * eta) * v2
        - 8.0 / 5.0 * (4.0 * np.pi - beta) * v * v2
        + (3058673.0 / 508032.0 + 5429.0 / 504.0 * eta
           + 617.0 / 72.0 * eta**2) * v2**2
    )


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

    **Time at a reference frequency.**  The merger time is degenerate with
    the chirp mass, mass ratio and spins: a change in those moves the merger
    relative to the part of the signal the detector measures best.  With
    ``reference_frequency`` the coordinate is instead the arrival time at the
    detector of the moment the 22 mode passes that frequency (Tissino et al.
    2026, Eq. 18),

        t_f = t_det - (tau(f) - tau_ref)

    with ``tau`` the 2PN time to merger (:func:`pn_time_to_merger`) and
    ``tau_ref`` a constant that keeps the coordinate near zero.  Given the
    masses and spins this is a shift of ``t_det``: the Jacobian is still
    ``1 / scale``.  With the Earth's orbital motion (``doppler_vector``,
    as for ``doppler-chirp-mass``) ``tau`` is computed from the chirp mass
    in the frame comoving with the Earth, ``chirp_mass (1 - n.v/c)``: that
    is the time at the detector's own position when the signal it receives
    passes ``f``, and it is invariant under the triangular group, like
    ``t_det``, as long as the chirp mass is sampled with the same
    ``doppler_vector``.  (The paper's form, barycentric masses and a fixed
    barycentric reference point, reaches the same minimum variance only by
    moving the point ~(5/3) v tau against the Earth's velocity.)

    ``reference_frequency="adaptive"`` chooses ``f`` before every training
    (:meth:`update`, on the training points) as the frequency that minimises
    the spread of ``t_f`` over them, as in the paper's Fig. 4; the search is a
    vectorised grid over ``frequency_range`` and costs milliseconds.  The
    masses, mass ratio and spins are then also read on the inverse pass, so
    they are inverted first.

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
    reference_frequency : float or "adaptive", optional
        Gravitational-wave frequency (Hz) of the 22 mode at which the time is
        measured; ``None`` (default) for the merger time.
    frequency_range : tuple of float, optional
        Search range (Hz) for ``reference_frequency="adaptive"``; also its
        first value before any update is the geometric mean of the range.
    doppler_vector : array_like, optional
        :math:`\\mathbf{v}_\\oplus / c` in the axes of ``(ra, dec)``, as for
        ``doppler-chirp-mass``; ``None`` for no orbital motion.
    chirp_mass, mass_ratio : str, optional
        Names of the chirp mass and mass ratio.
    spins : tuple of str or None, optional
        Names of the aligned spins of the heavier and lighter mass, or
        ``None`` for no spins.
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
        reference_frequency=None,
        frequency_range=(8.0, 300.0),
        doppler_vector=None,
        chirp_mass="chirp_mass",
        mass_ratio="mass_ratio",
        spins=("chi_1", "chi_2"),
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

        self.adaptive = reference_frequency == "adaptive"
        self.frequency_range = tuple(float(f) for f in frequency_range)
        if len(self.frequency_range) != 2 or not (
            0 < self.frequency_range[0] < self.frequency_range[1]
        ):
            raise ValueError(
                f"`frequency_range` must be (low, high) with 0 < low < high; "
                f"got {frequency_range}"
            )
        if self.adaptive:
            reference_frequency = float(np.sqrt(np.prod(self.frequency_range)))
        elif reference_frequency is not None:
            reference_frequency = float(reference_frequency)
            if not reference_frequency > 0:
                raise ValueError("`reference_frequency` must be positive")
        self.reference_frequency = reference_frequency
        self.doppler_vector = (
            None if doppler_vector is None
            else np.asarray(doppler_vector, dtype=float)
        )
        if self.doppler_vector is not None and self.doppler_vector.shape != (3,):
            raise ValueError(
                f"`doppler_vector` must have 3 components; got {doppler_vector}"
            )
        self._chirp_mass, self._mass_ratio = chirp_mass, mass_ratio
        self._spins = None if spins is None else tuple(spins)
        if self._spins is not None and len(self._spins) != 2:
            raise ValueError(f"`spins` must name two parameters; got {spins}")
        self._tau_ref = None
        if self.reference_frequency is not None:
            self.requires += [chirp_mass, mass_ratio] + list(self._spins or [])

        if hasattr(self, "inverse_input_parameters"):
            self.inverse_input_parameters = list(
                dict.fromkeys(
                    list(self.inverse_input_parameters or []) + self.requires
                )
            )
        self._log_j = -float(np.log(self.scale))

    def _delay(self, x):
        return geocenter_time_delay(
            self._vertex, self._gmst, x["ra"], x["dec"]
        )

    def _tau(self, x, frequency=None):
        """2PN time to merger (s) from ``frequency`` (default: the reference
        frequency), with the chirp mass in the frame of the Earth."""
        chirp_mass = x[self._chirp_mass]
        if self.doppler_vector is not None:
            chirp_mass = chirp_mass * doppler_factor(
                x["ra"], x["dec"], self.doppler_vector
            )
        if self._spins is None:
            chi_1 = chi_2 = 0.0
        else:
            chi_1, chi_2 = x[self._spins[0]], x[self._spins[1]]
        return pn_time_to_merger(
            chirp_mass, x[self._mass_ratio], chi_1, chi_2,
            self.reference_frequency if frequency is None else frequency,
        )

    def _shift(self, x):
        """``tau(f) - tau_ref`` (s): what is subtracted from ``t_det``."""
        if self.reference_frequency is None:
            return 0.0
        tau = self._tau(x)
        if self._tau_ref is None:
            finite = np.isfinite(tau)
            self._tau_ref = (
                float(np.median(tau[finite])) if finite.any() else 0.0
            )
        return tau - self._tau_ref

    def update(self, x, x_prime=None):
        """With ``reference_frequency="adaptive"``, move the frequency to
        the one that minimises the spread of the coordinate over ``x`` (the
        training points); with any reference frequency, recentre
        ``tau_ref`` on them."""
        if self.reference_frequency is None or len(x) < 2:
            return
        t_det = (x["geocent_time"] - self._reference_time) + self._delay(x)
        if self.adaptive:
            low, high = self.frequency_range
            freqs = np.geomspace(low, high, 97)
            spreads = np.array([
                np.nanstd(t_det - self._tau(x, f)) for f in freqs
            ])
            if np.isfinite(spreads).any():
                i = int(np.nanargmin(spreads))
                # refine between the neighbouring grid points
                fine = np.geomspace(freqs[max(i - 1, 0)],
                                    freqs[min(i + 1, len(freqs) - 1)], 33)
                fine_spreads = np.array([
                    np.nanstd(t_det - self._tau(x, f)) for f in fine
                ])
                j = int(np.nanargmin(fine_spreads))
                previous = self.reference_frequency
                self.reference_frequency = float(fine[j])
                logger.info(
                    "Arrival time measured at %.1f Hz (was %.1f Hz): spread "
                    "over the training points %.3g ms, %.3g ms at merger",
                    self.reference_frequency, previous,
                    1e3 * fine_spreads[j], 1e3 * np.std(t_det),
                )
        tau = self._tau(x)
        finite = np.isfinite(tau)
        if finite.any():
            self._tau_ref = float(np.median(tau[finite]))

    def reset(self):
        self._tau_ref = None

    def __setstate__(self, state):
        # checkpoints pickled before the reference frequency existed
        for key, value in (
            ("reference_frequency", None), ("adaptive", False),
            ("frequency_range", (8.0, 300.0)), ("doppler_vector", None),
            ("_chirp_mass", "chirp_mass"), ("_mass_ratio", "mass_ratio"),
            ("_spins", ("chi_1", "chi_2")), ("_tau_ref", None),
        ):
            state.setdefault(key, value)
        self.__dict__.update(state)

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        # Subtract the GPS epoch *before* adding the delay: at ~1e9 s a float64
        # only resolves ~0.24 us, so ``geocent_time + delay`` would round
        # ``t_det`` onto that grid -- a comb of only ~20 teeth across the
        # timing posterior of an SNR ~ 700 signal.
        t_det = (x["geocent_time"] - self._reference_time) + self._delay(x)
        x_prime[self.prime_parameters[0]] = (t_det - self._shift(x)) / self.scale
        return x, x_prime, log_j + self._log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        t_det = x_prime[self.prime_parameters[0]] * self.scale + self._shift(x)
        x["geocent_time"] = self._reference_time + (t_det - self._delay(x))
        return x, x_prime, log_j - self._log_j
