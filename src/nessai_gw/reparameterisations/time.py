"""Timing reparameterisations for single-site triangular detectors."""

import importlib
import inspect
from itertools import combinations_with_replacement

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

    **The waveform's own time.**  2PN leaves out the higher orders and the
    tides, and the merger the time refers to is itself shifted by the tides
    (more deformable stars merge earlier, by milliseconds for a BNS): with
    ``time_to_merger``, a function giving the waveform's
    :math:`-\\frac{1}{2\\pi}\\partial_f\\phi_{22}`, ``tau`` is 2PN plus a
    polynomial correction in the intrinsic parameters (``correction_degree``)
    fitted, at every :meth:`update`, to ``tau_waveform - tau_2PN`` on the
    training points; the adaptive frequency is then chosen with the
    waveform's time.  The waveform is only evaluated there (it is far too
    slow for every flow draw); any smooth correction keeps the map exactly
    invertible, the fit only sets how well the coordinate decorrelates.

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
    time_to_merger : callable or str, optional
        ``f(params, frequencies) -> tau`` of shape ``(N, len(frequencies))``,
        the waveform's time (s) from each frequency of the 22 mode to the
        merger, ``params`` a dict of arrays ``chirp_mass`` (in the Earth's
        frame), ``mass_ratio`` (``<= 1``), ``chi_1``, ``chi_2``, ``lambda_1``,
        ``lambda_2``; NaN where the waveform is not defined.  A
        ``"module:function"`` string is imported when first needed, and is
        what is pickled with the proposal (a checkpoint then loads without
        that module).  ``None`` (default): 2PN only.
    tides : tuple of str or None, optional
        Names of the tidal deformabilities of the heavier and lighter mass,
        read for ``time_to_merger`` (``None``: zero).
    correction_degree : int, optional
        Degree of the polynomial correction to 2PN (default 2).
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
        time_to_merger=None,
        tides=("lambda_1", "lambda_2"),
        correction_degree=2,
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
        self._time_to_merger = time_to_merger
        self._time_to_merger_fn = None
        self._tides = None if tides is None else tuple(tides)
        if self._tides is not None and len(self._tides) != 2:
            raise ValueError(f"`tides` must name two parameters; got {tides}")
        self._correction_degree = int(correction_degree)
        self._correction = None
        if time_to_merger is not None and self.reference_frequency is None:
            raise ValueError("`time_to_merger` needs a `reference_frequency`")
        if self.reference_frequency is not None:
            self.requires += [chirp_mass, mass_ratio] + list(self._spins or [])
            if time_to_merger is not None:
                self.requires += list(self._tides or [])

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

    def _params(self, x):
        """The intrinsic parameters ``tau`` depends on, as a dict of arrays
        (the chirp mass in the Earth's frame)."""
        chirp_mass = np.asarray(x[self._chirp_mass], dtype=float)
        if self.doppler_vector is not None:
            chirp_mass = chirp_mass * doppler_factor(
                x["ra"], x["dec"], self.doppler_vector
            )
        zeros = np.zeros_like(chirp_mass)
        params = dict(chirp_mass=chirp_mass,
                      mass_ratio=np.asarray(x[self._mass_ratio], dtype=float))
        for key, names in (("chi", self._spins), ("lambda", self._tides)):
            for i in (0, 1):
                params[f"{key}_{i + 1}"] = (
                    zeros if names is None or names[i] not in x.dtype.names
                    else np.asarray(x[names[i]], dtype=float)
                )
        return params

    def _tau_2pn(self, params, frequency):
        return pn_time_to_merger(
            params["chirp_mass"], params["mass_ratio"], params["chi_1"],
            params["chi_2"], frequency,
        )

    def _features(self, params):
        """Polynomial features for the correction, standardised on the
        training points of its fit (and clipped, so that the tails of the
        flow stay finite and smooth)."""
        c = self._correction
        u = np.stack([params[k] for k in c["keys"]], axis=-1)
        u = np.clip((u - c["mean"]) / c["std"], -8.0, 8.0)
        columns = [np.ones(len(u))]
        for degree in range(1, self._correction_degree + 1):
            for combo in combinations_with_replacement(range(u.shape[1]), degree):
                columns.append(np.prod(u[:, list(combo)], axis=1))
        return np.stack(columns, axis=1)

    def _tau(self, x, frequency=None):
        """Time to merger (s) from ``frequency`` (default: the reference
        frequency), with the chirp mass in the frame of the Earth: 2PN, plus
        the fitted waveform correction at the reference frequency."""
        params = self._params(x)
        tau = self._tau_2pn(
            params, self.reference_frequency if frequency is None else frequency
        )
        if frequency is None and getattr(self, "_correction", None) is not None:
            tau = tau + self._features(params) @ self._correction["beta"]
        return tau

    def _waveform(self):
        """The ``time_to_merger`` function, imported if given by name; None
        if there is none or it cannot be imported (then 2PN)."""
        spec = getattr(self, "_time_to_merger", None)
        if spec is None:
            return None
        if getattr(self, "_time_to_merger_fn", None) is None:
            if callable(spec):
                self._time_to_merger_fn = spec
            else:
                module, _, name = str(spec).partition(":")
                try:
                    self._time_to_merger_fn = getattr(
                        importlib.import_module(module), name
                    )
                except (ImportError, AttributeError) as exc:
                    logger.warning(
                        "Cannot import time_to_merger %r (%s): the arrival "
                        "time keeps its last correction", spec, exc,
                    )
                    self._time_to_merger = None
                    return None
        return self._time_to_merger_fn

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
        training points); with ``time_to_merger``, refit the correction to
        2PN there; with any reference frequency, recentre ``tau_ref`` on
        them."""
        if self.reference_frequency is None or len(x) < 2:
            return
        t_det = (x["geocent_time"] - self._reference_time) + self._delay(x)
        params = self._params(x)
        waveform = self._waveform()
        if self.adaptive:
            low, high = self.frequency_range
            freqs = np.geomspace(low, high, 97)
            tau_grid = None
            if waveform is not None:
                tau_grid = np.asarray(waveform(params, freqs), dtype=float)
            if tau_grid is None or not np.isfinite(tau_grid).any():
                tau_grid = np.stack(
                    [self._tau_2pn(params, f) for f in freqs], axis=1
                )
            with np.errstate(invalid="ignore"):
                spreads = np.nanstd(t_det[:, None] - tau_grid, axis=0)
            if np.isfinite(spreads).any():
                i = int(np.nanargmin(spreads))
                # refine between the neighbouring grid points
                fine = np.geomspace(freqs[max(i - 1, 0)],
                                    freqs[min(i + 1, len(freqs) - 1)], 17)
                if waveform is not None:
                    fine_grid = np.asarray(waveform(params, fine), dtype=float)
                else:
                    fine_grid = np.stack(
                        [self._tau_2pn(params, f) for f in fine], axis=1
                    )
                with np.errstate(invalid="ignore"):
                    fine_spreads = np.nanstd(t_det[:, None] - fine_grid, axis=0)
                j = int(np.nanargmin(fine_spreads))
                previous = self.reference_frequency
                self.reference_frequency = float(fine[j])
                logger.info(
                    "Arrival time measured at %.1f Hz (was %.1f Hz): spread "
                    "over the training points %.3g ms, %.3g ms at merger",
                    self.reference_frequency, previous,
                    1e3 * fine_spreads[j], 1e3 * np.std(t_det),
                )
        if waveform is not None:
            self._fit_correction(params, waveform)
        tau = self._tau(x)
        finite = np.isfinite(tau)
        if finite.any():
            self._tau_ref = float(np.median(tau[finite]))

    def _fit_correction(self, params, waveform):
        """Least-squares fit of ``tau_waveform - tau_2PN`` at the reference
        frequency, a polynomial in the standardised intrinsic parameters."""
        f = self.reference_frequency
        target = (np.asarray(waveform(params, np.array([f])), dtype=float)[:, 0]
                  - self._tau_2pn(params, f))
        ok = np.isfinite(target)
        keys = [k for k in ("chirp_mass", "mass_ratio", "chi_1", "chi_2",
                            "lambda_1", "lambda_2")
                if np.std(params[k][ok]) > 0]
        n_terms = len(list(combinations_with_replacement(
            range(len(keys) + 1), self._correction_degree)))
        if ok.sum() < 2 * n_terms:
            logger.warning("Too few points (%d) to fit the arrival-time "
                           "correction; keeping 2PN", ok.sum())
            self._correction = None
            return
        u = np.stack([params[k][ok] for k in keys], axis=-1)
        self._correction = dict(keys=keys, mean=u.mean(axis=0),
                                std=u.std(axis=0), beta=None)
        design = self._features({k: v[ok] for k, v in params.items()})
        beta, *_ = np.linalg.lstsq(design, target[ok], rcond=None)
        self._correction["beta"] = beta
        residual = target[ok] - design @ beta
        logger.info(
            "Arrival-time correction to 2PN at %.1f Hz from the waveform: "
            "spread %.3g ms, fit residual %.3g ms (%d terms)", f,
            1e3 * np.std(target[ok]), 1e3 * np.std(residual), len(beta),
        )

    def reset(self):
        self._tau_ref = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # the imported function goes with its name, not by value
        state["_time_to_merger_fn"] = None
        if callable(state.get("_time_to_merger")):
            fn = state["_time_to_merger"]
            state["_time_to_merger"] = f"{fn.__module__}:{fn.__qualname__}"
        return state

    def __setstate__(self, state):
        # checkpoints pickled before the reference frequency existed
        for key, value in (
            ("reference_frequency", None), ("adaptive", False),
            ("frequency_range", (8.0, 300.0)), ("doppler_vector", None),
            ("_chirp_mass", "chirp_mass"), ("_mass_ratio", "mass_ratio"),
            ("_spins", ("chi_1", "chi_2")), ("_tau_ref", None),
            ("_time_to_merger", None), ("_time_to_merger_fn", None),
            ("_tides", ("lambda_1", "lambda_2")), ("_correction_degree", 2),
            ("_correction", None),
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
