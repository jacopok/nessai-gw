import importlib
import inspect
from itertools import combinations_with_replacement

import numpy as np
from nessai.reparameterisations import (
    Reparameterisation,
)

from .. import nessai_logger
from .mass import doppler_factor

logger = nessai_logger.getChild(__name__)

_TWO_PI = 2.0 * np.pi


class DeltaPhaseReparameterisation(Reparameterisation):
    """Reparameterisation that converts phase to delta phase.

    The Jacobian determinant of this transformation is 1.

    Requires "psi" and "theta_jn".

    Parameters
    ----------
    parameters : Union[str, List[str]]
        Name(s) of the parameter(s).
    prior_bounds : Union[list, dict]
        Prior bounds for the parameters
    """

    def __init__(self, parameters=None, prior_bounds=None):
        super().__init__(parameters=parameters, prior_bounds=prior_bounds)
        self.requires = ["psi", "theta_jn"]
        self.prime_parameters = ["delta_phase"]

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        """
        Apply the reparameterisation to convert from x-space to x'-space.

        Parameters
        ----------
        x : structured array
            Array of inputs
        x_prime : structured array
            Array to be update
        log_j : array_like
            Log jacobian to be updated

        Returns
        -------
        x, x_prime : structured arrays
            Update version of the x and x_prime arrays
        log_j : array_like
            Updated log Jacobian determinant
        """
        x_prime[self.prime_parameters[0]] = (
            x[self.parameters[0]] + np.sign(np.cos(x["theta_jn"])) * x["psi"]
        )
        return x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        """
        Apply the reparameterisation to convert from x-space
        to x'-space

        Parameters
        ----------
        x : structured array
            Array
        x_prime : structured array
            Array to be update
        log_j : array_like
            Log jacobian to be updated

        Returns
        -------
        x, x_prime : structured arrays
            Update version of the x and x_prime arrays
        log_j : array_like
            Updated log Jacobian determinant
        """
        x[self.parameters[0]] = np.mod(
            x_prime[self.prime_parameters[0]]
            - np.sign(np.cos(x["theta_jn"])) * x["psi"],
            2 * np.pi,
        )
        return x, x_prime, log_j


class PolarisationPhaseReparameterisation(Reparameterisation):
    """Single ``delta_phase = phase + sign(cos theta_jn) * psi`` flow coordinate.

    For the dominant (2, 2) GW mode the extrinsic likelihood constrains only the
    combination ``delta_phase = phase + sign(cos theta_jn) * psi`` (``phase +
    psi`` face-on, ``phase - psi`` face-off); the orthogonal combination
    (``psi`` itself, carried separately as ``angle-pi``) is nearly flat.

    This reparameterisation replaces the ``phase`` ``angle-2pi`` **Cartesian
    pair** with a **single** coordinate ::

        delta_phase_prime =
            (((phase + sign(cos theta_jn) * psi) * scale) mod 2*pi) / pi - 1

    in ``[-1, 1)``.

    Dropping the pair removes the ``chi(2)`` auxiliary radius: with a
    well-measured ``delta_phase`` the pair's free radius turns the flow-facing
    marginal into a leptokurtic scale mixture (a radial "cusp" at the origin)
    that the single group-mixture base flow fits poorly -- offline ``v9``
    inspection measured ``delta_phase_x`` excess kurtosis ~+0.9 with ~12 % of
    the mass at radius < 0.5.  The single coordinate is instead a clean unimodal
    (or, for the 8-element group, ``pi``-periodic bimodal) bump.

    The coordinate is *not* wrapped for the flow -- it is a bounded ``[-1, 1)``
    linear coordinate -- but with the ``beta_f >= 0`` fold the canonical
    ``delta_phase`` sits well inside ``(0, 2*pi)`` (folded circular
    concentration ~0.9, mode ~1.9 rad) so the ``0`` / ``2*pi`` seam is a
    non-issue.  :class:`nessai_gw.group_mixture.PrimeSpaceTriangularGroupAction`
    still wraps modulo the period in prime space.

    The map ``phase -> delta_phase_prime`` has constant Jacobian
    ``|d delta_phase_prime / d phase| = scale / pi`` (``psi`` / ``theta_jn`` are
    held fixed -- the group action is what transforms them).
    :class:`~nessai_gw.group_mixture.PrimeSpaceTriangularGroupAction` recomputes
    ``delta_phase`` from the group-transformed ``psi`` and ``cos theta_jn`` on
    the encode side (``sign`` flips and ``psi -> pi - psi`` under a reflection).

    Requires ``psi`` and ``theta_jn`` on both the forward pass (to build
    ``delta_phase``) and the inverse (to recover ``phase``).

    Parameters
    ----------
    parameters : Union[str, List[str]]
        Name of the parameter; must be ``phase``.
    prior_bounds : Union[list, dict], optional
        Prior bounds for ``phase``.  Unused -- the coordinate is always rescaled
        from ``[0, 2*pi)``.
    scale : float, optional
        ``1.0`` (default) -> period ``2*pi``, an exact bijection.  ``2.0``
        (period ``pi``) folds out the ``phase -> phase + pi`` (2, 2)-mode
        degeneracy for a tighter target, at the cost of a ``2 -> 1`` map; prefer
        instead a 16-element group action
        (:class:`nessai_gw.group_mixture.ETTriangleGroupAction` with
        ``phase_reflection=True``), which folds ``phase <-> phase + pi`` while
        this coordinate stays a clean bijection.
    waveform_phase : callable or str, optional
        ``f(params, frequencies) -> phi_22`` of shape
        ``(N, len(frequencies))``, the (unwrapped) phase of the waveform's 22
        mode, in the convention where ``phase`` enters the 22 mode as
        ``2 phase + phi_22``; ``params`` as for
        :class:`~nessai_gw.reparameterisations.DetectorCenterTimeReparameterisation`'s
        ``time_to_merger`` (chirp mass in the Earth's frame).  A
        ``"module:function"`` string is imported when first needed and is
        what is pickled.  With it the coordinate is measured at a reference
        frequency (see the class notes); ``None`` (default): at the merger.
    frequency_range : tuple of float, optional
        Search range (Hz) for the reference frequency of the phase.
    correction_degree : int, optional
        Degree of the polynomial fit of the phase offset (default 4).
    doppler_vector : array_like, optional
        :math:`\\mathbf{v}_\\oplus / c` in the axes of ``(ra, dec)``, as for
        ``doppler-chirp-mass``, for the chirp mass in the Earth's frame.
    chirp_mass, mass_ratio : str, optional
        Names of the chirp mass and mass ratio.
    spins, tides : tuple of str or None, optional
        Names of the aligned spins and tidal deformabilities of the heavier
        and lighter mass (``None``: zero).
    prior : optional
        Accepted for registry compatibility and ignored (the flow models the
        prime coordinate directly).

    Notes
    -----
    **Phase at a reference frequency.**  ``phase`` is the orbital phase at
    the merger.  What the data measure is the 22 mode's phase over the band
    the detector is most sensitive to: given the arrival time, a change of
    the masses, spins or tides changes the phase accumulated between that
    band and the merger, and ``phase`` moves to compensate -- by tens of
    radians across a high-SNR posterior, which winds ``delta_phase`` into a
    helix around the intrinsic parameters that the flow cannot follow (on
    the ET-2L v6 live points the circular concentration ``R`` of
    ``delta_phase`` is 0.02; with the offset below it is 0.57).  With
    ``waveform_phase`` the coordinate is ::

        delta_phase = phase + sign(cos theta_jn) * psi + a(theta) / 2

    with ``a`` the intercept at ``f = 0`` of the tangent to ``phi_22`` at a
    reference frequency ``f_p`` (the phase there, with its slope -- the
    time, carried by ``t_det`` -- removed).  At every :meth:`update` (on the
    training points) ``f_p`` is the one in ``frequency_range`` that makes
    ``delta_phase`` the most concentrated, and ``a`` is fitted there as a
    polynomial in the standardised intrinsic parameters (the waveform is
    far too slow for every flow draw).  The offset depends only on the
    intrinsic parameters, so the Jacobian is unchanged; it is invariant
    under the triangular group (with the chirp mass in the Earth's frame,
    as for the time), which translates the phase by constants.  The
    intrinsic parameters are read on the inverse pass, so this has to be
    ranked ahead of their reparameterisations.  If no frequency
    concentrates ``delta_phase`` more than the merger phase does, beyond the
    noise (early in a run), the offset is left out.
    """

    one_to_one = False

    def __init__(
        self,
        parameters=None,
        prior_bounds=None,
        scale=1.0,
        waveform_phase=None,
        frequency_range=(8.0, 300.0),
        correction_degree=4,
        doppler_vector=None,
        chirp_mass="chirp_mass",
        mass_ratio="mass_ratio",
        spins=("chi_1", "chi_2"),
        tides=("lambda_1", "lambda_2"),
        prior=None,
        rng=None,
        **kwargs,
    ):
        # Only forward kwargs the installed nessai ``Reparameterisation`` accepts
        # (0.15.x: ``parameters``, ``prior_bounds``, ``rng``; newer versions also
        # ``input_parameters`` / ``output_parameters``).
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

        if self.parameters != ["phase"]:
            raise RuntimeError(
                "PolarisationPhaseReparameterisation must act on 'phase'; got "
                f"{self.parameters}"
            )
        self.scale = float(scale)
        # single flow-facing coordinate
        self.prime_parameters = ["delta_phase"]
        if hasattr(self, "output_parameters"):
            self.output_parameters = ["delta_phase"]
        # psi and theta_jn are needed both to build delta_phase (forward) and to
        # recover phase (inverse).
        self.requires = ["psi", "theta_jn"]
        self._waveform_phase = waveform_phase
        self._waveform_phase_fn = None
        self.frequency_range = tuple(float(f) for f in frequency_range)
        if len(self.frequency_range) != 2 or not (
            0 < self.frequency_range[0] < self.frequency_range[1]
        ):
            raise ValueError(
                f"`frequency_range` must be (low, high) with 0 < low < high; "
                f"got {frequency_range}"
            )
        self._correction_degree = int(correction_degree)
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
        self._tides = None if tides is None else tuple(tides)
        for key, names in (("spins", self._spins), ("tides", self._tides)):
            if names is not None and len(names) != 2:
                raise ValueError(f"`{key}` must name two parameters; got {names}")
        #: frequency (Hz) the phase is measured at; None: the merger
        self.reference_frequency = None
        self._offset = None
        if waveform_phase is not None:
            self.requires += [chirp_mass, mass_ratio]
            self.requires += list(self._spins or []) + list(self._tides or [])
            if self.doppler_vector is not None:
                self.requires += ["ra", "dec"]
        if hasattr(self, "inverse_input_parameters"):
            self.inverse_input_parameters = list(
                dict.fromkeys(
                    list(self.inverse_input_parameters or []) + self.requires
                )
            )
        self._log_j = float(np.log(self.scale / np.pi))

    @staticmethod
    def _psi_sign(x):
        """``sign(cos theta_jn)`` (0 exactly at edge-on -- a measure-zero set)."""
        return np.sign(np.cos(x["theta_jn"]))

    def _waveform(self):
        """The ``waveform_phase`` function, imported if given by name; None
        if there is none or it cannot be imported."""
        spec = getattr(self, "_waveform_phase", None)
        if spec is None:
            return None
        if getattr(self, "_waveform_phase_fn", None) is None:
            if callable(spec):
                self._waveform_phase_fn = spec
            else:
                module, _, name = str(spec).partition(":")
                try:
                    self._waveform_phase_fn = getattr(
                        importlib.import_module(module), name
                    )
                except (ImportError, AttributeError) as exc:
                    logger.warning(
                        "Cannot import waveform_phase %r (%s): the phase "
                        "keeps its last offset", spec, exc,
                    )
                    self._waveform_phase = None
                    return None
        return self._waveform_phase_fn

    def _params(self, x):
        """The intrinsic parameters the offset depends on, as a dict of
        arrays (the chirp mass in the Earth's frame)."""
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

    def _tangent_intercepts(self, params, frequencies, step=1e-3):
        """``phi_22(f) - f phi_22'(f)`` at each frequency, shape
        ``(N, len(frequencies))`` (central differences)."""
        f = np.asarray(frequencies, dtype=float)
        grid = np.concatenate([f * (1 - step), f, f * (1 + step)])
        phi = np.asarray(self._waveform()(params, grid), dtype=float)
        lo, mid, hi = np.split(phi, 3, axis=1)
        return mid - (hi - lo) / (2 * step)

    def _features(self, params):
        """Polynomial features of the standardised intrinsic parameters
        (clipped, so that the tails of the flow stay finite and smooth)."""
        c = self._offset
        u = np.stack([params[k] for k in c["keys"]], axis=-1)
        u = np.clip((u - c["mean"]) / c["std"], -8.0, 8.0)
        columns = [np.ones(len(u))]
        for degree in range(1, self._correction_degree + 1):
            for combo in combinations_with_replacement(range(u.shape[1]), degree):
                columns.append(np.prod(u[:, list(combo)], axis=1))
        return np.stack(columns, axis=1)

    def _shift(self, x):
        """``a(theta) / 2``: what is added to the phase."""
        if getattr(self, "_offset", None) is None:
            return 0.0
        return 0.5 * (self._features(self._params(x)) @ self._offset["beta"])

    def _concentration(self, angle):
        """Circular concentration of ``delta_phase`` at the first two
        harmonics of its period (the group mixture can leave it
        ``pi``-periodic), the larger of the two."""
        angle = np.asarray(angle) * self.scale
        return np.maximum.reduce([
            np.abs(np.mean(np.exp(1j * k * angle), axis=0)) for k in (1, 2)
        ])

    def update(self, x, x_prime=None):
        """With ``waveform_phase``, choose the reference frequency of the
        phase and refit its offset on ``x`` (the training points)."""
        if self._waveform() is None or len(x) < 2:
            return
        params = self._params(x)
        base = x["phase"] + self._psi_sign(x) * x["psi"]
        low, high = self.frequency_range
        freqs = np.geomspace(low, high, 33)
        offsets = self._tangent_intercepts(params, freqs)
        ok = np.isfinite(offsets).all(axis=0)
        r_merger = float(self._concentration(base))
        if not ok.any():
            logger.warning("waveform_phase gave no finite phases: phase "
                           "measured at the merger")
            self.reference_frequency, self._offset = None, None
            return
        r = np.where(ok, self._concentration(base[:, None] + 0.5 * offsets), -1.0)
        i = int(np.argmax(r))
        keys = [k for k in ("chirp_mass", "mass_ratio", "chi_1", "chi_2",
                            "lambda_1", "lambda_2") if np.std(params[k]) > 0]
        n_terms = len(list(combinations_with_replacement(
            range(len(keys) + 1), self._correction_degree)))
        # a gain within the noise of the best of many frequencies is chance
        # (uniform phases: 2 N R^2 ~ chi^2_2, so ~e^-16 per frequency)
        if r[i] <= r_merger + 4.0 / np.sqrt(len(x)) or len(x) < 2 * n_terms:
            logger.info(
                "Phase measured at the merger (concentration %.3f; best %.3f "
                "at %.1f Hz, %d points for %d terms)", r_merger, r[i],
                freqs[i], len(x), n_terms,
            )
            self.reference_frequency, self._offset = None, None
            return
        target = offsets[:, i]
        u = np.stack([params[k] for k in keys], axis=-1)
        self._offset = dict(keys=keys, mean=u.mean(axis=0), std=u.std(axis=0),
                            beta=None)
        design = self._features(params)
        beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        self._offset["beta"] = beta
        self.reference_frequency = float(freqs[i])
        residual = target - design @ beta
        logger.info(
            "Phase measured at %.1f Hz: offset spread %.3g rad, fit residual "
            "%.3g rad (%d terms); concentration of delta_phase %.3f (%.3f at "
            "the merger)", self.reference_frequency, np.std(target),
            np.std(residual), len(beta),
            self._concentration(base + self._shift(x)), r_merger,
        )

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        angle = np.mod(
            (x["phase"] + self._psi_sign(x) * x["psi"] + self._shift(x))
            * self.scale, _TWO_PI
        )
        x_prime[self.prime_parameters[0]] = angle / np.pi - 1.0
        return x, x_prime, log_j + self._log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        angle = np.mod(
            (x_prime[self.prime_parameters[0]] + 1.0) * np.pi, _TWO_PI
        )
        x["phase"] = np.mod(
            angle / self.scale - self._psi_sign(x) * x["psi"] - self._shift(x),
            _TWO_PI,
        )
        return x, x_prime, log_j - self._log_j

    def __getstate__(self):
        state = self.__dict__.copy()
        # the imported function goes with its name, not by value
        state["_waveform_phase_fn"] = None
        if callable(state.get("_waveform_phase")):
            fn = state["_waveform_phase"]
            state["_waveform_phase"] = f"{fn.__module__}:{fn.__qualname__}"
        return state

    def __setstate__(self, state):
        # checkpoints pickled before the reference-frequency phase existed
        for key, value in (
            ("_waveform_phase", None), ("_waveform_phase_fn", None),
            ("frequency_range", (8.0, 300.0)), ("_correction_degree", 4),
            ("doppler_vector", None), ("_chirp_mass", "chirp_mass"),
            ("_mass_ratio", "mass_ratio"), ("_spins", ("chi_1", "chi_2")),
            ("_tides", ("lambda_1", "lambda_2")),
            ("reference_frequency", None), ("_offset", None),
        ):
            state.setdefault(key, value)
        self.__dict__.update(state)


class ArgAlphaBetaReparameterisation(Reparameterisation):
    r"""Polarisation/phase block as the two circular-polarisation phases.

    For the dominant (2, 2) mode the strain of a triangular detector in the
    frozen long-wavelength limit is ``h = alpha Z(n) + beta conj(Z(n))`` with

        alpha = c_+ e^{-2i psi} e^{2i phi_c} / d_L ,
        beta  = c_- e^{+2i psi} e^{2i phi_c} / d_L ,

    so the *phases* the extrinsic likelihood actually constrains are

        arg(alpha) = 2 phi_c - 2 psi ,      arg(beta) = 2 phi_c + 2 psi .

    Half of each -- ``(phase - psi)`` and ``(phase + psi)`` -- is a tight ridge
    coordinate (folded circular std ~0.4-0.75 rad on the octomodal ET BNS
    posterior, vs ~2.5 rad for ``delta_phase = phase + sign(cos theta_jn) psi``
    and ~2.3 rad for bare ``psi``).  Putting *both* circular-polarisation phases
    on the flow axes (rather than ``delta_phase`` + a separate broad ``psi``)
    aligns the coordinate grid with the likelihood ridge -- a flow-fit study on
    the folded octomodal posterior cut NLL by ~1 nat relative to the
    equivalently-folded ``(psi, delta_phase)`` block, with ``(psi, phi_c)`` on
    the axes ("Poincare" chart) giving no gain.

    This reparameterisation replaces **both** the ``psi`` (``angle-pi`` /
    ``SingleAngle``) and ``phase`` (``polarisation-phase``) flow coordinates with
    the single bounded pair ::

        arg_alpha = ((phase - psi)  mod 2*pi) / pi - 1   in [-1, 1)
        arg_beta  = ((phase + psi)  mod 2*pi) / pi - 1   in [-1, 1)

    On ``psi in [0, pi)``, ``phase in [0, 2*pi)`` this map is an exact
    **bijection** (a shear + scale of the ``(psi, phase)`` fundamental cell onto
    the ``[-1, 1)^2`` square), constant Jacobian
    ``|d(arg_alpha, arg_beta) / d(psi, phase)| = 2 / pi**2``.  It does *not* fold
    the ``phi_c -> phi_c + pi/2`` polarisation-quarter degeneracy -- that stays a
    weight-learned factor of the group mixture
    (:class:`~nessai_gw.group_mixture.TriangularDetectorGroupAction` with
    ``polarisation_quarter=True``), on which the group acts here by half-integer
    shifts of ``arg_alpha`` / ``arg_beta`` (and an ``alpha <-> beta`` swap under
    the plane reflection).

    Requires nothing beyond ``psi`` and ``phase`` themselves (independent of the
    inclination, unlike ``polarisation-phase``).  The inverse returns the
    representative with ``psi in [0, pi)``.

    Parameters
    ----------
    parameters : list of str
        Must be ``["psi", "phase"]`` (in any order).
    prior_bounds : list or dict, optional
        Unused -- the coordinates are always rescaled from ``[0, 2*pi)``.
    prior : optional
        Accepted for registry compatibility and ignored.
    """

    one_to_one = False

    def __init__(
        self,
        parameters=None,
        prior_bounds=None,
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

        if set(self.parameters) != {"psi", "phase"}:
            raise RuntimeError(
                "ArgAlphaBetaReparameterisation must act on ['psi', 'phase']; "
                f"got {self.parameters}"
            )
        # deterministic coordinate order regardless of the input order.  On
        # 0.15.x ``parameters`` is a writable attribute; on newer nessai it is a
        # read-only alias for ``input_parameters`` -- set whichever exists.
        if isinstance(
            getattr(type(self), "parameters", None), property
        ):
            self.input_parameters = ["psi", "phase"]
        else:
            self.parameters = ["psi", "phase"]
        self.prime_parameters = ["arg_alpha", "arg_beta"]
        if hasattr(self, "output_parameters"):
            self.output_parameters = list(self.prime_parameters)
        # 2 / pi**2 = (1/pi) * (1/pi) * |det [[-1, 1], [1, 1]]|
        self._log_j = float(np.log(2.0) - 2.0 * np.log(np.pi))

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        psi = x["psi"]
        phase = x["phase"]
        au = np.mod(phase - psi, _TWO_PI)
        av = np.mod(phase + psi, _TWO_PI)
        x_prime[self.prime_parameters[0]] = au / np.pi - 1.0
        x_prime[self.prime_parameters[1]] = av / np.pi - 1.0
        return x, x_prime, log_j + self._log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        au = np.mod((x_prime[self.prime_parameters[0]] + 1.0) * np.pi, _TWO_PI)
        av = np.mod((x_prime[self.prime_parameters[1]] + 1.0) * np.pi, _TWO_PI)
        # au = phase - psi, av = phase + psi (both mod 2 pi).  psi is fixed mod
        # pi; phase is fixed only mod pi, so pick the [0, 2 pi) branch that
        # reproduces au.
        psi = np.mod(0.5 * (av - au), np.pi)
        phase = np.mod(0.5 * (av + au), np.pi)
        e0 = np.cos(np.mod(phase - psi, _TWO_PI) - au)
        e1 = np.cos(np.mod(phase + np.pi - psi, _TWO_PI) - au)
        x["phase"] = np.where(e1 > e0, np.mod(phase + np.pi, _TWO_PI), phase)
        x["psi"] = psi
        return x, x_prime, log_j - self._log_j


class FittedPhaseRotation(Reparameterisation):
    r"""A fitted, branch-local rotation of ``(psi, phase)`` -- decorrelates
    what :class:`ArgAlphaBetaReparameterisation`'s fixed 45-degree shear and
    the identity (plain ``psi``, ``phase``) both leave correlated.

    Neither of the two "natural" bases is generically the right one: the
    correlation between the flow-facing coordinates ``ArgAlphaBetaReparameterisation``
    builds is set by the *local* ratio ``c_-/c_+ = ((1-cos iota)/(1+cos
    iota))**2``, which is small near circular polarisation and of order 1
    near the plateau -- so the 45-degree shear undershoots the right angle
    for one regime and overshoots it for the other, and plain ``(psi,
    phase)`` (a 0-degree "rotation") fares no better.  Measured on the
    ET-Delta v70 plateau/clump split
    (:func:`nessai_gw._polarisation_branch.fit_distance_floor_split`): raw
    ``corr(psi, phase)`` is -0.55 (plateau) and +0.76 (clump); a fixed
    45-degree shear leaves correlations of 0.59 and 0.86 respectively.  The
    angle that actually diagonalises each branch's covariance
    (:func:`nessai_gw._polarisation_branch.fit_phase_rotation`) is 111
    degrees for the plateau and 73 degrees for the clump -- neither the 0
    nor the 45 built into the other two options, and different from each
    other, which is why one fixed choice cannot serve both branches.

    This applies that fitted rotation directly, about a fitted centre ::

        p1 = (phase - phase0) cos(angle) - (psi - psi0) sin(angle)
        p2 = (phase - phase0) sin(angle) + (psi - psi0) cos(angle)

    A pure rotation (plus a constant shift) has unit Jacobian determinant, so
    ``log_j`` is unchanged.  Unlike :class:`ArgAlphaBetaReparameterisation`
    and :class:`SingleAngleReparameterisation`, ``p1``/``p2`` are **not**
    wrapped to a period -- ``angle``, ``psi0``, ``phase0`` are fit once from a
    single branch's own live points (via
    :func:`~nessai_gw._polarisation_branch.fit_phase_rotation`), whose
    ``(psi, phase)`` occupy a bounded sub-region that does not itself wrap, so
    there is nothing to fold.  This is a *branch-local* coordinate: it is
    only meaningful for live points already restricted to the branch it was
    fit on (see the plateau/clump split), not as a global replacement for
    ``ArgAlphaBetaReparameterisation``.

    Diagonalising the covariance does **not** remove every feature of the
    ``(psi, phase)`` corner: on both branches the rotated coordinates still
    show two parallel, offset ridges rather than one blob (a real, roughly
    ``phase -> phase + pi/2``-shaped discrete near-degeneracy this
    reparameterisation was not built to fold) -- see the module docstring of
    :mod:`nessai_gw._polarisation_branch`. Rotating removes the *correlation*
    a suboptimal linear basis introduces; it does not fold an additional
    discrete symmetry, which is a different problem.

    Parameters
    ----------
    parameters : list of str
        Must be ``["psi", "phase"]`` (in any order).
    prior_bounds : list or dict, optional
        Unused.
    angle : float
        Rotation angle in radians, e.g. from
        :func:`~nessai_gw._polarisation_branch.fit_phase_rotation`.
    psi0, phase0 : float
        Centre the rotation is taken about (typically each branch's own
        mean).
    prior : optional
        Accepted for registry compatibility and ignored.
    """

    one_to_one = False

    def __init__(
        self,
        parameters=None,
        prior_bounds=None,
        angle=0.0,
        psi0=0.0,
        phase0=0.0,
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

        if set(self.parameters) != {"psi", "phase"}:
            raise RuntimeError(
                "FittedPhaseRotation must act on ['psi', 'phase']; got "
                f"{self.parameters}"
            )
        if isinstance(
            getattr(type(self), "parameters", None), property
        ):
            self.input_parameters = ["psi", "phase"]
        else:
            self.parameters = ["psi", "phase"]
        self.prime_parameters = ["phase_rot_1", "phase_rot_2"]
        if hasattr(self, "output_parameters"):
            self.output_parameters = list(self.prime_parameters)
        self.angle = float(angle)
        self.psi0 = float(psi0)
        self.phase0 = float(phase0)
        self._cos, self._sin = np.cos(self.angle), np.sin(self.angle)

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        dpsi = x["psi"] - self.psi0
        dphase = x["phase"] - self.phase0
        x_prime[self.prime_parameters[0]] = dphase * self._cos - dpsi * self._sin
        x_prime[self.prime_parameters[1]] = dphase * self._sin + dpsi * self._cos
        return x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        p1 = x_prime[self.prime_parameters[0]]
        p2 = x_prime[self.prime_parameters[1]]
        dphase = p1 * self._cos + p2 * self._sin
        dpsi = -p1 * self._sin + p2 * self._cos
        x["phase"] = dphase + self.phase0
        x["psi"] = dpsi + self.psi0
        return x, x_prime, log_j


class SingleAngleReparameterisation(Reparameterisation):
    r"""Map one periodic angle to a single bounded coordinate -- no radius.

    ``angle_prime = ((a * scale) mod 2*pi) / pi - 1``  in ``[-1, 1)``, with
    constant Jacobian ``|d angle_prime / d a| = scale / pi``.

    This is the ``Angle`` (``angle-pi`` / ``angle-2pi``) reparameterisation
    **without** the Cartesian ``(x, y)`` pair and its ``chi(2)`` auxiliary
    radius.  A well-measured angle through ``Angle`` becomes a leptokurtic
    scale mixture (the free radius mixes scales, giving an origin cusp the
    flow fits poorly); the single coordinate is a clean unimodal bump -- the
    same trade :class:`PolarisationPhaseReparameterisation` makes for the
    informative ``phase``/``psi`` combination.  Use it for the *orthogonal*
    combination (bare ``psi``), which is only weakly constrained.

    Parameters
    ----------
    parameters : str or list
        The angle name (e.g. ``"psi"``).  Exactly one.
    prior_bounds : list or dict
        Unused for the map itself (the coordinate is always rescaled from the
        angle's natural period); required by the framework.
    scale : float, optional
        ``2.0`` (default) -> period ``pi`` (``psi``); ``1.0`` -> period
        ``2*pi``.
    prior : optional
        Accepted for registry compatibility and ignored.
    """

    one_to_one = False
    requires_bounded_prior = True

    def __init__(
        self,
        parameters=None,
        prior_bounds=None,
        scale=2.0,
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

        if len(self.parameters) != 1:
            raise RuntimeError(
                "SingleAngleReparameterisation acts on exactly one angle; got "
                f"{self.parameters}"
            )
        self.scale = float(scale)
        base = self.parameters[0]
        self.prime_parameters = [f"{base}_prime"]
        self.output_parameters = [f"{base}_prime"]
        self._period = _TWO_PI / self.scale
        self._log_j = float(np.log(self.scale / np.pi))

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        angle = np.mod(x[self.parameters[0]] * self.scale, _TWO_PI)
        x_prime[self.prime_parameters[0]] = angle / np.pi - 1.0
        return x, x_prime, log_j + self._log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        angle = np.mod((x_prime[self.prime_parameters[0]] + 1.0) * np.pi, _TWO_PI)
        x[self.parameters[0]] = np.mod(angle / self.scale, self._period)
        return x, x_prime, log_j - self._log_j
