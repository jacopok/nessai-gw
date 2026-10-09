"""Chirp/effective-distance reparameterisation.

Roulet et al. 2022 (arXiv:2207.03508, Eqs. 9 and 18) replace ``luminosity_distance``
by ::

    chirp_distance = luminosity_distance / (chirp_mass ** (5/6) * |R_k0(iota, n, psi)|)

with ``R_k0`` the dominant-(2,2)-mode response (Eq. 9) at a fixed reference
("loudest") detector ``k0``.  This strips out the amplitude scaling that
inclination, sky position and polarisation angle impose through the antenna
pattern -- exactly the ``luminosity_distance`` <-> ``theta_jn`` correlation
that dominates a single-site triangular-detector posterior (``corr = 0.998``
on the ET-Delta live points, reduced to ``0.085`` by this coordinate; see
``xg_inference.replay_diagnostics.effective_distance_experiment``).

The geometry (``R_k``, choosing ``k0``) lives in :mod:`nessai_gw._effective_distance`
and is reused here unchanged; this module only wraps it as a
:class:`~nessai.reparameterisations.Reparameterisation`.  See that module's
docstring for why ``R_k0`` must be built on the *real* detector tensors (not
the idealised triangle), why ``k0 = argmax_k |R_k(fiducial)|`` is the right way
to pick the reference detector for the co-located ET-EMR sub-interferometers,
and why the coordinate is invariant under the group actions.

With ``boundary_inversion`` the prior edge of ``luminosity_distance`` the
live points pile up against (typically the upper one: a source-frame-uniform
prior grows towards ``d_max``) is reflected in the *physical* distance,
``d -> 2 d_max - d``: the flow is trained on the live points plus their
mirror copies (``chirp_distance = (2 d_max - d) / c``), so it sees no cliff
at the edge, and draws past it are folded back.  The edge sits at a
parameter-dependent ``chirp_distance = d_max / c``, so this cannot be done
by a fixed-bound inversion in the prime space; the reflection leaves ``c``
(hence the group invariance) untouched and has unit Jacobian.
"""

from __future__ import annotations

import inspect

import numpy as np
from nessai.reparameterisations import Reparameterisation

from .. import nessai_logger
from .._effective_distance import dominant_detector, response_R
from .._geometry import greenwich_mean_sidereal_time

logger = nessai_logger.getChild(__name__)


class ChirpDistanceReparameterisation(Reparameterisation):
    """``luminosity_distance -> chirp_distance`` (Roulet et al. 2022, Eq. 18).

    Parameters
    ----------
    parameters : str or list
        Must be ``luminosity_distance`` (or a single-element list containing
        it).
    prior_bounds : optional
        Accepted for registry compatibility; unused (this is a residual-style
        map like :class:`~nessai_gw.reparameterisations.PolarisationEllipseReparameterisation`,
        not a rescaling to a bounded prime range -- the group-mixture wrapper
        standardises the prime coordinate regardless).
    tensors : array_like
        ``(n_det, 3, 3)`` (or a single ``(3, 3)``) Earth-fixed detector
        tensors, e.g. :func:`nessai_gw.group_mixture.detector_tensors` or
        :data:`nessai_gw.group_mixture.ET_EMR_DETECTOR_TENSORS`.  These must be
        the real tensors: :func:`nessai_gw._ellipse.ideal_triangle_tensors`
        does not know the arms' in-plane orientation, which ``|R_k0|`` of a
        single sub-detector depends on.
    gmst, reference_time : float, optional
        Greenwich mean sidereal time (rad), or the geocentric GPS time it is
        computed from.  Exactly one is needed.
    fiducial : mapping, optional
        ``ra``, ``dec``, ``psi``, ``theta_jn`` of a reference signal (typically
        the injection or the maximum-likelihood point), used once at
        construction to pick the reference detector ``k0 = argmax_k |R_k|``.
        Required unless ``k0`` is given directly.
    k0 : int, optional
        The reference detector index directly, bypassing ``fiducial``.
    chirp_mass, theta_jn, ra, dec, psi : str, optional
        Names of the parameters ``R_k0`` depends on (defaults match the
        standard ``xg_inference`` parameter names).
    boundary_inversion : bool, optional
        Reflect the luminosity distance about the prior edge the live points
        pile up against (detected with :func:`nessai.utils.detect_edge` at
        every :meth:`update`, i.e. before every training), training on mirror
        copies and folding draws back.  Needs ``prior_bounds``.
    inversion_bounds : sequence of str, optional
        The edges that may be inverted, from ``"lower"``/``"upper"``
        (default ``("upper",)``: the lower one sits near zero distance).
    detect_edges_kwargs : dict, optional
        Extra keyword arguments for :func:`nessai.utils.detect_edge`.
    """

    one_to_one = False

    def __init__(
        self,
        parameters=None,
        prior_bounds=None,
        tensors=None,
        gmst=None,
        reference_time=None,
        fiducial=None,
        k0=None,
        chirp_mass="chirp_mass",
        theta_jn="theta_jn",
        ra="ra",
        dec="dec",
        psi="psi",
        prior=None,
        rng=None,
        boundary_inversion=False,
        inversion_bounds=("upper",),
        detect_edges_kwargs=None,
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

        if self.parameters != ["luminosity_distance"]:
            raise RuntimeError(
                "ChirpDistanceReparameterisation must act on "
                f"'luminosity_distance'; got {self.parameters}"
            )

        if tensors is None:
            raise ValueError(
                "ChirpDistanceReparameterisation requires the detector "
                "`tensors`."
            )
        tensors = np.asarray(tensors, dtype=float)
        if tensors.ndim == 2:
            tensors = tensors[None]
        if tensors.ndim != 3 or tensors.shape[1:] != (3, 3):
            raise ValueError(
                "`tensors` must have shape (n_det, 3, 3) or (3, 3); got "
                f"{tensors.shape}"
            )
        if (gmst is None) == (reference_time is None):
            raise ValueError(
                "ChirpDistanceReparameterisation requires exactly one of "
                "`gmst` and `reference_time`."
            )
        if gmst is None:
            gmst = greenwich_mean_sidereal_time(reference_time)
        self.tensors = tensors
        self.gmst = float(gmst)

        if k0 is None:
            if fiducial is None:
                raise ValueError(
                    "ChirpDistanceReparameterisation requires either `k0` "
                    "or `fiducial` (ra, dec, psi, theta_jn of a reference "
                    "signal) to pick the reference detector."
                )
            missing = {"ra", "dec", "psi", "theta_jn"} - set(fiducial)
            if missing:
                raise ValueError(f"`fiducial` is missing {sorted(missing)}.")
            k0, _ = dominant_detector(
                self.tensors,
                fiducial["ra"],
                fiducial["dec"],
                fiducial["psi"],
                fiducial["theta_jn"],
                self.gmst,
            )
        self.k0 = int(k0)
        if not 0 <= self.k0 < len(self.tensors):
            raise ValueError(
                f"k0={self.k0} is out of range for {len(self.tensors)} "
                "detector tensors."
            )
        logger.info(
            "ChirpDistanceReparameterisation: reference detector k0=%d",
            self.k0,
        )

        self._chirp_mass = chirp_mass
        self._theta_jn = theta_jn
        self._ra = ra
        self._dec = dec
        self._psi = psi
        self.requires = [chirp_mass, theta_jn, ra, dec, psi]

        self.boundary_inversion = bool(boundary_inversion)
        self.inversion_bounds = tuple(inversion_bounds)
        if not set(self.inversion_bounds) <= {"lower", "upper"}:
            raise ValueError(
                f"inversion_bounds must be from 'lower'/'upper'; got "
                f"{self.inversion_bounds}"
            )
        self.detect_edges_kwargs = dict(detect_edges_kwargs or {})
        self._edge = None
        if self.boundary_inversion:
            if prior_bounds is None:
                raise ValueError("boundary_inversion needs `prior_bounds`.")
            bounds = (
                prior_bounds["luminosity_distance"]
                if isinstance(prior_bounds, dict) else prior_bounds
            )
            self.distance_bounds = tuple(
                float(b) for b in np.asarray(bounds, dtype=float).ravel()
            )

        self.prime_parameters = ["chirp_distance"]
        if hasattr(self, "output_parameters"):
            self.output_parameters = ["chirp_distance"]
        if hasattr(self, "inverse_input_parameters"):
            self.inverse_input_parameters = list(
                dict.fromkeys(
                    list(self.inverse_input_parameters or []) + self.requires
                )
            )

    # ------------------------------------------------------------------
    def _abs_r(self, x):
        r = response_R(
            self.tensors,
            x[self._ra],
            x[self._dec],
            x[self._psi],
            x[self._theta_jn],
            self.gmst,
        )[..., self.k0]
        return np.abs(r)

    def _detect_edge(self, distance):
        from nessai.utils import detect_edge

        kwargs = dict(
            x_range=self.distance_bounds,
            allow_none=True,
            allowed_bounds=list(self.inversion_bounds),
        )
        kwargs.update(self.detect_edges_kwargs)
        edge = detect_edge(np.asarray(distance, dtype=float), **kwargs)
        if edge == "both":     # a single reflection: take the denser edge
            lo, hi = self.distance_bounds
            d = np.asarray(distance, dtype=float)
            edge = "upper" if np.median(d) > 0.5 * (lo + hi) else "lower"
        self._edge = edge or False
        logger.debug(
            "ChirpDistanceReparameterisation: distance inversion at %s edge",
            self._edge,
        )

    def _mirror(self, distance):
        lo, hi = self.distance_bounds
        return 2.0 * (hi if self._edge == "upper" else lo) - distance

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        c = x[self._chirp_mass] ** (5.0 / 6.0) * self._abs_r(x)
        x_prime[self.prime_parameters[0]] = x["luminosity_distance"] / c
        log_j = log_j - np.log(c)
        if self.boundary_inversion:
            if self._edge is None:
                self._detect_edge(x["luminosity_distance"])
            if self._edge:
                # mirror copies in a block after the originals (the
                # reflection has unit Jacobian, so log_j is shared)
                x_inv = x_prime.copy()
                x_inv[self.prime_parameters[0]] = (
                    self._mirror(x["luminosity_distance"]) / c
                )
                x_prime = np.concatenate([x_prime, x_inv])
                x = np.concatenate([x, x])
                log_j = np.concatenate([log_j, log_j])
        return x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        c = x[self._chirp_mass] ** (5.0 / 6.0) * self._abs_r(x)
        distance = x_prime[self.prime_parameters[0]] * c
        if self.boundary_inversion and self._edge:
            lo, hi = self.distance_bounds
            past = distance > hi if self._edge == "upper" else distance < lo
            distance = np.where(past, self._mirror(distance), distance)
        x["luminosity_distance"] = distance
        return x, x_prime, log_j + np.log(c)

    def update(self, x, x_prime=None):
        """Re-detect the inversion edge on the points about to be trained on."""
        if self.boundary_inversion:
            self._detect_edge(x["luminosity_distance"])

    def reset(self):
        self._edge = None

    def __setstate__(self, state):
        # checkpoints pickled before boundary inversion existed
        state.setdefault("boundary_inversion", False)
        state.setdefault("inversion_bounds", ("upper",))
        state.setdefault("detect_edges_kwargs", {})
        state.setdefault("_edge", None)
        self.__dict__.update(state)
