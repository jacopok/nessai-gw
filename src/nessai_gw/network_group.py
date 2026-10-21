"""Group-mixture proposal for a network of separated detectors.

:mod:`nessai_gw.group_mixture` is built around a single triangular site, whose
32-element group mixes a sky ``Z4 x Z2`` (the triangle's rotation / plane
reflection) with a polarisation/phase ``Z4``.  Only the latter survives for a
general network: the quarter turn

    g : psi -> psi + pi/2,   phase -> phase - pi/2

flips the sign of both antenna patterns ``F_+, F_x ~ (cos 2 psi, sin 2 psi)``
*and* of the ``m = 2`` mode ``~ e^{-2 i phase}``, so it leaves every
detector's (2, 2)-mode response unchanged whatever the detector geometry.
Its square ``g^2 : phase -> phase + pi`` is the familiar ``(2, 2)``
reflection.  Higher modes break it weakly, which the mixture absorbs in its
per-element weights (and importance sampling corrects exactly), just as on the
triangular path.  This module provides that 4-element group
(:class:`PolarisationPhaseGroupAction`, :class:`PrimeSpacePolarisationPhaseAction`)
together with the two frame choices a separated network wants:

* **time at the SNR-weighted barycentre.**  ``t_bar = geocent_time +
  delay(n; r_bar)`` with ``r_bar = sum_i w_i r_i``, ``w_i ~ rho_i**2``.
  ``delay`` is linear in the vertex, so ``t_bar = sum_i w_i t_i`` is the
  inverse-variance average of the per-detector arrival times (timing
  precision scales as ``1 / rho_i`` for detectors of similar bandwidth).  It
  is the ``detector-center-time`` reparameterisation with ``vertex = r_bar``.
* **sky aligned to the baseline** of the two highest-SNR sites.  Their
  arrival-time difference fixes the angle ``alpha`` to the baseline, so the
  posterior is a (piece of a) ring about it: ``cos alpha`` is sharp and all
  remaining structure sits on the azimuth ``beta`` about the baseline.  It is
  :class:`~nessai_gw.reparameterisations.sky.EqualAreaSky` with the baseline
  as the frame's ``z`` axis, ``sky_v = (1 - cos alpha) / 2`` and
  ``sky_u = beta / (2 pi)``.
* **chirp distance** at the loudest detector (Roulet et al. 2022,
  arXiv:2207.03508, Eq. 18; the ``chirp-distance`` reparameterisation):
  ``luminosity_distance / (chirp_mass^{5/6} |R_k0|)``, which divides out the
  antenna-pattern amplitude the reference detector sees and with it the
  ``luminosity_distance`` <-> ``theta_jn`` correlation.  ``|R_k0|`` is
  invariant under ``g`` (``F_+, F_x -> -F_+, -F_x``), so the group action
  passes the coordinate through untouched.  It needs the detector tensors
  (read by :meth:`DetectorNetworkGeometry.from_interferometers`).
* **effective spins** (Roulet et al. Sec. IV; the ``effective-spin``
  reparameterisation): aligned ``chi_1``, ``chi_2`` become
  ``chi_eff_prime`` (a function of ``chi_eff`` at fixed mass ratio) and
  ``chi_diff_prime`` (the prior-weighted ``cumchidiff``), both ``N(0, 1)``
  under the ``AlignedSpin`` prior, so a well-measured ``chi_eff`` is one axis
  instead of a curved ridge.  Spins, aligned or precessing (the
  ``precessing-spin`` reparameterisation), are invariant under ``g``, which
  only moves ``psi`` and ``phase``.

Co-located interferometers (the three of a triangular ET) are merged into one
*site* first, so the baseline always joins two distinct locations.  With a
single site there is no baseline and the sky falls back to the equatorial
equal-area frame; use :mod:`nessai_gw.group_mixture` for a lone triangle.

Measured on the ET 2L (Sardinia + Lusatia) injections ``ET_2L_MisA_v70-v108``
(``xg_inference.nongaussianity.twoL_frames``): relative to the physical frame,
detector time lowers the peak ``KL(p || Gaussian)`` of the constrained prior by
~3.4 nats (median), the baseline sky by another ~1.4, while the ``Angle`` /
``AnglePair`` Cartesian lifts *raise* it by 1-2 nats at low SNR and 8-12 nats
at SNR ~ 700.
"""

from __future__ import annotations

import functools
import inspect

import numpy as np
import torch

from . import nessai_logger
from ._geometry import greenwich_mean_sidereal_time
from .group_mixture import (
    _DELTA_PHASE_SCALE,
    _MIN_CANON_STD,
    AMPLITUDE_ARGUMENT_CIRCLE,
    POLARISATION_PHASE_CIRCLE,
    AdaptiveFundamentalDomain,
    CircularCanonicalTransform,
    SkyOctantProbit,
    _make_group_proposal_class,
    _prime_parameter_names,
    _seam_note,
    _select_flow_model_factory,
    _wrap_angle,
    curvature_shear_kwargs,
    detector_tensors,
    triangular_group_reparameterisations,
)

from .reparameterisations.network_amplitudes import (
    NETWORK_AMPLITUDE_ANGLES,
    NETWORK_AMPLITUDE_PRIME_PARAMETERS,
)

logger = nessai_logger.getChild(__name__)

_TWO_PI = 2.0 * np.pi

#: Size of the polarisation/phase group.
POLARISATION_PHASE_GROUP_SIZE = 4

#: Size of the optional baseline half-turn factor (see
#: :class:`PrimeSpacePolarisationPhaseAction`, ``baseline_flip``).
BASELINE_FLIP_GROUP_SIZE = 2

#: Interferometers closer than this (metres) are one site (ET's triangle arms
#: are ~10 km apart; the closest planned separated sites are ~1000 km).
DEFAULT_SITE_TOLERANCE = 100e3


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
class DetectorNetworkGeometry:
    """Timing barycentre and baseline sky frame of a detector network.

    Parameters
    ----------
    vertices : array_like
        ``(n, 3)`` Earth-fixed detector positions in metres.
    snrs : array_like
        ``(n,)`` per-detector SNRs (optimal or matched-filter; only their
        relative sizes matter).
    reference_time : float
        Geocentric GPS time of the event; its GMST rotates the Earth-fixed
        baseline frame to equatorial coordinates.
    names : list of str, optional
        Detector names, for logging.
    site_tolerance : float, optional
        Detectors within this distance (metres) of a site's loudest member are
        merged into that site (default :data:`DEFAULT_SITE_TOLERANCE`).
    azimuth_offset : float, optional
        Rotation (rad) of the baseline frame about the baseline, i.e. where the
        ``sky_u`` seam sits.  At ``0`` the seam (``beta = 0``) is the horizontal
        direction perpendicular to the baseline at its midpoint, where
        ground-based antenna patterns are weakest; ``beta = pi/2`` is the local
        vertical.
    tensors : array_like, optional
        ``(n, 3, 3)`` Earth-fixed detector tensors, needed for the
        ``chirp-distance`` reparameterisation (see
        :func:`network_group_reparameterisations`).

    Attributes
    ----------
    timing_vertex : numpy.ndarray
        ``sum_i w_i r_i`` with ``w_i = rho_i**2 / sum_j rho_j**2``.
    sites : list of dict
        ``{"vertex", "snr", "members"}`` per site, loudest first; a site's
        SNR is the quadrature sum of its members'.
    baseline : numpy.ndarray or None
        Unit vector from the second- to the loudest site, or ``None`` with a
        single site.
    sky_frame_rotation : numpy.ndarray
        ``(3, 3)`` equatorial -> sky-frame rotation (GMST folded in) for
        :class:`~nessai_gw.reparameterisations.sky.EqualAreaSky`.
    reference_detector : int
        Index of the loudest detector, the chirp distance's reference ``k0``
        (Roulet et al. sort the detectors by SNR).
    tensors : numpy.ndarray or None
        The ``(n, 3, 3)`` detector tensors, if given.
    """

    def __init__(
        self,
        vertices,
        snrs,
        reference_time,
        names=None,
        site_tolerance=DEFAULT_SITE_TOLERANCE,
        azimuth_offset=0.0,
        tensors=None,
    ):
        vertices = np.atleast_2d(np.asarray(vertices, dtype=float))
        snrs = np.atleast_1d(np.asarray(snrs, dtype=float))
        if vertices.shape[1] != 3 or len(vertices) != len(snrs):
            raise ValueError(
                f"need (n, 3) vertices and n SNRs; got {vertices.shape} and "
                f"{snrs.shape}"
            )
        if np.any(snrs < 0) or not np.all(np.isfinite(snrs)):
            raise ValueError(f"SNRs must be finite and non-negative; got {snrs}")
        if tensors is not None:
            tensors = np.asarray(tensors, dtype=float)
            if tensors.shape != (len(snrs), 3, 3):
                raise ValueError(
                    f"need ({len(snrs)}, 3, 3) detector tensors; got "
                    f"{tensors.shape}"
                )
        self.tensors = tensors
        self.reference_detector = int(np.argmax(snrs))
        self.vertices = vertices
        self.snrs = snrs
        self.names = (
            list(names) if names is not None
            else [f"det{i}" for i in range(len(snrs))]
        )
        self.reference_time = float(reference_time)
        self.gmst = greenwich_mean_sidereal_time(self.reference_time)
        self.azimuth_offset = float(azimuth_offset)

        w = snrs**2
        w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
        self.timing_weights = w
        self.timing_vertex = w @ vertices

        self.sites = self._group_sites(float(site_tolerance))
        if len(self.sites) >= 2:
            b = self.sites[0]["vertex"] - self.sites[1]["vertex"]
            self.baseline = b / np.linalg.norm(b)
        else:
            self.baseline = None
            logger.warning(
                "Single-site network (%s): no baseline, the sky stays in the "
                "equatorial equal-area frame.",
                ", ".join(self.names),
            )
        self.sky_frame_rotation = self._sky_frame_rotation()

    @classmethod
    def from_interferometers(cls, interferometers, reference_time, snrs=None,
                             **kwargs):
        """Build from bilby ``Interferometer`` objects.

        ``snrs`` may be a sequence (in interferometer order), a dict keyed by
        interferometer name, or ``None`` to read each interferometer's
        ``meta_data["optimal_SNR"]`` (set by bilby on injection).  The detector
        tensors are read too (for the chirp distance) unless ``tensors`` is
        passed.
        """
        ifos = list(interferometers)
        names = [ifo.name for ifo in ifos]
        vertices = [
            np.asarray(
                getattr(ifo, "geometry", ifo).vertex, dtype=float
            )
            for ifo in ifos
        ]
        if kwargs.get("tensors") is None:
            try:
                kwargs["tensors"] = detector_tensors(ifos)
            except AttributeError:
                kwargs["tensors"] = None
        if snrs is None:
            try:
                snrs = [float(ifo.meta_data["optimal_SNR"]) for ifo in ifos]
            except (AttributeError, KeyError) as exc:
                raise ValueError(
                    "no `snrs` given and the interferometers carry no "
                    "meta_data['optimal_SNR']; pass the per-detector SNRs."
                ) from exc
        elif isinstance(snrs, dict):
            snrs = [float(snrs[n]) for n in names]
        return cls(vertices, snrs, reference_time, names=names, **kwargs)

    def site_tensors(self):
        """``(2, 3, 3)`` tensors of the two loudest sites' detectors, for the
        ``network-amplitudes`` reparameterisation.  Each site must be a single
        interferometer: a co-located triangle measures two amplitudes."""
        if self.tensors is None or len(self.sites) < 2:
            raise ValueError(
                "network amplitudes need two sites and the detector tensors."
            )
        out = []
        for site in self.sites[:2]:
            if len(site["members"]) != 1:
                raise ValueError(
                    "network amplitudes need one interferometer per site; "
                    f"site {'+'.join(site['members'])} has "
                    f"{len(site['members'])}."
                )
            out.append(self.tensors[self.names.index(site["members"][0])])
        return np.stack(out)

    def _group_sites(self, tol):
        sites = []
        for i in np.argsort(-self.snrs, kind="stable"):
            for site in sites:
                if np.linalg.norm(self.vertices[i] - site["anchor"]) < tol:
                    site["members"].append(int(i))
                    break
            else:
                sites.append(
                    {"anchor": self.vertices[i], "members": [int(i)]}
                )
        out = []
        for site in sites:
            m = site["members"]
            w = self.snrs[m] ** 2
            w = w / w.sum() if w.sum() > 0 else np.full(len(m), 1.0 / len(m))
            out.append({
                "vertex": w @ self.vertices[m],
                "snr": float(np.sqrt(np.sum(self.snrs[m] ** 2))),
                "members": [self.names[j] for j in m],
            })
        return sorted(out, key=lambda s: -s["snr"])

    def _sky_frame_rotation(self):
        g = self.gmst
        # equatorial -> Earth-fixed (ra -> ra - gmst)
        rz = np.array(
            [[np.cos(g), np.sin(g), 0.0], [-np.sin(g), np.cos(g), 0.0],
             [0.0, 0.0, 1.0]]
        )
        if self.baseline is None:
            return np.eye(3)
        b = self.baseline
        up = self.sites[0]["vertex"] + self.sites[1]["vertex"]
        up = up - (up @ b) * b
        e2 = up / np.linalg.norm(up)
        e1 = np.cross(e2, b)
        if self.azimuth_offset:
            c, s = np.cos(self.azimuth_offset), np.sin(self.azimuth_offset)
            e1, e2 = c * e1 + s * e2, -s * e1 + c * e2
        return np.stack([e1, e2, b]) @ rz

    def __repr__(self):
        sites = "; ".join(
            f"{'+'.join(s['members'])} (SNR {s['snr']:.1f})" for s in self.sites
        )
        return f"DetectorNetworkGeometry({sites})"


# ---------------------------------------------------------------------------
# group actions
# ---------------------------------------------------------------------------
def _wrap(x, period):
    """``x mod period`` in ``[0, period)``, also when rounding hits ``period``."""
    y = torch.remainder(x, period)
    return torch.where(y >= period, y - period, y)


def _quarter_turns(modes, inverse, like):
    k = torch.as_tensor(modes, device=like.device)
    if inverse:
        k = torch.remainder(-k, POLARISATION_PHASE_GROUP_SIZE)
    return (0.5 * np.pi) * torch.remainder(k, 4).to(like.dtype)


def _sign_cos(cos_theta_jn):
    return torch.where(cos_theta_jn < 0, -1.0, 1.0).to(cos_theta_jn.dtype)


class PolarisationPhaseGroupAction:
    """The polarisation/phase ``Z4`` in physical coordinates.

    Mode ``k`` in ``0..3`` applies ``g^k``: ``psi -> psi + k pi/2 (mod pi)``,
    ``phase -> phase - k pi/2 (mod 2 pi)``; ``cos_theta_jn`` is unchanged and
    only enters the fundamental domain, which is taken on the flow coordinates
    ``psi`` and ``delta_phase = phase + sign(cos theta_jn) psi``::

        (psi - psi_offset) mod pi < pi/2   and
        (delta_phase - delta_offset) mod 2 pi < pi

    ``g`` moves ``psi`` by ``pi/2`` and ``g^2`` moves ``delta_phase`` by
    ``pi`` with ``psi`` fixed, so exactly one of the four images of a point
    satisfies both.  The action is a piecewise translation: unit Jacobian.
    """

    parameters = ("cos_theta_jn", "psi", "phase")
    group_size = POLARISATION_PHASE_GROUP_SIZE
    mode_factor_sizes = (POLARISATION_PHASE_GROUP_SIZE,)

    def __init__(self, psi_offset=0.0, delta_offset=0.0):
        self.psi_offset = float(psi_offset)
        self.delta_offset = float(delta_offset)

    def __call__(self, point_dict, modes, inverse=False):
        psi, phase = point_dict["psi"], point_dict["phase"]
        q = _quarter_turns(modes, inverse, psi)
        out = dict(point_dict)
        out["psi"] = _wrap(psi + q, np.pi)
        out["phase"] = _wrap(phase - q, _TWO_PI)
        return out

    def in_fundamental_domain(self, point_dict):
        psi = point_dict["psi"]
        delta = point_dict["phase"] + _sign_cos(point_dict["cos_theta_jn"]) * psi
        return (_wrap(psi - self.psi_offset, np.pi) < 0.5 * np.pi) & (
            _wrap(delta - self.delta_offset, _TWO_PI) < np.pi
        )


class PrimeSpacePolarisationPhaseAction:
    """:class:`PolarisationPhaseGroupAction` on the nessai-gw prime coordinates.

    Acts on ``psi_prime = (2 psi mod 2 pi) / pi - 1``
    (:class:`~nessai_gw.reparameterisations.phase.SingleAngleReparameterisation`)
    and ``delta_phase = (delta mod 2 pi) / pi - 1``
    (``polarisation-phase``), reading ``sign(cos theta_jn) = -sign(u)`` from the
    fixed-bounds ``angle-sine`` coordinate ``u = theta_jn_prime``.  Every other
    prime coordinate -- sky, barycentre time, distance, masses -- passes
    through untouched, so the map is a piecewise translation of two
    coordinates with unit Jacobian.

    The fundamental domain is ``psi_prime, delta_phase in [-1, 0)`` shifted by
    ``psi_offset`` / ``delta_offset`` (radians of ``psi`` / ``delta_phase``);
    prefer :class:`AdaptiveNetworkDomain`, which moves those seams off the
    posterior mass every training round.

    **Baseline half-turn** (``baseline_flip``).  The antipode is an exact
    degeneracy of every detector's antenna response:
    ``(n, theta_jn, psi, phase) -> (-n, pi - theta_jn, -psi, phase)`` leaves
    the response of each detector (all modes of an aligned-spin source)
    unchanged.  Only the arrival times break it, and with two sites they
    constrain ``n . b`` alone (``b`` the baseline), which the antipode flips.
    The image on the same delay ring nearest the antipode is the half-turn
    about the baseline, ``beta -> beta + pi``: it *is* the antipode at
    ``alpha = 90 deg`` and departs from it by ``2 |90 deg - alpha|``.  So
    a 2-site posterior grows a second mode half a turn round the ring with
    the inclination flipped, exactly degenerate while the ring is wide and
    dying as the timing pins ``alpha`` away from 90 deg (ET 2L
    mlgw_bns_2L_v4: two equal modes 180.7 deg apart at ``alpha ~ 80 deg``;
    v2, ``alpha ~ 55 deg``: the second mode ``Delta ln L ~ 7800`` below).
    The generator

        h : sky_u -> sky_u + 1/2,  theta_jn_prime -> -theta_jn_prime,
            psi -> -psi,  phase -> phase + pi

    (the antipodal map times ``g^2``; ``delta_phase`` is then unchanged
    off the ``psi = 0`` seam) commutes with the quarter turn ``g``, so the
    group becomes ``Z4 x Z2`` (8 elements, mode ``k + 4 f`` applying
    ``g^k h^f``) with factor sizes ``(4, 2)``.  The barycentre time is kept
    (the site offsets from the barycentre lie along ``b``, so both arrival
    times are kept), as is the chirp distance.  ``h`` is measure preserving
    (equal-area sky, ``sin theta_jn`` prior); the per-element weights absorb
    how far the data break it and importance sampling corrects the rest.
    The domain gains ``(sky_u - sky_u_offset) mod 1 < 1/2``.
    """

    def __init__(self, prime_names, psi_offset=0.0, delta_offset=0.0,
                 baseline_flip=False, sky_u_offset=0.0):
        self.psi_offset = float(psi_offset)
        self.delta_offset = float(delta_offset)
        self.baseline_flip = bool(baseline_flip)
        self.sky_u_offset = float(sky_u_offset)
        self.group_size = POLARISATION_PHASE_GROUP_SIZE * (
            BASELINE_FLIP_GROUP_SIZE if self.baseline_flip else 1
        )
        self.mode_factor_sizes = (
            (POLARISATION_PHASE_GROUP_SIZE, BASELINE_FLIP_GROUP_SIZE)
            if self.baseline_flip
            else (POLARISATION_PHASE_GROUP_SIZE,)
        )
        self.bind(prime_names)

    @property
    def sky_u_period(self):
        """Width of the fundamental domain in ``sky_u``."""
        return 0.5 if self.baseline_flip else 1.0

    def bind(self, prime_names):
        names = list(prime_names)
        self._prime_names = names
        self._delta = next(
            (c for c in ("delta_phase", "delta_phase_prime") if c in names),
            None,
        )
        missing = [
            n for n, ok in (
                ("psi_prime", "psi_prime" in names),
                ("delta_phase", self._delta is not None),
                ("theta_jn_prime", "theta_jn_prime" in names),
                ("sky_u", "sky_u" in names or not self.baseline_flip),
            ) if not ok
        ]
        if missing:
            raise RuntimeError(
                f"prime space is missing {missing}, which the polarisation/"
                "phase group action needs (single-angle psi, "
                "polarisation-phase phase, fixed-bounds angle-sine theta_jn"
                "; the baseline-frame sky_u with baseline_flip); "
                f"got {names}."
            )

    @staticmethod
    def _sign(theta_jn_prime, dtype):
        return torch.where(theta_jn_prime > 0, -1.0, 1.0).to(dtype)

    def _decode(self, point_dict):
        psi = _wrap((point_dict["psi_prime"] + 1.0) * (0.5 * np.pi), np.pi)
        delta = _wrap(
            (point_dict[self._delta] + 1.0) * np.pi / _DELTA_PHASE_SCALE,
            _TWO_PI,
        )
        sign = self._sign(point_dict["theta_jn_prime"], psi.dtype)
        return psi, delta - sign * psi, sign

    def __call__(self, point_dict, modes, inverse=False):
        psi, phase, sign = self._decode(point_dict)
        out = dict(point_dict)
        modes = torch.as_tensor(modes, device=psi.device)
        if self.baseline_flip:
            # g and h commute and h is an involution: the inverse of
            # g^k h^f is g^-k h^f, so h is applied first either way
            flip = torch.div(
                modes, POLARISATION_PHASE_GROUP_SIZE, rounding_mode="floor"
            ) % BASELINE_FLIP_GROUP_SIZE == 1
            theta = point_dict["theta_jn_prime"]
            out["theta_jn_prime"] = torch.where(flip, -theta, theta)
            u = point_dict["sky_u"]
            out["sky_u"] = torch.where(flip, _wrap(u + 0.5, 1.0), u)
            psi = torch.where(flip, _wrap(-psi, np.pi), psi)
            phase = torch.where(flip, _wrap(phase + np.pi, _TWO_PI), phase)
            sign = self._sign(out["theta_jn_prime"], psi.dtype)
            modes = torch.remainder(modes, POLARISATION_PHASE_GROUP_SIZE)
        q = _quarter_turns(modes, inverse, psi)
        psi_t = _wrap(psi + q, np.pi)
        delta_t = _wrap((phase - q + sign * psi_t) * _DELTA_PHASE_SCALE, _TWO_PI)
        out["psi_prime"] = _wrap(2.0 * psi_t, _TWO_PI) / np.pi - 1.0
        out[self._delta] = delta_t / np.pi - 1.0
        return {n: out[n] for n in self._prime_names}

    def in_fundamental_domain(self, point_dict):
        psi, phase, sign = self._decode(point_dict)
        delta = phase + sign * psi
        ok = (_wrap(psi - self.psi_offset, np.pi) < 0.5 * np.pi) & (
            _wrap(delta - self.delta_offset, _TWO_PI) < np.pi
        )
        if self.baseline_flip:
            ok = ok & (_wrap(point_dict["sky_u"] - self.sky_u_offset, 1.0) < 0.5)
        return ok


class TrivialPrimeSpaceAction:
    """The one-element group, for the ``network-amplitudes`` coordinates.

    The two sites' complex amplitudes are invariant under the
    polarisation/phase quarter turn (the reparameterisation is the quotient
    by it), so nothing is left to fold; the group mixture then reduces to its
    base flow with the same machinery (standardisation, circular angles).
    """

    group_size = 1
    mode_factor_sizes = [1]
    sky_u_period = 1.0

    def bind(self, prime_names):
        pass

    def __call__(self, point_dict, modes, inverse=False):
        return dict(point_dict)

    def in_fundamental_domain(self, point_dict):
        any_value = next(iter(point_dict.values()))
        return torch.ones_like(any_value, dtype=torch.bool)


def _identity_group_action(point_dict, modes, inverse=False):
    """Physical ``group_action_fn`` of :class:`TrivialPrimeSpaceAction`."""
    return dict(point_dict)


class AdaptiveNetworkDomain(AdaptiveFundamentalDomain):
    """Data-driven seams for the network group proposal (a ``base_reparam``).

    The network analogue of :class:`~nessai_gw.group_mixture.AdaptiveFundamentalDomain`
    (whose seam-finding it reuses), restricted to the ``"delta"`` family:
    each expert moves

    * the ``psi`` and ``delta_phase`` fold seams (``psi mod pi/2``,
      ``delta_phase mod pi``) by applying the unique group element that takes
      the point into the shifted domain, and
    * the seam of the baseline azimuth ``sky_u``: with the action's
      ``baseline_flip`` a fold seam (``sky_u mod 1/2``, chosen on the folded
      ``sky_u``); without it a translation mod 1 (no group acts on the sky;
      the seam is still a cut through the ring when the posterior wraps round
      the baseline),

    to where its folded posterior has the least mass.  All pieces are
    translations or group elements, so the map has unit Jacobian.  ``sky_u``
    stays in ``[0, P)`` (``P = action.sky_u_period``), so a
    :class:`~nessai_gw.group_mixture.SkyOctantProbit` with ``u_scale = 1/P``
    still applies after it.

    With ``circular_psi_phase`` the ``psi`` / ``delta_phase`` seams stay at
    zero and, with ``allow_phase_mode``, each expert picks its basis of the
    ``(psi_prime, delta_phase)`` torus instead, as in the parent class (e.g.
    ``phase mod pi/2`` where the phase is pinned rather than ``delta_phase``).
    """

    def __init__(self, action, param_names=None, circular_psi_phase=False,
                 circular_sky_u=False, allow_phase_mode=False):
        if allow_phase_mode and not circular_psi_phase:
            raise ValueError(
                "AdaptiveNetworkDomain: allow_phase_mode needs circular_psi_phase"
            )
        super().__init__(
            action, param_names, allow_phase_mode=allow_phase_mode,
            circular_psi_phase=circular_psi_phase,
        )
        #: ``sky_u`` is an angle of a circular base flow: keep its seam.
        self.circular_sky_u = bool(circular_sky_u)

    @property
    def _u_period(self):
        return getattr(self._action, "sky_u_period", 1.0)

    def bind(self, param_names):
        names = list(param_names)
        need = ("psi_prime", "delta_phase", "theta_jn_prime")
        if self._u_period < 1.0:
            need = need + ("sky_u",)
        missing = [n for n in need if n not in names]
        if missing:
            raise RuntimeError(
                f"AdaptiveNetworkDomain needs {missing} in the prime "
                f"parameters; got {names}."
            )
        self._names = names
        self._ipsi, self._idp, self._ith = (names.index(n) for n in need[:3])
        self._iu = names.index("sky_u") if "sky_u" in names else None

    def _in_target(self, z, seams, phase_mode):
        cu, cpsi, cdel, _ = seams
        ok = torch.remainder(z[:, self._ipsi] + 1 - cpsi, 2.0) < 1.0
        ok = ok & (torch.remainder(z[:, self._idp] + 1 - cdel, 2.0) < 1.0)
        if self._u_period < 1.0:
            ok = ok & (_wrap(z[:, self._iu] - cu, 1.0) < self._u_period)
        return ok

    def _map_to(self, z, seams, phase_mode):
        out = z.clone()
        found = torch.zeros(z.shape[0], dtype=torch.bool, device=z.device)
        for k in range(self._action.group_size):
            cand = self._act(z, torch.full((z.shape[0],), k, device=z.device))
            m = self._in_target(cand, seams, phase_mode) & ~found
            out[m] = cand[m]
            found |= m
        return out, found

    def _to_box(self, z, seams, phase_mode):
        cu, cpsi, cdel, _ = seams
        y = z.clone()
        if self._iu is not None:
            u = _wrap(z[:, self._iu] - cu, 1.0)
            if self._u_period < 1.0:
                u = torch.clamp(u, max=self._u_period * (1 - 1e-7))
            y[:, self._iu] = u
        y[:, self._ipsi] = torch.remainder(z[:, self._ipsi] + 1 - cpsi, 2.0) - 1
        y[:, self._idp] = torch.remainder(z[:, self._idp] + 1 - cdel, 2.0) - 1
        return y

    def _box_valid(self, y, phase_mode):
        a, b = y[:, self._ipsi], y[:, self._idp]
        ok = (a >= -1) & (a < 0) & (b >= -1) & (b < 0)
        if self._iu is not None:
            u = y[:, self._iu]
            ok = ok & (u >= 0) & (u < self._u_period)
        return ok

    def _from_box(self, y, seams, phase_mode):
        cu, cpsi, cdel, _ = seams
        z = y.clone()
        if self._iu is not None:
            z[:, self._iu] = _wrap(y[:, self._iu] + cu, 1.0)
        z[:, self._ipsi] = torch.remainder(y[:, self._ipsi] + 1 + cpsi, 2.0) - 1
        z[:, self._idp] = torch.remainder(y[:, self._idp] + 1 + cdel, 2.0) - 1
        return z

    @torch.no_grad()
    def update(self, canon):
        if canon.shape[0] < self.min_points:
            return False
        before = self._state()
        torus_before = self._torus_mn()
        torus_note = ""
        (cu, cpsi, cdel, _), _ = before
        du = dpsi = ddel = float("nan")
        if self._iu is not None and not getattr(self, "circular_sky_u", False):
            p = self._u_period
            u = canon[:, self._iu].double().cpu().numpy()
            cu, du, _ = self._choose_seam(u / p, 1.0, cu / p)
            cu = cu * p
        if getattr(self, "circular_psi_phase", False):
            cpsi = cdel = 0.0
            if self.allow_phase_mode:
                z, _ = self._map_to(canon, (cu, 0.0, 0.0, 0.0), False)
                torus_note = self._update_torus(z)
        else:
            z, _ = self._map_to(canon, (cu, 0.0, 0.0, 0.0), False)
            psi = (z[:, self._ipsi] + 1).double().cpu().numpy()
            cpsi, dpsi, _ = self._choose_seam(psi, 1.0, cpsi)
            z, _ = self._map_to(canon, (cu, cpsi, 0.0, 0.0), False)
            dl = (z[:, self._idp] + 1).double().cpu().numpy()
            cdel, ddel, _ = self._choose_seam(dl, 1.0, cdel)

        self._seams.copy_(
            torch.tensor([cu, cpsi, cdel, 0.0], dtype=torch.float64)
        )
        self._seen.fill_(True)
        changed = any(
            abs(a - b) > 1e-12 for a, b in zip(self._state()[0], before[0])
        ) or self._torus_mn() != torus_before
        logger.debug(
            "Adaptive network domain: seams sky_u %.4f psi %.4f delta %.4f; "
            "density at old seams sky %.2f psi %.2f delta %.2f (x uniform), "
            "n=%d%s",
            cu, cpsi, cdel, du, dpsi, ddel, canon.shape[0],
            " [changed]" if changed else "",
        )
        self._retrain_note = _seam_note(
            [("sky_u", cu, du), ("psi", cpsi, dpsi), ("delta", cdel, ddel)],
            changed,
        )
        if torus_note:
            self._retrain_note = (
                torus_note + "; " + self._retrain_note
                if self._retrain_note else torus_note
            )
        return changed


class BaselineFlipSymmetry:
    """The baseline half-turn ``h`` in the circular base frame (a
    ``base_symmetry``).

    With ``baseline_flip`` the group folds ``h`` (see
    :class:`PrimeSpacePolarisationPhaseAction`) onto half the ``sky_u``
    circle, gluing its two ends with a twist (``theta_jn_prime -> -theta_jn_prime``,
    ``psi -> -psi``), which a product of circles cannot represent.  The base
    flow instead models the whole ``sky_u`` circle (a
    :class:`~nessai_gw.group_mixture.CircularCanonicalTransform` with
    ``sky_u`` of length 1), the canonical points fill its first half, and the
    base density is summed over ``h``, which keeps the folded density
    continuous across the twisted seam.  In the angles of that frame ``h`` is

        t_u -> t_u + pi,   theta_jn_prime -> -theta_jn_prime,
        t_psi -> -t_psi,   t_delta unchanged

    (``psi -> pi - psi`` is ``psi_prime -> -psi_prime`` mod 2, and
    ``delta_phase = phase + sign(cos theta_jn) psi`` is invariant).  Unit
    Jacobian; an involution.  Needs ``psi_prime`` / ``delta_phase`` on the
    zero-offset polarisation/phase torus and ``sky_u`` starting the circle at
    the fundamental domain's ``sky_u_offset``.
    """

    def __init__(self, param_names=None):
        self._idx = None
        if param_names is not None:
            self.bind(param_names)

    def bind(self, param_names):
        names = list(param_names)
        need = ("sky_u", "theta_jn_prime", "psi_prime")
        missing = [n for n in need if n not in names]
        if missing:
            raise RuntimeError(
                f"BaselineFlipSymmetry needs {missing} in the prime "
                f"parameters; got {names}."
            )
        self._idx = tuple(names.index(n) for n in need)

    def _h(self, t):
        iu, ith, ipsi = self._idx
        out = t.clone()
        out[:, iu] = _wrap_angle(t[:, iu] + np.pi)
        out[:, ith] = -t[:, ith]
        out[:, ipsi] = _wrap_angle(-t[:, ipsi])
        return out

    def images(self, t):
        """The non-identity image of each row."""
        return [self._h(t)]

    def fold(self, t):
        """Map each row onto the canonical half circle, ``t_u in [-pi, 0)``."""
        second = (t[:, self._idx[0]] >= 0)[:, None]
        return torch.where(second, self._h(t), t)


# ---------------------------------------------------------------------------
# reparameterisations + proposal
# ---------------------------------------------------------------------------
def _no_physical_baseline_flip(point_dict, modes, inverse=False):
    """Placeholder ``group_action_fn`` with ``baseline_flip``: only the
    prime-space action implements the half-turn."""
    raise NotImplementedError(
        "the baseline half-turn is only implemented in the prime space "
        "(PrimeSpacePolarisationPhaseAction)"
    )


def _chirp_distance_kwargs(geometry, chirp_distance):
    """``triangular_group_reparameterisations`` chirp-distance arguments."""
    if not chirp_distance:
        return {"chirp_distance": False}
    if geometry.tensors is None:
        raise ValueError(
            "chirp_distance needs the detector tensors: build the geometry "
            "with DetectorNetworkGeometry.from_interferometers (or pass "
            "`tensors`), or pass chirp_distance=False."
        )
    return {
        "chirp_distance": True,
        "chirp_distance_tensors": geometry.tensors,
        "chirp_distance_k0": geometry.reference_detector,
    }


def network_group_reparameterisations(
    sampling_parameters,
    geometry,
    chirp_distance=True,
    effective_spin=True,
    log_mass_ratio=True,
    precessing_spin=False,
    effective_tidal_deformability=False,
    doppler_vector=None,
    time_reference_frequency=None,
    time_to_merger=None,
    chirp_distance_inversion=False,
    waveform_phase=None,
    network_amplitudes=False,
):
    """``reparameterisations`` dict for :func:`make_network_group_flow_proposal`.

    The triangular profile
    (:func:`~nessai_gw.group_mixture.triangular_group_reparameterisations`)
    with the ``detector-center-time`` vertex at the network's SNR-weighted
    barycentre: ``delta_phase`` (``polarisation-phase``) for ``phase``,
    fixed-bounds ``angle-sine`` for ``theta_jn``, ``aligned-spin`` for aligned
    spins, ``logit`` for tides, ``log-mass-ratio`` (``ln q``) for
    ``mass_ratio`` and, by default, the ``chirp-distance`` at the
    loudest detector (``geometry.reference_detector``) for
    ``luminosity_distance`` and the joint ``effective-spin``
    ``(chi_eff_prime, chi_diff_prime)`` for aligned ``chi_1``, ``chi_2``.  ``psi`` and the sky are added by the proposal
    (single-angle ``psi_prime``; baseline-frame
    :class:`~nessai_gw.reparameterisations.sky.EqualAreaSky`).

    Parameters
    ----------
    sampling_parameters : list of str
        Every parameter nessai samples.
    geometry : DetectorNetworkGeometry
        The network; must carry ``tensors`` when ``chirp_distance`` is set.
    chirp_distance : bool, optional
        Reparameterise ``luminosity_distance`` as the Roulet et al. chirp
        distance (needs ``chirp_mass``, ``theta_jn``, ``ra``, ``dec`` and
        ``psi`` to be sampled).  Ignored when ``luminosity_distance`` is not
        sampled.  Must match :func:`make_network_group_flow_proposal`.
        Default ``True``.
    effective_spin : bool, optional
        Carry aligned ``chi_1``, ``chi_2`` as the effective-spin pair (needs
        ``mass_ratio``).  Ignored without both spins.  Must match
        :func:`make_network_group_flow_proposal`.  Default ``True``.
    log_mass_ratio : bool, optional
        Sample ``mass_ratio`` as ``ln q`` (``log-mass-ratio``).  Must match
        :func:`make_network_group_flow_proposal`.  Default ``True``.
    precessing_spin : bool, optional
        Carry the six precessing spin parameters as the ``precessing-spin``
        coordinates (needs ``mass_ratio``).  Must match
        :func:`make_network_group_flow_proposal`.  Default ``False``.
    effective_tidal_deformability : bool, optional
        Carry ``lambda_1``, ``lambda_2`` as ``(lambda_tilde_prime,
        lambda_diff_prime)`` (needs ``mass_ratio``).  Must match
        :func:`make_network_group_flow_proposal`.  Default ``False``.
    doppler_vector : array_like, optional
        The Earth's barycentric velocity over ``c`` at the coalescence time:
        sample the Doppler-corrected chirp mass (runs with the orbital motion
        only).  Must match :func:`make_network_group_flow_proposal`.  Default
        ``None``.
    time_reference_frequency : float or "adaptive", optional
        Measure the arrival time at the barycentre when the 22 mode passes
        this frequency instead of at merger (``"adaptive"``: the frequency of
        least spread over the training points, chosen before every training);
        see :func:`~nessai_gw.group_mixture.triangular_group_reparameterisations`.
        Default ``None``.
    time_to_merger : str, optional
        The waveform's time to merger, ``"module:function"``, as in
        :func:`~nessai_gw.group_mixture.triangular_group_reparameterisations`.
        Default ``None`` (2PN).
    chirp_distance_inversion : bool, optional
        Boundary inversion of the luminosity distance under the chirp
        distance, as in
        :func:`~nessai_gw.group_mixture.triangular_group_reparameterisations`.
        Default ``False``.
    waveform_phase : str, optional
        The waveform's 22-mode phase, ``"module:function"``, as in
        :func:`~nessai_gw.group_mixture.triangular_group_reparameterisations`.
        Default ``None`` (merger phase).
    network_amplitudes : bool, optional
        Carry ``(luminosity_distance, theta_jn, psi, phase)`` as the two
        sites' complex amplitudes (``network-amplitudes``; see
        :class:`~nessai_gw.reparameterisations.NetworkAmplitudesReparameterisation`),
        which replaces ``chirp_distance`` and ``delta_phase``.  Needs two
        sites with one interferometer each.  Must match
        :func:`make_network_group_flow_proposal`.  Default ``False``.
    """
    if network_amplitudes:
        return _network_amplitude_reparameterisations(
            sampling_parameters, geometry,
            effective_spin=effective_spin, log_mass_ratio=log_mass_ratio,
            precessing_spin=precessing_spin,
            effective_tidal_deformability=effective_tidal_deformability,
            doppler_vector=doppler_vector,
            time_reference_frequency=time_reference_frequency,
            time_to_merger=time_to_merger, waveform_phase=waveform_phase,
        )
    return triangular_group_reparameterisations(
        sampling_parameters,
        geometry.reference_time,
        vertex=geometry.timing_vertex,
        phase_coordinates="polarisation-phase",
        effective_spin=effective_spin,
        log_mass_ratio=log_mass_ratio,
        precessing_spin=precessing_spin,
        effective_tidal_deformability=effective_tidal_deformability,
        doppler_vector=doppler_vector,
        time_reference_frequency=time_reference_frequency,
        time_to_merger=time_to_merger,
        chirp_distance_inversion=chirp_distance_inversion,
        waveform_phase=waveform_phase,
        **_chirp_distance_kwargs(
            geometry,
            chirp_distance and "luminosity_distance" in sampling_parameters,
        ),
    )


def _network_amplitude_reparameterisations(sampling_parameters, geometry,
                                           **kwargs):
    """The network profile with ``network-amplitudes`` (the two loudest
    sites') in place of the distance, ``delta_phase`` and ``theta_jn``."""
    return triangular_group_reparameterisations(
        sampling_parameters,
        geometry.reference_time,
        vertex=geometry.timing_vertex,
        amplitudes=True,
        amplitude_tensors=geometry.site_tensors(),
        **kwargs,
    )


def make_network_group_flow_proposal(
    sampling_parameters,
    geometry,
    gaussianise_sky=True,
    adaptive_domain=True,
    psi_offset=0.0,
    delta_offset=0.0,
    n_clusters_max=1,
    flow_model_factory=None,
    chirp_distance=True,
    effective_spin=True,
    log_mass_ratio=True,
    precessing_spin=False,
    effective_tidal_deformability=False,
    doppler_vector=None,
    baseline_flip=False,
    sky_u_offset=0.0,
    circular_psi_phase=False,
    circular_sky_u=False,
    circular_standardisation=False,
    mass_ratio_shear=False,
    time_reference_frequency=None,
    time_to_merger=None,
    chirp_distance_inversion=False,
    waveform_phase=None,
    phase_recanon=False,
    network_amplitudes=False,
):
    """``FlowProposal`` subclass for any detector network.

    Pass the returned class as ``flow_proposal_class`` to
    ``bilby.run_sampler(sampler="nessai", ...)`` together with
    ``reparameterisations=network_group_reparameterisations(names, geometry)``.

    The flow sees the barycentre time ``t_det``, the baseline-frame sky
    ``(sky_u, sky_v)``, ``psi_prime``, ``delta_phase`` and (by default)
    ``chirp_distance`` and ``chi_eff_prime`` / ``chi_diff_prime``; the group
    mixture folds the polarisation/phase ``Z4``
    (4 elements; 8 with ``baseline_flip``) and learns its weights.

    Parameters
    ----------
    sampling_parameters : list of str
        Every parameter nessai samples.  Must contain ``theta_jn``, ``psi`` and
        ``phase``; ``ra`` / ``dec`` / ``geocent_time`` get the baseline sky and
        barycentre time when present.
    geometry : DetectorNetworkGeometry
        The network (timing barycentre, baseline sky frame).
    gaussianise_sky : bool, optional
        Probit the equal-area ``(sky_u, sky_v)`` square to two standard normals
        in front of the base flow (:class:`~nessai_gw.group_mixture.SkyOctantProbit`
        with unit scales), so the base flow sees no hard prior edge.  Default
        ``True``.
    adaptive_domain : bool, optional
        Give each group-mixture expert an :class:`AdaptiveNetworkDomain`, which
        moves the ``psi`` / ``delta_phase`` fold seams and the ``sky_u`` seam off
        its posterior mass every round.  Default ``True``.
    psi_offset, delta_offset : float, optional
        Fixed seams (radians) of the base fundamental domain; see
        :class:`PrimeSpacePolarisationPhaseAction`.
    n_clusters_max, flow_model_factory
        Clustered base flow, as in
        :func:`~nessai_gw.group_mixture.make_triangular_group_flow_proposal`.
    chirp_distance : bool, optional
        Whether ``luminosity_distance`` is carried as the chirp distance; must
        match :func:`network_group_reparameterisations`.  Default ``True``.
    effective_spin : bool, optional
        Whether aligned spins are carried as ``(chi_eff_prime,
        chi_diff_prime)``; must match :func:`network_group_reparameterisations`.
        Default ``True``.
    log_mass_ratio : bool, optional
        Whether ``mass_ratio`` is sampled as ``ln q``; must match
        :func:`network_group_reparameterisations`.  Default ``True``.
    precessing_spin : bool, optional
        Whether precessing spins are carried as the ``precessing-spin``
        coordinates; must match :func:`network_group_reparameterisations`.
        Default ``False``.
    effective_tidal_deformability, doppler_vector : optional
        Must match :func:`network_group_reparameterisations`.
    baseline_flip : bool, optional
        Also fold the half-turn about the baseline with the inclination
        flipped (the delay-preserving image of the antipodal degeneracy; see
        :class:`PrimeSpacePolarisationPhaseAction`), halving the ``sky_u``
        domain.  Needs the sky.  Default ``False``.
    sky_u_offset : float, optional
        Fixed ``sky_u`` seam of the base fundamental domain with
        ``baseline_flip`` (the adaptive domain moves it per expert).
    circular_psi_phase : bool, optional
        Carry ``psi_prime`` and ``delta_phase`` as angles of a circular base
        flow (:class:`~nessai_gw.group_mixture.CircularCanonicalTransform`):
        the quarter turn folds them to a torus, which the base flow models
        without seams, so the adaptive domain leaves their seams alone.
        Needs a circular flow (``flow_config['ftype'] = 'circular'``) and zero
        ``psi_offset`` / ``delta_offset``.  Default ``False``.
    circular_sky_u : bool, optional
        Also carry the baseline azimuth ``sky_u`` as an angle (the posterior
        ring about the baseline then has no cut); ``sky_v`` alone is
        probit-mapped.  With ``baseline_flip`` the base flow models the whole
        ``sky_u`` circle summed over the half-turn (:class:`BaselineFlipSymmetry`).
        Needs ``circular_psi_phase``.  Default ``False``.
    circular_standardisation : bool, optional
        Standardise the circular angles as well: each is rotated to its mean
        direction and spread over the circle by a Möbius map fitted to the
        live points (nessai's ``circular_standardisation``), so a tightly
        measured angle (``delta_phase``, ``sky_u`` late in a run) reaches the
        uniform latent as an O(1) shape rather than a spike.  Needs circular
        angles.  Default ``False``.
    mass_ratio_shear : bool, optional
        Shear the other non-circular standardised coordinates by an even
        polynomial in ``mass_ratio_prime`` about the ``q = 1`` edge (see
        :func:`~nessai_gw.group_mixture.curvature_shear_kwargs`).  Default
        ``False``.
    time_reference_frequency : optional
        Must match :func:`network_group_reparameterisations` (the prime names
        do not depend on it; the probe checks the order of the inverse pass
        with it).  Default ``None``.
    phase_recanon : bool, optional
        Let each expert's :class:`AdaptiveNetworkDomain` pick its basis of the
        ``(psi_prime, delta_phase)`` torus off its live points (``delta_phase``
        on a clump, ``phase mod pi/2`` where the phase is pinned instead).
        Needs ``circular_psi_phase`` and ``adaptive_domain``.  Default
        ``False``.
    network_amplitudes : bool, optional
        The flow sees the two sites' complex amplitudes
        ``(ln_amp_k, arg_amp_k)`` instead of ``chirp_distance``,
        ``theta_jn_prime``, ``psi_prime`` and ``delta_phase``; must match
        :func:`network_group_reparameterisations`.  The amplitudes are
        invariant under the polarisation/phase quarter turn, so the group
        is trivial (:class:`TrivialPrimeSpaceAction`) and the ``arg_amp_k``
        are angles of a circular base flow.  Excludes ``baseline_flip``,
        ``circular_psi_phase``, ``phase_recanon`` and the ``psi`` /
        ``delta_phase`` offsets; ``circular_sky_u`` and
        ``circular_standardisation`` work as usual.  Default ``False``.
    """
    if network_amplitudes:
        return _make_network_amplitude_flow_proposal(
            sampling_parameters, geometry,
            gaussianise_sky=gaussianise_sky,
            adaptive_domain=adaptive_domain,
            n_clusters_max=n_clusters_max,
            flow_model_factory=flow_model_factory,
            effective_spin=effective_spin,
            log_mass_ratio=log_mass_ratio,
            precessing_spin=precessing_spin,
            effective_tidal_deformability=effective_tidal_deformability,
            doppler_vector=doppler_vector,
            circular_sky_u=circular_sky_u,
            circular_standardisation=circular_standardisation,
            mass_ratio_shear=mass_ratio_shear,
            time_reference_frequency=time_reference_frequency,
            time_to_merger=time_to_merger,
            waveform_phase=waveform_phase,
            unsupported=dict(
                baseline_flip=baseline_flip,
                circular_psi_phase=circular_psi_phase,
                phase_recanon=phase_recanon,
                psi_offset=psi_offset, delta_offset=delta_offset,
                sky_u_offset=sky_u_offset,
            ),
        )
    if phase_recanon and not (circular_psi_phase and adaptive_domain):
        raise ValueError(
            "phase_recanon needs circular_psi_phase and adaptive_domain."
        )
    try:
        from nessai.flowmodel.group_mixture import make_group_mixture_flow
    except ImportError as exc:  # pragma: no cover - depends on nessai version
        raise RuntimeError(
            "make_network_group_flow_proposal requires a version of nessai "
            "that ships nessai.flowmodel.group_mixture."
        ) from exc

    names = list(sampling_parameters)
    missing = sorted({"theta_jn", "psi", "phase"} - set(names))
    if missing:
        raise RuntimeError(
            f"The sampling space is missing {missing}, which the "
            "polarisation/phase group action needs."
        )

    make_flow = _select_flow_model_factory(
        n_clusters_max,
        flow_model_factory,
    )
    prime_names = _prime_parameter_names(
        names, geometry.reference_time, vertex=geometry.timing_vertex,
        psi_single=True, effective_spin=effective_spin,
        precessing_spin=precessing_spin,
        log_mass_ratio=log_mass_ratio,
        effective_tidal_deformability=effective_tidal_deformability,
        doppler_vector=doppler_vector,
        time_reference_frequency=time_reference_frequency,
        time_to_merger=time_to_merger,
        chirp_distance_inversion=chirp_distance_inversion,
        waveform_phase=waveform_phase,
        **_chirp_distance_kwargs(
            geometry, chirp_distance and "luminosity_distance" in names
        ),
    )
    if baseline_flip and "sky_u" not in prime_names:
        raise RuntimeError(
            "baseline_flip needs the sky (ra, dec) in the sampling parameters."
        )
    action = PrimeSpacePolarisationPhaseAction(
        prime_names, psi_offset=psi_offset, delta_offset=delta_offset,
        baseline_flip=baseline_flip, sky_u_offset=sky_u_offset,
    )
    if baseline_flip:
        physical = _no_physical_baseline_flip
    else:
        physical = PolarisationPhaseGroupAction(psi_offset, delta_offset)

    if circular_sky_u and not circular_psi_phase:
        raise ValueError("circular_sky_u needs circular_psi_phase.")
    if circular_sky_u and "sky_u" not in prime_names:
        raise RuntimeError(
            "circular_sky_u needs the sky (ra, dec) in the sampling parameters."
        )
    if circular_psi_phase and (psi_offset or delta_offset):
        raise ValueError(
            "circular_psi_phase needs psi_offset = delta_offset = 0 (the "
            "circular base flow has no seams to move)."
        )

    gm_params = inspect.signature(make_group_mixture_flow).parameters
    gm_kwargs = {"mode_factor_sizes": action.mode_factor_sizes}
    canonical = None
    if gaussianise_sky and "sky_u" in prime_names:
        canonical = SkyOctantProbit(
            prime_names,
            u_scale=None if circular_sky_u else 1.0 / action.sky_u_period,
            v_scale=1.0,
        )
    circles = {}
    if circular_psi_phase:
        circles["psi_prime"] = POLARISATION_PHASE_CIRCLE
        circles[action._delta] = POLARISATION_PHASE_CIRCLE
    if circular_sky_u:
        # the whole circle, starting at the fundamental domain's seam
        start = float(sky_u_offset) if baseline_flip else 0.0
        circles["sky_u"] = (start, 1.0, (0.0, 1.0))
    if circles:
        if "circular_parameters" not in gm_params:
            raise RuntimeError(
                "circular_psi_phase requires a version of nessai whose "
                "make_group_mixture_flow accepts `circular_parameters`."
            )
        canonical = CircularCanonicalTransform(
            circles, prime_names, inner=canonical
        )
        gm_kwargs["circular_parameters"] = list(circles)
        if circular_sky_u and baseline_flip:
            gm_kwargs["base_symmetry"] = BaselineFlipSymmetry(prime_names)
        if circular_standardisation:
            if "circular_standardisation" not in gm_params:
                raise RuntimeError(
                    "circular_standardisation requires a version of nessai "
                    "whose make_group_mixture_flow accepts it."
                )
            gm_kwargs["circular_standardisation"] = True
    elif circular_standardisation:
        raise ValueError("circular_standardisation needs circular angles.")
    if canonical is not None:
        if "canonical_transform" not in gm_params:
            raise RuntimeError(
                "gaussianise_sky requires a version of nessai whose "
                "make_group_mixture_flow accepts `canonical_transform`."
            )
        gm_kwargs["canonical_transform"] = canonical
    # with circular angles the adaptive domain only has the sky_u seam left
    # to move, and none if sky_u is circular too
    moves_seams = not circular_psi_phase or bool(phase_recanon) or (
        "sky_u" in prime_names and not circular_sky_u
    )
    if adaptive_domain and moves_seams:
        if "base_reparam_factory" not in gm_params:
            raise RuntimeError(
                "adaptive_domain requires a version of nessai whose "
                "make_group_mixture_flow accepts `base_reparam_factory`."
            )
        gm_kwargs["base_reparam_factory"] = functools.partial(
            AdaptiveNetworkDomain, action, list(prime_names),
            circular_psi_phase=bool(circular_psi_phase),
            circular_sky_u=bool(circular_sky_u),
            allow_phase_mode=bool(phase_recanon),
        )

    gm_kwargs.update(
        curvature_shear_kwargs(mass_ratio_shear, prime_names, gm_params)
    )
    flow_model_cls = make_flow(
        group_action_fn=physical,  # ignored on the prime-space path
        group_size=action.group_size,
        param_names=prime_names,
        prime_space_action=action,
        prime_space_in_domain=action.in_fundamental_domain,
        min_canon_std=_MIN_CANON_STD,
        **gm_kwargs,
    )
    logger.info("Network group proposal for %r", geometry)
    return _make_group_proposal_class(
        "NetworkGroupFlowProposal",
        flow_model_cls,
        action,
        sky_rotation=geometry.sky_frame_rotation,
        psi_single=True,
        phase_coordinates="polarisation-phase",
    )


def _make_network_amplitude_flow_proposal(
    sampling_parameters,
    geometry,
    gaussianise_sky,
    adaptive_domain,
    n_clusters_max,
    flow_model_factory,
    effective_spin,
    log_mass_ratio,
    precessing_spin,
    effective_tidal_deformability,
    doppler_vector,
    circular_sky_u,
    circular_standardisation,
    mass_ratio_shear,
    time_reference_frequency,
    time_to_merger,
    waveform_phase,
    unsupported,
):
    """:func:`make_network_group_flow_proposal` with ``network_amplitudes``."""
    from nessai.flowmodel.group_mixture import make_group_mixture_flow

    bad = sorted(k for k, v in unsupported.items() if v)
    if bad:
        raise ValueError(f"network_amplitudes excludes {bad}.")
    names = list(sampling_parameters)
    prime = _prime_parameter_names(
        names, geometry.reference_time, vertex=geometry.timing_vertex,
        psi_single=True, effective_spin=effective_spin,
        precessing_spin=precessing_spin, log_mass_ratio=log_mass_ratio,
        effective_tidal_deformability=effective_tidal_deformability,
        doppler_vector=doppler_vector,
        time_reference_frequency=time_reference_frequency,
        time_to_merger=time_to_merger, waveform_phase=waveform_phase,
        chirp_distance=False,
    )
    # the placeholder order; initialise re-binds to nessai's actual one
    replaced = {"delta_phase", "psi_prime", "theta_jn_prime",
                "luminosity_distance_prime"}
    prime_names = list(NETWORK_AMPLITUDE_PRIME_PARAMETERS) + [
        n for n in prime if n not in replaced
    ]
    action = TrivialPrimeSpaceAction()
    gm_params = inspect.signature(make_group_mixture_flow).parameters
    if "circular_parameters" not in gm_params:
        raise RuntimeError(
            "network_amplitudes requires a version of nessai whose "
            "make_group_mixture_flow accepts `circular_parameters`."
        )
    gm_kwargs = {"mode_factor_sizes": action.mode_factor_sizes}
    canonical = None
    if gaussianise_sky and "sky_u" in prime_names:
        canonical = SkyOctantProbit(
            prime_names, u_scale=None if circular_sky_u else 1.0, v_scale=1.0,
        )
    circles = {name: AMPLITUDE_ARGUMENT_CIRCLE
               for name in NETWORK_AMPLITUDE_ANGLES}
    if circular_sky_u:
        if "sky_u" not in prime_names:
            raise RuntimeError(
                "circular_sky_u needs the sky (ra, dec) in the sampling "
                "parameters."
            )
        circles["sky_u"] = (0.0, 1.0, (0.0, 1.0))
    canonical = CircularCanonicalTransform(circles, prime_names, inner=canonical)
    gm_kwargs["circular_parameters"] = list(circles)
    gm_kwargs["canonical_transform"] = canonical
    if circular_standardisation:
        gm_kwargs["circular_standardisation"] = True
    # the sky_u seam is the only one left for the adaptive domain to move,
    # and AdaptiveNetworkDomain is built around psi_prime / delta_phase
    if adaptive_domain and not circular_sky_u:
        logger.info("network_amplitudes: the sky_u seam stays fixed")
    gm_kwargs.update(
        curvature_shear_kwargs(mass_ratio_shear, prime_names, gm_params)
    )
    flow_model_cls = _select_flow_model_factory(
        n_clusters_max, flow_model_factory
    )(
        group_action_fn=_identity_group_action,
        group_size=action.group_size,
        param_names=prime_names,
        prime_space_action=action,
        prime_space_in_domain=action.in_fundamental_domain,
        min_canon_std=_MIN_CANON_STD,
        **gm_kwargs,
    )
    logger.info("Network amplitude proposal for %r", geometry)
    return _make_group_proposal_class(
        "NetworkAmplitudeFlowProposal",
        flow_model_cls,
        action,
        sky_rotation=geometry.sky_frame_rotation,
        psi_single=True,
        phase_coordinates="polarisation-phase",
    )
