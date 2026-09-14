"""Inclination reparameterisations for single-site triangular detectors."""

import inspect

import numpy as np
from nessai.reparameterisations import Reparameterisation

from .. import nessai_logger
from .._ellipse import PolarisationEllipse

logger = nessai_logger.getChild(__name__)

#: Value written into ``theta_jn`` for prime points whose implied
#: ``cos theta_jn`` falls outside ``[-1, 1]``.  It is outside the ``[0, pi]``
#: prior range, so nessai's prior-bounds check drops the point rather than
#: silently accepting a clipped -- and therefore non-invertible -- angle.
_OUT_OF_BOUNDS = -1.0


class PolarisationEllipseReparameterisation(Reparameterisation):
    """Inclination measured *relative to the value the sky position forces*.

    For a single triangular detector the frozen long-wavelength extrinsic
    likelihood is degenerate over the whole sky, not just under the discrete
    group :mod:`nessai_gw.group_mixture` folds: at any sky direction ``n`` the
    two-dimensional model span ``{alpha Z(n) + beta conj(Z(n))}`` covers the
    entire null-stream-orthogonal subspace the data live in.  What survives is
    a *lock* between the sky and the polarisation ellipse -- fixing ``n`` fixes
    the inclination.  The posterior is therefore a thin sheet
    ``cos theta_jn ~ cos iota*(n)`` winding across the sphere, and because
    ``cos iota*`` runs up to ``+-1`` at the eight sky points where the signal
    reads as exactly circularly polarised and saturates on a plateau elsewhere,
    the ``cos theta_jn`` marginal comes out sharply bimodal with an empty band
    in between.  Neither the sheet nor the bimodality is removed by the discrete
    fold.

    This reparameterisation makes the sheet an axis by subtracting the lock ::

        theta_jn_prime = (cos(theta_jn) - cos iota*(ra, dec)) / scale

    with ``cos iota*`` from :class:`nessai_gw._ellipse.PolarisationEllipse`.  On
    the ET-Delta runs the sampled ``cos theta_jn`` tracks ``cos iota*`` with a
    correlation of 0.99, so the prime coordinate is a single unimodal residual
    roughly six times narrower than the raw marginal, and the flow no longer has
    to represent a curved multimodal ridge across ``theta_jn`` and the sky.

    Compatible with the group fold without any change to it.  The map is built
    on the idealised planar triangle implied by ``plane_normal``, for which a
    quarter turn of the source azimuth leaves ``cos iota*`` invariant and a
    reflection through the detector plane flips its sign -- exactly how
    ``cos theta_jn`` itself transforms.  The residual is therefore equivariant,
    and :class:`nessai_gw.group_mixture.PrimeSpaceTriangularGroupAction` keeps
    negating the coordinate under a reflection as before.  It does need the
    ellipse to recover ``sign(cos theta_jn)`` for the ``delta_phase``
    coordinate, which is why the same object is handed to the proposal.

    ``ra`` and ``dec`` are held fixed by this map (the group action is what
    transforms them), so the Jacobian is that of ``theta_jn -> cos theta_jn``
    alone::

        log |d theta_jn_prime / d theta_jn| = log sin(theta_jn) - log scale.

    Requires ``ra`` and ``dec`` on both the forward and the inverse pass.

    Parameters
    ----------
    parameters : Union[str, List[str]]
        Name of the parameter; must be ``theta_jn``.
    prior_bounds : Union[list, dict], optional
        Prior bounds for ``theta_jn``.  Unused -- the coordinate is a residual
        about a sky-dependent centre, not a rescaling of the prior range.
    ellipse : PolarisationEllipse
        The sky -> inclination map.  Alternatively pass ``plane_normal``,
        ``reference_time`` and ``fiducial`` and one is built here.
    plane_normal, reference_time, fiducial, azimuth_offset : optional
        Forwarded to :class:`~nessai_gw._ellipse.PolarisationEllipse` when
        ``ellipse`` is not given.
    coordinate : {"angle", "cos"}, optional
        Which residual to take.  ``"angle"`` (the default) uses
        ``theta_jn - arccos(cos iota*)``, rescaled by ``2 / pi`` so it matches
        the ``angle-sine`` coordinate it replaces: the Jacobian is then constant
        and the prime-space prior keeps ``angle-sine``'s smooth ``sin(theta_jn)``
        bump, which is what the live points look like early in a run, before the
        likelihood has locked the sheet.  ``"cos"`` uses
        ``cos(theta_jn) - cos iota*``, i.e. the residual in the coordinate the
        lock is naturally linear in.  Both are odd under the plane reflection,
        so either works with the fold.
    scale : float, optional
        Prime-space scale (default 1, so the coordinate spans at most
        ``[-2, 2]``).  The group-mixture wrapper standardises each prime
        dimension anyway, so this mainly sets the units of the stored value.
    adaptive_width : bool, optional
        Divide the residual by a fitted sky-dependent width
        ``s(a) = sqrt(w0**2 + (w1 a)**2)``, ``a = |cos iota*(n)|``, so the
        flow-facing coordinate is homoskedastic.  The plain residual is centred
        but fans out towards the eight face-on points (there ``cos iota*`` is a
        steep function of ``n``); ``s`` is refit from the live points on every
        :meth:`update`, ``(w0, w1)`` stored on the shared ellipse so the group
        action reads the same width.  Default ``False`` (constant width, the
        original coordinate).  Measured on the ET-Delta v70/v96 folds it roughly
        doubles the mid-run non-Gaussianity reduction of the plain lock.
    width_bins : int, optional
        Number of ``|cos iota*|`` quantile bins for the width fit (default 10).
    width_min_points : int, optional
        Below this many finite live points :meth:`update` leaves the width
        unchanged (default 400).
    width_min_corr : float, optional
        :meth:`update` only fits a width when ``corr(|cos iota*|, |residual|)``
        is at least this (default 0.15); below it -- early in a run, before the
        sheet has locked, when the residual is prior-wide everywhere -- the
        width is reset to the constant 1 so a monotone ``s`` cannot
        anti-whiten.
    prior : optional
        Accepted for registry compatibility and ignored.
    """

    one_to_one = False

    def __init__(
        self,
        parameters=None,
        prior_bounds=None,
        ellipse=None,
        plane_normal=None,
        reference_time=None,
        fiducial=None,
        azimuth_offset=0.0,
        coordinate="angle",
        scale=1.0,
        adaptive_width=False,
        width_bins=10,
        width_min_points=400,
        width_min_corr=0.15,
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

        if self.parameters != ["theta_jn"]:
            raise RuntimeError(
                "PolarisationEllipseReparameterisation must act on 'theta_jn'; "
                f"got {self.parameters}"
            )
        if ellipse is None:
            if plane_normal is None or reference_time is None or fiducial is None:
                raise ValueError(
                    "PolarisationEllipseReparameterisation requires either "
                    "`ellipse` or all of `plane_normal`, `reference_time` and "
                    "`fiducial`."
                )
            ellipse = PolarisationEllipse(
                plane_normal, reference_time, fiducial,
                azimuth_offset=azimuth_offset,
            )
        self.ellipse = ellipse
        if coordinate not in ("angle", "cos"):
            raise ValueError(
                f"`coordinate` must be 'angle' or 'cos', got {coordinate!r}."
            )
        self.coordinate = coordinate
        self.scale = float(scale)
        if self.scale <= 0.0:
            raise ValueError(f"`scale` must be positive, got {self.scale}.")
        #: Fit a sky-dependent residual width ``s(|cos iota*|)`` from the
        #: live points each round and divide the residual by it, so the
        #: flow sees a homoskedastic coordinate (see the class notes).
        self.adaptive_width = bool(adaptive_width)
        self.width_bins = int(width_bins)
        self.width_min_points = int(width_min_points)
        self.width_min_corr = float(width_min_corr)

        self.prime_parameters = ["theta_jn_prime"]
        if hasattr(self, "output_parameters"):
            self.output_parameters = ["theta_jn_prime"]
        self.requires = ["ra", "dec"]
        if hasattr(self, "inverse_input_parameters"):
            self.inverse_input_parameters = list(
                dict.fromkeys(
                    list(self.inverse_input_parameters or []) + ["ra", "dec"]
                )
            )
        self._log_scale = float(np.log(self.scale))
        #: ``angle`` mode matches ``angle-sine``'s 2 / pi rescaling.
        self._angle_gain = 2.0 / (np.pi * self.scale)
        self._log_angle_gain = float(np.log(self._angle_gain))

    # ------------------------------------------------------------------
    def _cos_iota(self, x):
        return self.ellipse.cos_iota(x["ra"], x["dec"])

    def _centre(self, cos_iota):
        """``cos_iota`` -> the lock in the active coordinate."""
        if self.coordinate == "cos":
            return cos_iota
        return np.arccos(np.clip(cos_iota, -1.0, 1.0))

    def _gain(self, cos_iota):
        """Forward multiplier and its log; heteroskedastic when adaptive.

        ``x_prime = (residual) * gain``.  For ``"cos"`` the residual is
        ``cos theta_jn - cos iota*`` and ``gain = 1 / (scale s)``; for
        ``"angle"`` it is ``theta_jn - arccos(cos iota*)`` and
        ``gain = angle_gain / s``.  ``s = 1`` unless ``adaptive_width``.
        """
        if self.adaptive_width:
            s = self.ellipse.residual_width(np.abs(np.clip(cos_iota, -1.0, 1.0)))
        else:
            s = 1.0
        if self.coordinate == "cos":
            gain = 1.0 / (self.scale * s)
        else:
            gain = self._angle_gain / s
        return gain, np.log(gain)

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        theta = x["theta_jn"]
        cos_iota = self._cos_iota(x)
        centre = self._centre(cos_iota)
        gain, log_gain = self._gain(cos_iota)
        if self.coordinate == "cos":
            x_prime[self.prime_parameters[0]] = (np.cos(theta) - centre) * gain
            return (x, x_prime,
                    log_j + np.log(np.abs(np.sin(theta))) + log_gain)
        x_prime[self.prime_parameters[0]] = (theta - centre) * gain
        return x, x_prime, log_j + log_gain

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        u = x_prime[self.prime_parameters[0]]
        cos_iota = self._cos_iota(x)
        centre = self._centre(cos_iota)
        gain, log_gain = self._gain(cos_iota)
        if self.coordinate == "cos":
            cos_theta = u / gain + centre
            x["theta_jn"] = np.where(
                np.abs(cos_theta) <= 1.0,
                np.arccos(np.clip(cos_theta, -1.0, 1.0)),
                _OUT_OF_BOUNDS,
            )
            sin_theta = np.sqrt(np.clip(1.0 - cos_theta ** 2, 1e-300, None))
            return x, x_prime, log_j - np.log(sin_theta) - log_gain
        theta = u / gain + centre
        x["theta_jn"] = np.where(
            (theta >= 0.0) & (theta <= np.pi), theta, _OUT_OF_BOUNDS
        )
        return x, x_prime, log_j - log_gain

    def update(self, x, x_prime=None):
        """Refit the residual width ``s(|cos iota*|)`` from the live points.

        No-op unless ``adaptive_width``.  The residual is centred by the lock but
        its spread fans out towards the face-on points once the sheet is locked;
        this fits ``s(a)**2 = w0**2 + (w1 a)**2`` (a monotone, non-negative
        inductive bias for the fan) to the robust per-bin width and stores it as
        knots on the shared ellipse, which the group action then reads.

        Guarded two ways so it does nothing before the fan exists (early in a
        run the residual is prior-wide *everywhere* and a monotone ``s`` would
        mildly anti-whiten it): it needs ``width_min_points`` finite points and
        a correlation of at least ``width_min_corr`` between ``|cos iota*|`` and
        ``|residual|``.  Below that the width is reset to the constant ``1``.
        """
        if not self.adaptive_width:
            return
        ra, dec, theta = x["ra"], x["dec"], x["theta_jn"]
        finite = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(theta)
        if int(finite.sum()) < self.width_min_points:
            return
        cos_iota = self.ellipse.cos_iota(ra[finite], dec[finite])
        a = np.abs(np.clip(cos_iota, -1.0, 1.0))
        if self.coordinate == "cos":
            resid = np.cos(theta[finite]) - cos_iota
        else:
            resid = theta[finite] - np.arccos(np.clip(cos_iota, -1.0, 1.0))
        if np.std(a) < 1e-6 or (
            np.corrcoef(a, np.abs(resid))[0, 1] < self.width_min_corr
        ):
            self.ellipse.set_width([0.0, 1.0], [1.0, 1.0])
            return
        edges = np.quantile(a, np.linspace(0.0, 1.0, self.width_bins + 1))
        edges[-1] += 1e-9
        idx = np.clip(np.digitize(a, edges) - 1, 0, self.width_bins - 1)
        cen, var = [], []
        for i in range(self.width_bins):
            m = idx == i
            if int(m.sum()) < 20:
                continue
            r = resid[m]
            mad = np.median(np.abs(r - np.median(r)))
            cen.append(float(a[m].mean()))
            var.append(float((1.4826 * mad) ** 2))
        if len(cen) < 3:
            return
        cen = np.asarray(cen)
        design = np.column_stack([np.ones_like(cen), cen ** 2])
        coef, *_ = np.linalg.lstsq(design, np.asarray(var), rcond=None)
        w0 = float(np.sqrt(max(coef[0], 1e-6)))
        w1 = float(np.sqrt(max(coef[1], 0.0)))
        grid = np.linspace(0.0, 1.0, 9)
        self.ellipse.set_width(grid, np.sqrt(w0 ** 2 + (w1 * grid) ** 2))
