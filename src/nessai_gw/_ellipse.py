"""The sky -> polarisation-ellipse map of a single triangular detector.

Why this exists
---------------
In the frozen long-wavelength limit the three nested interferometers of a
triangle have detector tensors that sum to zero, so the null stream
``sum_k h_k = 0`` is exact and the data span two complex dimensions at each
frequency -- the same dimension as ``(h_+, h_x)``.  Writing
``Z_k = F_+,k + i F_x,k`` at ``psi = 0``, the dominant-mode response is

    h_k = [ alpha Z_k + beta conj(Z_k) ] A(f),

    alpha = c_+ e^{-2i psi} e^{2i phi_c} / d_L,   c_+ = (1 + cos iota)^2 / 4,
    beta  = c_- e^{+2i psi} e^{2i phi_c} / d_L,   c_- = (1 - cos iota)^2 / 4,

and ``span{Z(n), conj(Z(n))}`` is that whole two-dimensional space for *every*
sky direction ``n``.  The sky is therefore exactly degenerate: any ``n`` fits
the data, with ``(psi, iota, phi_c, d_L)`` solved for.  The discrete group
folded by :mod:`nessai_gw.group_mixture` is a subgroup of that continuous
degeneracy, not the whole of it.

What survives is that the *inclination is determined by the assumed sky
position*.  Solving ``alpha Z(n) + beta conj(Z(n)) = a_0`` in the least-squares
sense for a fiducial response pattern ``a_0`` gives

    cos iota*(n) = (sqrt|alpha| - sqrt|beta|) / (sqrt|alpha| + sqrt|beta|),

a smooth function of ``n`` alone.  On the ET-Delta runs the posterior samples
track it with a correlation of 0.99, and subtracting it turns a strongly
bimodal ``cos theta_jn`` marginal into a single unimodal bump.  That is what
:class:`nessai_gw.reparameterisations.PolarisationEllipseReparameterisation`
does with it.

Equivariance
------------
The map is built on the *idealised* planar triangle implied by the group
action's ``plane_normal`` -- three 60-degree V interferometers sharing one
plane and one vertex -- rather than on the real, slightly non-planar geometry.
That is deliberate: with the idealised tensors

* a quarter turn of the source azimuth about the plane normal sends
  ``Z -> -Z``, hence ``(alpha, beta) -> (-alpha, -beta)`` and leaves
  ``cos iota*`` unchanged;
* reflection through the detector plane sends ``Z -> conj(Z)``, which exchanges
  ``alpha`` and ``beta`` and sends ``cos iota* -> -cos iota*``,

both to machine precision, so the residual coordinate
``cos theta_jn - cos iota*(n)`` transforms under the group exactly as
``cos theta_jn`` itself does and the fold needs no modification.  With the real
ET-EMR tensors those identities hold only to ~1e-3, and the accuracy against
the samples is indistinguishable.
"""

from __future__ import annotations

import numpy as np

from ._geometry import greenwich_mean_sidereal_time

_TWO_PI = 2.0 * np.pi
#: Floor on ``1 - |g12/g11|^2``, the conditioning of the 2x2 solve.  It goes to
#: zero on the great circle in the detector plane, where ``F_x`` vanishes for
#: all three interferometers, ``Z`` is real and the model span collapses to one
#: complex dimension.  ``cos iota*`` tends to zero there continuously, so the
#: floor only stops the division blowing up on that measure-zero circle.
_MIN_CONDITION = 1e-9


def detector_frame_basis(plane_normal, azimuth_offset: float = 0.0):
    """Orthonormal Earth-fixed basis ``[e1, e2, e3]`` with ``e3`` the normal.

    Identical to :func:`nessai_gw.group_mixture._detector_frame_basis`;
    duplicated here to keep this a dependency-light leaf module.
    """
    e3 = np.asarray(plane_normal, dtype=float)
    e3 = e3 / np.linalg.norm(e3)
    e1 = np.array([0.0, 0.0, 1.0]) - e3[2] * e3
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(e3, e1)
    if azimuth_offset:
        c, s = np.cos(azimuth_offset), np.sin(azimuth_offset)
        e1, e2 = c * e1 + s * e2, -s * e1 + c * e2
    return np.stack([e1, e2, e3])


def ideal_triangle_tensors(plane_normal, azimuth_offset: float = 0.0):
    """Detector tensors of a perfect triangle with the given plane normal.

    Three arms at 0, 120 and 240 degrees in the plane; nested interferometer
    ``k`` uses arms ``k`` and ``k + 1``.  The three tensors sum to zero
    identically, which is what makes the null stream -- and hence the
    equivariance described in the module docstring -- exact.
    """
    e1, e2, _ = detector_frame_basis(plane_normal, azimuth_offset)
    arms = [np.cos(g) * e1 + np.sin(g) * e2 for g in np.radians([0.0, 120.0, 240.0])]
    return np.array([
        0.5 * (np.outer(arms[k], arms[k])
               - np.outer(arms[(k + 1) % 3], arms[(k + 1) % 3]))
        for k in range(3)
    ])


def _complex_response(tensors, ra, dec, gmst):
    """``Z = F_+ + i F_x`` at ``psi = 0``, shape ``(..., n_det)``.

    bilby's (Nishizawa et al. 2009) convention, in which the general-``psi``
    response is ``Z(psi) = exp(-2i psi) Z(0)``.
    """
    theta = 0.5 * np.pi - np.asarray(dec, dtype=float)
    phi = np.asarray(ra, dtype=float) - gmst
    theta, phi = np.broadcast_arrays(theta, phi)
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    u = np.stack([cp * ct, ct * sp, -st], axis=-1)
    v = np.stack([-sp, cp, np.zeros_like(cp)], axis=-1)
    # psi = 0: m = -v, n = -u
    m, n = -v, -u
    ep = m[..., :, None] * m[..., None, :] - n[..., :, None] * n[..., None, :]
    ec = m[..., :, None] * n[..., None, :] + n[..., :, None] * m[..., None, :]
    fp = np.einsum('dij,...ij->...d', tensors, ep)
    fc = np.einsum('dij,...ij->...d', tensors, ec)
    return fp + 1j * fc


class PolarisationEllipse:
    """The inclination a sky position forces, given a fiducial signal.

    Parameters
    ----------
    plane_normal : array_like
        Unit normal to the detector plane in the Earth-fixed frame (e.g.
        :data:`nessai_gw.group_mixture.ET_EMR_PLANE_NORMAL`).
    reference_time : float
        Geocentric GPS time of the event.  Its GMST rotates between the
        equatorial and Earth-fixed frames.  The map is evaluated at this single
        instant -- the frozen limit -- which is also what the group action
        assumes.
    fiducial : mapping
        ``ra``, ``dec``, ``psi`` and ``theta_jn`` of a reference signal,
        typically the injection or the maximum-likelihood point.  Only the
        *direction* of the resulting response pattern matters, so no distance
        or amplitude calibration is needed.
    azimuth_offset : float, optional
        In-plane rotation of the detector frame, matching the group action's
        ``azimuth_offset``.  It cancels out of ``cos iota*`` (it only shifts
        ``psi``), and is accepted so the two can be configured identically.
    """

    def __deepcopy__(self, memo):
        # The ellipse is a stateless geometric map plus a *deliberately shared*
        # mutable residual width: the reparameterisation fits it and the group
        # action reads it, and nessai deep-copies reparameterisation specs
        # (nessai.reparameterisations.utils).  Sharing one instance is the
        # point, so a deep copy returns self.
        return self

    def __init__(self, plane_normal, reference_time, fiducial,
                 azimuth_offset: float = 0.0):
        self.plane_normal = np.asarray(plane_normal, dtype=float)
        self.reference_time = float(reference_time)
        self.gmst = greenwich_mean_sidereal_time(self.reference_time)
        self.azimuth_offset = float(azimuth_offset)
        self.tensors = ideal_triangle_tensors(self.plane_normal, self.azimuth_offset)

        missing = {"ra", "dec", "psi", "theta_jn"} - set(fiducial)
        if missing:
            raise ValueError(
                f"`fiducial` is missing {sorted(missing)}; it must give the sky "
                "position, polarisation angle and inclination of a reference signal."
            )
        self.fiducial = {k: float(fiducial[k]) for k in
                         ("ra", "dec", "psi", "theta_jn")}

        c = np.cos(self.fiducial["theta_jn"])
        z0 = _complex_response(
            self.tensors,
            np.array([self.fiducial["ra"]]),
            np.array([self.fiducial["dec"]]),
            self.gmst,
        )[0]
        e = np.exp(-2j * self.fiducial["psi"])
        #: The fiducial response pattern, up to an irrelevant complex scale.
        self.pattern = (1 + c) ** 2 / 4 * e * z0 + (1 - c) ** 2 / 4 * np.conj(e * z0)
        norm = np.abs(self.pattern).max()
        if norm == 0.0:
            raise ValueError("the fiducial signal has an identically zero response")
        self.pattern = self.pattern / norm

    # ------------------------------------------------------------------
    def coefficients(self, ra, dec):
        """The ``(alpha, beta)`` that best reproduce the fiducial pattern."""
        z = _complex_response(self.tensors, ra, dec, self.gmst)
        g11 = np.sum(np.abs(z) ** 2, axis=-1)
        g12 = np.sum(np.conj(z) ** 2, axis=-1)
        p1 = np.sum(np.conj(z) * self.pattern, axis=-1)
        p2 = np.sum(z * self.pattern, axis=-1)
        cond = np.maximum(1.0 - np.abs(g12 / g11) ** 2, _MIN_CONDITION)
        det = cond * g11 ** 2
        return (g11 * p1 - g12 * p2) / det, (-np.conj(g12) * p1 + g11 * p2) / det

    def cos_iota(self, ra, dec):
        """``cos iota*(n)``: the inclination the sky position forces."""
        alpha, beta = self.coefficients(ra, dec)
        ra_, rb = np.sqrt(np.abs(alpha)), np.sqrt(np.abs(beta))
        return (ra_ - rb) / (ra_ + rb)

    __call__ = cos_iota

    # ------------------------------------------------------------------
    # Heteroskedastic residual width.
    #
    # The residual ``cos theta_jn - cos iota*(n)`` is centred on zero but its
    # spread is *not* constant over the sky: it is tight where ``|cos iota*|``
    # is small (the plateau / diffuse mode) and fans out towards the eight
    # face-on points, because there ``cos iota*`` is a steep function of ``n``
    # and a given sky spread maps to a wide ``cos theta_jn`` spread.  Left in,
    # that fan is what keeps the flow-facing coordinate non-Gaussian even after
    # the lock is subtracted.
    #
    # :class:`nessai_gw.reparameterisations.PolarisationEllipseReparameterisation`
    # with ``adaptive_width=True`` divides the residual by ``s(a)``,
    # ``a = |cos iota*(n)|``, refitting it from the live points on every
    # training round (its ``update``) as ``s(a) = sqrt(w0**2 + (w1 a)**2)``
    # sampled onto the knots below.  The knots live here, on the shared ellipse,
    # so :class:`nessai_gw.group_mixture.PrimeSpaceTriangularGroupAction` --
    # handed the same instance -- reads the current width when it reconstructs
    # ``sign(cos theta_jn)`` for ``delta_phase`` (hence :meth:`__deepcopy__`).
    #: Knots of the piecewise-linear width ``s(a)`` in ``a = |cos iota*(n)|``,
    #: linearly interpolated and clamped to the end knots outside the range.
    #: The default -- two knots at height 1 -- is a constant unit width, i.e.
    #: exactly the un-rescaled residual.
    width_knots_a = np.array([0.0, 1.0])
    width_knots_s = np.array([1.0, 1.0])

    def set_width(self, knots_a, knots_s):
        """Set the width knots (see the class notes).  ``knots_a`` must be
        sorted ascending; ``knots_s`` is floored at a small positive value."""
        a = np.asarray(knots_a, dtype=float)
        sv = np.maximum(np.asarray(knots_s, dtype=float), 1e-4)
        if a.ndim != 1 or a.shape != sv.shape or a.size < 2:
            raise ValueError("width knots must be two 1-D arrays of equal size")
        self.width_knots_a = a
        self.width_knots_s = sv

    def residual_width(self, abs_cos_iota):
        """``s(a)`` for numpy input (``np.interp``, so clamped past the knots)."""
        return np.interp(
            np.asarray(abs_cos_iota, dtype=float),
            self.width_knots_a, self.width_knots_s,
        )

    def residual_width_torch(self, abs_cos_iota):
        """:meth:`residual_width` for a torch tensor (linear interp + clamp)."""
        import torch

        ka = torch.as_tensor(
            self.width_knots_a, dtype=abs_cos_iota.dtype,
            device=abs_cos_iota.device,
        )
        ks = torch.as_tensor(
            self.width_knots_s, dtype=abs_cos_iota.dtype,
            device=abs_cos_iota.device,
        )
        a = torch.clamp(abs_cos_iota, float(ka[0]), float(ka[-1]))
        j = torch.clamp(torch.searchsorted(ka, a, right=True), 1, ka.numel() - 1)
        a0, a1 = ka[j - 1], ka[j]
        s0, s1 = ks[j - 1], ks[j]
        t = torch.where(a1 > a0, (a - a0) / (a1 - a0), torch.zeros_like(a))
        return s0 + t * (s1 - s0)

    # ------------------------------------------------------------------
    def cos_iota_torch(self, ra, sin_dec):
        """:meth:`cos_iota` for torch tensors, from ``ra`` and ``sin(dec)``.

        Used by :class:`nessai_gw.group_mixture.PrimeSpaceTriangularGroupAction`
        to recover ``sign(cos theta_jn)`` from the residual prime coordinate.
        """
        import torch

        dtype = ra.dtype
        tensors = torch.as_tensor(self.tensors, dtype=dtype, device=ra.device)
        pattern = torch.as_tensor(
            np.stack([self.pattern.real, self.pattern.imag]),
            dtype=dtype, device=ra.device,
        )
        sd = torch.clamp(sin_dec, -1.0, 1.0)
        cd = torch.sqrt(torch.clamp(1.0 - sd * sd, min=0.0))
        phi = ra - self.gmst
        sp, cp = torch.sin(phi), torch.cos(phi)
        # theta = pi/2 - dec  =>  sin(theta) = cos(dec), cos(theta) = sin(dec)
        u = torch.stack([cp * sd, sd * sp, -cd], dim=-1)
        v = torch.stack([-sp, cp, torch.zeros_like(cp)], dim=-1)
        m, n = -v, -u
        ep = m[..., :, None] * m[..., None, :] - n[..., :, None] * n[..., None, :]
        ec = m[..., :, None] * n[..., None, :] + n[..., :, None] * m[..., None, :]
        fp = torch.einsum('dij,...ij->...d', tensors, ep)
        fc = torch.einsum('dij,...ij->...d', tensors, ec)

        # complex arithmetic written out in real pairs (torch.complex support in
        # older builds is patchy under autograd; nothing here needs gradients).
        zr, zi = fp, fc
        ar, ai = pattern[0], pattern[1]
        g11 = torch.sum(zr * zr + zi * zi, dim=-1)
        g12r = torch.sum(zr * zr - zi * zi, dim=-1)
        g12i = torch.sum(-2.0 * zr * zi, dim=-1)
        p1r = torch.sum(zr * ar + zi * ai, dim=-1)
        p1i = torch.sum(zr * ai - zi * ar, dim=-1)
        p2r = torch.sum(zr * ar - zi * ai, dim=-1)
        p2i = torch.sum(zr * ai + zi * ar, dim=-1)
        cond = torch.clamp(
            1.0 - (g12r * g12r + g12i * g12i) / (g11 * g11), min=_MIN_CONDITION
        )
        det = cond * g11 * g11
        alr = (g11 * p1r - (g12r * p2r - g12i * p2i)) / det
        ali = (g11 * p1i - (g12r * p2i + g12i * p2r)) / det
        ber = (-(g12r * p1r + g12i * p1i) + g11 * p2r) / det
        bei = (-(g12r * p1i - g12i * p1r) + g11 * p2i) / det
        ma = torch.sqrt(torch.sqrt(alr * alr + ali * ali))
        mb = torch.sqrt(torch.sqrt(ber * ber + bei * bei))
        return (ma - mb) / (ma + mb)
