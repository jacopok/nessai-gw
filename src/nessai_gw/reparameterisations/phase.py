import inspect

import numpy as np
from nessai.reparameterisations import (
    Reparameterisation,
)

from .. import nessai_logger

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
    prior : optional
        Accepted for registry compatibility and ignored (the flow models the
        prime coordinate directly).
    """

    one_to_one = False

    def __init__(
        self,
        parameters=None,
        prior_bounds=None,
        scale=1.0,
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
        if hasattr(self, "inverse_input_parameters"):
            self.inverse_input_parameters = list(
                dict.fromkeys(
                    list(self.inverse_input_parameters or [])
                    + ["psi", "theta_jn"]
                )
            )
        self._log_j = float(np.log(self.scale / np.pi))

    @staticmethod
    def _psi_sign(x):
        """``sign(cos theta_jn)`` (0 exactly at edge-on -- a measure-zero set)."""
        return np.sign(np.cos(x["theta_jn"]))

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        angle = np.mod(
            (x["phase"] + self._psi_sign(x) * x["psi"]) * self.scale, _TWO_PI
        )
        x_prime[self.prime_parameters[0]] = angle / np.pi - 1.0
        return x, x_prime, log_j + self._log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        angle = np.mod(
            (x_prime[self.prime_parameters[0]] + 1.0) * np.pi, _TWO_PI
        )
        x["phase"] = np.mod(
            angle / self.scale - self._psi_sign(x) * x["psi"], _TWO_PI
        )
        return x, x_prime, log_j - self._log_j


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
