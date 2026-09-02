import inspect

import numpy as np
from nessai.reparameterisations import (
    Reparameterisation,
)
from nessai.reparameterisations.angle import Angle

from .. import nessai_logger

logger = nessai_logger.getChild(__name__)


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


class PolarisationPhaseReparameterisation(Angle):
    """Periodic ``delta_phase = phase + sign(cos theta_jn) * psi`` coordinate.

    For the dominant (2, 2) GW mode the extrinsic likelihood constrains only the
    combination ``phase + sign(cos theta_jn) * psi`` (``phase + psi`` face-on,
    ``phase - psi`` face-off); the orthogonal combination is nearly flat.  This
    reparameterisation replaces the ``phase`` Cartesian pair of
    :class:`~nessai.reparameterisations.angle.Angle` (``angle-2pi``) with a
    single periodic coordinate ``delta_phase``, so the constrained ridge is an
    explicit flow axis and ``psi`` is left as the broad orthogonal coordinate.

    The sign is taken from ``cos theta_jn`` (this is the same combination as
    :class:`DeltaPhaseReparameterisation`).  A flow-fit study on the octomodal
    ET BNS posterior (``examples/validate_polarisation_phase.py``) showed that

    * the *sign-free* ``phase + psi`` is worse than the plain ``phase`` pair
      when the group fold canonicalises the detector-plane hemisphere
      (``beta_f >= 0``, what :meth:`ETTriangleGroupAction.in_fundamental_domain`
      does) rather than face-on/off -- that fold leaves the folded posterior
      face-off dominated, so ``phase + psi`` is the *flat* direction;
    * ``phase + sign(cos theta_jn) * psi`` picks the constrained combination
      regardless of which reflection the fold chose (~1.5 nat lower flow NLL),
      and is exactly invariant modulo ``pi`` under the group reflection
      (``cos theta_jn -> -cos theta_jn``, ``psi -> pi - psi``).

    :class:`nessai_gw.group_mixture.PrimeSpaceTriangularGroupAction` recomputes
    ``delta_phase`` from the group-transformed ``psi`` and ``cos theta_jn`` on
    the encode side.

    The transformation is a rotation of the Cartesian pair at fixed radius, so
    its Jacobian determinant is 1 (the ``chi(2)`` auxiliary radius is inherited
    from :class:`~nessai.reparameterisations.angle.Angle` unchanged).

    Requires ``psi`` and ``theta_jn`` (to build ``delta_phase`` on the forward
    pass and to recover ``phase`` on the inverse).

    Parameters
    ----------
    parameters : Union[str, List[str]]
        Name(s) of the parameter(s); the angle must be ``phase``.
    prior_bounds : Union[list, dict]
        Prior bounds for the parameters; ``phase`` must have a lower bound of 0.
    scale : float, optional
        Angle rescaling before the Cartesian conversion.  ``1.0`` (default)
        gives period ``2*pi`` and keeps the map an exact bijection.  ``2.0``
        (period ``pi``) fits noticeably tighter still by folding out the
        ``phase -> phase + pi`` (2, 2)-mode degeneracy, at the cost of
        discarding that (weak, for BNS) distinction and making the map 2->1.
        The **preferred** way to exploit that degeneracy is instead a
        16-element group action
        (:class:`nessai_gw.group_mixture.ETTriangleGroupAction` with
        ``phase_reflection=True``): the mixture folds ``phase <-> phase + pi``
        and its fitted weight absorbs the inexactness, while this
        reparameterisation stays a clean bijection at ``scale=1.0``.
    prior : {"uniform", "sine", None}, optional
        Passed through to :class:`~nessai.reparameterisations.angle.Angle` on
        versions of nessai that accept it.
    """

    def __init__(
        self,
        parameters=None,
        prior_bounds=None,
        scale=1.0,
        prior="uniform",
        rng=None,
        **kwargs,
    ):
        parent_params = inspect.signature(Angle.__init__).parameters
        call = dict(
            parameters=parameters, prior_bounds=prior_bounds, scale=scale,
            rng=rng,
        )
        if "prior" in parent_params:
            call["prior"] = prior
        # forward any framework-injected kwargs the installed nessai accepts
        # (e.g. ``input_parameters`` on newer versions)
        for key, value in kwargs.items():
            if key in parent_params:
                call[key] = value
        super().__init__(**call)

        # psi and theta_jn are needed on the forward pass
        # (delta_phase = phase + sign(cos theta_jn) * psi) and on the inverse.
        self.requires = ["psi", "theta_jn"]
        if hasattr(self, "inverse_input_parameters"):
            self.inverse_input_parameters = list(
                dict.fromkeys(
                    list(self.inverse_input_parameters or [])
                    + ["psi", "theta_jn"]
                )
            )

        # Rename the flow-facing coordinate ``phase`` -> ``delta_phase``.
        self.prime_parameters = ["delta_phase_x", "delta_phase_y"]
        if hasattr(self, "output_parameters"):
            self.output_parameters = ["delta_phase_x", "delta_phase_y"]

        radial = "delta_phase_radial"
        if getattr(self, "chi", False):
            # nessai <= 0.15: the auxiliary radial lives in ``self.parameters``.
            if len(self.parameters) > 1:
                self.parameters[1] = radial
            # newer nessai: it lives in ``self.auxiliary_parameters``.
            if getattr(self, "auxiliary_parameters", None):
                self.auxiliary_parameters = [radial]

        if not self._zero_bound:
            raise RuntimeError(
                "PolarisationPhaseReparameterisation requires a phase lower "
                "bound of 0 so the inverse wraps modulo 2*pi."
            )

    @staticmethod
    def _psi_sign(x):
        """``sign(cos theta_jn)`` (0 exactly at edge-on -- a measure-zero set)."""
        return np.sign(np.cos(x["theta_jn"]))

    def _rescale_angle(self, x, x_prime, log_j, **kwargs):
        angle = (x["phase"] + self._psi_sign(x) * x["psi"]) * self.scale
        return angle, x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        # ``Angle`` writes the recovered ``delta_phase`` angle into ``x["phase"]``
        # and the radius into the auxiliary parameter, and subtracts ``log r``.
        x, x_prime, log_j = super().inverse_reparameterise(
            x, x_prime, log_j, **kwargs
        )
        x["phase"] = np.mod(
            x["phase"] - self._psi_sign(x) * x["psi"], 2 * np.pi
        )
        return x, x_prime, log_j
