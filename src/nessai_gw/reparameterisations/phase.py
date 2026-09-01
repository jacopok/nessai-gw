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
    """Periodic ``delta_phase = phase + psi`` coordinate for the ET group flow.

    For the dominant (2, 2) GW mode the extrinsic likelihood constrains only the
    combination ``phase + sign(cos theta_jn) * psi`` (``phase + psi`` face-on,
    ``phase - psi`` face-off); the orthogonal combination is nearly flat.  This
    reparameterisation replaces the ``phase`` Cartesian pair of
    :class:`~nessai.reparameterisations.angle.Angle` (``angle-2pi``) with a
    single periodic coordinate ``delta_phase = phase + psi``, so the constrained
    ridge is an explicit axis and ``psi`` is left as the broad orthogonal
    coordinate.

    The map is **sign-free**: it does not read ``sign(cos theta_jn)`` (which is
    discontinuous at edge-on).  The face-on / face-off distinction is instead
    carried by the group fold -- a face-off point is assigned to a reflected
    branch, whose ``psi -> pi - psi`` turns ``phase + psi`` into
    ``phase - psi + pi`` automatically -- and
    :class:`nessai_gw.group_mixture.PrimeSpaceETGroupAction` recomputes
    ``delta_phase`` from the group-transformed ``psi`` on the encode side.

    The transformation is a rotation of the Cartesian pair at fixed radius, so
    its Jacobian determinant is 1 (the ``chi(2)`` auxiliary radius is inherited
    from :class:`~nessai.reparameterisations.angle.Angle` unchanged).

    Requires ``psi`` (both to build ``delta_phase`` on the forward pass and to
    recover ``phase`` on the inverse).

    Parameters
    ----------
    parameters : Union[str, List[str]]
        Name(s) of the parameter(s); the angle must be ``phase``.
    prior_bounds : Union[list, dict]
        Prior bounds for the parameters; ``phase`` must have a lower bound of 0.
    scale : float, optional
        Angle rescaling before the Cartesian conversion.  ``1.0`` (default)
        gives period ``2*pi``, matching ``angle-2pi`` for ``phase``.
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

        # psi is needed on the forward pass (delta_phase = phase + psi) and on
        # the inverse (phase = delta_phase - psi).
        self.requires = ["psi"]
        if hasattr(self, "inverse_input_parameters"):
            self.inverse_input_parameters = list(
                dict.fromkeys(
                    list(self.inverse_input_parameters or []) + ["psi"]
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

    def _rescale_angle(self, x, x_prime, log_j, **kwargs):
        angle = (x["phase"] + x["psi"]) * self.scale
        return angle, x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        # ``Angle`` writes the recovered ``delta_phase`` angle into ``x["phase"]``
        # and the radius into the auxiliary parameter, and subtracts ``log r``.
        x, x_prime, log_j = super().inverse_reparameterise(
            x, x_prime, log_j, **kwargs
        )
        x["phase"] = np.mod(x["phase"] - x["psi"], 2 * np.pi)
        return x, x_prime, log_j
