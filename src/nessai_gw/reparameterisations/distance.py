from nessai.reparameterisations import (
    RescaleToBounds,
)

from .. import nessai_logger
from .distance_converters import get_distance_converter

logger = nessai_logger.getChild(__name__)


class DistanceReparameterisation(RescaleToBounds):
    """Reparameterisation for distance.

    If the prior is specified and is one of the known priors then a rescaling
    is applied such that the resulting parameter has a uniform prior. If the
    prior is not specified, then the distance is rescaled an inversion is
    allowed on only the upper bound.

    Known priors
    ------------
    * Power-law: requires specifying the power. See converter kwargs.
    * Uniform-comoving-volume: uses a lookup table to convert to co-moving
    distance.

    Parameters
    ----------
    parameters : str
        Name of distance parameter to rescale.
    prior : {'power-law', 'uniform-comoving-volume'}, optional
        Prior used for the distance parameter
    prior_bounds : tuple
        Tuple of lower and upper bounds on the prior
    converter_kwargs : dict, optional
        Keyword arguments parsed to converter object that converts the distance
        to a parameter with a uniform prior.
    allowed_bounds : list, optional
        List of the allowed bounds for inversion
    kwargs :
        Additional kwargs are parsed to the parent class.
    """

    requires_bounded_prior = True

    def __init__(
        self,
        parameters=None,
        input_parameters=None,
        allowed_bounds=["upper"],
        allow_both=False,
        converter_kwargs=None,
        prior=None,
        prior_bounds=None,
        rng=None,
        **kwargs,
    ):
        # FlowProposal's generic reparameterisation-spec path (an explicit
        # ``reparameterisations={"luminosity_distance": {...}}`` dict, as
        # opposed to GWReparamMixin's own default-reparameterisation path,
        # which calls ``ReparamClass(parameters=p, ...)`` directly) only ever
        # fills in ``input_parameters``, never ``parameters`` -- any
        # ``"parameters"`` key in the config is consumed as a synonym for
        # ``input_parameters`` before this class ever sees it. Without this
        # fallback, ``parameters`` stays ``None`` and the ``len()`` below
        # raises a ``TypeError`` for every explicit "distance" spec.
        if parameters is None:
            parameters = input_parameters
        if isinstance(parameters, str):
            parameters = [parameters]

        if len(parameters) > 1:
            raise RuntimeError(
                "DistanceReparameterisation only supports one parameter"
            )

        dc_class = get_distance_converter(prior)

        if converter_kwargs is None:
            converter_kwargs = {}
        self.distance_converter = dc_class(
            d_min=prior_bounds[parameters[0]][0],
            d_max=prior_bounds[parameters[0]][1],
            **converter_kwargs,
        )

        if self.distance_converter.has_jacobian is False:
            raise RuntimeError(
                "Reparameterisations without tractable Jacobians are no"
                " longer supported. Consider using unit hypercube sampling."
            )

        pre_rescaling = (
            self.distance_converter.to_uniform_parameter,
            self.distance_converter.from_uniform_parameter,
        )

        super().__init__(
            parameters=parameters,
            prior_bounds=prior_bounds,
            pre_rescaling=pre_rescaling,
            rng=rng,
            **kwargs,
        )

        self.detect_edges_kwargs["allowed_bounds"] = allowed_bounds
        self.detect_edges_kwargs["allow_both"] = allow_both
        self.detect_edges_kwargs["x_range"] = self.prior_bounds[
            self.parameters[0]
        ]
