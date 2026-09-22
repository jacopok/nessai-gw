import numpy as np
from nessai.reparameterisations import (
    Reparameterisation,
    RescaleToBounds,
)

from .. import nessai_logger
from .._ellipse import PolarisationEllipse
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


class PolarisationDistanceReparameterisation(Reparameterisation):
    r"""Distance measured relative to the value the sky position forces.

    Companion to
    :class:`~nessai_gw.reparameterisations.inclination.PolarisationEllipseReparameterisation`:
    for a single triangular detector in the frozen long-wavelength limit,
    solving for the two-complex-dimensional amplitude block at every sky
    direction ``n`` against a fixed fiducial response also forces a distance
    ``d_L*(n) = d_{L,0} \times`` :meth:`PolarisationEllipse.distance_scale`,
    ``d_{L,0}`` the fiducial signal's own distance
    (``triangle_degeneracy/triangle_manifold_reparameterization.tex``). This
    reparameterisation makes it an axis::

        distance_prime = ln(d_L) - ln(d_L*(n))

    Unlike the inclination lock this residual needs no special handling in
    :class:`~nessai_gw.group_mixture.PrimeSpaceTriangularGroupAction`:
    ``d_L*(n)`` is exactly invariant under the discrete group (a plane
    reflection swaps ``alpha <-> beta``, which swaps the two terms of
    ``sqrt|alpha| + sqrt|beta|`` and leaves it unchanged; a quarter turn
    multiplies both by a unit-modulus phase), the group never touches
    ``luminosity_distance`` at all, and so this coordinate is simply carried
    through the fold unchanged, exactly like an ordinary z-scored parameter.

    ``ra`` and ``dec`` are held fixed by this map, so the Jacobian is that of
    ``d_L -> ln(d_L)`` alone::

        log |d distance_prime / d d_L| = -log(d_L)

    Requires ``ra`` and ``dec`` on both the forward and the inverse pass.

    Parameters
    ----------
    parameters : Union[str, List[str]]
        Name of the parameter; must be ``luminosity_distance``.
    prior_bounds : Union[list, dict], optional
        Unused -- the coordinate is a residual about a sky-dependent centre,
        not a rescaling of the prior range.
    ellipse : PolarisationEllipse
        The sky -> amplitude-block map (shared with the theta_jn lock).
    fiducial_distance : float
        The fiducial signal's own ``d_L*(n_0) = d_{L,0}``: the ellipse's own
        ``fiducial`` dict has no ``luminosity_distance`` field (solving for the
        amplitude block is scale-invariant in the reference response), so this
        calibration point has to be given separately.
    plane_normal, reference_time, fiducial, azimuth_offset : optional
        Forwarded to :class:`~nessai_gw._ellipse.PolarisationEllipse` when
        ``ellipse`` is not given.
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
        fiducial_distance=None,
        prior=None,
        rng=None,
        **kwargs,
    ):
        import inspect

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
                "PolarisationDistanceReparameterisation must act on "
                f"'luminosity_distance'; got {self.parameters}"
            )
        if ellipse is None:
            if plane_normal is None or reference_time is None or fiducial is None:
                raise ValueError(
                    "PolarisationDistanceReparameterisation requires either "
                    "`ellipse` or all of `plane_normal`, `reference_time` and "
                    "`fiducial`."
                )
            ellipse = PolarisationEllipse(
                plane_normal, reference_time, fiducial,
                azimuth_offset=azimuth_offset,
            )
        self.ellipse = ellipse
        if fiducial_distance is None:
            raise ValueError(
                "PolarisationDistanceReparameterisation requires "
                "`fiducial_distance` (the fiducial signal's own distance)."
            )
        self.fiducial_distance = float(fiducial_distance)
        self._log_fiducial_distance = float(np.log(self.fiducial_distance))

        self.prime_parameters = ["distance_lock"]
        if hasattr(self, "output_parameters"):
            self.output_parameters = ["distance_lock"]
        self.requires = ["ra", "dec"]
        if hasattr(self, "inverse_input_parameters"):
            self.inverse_input_parameters = list(
                dict.fromkeys(
                    list(self.inverse_input_parameters or []) + ["ra", "dec"]
                )
            )

    def _log_dstar(self, x):
        return self._log_fiducial_distance + np.log(
            self.ellipse.distance_scale(x["ra"], x["dec"])
        )

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        dl = x["luminosity_distance"]
        x_prime[self.prime_parameters[0]] = np.log(dl) - self._log_dstar(x)
        return x, x_prime, log_j - np.log(dl)

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        u = x_prime[self.prime_parameters[0]]
        log_dl = u + self._log_dstar(x)
        x["luminosity_distance"] = np.exp(log_dl)
        return x, x_prime, log_j + log_dl
