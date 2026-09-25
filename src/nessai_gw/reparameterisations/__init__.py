from nessai.reparameterisations import (
    AnglePair,
    ReparameterisationDict,
    RescaleToBounds,
)
from nessai.reparameterisations import (
    default_reparameterisations as base_reparameterisations,
)

from .chirp_distance import ChirpDistanceReparameterisation
from .distance import DistanceReparameterisation
from .inclination import PolarisationEllipseReparameterisation
from .phase import (
    ArgAlphaBetaReparameterisation,
    DeltaPhaseReparameterisation,
    FittedPhaseRotation,
    PolarisationPhaseReparameterisation,
    SingleAngleReparameterisation,
)
from .sky import EqualAreaSky, RotatedAnglePair
from .spin import AlignedSpinReparameterisation
from .time import DetectorCenterTimeReparameterisation

known_reparameterisations = ReparameterisationDict()
known_reparameterisations.add_reparameterisation(
    "distance",
    DistanceReparameterisation,
    {
        "boundary_inversion": True,
        "detect_edges": True,
        "inversion_type": "duplicate",
    },
)
known_reparameterisations.add_reparameterisation(
    "time",
    RescaleToBounds,
    {"offset": True, "update_bounds": True},
)
known_reparameterisations.add_reparameterisation(
    "sky-ra-dec",
    AnglePair,
    {"convention": "ra-dec"},
)
known_reparameterisations.add_reparameterisation(
    "sky-az-zen",
    AnglePair,
    {"convention": "az-zen"},
)
known_reparameterisations.add_reparameterisation(
    "mass_ratio",
    RescaleToBounds,
    {
        "detect_edges": True,
        "boundary_inversion": True,
        "inversion_type": "duplicate",
        "update_bounds": True,
    },
)
known_reparameterisations.add_reparameterisation(
    "mass",
    RescaleToBounds,
    {"update_bounds": True},
)
known_reparameterisations.add_reparameterisation(
    "aligned-spin",
    AlignedSpinReparameterisation,
    {},
)
known_reparameterisations.add_reparameterisation(
    "delta_phase",
    DeltaPhaseReparameterisation,
    {},
)
known_reparameterisations.add_reparameterisation(
    "delta-phase",
    DeltaPhaseReparameterisation,
    {},
)
known_reparameterisations.add_reparameterisation(
    "polarisation-phase",
    PolarisationPhaseReparameterisation,
    {"scale": 1.0},
)
known_reparameterisations.add_reparameterisation(
    "arg-alpha-beta",
    ArgAlphaBetaReparameterisation,
    {},
)
known_reparameterisations.add_reparameterisation(
    "single-angle",
    SingleAngleReparameterisation,
    {"scale": 2.0},
)
known_reparameterisations.add_reparameterisation(
    "fitted-phase-rotation",
    FittedPhaseRotation,
    {"angle": 0.0, "psi0": 0.0, "phase0": 0.0},
)

known_reparameterisations.add_reparameterisation(
    "chirp-distance",
    ChirpDistanceReparameterisation,
    {},
)

known_reparameterisations.add_reparameterisation(
    "polarisation-ellipse",
    PolarisationEllipseReparameterisation,
    {"scale": 1.0, "adaptive_width": False},
)
known_reparameterisations.add_reparameterisation(
    "polarization-ellipse",
    PolarisationEllipseReparameterisation,
    {"scale": 1.0, "adaptive_width": False},
)

known_reparameterisations.add_reparameterisation(
    "detector-center-time",
    DetectorCenterTimeReparameterisation,
    {},
)
known_reparameterisations.add_reparameterisation(
    "detector-centre-time",
    DetectorCenterTimeReparameterisation,
    {},
)

known_reparameterisations.update(base_reparameterisations)
