"""Tests for :mod:`nessai_gw.group_mixture`.

The physical check uses the octomodal ET BNS posterior from nessai's
``replay_optimization/BNS_result.json``; a small extract of the relevant
columns is stored in ``tests/data/bns_et_octomodal_posterior.npz``.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nessai_gw.group_mixture import (  # noqa: E402
    ET_TRIANGLE_GROUP_SIZE,
    ET_TRIANGLE_PARAMETERS,
    ETTriangleGroupAction,
    detector_plane_normal,
)

REFERENCE_TIME = 1187008882.4
DATA = (
    __import__("pathlib").Path(__file__).parent
    / "data"
    / "bns_et_octomodal_posterior.npz"
)


@pytest.fixture(scope="module")
def action():
    return ETTriangleGroupAction(reference_time=REFERENCE_TIME)


@pytest.fixture(scope="module")
def random_points():
    rng = np.random.default_rng(1234)
    n = 5000
    return {
        "ra": torch.as_tensor(rng.uniform(0, 2 * np.pi, n)),
        "dec": torch.as_tensor(np.arcsin(rng.uniform(-1, 1, n))),
        "psi": torch.as_tensor(rng.uniform(0, np.pi, n)),
        "theta_jn": torch.as_tensor(np.arccos(rng.uniform(-1, 1, n))),
        "phase": torch.as_tensor(rng.uniform(0, 2 * np.pi, n)),
    }


def _ang_dist(a, b, period):
    d = np.abs(np.asarray(a) - np.asarray(b)) % period
    return np.minimum(d, period - d)


PERIODS = {
    "ra": 2 * np.pi,
    "dec": np.pi,
    "psi": np.pi,
    "theta_jn": np.pi,
    "phase": 2 * np.pi,
}


def _np(d):
    return {k: v.numpy() for k, v in d.items()}


def test_metadata(action):
    assert action.group_size == ET_TRIANGLE_GROUP_SIZE == 8
    assert action.parameters == ET_TRIANGLE_PARAMETERS
    assert np.isclose(np.linalg.norm(action.plane_normal), 1.0)


def test_identity_element(action, random_points):
    modes = torch.zeros(len(random_points["ra"]), dtype=torch.long)
    out = _np(action(random_points, modes))
    ref = _np(random_points)
    for key, period in PERIODS.items():
        assert _ang_dist(out[key], ref[key], period).max() < 1e-9


@pytest.mark.parametrize("g", range(ET_TRIANGLE_GROUP_SIZE))
def test_inverse_round_trip(action, random_points, g):
    modes = torch.full((len(random_points["ra"]),), g, dtype=torch.long)
    transformed = action(random_points, modes)
    recovered = _np(action(transformed, modes, inverse=True))
    ref = _np(random_points)
    for key, period in PERIODS.items():
        assert _ang_dist(recovered[key], ref[key], period).max() < 1e-8


def test_outputs_in_prior_ranges(action, random_points):
    for g in range(ET_TRIANGLE_GROUP_SIZE):
        modes = torch.full((len(random_points["ra"]),), g, dtype=torch.long)
        out = _np(action(random_points, modes))
        assert np.all((out["ra"] >= 0) & (out["ra"] <= 2 * np.pi))
        assert np.all((out["dec"] >= -np.pi / 2) & (out["dec"] <= np.pi / 2))
        assert np.all((out["psi"] >= 0) & (out["psi"] <= np.pi))
        assert np.all((out["theta_jn"] >= 0) & (out["theta_jn"] <= np.pi))


def test_fundamental_domain_partitions_every_orbit(action, random_points):
    """Exactly one of the eight images of each point is canonical."""
    n = len(random_points["ra"])
    in_domain = np.zeros((ET_TRIANGLE_GROUP_SIZE, n), dtype=bool)
    for g in range(ET_TRIANGLE_GROUP_SIZE):
        modes = torch.full((n,), g, dtype=torch.long)
        image = action(random_points, modes)
        in_domain[g] = action.in_fundamental_domain(image).numpy()
    assert np.array_equal(in_domain.sum(axis=0), np.ones(n, dtype=int))


def test_group_closure(action, random_points):
    """Composing two elements gives another element of the group."""
    n = len(random_points["ra"])
    for g in [1, 2, 3, 5, 6]:
        for h in [1, 3, 4, 7]:
            mg = torch.full((n,), g, dtype=torch.long)
            mh = torch.full((n,), h, dtype=torch.long)
            composed = _np(action(action(random_points, mh), mg))
            # find the single element reproducing the composition
            matches = 0
            for k in range(ET_TRIANGLE_GROUP_SIZE):
                mk = torch.full((n,), k, dtype=torch.long)
                cand = _np(action(random_points, mk))
                if all(
                    _ang_dist(cand[p], composed[p], per).max() < 1e-7
                    for p, per in PERIODS.items()
                ):
                    matches += 1
            assert matches == 1, (g, h, matches)


# ---------------------------------------------------------------------------
# Physical checks against the octomodal ET BNS posterior
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def posterior():
    if not DATA.exists():
        pytest.skip(f"missing test data: {DATA}")
    d = np.load(DATA)
    return {k: d[k] for k in d.files}


@pytest.fixture(scope="module")
def posterior_action(posterior):
    return ETTriangleGroupAction(
        reference_time=float(np.mean(posterior["geocent_time"]))
    )


def test_detector_frame_azimuth_modes_are_quarter_turns(
    posterior, posterior_action
):
    """The sky posterior peaks are spaced by pi/2 in detector-frame azimuth
    and are bimodal in +-beta -- i.e. the eight modes of Eqs. (63)-(65)."""
    tp = {
        k: torch.as_tensor(posterior[k]) for k in ["ra", "dec", "psi"]
    }
    lam, beta, _ = posterior_action._to_frame(tp["ra"], tp["dec"], tp["psi"])
    lam = np.mod(lam.numpy(), 2 * np.pi)
    beta = beta.numpy()
    weights = np.exp(
        posterior["log_likelihood"] - posterior["log_likelihood"].max()
    )

    # Folding the azimuth modulo pi/2 collapses the four longitude peaks.
    def circ_std(x, period):
        ang = x * (2 * np.pi / period)
        c = np.average(np.cos(ang), weights=weights)
        s = np.average(np.sin(ang), weights=weights)
        return np.sqrt(-2 * np.log(np.hypot(c, s))) * (period / (2 * np.pi))

    assert circ_std(lam % (np.pi / 2), np.pi / 2) < 0.5 * circ_std(
        lam, 2 * np.pi
    )
    # Both sides of the detector plane are populated.
    frac_north = np.average(beta > 0, weights=weights)
    assert 0.15 < frac_north < 0.85


def test_folding_concentrates_the_posterior(posterior, posterior_action):
    """Mapping every sample to the fundamental domain shrinks the sky
    posterior; the 8-fold orbit of the folded set covers the original."""
    action = posterior_action
    keys = ET_TRIANGLE_PARAMETERS
    n = len(posterior["ra"])
    tp = {k: torch.as_tensor(posterior[k]) for k in keys}

    folded = {k: np.array(posterior[k]) for k in keys}
    assigned = action.in_fundamental_domain(tp).numpy()
    for g in range(1, ET_TRIANGLE_GROUP_SIZE):
        modes = torch.full((n,), g, dtype=torch.long)
        image = _np(action(tp, modes))
        take = action.in_fundamental_domain(
            {k: torch.as_tensor(v) for k, v in image.items()}
        ).numpy() & ~assigned
        for k in keys:
            folded[k] = np.where(take, image[k], folded[k])
        assigned |= take
    assert assigned.all()

    # every folded point really is canonical
    assert action.in_fundamental_domain(
        {k: torch.as_tensor(v) for k, v in folded.items()}
    ).numpy().all()

    # the sky localisation is much tighter after folding
    assert np.std(folded["ra"]) < 0.6 * np.std(posterior["ra"])
    assert np.std(folded["dec"]) < 0.6 * np.std(posterior["dec"])
    assert np.std(folded["theta_jn"]) < 0.6 * np.std(posterior["theta_jn"])

    # spreading the folded set back over the orbit recovers the sky extent
    orbit_ra = np.concatenate(
        [
            _np(
                action(
                    {k: torch.as_tensor(v) for k, v in folded.items()},
                    torch.full((n,), g, dtype=torch.long),
                )
            )["ra"]
            for g in range(ET_TRIANGLE_GROUP_SIZE)
        ]
    )
    assert np.std(orbit_ra) > 0.9 * np.std(posterior["ra"])


@pytest.mark.requires("bilby")
def test_long_wavelength_response_invariance():
    """Each group element leaves the long-wavelength single-triangle antenna
    response (hence the extrinsic likelihood) invariant."""
    import bilby

    ifos = bilby.gw.detector.InterferometerList(["ET"])
    normal = detector_plane_normal(ifos)
    action = ETTriangleGroupAction(
        reference_time=REFERENCE_TIME, plane_normal=normal
    )

    def coeffs(ra, dec, psi, iota, phase):
        out = []
        for ifo in ifos:
            fp = ifo.antenna_response(ra, dec, REFERENCE_TIME, psi, "plus")
            fc = ifo.antenna_response(ra, dec, REFERENCE_TIME, psi, "cross")
            out.append(
                (
                    fp * (1 + np.cos(iota) ** 2) / 2
                    - 1j * fc * np.cos(iota)
                )
                * np.exp(-2j * phase)
            )
        return np.array(out)

    rng = np.random.default_rng(0)
    for _ in range(50):
        p = dict(
            ra=rng.uniform(0, 2 * np.pi),
            dec=np.arcsin(rng.uniform(-1, 1)),
            psi=rng.uniform(0, np.pi),
            theta_jn=np.arccos(rng.uniform(-1, 1)),
            phase=rng.uniform(0, 2 * np.pi),
        )
        c0 = coeffs(p["ra"], p["dec"], p["psi"], p["theta_jn"], p["phase"])
        pt = {k: torch.tensor([v]) for k, v in p.items()}
        for g in range(ET_TRIANGLE_GROUP_SIZE):
            out = action(pt, torch.tensor([g]))
            c1 = coeffs(*[float(out[k]) for k in ET_TRIANGLE_PARAMETERS])
            assert np.abs(c1 - c0).max() / np.abs(c0).max() < 0.02


def test_make_flow_factory():
    pytest.importorskip("nessai.flowmodel.group_mixture")
    from nessai_gw.group_mixture import make_et_triangle_group_mixture_flow

    cls = make_et_triangle_group_mixture_flow(reference_time=REFERENCE_TIME)
    assert cls.group_size == ET_TRIANGLE_GROUP_SIZE
    assert cls.param_names == ET_TRIANGLE_PARAMETERS
    assert callable(cls.group_action_fn)
    assert callable(cls.in_fundamental_domain)
