"""Tests for :mod:`nessai_gw.group_mixture`.

The physical check uses the octomodal ET BNS posterior from nessai's
``replay_optimization/BNS_result.json``; a small extract of the relevant
columns is stored in ``tests/data/bns_et_octomodal_posterior.npz``.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nessai_gw.group_mixture import (  # noqa: E402
    ET_EMR_PLANE_NORMAL,
    ET_EMR_VERTEX,
    TRIANGULAR_DETECTOR_GROUP_SIZE,
    TRIANGULAR_DETECTOR_GROUP_SIZE_PHASE,
    TRIANGULAR_DETECTOR_PARAMETERS,
    ETTriangleGroupAction,
    TriangularDetectorGroupAction,
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
def phase_action():
    """16-element action (extra ``phase -> phase + pi`` reflection)."""
    return ETTriangleGroupAction(
        reference_time=REFERENCE_TIME, phase_reflection=True
    )


@pytest.fixture(scope="module")
def random_points():
    rng = np.random.default_rng(1234)
    n = 5000
    return {
        "ra": torch.as_tensor(rng.uniform(0, 2 * np.pi, n)),
        "sin_dec": torch.as_tensor(rng.uniform(-1, 1, n)),
        "cos_theta_jn": torch.as_tensor(rng.uniform(-1, 1, n)),
        "psi": torch.as_tensor(rng.uniform(0, np.pi, n)),
        "phase": torch.as_tensor(rng.uniform(0, 2 * np.pi, n)),
        "geocent_time": torch.as_tensor(
            REFERENCE_TIME + rng.uniform(-0.1, 0.1, n)
        ),
    }


# How to compare each acted-on coordinate: circular (with period) or linear.
METRICS = {
    "ra": ("circ", 2 * np.pi),
    "sin_dec": ("lin", None),
    "cos_theta_jn": ("lin", None),
    "psi": ("circ", np.pi),
    "phase": ("circ", 2 * np.pi),
    "geocent_time": ("lin", None),
}


def _dist(key, a, b):
    kind, period = METRICS[key]
    d = np.abs(np.asarray(a) - np.asarray(b))
    if kind == "lin":
        return d
    d = d % period
    return np.minimum(d, period - d)


def _np(d):
    return {k: v.numpy() for k, v in d.items()}


def _to_coords(ra, dec, psi, theta_jn, phase, geocent_time=REFERENCE_TIME):
    """Physical angles -> the measure-preserving coordinates of the action."""
    return {
        "ra": np.asarray(ra),
        "sin_dec": np.sin(dec),
        "cos_theta_jn": np.cos(theta_jn),
        "psi": np.asarray(psi),
        "phase": np.asarray(phase),
        "geocent_time": np.broadcast_to(
            np.asarray(geocent_time, dtype=float), np.shape(ra)
        ).copy(),
    }


def test_metadata(action):
    assert action.group_size == TRIANGULAR_DETECTOR_GROUP_SIZE == 8
    assert action.phase_reflection is False
    assert action.parameters == TRIANGULAR_DETECTOR_PARAMETERS
    assert TRIANGULAR_DETECTOR_PARAMETERS == [
        "ra",
        "sin_dec",
        "cos_theta_jn",
        "psi",
        "phase",
        "geocent_time",
    ]
    assert np.isclose(np.linalg.norm(action.plane_normal), 1.0)


def test_et_is_thin_subclass():
    """``ETTriangleGroupAction`` only fills in the ET-EMR geometry."""
    et = ETTriangleGroupAction(reference_time=REFERENCE_TIME)
    assert isinstance(et, TriangularDetectorGroupAction)
    assert np.allclose(et.plane_normal, ET_EMR_PLANE_NORMAL)
    assert np.allclose(et.vertex, ET_EMR_VERTEX)

    generic = TriangularDetectorGroupAction(
        REFERENCE_TIME,
        plane_normal=ET_EMR_PLANE_NORMAL,
        vertex=ET_EMR_VERTEX,
    )
    rng = np.random.default_rng(7)
    n = 256
    pts = {
        "ra": torch.as_tensor(rng.uniform(0, 2 * np.pi, n)),
        "sin_dec": torch.as_tensor(rng.uniform(-1, 1, n)),
        "cos_theta_jn": torch.as_tensor(rng.uniform(-1, 1, n)),
        "psi": torch.as_tensor(rng.uniform(0, np.pi, n)),
        "phase": torch.as_tensor(rng.uniform(0, 2 * np.pi, n)),
        "geocent_time": torch.as_tensor(REFERENCE_TIME + rng.uniform(-0.1, 0.1, n)),
    }
    for g in range(TRIANGULAR_DETECTOR_GROUP_SIZE):
        modes = torch.full((n,), g, dtype=torch.long)
        oe = et(pts, modes)
        og = generic(pts, modes)
        for key in METRICS:
            assert torch.allclose(oe[key], og[key])


def test_metadata_phase_reflection(phase_action):
    assert phase_action.group_size == TRIANGULAR_DETECTOR_GROUP_SIZE_PHASE == 16
    assert phase_action.phase_reflection is True
    # class default is unchanged
    assert ETTriangleGroupAction.group_size == 8


@pytest.mark.parametrize("phase_reflection", [False, True])
def test_public_mode_helpers_match_private(phase_reflection):
    action = ETTriangleGroupAction(
        reference_time=REFERENCE_TIME, phase_reflection=phase_reflection
    )
    modes = torch.arange(action.group_size)
    k_pub, refl_pub, pf_pub = action.decode_modes(modes)
    k_priv, refl_priv, pf_priv = ETTriangleGroupAction._decode(modes)
    assert torch.equal(k_pub, k_priv)
    assert torch.equal(refl_pub, refl_priv)
    assert torch.equal(pf_pub, pf_priv)
    if not phase_reflection:
        assert not pf_pub.any()
    else:
        assert torch.equal(pf_pub, modes >= 8)
    assert torch.equal(
        action.invert_modes(modes), ETTriangleGroupAction._invert_modes(modes)
    )
    # every element composed with its inverse is the identity
    assert torch.equal(
        action.invert_modes(action.invert_modes(modes)), modes
    )
    # the inverse is a permutation of 0..group_size-1
    assert torch.equal(
        torch.sort(action.invert_modes(modes)).values, modes
    )


def test_identity_element(action, random_points):
    modes = torch.zeros(len(random_points["ra"]), dtype=torch.long)
    out = _np(action(random_points, modes))
    ref = _np(random_points)
    for key in METRICS:
        assert _dist(key, out[key], ref[key]).max() < 1e-9


@pytest.mark.parametrize("g", range(TRIANGULAR_DETECTOR_GROUP_SIZE))
def test_inverse_round_trip(action, random_points, g):
    modes = torch.full((len(random_points["ra"]),), g, dtype=torch.long)
    transformed = action(random_points, modes)
    recovered = _np(action(transformed, modes, inverse=True))
    ref = _np(random_points)
    for key in METRICS:
        assert _dist(key, recovered[key], ref[key]).max() < 1e-8


def test_outputs_in_prior_ranges(action, random_points):
    for g in range(TRIANGULAR_DETECTOR_GROUP_SIZE):
        modes = torch.full((len(random_points["ra"]),), g, dtype=torch.long)
        out = _np(action(random_points, modes))
        assert np.all((out["ra"] >= 0) & (out["ra"] <= 2 * np.pi))
        assert np.all(np.abs(out["sin_dec"]) <= 1 + 1e-9)
        assert np.all((out["psi"] >= 0) & (out["psi"] <= np.pi))
        assert np.all(np.abs(out["cos_theta_jn"]) <= 1 + 1e-9)


def test_action_is_measure_preserving(action, random_points):
    """The action has unit Jacobian in ``(ra, sin_dec, cos_theta_jn, psi)``.

    Estimated by finite differences on a handful of points for every element.
    """
    keys = ["ra", "sin_dec", "cos_theta_jn", "psi"]
    base = {k: random_points[k][:64].clone() for k in METRICS}
    eps = 1e-6
    for g in range(TRIANGULAR_DETECTOR_GROUP_SIZE):
        n = len(base["ra"])
        modes = torch.full((n,), g, dtype=torch.long)
        f0 = _np(action(base, modes))
        jac = np.zeros((n, len(keys), len(keys)))
        for j, kj in enumerate(keys):
            pert = {k: v.clone() for k, v in base.items()}
            pert[kj] = pert[kj] + eps
            f1 = _np(action(pert, modes))
            for i, ki in enumerate(keys):
                d = f1[ki] - f0[ki]
                # unwrap circular coordinates
                if METRICS[ki][0] == "circ":
                    p = METRICS[ki][1]
                    d = (d + p / 2) % p - p / 2
                jac[:, i, j] = d / eps
        det = np.abs(np.linalg.det(jac))
        assert np.allclose(det, 1.0, atol=1e-3), (g, det.min(), det.max())


def test_fundamental_domain_partitions_every_orbit(action, random_points):
    """Exactly one of the eight images of each point is canonical."""
    n = len(random_points["ra"])
    in_domain = np.zeros((TRIANGULAR_DETECTOR_GROUP_SIZE, n), dtype=bool)
    for g in range(TRIANGULAR_DETECTOR_GROUP_SIZE):
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
            matches = 0
            for k in range(TRIANGULAR_DETECTOR_GROUP_SIZE):
                mk = torch.full((n,), k, dtype=torch.long)
                cand = _np(action(random_points, mk))
                if all(
                    _dist(p, cand[p], composed[p]).max() < 1e-7
                    for p in METRICS
                ):
                    matches += 1
            assert matches == 1, (g, h, matches)


# ---------------------------------------------------------------------------
# 16-element variant (phase_reflection=True)
# ---------------------------------------------------------------------------


def test_phase_reflection_element(phase_action, action, random_points):
    """Modes 8..15 = modes 0..7 with ``phase -> (phase + pi) mod 2pi``."""
    n = len(random_points["ra"])
    for base_g in range(TRIANGULAR_DETECTOR_GROUP_SIZE):
        m8 = torch.full((n,), base_g, dtype=torch.long)
        m16 = torch.full((n,), base_g + 8, dtype=torch.long)
        out8 = _np(phase_action(random_points, m8))
        out16 = _np(phase_action(random_points, m16))
        # the 8-element action agrees with the 16-element action on modes 0..7
        ref8 = _np(action(random_points, m8))
        for key in METRICS:
            assert _dist(key, out8[key], ref8[key]).max() < 1e-9, (base_g, key)
        for key in METRICS:
            if key == "phase":
                shifted = (out8["phase"] + np.pi) % (2 * np.pi)
                assert _dist("phase", out16["phase"], shifted).max() < 1e-9
            else:
                assert _dist(key, out16[key], out8[key]).max() < 1e-9, key


@pytest.mark.parametrize("g", range(TRIANGULAR_DETECTOR_GROUP_SIZE_PHASE))
def test_inverse_round_trip_16(phase_action, random_points, g):
    modes = torch.full((len(random_points["ra"]),), g, dtype=torch.long)
    transformed = phase_action(random_points, modes)
    recovered = _np(phase_action(transformed, modes, inverse=True))
    ref = _np(random_points)
    for key in METRICS:
        assert _dist(key, recovered[key], ref[key]).max() < 1e-8, (g, key)


def test_outputs_in_prior_ranges_16(phase_action, random_points):
    for g in range(TRIANGULAR_DETECTOR_GROUP_SIZE_PHASE):
        modes = torch.full((len(random_points["ra"]),), g, dtype=torch.long)
        out = _np(phase_action(random_points, modes))
        assert np.all((out["ra"] >= 0) & (out["ra"] <= 2 * np.pi))
        assert np.all(np.abs(out["sin_dec"]) <= 1 + 1e-9)
        assert np.all((out["psi"] >= 0) & (out["psi"] <= np.pi))
        assert np.all(np.abs(out["cos_theta_jn"]) <= 1 + 1e-9)
        assert np.all((out["phase"] >= 0) & (out["phase"] <= 2 * np.pi))


def test_action_is_measure_preserving_16(phase_action, random_points):
    """Unit Jacobian in ``(ra, sin_dec, cos_theta_jn, psi, phase)`` for all 16."""
    keys = ["ra", "sin_dec", "cos_theta_jn", "psi", "phase"]
    base = {k: random_points[k][:64].clone() for k in METRICS}
    eps = 1e-6
    for g in range(TRIANGULAR_DETECTOR_GROUP_SIZE_PHASE):
        n = len(base["ra"])
        modes = torch.full((n,), g, dtype=torch.long)
        f0 = _np(phase_action(base, modes))
        jac = np.zeros((n, len(keys), len(keys)))
        for j, kj in enumerate(keys):
            pert = {k: v.clone() for k, v in base.items()}
            pert[kj] = pert[kj] + eps
            f1 = _np(phase_action(pert, modes))
            for i, ki in enumerate(keys):
                d = f1[ki] - f0[ki]
                if METRICS[ki][0] == "circ":
                    p = METRICS[ki][1]
                    d = (d + p / 2) % p - p / 2
                jac[:, i, j] = d / eps
        det = np.abs(np.linalg.det(jac))
        assert np.allclose(det, 1.0, atol=1e-3), (g, det.min(), det.max())


def test_fundamental_domain_partitions_every_orbit_16(
    phase_action, random_points
):
    """Exactly one of the sixteen images of each point is canonical."""
    n = len(random_points["ra"])
    in_domain = np.zeros((TRIANGULAR_DETECTOR_GROUP_SIZE_PHASE, n), dtype=bool)
    for g in range(TRIANGULAR_DETECTOR_GROUP_SIZE_PHASE):
        modes = torch.full((n,), g, dtype=torch.long)
        image = phase_action(random_points, modes)
        in_domain[g] = phase_action.in_fundamental_domain(image).numpy()
    assert np.array_equal(in_domain.sum(axis=0), np.ones(n, dtype=int))


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

    def circ_std(x, period):
        ang = x * (2 * np.pi / period)
        c = np.average(np.cos(ang), weights=weights)
        s = np.average(np.sin(ang), weights=weights)
        return np.sqrt(-2 * np.log(np.hypot(c, s))) * (period / (2 * np.pi))

    assert circ_std(lam % (np.pi / 2), np.pi / 2) < 0.5 * circ_std(
        lam, 2 * np.pi
    )
    frac_north = np.average(beta > 0, weights=weights)
    assert 0.15 < frac_north < 0.85


def test_folding_concentrates_the_posterior(posterior, posterior_action):
    """Mapping every sample to the fundamental domain shrinks the sky
    posterior; the 8-fold orbit of the folded set covers the original."""
    action = posterior_action
    keys = TRIANGULAR_DETECTOR_PARAMETERS
    n = len(posterior["ra"])
    base = _to_coords(
        posterior["ra"],
        posterior["dec"],
        posterior["psi"],
        posterior["theta_jn"],
        posterior["phase"],
    )
    tp = {k: torch.as_tensor(base[k]) for k in keys}

    folded = {k: np.array(base[k]) for k in keys}
    assigned = action.in_fundamental_domain(tp).numpy()
    for g in range(1, TRIANGULAR_DETECTOR_GROUP_SIZE):
        modes = torch.full((n,), g, dtype=torch.long)
        image = _np(action(tp, modes))
        take = action.in_fundamental_domain(
            {k: torch.as_tensor(v) for k, v in image.items()}
        ).numpy() & ~assigned
        for k in keys:
            folded[k] = np.where(take, image[k], folded[k])
        assigned |= take
    assert assigned.all()

    assert action.in_fundamental_domain(
        {k: torch.as_tensor(v) for k, v in folded.items()}
    ).numpy().all()

    assert np.std(folded["ra"]) < 0.6 * np.std(base["ra"])
    assert np.std(folded["sin_dec"]) < 0.6 * np.std(base["sin_dec"])
    assert np.std(folded["cos_theta_jn"]) < 0.6 * np.std(base["cos_theta_jn"])

    orbit_ra = np.concatenate(
        [
            _np(
                action(
                    {k: torch.as_tensor(v) for k, v in folded.items()},
                    torch.full((n,), g, dtype=torch.long),
                )
            )["ra"]
            for g in range(TRIANGULAR_DETECTOR_GROUP_SIZE)
        ]
    )
    assert np.std(orbit_ra) > 0.9 * np.std(base["ra"])


def _circ_std(x, period):
    a = np.asarray(x) * (2 * np.pi / period)
    c, s = np.mean(np.cos(a)), np.mean(np.sin(a))
    return np.sqrt(-2 * np.log(np.hypot(c, s))) * (period / (2 * np.pi))


def test_folding_concentrates_signed_delta_phase(posterior, posterior_action):
    """After folding to the fundamental domain, the polarisation-phase
    combination ``phase + sign(cos theta_jn) * psi`` is tight modulo pi -- much
    tighter than raw ``phase`` -- which is what ``polarisation-phase`` exploits.
    """
    action = posterior_action
    keys = TRIANGULAR_DETECTOR_PARAMETERS
    n = len(posterior["ra"])
    base = _to_coords(
        posterior["ra"], posterior["dec"], posterior["psi"],
        posterior["theta_jn"], posterior["phase"],
    )
    tp = {k: torch.as_tensor(base[k]) for k in keys}
    folded = {k: np.array(base[k]) for k in keys}
    assigned = action.in_fundamental_domain(tp).numpy()
    for g in range(1, TRIANGULAR_DETECTOR_GROUP_SIZE):
        image = _np(action(tp, torch.full((n,), g, dtype=torch.long)))
        take = action.in_fundamental_domain(
            {k: torch.as_tensor(v) for k, v in image.items()}
        ).numpy() & ~assigned
        for k in keys:
            folded[k] = np.where(take, image[k], folded[k])
        assigned |= take

    psi = folded["psi"]
    sign_ct = np.sign(folded["cos_theta_jn"])
    dphase = folded["phase"] + sign_ct * psi

    # tight modulo pi, and much tighter than raw phase
    assert _circ_std(dphase, np.pi) < 0.5
    assert _circ_std(dphase, np.pi) < 0.5 * _circ_std(folded["phase"], np.pi)
    # the sign matters: the opposite sign is markedly less concentrated
    assert _circ_std(folded["phase"] - sign_ct * psi, np.pi) > 1.5 * _circ_std(
        dphase, np.pi
    )


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
        coords = _to_coords(**p)
        pt = {k: torch.tensor([float(v)]) for k, v in coords.items()}
        for g in range(TRIANGULAR_DETECTOR_GROUP_SIZE):
            out = action(pt, torch.tensor([g]))
            ra = float(out["ra"])
            dec = float(np.arcsin(np.clip(float(out["sin_dec"]), -1, 1)))
            psi = float(out["psi"])
            iota = float(np.arccos(np.clip(float(out["cos_theta_jn"]), -1, 1)))
            phase = float(out["phase"])
            c1 = coeffs(ra, dec, psi, iota, phase)
            assert np.abs(c1 - c0).max() / np.abs(c0).max() < 0.02


@pytest.mark.requires("bilby")
def test_geocent_time_preserves_detector_arrival_time():
    """Each element carries geocent_time so the arrival time at the ET
    vertex is unchanged (Santoliquido et al. 2025, Eq. 22)."""
    import bilby

    from nessai_gw.group_mixture import detector_plane_normal, detector_vertex

    ifos = bilby.gw.detector.InterferometerList(["ET"])
    vertex = detector_vertex(ifos)
    action = ETTriangleGroupAction(
        reference_time=REFERENCE_TIME,
        plane_normal=detector_plane_normal(ifos),
        vertex=vertex,
    )

    def arrival(ra, dec, t):
        # geocenter -> ET vertex light-travel, bilby convention.
        gmst = action.gmst
        theta, phi = np.pi / 2 - dec, ra - gmst
        n = np.array(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ]
        )
        return t - np.dot(n, vertex) / 299792458.0

    rng = np.random.default_rng(3)
    for _ in range(30):
        ra = rng.uniform(0, 2 * np.pi)
        dec = np.arcsin(rng.uniform(-1, 1))
        t = REFERENCE_TIME + rng.uniform(-0.05, 0.05)
        pt = {
            k: torch.tensor([v], dtype=torch.float64)
            for k, v in {
                "ra": ra,
                "sin_dec": np.sin(dec),
                "cos_theta_jn": 0.3,
                "psi": 1.0,
                "phase": 2.0,
                "geocent_time": t,
            }.items()
        }
        t0 = arrival(ra, dec, t)
        flipped = False
        for g in range(TRIANGULAR_DETECTOR_GROUP_SIZE):
            out = action(pt, torch.tensor([g]))
            ra_t = float(out["ra"])
            dec_t = float(np.arcsin(np.clip(float(out["sin_dec"]), -1, 1)))
            t_t = float(out["geocent_time"])
            # the single-triangle sky degeneracy is only approximate for a
            # real (finite-size) detector, like the antenna response itself.
            assert abs(arrival(ra_t, dec_t, t_t) - t0) < 1e-3
            if abs(t_t - t) > 1e-4:
                flipped = True
        assert flipped, "reflection should shift geocent_time"


def test_make_flow_factory():
    pytest.importorskip("nessai.flowmodel.group_mixture")
    from nessai_gw.group_mixture import make_et_triangle_group_mixture_flow

    cls = make_et_triangle_group_mixture_flow(reference_time=REFERENCE_TIME)
    assert cls.group_size == TRIANGULAR_DETECTOR_GROUP_SIZE
    assert cls.param_names == TRIANGULAR_DETECTOR_PARAMETERS
    assert callable(cls.group_action_fn)
    assert callable(cls.in_fundamental_domain)

    cls16 = make_et_triangle_group_mixture_flow(
        reference_time=REFERENCE_TIME, phase_reflection=True
    )
    assert cls16.group_size == TRIANGULAR_DETECTOR_GROUP_SIZE_PHASE
