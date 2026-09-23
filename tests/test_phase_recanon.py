"""Tests for :class:`nessai_gw.group_mixture.PhaseQuarterRecanonicaliser`,
checked against the real prime-space group action and domain predicate."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nessai_gw.group_mixture import (  # noqa: E402
    ETTriangleGroupAction,
    PhaseQuarterRecanonicaliser,
    PrimeSpaceTriangularGroupAction,
    _prime_parameter_names,
    make_et_group_flow_proposal,
)

REFERENCE_TIME = 1187008882.4
BNS_PARAMETERS = [
    "chirp_mass", "mass_ratio", "chi_1", "chi_2", "luminosity_distance",
    "theta_jn", "psi", "phase", "ra", "dec", "geocent_time",
]


@pytest.fixture(scope="module")
def names():
    return _prime_parameter_names(
        BNS_PARAMETERS, REFERENCE_TIME, sky_2d=True, psi_single=True
    )


@pytest.fixture(scope="module")
def action(names):
    base = ETTriangleGroupAction(
        reference_time=REFERENCE_TIME, polarisation_quarter=True
    )
    return PrimeSpaceTriangularGroupAction(base, names)


def _as_dict(t, names):
    return {n: t[:, i] for i, n in enumerate(names)}


def _in_domain(action, t, names):
    return action.in_fundamental_domain(_as_dict(t, names)).to(torch.bool)


@pytest.fixture(scope="module")
def canon(action, names):
    """Random prime points folded to their canonical representative."""
    rng = np.random.default_rng(3)
    n = 3000
    cols = []
    for nm in names:
        if nm in ("sky_u", "sky_v"):
            cols.append(rng.uniform(0.0, 1.0, n))
        else:
            cols.append(rng.uniform(-1.0, 1.0, n))
    x = torch.as_tensor(np.stack(cols, axis=1), dtype=torch.float64)
    out = torch.full_like(x, float("nan"))
    found = torch.zeros(n, dtype=torch.bool)
    for g in range(action._action.group_size):
        modes = torch.full((n,), g, dtype=torch.long)
        img = action(_as_dict(x, names), modes, inverse=True)
        img = torch.stack([img[nm] for nm in names], dim=1)
        ok = _in_domain(action, img, names) & ~found
        out[ok] = img[ok]
        found |= ok
    assert found.all()
    return out


def _enabled(names, cbar=0.13):
    r = PhaseQuarterRecanonicaliser(names)
    r._enabled.fill_(True)
    r._cbar.fill_(cbar)
    return r


def test_disabled_is_identity(names, canon):
    r = PhaseQuarterRecanonicaliser(names)
    assert torch.equal(r.forward(canon), canon)
    assert torch.equal(r.inverse(canon), canon)


def test_bind_requires_coordinates():
    with pytest.raises(RuntimeError, match="psi_prime"):
        PhaseQuarterRecanonicaliser(["delta_phase", "theta_jn_prime"])


@pytest.mark.parametrize("cbar", [0.0, 0.13, 0.37, 0.49])
def test_forward_maps_domain_into_box_and_round_trips(names, action, canon, cbar):
    r = _enabled(names, cbar)
    tt = r.forward(canon)
    i_psi, i_dp = names.index("psi_prime"), names.index("delta_phase")
    assert ((tt[:, i_psi] >= -1) & (tt[:, i_psi] < 1)).all()
    assert ((tt[:, i_dp] >= 0) & (tt[:, i_dp] < 0.5)).all()
    back = r.inverse(tt)
    assert torch.allclose(back, canon, atol=1e-12)
    assert _in_domain(action, back, names).all()


@pytest.mark.parametrize("cbar", [0.0, 0.37])
def test_box_is_exactly_the_image(names, action, canon, cbar):
    """Every point of the target box comes from one canonical point."""
    r = _enabled(names, cbar)
    rng = np.random.default_rng(5)
    tt = canon.clone()
    i_psi, i_dp = names.index("psi_prime"), names.index("delta_phase")
    tt[:, i_psi] = torch.as_tensor(rng.uniform(-1, 1, len(tt)))
    tt[:, i_dp] = torch.as_tensor(rng.uniform(0, 0.5, len(tt)))
    back = r.inverse(tt)
    assert _in_domain(action, back, names).all()
    assert torch.allclose(r.forward(back), tt, atol=1e-12)


def test_inverse_outside_box_leaves_the_domain(names, action, canon):
    r = _enabled(names)
    rng = np.random.default_rng(6)
    n = len(canon)
    i_psi, i_dp = names.index("psi_prime"), names.index("delta_phase")
    tt = canon.clone()
    side = rng.integers(0, 4, n)
    tt[:, i_psi] = torch.as_tensor(np.where(
        side == 0, rng.uniform(1, 3, n),
        np.where(side == 1, rng.uniform(-3, -1, n), rng.uniform(-1, 1, n))))
    tt[:, i_dp] = torch.as_tensor(np.where(
        side == 2, rng.uniform(0.5, 2, n),
        np.where(side == 3, rng.uniform(-2, 0, n), rng.uniform(0, 0.5, n))))
    assert not _in_domain(action, r.inverse(tt), names).any()


def test_unit_jacobian(names, canon):
    r = _enabled(names)
    i_psi, i_dp = names.index("psi_prime"), names.index("delta_phase")
    for row in canon[:20]:
        jac = torch.autograd.functional.jacobian(
            lambda v: r.forward(v.unsqueeze(0))[0], row
        )
        block = jac[[i_psi, i_dp]][:, [i_psi, i_dp]]
        assert torch.linalg.det(block).item() == pytest.approx(1.0)
        assert torch.linalg.det(jac).item() == pytest.approx(1.0)


def _canonical_batch(names, psi, delta, face_on=True):
    t = torch.zeros(len(psi), len(names), dtype=torch.float64)
    t[:, names.index("psi_prime")] = torch.as_tensor(2 * psi / np.pi - 1)
    t[:, names.index("delta_phase")] = torch.as_tensor(delta / np.pi - 1)
    t[:, names.index("theta_jn_prime")] = -0.5 if face_on else 0.5
    return t


def _phase_pinned(names, rng, n=2000, width=0.05, phi0=1.1):
    psi = rng.uniform(0, np.pi / 2, n)
    phi = phi0 + (np.pi / 2) * rng.integers(0, 2, n) + rng.normal(0, width, n)
    return _canonical_batch(names, psi, np.mod(phi + psi, np.pi))


def _delta_pinned(names, rng, n=2000, width=0.05):
    psi = rng.uniform(0, np.pi / 2, n)
    return _canonical_batch(names, psi, np.mod(1.9 + rng.normal(0, width, n), np.pi))


def test_update_switches_on_for_phase_pinned_points(names):
    rng = np.random.default_rng(0)
    r = PhaseQuarterRecanonicaliser(names)
    t = _phase_pinned(names, rng)
    assert r.update(t) is True
    assert r.enabled
    p = r.forward(t)[:, names.index("delta_phase")].numpy()
    # one band, centred in [0, 1/2)
    assert abs(np.mean(p) - 0.25) < 0.02
    assert np.std(p) < 0.03
    assert r.update(t) is False


def test_update_stays_off_for_delta_pinned_points(names):
    rng = np.random.default_rng(1)
    r = PhaseQuarterRecanonicaliser(names)
    assert r.update(_delta_pinned(names, rng)) is False
    assert not r.enabled


def test_update_hysteresis_and_switch_off(names):
    rng = np.random.default_rng(2)
    r = PhaseQuarterRecanonicaliser(names)
    r.update(_phase_pinned(names, rng))
    assert r.enabled
    # phase band broadened so the ratio sits between on and off: stays on
    psi = rng.uniform(0, np.pi / 2, 4000)
    phi = 1.1 + (np.pi / 2) * rng.integers(0, 2, 4000) + rng.normal(0, 0.45, 4000)
    delta = np.mod(phi + psi + rng.normal(0, 0.35, 4000), np.pi)
    t = _canonical_batch(names, psi, delta)
    sp_p, _ = r._circ(np.mod((t[:, names.index("delta_phase")].numpy() + 1)
                             - (t[:, names.index("psi_prime")].numpy() + 1) / 2, 0.5), 0.5)
    sp_d, _ = r._circ(t[:, names.index("delta_phase")].numpy() + 1, 1.0)
    assert r.on_ratio < sp_p / sp_d < r.off_ratio
    r.update(t)
    assert r.enabled
    assert r.update(_delta_pinned(names, rng)) is True
    assert not r.enabled


def test_update_recentres_when_the_band_drifts(names):
    rng = np.random.default_rng(4)
    r = PhaseQuarterRecanonicaliser(names)
    r.update(_phase_pinned(names, rng, phi0=1.1))
    c0 = float(r._cbar)
    assert r.update(_phase_pinned(names, rng, phi0=1.1 + 0.08 * np.pi)) is False
    assert r.update(_phase_pinned(names, rng, phi0=1.1 + 0.2 * np.pi)) is True
    assert float(r._cbar) != c0


def test_face_off_points(names):
    """s = -1 (face-off): delta = phase - psi; still one band after the map."""
    rng = np.random.default_rng(8)
    psi = rng.uniform(0, np.pi / 2, 2000)
    phi = 0.7 + (np.pi / 2) * rng.integers(0, 2, 2000) + rng.normal(0, 0.05, 2000)
    t = _canonical_batch(names, psi, np.mod(phi - psi, np.pi), face_on=False)
    r = PhaseQuarterRecanonicaliser(names)
    assert r.update(t)
    p = r.forward(t)[:, names.index("delta_phase")].numpy()
    assert np.std(p) < 0.03
    assert torch.allclose(r.inverse(r.forward(t)), t, atol=1e-12)


def _has_base_reparam():
    try:
        import inspect

        from nessai.flowmodel.group_mixture import make_group_mixture_flow

        return "base_reparam_factory" in inspect.signature(
            make_group_mixture_flow
        ).parameters
    except ImportError:
        return False


@pytest.mark.skipif(not _has_base_reparam(), reason="nessai lacks base_reparam")
def test_make_et_group_flow_proposal_phase_recanon():
    cls = make_et_group_flow_proposal(
        BNS_PARAMETERS, REFERENCE_TIME, phase_recanon=True
    )
    factory = cls._FlowModelClass.base_reparam_factory
    a, b = factory(), factory()
    assert isinstance(a, PhaseQuarterRecanonicaliser)
    assert a is not b
    off = make_et_group_flow_proposal(BNS_PARAMETERS, REFERENCE_TIME)
    assert off._FlowModelClass.base_reparam_factory is None


@pytest.mark.skipif(not _has_base_reparam(), reason="nessai lacks base_reparam")
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(psi_single=False),
        dict(polarisation_quarter=False),
        dict(phase_coordinates="arg-alpha-beta"),
        dict(prime_space=False),
    ],
)
def test_make_et_group_flow_proposal_phase_recanon_incompatible(kwargs):
    with pytest.raises(RuntimeError, match="phase_recanon"):
        make_et_group_flow_proposal(
            BNS_PARAMETERS, REFERENCE_TIME, phase_recanon=True, **kwargs
        )


# --------------------------------------------------------------------------
# AdaptiveFundamentalDomain

from nessai_gw.group_mixture import AdaptiveFundamentalDomain  # noqa: E402

SEAM_CASES = [
    ((0.0, 0.0, 0.0, 0.0), False),
    ((0.11, 0.37, 0.62, 0.0), False),
    ((0.23, 0.9, 0.05, 0.0), False),
    ((0.07, 0.0, 0.0, 0.31), True),
    ((0.19, 1.43, 0.0, 0.04), True),
]


def _afd(names, action, seams, phase):
    r = AdaptiveFundamentalDomain(action, names, allow_phase_mode=True)
    r._seams.copy_(torch.tensor(seams, dtype=torch.float64))
    r._phase_mode.fill_(phase)
    r._seen.fill_(True)
    return r


def test_afd_identity_until_fitted(names, action, canon):
    r = AdaptiveFundamentalDomain(action, names)
    assert torch.equal(r.forward(canon), canon)
    assert torch.equal(r.inverse(canon), canon)


@pytest.mark.parametrize("seams,phase", SEAM_CASES)
def test_afd_round_trip_and_box(names, action, canon, seams, phase):
    r = _afd(names, action, seams, phase)
    y = r.forward(canon)
    assert r._box_valid(y, phase).all()
    back = r.inverse(y)
    assert torch.allclose(back, canon, atol=1e-9)
    assert _in_domain(action, back, names).all()


@pytest.mark.parametrize("seams,phase", SEAM_CASES)
def test_afd_forward_is_a_group_image(names, action, canon, seams, phase):
    """The pre-box image lies in the same orbit (re-folding it gives back the
    canonical point) and in the shifted target domain."""
    r = _afd(names, action, seams, phase)
    z, found = r._map_to(canon, seams, phase)
    assert found.all()
    assert r._in_target(z, seams, phase).all()
    back, found0 = r._map_to(z, (0.0, 0.0, 0.0, 0.0), False)
    assert found0.all()
    assert torch.allclose(back, canon, atol=1e-9)


@pytest.mark.parametrize("seams,phase", SEAM_CASES[1:])
def test_afd_box_is_exactly_the_image(names, action, canon, seams, phase):
    r = _afd(names, action, seams, phase)
    rng = np.random.default_rng(11)
    y = canon.clone()
    n = len(y)
    iu, ipsi, idp = (names.index(k) for k in ("sky_u", "psi_prime", "delta_phase"))
    y[:, iu] = torch.as_tensor(rng.uniform(0, 0.25, n))
    if phase:
        y[:, ipsi] = torch.as_tensor(rng.uniform(-1, 1, n))
        y[:, idp] = torch.as_tensor(rng.uniform(0, 0.5, n))
    else:
        y[:, ipsi] = torch.as_tensor(rng.uniform(-1, 0, n))
        y[:, idp] = torch.as_tensor(rng.uniform(-1, 0, n))
    back = r.inverse(y)
    assert _in_domain(action, back, names).all()
    assert torch.allclose(r.forward(back), y, atol=1e-9)


@pytest.mark.parametrize("seams,phase", SEAM_CASES[1:])
def test_afd_inverse_outside_box_leaves_domain(names, action, canon, seams, phase):
    r = _afd(names, action, seams, phase)
    rng = np.random.default_rng(12)
    n = len(canon)
    iu, idp = names.index("sky_u"), names.index("delta_phase")
    y = r.forward(canon)
    side = rng.integers(0, 2, n)
    y[:, iu] = torch.where(
        torch.as_tensor(side == 0), torch.as_tensor(rng.uniform(0.25, 1.0, n)),
        y[:, iu])
    hi = 0.5 if phase else 0.0
    y[:, idp] = torch.where(
        torch.as_tensor(side == 1), torch.as_tensor(rng.uniform(hi, hi + 1, n)),
        y[:, idp])
    assert not _in_domain(action, r.inverse(y), names).any()


@pytest.mark.parametrize("seams,phase", SEAM_CASES[1:])
def test_afd_unit_jacobian(names, action, canon, seams, phase):
    r = _afd(names, action, seams, phase)
    for row in canon[:15]:
        jac = torch.autograd.functional.jacobian(
            lambda v: r.forward(v.unsqueeze(0))[0], row
        )
        assert abs(torch.linalg.det(jac).item()) == pytest.approx(1.0, abs=1e-8)


def _d0_points(names, rng, n=4000, u=None, psi=None, delta=None, theta=-0.5):
    t = torch.zeros(n, len(names), dtype=torch.float64)
    t[:, names.index("sky_u")] = torch.as_tensor(
        rng.uniform(0, 0.25, n) if u is None else u)
    t[:, names.index("sky_v")] = torch.as_tensor(rng.uniform(0, 0.5, n))
    t[:, names.index("psi_prime")] = torch.as_tensor(
        rng.uniform(-1, 0, n) if psi is None else psi)
    t[:, names.index("delta_phase")] = torch.as_tensor(
        rng.uniform(-1, 0, n) if delta is None else delta)
    t[:, names.index("theta_jn_prime")] = theta
    return t


def test_afd_update_moves_sky_seam_off_a_peak(names, action):
    rng = np.random.default_rng(20)
    u = np.mod(rng.normal(0.0, 0.015, 4000), 0.25)     # straddles the seam
    r = AdaptiveFundamentalDomain(action, names)
    assert r.update(_d0_points(names, rng, u=u)) is True
    cu = float(r._seams[0])
    assert abs(cu - 0.125) < 0.03
    # the peak is now whole, in the middle of the box
    y = r.forward(_d0_points(names, rng, u=u))
    yu = y[:, names.index("sky_u")].numpy()
    assert abs(np.median(yu) - 0.125) < 0.02
    # a second identical round keeps the frame
    assert r.update(_d0_points(names, rng, u=u)) is False


def test_afd_update_moves_psi_and_delta_seams(names, action):
    rng = np.random.default_rng(21)
    psi = np.mod(rng.normal(0.0, 0.05, 4000) + 1.0, 1.0) - 1.0   # at psi' = -1/0
    r = AdaptiveFundamentalDomain(action, names)
    r.update(_d0_points(names, rng, psi=psi))
    assert abs(float(r._seams[1]) - 0.5) < 0.1
    assert float(r._seams[0]) == 0.0                   # flat sky: untouched
    assert not bool(r._phase_mode)


def test_afd_flat_coordinates_leave_seams_alone(names, action):
    rng = np.random.default_rng(22)
    r = AdaptiveFundamentalDomain(action, names)
    assert r.update(_d0_points(names, rng)) is False
    assert r._identity()


def test_afd_phase_family_for_phase_pinned_points(names, action):
    rng = np.random.default_rng(23)
    n = 4000
    psi = rng.uniform(0, np.pi / 2, n)
    phi = 1.1 + (np.pi / 2) * rng.integers(0, 2, n) + rng.normal(0, 0.05, n)
    delta = np.mod(phi + psi, np.pi)
    pts = _d0_points(names, rng, psi=2 * psi / np.pi - 1, delta=delta / np.pi - 1)
    off = AdaptiveFundamentalDomain(action, names, allow_phase_mode=False)
    off.update(pts)
    assert not bool(off._phase_mode)
    r = AdaptiveFundamentalDomain(action, names, allow_phase_mode=True)
    assert r.update(pts) is True
    assert bool(r._phase_mode)
    p = r.forward(pts)[:, names.index("delta_phase")].numpy()
    assert np.std(p) < 0.03 and 0.1 < np.mean(p) < 0.4


@pytest.mark.skipif(not _has_base_reparam(), reason="nessai lacks base_reparam")
def test_make_et_group_flow_proposal_adaptive_domain():
    cls = make_et_group_flow_proposal(
        BNS_PARAMETERS, REFERENCE_TIME, adaptive_domain=True, phase_recanon=True
    )
    a = cls._FlowModelClass.base_reparam_factory()
    assert isinstance(a, AdaptiveFundamentalDomain)
    assert a.allow_phase_mode
    b = make_et_group_flow_proposal(
        BNS_PARAMETERS, REFERENCE_TIME, adaptive_domain=True
    )._FlowModelClass.base_reparam_factory()
    assert not b.allow_phase_mode
    with pytest.raises(RuntimeError, match="sky_2d"):
        make_et_group_flow_proposal(
            BNS_PARAMETERS, REFERENCE_TIME, adaptive_domain=True, sky_2d=False
        )
