# Polarisation-aware phase coordinate for the ET-triangle group-mixture flow

## Context

**Problem.** Offline inspection of the running `v6` ET-triangle group-mixture run
(`group_flow_inspection/inspect_flow_v6.py` against the live
`nested_sampler_resume.pkl`) shows the folded / canonical live-point set that the
single base flow `q0` must fit has a **sharply cusped, heavy-tailed
`phase_x` / `phase_y` pair**, and in physical space `q0` washes out the real
`phase` banding and the `phase`–`psi` / `phase`–`theta_jn` correlations. `psi`,
`ra`/`dec` and `theta_jn` all fold cleanly; `phase` is the outlier.

**Cause.** For the dominant (2,2) GW mode the extrinsic likelihood constrains
only the combination `phase + sign(cos theta_jn) * psi` (face-on → `phase + psi`,
face-off → `phase − psi`); the orthogonal combination is nearly flat. With
`phase` and `psi` carried as **separate** `angle-2pi` / `angle-pi` Cartesian
pairs, that ridge is a thin diagonal correlation across four prime coordinates,
which a RealNVP fits as a cusped, fat-tailed marginal. The current
`PrimeSpaceETGroupAction` also treats the `phase` pair as fully group-invariant
and never touches it.

**Fix (what the user asked for).** Replace the `phase` pair with a single
periodic coordinate `delta_phase = phase + psi`, so the constrained combination
is an explicit axis and `psi` is left as the (broad) orthogonal coordinate. The
sign that distinguishes face-on/face-off is **not** taken from
`sign(cos theta_jn)` (discontinuous at edge-on, and the ET posterior has real
mass on both sides of `theta_jn = pi/2`) — instead the group fold does it: a
face-off point is assigned to a reflected branch, and the reflection sends
`psi -> pi - psi`, which turns `phase + psi` into `phase - psi + pi`
automatically. So the reparameterisation is **sign-free**, and
`PrimeSpaceETGroupAction` recomputes `delta_phase` from the group-transformed
`psi` on the encode side.

**Intended outcome.** `q0`'s canonical target becomes "one narrow periodic
coordinate (`delta_phase`) + one broad one (`psi`)" instead of a diagonal ridge;
the cusp disappears, `phase` structure and its correlations are reproduced, and
population acceptance for the ET runs improves.

**Decisions already taken (from the user):**
1. 1→1 map: replace `phase` with `delta_phase`, keep `psi` as its own coordinate
   (not a 2→2 sum/difference map — `psi` already folds cleanly).
2. Sign comes from the group's `reflected` state, not `sign(cos theta_jn)` →
   the reparameterisation is sign-free (`delta_phase = phase + psi`), no
   `theta_jn` dependence, no edge-on discontinuity.
3. Hard replace, no switch — the ET group wiring always uses the new coordinate;
   the old `angle-2pi` path for `phase` is removed from that wiring
   (`DeltaPhaseReparameterisation` / `angle-2pi` themselves stay registered for
   other users).
4. Scope: `nessai-gw` only. `xG_inference` keeps its own divergent copy; porting
   it is a separate follow-up (see Risks).

## Math

For a group element decoded as `(k, reflected)`, `k in {0,1,2,3}`
(`ETTriangleGroupAction.__call__`): `phase` is invariant; detector-frame
`psi_f -> (pi - psi_f if reflected else psi_f) + k*pi/2`;
`cos theta_jn -> -cos theta_jn` iff `reflected`.

With the sign-free `delta_phase = phase + psi` and `phase` invariant, the
group-invariant piece is `phase = delta_phase - psi`. Define, in
`PrimeSpaceETGroupAction`:

```
phase_inv       = delta_phase_in - psi_in          # == physical phase, invariant
delta_phase_out = phase_inv + psi_out              # psi_out = action's mapped psi
```

- rotation only (`reflected` false): `psi_out ≈ psi_in + k*pi/2` →
  `delta_phase_out ≈ delta_phase_in + k*pi/2`.
- reflection: `psi_out ≈ pi - psi_in` →
  `delta_phase_out ≈ phase + pi - psi_in` = the face-on constrained combination
  for the folded (now face-on) point.

`delta_phase_out` is computed **exactly** from `psi_out` (which the action
already returns), so it does not rely on the `k*pi/2` shift being a hard
constant — important, because a single ET triangle localises the sky poorly and
the equatorial-`psi` shift picks up a sky-dependent parallactic term (see Risk
R2). The map is a pure rotation of the Cartesian pair at fixed radius → unit
Jacobian.

Inverse round-trip: the wrapper calls with inverted modes; `psi` inverts,
`phase_inv = delta_phase_in - psi_in` is unchanged, so
`delta_phase_back = phase_inv + psi_back` round-trips with `psi`.

## Files to change (all under `/home/ku74xac/Documents/nessai-gw`)

### 1. `src/nessai_gw/reparameterisations/phase.py` — new class

Add `PolarisationPhaseReparameterisation(Angle)` (leave
`DeltaPhaseReparameterisation` untouched). Subclass
`nessai.reparameterisations.angle.Angle` so the coordinate is periodic
(Cartesian pair + chi(2) auxiliary radius), matching how `phase` is handled today.

- `__init__(self, parameters=None, prior_bounds=None, scale=1.0,
  prior="uniform", rng=None)`: `super().__init__(...)` with `parameters=["phase"]`;
  then `self.requires = ["psi"]`; override the generated names so the flow
  coordinate reads `delta_phase`
  (`self.prime_parameters = ["delta_phase_x", "delta_phase_y"]`, rename the
  auxiliary radial entry to `"delta_phase_radial"`, keep `Angle`'s `radial`
  bookkeeping consistent). Assert `self._zero_bound` (phase lower bound 0, so the
  inherited inverse mods by `2*pi`).
- override `_rescale_angle(...)` to return
  `((x["phase"] + x["psi"]) * self.scale, x, x_prime, log_j)`.
  `Angle.reparameterise` (draw `r ~ chi(2)`, write `r*cos`, `r*sin`,
  `log_j += log r`) is inherited unchanged.
- override `inverse_reparameterise(...)`: call
  `super().inverse_reparameterise(...)` (writes `delta_phase_radial`,
  `x["phase"] = atan2(y,x) % (2*pi) / scale` — i.e. the recovered `delta_phase`
  angle — and `log_j -= log r`), then
  `x["phase"] = np.mod(x["phase"] - x["psi"], 2*np.pi)`. `psi` is read from `x`
  (inverted first via `requires` ordering — verify the sort places `psi` before
  `delta_phase` on the inverse; `DeltaPhaseReparameterisation` already relies on
  the same pattern).
- `log_prior` inherited from `Angle` (chi(2) on the radius), identical to
  `angle-2pi`.

Coordinate/name check: coordinate is `delta_phase`, class is
`PolarisationPhaseReparameterisation`; no `chi`-named coordinate introduced.
`scale=1.0` → period `2*pi` (matches today's `angle-2pi` for `phase`); this is
the chosen default (see Risk R3).

### 2. `src/nessai_gw/reparameterisations/__init__.py`

```python
from .phase import (
    DeltaPhaseReparameterisation,
    PolarisationPhaseReparameterisation,
)
known_reparameterisations.add_reparameterisation(
    "polarisation-phase", PolarisationPhaseReparameterisation,
    {"scale": 1.0, "prior": "uniform"},
)
```

Export `PolarisationPhaseReparameterisation` from the package `__init__` (tests
and `group_mixture.py` import it by name). Leave `delta_phase` / `delta-phase`
pointing at the unchanged `DeltaPhaseReparameterisation`.

### 3. `src/nessai_gw/group_mixture.py`

**`_ANGLE_SCALE`** (~line 136): add `"delta_phase": 1.0`. Keep the `"phase"` key
too so `_decode_pair` stays generic, but the ET wiring will now only ever
produce `delta_phase`.

**`et_group_reparameterisations`** (~line 707): in the per-name loop add

```python
elif name == "phase":
    reps[name] = {"reparameterisation": "polarisation-phase"}
```

Update the docstring: `phase -> polarisation-phase` (`delta_phase` pair);
`psi` (`angle-pi`) is now a prerequisite of that coordinate's `requires`.

**`PrimeSpaceETGroupAction`**:

- `bind` (~line 565): the `_pair` discovery loop over `_ANGLE_SCALE` now finds
  `("delta_phase_x", "delta_phase_y")`. Replace the phase-pair handling: require
  `delta_phase` (not `phase`); the "prime space is missing …" message lists
  `delta_phase (delta_phase_x/_y)`. Drop the old `phase` pass-through branch
  entirely (hard replace). Update
  `test_prime_space_action_missing_coordinate`'s expected message.
- `_decode` (~line 619): after decoding `psi, r_psi`, decode
  `delta_phase, r_dphase` from `self._pair["delta_phase"]`. Set the action's
  `phase = delta_phase - psi` (feed the real invariant value through instead of
  the dummy). Stash `aux["r_dphase"]`, `aux["delta_phase"]`, `aux["psi_in"] = psi`.
  `cos_theta_jn` stays a dummy `0` (reflection still handled by the
  `theta_jn_prime` sign flip in `_encode`).
- `_encode` (~line 663): after the `psi` re-encode, compute
  `phase_inv = aux["delta_phase"] - aux["psi_in"]`;
  `delta_phase_new = phase_inv + mapped["psi"]`;
  `out[dpx], out[dpy] = _pair_from_angle(delta_phase_new, aux["r_dphase"],
  _ANGLE_SCALE["delta_phase"])`. No `sign`, no `reflected` term for this
  coordinate (the reflection's effect arrives through `mapped["psi"]`).
- `in_fundamental_domain` (~line 702): unchanged — it only consumes
  `ra`, `sin_dec`, `psi`.

**`_prime_parameter_names`** (~line 795): the probe path already returns
`delta_phase_x/_y` because it builds the real reparameterisations. In the
heuristic fallback branch, map `"phase" -> "delta_phase"` before the
`_ANGLE_SCALE` lookup so it emits `["delta_phase_x", "delta_phase_y"]`.
`_DUMMY_PRIOR_BOUNDS`: `phase` already has `(0, 2*pi)`; `delta_phase` is derived
and needs no entry — add a one-line comment.

**`make_et_group_flow_proposal`** (~line 870): no signature change (hard
replace). `_prime_parameter_names` → `PrimeSpaceETGroupAction(base_action,
prime_names)` binds to `delta_phase_x/_y` automatically; the
`ETGroupFlowProposal.initialise` re-`bind` path is unchanged.

### 4. `ETTriangleGroupAction` (physical-space path) — **no change**

`__call__` already returns `phase` invariant and transforms `psi` /
`cos_theta_jn`. On the `prime_space=False` path nessai's `ReparamBridge` applies
`PolarisationPhaseReparameterisation` to the physically-transformed point, so
`delta_phase_x/_y` come out correct with no extra code. The finite-difference
`test_action_is_measure_preserving` works in physical coords (raw `phase`) and is
unaffected. `ET_TRIANGLE_PARAMETERS` still lists physical `phase` — correct.

## Tests

### `tests/test_reparameterisations/test_phase.py`
Add a `TestPolarisationPhase` block mirroring the `DeltaPhase` tests:
- `test_init`: `prime_parameters == ["delta_phase_x", "delta_phase_y"]`,
  `requires == ["psi"]`, `scale == 1.0`.
- `test_reparameterise_values`: known `phase`, `psi`; assert
  `hypot(delta_phase_x, delta_phase_y)` == drawn radius and
  `atan2(y, x) % (2*pi)` == `(phase + psi) % (2*pi)`.
- `@pytest.mark.integration_test test_invertible`: `reparameterise` → null
  `phase` → `inverse_reparameterise` recovers `phase` mod `2*pi` (rtol 1e-10);
  `log_j` returns to input (the `±log r` cancel).

### `tests/test_group_mixture_proposal.py`
Swap `PRIME_NAMES` / `prime_points` to the `delta_phase_x/_y` layout (drop
`phase_x/_y`; hard replace). Adjust `prime_points` so the `delta_phase` pair uses
random radii like the other pairs.
- `test_prime_space_action_delta_phase_inverse_round_trip` —
  `parametrize("g", range(8))`, `atol/rtol 1e-6` on every prime name.
- `test_prime_space_action_identity` — mode 0 no-op (keep, now covers
  `delta_phase`).
- `test_prime_space_action_delta_phase_radius_preserved` —
  `hypot(delta_phase_x, delta_phase_y)` invariant across all 8 modes.
- `test_prime_space_action_physical_phase_invariant` — decode each mapped point,
  recompute `phase = delta_phase - psi`, assert it matches the original physical
  `phase` across all 8 modes (replaces the old
  `test_prime_space_action_phase_invariant`).
- `test_prime_space_action_delta_phase_matches_physical` — **the consistency
  check**: decode a prime point to physical `(ra, dec, psi, theta_jn, phase,
  geocent_time)`, run `ETTriangleGroupAction`, recompute
  `delta_phase = phase' + psi'` from that output, compare to the `delta_phase`
  implied by `prime_action`'s `delta_phase_x/_y`; `allclose`.
- `test_et_group_reparameterisations` — assert
  `reps["phase"]["reparameterisation"] == "polarisation-phase"`.
- `test_prime_parameter_names` — assert `delta_phase_x` / `delta_phase_y` in
  `prime`, `phase_x` not in `prime` (skip on probe fallback as today).
- `test_prime_space_action_missing_coordinate` — updated message.
- `test_make_et_group_flow_proposal` — unchanged params; still checks MRO,
  `__qualname__`, `_FlowModelClass`.

### `tests/test_group_mixture.py` (recommended)
- `test_folding_concentrates_delta_phase`: load
  `tests/data/bns_et_octomodal_posterior.npz`, fold with the physical action's
  `in_fundamental_domain`, compute `delta_phase = phase + psi` on canonical
  points, assert its circular std (period `2*pi`) is markedly smaller than the
  circular std of raw `phase` on the same points and smaller than pre-folding.
  Loose threshold (ratio < 0.8).

## Verification

From `/home/ku74xac/Documents/nessai-gw` (editable `nessai` dev checkout at
`/home/ku74xac/Documents/nessai` must be active — it provides
`nessai.flowmodel.group_mixture`):

```
uv run python -c "import nessai.flowmodel.group_mixture"
uv run pytest tests/test_reparameterisations/test_phase.py -q
uv run pytest tests/test_group_mixture.py tests/test_group_mixture_proposal.py -q
uv run pytest tests/test_reparameterisations -q -m integration_test
uv run pytest -q                      # full suite, no regressions
uv run python examples/validate_group_mixture.py
```

### Offline flow-inspection check (confirms the cusp is gone)
1. Start a fresh `v7` ET run whose proposal comes from
   `nessai_gw.group_mixture.make_et_group_flow_proposal` (so the prime space
   contains `delta_phase_x` / `delta_phase_y`). A resume of a `v6` checkpoint
   will not pick this up — it must be a fresh run.
2. Copy `group_flow_inspection/inspect_flow_v6.py` to `inspect_flow_v7.py`;
   point `RUN` at the new run dir; in the canonical-folded `want` list
   (~line 264) add `"delta_phase_x"`, `"delta_phase_y"` and drop
   `"phase_x"`/`"phase_y"`. The physical-space corner needs no change (it reads
   `fitted_reparam.prime_parameters`).
3. `uv run python .../inspect_flow_v7.py v7_group` and check: the
   `delta_phase_x`/`delta_phase_y` panel is a smooth unimodal blob (no cusp, no
   fat tails); the physical `phase` marginal and the `phase`–`psi` /
   `phase`–`theta_jn` 2-D structure that `v6` washed out are now reproduced;
   single-branch fallback fraction and mixture-weight entropy stay healthy.

## Risks / open items

- **R1 — `xG_inference` copy.** `xg_inference/src/xg_inference/group_mixture.py`
  is a sibling of the `nessai_gw` module and is what `inspect_flow_v6.py` (and
  possibly the production launch scripts) import. This plan changes only
  `nessai_gw`. If the `v7` launch script imports `xg_inference`, the same four
  edits (new class, `_ANGLE_SCALE`, `PrimeSpaceETGroupAction._decode/_encode`,
  `et_group_reparameterisations`) must be mirrored there. **Confirm which module
  the launch scripts use before running `v7`.**
- **R2 — equatorial-`psi` shift is not a hard constant.** The `_encode`
  recomputation from `mapped["psi"]` is exact, so correctness holds, but the
  per-element `delta_phase` offset has some spread over a poorly-localised sky
  posterior, so the "one shared base flow" assumption is only partly helped.
  Judge from the `v7` inspection; if disappointing, the next step (out of scope)
  is to carry `psi` into the base flow's detector frame too.
- **R3 — period `2*pi` vs `pi`.** Plan uses `scale=1.0` (period `2*pi`, matches
  today's `phase`). Period `pi` (`scale=2.0`) makes the reflection's `pi` shift
  vanish and gives a slightly tighter target, at the cost of discarding the
  `phase` vs `phase + pi` distinction (real but weak for low-spin BNS). Easy to
  switch by changing the registered `scale`.
- **R4 — `sign(0)` / exact edge-on.** The sign-free formula removes the
  discontinuity entirely; nothing branches on `theta_jn`. No action needed —
  noted only to contrast with the rejected `sign(cos theta_jn)` approach.
- **R5 — `Angle` prime prior.** `angle-2pi` registers with `prior="uniform"`;
  `PolarisationPhaseReparameterisation` inherits it. If the current
  `FlowProposal` path trips the deprecated `x_prime_log_prior` warning/raise,
  register with `prior=None` (as the production `psi`/`phase` config does) and
  re-check.
- **R6 — inverse sort order.** Confirm `CombinedReparameterisation` /
  `sort_reparameterisations` places `psi`'s inverse before `delta_phase`'s
  (driven by `requires = ["psi"]`). If not, the `phase = delta_phase - psi`
  step reads a stale `psi`.

## Where this doc lives in nessai-gw

`nessai-gw` has no `docs/` or `notes/` tree (physics derivations live in module
docstrings). On approval, copy this plan to
`nessai-gw/docs/plans/polarisation-phase-reparameterisation.md` (new `docs/plans/`
dir), and fold the "Math" section into the `group_mixture.py` module docstring
next to the existing group-action derivation.
