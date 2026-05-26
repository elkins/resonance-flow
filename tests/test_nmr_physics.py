"""
Additional physics-level tests for ResonanceFlow.

Tests in this module verify correct behaviour of the N-H proxy vector
computation and the bonded-exclusion feature of the steric clash loss
under a variety of edge cases.
"""

import jax
import jax.numpy as jnp
import numpy as np

from resonance_flow.losses import estimate_nh_proxy_vectors, get_steric_clash_loss


def test_nh_proxy_vectors_are_unit_length():
    """
    estimate_nh_proxy_vectors must return exactly unit vectors regardless
    of the scale of the input Cα coordinates.
    """
    ca_coords = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [7.6, 0.0, 0.0],
            [11.4, 0.0, 0.0],
            [15.2, 0.0, 0.0],
        ]
    )
    proxies = estimate_nh_proxy_vectors(ca_coords)
    norms = jnp.linalg.norm(proxies, axis=-1)
    assert jnp.allclose(norms, 1.0, atol=1e-5), (
        f"Proxy vectors must be unit length, got norms {norms}"
    )


def test_nh_proxy_vectors_output_shape():
    """
    For N Cα atoms, estimate_nh_proxy_vectors must return exactly N-2 vectors.
    This matches the number of interior residues (indices 1 … N-2 inclusive)
    for which both neighbouring Cα positions are available.
    """
    for seq_len in [4, 8, 16, 32]:
        ca_coords = jax.random.normal(jax.random.PRNGKey(seq_len), (seq_len, 3))
        proxies = estimate_nh_proxy_vectors(ca_coords)
        assert proxies.shape == (seq_len - 2, 3), (
            f"seq_len={seq_len}: expected ({seq_len - 2}, 3), got {proxies.shape}"
        )


def test_nh_proxy_vectors_gradient_flows():
    """
    Gradients must flow through estimate_nh_proxy_vectors so that the RDC
    loss can backpropagate into the Cα coordinate predictions.
    """
    ca_coords = jax.random.normal(jax.random.PRNGKey(0), (6, 3))

    def proxy_norm_sum(coords):
        proxies = estimate_nh_proxy_vectors(coords)
        return jnp.sum(proxies**2)

    grads = jax.grad(proxy_norm_sum)(ca_coords)
    assert grads.shape == ca_coords.shape
    # Gradients must be non-zero for interior residues.
    assert jnp.any(grads != 0.0), "Gradients through proxy vectors must be non-zero"


# ---------------------------------------------------------------------------
# Bonded-exclusion tests
# ---------------------------------------------------------------------------


def test_steric_bonded_exclusion_eliminates_adjacent_penalty():
    """
    BIOPHYSICAL VALIDATION: AMBER / CHARMM 1-2 exclusion convention.

    With exclude_bonded_range=1, directly bonded atoms (sequential index
    separation = 1) must not incur a steric penalty even when their vdW
    radii would otherwise overlap.  This mirrors the standard treatment in
    biomolecular force fields (Cornell et al., JACS 1995; MacKerell et al.,
    J. Phys. Chem. B 1998).
    """
    loss_fn_no_excl = get_steric_clash_loss(exclude_bonded_range=0)
    loss_fn_excl = get_steric_clash_loss(exclude_bonded_range=1)

    # Two atoms at the Cα–Cα distance (3.8 Å) with radii that sum to > 3.8 Å,
    # so they would register as clashing without bonded exclusion.
    positions = jnp.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]])
    atom_radii = jnp.array([2.5, 2.5])  # sum = 5.0 Å > 3.8 Å

    loss_without = loss_fn_no_excl(positions, atom_radii)
    loss_with = loss_fn_excl(positions, atom_radii)

    print(f"Steric loss (no exclusion):      {loss_without:.4f}")
    print(f"Steric loss (1-2 excl.):         {loss_with:.4f}")

    assert loss_without > 0.0, "Without exclusion, overlapping atoms must incur a penalty"
    assert loss_with == 0.0, (
        f"Adjacent (1-2) bonded atoms must not be penalised with exclusion, got {loss_with}"
    )


def test_steric_non_adjacent_still_penalised_with_exclusion():
    """
    Non-adjacent atoms (index separation > exclude_bonded_range) that
    genuinely overlap must still be penalised even when bonded exclusion
    is active.  Only truly bonded pairs should be silenced.
    """
    loss_fn = get_steric_clash_loss(exclude_bonded_range=1)

    # Atom 0 and atom 2: index separation = 2 > 1, so NOT excluded.
    # Place atom 2 very close to atom 0 so they clash.
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],  # adjacent to 0 (excluded)
            [0.5, 0.0, 0.0],  # NOT adjacent to 0 (sep=2): must be penalised
        ]
    )
    atom_radii = jnp.array([1.5, 1.5, 1.5])

    loss = loss_fn(positions, atom_radii)
    print(f"Steric loss (non-adjacent overlap, 1-2 excl.): {loss:.4f}")
    assert loss > 0.0, "Non-adjacent overlapping atoms must still incur a steric penalty"


def test_steric_exclusion_range_2_removes_13_pairs():
    """
    With exclude_bonded_range=2, both 1-2 AND 1-3 pairs (i.e. atoms
    separated by 1 or 2 positions in the chain) are excluded, matching
    the standard 1-3 exclusion in AMBER/CHARMM.
    """
    loss_fn_1 = get_steric_clash_loss(exclude_bonded_range=1)  # excludes 1-2 only
    loss_fn_2 = get_steric_clash_loss(exclude_bonded_range=2)  # excludes 1-2 and 1-3

    # Three collinear atoms: 0 and 2 are a 1-3 pair (sep=2).
    # Place them close enough to clash if not excluded.
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.3, 0.0, 0.0],  # close to atom 0 (sep=2)
        ]
    )
    atom_radii = jnp.array([1.0, 1.0, 1.0])

    loss_excl_1 = loss_fn_1(positions, atom_radii)
    loss_excl_2 = loss_fn_2(positions, atom_radii)

    print(f"Steric loss (1-2 excl.):      {loss_excl_1:.4f}")
    print(f"Steric loss (1-2+1-3 excl.):  {loss_excl_2:.4f}")

    # With only 1-2 exclusion, the 0–2 clash remains.
    assert loss_excl_1 > 0.0, "1-2 exclusion only: 1-3 clash should still be penalised"
    # With 1-2 and 1-3 exclusion, the 0–2 pair is also suppressed.
    assert loss_excl_2 < loss_excl_1, "1-3 exclusion must remove the 0-2 clash penalty"


def test_steric_default_matches_original_behaviour():
    """
    Regression: the default exclude_bonded_range=0 must reproduce the
    original single-diagonal mask (1 - eye), so existing tests are unaffected.
    """
    loss_fn_new = get_steric_clash_loss(exclude_bonded_range=0)

    # Two clashing atoms as used in test_validation.py::test_steric_clash_validation
    pos_clash = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    radii = jnp.array([1.5, 1.5])
    loss = loss_fn_new(pos_clash, radii)
    # overlap = (3.0 - 1.0) = 2.0; both [0,1] and [1,0] contribute → sum = 4+4, /2 = 4
    assert jnp.isclose(loss, 4.0), f"Default behaviour regression: expected 4.0, got {loss}"
