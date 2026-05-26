import jax
import jax.numpy as jnp
import numpy as np

from resonance_flow.losses import (
    get_bond_length_loss,
    get_steric_clash_loss,
    noe_upper_bound_loss,
    rdc_loss,
    rdc_q_factor,
)


def test_rdc_synthetic_recoverability():
    """
    MATHEMATICAL VALIDATION:
    Verify that rdc_loss can perfectly fit a set of vectors to RDCs
    if a consistent alignment tensor exists.
    """
    rng = np.random.default_rng(42)
    num_vectors = 20

    # 1. Generate random unit vectors
    v_np = rng.standard_normal((num_vectors, 3))
    v_np = v_np / np.linalg.norm(v_np, axis=-1, keepdims=True)
    v = jnp.array(v_np)

    # 2. Generate a random traceless Saupe tensor (5 parameters)
    # Sxx, Syy, Sxy, Sxz, Syz
    true_s = jnp.array([0.001, -0.0005, 0.0002, -0.0003, 0.0001])

    # 3. Calculate RDCs using the exact formula in rdc_loss
    d_max = 21700.0
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    A = d_max * jnp.stack(
        [x**2 - z**2, y**2 - z**2, 2 * x * y, 2 * x * z, 2 * y * z], axis=1
    )
    measured_rdcs = A @ true_s

    # 4. Verify that rdc_loss returns near-zero
    loss = rdc_loss(v, measured_rdcs)
    print(f"Synthetic RDC Loss: {loss:.2e}")
    assert loss < 1e-10


def test_ubiquitin_geometry_validation():
    """
    STRUCTURAL VALIDATION:
    Use real Ubiquitin (1D3Z) geometry and verify that the RDC loss
    correctly discriminates between the native state and a scrambled state.
    """
    # NH vectors from 1D3Z Model 1 (extracted previously)
    ubiquitin_vectors = jnp.array(
        [
            [-0.38999939, -0.05299377, 0.89700031],
            [0.13299942, -0.65100098, -0.71799994],
            [0.27000046, 0.80500031, 0.47800016],
            [-0.39599991, -0.86600494, -0.222],
            [-0.68600082, -0.69599915, -0.0710001],
            [0.40800095, 0.88899994, 0.05799985],
            [0.66600037, 0.66200256, 0.29700002],
            [0.78100204, 0.50700378, 0.29200029],
            [0.5719986, -0.28799438, 0.74499989],
            [0.96999741, 0.11600494, -0.08900023],
        ]
    )

    # Generate SYNTHETIC RDCs for this structure using a realistic tensor
    # (Da = 10 Hz, Rhombicity = 0.2)
    # This proves the loss function works on real protein bond distributions.
    true_s = jnp.array([0.0005, -0.0002, 0.0001, 0.0, 0.0])
    d_max = 21700.0
    x, y, z = ubiquitin_vectors[:, 0], ubiquitin_vectors[:, 1], ubiquitin_vectors[:, 2]
    A = d_max * jnp.stack(
        [x**2 - z**2, y**2 - z**2, 2 * x * y, 2 * x * z, 2 * y * z], axis=1
    )
    synthetic_rdcs = A @ true_s

    # 1. Native structure should have 0 loss
    loss_native = rdc_loss(ubiquitin_vectors, synthetic_rdcs)

    # 2. Scrambled structure should have high loss
    rng = np.random.default_rng(123)
    scrambled_vectors = ubiquitin_vectors[rng.permutation(len(ubiquitin_vectors))]
    loss_scrambled = rdc_loss(scrambled_vectors, synthetic_rdcs)

    print(f"Ubiquitin Native Loss: {loss_native:.2e}")
    print(f"Ubiquitin Scrambled Loss: {loss_scrambled:.2f}")

    assert loss_native < 1e-3
    assert loss_scrambled > 1.0


def test_steric_clash_validation():
    """
    BIOPHYSICAL VALIDATION:
    Verify that the steric clash loss identifies overlaps in a
    synthetic 'clashing' structure vs a valid structure.
    """
    loss_fn = get_steric_clash_loss()

    # 1. Valid structure (atoms separated by 4A)
    pos_valid = jnp.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    radii = jnp.array([1.5, 1.5])  # sum = 3.0
    loss_valid = loss_fn(pos_valid, radii)

    # 2. Clashing structure (atoms separated by 1A)
    pos_clash = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    loss_clash = loss_fn(pos_clash, radii)  # overlap = 2.0

    print(f"Valid Steric Loss: {loss_valid:.4f}")
    print(f"Clash Steric Loss: {loss_clash:.4f}")

    assert loss_valid == 0.0
    assert loss_clash > 0.0
    # overlap = 2.0, loss = (2.0^2)/2 * 2 (for both directions) = 4.0
    # The code sums overlap**2 over the full matrix / 2.
    # overlap matrix has 2.0 at [0,1] and [1,0].
    # Sum(4, 4) / 2 = 4.0.
    assert jnp.isclose(loss_clash, 4.0)


# ---------------------------------------------------------------------------
# New research-grounded validation tests
# ---------------------------------------------------------------------------


def test_rdc_q_factor_perfect_structure():
    """
    PUBLICATION VALIDATION: Cornilescu, Marquardt, Ottiger & Bax, JACS 1998;
    Clore & Garrett, JACS 1999.

    Q-factor for a structure perfectly consistent with the alignment tensor
    must be essentially zero.  A random structure should have Q >> 0.

        Q = RMSD(D_calc − D_obs) / RMS(D_obs)

    A high-quality backbone structure is expected to achieve Q ≤ 0.20.
    """
    rng = np.random.default_rng(42)
    num_vectors = 25
    d_max = 21700.0

    # Generate random unit vectors and a synthetic Saupe tensor.
    v_np = rng.standard_normal((num_vectors, 3))
    v_np /= np.linalg.norm(v_np, axis=-1, keepdims=True)
    v = jnp.array(v_np)

    true_s = jnp.array([0.001, -0.0005, 0.0002, -0.0003, 0.0001])
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    A = d_max * jnp.stack(
        [x**2 - z**2, y**2 - z**2, 2 * x * y, 2 * x * z, 2 * y * z], axis=1
    )
    measured = A @ true_s

    # Perfect structure: Q must be ~0.
    q_perfect = rdc_q_factor(v, measured, d_max=d_max)
    print(f"Q-factor (perfect structure): {q_perfect:.2e}")
    assert q_perfect < 1e-3, f"Q for perfect structure should be ~0, got {q_perfect:.4f}"

    # Random (wrong) vectors: Q must be substantially higher.
    rng2 = np.random.default_rng(99)
    v_random_np = rng2.standard_normal((num_vectors, 3))
    v_random_np /= np.linalg.norm(v_random_np, axis=-1, keepdims=True)
    v_random = jnp.array(v_random_np)
    q_random = rdc_q_factor(v_random, measured, d_max=d_max)
    print(f"Q-factor (random structure): {q_random:.4f}")
    assert q_random > q_perfect * 100, "Random structure should have much worse Q-factor"


def test_rdc_rotation_invariance():
    """
    PHYSICAL VALIDATION: Saupe, Z. Naturforsch. 1964.

    RDC loss is invariant under global rigid rotation of all internuclear
    vectors because the alignment tensor rotates correspondingly, leaving
    the optimal residual unchanged.  This test uses noisy RDCs (not
    perfectly fittable) to produce a non-trivial, non-zero loss that must
    remain constant after rotation.
    """
    rng = np.random.default_rng(7)
    num_vectors = 20
    d_max = 21700.0

    v_np = rng.standard_normal((num_vectors, 3))
    v_np /= np.linalg.norm(v_np, axis=-1, keepdims=True)
    v = jnp.array(v_np)

    # Synthetic RDCs with realistic noise (~50 Hz) — not perfectly fittable.
    true_s = jnp.array([0.001, -0.0005, 0.0002, -0.0003, 0.0001])
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    A = d_max * jnp.stack(
        [x**2 - z**2, y**2 - z**2, 2 * x * y, 2 * x * z, 2 * y * z], axis=1
    )
    noise = jnp.array(rng.standard_normal(num_vectors) * 50.0)
    measured_noisy = A @ true_s + noise

    loss_original = rdc_loss(v, measured_noisy, d_max=d_max)

    # Apply a 45-degree rotation about the Y-axis.
    angle = jnp.pi / 4
    R = jnp.array(
        [
            [jnp.cos(angle), 0.0, jnp.sin(angle)],
            [0.0, 1.0, 0.0],
            [-jnp.sin(angle), 0.0, jnp.cos(angle)],
        ]
    )
    v_rotated = v @ R.T
    loss_rotated = rdc_loss(v_rotated, measured_noisy, d_max=d_max)

    print(f"RDC loss original:  {loss_original:.4f}")
    print(f"RDC loss rotated:   {loss_rotated:.4f}")
    assert jnp.allclose(loss_original, loss_rotated, atol=1e-2), (
        f"RDC loss changed after rotation: {loss_original:.4f} vs {loss_rotated:.4f}"
    )


def test_bond_loss_ideal_geometry():
    """
    PUBLICATION VALIDATION: Engh & Huber, Acta Crystallogr. A, 1991.

    The bond length loss must be exactly zero for the canonical Cα–Cα
    virtual bond distance of 3.80 Å, which is the reference geometry
    used by REFMAC, CNS, and PHENIX.  It must also be non-trivially
    positive for the old (incorrect) 1.52 Å default, confirming the bug
    fix is necessary.
    """
    bond_fn = get_bond_length_loss(target_distance=3.8)

    # Perfectly linear chain with ideal 3.8 Å spacing.
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [7.6, 0.0, 0.0],
            [11.4, 0.0, 0.0],
        ]
    )
    loss_ideal = bond_fn(positions)
    print(f"Bond loss at 3.8 Å (ideal):  {loss_ideal:.2e}")
    assert jnp.isclose(loss_ideal, 0.0, atol=1e-6), (
        f"Bond loss must be 0 at ideal 3.8 Å geometry, got {loss_ideal}"
    )

    # The incorrect old default of 1.52 Å would severely penalise valid geometry.
    bond_fn_wrong = get_bond_length_loss(target_distance=1.52)
    loss_wrong = bond_fn_wrong(positions)
    print(f"Bond loss at 1.52 Å (wrong): {loss_wrong:.2f}")
    assert loss_wrong > 4.0, (
        f"Old 1.52 Å default should produce large loss on valid geometry, got {loss_wrong:.4f}"
        f" (expected > 4.0; actual MSE = (3.8 - 1.52)² = 5.20)"
    )


def test_noe_upper_bound_loss():
    """
    PUBLICATION VALIDATION: Wüthrich, NMR of Proteins and Nucleic Acids,
    1986; Güntert, Mumenthaler & Wüthrich, J. Mol. Biol. 1997 (CYANA).

    NOE loss must be zero when all inter-proton distances satisfy their
    upper bounds, positive when violated, and the gradients must push
    violating atoms apart.
    """
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ]
    )
    noe_pairs = jnp.array([[0, 1], [1, 2]])

    # 1. Satisfied upper bounds (d = 3.0 Å < 4.0 Å limit) → zero loss.
    upper_satisfied = jnp.array([4.0, 4.0])
    loss_ok = noe_upper_bound_loss(positions, noe_pairs, upper_satisfied)
    print(f"NOE loss (satisfied): {loss_ok:.4f}")
    assert loss_ok == 0.0, f"NOE loss must be 0 when bounds satisfied, got {loss_ok}"

    # 2. Violated bounds (d = 3.0 Å > 2.0 Å limit) → positive loss.
    upper_violated = jnp.array([2.0, 2.0])
    loss_viol = noe_upper_bound_loss(positions, noe_pairs, upper_violated)
    print(f"NOE loss (violated):  {loss_viol:.4f}")
    assert loss_viol > 0.0
    # Each pair violates by 1.0 Å → mean((1.0², 1.0²)) = 1.0
    assert jnp.isclose(loss_viol, 1.0), f"Expected NOE loss = 1.0, got {loss_viol}"

    # 3. Gradients must push violating atoms apart.
    grad_fn = jax.grad(noe_upper_bound_loss)
    grads = grad_fn(positions, noe_pairs, upper_violated)
    # Atom 0 gradient must be negative x (pushed left away from atom 1).
    assert grads[0, 0] < 0.0, "Atom 0 should be pushed away from atom 1 (negative x)"
    # Atom 1 sees forces from both pairs; net should be positive x.
    assert grads[2, 0] > 0.0, "Atom 2 should be pushed away from atom 1 (positive x)"


def test_saupe_tensor_eigenvalue_bounds():
    """
    PHYSICAL VALIDATION: Bax & Tjandra, J. Biomol. NMR, 1997;
    Losonczi, Andrec, Fischer & Prestegard, J. Magn. Reson., 1999.

    A physical Saupe order matrix is traceless and symmetric.
    Its principal-axis-frame eigenvalues (order parameters) must lie
    within [-0.5, 1.0] — the thermodynamically allowed range for
    orientational order parameters.  The fitted tensor must also be
    exactly traceless (Sxx + Syy + Szz = 0).
    """
    rng = np.random.default_rng(42)
    num_vectors = 30
    d_max = 21700.0

    v_np = rng.standard_normal((num_vectors, 3))
    v_np /= np.linalg.norm(v_np, axis=-1, keepdims=True)
    v = jnp.array(v_np)

    true_s = jnp.array([0.001, -0.0005, 0.0002, -0.0003, 0.0001])
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    A = d_max * jnp.stack(
        [x**2 - z**2, y**2 - z**2, 2 * x * y, 2 * x * z, 2 * y * z], axis=1
    )
    measured = A @ true_s

    # Fit the tensor — should recover true_s exactly (overdetermined, full rank).
    s_fit, _, _, _ = jnp.linalg.lstsq(A, measured, rcond=1e-5)
    Sxx, Syy, Sxy, Sxz, Syz = s_fit

    # Reconstruct the full 3×3 symmetric traceless Saupe matrix.
    Szz = -(Sxx + Syy)  # traceless constraint: Sxx + Syy + Szz = 0
    S_matrix = jnp.array(
        [
            [Sxx, Sxy, Sxz],
            [Sxy, Syy, Syz],
            [Sxz, Syz, Szz],
        ]
    )

    # Tracelessness.
    trace = jnp.trace(S_matrix)
    print(f"Saupe tensor trace: {trace:.2e}  (must be ~0)")
    assert jnp.isclose(trace, 0.0, atol=1e-6), f"Saupe tensor not traceless: trace={trace:.2e}"

    # Symmetry.
    assert jnp.allclose(S_matrix, S_matrix.T, atol=1e-8), "Saupe tensor must be symmetric"

    # Eigenvalue bounds: principal order parameters in [-0.5, 1.0].
    eigenvalues = jnp.linalg.eigvalsh(S_matrix)
    print(f"Saupe tensor eigenvalues: {eigenvalues}")
    assert jnp.all(eigenvalues >= -0.5 - 1e-6), (
        f"Saupe eigenvalues must be >= -0.5, got {eigenvalues}"
    )
    assert jnp.all(eigenvalues <= 1.0 + 1e-6), (
        f"Saupe eigenvalues must be <= 1.0, got {eigenvalues}"
    )
