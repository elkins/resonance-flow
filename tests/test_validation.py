import jax.numpy as jnp
import numpy as np

from resonance_flow.losses import get_steric_clash_loss, rdc_loss


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
    # Wait, the code says jnp.sum(overlap**2) / 2.0.
    # overlap matrix has 2.0 at [0,1] and [1,0].
    # Sum(4, 4) / 2 = 4.0.
    assert jnp.isclose(loss_clash, 4.0)
