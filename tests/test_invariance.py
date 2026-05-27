import jax.numpy as jnp

from resonance_flow.losses import get_steric_clash_loss


def test_rotation_invariance() -> None:
    """Verify that the loss is invariant to global rotation of the structure."""
    loss_fn = get_steric_clash_loss()
    positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    atom_radii = jnp.array([1.0, 1.0, 1.0])

    initial_loss = loss_fn(positions, atom_radii)

    # Apply a 90-degree rotation around Z axis
    rotation_matrix = jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated_positions = positions @ rotation_matrix

    rotated_loss = loss_fn(rotated_positions, atom_radii)

    assert jnp.allclose(
        initial_loss, rotated_loss
    ), f"Loss changed after rotation: {initial_loss} vs {rotated_loss}"
    print("Rotation invariance test passed!")


def test_translation_invariance() -> None:
    """Verify that the loss is invariant to global translation."""
    loss_fn = get_steric_clash_loss()
    positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    atom_radii = jnp.array([1.0, 1.0])

    initial_loss = loss_fn(positions, atom_radii)

    # Translate by [10, 20, 30]
    translated_positions = positions + jnp.array([10.0, 20.0, 30.0])

    translated_loss = loss_fn(translated_positions, atom_radii)

    assert jnp.allclose(
        initial_loss, translated_loss
    ), f"Loss changed after translation: {initial_loss} vs {translated_loss}"
    print("Translation invariance test passed!")


if __name__ == "__main__":
    test_rotation_invariance()
    test_translation_invariance()
