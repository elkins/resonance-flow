import jax
import jax.numpy as jnp

from resonance_flow.model import TransformerCoordinatePredictor
from resonance_flow.train import create_train_state, train_step


def test_model_output_shape():
    rng = jax.random.PRNGKey(0)
    batch_size = 2
    seq_len = 8
    model = TransformerCoordinatePredictor(vocab_size=21, d_model=32, num_heads=2, num_layers=1)
    x = jax.random.randint(rng, (batch_size, seq_len), 0, 21)
    params = model.init(rng, x, deterministic=True)
    coords = model.apply(params, x, deterministic=True)
    assert isinstance(coords, jnp.ndarray)
    assert coords.shape == (batch_size, seq_len, 3)


def test_train_step_updates_params():
    rng = jax.random.PRNGKey(42)
    rng, init_rng, step_rng = jax.random.split(rng, 3)
    seq_len = 5
    model = TransformerCoordinatePredictor(vocab_size=21, d_model=16, num_heads=2, num_layers=1)
    state = create_train_state(init_rng, model, learning_rate=0.1)
    initial_params = state.params
    batch = jnp.array([[1, 2, 3, 4, 5]])
    atom_radii = jnp.ones((seq_len,)) * 1.5
    # estimate_nh_proxy_vectors returns N-2 proxy vectors for N Cα atoms.
    measured_rdcs = jnp.ones((seq_len - 2,))

    new_state, loss, _, _, _ = train_step(state, batch, step_rng, atom_radii, measured_rdcs)

    def params_changed(p1, p2):
        return any(
            not jnp.array_equal(v1, v2)
            for v1, v2 in zip(jax.tree_util.tree_leaves(p1), jax.tree_util.tree_leaves(p2))
        )

    assert params_changed(initial_params, new_state.params)
