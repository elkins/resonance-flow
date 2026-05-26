# ResonanceFlow 🧬📡

**Self-correcting protein structure prediction with differentiable NMR constraints.**

ResonanceFlow is a JAX-native framework that integrates experimental NMR
observables directly into the training loop of a protein structure model.
Physical violations (atomic clashes, bad bond geometry) and NMR
mismatches (RDC residuals, NOE distance violations) are expressed as
differentiable loss functions whose gradients flow back into the
Transformer weights — allowing the model to *self-correct* in real time.

---

## Installation

```bash
pip install resonance-flow
```

For development:

```bash
git clone https://github.com/elkins/resonance-flow.git
cd resonance-flow
pip install -e ".[dev]"
```

---

## Quick Start

### Run the prototype training loop

```python
from resonance_flow.train import main

state = main(num_steps=100)
```

This runs a small self-contained demo: a 10-residue sequence is passed
through the Transformer, and the predicted Cα coordinates are
simultaneously optimised against three loss terms (steric clash, bond
geometry, RDC).

### Use individual loss functions

```python
import jax.numpy as jnp
from resonance_flow import (
    get_steric_clash_loss,
    get_bond_length_loss,
    rdc_loss,
    rdc_q_factor,
    noe_upper_bound_loss,
    estimate_nh_proxy_vectors,
)

# Steric clash (free space, 1-2 bonded pairs excluded)
clash_fn = get_steric_clash_loss(exclude_bonded_range=1)
positions  = jnp.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
atom_radii = jnp.array([1.5, 1.5])
print(clash_fn(positions, atom_radii))  # 0.0 — no overlap

# Bond length (Cα–Cα virtual bond, target 3.8 Å)
bond_fn = get_bond_length_loss()
ca_chain = jnp.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0]])
print(bond_fn(ca_chain))  # ~0.0

# RDC loss (Saupe tensor fitting)
nh_vectors   = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                           [0.7, 0.7, 0.0], [0.7, 0.0, 0.7], [0.0, 0.7, 0.7]])
measured_rdc = jnp.array([10.0, -5.0, 2.0, 0.0, 4.0, 8.0])
print(rdc_loss(nh_vectors, measured_rdc))

# RDC Q-factor (structure quality metric, Q ≤ 0.20 = good)
print(rdc_q_factor(nh_vectors, measured_rdc))

# N-H proxy vectors from Cα coordinates (Cα-only models)
ca_coords = jnp.ones((8, 3))  # replace with real coordinates
nh_proxy  = estimate_nh_proxy_vectors(ca_coords)  # shape (6, 3)

# NOE upper-bound distance restraints
noe_pairs    = jnp.array([[0, 2], [1, 3]])
upper_bounds = jnp.array([5.0, 5.0])
print(noe_upper_bound_loss(positions, noe_pairs[:1], upper_bounds[:1]))
```

---

## Key Concepts

| Concept | Description |
|---|---|
| **Differentiable constraints** | Every loss term is a JAX-traceable function — gradients flow directly into model weights |
| **Saupe tensor fitting** | The RDC alignment tensor is fit at every forward pass via `jnp.linalg.lstsq` (SVD) |
| **Cα-only model** | The Transformer predicts one coordinate per residue; N-H proxy vectors approximate amide orientations |
| **PBC support** | Steric clash loss supports periodic boundary conditions via `jax-md` |
| **Bonded exclusion** | Standard AMBER/CHARMM 1-2 (and 1-3) pair exclusions available via `exclude_bonded_range` |

---

## Documentation

- [Theory](theory.md) — Mathematical derivations and literature references
- [API Reference — Losses](api/losses.md) — All loss functions and helpers
- [API Reference — Model](api/model.md) — `TransformerCoordinatePredictor`
- [API Reference — Training](api/train.md) — Training loop utilities
