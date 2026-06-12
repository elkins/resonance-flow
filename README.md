# 🧬 Resonance-Flow: Differentiable Protein Structure Prediction with NMR Self-Correction

[![Tests](https://github.com/elkins/resonance-flow/actions/workflows/test.yml/badge.svg)](https://github.com/elkins/resonance-flow/actions/workflows/test.yml)
[![Docs](https://github.com/elkins/resonance-flow/actions/workflows/docs.yml/badge.svg)](https://github.com/elkins/resonance-flow/actions/workflows/docs.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-mkdocs--material-blue.svg)](https://elkins.github.io/resonance-flow/)
[![PyPI version](https://img.shields.io/pypi/v/resonance-flow)](https://pypi.org/project/resonance-flow/)
[![Python versions](https://img.shields.io/pypi/pyversions/resonance-flow)](https://pypi.org/project/resonance-flow/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)
[![JAX](https://img.shields.io/badge/framework-JAX%20%2B%20Flax-9cf.svg)](https://jax.readthedocs.io/)

**Resonance-Flow** is a JAX-native protein structure prediction framework that integrates differentiable biophysics with experimental NMR constraints. It allows models to "self-correct" by propagating gradients from physical violations (atomic clashes, bad geometry) and NMR observables (RDCs, NOE distances) back into the neural network architecture — end-to-end, with no manual refinement step.

---

## 🚀 Key Features

- **JAX-Native Gradient Flow** — End-to-end differentiability from experimental constraints to model weights via `jax.grad`.
- **Saupe Tensor RDC Loss** — Differentiable least-squares fitting of the alignment tensor at every forward pass (Bax & Tjandra 1997; Cornilescu et al. 1998).
- **NOE Distance Restraints** — Flat-bottomed harmonic penalty on upper-bound violations, the primary 3D information source in protein NMR (Wüthrich 1986; Güntert et al. 1997).
- **Biophysically Correct Geometry** — Bond length loss calibrated to the canonical Cα–Cα distance of 3.80 Å (Engh & Huber 1991).
- **Differentiable Steric Clash** — Harmonic atom-overlap penalty with optional AMBER/CHARMM-style 1-2/1-3 bonded exclusions, powered by `jax-md`.
- **RDC Quality Metric** — Built-in Q-factor and Q_free cross-validation (Cornilescu et al. 1998; Clore & Garrett 1999) for structural validation without additional tooling.
- **Backbone Conformational Checks** — Pseudo-torsion angle calculation (Oldfield & Hubbard 1994) to verify secondary structure plausibility in Cα-only models.
- **PBC Support** — Periodic boundary conditions for simulation-box contexts.
- **Transformer-to-Coords** — A pre-LN Transformer architecture that maps amino acid sequences directly to physical 3D Cα coordinates.

---

## 🧠 The Concept: "Self-Correction"

Traditional folding models are trained on static PDB snapshots.  Resonance-Flow instead teaches a model to *listen* to physical laws and NMR data during training itself:

```
  Sequence  →  [Transformer]  →  Cα Coordinates
                                       │
                  ┌────────────────────┼──────────────────────┐
                  ▼                    ▼                       ▼
           Steric Clash          Bond Length              RDC / NOE
             Penalty               Loss                  Mismatch
                  └────────────────────┼──────────────────────┘
                                       │  ∇θ L_total
                                       ▼
                              [Optimizer Step]
```

Gradients from every constraint flow back simultaneously into the model weights — the model learns not just from data, but from physics.

---

## 🛠️ Installation

```bash
pip install resonance-flow
```

For development (includes linting, type-checking, testing, and docs):

```bash
git clone https://github.com/elkins/resonance-flow.git
cd resonance-flow
pip install -e ".[dev]"
```

**Requirements:** Python 3.10+, JAX ≥ 0.4, Flax, Optax, jax-md, NumPy.

---

## 🧪 Quick Start

### Run the self-correction demo

```python
from resonance_flow.train import main

state = main(num_steps=100)
# Step   0 | Total Loss: 12.3421 | Steric: 0.0012 | Bond: 1.2034 | RDC: 0.0087
# Step  10 | Total Loss:  4.1823 | ...
# Step 100 | Total Loss:  0.0031 | ...
```

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

# ── Steric clash (AMBER-style 1-2 bonded exclusion) ──────────────────────────
clash_fn   = get_steric_clash_loss(exclude_bonded_range=1)
positions  = jnp.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
atom_radii = jnp.array([1.5, 1.5])
clash_fn(positions, atom_radii)          # → 0.0  (no overlap)

# ── Bond length (Cα–Cα virtual bond, Engh & Huber 1991) ─────────────────────
bond_fn  = get_bond_length_loss()        # default target = 3.8 Å
ca_chain = jnp.array([[0.0,0.0,0.0],[3.8,0.0,0.0],[7.6,0.0,0.0]])
bond_fn(ca_chain)                        # → ~0.0

# ── RDC loss (Saupe tensor fitting) ─────────────────────────────────────────
nh_vecs      = jnp.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],
                           [0.7,0.7,0.],[0.7,0.,0.7],[0.,0.7,0.7]])
measured_rdc = jnp.array([10., -5., 2., 0., 4., 8.])
rdc_loss(nh_vecs, measured_rdc)          # → scalar MSE

# ── RDC Q-factor (structure quality; Q ≤ 0.20 = high quality) ───────────────
rdc_q_factor(nh_vecs, measured_rdc)      # → 0 – 1 (lower is better)
train_mask = jnp.array([True, True, True, False, False, False])
rdc_q_free(nh_vecs, measured_rdc, train_mask)  # → Q-factor on held-out data

# ── N-H proxy vectors from Cα coordinates (Cα-only models) ──────────────────
ca_coords = jax.random.normal(jax.random.PRNGKey(0), (10, 3))
nh_proxy  = estimate_nh_proxy_vectors(ca_coords)   # → (8, 3) unit vectors

# ── NOE upper-bound distance restraints (Wüthrich 1986) ─────────────────────
noe_pairs    = jnp.array([[0, 2], [1, 3]])
upper_bounds = jnp.array([5.0, 4.5])
noe_upper_bound_loss(positions, noe_pairs[:1], upper_bounds[:1])  # → 0.0
```

---

## 🎓 Interactive Tutorial Catalog

Experience **Resonance-Flow** directly in your browser via Google Colab. These interactive Jupyter Notebook tutorials cover everything from basic biophysics to advanced structural self-correction.

| Tutorial | Difficulty | Time | Action |
| :--- | :--- | :--- | :--- |
| **Self-Correction Demo** | ⭐ Beginner | 15 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins/resonance-flow/blob/main/examples/interactive_tutorials/self_correction_demo.ipynb) |
| **Biophysical Constraints** | ⭐ Beginner | 15 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins/resonance-flow/blob/main/examples/interactive_tutorials/biophysical_constraints.ipynb) |
| **Differentiable NMR** | ⭕ Intermediate | 25 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins/resonance-flow/blob/main/examples/interactive_tutorials/differentiable_nmr.ipynb) |
| **Transformer-to-Coords** | 🏔️ Advanced | 30 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins/resonance-flow/blob/main/examples/interactive_tutorials/transformer_to_coords.ipynb) |

---

## 🔬 Scientific Basis

All loss functions and validation metrics are grounded in published, peer-reviewed NMR methodology:

| Loss / Metric | Scientific Basis |
|---|---|
| RDC loss — Saupe tensor | Bax & Tjandra, *J. Biomol. NMR* 1997; Cornilescu et al., *JACS* 1998 |
| RDC Q-factor | Cornilescu et al., *JACS* 1998; Clore & Garrett, *JACS* 1999 |
| NOE distance restraints | Wüthrich, *NMR of Proteins and Nucleic Acids* 1986; Güntert et al., *J. Mol. Biol.* 1997 |
| Cα–Cα bond distance (3.8 Å) | Engh & Huber, *Acta Crystallogr. A* 1991 |
| N-H proxy vectors | Zweckstetter & Bax, *JACS* 2000 |
| Bonded exclusion (1-2/1-3) | Cornell et al. (AMBER), *JACS* 1995; MacKerell et al. (CHARMM), *J. Phys. Chem. B* 1998 |
| d_max = 21 700 Hz | Ottiger & Bax, *JACS* 1998 |

---

## 🧬 Architecture

```
TransformerCoordinatePredictor
├── Embedding         (vocab_size=21, d_model=128)
├── Positional Embed  (learned, max_len=512)
├── N × Pre-LN Block
│   ├── LayerNorm → MultiHeadDotProductAttention → Residual
│   └── LayerNorm → FFN (4× expand, GELU) → Residual
└── LayerNorm → Linear(3)   # → (batch, seq_len, 3) Cα coordinates
```

The pre-LN (LayerNorm before attention) layout avoids gradient
explosion and follows the convention recommended by Xiong et al. 2020.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or pull request. The project follows:

- **Formatting + Linting:** `ruff` / `ruff format`
- **Type checking:** `mypy`
- **Testing:** `pytest` with coverage

```bash
# Run the full quality pipeline before submitting a PR
ruff check resonance_flow tests
ruff format resonance_flow tests
mypy resonance_flow tests
pytest --cov=resonance_flow tests
```

---

## 📚 Documentation

Full theory, API reference, and examples at **[elkins.github.io/resonance-flow](https://elkins.github.io/resonance-flow/)**.

---

## ⚖️ License

MIT © George Elkins

---

## 🔗 Related Projects

Resonance-Flow is the most complete end-to-end model in this ecosystem, depending on:

- [diff-biophys](https://github.com/elkins/diff-biophys) — Differentiable RDC, NOE, bond-length, and clash kernels
- [synth-nmr](https://github.com/elkins/synth-nmr) — NMR parameter libraries (chemical shifts, Karplus, RDC)
- [synth-pdb](https://github.com/elkins/synth-pdb) — Protein structure data generation
- [torsion-tuner](https://github.com/elkins/torsion-tuner) — Single-structure refinement using similar torsion-space kinematics
- [diff-ensemble](https://github.com/elkins/diff-ensemble) — Ensemble counterpart for IDPs

---

## 📖 Citation

```bibtex
@software{resonance_flow,
  author  = {Elkins, George},
  title   = {Resonance-Flow: Differentiable protein structure prediction with NMR self-correction},
  year    = {2026},
  url     = {https://github.com/elkins/resonance-flow},
  version = {0.1.0}
}
```
