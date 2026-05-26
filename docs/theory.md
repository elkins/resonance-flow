# Theory: Differentiable Physics in ResonanceFlow

ResonanceFlow treats protein structure prediction as a differentiable
optimization problem. The total training loss is a weighted sum of
physical and experimental constraint terms:

$$\mathcal{L}_{total} = \mathcal{L}_{steric} + \lambda_{bond}\,\mathcal{L}_{bond} + \mathcal{L}_{RDC} + \mathcal{L}_{NOE}$$

Because every term is implemented in JAX, gradients $\nabla_\theta \mathcal{L}_{total}$
flow back through the Transformer weights $\theta$, allowing the model to
learn physically valid protein geometries end-to-end.

---

## Steric Clash Penalty

A harmonic repulsion penalises any pair of atoms whose van der Waals
radii overlap. Given atoms $i$ and $j$ with radii $r_i$, $r_j$ at
positions $\mathbf{x}_i$, $\mathbf{x}_j$, the pairwise overlap is:

$$o_{ij} = \max\!\left(0,\; (r_i + r_j) - \|\mathbf{x}_i - \mathbf{x}_j\|\right)$$

The total loss sums over all non-excluded pairs:

$$\mathcal{L}_{steric} = \frac{1}{2} \sum_{i \neq j,\; |i-j|>k} o_{ij}^2$$

where $k$ is the `exclude_bonded_range` parameter (0 = self only;
1 = AMBER 1-2 exclusion; 2 = 1-2 and 1-3 exclusion). Periodic boundary
conditions are supported via `jax-md`.

---

## Bond Length Constraints

Backbone geometry is constrained by a mean-squared error loss on
consecutive C$\alpha$–C$\alpha$ virtual bond distances. The canonical
C$\alpha$–C$\alpha$ distance in a peptide chain is **3.80 ± 0.02 Å**
(Engh & Huber, *Acta Crystallogr. A*, 1991):

$$\mathcal{L}_{bond} = \frac{1}{N-1} \sum_{i=1}^{N-1} \left(\|\mathbf{x}_{i+1} - \mathbf{x}_i\| - d_{ideal}\right)^2, \quad d_{ideal} = 3.8\,\text{Å}$$

> **Note**: The ideal distance is the *virtual* bond between consecutive
> alpha-carbons spanning the full peptide unit, **not** the C–C covalent
> bond length (1.52 Å).

---

## Residual Dipolar Couplings (RDCs)

RDCs report the average orientation of an internuclear bond vector
relative to the external magnetic field, providing long-range global
restraints that complement short-range NOE distances
(Bax & Tjandra, *J. Biomol. NMR*, 1997).

### Saupe Tensor Formulation

The RDC for an internuclear vector $\hat{\mathbf{v}} = (x, y, z)$ is:

$$D = D_{max} \left[ S_{xx}(x^2 - z^2) + S_{yy}(y^2 - z^2) + 2S_{xy}\,xy + 2S_{xz}\,xz + 2S_{yz}\,yz \right]$$

where $\mathbf{S}$ is the **Saupe order matrix** — a traceless,
symmetric $3 \times 3$ tensor with five independent components
(Saupe, *Z. Naturforsch.*, 1964). Writing $\mathbf{s} = [S_{xx}, S_{yy}, S_{xy}, S_{xz}, S_{yz}]^\top$,
the equations for all $N$ bond vectors are the linear system $\mathbf{A}\mathbf{s} = \mathbf{D}$.

### Differentiable Fitting

ResonanceFlow fits $\mathbf{S}$ at every forward pass via the
differentiable least-squares solver `jnp.linalg.lstsq` (SVD-based,
with a ridge penalty $r_{cond} = 10^{-5}$ for stability). The loss is
the MSE of the residual:

$$\mathcal{L}_{RDC} = \frac{1}{N}\sum_{i=1}^N \left(D_i^{calc} - D_i^{obs}\right)^2$$

Gradient flow through the solver allows the model to learn backbone
orientations that are consistent with the measured RDCs.

### Quality Factor

The **Q-factor** (Cornilescu, Marquardt, Ottiger & Bax, *JACS*, 1998)
quantifies structural quality analogously to the crystallographic R-factor:

$$Q = \frac{\sqrt{\langle (D^{calc} - D^{obs})^2 \rangle}}{\sqrt{\langle (D^{obs})^2 \rangle}}$$

A high-quality structure has $Q \leq 0.20$. To detect overfitting,
$Q_{free}$ should be computed on a held-out subset of RDCs not used
during fitting (Clore & Garrett, *JACS*, 1999).

### N-H Proxy Vectors (Cα-only Models)

Because the current model predicts only C$\alpha$ coordinates, true
N–H internuclear vectors are not directly available. ResonanceFlow
uses the **anti-parallel virtual-bond approximation**:

$$\hat{\mathbf{v}}_i^{proxy} = \frac{\mathbf{x}_{i-1} - \mathbf{x}_{i+1}}{\|\mathbf{x}_{i-1} - \mathbf{x}_{i+1}\|}$$

This proxy correlates with the amide N-H orientation in both
$\alpha$-helices and $\beta$-strands (Zweckstetter & Bax,
*J. Am. Chem. Soc.*, 2000). A full-atom model should supply true N–H
vectors directly to `rdc_loss`.

---

## NOE Distance Restraints

Nuclear Overhauser Effect (NOE) cross-peaks provide upper bounds on
inter-proton distances, typically in the range 1.8–6.0 Å (Wüthrich,
*NMR of Proteins and Nucleic Acids*, 1986). ResonanceFlow applies a
flat-bottomed harmonic penalty only to **upper-bound violations**
(no lower-bound penalty, since an NOE cross-peak is only observed when
protons are close):

$$\mathcal{L}_{NOE} = \frac{1}{M} \sum_{k=1}^{M} \max\!\left(0,\; d_k - d_k^{upper}\right)^2$$

This form matches the standard restraint potential used in CYANA /
DYANA (Güntert, Mumenthaler & Wüthrich, *J. Mol. Biol.*, 1997).

---

## References

| Citation | Role in ResonanceFlow |
|---|---|
| Saupe, *Z. Naturforsch.* **19a**, 161 (1964) | Definition of the alignment (Saupe) tensor |
| Engh & Huber, *Acta Crystallogr. A* **47**, 392 (1991) | Ideal Cα–Cα bond distance (3.8 Å) |
| Wüthrich, *NMR of Proteins and Nucleic Acids* (1986) | NOE distance restraints |
| Bax & Tjandra, *J. Biomol. NMR* **10**, 289 (1997) | RDC structure determination |
| Cornilescu, Marquardt, Ottiger & Bax, *JACS* **120**, 6836 (1998) | RDC Q-factor definition |
| Clore & Garrett, *JACS* **121**, 9008 (1999) | Q_free cross-validation |
| Güntert, Mumenthaler & Wüthrich, *J. Mol. Biol.* **273**, 283 (1997) | NOE restraint potential (CYANA) |
| Zweckstetter & Bax, *JACS* **122**, 3791 (2000) | Cα-based N-H proxy vectors |
| Ottiger & Bax, *JACS* **120**, 12334 (1998) | d_max for ¹⁵N-¹H at 600 MHz |
