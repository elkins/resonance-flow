import jax.numpy as jnp
from jax_md import space


def get_steric_clash_loss(box_size=None, exclude_bonded_range=0):
    """
    Returns a function to compute the steric clash (atom overlap) penalty.

    Args:
        box_size: Optional. If provided, uses periodic boundary conditions.
                  Otherwise, assumes free space.
        exclude_bonded_range: Exclude atom pairs whose sequential index
                              separation is <= this value.  Default 0 excludes
                              only self-interactions (original behaviour).
                              Set to 1 to also exclude directly bonded 1-2
                              neighbours, or 2 for 1-2 and 1-3 pairs
                              (standard AMBER / CHARMM convention).
    """
    if box_size is None:
        displacement_fn, _ = space.free()
    else:
        displacement_fn, _ = space.periodic(box_size)

    space_metric = space.metric(displacement_fn)

    def steric_clash_loss(positions, atom_radii):
        """
        Computes the penalty for overlapping atoms.

        Args:
            positions: (N, 3) array of atomic coordinates.
            atom_radii: (N,) array of atomic van der Waals radii.

        Returns:
            A scalar loss representing the total steric clash penalty.
        """
        n = positions.shape[0]
        dr = space.map_product(space_metric)(positions, positions)
        radii_sum = atom_radii[:, None] + atom_radii[None, :]
        overlap = jnp.maximum(radii_sum - dr, 0.0)

        # Build mask: exclude pairs with index separation <= exclude_bonded_range.
        # exclude_bonded_range=0  →  only self excluded  (original behaviour)
        # exclude_bonded_range=1  →  self + 1-2 bonded excluded
        # exclude_bonded_range=2  →  self + 1-2 + 1-3 excluded
        indices = jnp.arange(n)
        pair_sep = jnp.abs(indices[:, None] - indices[None, :])
        mask = (pair_sep > exclude_bonded_range).astype(jnp.float32)
        overlap = overlap * mask

        loss = jnp.sum(overlap**2) / 2.0
        return loss

    return steric_clash_loss


def get_bond_length_loss(target_distance=3.8):
    """
    Penalises deviations from the ideal Cα–Cα virtual bond length.

    The canonical Cα–Cα distance in a peptide chain is 3.80 ± 0.02 Å
    (Engh & Huber, Acta Crystallogr. A, 1991).  This is the virtual bond
    between sequential alpha-carbons across the full peptide unit; it is
    NOT the C–C covalent bond length (1.52 Å).

    Args:
        target_distance: Ideal Cα–Cα virtual bond length in Angstroms.
                         Default 3.8 Å (Engh & Huber 1991).
    """

    def bond_length_loss(positions):
        # Compute distances between consecutive Cα atoms.
        diffs = positions[1:] - positions[:-1]
        distances = jnp.linalg.norm(diffs, axis=-1)
        return jnp.mean((distances - target_distance) ** 2)

    return bond_length_loss


def estimate_nh_proxy_vectors(ca_coords):
    """
    Estimates backbone N-H proxy vectors from Cα coordinates.

    Uses the anti-parallel virtual-bond approximation: for each interior
    residue i the proxy N-H direction is taken as the unit vector from
    Cα(i+1) to Cα(i-1), which is roughly anti-parallel to the local
    backbone tangent and correlates with the amide N-H orientation in
    both α-helices and β-strands.  This is a standard Cα-only coarse-
    graining strategy for alignment tensor calculations (see Zweckstetter &
    Bax, J. Am. Chem. Soc. 2000, for the geometric relationship between
    Cα positions and alignment-frame vectors).

    Note: for full-atom models, real N–H internuclear vectors should be
    supplied directly to rdc_loss instead of using this approximation.

    Args:
        ca_coords: (N, 3) array of Cα coordinates.

    Returns:
        (N-2, 3) unit proxy vectors for residues 1 … N-2.
    """
    # Anti-parallel virtual bond: Cα(i-1) − Cα(i+1), normalised.
    raw = ca_coords[:-2] - ca_coords[2:]  # shape (N-2, 3)
    norms = jnp.linalg.norm(raw, axis=-1, keepdims=True)
    return raw / (norms + 1e-8)


def rdc_loss(predicted_vectors, measured_rdcs, d_max=21700.0):
    """
    Scientifically correct RDC loss using Saupe tensor fitting.
    Fits the alignment tensor to the structure, then calculates the residual.

    The formula follows the Saupe order matrix parametrisation:

        D = d_max * (Sxx*(x²-z²) + Syy*(y²-z²) + 2*Sxy*xy + 2*Sxz*xz + 2*Syz*yz)

    using the five independent components of the traceless symmetric tensor.

    References:
        Bax & Tjandra, J. Biomol. NMR, 1997.
        Cornilescu, Marquardt, Ottiger & Bax, J. Am. Chem. Soc., 1998.

    Args:
        predicted_vectors: (N, 3) internuclear vectors from the model.
        measured_rdcs: (N,) experimental RDC values in Hz.
        d_max: Maximum dipolar coupling constant.
               Default 21 700 Hz for ¹⁵N-¹H backbone amide bonds at 600 MHz
               (Ottiger & Bax, J. Am. Chem. Soc., 1998).

    Returns:
        Scalar MSE loss between measured and back-calculated RDCs after
        optimal Saupe tensor fitting.
    """
    # 1. Normalise vectors.
    norms = jnp.linalg.norm(predicted_vectors, axis=-1, keepdims=True)
    v = predicted_vectors / (norms + 1e-8)

    # 2. Build the design matrix A so that A @ s = D_calc.
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    A = d_max * jnp.stack([x**2 - z**2, y**2 - z**2, 2 * x * y, 2 * x * z, 2 * y * z], axis=1)

    # 3. Fit Saupe tensor (least squares; ridge penalty for numerical stability).
    s, _, _, _ = jnp.linalg.lstsq(A, measured_rdcs, rcond=1e-5)

    # 4. Back-calculate RDCs and return MSE.
    predicted_rdcs = A @ s
    return jnp.mean((predicted_rdcs - measured_rdcs) ** 2)


def rdc_q_factor(predicted_vectors, measured_rdcs, d_max=21700.0):
    """
    Computes the RDC Q-factor (Cornilescu, Marquardt, Ottiger & Bax, JACS 1998).

    The Q-factor is the NMR analogue of the crystallographic R-factor:

        Q = RMSD(D_calc − D_obs) / RMS(D_obs)

    A high-quality backbone structure typically has Q ≤ 0.20 for ¹⁵N-¹H
    RDCs.  To guard against overfitting, a Q_free computed on data *not*
    used during fitting is preferred (Clore & Garrett, JACS 1999).

    Args:
        predicted_vectors: (N, 3) internuclear vectors from the model.
        measured_rdcs: (N,) experimental RDC values in Hz.
        d_max: Maximum dipolar coupling constant (Hz).

    Returns:
        Q-factor (dimensionless, 0–1; lower is better).
    """
    norms = jnp.linalg.norm(predicted_vectors, axis=-1, keepdims=True)
    v = predicted_vectors / (norms + 1e-8)

    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    A = d_max * jnp.stack([x**2 - z**2, y**2 - z**2, 2 * x * y, 2 * x * z, 2 * y * z], axis=1)
    s, _, _, _ = jnp.linalg.lstsq(A, measured_rdcs, rcond=1e-5)
    predicted_rdcs = A @ s

    rmsd = jnp.sqrt(jnp.mean((predicted_rdcs - measured_rdcs) ** 2))
    rms_obs = jnp.sqrt(jnp.mean(measured_rdcs**2))
    return rmsd / (rms_obs + 1e-10)


def noe_upper_bound_loss(positions, noe_pairs, upper_bounds):
    """
    Penalises violations of NOE-derived inter-proton distance upper bounds.

    NOE distance restraints are the primary source of 3D structural
    information in protein NMR, providing upper bounds on inter-proton
    distances typically in the range 1.8–6.0 Å (Wüthrich, *NMR of Proteins
    and Nucleic Acids*, 1986; Güntert et al., J. Mol. Biol., 1997).

    A flat-bottomed harmonic penalty is applied only to upper-bound
    violations (no lower-bound penalty, since NOE cross-peaks are only
    observed when protons are close):

        L_NOE = mean( max(0, d_ij − d_upper)² )

    Args:
        positions: (N, 3) atomic coordinates in Angstroms.
        noe_pairs: (M, 2) integer array of atom-index pairs.
        upper_bounds: (M,) upper distance bounds in Angstroms.

    Returns:
        Scalar NOE violation loss.
    """
    ri = positions[noe_pairs[:, 0]]
    rj = positions[noe_pairs[:, 1]]
    dists = jnp.linalg.norm(ri - rj, axis=-1)
    violations = jnp.maximum(dists - upper_bounds, 0.0)
    return jnp.mean(violations**2)
