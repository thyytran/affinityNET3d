"""PDBBind complex -> point cloud extraction.

Converts a protein (PDB) + ligand (SDF/MOL2) into a point cloud suitable for
PointNet / Point Transformer:

    coords:   (N, 3)  float32  atom xyz positions (angstroms)
    features: (N, F)  float32  per-atom features

The pocket is defined as protein atoms within ``cutoff`` angstroms of any
ligand atom; all ligand atoms are always kept. Hydrogens are dropped by
default (PDBBind structures are inconsistent about protonation).

The feature dimension ``F`` produced here is exactly ``FEATURE_DIM`` below.
Pass that value as ``in_features`` when constructing a model:

    from pointcloud_affinity.data.extract import FEATURE_DIM
    model = build_model("point_transformer", in_features=FEATURE_DIM)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Feature scheme
# --------------------------------------------------------------------------- #
# Element one-hot vocabulary. Atoms outside this set map to the trailing
# "other" slot, so the dimension is fixed regardless of input chemistry.
ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "other"]
_ELEMENT_INDEX = {sym: i for i, sym in enumerate(ELEMENTS)}

# Per-atom feature layout:
#   [ element one-hot (len ELEMENTS) ]
#   [ is_ligand flag                 ]  1.0 ligand, 0.0 protein
#   [ partial charge                 ]  Gasteiger (ligand) / 0.0 (protein)
#   [ degree (heavy-atom neighbors)  ]
#   [ is_aromatic                    ]
#   [ is_in_ring                     ]
FEATURE_DIM = len(ELEMENTS) + 5


def _element_onehot(symbol: str) -> np.ndarray:
    vec = np.zeros(len(ELEMENTS), dtype=np.float32)
    vec[_ELEMENT_INDEX.get(symbol, _ELEMENT_INDEX["other"])] = 1.0
    return vec


# --------------------------------------------------------------------------- #
# Ligand parsing (RDKit)
# --------------------------------------------------------------------------- #
def _load_ligand(path: str):
    """Load a ligand from SDF or MOL2 into an RDKit Mol with a conformer."""
    from rdkit import Chem

    suffix = Path(path).suffix.lower()
    if suffix == ".sdf":
        supplier = Chem.SDMolSupplier(path, removeHs=True, sanitize=True)
        mol = next((m for m in supplier if m is not None), None)
    elif suffix in (".mol2", ".mol"):
        mol = Chem.MolFromMol2File(path, removeHs=True, sanitize=True)
    else:
        raise ValueError(f"Unsupported ligand format: {suffix}")

    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError(f"Could not parse a 3D ligand from {path}")
    return mol


def _ligand_atoms(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (coords (M,3), features (M,F)) for ligand heavy atoms."""
    from rdkit.Chem import AllChem

    mol = _load_ligand(path)
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:  # noqa: BLE001 - charges are optional, default to 0
        logger.warning("Gasteiger charge computation failed; using 0.0")

    conf = mol.GetConformer()
    coords, feats = [], []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])

        try:
            charge = float(atom.GetDoubleProp("_GasteigerCharge"))
            if not np.isfinite(charge):
                charge = 0.0
        except (KeyError, ValueError):
            charge = 0.0

        feats.append(
            np.concatenate([
                _element_onehot(atom.GetSymbol()),
                np.array([
                    1.0,                                   # is_ligand
                    charge,
                    float(atom.GetDegree()),
                    float(atom.GetIsAromatic()),
                    float(atom.IsInRing()),
                ], dtype=np.float32),
            ])
        )
    return np.asarray(coords, dtype=np.float32), np.asarray(feats, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Protein parsing (BioPython)
# --------------------------------------------------------------------------- #
def _protein_atoms(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (coords (P,3), features (P,F)) for protein heavy atoms."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", path)

    coords, feats = [], []
    for atom in structure.get_atoms():
        element = atom.element.strip().capitalize() if atom.element else ""
        if element == "H" or element == "":
            continue  # drop hydrogens / unlabeled
        coords.append(atom.get_coord())
        feats.append(
            np.concatenate([
                _element_onehot(element),
                np.array([
                    0.0,   # is_ligand
                    0.0,   # charge (not computed for protein here)
                    0.0,   # degree (no bond graph from PDB coords alone)
                    0.0,   # is_aromatic
                    0.0,   # is_in_ring
                ], dtype=np.float32),
            ])
        )
    if not coords:
        raise ValueError(f"No heavy atoms parsed from protein {path}")
    return np.asarray(coords, dtype=np.float32), np.asarray(feats, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Pocket selection + assembly
# --------------------------------------------------------------------------- #
def _pocket_mask(
    protein_coords: np.ndarray, ligand_coords: np.ndarray, cutoff: float
) -> np.ndarray:
    """Boolean mask of protein atoms within ``cutoff`` of any ligand atom."""
    # Pairwise distances (P, M); fine for pocket-sized inputs.
    diff = protein_coords[:, None, :] - ligand_coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    return dist.min(axis=1) <= cutoff


def complex_to_point_cloud(
    protein_path: str,
    ligand_path: str,
    cutoff: float = 8.0,
    center: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a point cloud from a protein-ligand complex.

    Args:
        protein_path: path to a .pdb file.
        ligand_path:  path to a .sdf / .mol2 file.
        cutoff:       pocket radius in angstroms around the ligand.
        center:       if True, translate coords so the ligand centroid is at
                      the origin (removes absolute-position bias).

    Returns:
        coords:   (N, 3) float32
        features: (N, FEATURE_DIM) float32
        where N = (pocket protein atoms) + (ligand atoms).
    """
    lig_coords, lig_feats = _ligand_atoms(ligand_path)
    prot_coords, prot_feats = _protein_atoms(protein_path)

    mask = _pocket_mask(prot_coords, lig_coords, cutoff)
    prot_coords, prot_feats = prot_coords[mask], prot_feats[mask]
    logger.info(
        "Pocket: %d protein atoms within %.1f A, %d ligand atoms",
        len(prot_coords), cutoff, len(lig_coords),
    )

    coords = np.concatenate([lig_coords, prot_coords], axis=0)
    feats = np.concatenate([lig_feats, prot_feats], axis=0)

    if center and len(lig_coords) > 0:
        coords = coords - lig_coords.mean(axis=0, keepdims=True)

    return coords.astype(np.float32), feats.astype(np.float32)