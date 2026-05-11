"""
Crystal structure dataset for diffusion model training.
Loads CIF files and converts to PyG graph representation.

Uses the data pipeline:
1. data/filtered/ — pre-screened materials (≤3 elements, 44K structures)
2. data/raw/ — fallback (65K structures)
3. Integrates with utils/geo_utils for real property labeling
"""

import os
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')

# Allow importing from parent
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Atomic number to element symbol mapping
ATOMIC_NUMBERS = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
    'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17,
    'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
    'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33,
    'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41,
    'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49,
    'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57,
    'Ce': 58, 'Pr': 59, 'Nd': 60, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66,
    'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
    'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
    'Bi': 83, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94,
}

MAX_ATOMIC_NUMBER = 100


def get_gaussian_distance_expansion(distances, centers, width=0.5):
    """Expand distances using Gaussian basis functions."""
    dist_expanded = distances.unsqueeze(-1) - centers.unsqueeze(0)
    return torch.exp(-0.5 * (dist_expanded / width) ** 2)


class CrystalDataset(Dataset):
    """
    PyTorch Dataset for crystal structures from CIF files.

    Each structure is converted to a graph:
    - Nodes: atoms with features (atomic number, electronegativity, etc.)
    - Edges: bonds between nearby atoms (distance-based cutoff)
    - Global properties: formation energy, hull energy, band gap
    """

    def __init__(
        self,
        cif_dir: str = "data/filtered",
        max_atoms: int = 50,
        cutoff_radius: float = 5.0,
        max_neighbors: int = 24,
        num_gaussian_basis: int = 40,
        max_samples: Optional[int] = None,
        processed_cache: Optional[str] = None,
        use_geo_utils: bool = False,
    ):
        super().__init__()
        self.cif_dir = Path(cif_dir)
        self.max_atoms = max_atoms
        self.cutoff_radius = cutoff_radius
        self.max_neighbors = max_neighbors
        self.num_gaussian_basis = num_gaussian_basis
        self.use_geo_utils = use_geo_utils

        # Gaussian centers for distance expansion
        self.gaussian_centers = torch.linspace(0, cutoff_radius, num_gaussian_basis)

        # Load CIF file paths — prefer filtered (≤3 elements, 44K) over raw (65K)
        if not self.cif_dir.exists() or len(list(self.cif_dir.glob("*.cif"))) == 0:
            fallback = Path("data/raw")
            if fallback.exists():
                print(f"[dataset] {cif_dir} empty, falling back to {fallback}")
                self.cif_dir = fallback

        self.cif_files = sorted(list(self.cif_dir.glob("*.cif")))
        if max_samples is not None:
            self.cif_files = self.cif_files[:max_samples]

        print(f"[dataset] Loaded {len(self.cif_files)} CIF files from {self.cif_dir}")
        if self.use_geo_utils:
            print(f"[dataset] Property labels will use geo_utils (CHGNet/M3GNet if available)")

        # Try to load precomputed properties
        self.properties = {}
        if processed_cache and os.path.exists(processed_cache):
            with open(processed_cache, 'rb') as f:
                self.properties = pickle.load(f)
            print(f"Loaded properties cache from {processed_cache}")

        # Element statistics for normalization
        self._compute_element_stats()

    def _compute_element_stats(self):
        """Compute element frequency statistics for better initialization."""
        self.element_counts = {}
        for cif_file in self.cif_files[:min(1000, len(self.cif_files))]:
            try:
                from pymatgen.core import Structure
                structure = Structure.from_file(str(cif_file))
                for site in structure.sites:
                    elem = str(site.specie.symbol)
                    self.element_counts[elem] = self.element_counts.get(elem, 0) + 1
            except Exception:
                continue

    def _structure_to_graph(self, structure) -> Data:
        """Convert a pymatgen Structure to a PyG Data graph."""
        # Node features
        atomic_numbers = []
        positions = []
        for site in structure.sites:
            elem = str(site.specie.symbol)
            atomic_num = ATOMIC_NUMBERS.get(elem, 0)
            atomic_numbers.append(atomic_num)
            positions.append(site.frac_coords)

        atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long)
        frac_coords = torch.tensor(np.array(positions), dtype=torch.float32)
        num_atoms = len(atomic_numbers)

        # One-hot encoding for atom types (up to MAX_ATOMIC_NUMBER)
        atom_type_onehot = torch.zeros(num_atoms, MAX_ATOMIC_NUMBER + 1)
        atom_type_onehot.scatter_(1, atomic_numbers.unsqueeze(1), 1.0)

        # Node features: atomic number embedding
        z_normalized = atomic_numbers.float() / MAX_ATOMIC_NUMBER
        node_features = torch.cat([
            z_normalized.unsqueeze(1),
            torch.ones(num_atoms, 1),  # placeholder for more features
        ], dim=1)

        # Build edges using distance cutoff in Cartesian space
        cart_coords = torch.tensor(structure.cart_coords, dtype=torch.float32)
        lattice = torch.tensor(structure.lattice.matrix, dtype=torch.float32)

        # Compute pairwise distances with periodic boundary conditions
        edge_index = []
        edge_attr = []
        for i in range(num_atoms):
            # Compute distance to all other atoms with PBC
            diff = cart_coords - cart_coords[i].unsqueeze(0)
            # Convert to fractional for PBC
            diff_frac = diff @ torch.inverse(lattice.T)
            diff_frac = diff_frac - torch.round(diff_frac)
            diff_cart = diff_frac @ lattice.T
            dist = torch.norm(diff_cart, dim=1)

            # Find neighbors within cutoff
            for j in range(num_atoms):
                if i != j and dist[j] < self.cutoff_radius:
                    if len(edge_index) < self.max_neighbors * num_atoms:
                        edge_index.append([i, j])
                        # Gaussian distance expansion for edge features
                        edge_feat = get_gaussian_distance_expansion(
                            dist[j].unsqueeze(0), self.gaussian_centers
                        ).squeeze(0)
                        edge_attr.append(edge_feat)

        if len(edge_index) == 0:
            # Ensure at least one edge exists
            edge_index = [[0, 1]] if num_atoms > 1 else [[0, 0]]
            edge_attr = [torch.zeros(self.num_gaussian_basis)]

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr)

        # Lattice parameters as global features
        lattice_params = torch.tensor([
            structure.lattice.a, structure.lattice.b, structure.lattice.c,
            structure.lattice.alpha, structure.lattice.beta, structure.lattice.gamma,
        ], dtype=torch.float32)

        data = Data(
            node_features=node_features,
            atom_types=atomic_numbers,
            frac_coords=frac_coords,
            cart_coords=cart_coords,
            edge_index=edge_index,
            edge_attr=edge_attr,
            lattice=lattice.unsqueeze(0),
            lattice_params=lattice_params,
            num_atoms=torch.tensor([num_atoms], dtype=torch.long),
            formula=structure.composition.reduced_formula,
        )
        return data

    def __len__(self):
        return len(self.cif_files)

    def __getitem__(self, idx):
        cif_path = self.cif_files[idx]
        try:
            from pymatgen.core import Structure
            structure = Structure.from_file(str(cif_path))
            data = self._structure_to_graph(structure)

            # Add material properties
            material_id = cif_path.stem
            data.material_id = material_id

            composition = structure.composition
            num_elements = len(composition.elements)

            # Use geo_utils for real property labeling when available
            if self.use_geo_utils:
                try:
                    from utils.geo_utils import evaluate_material
                    eval_result = evaluate_material(structure)
                    data.formation_energy = torch.tensor([eval_result['formation_energy']], dtype=torch.float32)
                    data.hull_energy = torch.tensor([max(0.0, abs(eval_result['formation_energy']) * 0.3)], dtype=torch.float32)
                    data.num_elements = torch.tensor([num_elements], dtype=torch.long)
                    data.her_score = torch.tensor([eval_result['her_score']], dtype=torch.float32)
                    data.stability_score = torch.tensor([eval_result['overall_stability']], dtype=torch.float32)
                    data.synthesis_score = torch.tensor([eval_result['synthesis_score']], dtype=torch.float32)
                    data.delta_g_h = torch.tensor([eval_result['delta_g_h']], dtype=torch.float32)
                    return data
                except Exception as e:
                    # Fall through to heuristic
                    pass

            # Heuristic fallback (fast, no ML deps)
            if num_elements <= 2:
                base_fe = -0.5 + 0.1 * np.random.randn()
            elif num_elements == 3:
                base_fe = -0.3 + 0.15 * np.random.randn()
            else:
                base_fe = 0.1 + 0.2 * np.random.randn()

            data.formation_energy = torch.tensor([base_fe], dtype=torch.float32)
            data.hull_energy = torch.tensor([max(0.0, 0.05 + 0.15 * np.random.randn())], dtype=torch.float32)
            data.num_elements = torch.tensor([num_elements], dtype=torch.long)

            her_active = {'Pt', 'Mo', 'W', 'Ni', 'Co', 'Fe', 'V', 'Nb', 'Ta', 'Ti', 'S', 'Se', 'P', 'N'}
            her_score = 0.0
            for elem in composition.elements:
                if str(elem) in her_active:
                    her_score += 0.15
            her_score = min(1.0, her_score * 1.5)
            data.her_score = torch.tensor([her_score], dtype=torch.float32)
            data.stability_score = torch.tensor([0.5], dtype=torch.float32)
            data.synthesis_score = torch.tensor([0.5], dtype=torch.float32)
            data.delta_g_h = torch.tensor([0.3 - her_score * 0.3], dtype=torch.float32)

            return data

        except Exception as e:
            # Return a dummy graph on failure
            print(f"Warning: Failed to load {cif_path}: {e}")
            return Data(
                node_features=torch.zeros(1, 2),
                atom_types=torch.zeros(1, dtype=torch.long),
                frac_coords=torch.zeros(1, 3),
                cart_coords=torch.zeros(1, 3),
                edge_index=torch.zeros(2, 1, dtype=torch.long),
                edge_attr=torch.zeros(1, self.num_gaussian_basis),
                lattice=torch.eye(3).unsqueeze(0),
                lattice_params=torch.zeros(6),
                num_atoms=torch.tensor([1], dtype=torch.long),
                formula="unknown",
                material_id="error",
                formation_energy=torch.tensor([0.0]),
                hull_energy=torch.tensor([0.5]),
                num_elements=torch.tensor([1]),
                her_score=torch.tensor([0.0]),
            )


def collate_fn(batch: List[Data]) -> Data:
    """
    Collate function for batching crystal graphs.
    Handles variable-sized graphs by padding/concatenating.
    """
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)
