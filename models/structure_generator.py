"""
Structure Generator: converts diffusion model outputs to pymatgen Structure objects.

Handles:
- Converting fractional coordinates + atom types + lattice to crystal structures
- Post-processing (symmetry analysis, structure refinement)
- Batch generation with property conditioning
"""

import torch
import numpy as np
from typing import List, Optional, Dict
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Atomic number to element symbol reverse mapping
ATOMIC_NUMBER_TO_SYMBOL = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl',
    18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn',
    26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As',
    34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb',
    42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In',
    50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La',
    58: 'Ce', 59: 'Pr', 60: 'Nd', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy',
    67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W',
    75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb',
    83: 'Bi', 90: 'Th', 91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu',
}

# HER-active elements
HER_ACTIVE_ELEMENTS = {'Pt', 'Mo', 'W', 'Ni', 'Co', 'Fe', 'V', 'Nb', 'Ta', 'Ti'}
CHALCOGEN_ELEMENTS = {'S', 'Se', 'Te'}
PNICTOGEN_ELEMENTS = {'N', 'P', 'As'}


class StructureGenerator:
    """
    Generates pymatgen Structure objects from diffusion model samples.
    """

    # Allowlist: common elements that appear in typical inorganic materials
    # Excludes radioactive, noble gases, and extremely rare elements
    ALLOWED_ATOMIC_NUMBERS = sorted(set([
        # Alkali/Alkaline earth
        3, 4, 11, 12, 19, 20, 37, 38, 55, 56,
        # Transition metals (HER-relevant)
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        39, 40, 41, 42, 44, 45, 46, 47, 48,
        72, 73, 74, 75, 76, 77, 78, 79,
        # Main group
        5, 6, 7, 8, 9, 13, 14, 15, 16, 17,
        31, 32, 33, 34, 35, 49, 50, 51, 52, 53,
        81, 82, 83,
        # Lanthanides (common)
        57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
    ]))

    # HER-preferring compositions: (active_metal, chalcogen/pnictogen) pairs
    HER_FAVORED_ELEMENT_GROUPS = [
        # MoS2 family
        [42, 16],  # Mo, S
        [74, 16],  # W, S
        [42, 34],  # Mo, Se
        [74, 34],  # W, Se
        # Noble metal chalcogenides
        [78, 16], [78, 34],  # Pt-S, Pt-Se
        [46, 16], [46, 34],  # Pd-S, Pd-Se
        # Transition metal phosphides
        [27, 15], [28, 15],  # Co-P, Ni-P
        [42, 15], [74, 15],  # Mo-P, W-P
        # Layered double hydroxides / oxides
        [25, 8], [26, 8], [27, 8], [28, 8],  # Mn/Fe/Co/Ni oxides
        # Nitrides
        [42, 7], [74, 7], [22, 7],  # Mo-N, W-N, Ti-N
    ]

    def __init__(
        self,
        model,
        device: str = 'cuda',
        max_atoms: int = 50,
        prefer_2d: bool = True,
        her_bias: bool = True,
        cif_dir: str = 'data/filtered',
    ):
        self.model = model
        self.device = device
        self.max_atoms = max_atoms
        self.prefer_2d = prefer_2d
        self.her_bias = her_bias
        self.model.eval()

        # Learn element distribution from training data
        self.element_weights = self._compute_element_weights(cif_dir)

    def _compute_element_weights(self, cif_dir: str) -> torch.Tensor:
        """Compute sampling weights for elements based on training data frequency."""
        from pathlib import Path
        weights = torch.ones(max(self.ALLOWED_ATOMIC_NUMBERS) + 1)
        cif_path = Path(cif_dir)
        count = 0
        for cif_file in list(cif_path.glob('*.cif'))[:500]:
            try:
                from pymatgen.core import Structure
                s = Structure.from_file(str(cif_file))
                for site in s.sites:
                    z = site.specie.Z
                    if z < len(weights):
                        weights[z] += 1
                count += 1
            except Exception:
                pass
        # Normalize and ensure minimum weight for allowed elements
        weights = weights / weights.sum()
        for z in self.ALLOWED_ATOMIC_NUMBERS:
            if weights[z] < 0.001:
                weights[z] = 0.001
        return weights

    def _generate_lattice_2d(self, num_atoms: int, batch_size: int = 1) -> torch.Tensor:
        """
        Generate lattice parameters favoring 2D structures.
        Returns [B, 3, 3] lattice matrices.
        """
        lattices = []
        for _ in range(batch_size):
            # Generate a 2D-favoring lattice (larger a,b; smaller c)
            a = 3.0 + 2.0 * np.random.random()  # 3.0-5.0 Angstrom
            b = 3.0 + 2.0 * np.random.random()
            c = 8.0 + 8.0 * np.random.random()  # 8.0-16.0 Angstrom (large interlayer)

            alpha = 85.0 + 10.0 * np.random.random()  # 85-95 degrees
            beta = 85.0 + 10.0 * np.random.random()
            gamma = 55.0 + 70.0 * np.random.random()  # 55-125 degrees (variety)

            alpha_rad = np.deg2rad(alpha)
            beta_rad = np.deg2rad(beta)
            gamma_rad = np.deg2rad(gamma)

            # Build lattice matrix
            lattice = np.zeros((3, 3))
            lattice[0, 0] = a
            lattice[1, 0] = b * np.cos(gamma_rad)
            lattice[1, 1] = b * np.sin(gamma_rad)
            lattice[2, 0] = c * np.cos(beta_rad)
            lattice[2, 1] = c * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad)
            lattice[2, 2] = c * np.sqrt(1 - np.cos(beta_rad)**2 -
                                         ((np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad))**2)
            lattices.append(lattice)

        return torch.tensor(np.array(lattices), dtype=torch.float32, device=self.device)

    def _determine_num_atoms_her(self, formula_hint: Optional[str] = None) -> int:
        """Determine number of atoms based on HER-active elements preference."""
        if formula_hint:
            # Parse formula to get atom count
            return min(self.max_atoms, max(4, len(formula_hint) * 2))
        # Default: binary/ternary compounds with reasonable cell size
        return np.random.randint(6, min(25, self.max_atoms))

    def generate(
        self,
        num_samples: int = 10,
        num_atoms: Optional[int] = None,
        target_her_score: Optional[float] = None,
        target_stability: Optional[float] = None,
        target_synthesis: Optional[float] = None,
        guidance_scale: float = 1.0,
    ) -> List:
        """
        Generate new crystal structures with composition-aware element selection.
        """
        from pymatgen.core import Structure, Lattice

        from tqdm import tqdm

        structures = []
        allowed_cpu = torch.tensor(self.ALLOWED_ATOMIC_NUMBERS)

        for i in tqdm(range(num_samples), desc="Generating"):
            n_atoms = num_atoms if num_atoms is not None else self._determine_num_atoms_her()

            # --- Step 1: Pick 2-3 distinct elements (composition-level) ---
            if self.her_bias and np.random.random() < 0.5:
                # HER-favored: pick a known active group (e.g. Mo-S)
                group = self.HER_FAVORED_ELEMENT_GROUPS[
                    np.random.randint(len(self.HER_FAVORED_ELEMENT_GROUPS))
                ]
                distinct_elements = group[:2]  # binary compound
                if np.random.random() < 0.3 and len(group) > 2:
                    distinct_elements = group[:3]  # ternary
            else:
                # Sample 2-3 elements from data distribution
                n_elem = np.random.choice([2, 2, 2, 3, 3])  # bias towards binary
                weights_cpu = self.element_weights
                idx = torch.multinomial(weights_cpu[allowed_cpu], n_elem, replacement=False)
                distinct_elements = allowed_cpu[idx].tolist()

            # --- Step 2: Assign stoichiometry ---
            if len(distinct_elements) == 2:
                ratios = [[1, 1], [1, 2], [2, 1], [1, 3], [3, 1], [2, 3], [3, 2]]
            else:
                ratios = [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2], [2, 1, 2]]

            ratio = ratios[np.random.randint(len(ratios))]
            # Scale ratio to reach target atom count
            ratio_sum = sum(ratio)
            repeats = max(1, n_atoms // ratio_sum)
            atom_z_list = []
            for elem, r in zip(distinct_elements, ratio):
                atom_z_list.extend([elem] * (r * repeats))
            # Trim or pad to exact count
            atom_z_list = atom_z_list[:n_atoms]
            while len(atom_z_list) < n_atoms:
                atom_z_list.append(distinct_elements[0])

            # Shuffle
            np.random.shuffle(atom_z_list)

            # Generate 2D-favoring lattice
            lattice = self._generate_lattice_2d(n_atoms, batch_size=1)

            # Build property conditioning
            if target_her_score is not None or target_stability is not None or target_synthesis is not None:
                properties = torch.tensor([[
                    target_her_score if target_her_score is not None else 0.5,
                    target_stability if target_stability is not None else 0.5,
                    target_synthesis if target_synthesis is not None else 0.5,
                ]], device=self.device)
            else:
                properties = torch.tensor([[0.8, 0.7, 0.7]], device=self.device)

            # Use data-driven element selection as initial atom types
            atom_types_tensor = torch.tensor(atom_z_list, device=self.device, dtype=torch.long)

            # Sample from diffusion model (coordinates evolve, atom types stay)
            try:
                result = self.model.sample(
                    num_atoms=n_atoms,
                    batch_size=1,
                    lattice=lattice,
                    properties=properties,
                    guidance_scale=guidance_scale,
                    initial_atom_types=atom_types_tensor,
                )
            except Exception as e:
                print(f"Warning: sampling failed for sample {i}: {e}")
                continue

            try:
                structure = self._tensor_to_structure(
                    result['frac_coords'],
                    atom_types_tensor,
                    result['lattice'],
                )
                if structure is not None:
                    structures.append(structure)
            except Exception as e:
                print(f"Warning: conversion failed for sample {i}: {e}")

        return structures

    def _tensor_to_structure(
        self,
        frac_coords: torch.Tensor,
        atom_types: torch.Tensor,
        lattice: torch.Tensor,
    ) -> Optional:
        """Convert tensors to pymatgen Structure."""
        from pymatgen.core import Structure, Lattice

        # Move to CPU
        frac_coords = frac_coords.cpu().numpy()
        atom_types = atom_types.cpu().numpy()
        lattice_matrix = lattice[0].cpu().numpy()

        # Convert atom type indices to element symbols
        symbols = []
        valid_indices = []
        for i, z in enumerate(atom_types):
            symbol = ATOMIC_NUMBER_TO_SYMBOL.get(int(z))
            if symbol is not None:
                symbols.append(symbol)
                valid_indices.append(i)

        if len(symbols) == 0:
            return None

        # Filter coordinates for valid atoms
        frac_coords = frac_coords[valid_indices]

        try:
            lattice_obj = Lattice(lattice_matrix)
            structure = Structure(
                lattice=lattice_obj,
                species=symbols,
                coords=frac_coords,
                coords_are_cartesian=False,
            )
            return structure
        except Exception:
            return None


def save_structures_to_cif(structures: List, output_dir: str = "results"):
    """Save generated structures as CIF files."""
    from pymatgen.io.cif import CifWriter

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cif_files = []
    for i, structure in enumerate(structures):
        try:
            formula = structure.composition.reduced_formula
            safe_formula = formula.replace(" ", "_")
            filename = output_path / f"generated_{i+1:03d}_{safe_formula}.cif"
            writer = CifWriter(structure)
            writer.write_file(str(filename))
            cif_files.append(str(filename))
        except Exception as e:
            print(f"Warning: failed to save structure {i}: {e}")

    return cif_files
