"""
Property Optimization Module.

Implements:
1. HER activity predictor (ΔG_H prediction)
2. Stability predictor (formation energy + hull energy)
3. Synthesizability predictor
4. Multi-objective optimization with loss weighting
5. Property-guided generation guidance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from typing import Dict, Optional, Tuple
import math


# ============================================================================
# HER Activity Predictor
# ============================================================================

class HERActivityPredictor(nn.Module):
    """
    Predicts HER catalytic activity (ΔG_H) from crystal graph.

    Uses the Sabatier principle via Volcano plot:
    - Optimal ΔG_H ≈ 0 eV (Pt-like activity)
    - Activity ∝ exp(-(ΔG_H - ΔG_H_opt)² / 2σ²)

    The network predicts the H adsorption energy, which is then
    mapped to HER activity via the volcano relationship.
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim

        input_dim = hidden_dim + 3  # node features from denoiser + frac_coords

        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Message passing layers (simplified GNN)
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2 + 40, hidden_dim),  # h_i + h_j + edge_attr
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])

        self.node_update = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])

        # Global pooling + prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 2),  # [ΔG_H, her_score]
        )

        self.sigma = 0.1  # Volcano plot width

    def forward(self, node_features, frac_coords, edge_index, edge_attr, batch):
        """Predict HER activity from crystal graph."""
        h = torch.cat([node_features, frac_coords], dim=-1)
        h = self.node_encoder(h)

        for conv, update in zip(self.conv_layers, self.node_update):
            row, col = edge_index
            edge_input = torch.cat([h[row], h[col], edge_attr], dim=-1)
            messages = conv(edge_input)

            # Aggregate messages
            aggregated = torch.zeros_like(h)
            aggregated.scatter_add_(0, row.unsqueeze(-1).expand(-1, h.size(-1)), messages)

            # Update nodes
            h = update(torch.cat([h, aggregated], dim=-1)) + h

        # Global pooling
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_global = torch.cat([h_mean, h_max], dim=-1)

        # Predict
        output = self.pred_head(h_global)
        delta_g_h = output[:, 0:1]  # H adsorption free energy
        her_score_raw = output[:, 1:2]

        # Apply volcano relationship for HER activity
        her_score = torch.exp(-(delta_g_h ** 2) / (2 * self.sigma ** 2))

        # Clamp to valid range
        delta_g_h = torch.clamp(delta_g_h, -2.0, 2.0)
        her_score = torch.clamp(her_score, 0.0, 1.0)

        return {
            'delta_g_h': delta_g_h,
            'her_score': her_score,
            'her_score_raw': her_score_raw,
        }


# ============================================================================
# Stability Predictor
# ============================================================================

class StabilityPredictor(nn.Module):
    """
    Predicts material stability from crystal graph.

    Predicts:
    1. Formation energy (eV/atom) - thermodynamic stability
    2. Energy above hull (eV/atom) - thermodynamic stability
    3. Stability score [0, 1] - combined metric
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        input_dim = hidden_dim + 3  # node features from denoiser + frac_coords

        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2 + 40, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])

        self.node_update = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])

        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),  # +1 for num_elements
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3),  # [formation_energy, hull_energy, stability_score]
        )

    def forward(self, node_features, frac_coords, edge_index, edge_attr, batch, num_atoms_per_graph=None):
        """Predict stability properties."""
        h = torch.cat([node_features, frac_coords], dim=-1)
        h = self.node_encoder(h)

        for conv, update in zip(self.conv_layers, self.node_update):
            row, col = edge_index
            edge_input = torch.cat([h[row], h[col], edge_attr], dim=-1)
            messages = conv(edge_input)

            aggregated = torch.zeros_like(h)
            aggregated.scatter_add_(0, row.unsqueeze(-1).expand(-1, h.size(-1)), messages)

            h = update(torch.cat([h, aggregated], dim=-1)) + h

        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)

        # Calculate num elements per graph
        if num_atoms_per_graph is None:
            num_elements = torch.ones(len(h_mean), 1, device=h_mean.device) * 2.0
        else:
            num_elements = num_atoms_per_graph.float().unsqueeze(-1)

        h_global = torch.cat([h_mean, h_max, num_elements], dim=-1)

        output = self.pred_head(h_global)
        formation_energy = output[:, 0:1]
        hull_energy = torch.clamp(output[:, 1:2], min=0.0)  # hull energy >= 0
        stability_score = torch.sigmoid(output[:, 2:3])  # [0, 1]

        return {
            'formation_energy': formation_energy,
            'hull_energy': hull_energy,
            'stability_score': stability_score,
        }


# ============================================================================
# Synthesizability Predictor
# ============================================================================

class SynthesisPredictor(nn.Module):
    """
    Predicts experimental synthesizability from crystal graph.

    Factors considered:
    - Number of elements (fewer = easier to synthesize)
    - Element commonality (abundant elements easier)
    - Structural complexity (simpler = more synthesizable)
    - Known synthesis routes for similar compositions
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        input_dim = hidden_dim + 3  # node features from denoiser + frac_coords

        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2 + 40, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])

        self.node_update = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])

        # Prediction head (includes element count as feature)
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 2),  # [synthesis_score, complexity]
        )

    def forward(self, node_features, frac_coords, edge_index, edge_attr, batch, num_elements_per_graph=None):
        """Predict synthesizability."""
        h = torch.cat([node_features, frac_coords], dim=-1)
        h = self.node_encoder(h)

        for conv, update in zip(self.conv_layers, self.node_update):
            row, col = edge_index
            edge_input = torch.cat([h[row], h[col], edge_attr], dim=-1)
            messages = conv(edge_input)

            aggregated = torch.zeros_like(h)
            aggregated.scatter_add_(0, row.unsqueeze(-1).expand(-1, h.size(-1)), messages)

            h = update(torch.cat([h, aggregated], dim=-1)) + h

        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)

        if num_elements_per_graph is None:
            num_elements = torch.ones(len(h_mean), 1, device=h_mean.device) * 2.0
        else:
            num_elements = num_elements_per_graph.float().unsqueeze(-1)

        h_global = torch.cat([h_mean, h_max, num_elements], dim=-1)

        output = self.pred_head(h_global)
        synthesis_score = torch.sigmoid(output[:, 0:1])
        complexity = output[:, 1:2]

        return {
            'synthesis_score': synthesis_score,
            'complexity': complexity,
        }


# ============================================================================
# Multi-Objective Loss Function
# ============================================================================

class MultiObjectiveLoss(nn.Module):
    """
    Multi-task loss combining HER, stability, and synthesis objectives.

    L_total = w_diff * L_diffusion + w_her * L_HER + w_stab * L_stability + w_syn * L_synthesis
    """

    def __init__(
        self,
        w_diffusion: float = 1.0,
        w_her: float = 0.5,
        w_stability: float = 0.3,
        w_synthesis: float = 0.2,
        target_delta_g_h: float = 0.0,  # Ideal ΔG_H = 0 eV
        target_stability: float = 1.0,
        target_synthesis: float = 1.0,
    ):
        super().__init__()
        self.w_diffusion = w_diffusion
        self.w_her = w_her
        self.w_stability = w_stability
        self.w_synthesis = w_synthesis

        self.target_delta_g_h = target_delta_g_h
        self.target_stability = target_stability
        self.target_synthesis = target_synthesis

        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(
        self,
        diffusion_loss: torch.Tensor,
        her_output: Dict[str, torch.Tensor],
        stability_output: Dict[str, torch.Tensor],
        synthesis_output: Dict[str, torch.Tensor],
        her_targets: Optional[Dict[str, torch.Tensor]] = None,
        stability_targets: Optional[Dict[str, torch.Tensor]] = None,
        synthesis_targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total multi-objective loss.

        Args:
            diffusion_loss: Loss from diffusion model
            her_output: HER predictor output dict
            stability_output: Stability predictor output dict
            synthesis_output: Synthesis predictor output dict
            her_targets: Optional target values for HER
            stability_targets: Optional target values for stability
            synthesis_targets: Optional target values for synthesis
        """
        losses = {}

        # Diffusion loss
        losses['diffusion_loss'] = diffusion_loss

        # HER loss: minimize |ΔG_H - 0|
        her_target_delta_g = torch.zeros_like(her_output['delta_g_h']) + self.target_delta_g_h
        her_target_score = torch.zeros_like(her_output['her_score']) + 1.0

        if her_targets is not None:
            if 'delta_g_h' in her_targets:
                her_target_delta_g = her_targets['delta_g_h']
            if 'her_score' in her_targets:
                her_target_score = her_targets['her_score']

        losses['her_delta_g_loss'] = self.mse(her_output['delta_g_h'], her_target_delta_g)
        losses['her_score_loss'] = self.mae(her_output['her_score'], her_target_score)
        losses['her_loss'] = losses['her_delta_g_loss'] + 0.5 * losses['her_score_loss']

        # Stability loss
        stab_target_fe = torch.zeros_like(stability_output['formation_energy']) - 0.5  # negative formation energy
        stab_target_hull = torch.zeros_like(stability_output['hull_energy'])  # hull energy near 0
        stab_target_score = torch.zeros_like(stability_output['stability_score']) + self.target_stability

        if stability_targets is not None:
            if 'formation_energy' in stability_targets:
                stab_target_fe = stability_targets['formation_energy']
            if 'hull_energy' in stability_targets:
                stab_target_hull = stability_targets['hull_energy']
            if 'stability_score' in stability_targets:
                stab_target_score = stability_targets['stability_score']

        losses['stab_formation_loss'] = self.mse(stability_output['formation_energy'], stab_target_fe)
        losses['stab_hull_loss'] = self.mse(stability_output['hull_energy'], stab_target_hull)
        losses['stab_score_loss'] = self.mae(stability_output['stability_score'], stab_target_score)
        losses['stability_loss'] = (
            losses['stab_formation_loss'] + losses['stab_hull_loss'] + 0.5 * losses['stab_score_loss']
        )

        # Synthesis loss
        syn_target_score = torch.zeros_like(synthesis_output['synthesis_score']) + self.target_synthesis

        if synthesis_targets is not None and 'synthesis_score' in synthesis_targets:
            syn_target_score = synthesis_targets['synthesis_score']

        losses['synthesis_loss'] = self.mae(synthesis_output['synthesis_score'], syn_target_score)

        # Total loss
        losses['total_loss'] = (
            self.w_diffusion * diffusion_loss +
            self.w_her * losses['her_loss'] +
            self.w_stability * losses['stability_loss'] +
            self.w_synthesis * losses['synthesis_loss']
        )

        return losses


# ============================================================================
# Property Optimization Wrapper
# ============================================================================

class MaterialPropertyOptimizer(nn.Module):
    """
    Wrapper that combines all property predictors for joint optimization.

    Used during both training (as auxiliary tasks) and generation (as guidance).
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.her_predictor = HERActivityPredictor(hidden_dim=hidden_dim)
        self.stability_predictor = StabilityPredictor(hidden_dim=hidden_dim)
        self.synthesis_predictor = SynthesisPredictor(hidden_dim=hidden_dim)

    def forward(self, node_features, frac_coords, edge_index, edge_attr, batch, num_atoms_per_graph=None):
        """Predict all material properties jointly."""
        her_out = self.her_predictor(node_features, frac_coords, edge_index, edge_attr, batch)
        stab_out = self.stability_predictor(node_features, frac_coords, edge_index, edge_attr, batch, num_atoms_per_graph)

        num_elements = None
        if num_atoms_per_graph is not None:
            num_elements = num_atoms_per_graph

        syn_out = self.synthesis_predictor(node_features, frac_coords, edge_index, edge_attr, batch, num_elements)

        return {
            'her': her_out,
            'stability': stab_out,
            'synthesis': syn_out,
        }
