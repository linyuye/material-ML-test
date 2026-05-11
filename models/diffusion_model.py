"""
EGNN-based Diffusion Model for Crystal Structure Generation.

Implements an E(n) Equivariant Graph Neural Network (EGNN) as the denoising backbone
for a diffusion model that generates crystal structures (atom types + fractional coordinates).

Architecture:
- Forward diffusion: gradually adds Gaussian noise to coordinates and atom type logits
- Reverse diffusion: EGNN predicts the noise to denoise step by step
- Property conditioning: conditions on HER activity, stability, synthesizability
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from typing import Optional, Tuple, Dict


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


# ============================================================================
# Noise Scheduler
# ============================================================================

class CosineNoiseSchedule:
    """Cosine noise schedule for the diffusion process."""

    def __init__(self, timesteps: int = 1000, s: float = 0.008):
        self.timesteps = timesteps
        self.s = s
        self._build_schedule()

    def _build_schedule(self):
        steps = torch.arange(self.timesteps + 1, dtype=torch.float32)
        t = steps / self.timesteps
        alpha_bar = torch.cos((t + self.s) / (1 + self.s) * math.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]  # normalize so alpha_bar[0] = 1.0

        self.betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        self.betas = torch.clamp(self.betas, max=0.999)

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)


class LinearNoiseSchedule:
    """Linear noise schedule."""

    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.timesteps = timesteps
        self._build_schedule(beta_start, beta_end)

    def _build_schedule(self, beta_start, beta_end):
        self.betas = torch.linspace(beta_start, beta_end, self.timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)


# ============================================================================
# EGNN Layer
# ============================================================================

class EGNNLayer(nn.Module):
    """
    E(n) Equivariant Graph Neural Network layer.
    Updates node features and coordinates in an E(n)-equivariant manner.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        edge_dim: int = 40,
        activation: nn.Module = nn.SiLU(),
        residual: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.residual = residual
        self.activation = activation

        # Edge MLP: (2 * hidden_dim + edge_dim) -> hidden_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim * 2),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Message MLP: (hidden_dim + hidden_dim) -> hidden_dim
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim * 2),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Coordinate update MLP: hidden_dim -> 1
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            activation,
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

        # Node update MLP: (hidden_dim + hidden_dim) -> hidden_dim
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim * 2),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Initialize coord_mlp final layer to small values
        nn.init.constant_(self.coord_mlp[-1].weight, 0)

    def forward(
        self,
        h: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Node features [N, hidden_dim]
            coords: Node coordinates [N, 3]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, edge_dim]
            batch: Batch indices [N]
        Returns:
            h_updated, coords_updated
        """
        row, col = edge_index

        # Compute coordinate differences
        coord_diff = coords[row] - coords[col]  # [E, 3]
        dist = torch.norm(coord_diff, dim=-1, keepdim=True)  # [E, 1]

        # Edge features: combine node features + edge attributes
        edge_input = torch.cat([h[row], h[col], edge_attr], dim=-1)
        edge_messages = self.edge_mlp(edge_input)  # [E, hidden_dim]

        # Coordinate update (equivariant)
        coord_update = self.coord_mlp(edge_messages) * coord_diff / (dist + 1e-8)
        coord_updates = torch.zeros_like(coords)
        coord_updates.scatter_add_(0, row.unsqueeze(-1).expand(-1, 3), coord_update)

        # Aggregate messages per node
        messages = self.message_mlp(torch.cat([edge_messages, h[row]], dim=-1))
        aggregated = torch.zeros_like(h)
        aggregated.scatter_add_(0, row.unsqueeze(-1).expand(-1, messages.size(-1)), messages)

        # Node update
        h_new = self.node_mlp(torch.cat([h, aggregated], dim=-1))

        if self.residual:
            h_new = h + h_new
            coord_updates = coords + coord_updates

        return h_new, coord_updates


# ============================================================================
# Property Encoder (for conditioning)
# ============================================================================

class PropertyEncoder(nn.Module):
    """Encodes target properties into a conditioning vector."""

    def __init__(self, num_properties: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.num_properties = num_properties
        self.encoder = nn.Sequential(
            nn.Linear(num_properties, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, properties: torch.Tensor) -> torch.Tensor:
        """
        Args:
            properties: [B, num_properties] target values (ΔG_H, stability, synthesis)
        Returns:
            condition: [B, hidden_dim]
        """
        return self.encoder(properties)


# ============================================================================
# EGNN Denoiser (main model)
# ============================================================================

class EGNNDenoiser(nn.Module):
    """
    EGNN-based denoising network for crystal structure diffusion.

    Handles:
    - Continuous noise prediction for fractional coordinates
    - Discrete noise prediction for atom types (logits)
    - Property conditioning for guided generation
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        max_atomic_number: int = 100,
        edge_dim: int = 40,
        num_timesteps: int = 1000,
        condition_on_property: bool = True,
        num_properties: int = 3,
        attn_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_atomic_number = max_atomic_number
        self.num_timesteps = num_timesteps
        self.condition_on_property = condition_on_property

        # Time embedding (sinusoidal)
        self.time_dim = hidden_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Node feature embedding
        self.atom_embedding = nn.Embedding(max_atomic_number + 1, hidden_dim)
        self.coord_embedding = nn.Linear(3, hidden_dim)

        # Initial node projection
        self.node_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Property encoder
        if condition_on_property:
            self.property_encoder = PropertyEncoder(num_properties, hidden_dim)
        else:
            self.property_encoder = None

        # EGNN layers
        self.egnn_layers = nn.ModuleList([
            EGNNLayer(hidden_dim=hidden_dim, edge_dim=edge_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Output heads
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

        self.atom_type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max_atomic_number + 1),
        )

        # Property prediction head (auxiliary task)
        self.property_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_properties),
        )

        # Initialize output heads
        nn.init.xavier_uniform_(self.coord_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.coord_head[-1].bias)
        nn.init.xavier_uniform_(self.atom_type_head[-1].weight, gain=0.01)

    def _get_timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal timestep embeddings."""
        half_dim = self.time_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.time_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(
        self,
        atom_types: torch.Tensor,
        frac_coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        t: torch.Tensor,
        batch: torch.Tensor,
        lattice: Optional[torch.Tensor] = None,
        properties: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            atom_types: Long tensor [N] of atom type indices (noisy)
            frac_coords: [N, 3] noisy fractional coordinates
            edge_index: [2, E]
            edge_attr: [E, edge_dim]
            t: [B] timestep
            batch: [N] batch indices
            lattice: [B, 3, 3] lattice matrices
            properties: [B, num_properties] target property values
        Returns:
            dict with 'coord_noise', 'atom_logits', 'property_pred'
        """
        # Embed atom types
        h_atom = self.atom_embedding(atom_types)  # [N, hidden_dim]

        # Embed coordinates
        h_coord = self.coord_embedding(frac_coords)  # [N, hidden_dim]

        # Time embedding
        time_emb = self._get_timestep_embedding(t)  # [B, hidden_dim]
        time_emb_per_node = time_emb[batch]  # [N, hidden_dim]

        # Combine node features
        h = torch.cat([h_atom, h_coord, time_emb_per_node], dim=-1)
        h = self.node_proj(h)

        # Add property conditioning
        if self.condition_on_property and properties is not None:
            prop_emb = self.property_encoder(properties)  # [B, hidden_dim]
            prop_emb_per_node = prop_emb[batch]  # [N, hidden_dim]
            h = h + prop_emb_per_node

        # EGNN layers
        for layer in self.egnn_layers:
            h, frac_coords_out = layer(h, frac_coords, edge_index, edge_attr, batch)

        # Output predictions
        coord_pred = self.coord_head(h)  # [N, 3] - predicted noise on coordinates
        atom_logits = self.atom_type_head(h)  # [N, max_atomic+1]
        prop_pred = self.property_head(global_mean_pool(h, batch))  # [B, num_properties]

        return {
            'coord_noise': coord_pred,
            'atom_logits': atom_logits,
            'property_pred': prop_pred,
            'frac_coords_out': frac_coords_out,
            'node_features': h,
        }


# ============================================================================
# Full Diffusion Model
# ============================================================================

class CrystalDiffusionModel(nn.Module):
    """
    Complete diffusion model for crystal structure generation.

    Combines the noise schedule, EGNN denoiser, and sampling procedures.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        max_atomic_number: int = 100,
        edge_dim: int = 40,
        num_timesteps: int = 1000,
        schedule_type: str = 'cosine',
        condition_on_property: bool = True,
        num_properties: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        self.max_atomic_number = max_atomic_number

        # Noise schedule
        if schedule_type == 'cosine':
            self.schedule = CosineNoiseSchedule(timesteps=num_timesteps)
        else:
            self.schedule = LinearNoiseSchedule(timesteps=num_timesteps)

        # Denoiser network
        self.denoiser = EGNNDenoiser(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_atomic_number=max_atomic_number,
            edge_dim=edge_dim,
            num_timesteps=num_timesteps,
            condition_on_property=condition_on_property,
            num_properties=num_properties,
            dropout=dropout,
        )

        # Loss functions
        self.coord_loss_fn = nn.MSELoss()
        self.atom_loss_fn = nn.CrossEntropyLoss()
        self.prop_loss_fn = nn.MSELoss()

    def q_sample(
        self,
        frac_coords: torch.Tensor,
        atom_types: torch.Tensor,
        t: torch.Tensor,
        noise_coords: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: add noise to coordinates and atom types.
        """
        batch_size = t.shape[0]
        device = frac_coords.device

        sqrt_alpha = self.schedule.sqrt_alphas_cumprod.to(device)[t]
        sqrt_one_minus_alpha = self.schedule.sqrt_one_minus_alphas_cumprod.to(device)[t]

        # Noise on coordinates
        if noise_coords is None:
            noise_coords = torch.randn_like(frac_coords)

        noisy_coords = (
            sqrt_alpha.unsqueeze(-1) * frac_coords +
            sqrt_one_minus_alpha.unsqueeze(-1) * noise_coords
        )

        # For atom types, we use a simple categorical noise
        # (random atom types mixed in proportion to sqrt_alpha)
        noisy_atom_types = atom_types.clone()
        # Randomly replace some atom types based on noise level
        random_mask = torch.rand(atom_types.shape[0], device=device) < (1 - sqrt_alpha[batch[atom_types.shape[0] // batch_size]]).item()
        # simplified: just keep original types, atom type diffusion handled separately
        noise_atom = noise_coords  # reuse for API consistency

        return noisy_coords, noisy_atom_types, noise_coords, noise_atom

    def p_loss(
        self,
        data,
        properties: Optional[torch.Tensor] = None,
        noise_coords: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss for one batch.

        Args:
            data: PyG batch with atom_types, frac_coords, edge_index, edge_attr, lattice
            properties: [B, 3] target property values (optional)
        Returns:
            dict of losses
        """
        batch_size = data.num_graphs
        device = data.atom_types.device

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)

        # Add noise to coordinates
        sqrt_alpha = self.schedule.sqrt_alphas_cumprod.to(device)[t]
        sqrt_one_minus_alpha = self.schedule.sqrt_one_minus_alphas_cumprod.to(device)[t]

        if noise_coords is None:
            noise_coords = torch.randn_like(data.frac_coords)

        noisy_frac_coords = (
            sqrt_alpha[data.batch].unsqueeze(-1) * data.frac_coords +
            sqrt_one_minus_alpha[data.batch].unsqueeze(-1) * noise_coords
        )

        # Forward pass through denoiser
        output = self.denoiser(
            atom_types=data.atom_types,
            frac_coords=noisy_frac_coords,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            t=t,
            batch=data.batch,
            lattice=getattr(data, 'lattice', None),
            properties=properties,
        )

        # Coordinate MSE loss (predict the noise)
        coord_loss = self.coord_loss_fn(output['coord_noise'], noise_coords)

        # Atom type cross-entropy loss
        atom_loss = self.atom_loss_fn(output['atom_logits'], data.atom_types)

        # Property prediction loss (auxiliary)
        if properties is not None:
            prop_loss = self.prop_loss_fn(output['property_pred'], properties)
        else:
            prop_loss = torch.tensor(0.0, device=device)

        # Total diffusion loss
        diffusion_loss = coord_loss + 0.1 * atom_loss

        return {
            'diffusion_loss': diffusion_loss,
            'coord_loss': coord_loss,
            'atom_loss': atom_loss,
            'prop_loss': prop_loss,
        }

    @torch.no_grad()
    def sample(
        self,
        num_atoms: int,
        batch_size: int = 1,
        lattice: Optional[torch.Tensor] = None,
        properties: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        guidance_scale: float = 0.0,
        return_trajectory: bool = False,
        initial_atom_types: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample new crystal structures from the diffusion model.

        Args:
            num_atoms: Number of atoms per sample
            batch_size: Number of samples to generate
            lattice: [B, 3, 3] lattice matrices (can be None for random)
            properties: [B, 3] target property values for conditional generation
            edge_index: Pre-defined edge connectivity (can be None for fully connected)
            edge_attr: Pre-defined edge features
            guidance_scale: Classifier-free guidance strength
            return_trajectory: Return the full denoising trajectory
        Returns:
            dict with 'frac_coords', 'atom_types', 'lattice'
        """
        device = next(self.parameters()).device
        total_nodes = num_atoms * batch_size

        # Create batch indices
        batch = torch.arange(batch_size, device=device).repeat_interleave(num_atoms)

        # Random initial coordinates (from standard normal)
        frac_coords = torch.randn(total_nodes, 3, device=device)

        # Atom types: use provided initial values or random (for fully trained model)
        if initial_atom_types is not None:
            atom_types = initial_atom_types.to(device)
        else:
            atom_types = torch.randint(1, self.max_atomic_number + 1, (total_nodes,), device=device)

        # Default lattice if not provided
        if lattice is None:
            lattice = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
            # Add random distortion for variety
            lattice = lattice + 0.1 * torch.randn(batch_size, 3, 3, device=device)

        # Build fully-connected edge index if not provided
        if edge_index is None:
            edges = []
            for i in range(batch_size):
                offset = i * num_atoms
                for j in range(num_atoms):
                    for k in range(num_atoms):
                        if j != k:
                            edges.append([offset + j, offset + k])
            edge_index = torch.tensor(edges, device=device).t().contiguous()
            edge_attr = torch.zeros(edge_index.size(1), 40, device=device)

        trajectory = [] if return_trajectory else None

        # Denoising loop
        for t_step in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), t_step, device=device, dtype=torch.long)

            # Predict noise
            output = self.denoiser(
                atom_types=atom_types,
                frac_coords=frac_coords,
                edge_index=edge_index,
                edge_attr=edge_attr,
                t=t,
                batch=batch,
                lattice=lattice,
                properties=properties,
            )

            pred_noise = output['coord_noise']

            # DDPM sampling step
            alpha_t = self.schedule.alphas.to(device)[t]  # [batch_size]
            alpha_cumprod_t = self.schedule.alphas_cumprod.to(device)[t]  # [batch_size]
            beta_t = self.schedule.betas.to(device)[t]  # [batch_size]

            # Expand to per-node shape [total_nodes, 1]
            alpha_per_node = alpha_t[batch].unsqueeze(-1)
            alpha_cumprod_per_node = alpha_cumprod_t[batch].unsqueeze(-1)
            beta_per_node = beta_t[batch].unsqueeze(-1)

            # DDPM update: x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * noise_pred)
            coef_x = 1.0 / torch.sqrt(alpha_per_node)
            coef_pred = (1.0 - alpha_per_node) / torch.sqrt(1.0 - alpha_cumprod_per_node + 1e-8)

            frac_coords = coef_x * frac_coords - coef_pred * pred_noise

            # Add noise for intermediate steps
            if t_step > 0:
                noise = torch.randn_like(frac_coords)
                sigma = torch.sqrt(beta_per_node)
                frac_coords = frac_coords + sigma * noise

            # Update atom types (softmax + sampling for discrete features)
            atom_logits = output['atom_logits']
            if t_step > 0:
                # Gumbel-softmax for discrete diffusion
                atom_types = torch.argmax(atom_logits + torch.randn_like(atom_logits) * 0.1, dim=-1)
            else:
                atom_types = torch.argmax(atom_logits, dim=-1)

            if return_trajectory:
                trajectory.append({
                    'frac_coords': frac_coords.clone(),
                    'atom_types': atom_types.clone(),
                    't': t_step,
                })

        # Wrap coordinates to [0, 1] range (fractional)
        frac_coords = frac_coords % 1.0

        result = {
            'frac_coords': frac_coords,
            'atom_types': atom_types,
            'lattice': lattice,
            'batch': batch,
        }

        if return_trajectory:
            result['trajectory'] = trajectory

        return result
