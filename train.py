"""
Training script for the Crystal Diffusion Model with Multi-Task Optimization.

Training pipeline:
1. Load CIF crystal structures from Materials Project database
2. Train EGNN-based diffusion model to learn structure distributions
3. Jointly optimize HER activity, stability, and synthesizability
4. Save model checkpoints and training metrics
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from pathlib import Path
from typing import Optional, Dict, List
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.material_dataset import CrystalDataset, collate_fn
from models.diffusion_model import CrystalDiffusionModel
from models.optimization import (
    MaterialPropertyOptimizer,
    MultiObjectiveLoss,
)
from utils.geo_utils import batch_evaluate_materials, evaluate_material
from utils.vis import create_all_visualizations, plot_loss_curve


def parse_args():
    parser = argparse.ArgumentParser(description="Train Crystal Diffusion Model")

    # Data
    parser.add_argument('--cif_dir', type=str, default='data/raw',
                        help='Directory containing CIF files')
    parser.add_argument('--max_samples', type=int, default=5000,
                        help='Maximum number of training samples')
    parser.add_argument('--max_atoms', type=int, default=50,
                        help='Maximum atoms per structure')

    # Model
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension of EGNN')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of EGNN layers')
    parser.add_argument('--num_timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    parser.add_argument('--schedule', type=str, default='cosine',
                        choices=['cosine', 'linear'])

    # Training
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Training device')

    # Multi-task weights
    parser.add_argument('--w_diffusion', type=float, default=1.0,
                        help='Weight for diffusion loss')
    parser.add_argument('--w_her', type=float, default=0.5,
                        help='Weight for HER loss')
    parser.add_argument('--w_stability', type=float, default=0.3,
                        help='Weight for stability loss')
    parser.add_argument('--w_synthesis', type=float, default=0.2,
                        help='Weight for synthesis loss')

    # Output
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_every', type=int, default=10,
                        help='Log every N steps')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers (0 for Windows)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: CrystalDiffusionModel,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    property_optimizer: MaterialPropertyOptimizer,
    loss_fn: MultiObjectiveLoss,
    device: torch.device,
    epoch: int,
    log_every: int = 10,
) -> Dict[str, float]:
    """Train one epoch."""
    model.train()
    property_optimizer.train()

    total_loss = 0.0
    total_diff_loss = 0.0
    total_her_loss = 0.0
    total_stab_loss = 0.0
    total_syn_loss = 0.0
    num_batches = 0

    for batch_idx, data in enumerate(dataloader):
        data = data.to(device)

        # Get property targets
        batch_size = data.num_graphs
        properties = torch.stack([
            data.her_score,
            torch.zeros(batch_size, device=device),  # placeholder stability
            torch.zeros(batch_size, device=device),  # placeholder synthesis
        ], dim=1)

        # Forward pass: diffusion loss
        diff_outputs = model.p_loss(data, properties=properties)
        diffusion_loss = diff_outputs['diffusion_loss']

        # Get node features from denoiser for property prediction
        with torch.no_grad():
            output = model.denoiser(
                atom_types=data.atom_types,
                frac_coords=data.frac_coords,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                t=torch.randint(0, model.num_timesteps, (batch_size,), device=device),
                batch=data.batch,
                properties=properties,
            )

        # Property prediction
        prop_outputs = property_optimizer(
            node_features=output['node_features'],
            frac_coords=output['frac_coords_out'],
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=data.batch,
            num_atoms_per_graph=data.num_atoms,
        )

        # Multi-task loss
        losses = loss_fn(
            diffusion_loss=diffusion_loss,
            her_output=prop_outputs['her'],
            stability_output=prop_outputs['stability'],
            synthesis_output=prop_outputs['synthesis'],
        )

        # Backward pass
        optimizer.zero_grad()
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(property_optimizer.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate
        total_loss += losses['total_loss'].item()
        total_diff_loss += losses['diffusion_loss'].item()
        total_her_loss += losses.get('her_loss', torch.tensor(0.0)).item()
        total_stab_loss += losses.get('stability_loss', torch.tensor(0.0)).item()
        total_syn_loss += losses.get('synthesis_loss', torch.tensor(0.0)).item()
        num_batches += 1

        if batch_idx % log_every == 0:
            print(f"  Epoch {epoch:3d} | Batch {batch_idx:4d}/{len(dataloader)} | "
                  f"Loss: {losses['total_loss'].item():.4f} | "
                  f"Diff: {diffusion_loss.item():.4f} | "
                  f"HER: {losses.get('her_loss', torch.tensor(0.0)).item():.4f}")

    return {
        'train_loss': total_loss / num_batches,
        'diff_loss': total_diff_loss / num_batches,
        'her_loss': total_her_loss / num_batches,
        'stab_loss': total_stab_loss / num_batches,
        'syn_loss': total_syn_loss / num_batches,
    }


@torch.no_grad()
def validate(
    model: CrystalDiffusionModel,
    dataloader: DataLoader,
    property_optimizer: MaterialPropertyOptimizer,
    loss_fn: MultiObjectiveLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    property_optimizer.eval()

    total_loss = 0.0
    num_batches = 0

    for data in dataloader:
        data = data.to(device)

        batch_size = data.num_graphs
        properties = torch.stack([
            data.her_score,
            torch.zeros(batch_size, device=device),
            torch.zeros(batch_size, device=device),
        ], dim=1)

        diff_outputs = model.p_loss(data, properties=properties)
        diffusion_loss = diff_outputs['diffusion_loss']

        output = model.denoiser(
            atom_types=data.atom_types,
            frac_coords=data.frac_coords,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            t=torch.randint(0, model.num_timesteps, (batch_size,), device=device),
            batch=data.batch,
            properties=properties,
        )

        prop_outputs = property_optimizer(
            node_features=output['node_features'],
            frac_coords=output['frac_coords_out'],
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=data.batch,
            num_atoms_per_graph=data.num_atoms,
        )

        losses = loss_fn(
            diffusion_loss=diffusion_loss,
            her_output=prop_outputs['her'],
            stability_output=prop_outputs['stability'],
            synthesis_output=prop_outputs['synthesis'],
        )

        total_loss += losses['total_loss'].item()
        num_batches += 1

    return {'val_loss': total_loss / num_batches}


def save_checkpoint(
    model: CrystalDiffusionModel,
    property_optimizer: MaterialPropertyOptimizer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    path: str,
):
    """Save model checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'property_optimizer_state_dict': property_optimizer.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': {
            'hidden_dim': model.hidden_dim,
            'num_layers': model.denoiser.num_layers,
            'num_timesteps': model.num_timesteps,
        },
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: CrystalDiffusionModel,
    property_optimizer: MaterialPropertyOptimizer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device('cuda'),
) -> int:
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    property_optimizer.load_state_dict(checkpoint['property_optimizer_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    print(f"Loaded checkpoint from {path} (epoch {epoch})")
    return epoch


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training config: {vars(args)}")

    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Save config
    with open(Path(args.output_dir) / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # ====================================================================
    # Dataset
    # ====================================================================
    print("\nLoading dataset...")
    dataset = CrystalDataset(
        cif_dir=args.cif_dir,
        max_atoms=args.max_atoms,
        max_samples=args.max_samples,
        cutoff_radius=5.0,
    )
    print(f"Dataset size: {len(dataset)}")

    # Split into train/val (80/20, larger val for reliable loss estimation)
    val_size = max(int(0.2 * len(dataset)), 2000)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        drop_last=False,
    )

    # ====================================================================
    # Model
    # ====================================================================
    print("\nInitializing models...")
    model = CrystalDiffusionModel(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_timesteps=args.num_timesteps,
        schedule_type=args.schedule,
        condition_on_property=True,
        num_properties=3,
        dropout=0.1,
    ).to(device)

    property_optimizer = MaterialPropertyOptimizer(
        hidden_dim=args.hidden_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) + \
                   sum(p.numel() for p in property_optimizer.parameters())
    print(f"Total parameters: {total_params:,}")

    # ====================================================================
    # Optimizer & Loss
    # ====================================================================
    all_params = list(model.parameters()) + list(property_optimizer.parameters())
    optimizer = AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
    # Reduce LR only when val loss plateaus — avoids premature decay
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                  min_lr=1e-6)

    loss_fn = MultiObjectiveLoss(
        w_diffusion=args.w_diffusion,
        w_her=args.w_her,
        w_stability=args.w_stability,
        w_synthesis=args.w_synthesis,
    )

    # ====================================================================
    # Resume from checkpoint
    # ====================================================================
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        start_epoch = load_checkpoint(args.resume, model, property_optimizer, optimizer, device)

    # ====================================================================
    # Training Loop
    # ====================================================================
    print(f"\nStarting training from epoch {start_epoch + 1}...")
    print("=" * 70)

    train_losses = []
    val_losses = []
    her_losses_list = []
    stab_losses_list = []
    syn_losses_list = []

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 15

    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, optimizer, train_loader,
            property_optimizer, loss_fn, device,
            epoch, args.log_every,
        )

        # Validate
        val_metrics = validate(
            model, val_loader,
            property_optimizer, loss_fn, device,
        )

        scheduler.step(val_metrics['val_loss'])

        # Record metrics
        train_losses.append(train_metrics['train_loss'])
        val_losses.append(val_metrics['val_loss'])
        her_losses_list.append(train_metrics['her_loss'])
        stab_losses_list.append(train_metrics['stab_loss'])
        syn_losses_list.append(train_metrics['syn_loss'])

        epoch_time = time.time() - epoch_start

        # Print progress
        print(f"Epoch {epoch:3d}/{args.epochs} | Time: {epoch_time:.1f}s | "
              f"Train: {train_metrics['train_loss']:.4f} | Val: {val_metrics['val_loss']:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0
            save_checkpoint(
                model, property_optimizer, optimizer, epoch,
                {'val_loss': best_val_loss, 'train_loss': train_metrics['train_loss']},
                str(Path(args.checkpoint_dir) / 'best_model.pt'),
            )
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch}")
            break

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(
                model, property_optimizer, optimizer, epoch,
                {'val_loss': val_metrics['val_loss'], 'train_loss': train_metrics['train_loss']},
                str(Path(args.checkpoint_dir) / f'checkpoint_epoch_{epoch}.pt'),
            )

    # ====================================================================
    # Save final model
    # ====================================================================
    save_checkpoint(
        model, property_optimizer, optimizer, args.epochs,
        {'val_loss': val_losses[-1] if val_losses else 0, 'train_loss': train_losses[-1] if train_losses else 0},
        str(Path(args.checkpoint_dir) / 'final_model.pt'),
    )

    # ====================================================================
    # Save training metrics
    # ====================================================================
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'her_losses': her_losses_list,
        'stab_losses': stab_losses_list,
        'syn_losses': syn_losses_list,
        'best_val_loss': best_val_loss,
        'final_val_loss': val_losses[-1] if val_losses else 0,
    }
    with open(Path(args.output_dir) / 'training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Quick visualization of training progress
    plot_loss_curve(
        train_losses, val_losses,
        her_losses=her_losses_list,
        stab_losses=stab_losses_list,
        syn_losses=syn_losses_list,
        save_path=str(Path(args.output_dir) / 'loss_curve.png'),
    )

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"Metrics saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
