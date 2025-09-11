"""
Training script for EigenPlaces algorithm
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from datasets.gps_dataset import GPSImageDataset, TripletDataset
from utils.evaluation import VPREvaluator
from algorithms.eigenplaces.eigenplaces_model import EigenPlacesModel, EigenPlacesLoss, get_eigenplaces_transforms


def setup_logging(output_dir: Path):
    """Setup logging configuration"""
    log_file = output_dir / 'training.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_datasets_and_loaders(config: dict):
    """Create datasets and data loaders"""
    
    # Get transforms
    train_transform = get_eigenplaces_transforms(
        image_size=config.get('image_size', 224),
        augment=True
    )
    test_transform = get_eigenplaces_transforms(
        image_size=config.get('image_size', 224),
        augment=False
    )
    
    # Create datasets
    train_dataset = GPSImageDataset(
        csv_file=config['train_csv'],
        base_path=config['train_base_path'],
        transform=train_transform,
        camera_filter=config.get('camera_filter', 'all')
    )
    
    test_dataset = GPSImageDataset(
        csv_file=config['test_csv'],
        base_path=config['test_base_path'],
        transform=test_transform,
        camera_filter=config.get('camera_filter', 'all')
    )
    
    # Create triplet dataset for training
    triplet_dataset = TripletDataset(
        train_dataset,
        positive_threshold=config.get('positive_threshold', 25),
        negative_threshold=config.get('negative_threshold', 200)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        triplet_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, test_loader, test_dataset


def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: EigenPlacesLoss, 
                optimizer: optim.Optimizer, device: torch.device, epoch: int, 
                logger: logging.Logger, config: dict):
    """Train for one epoch"""
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (anchor, positive, negative) in enumerate(progress_bar):
        # Move to device
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        anchor_desc = model(anchor)
        positive_desc = model(positive)
        negative_desc = model(negative)
        
        # Compute loss
        loss = criterion(
            anchor_desc, 
            positive_desc, 
            negative_desc, 
            loss_type=config.get('loss_type', 'viewpoint_contrastive')
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Log progress
        if batch_idx % config.get('log_interval', 100) == 0:
            logger.info(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                       f'Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / num_batches
    logger.info(f'Epoch {epoch} completed - Avg Loss: {avg_loss:.4f}')
    
    return avg_loss


def evaluate_model(model: nn.Module, test_loader: DataLoader, test_dataset: GPSImageDataset,
                  device: torch.device, config: dict, logger: logging.Logger):
    """Evaluate model on test set"""
    
    model.eval()
    
    logger.info("Extracting features for evaluation...")
    
    # Extract features for all test images
    all_features = []
    all_locations = []
    
    with torch.no_grad():
        for batch_idx, (images, _, locations, _) in enumerate(tqdm(test_loader, desc="Extracting features")):
            images = images.to(device)
            
            # Extract features
            features = model(images)
            
            all_features.append(features.cpu())
            all_locations.extend(locations)
    
    # Concatenate all features
    all_features = torch.cat(all_features, dim=0).numpy()
    
    logger.info(f"Extracted features shape: {all_features.shape}")
    
    # Create evaluator
    evaluator = VPREvaluator(
        distance_threshold=config.get('distance_threshold', 25),
        recall_k_values=config.get('recall_k_values', [1, 5, 10, 20])
    )
    
    # Evaluate (using cosine distance for eigenplaces)
    results = evaluator.evaluate(all_features, all_locations, distance_metric='cosine')
    
    # Log results
    logger.info("Evaluation Results:")
    logger.info(f"Total queries: {results['total_queries']}")
    logger.info(f"Queries with matches: {results['queries_with_matches']}")
    
    for k, recall in results['recall_at_k'].items():
        logger.info(f"Recall@{k}: {recall:.3f}")
    
    for k, precision in results['precision_at_k'].items():
        logger.info(f"Precision@{k}: {precision:.3f}")
    
    logger.info(f"mAP: {results['mean_average_precision']:.3f}")
    
    return results


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, 
                   loss: float, results: dict, output_dir: Path, is_best: bool = False):
    """Save model checkpoint"""
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'results': results
    }
    
    # Save regular checkpoint
    checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = output_dir / 'best_model.pth'
        torch.save(checkpoint, best_path)


def main():
    parser = argparse.ArgumentParser(description='Train EigenPlaces for VPR')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    logger.info("Starting EigenPlaces training...")
    logger.info(f"Configuration: {config}")
    
    # Set device
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"Using device: {device}")
    
    # Create datasets and loaders
    train_loader, test_loader, test_dataset = create_datasets_and_loaders(config)
    logger.info(f"Training set size: {len(train_loader.dataset)}")
    logger.info(f"Test set size: {len(test_dataset)}")
    
    # Create model
    model = EigenPlacesModel(
        backbone=config.get('backbone', 'resnet50'),
        output_dim=config.get('output_dim', 2048),
        eigen_scales=config.get('eigen_scales', [1, 2, 4]),
        pretrained=config.get('pretrained', True)
    )
    model = model.to(device)
    
    logger.info(f"Model created: {model.__class__.__name__}")
    logger.info(f"Output dimension: {config.get('output_dim', 2048)}")
    logger.info(f"Eigen scales: {config.get('eigen_scales', [1, 2, 4])}")
    
    # Create loss function
    criterion = EigenPlacesLoss(
        margin=config.get('margin', 0.1),
        temperature=config.get('temperature', 0.07),
        viewpoint_weight=config.get('viewpoint_weight', 0.1)
    )
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', 0.001),
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get('lr_step_size', 20),
        gamma=config.get('lr_gamma', 0.5)
    )
    
    # Training loop
    best_recall = 0.0
    num_epochs = config.get('num_epochs', 50)
    
    for epoch in range(1, num_epochs + 1):
        # Train
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, logger, config)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Learning rate: {current_lr}")
        
        # Evaluate
        if epoch % config.get('eval_every', 5) == 0:
            results = evaluate_model(model, test_loader, test_dataset, device, config, logger)
            
            # Check if this is the best model
            current_recall = results['recall_at_k'][1]  # Recall@1
            is_best = current_recall > best_recall
            if is_best:
                best_recall = current_recall
                logger.info(f"New best model! Recall@1: {best_recall:.3f}")
            
            # Save checkpoint
            save_checkpoint(model, optimizer, epoch, avg_loss, results, output_dir, is_best)
        
        # Save regular checkpoint
        if epoch % config.get('save_every_n_epochs', 10) == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss, {}, output_dir)
    
    logger.info("Training completed!")
    logger.info(f"Best Recall@1: {best_recall:.3f}")


if __name__ == "__main__":
    main()
