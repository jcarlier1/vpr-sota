"""
NetVLAD Training Script for Visual Place Recognition
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from datasets.gps_dataset import create_datasets, create_dataloaders, TripletDataset
from algorithms.netvlad.netvlad_model import (
    create_netvlad_model, 
    extract_features_for_clustering,
    compute_cluster_centers,
    TripletLoss
)
from utils.evaluation import VPREvaluator, print_evaluation_results


def setup_logging(log_dir):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"netvlad_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def cluster_features(model, train_loader, num_clusters, device, logger):
    """Initialize NetVLAD by clustering features"""
    logger.info("Extracting features for clustering...")
    
    # Extract features from training data
    descriptors = extract_features_for_clustering(
        model=model,
        dataloader=train_loader,
        num_descriptors=50000,
        device=device
    )
    
    logger.info(f"Extracted {len(descriptors)} descriptors")
    
    # Compute cluster centers
    logger.info(f"Computing {num_clusters} cluster centers...")
    cluster_centers = compute_cluster_centers(descriptors, num_clusters)
    
    # Initialize NetVLAD parameters
    model.init_vlad_params(cluster_centers, descriptors)
    
    logger.info("NetVLAD initialization complete")
    
    return cluster_centers


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, logger):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        if batch is None:  # Skip invalid triplets
            continue
            
        # Extract triplet data
        anchor_data = batch['anchor']
        positive_data = batch['positive']
        negative_data = batch['negative']
        
        # Move to device
        anchor_images = anchor_data['image'].to(device)
        positive_images = positive_data['image'].to(device)
        negative_images = negative_data['image'].to(device)
        
        # Forward pass
        anchor_features = model(anchor_images)
        positive_features = model(positive_images)
        negative_features = model(negative_images)
        
        # Compute loss
        loss = criterion(anchor_features, positive_features, negative_features)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 50 == 0:
            logger.info(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / max(num_batches, 1)
    logger.info(f'Epoch {epoch} completed. Average loss: {avg_loss:.4f}')
    
    return avg_loss


def extract_all_features(model, data_loader, device):
    """Extract features for all images in the dataset"""
    model.eval()
    features = []
    gps_coords = []
    
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, dict):
                images = batch['image'].to(device)
                gps = batch['gps_coords'].cpu().numpy()
            else:
                images, _, gps = batch[0].to(device), batch[1], batch[2].cpu().numpy()
            
            # Extract features
            batch_features = model(images)
            features.append(batch_features.cpu().numpy())
            gps_coords.append(gps)
    
    features = np.vstack(features)
    gps_coords = np.vstack(gps_coords)
    
    return features, gps_coords


def evaluate_model(model, train_dataset, test_dataset, device, logger):
    """Evaluate the model on test data"""
    logger.info("Evaluating model...")
    
    # Create evaluation data loaders
    train_eval_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )
    test_eval_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Extract features
    train_features, train_gps = extract_all_features(model, train_eval_loader, device)
    test_features, test_gps = extract_all_features(model, test_eval_loader, device)
    
    # Evaluate
    evaluator = VPREvaluator(distance_threshold=25.0, k_values=[1, 5, 10, 20])
    
    results = evaluator.evaluate(
        query_features=test_features,
        database_features=train_features,
        query_gps=test_gps,
        database_gps=train_gps
    )
    
    print_evaluation_results(results)
    
    return results


def save_checkpoint(model, optimizer, epoch, loss, results, checkpoint_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'results': results,
        'model_config': {
            'num_clusters': model.num_clusters,
            'encoder_dim': model.encoder_dim
        }
    }
    
    torch.save(checkpoint, checkpoint_path)


def train_netvlad(config):
    """Main training function"""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logging(config['output_dir'])
    
    logger.info(f"Using device: {device}")
    logger.info(f"Configuration: {config}")
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset, test_dataset = create_datasets(
        train_csv=config['train_csv'],
        test_csv=config['test_csv'],
        train_base_path=config['train_base_path'],
        test_base_path=config['test_base_path'],
        camera_filter=config.get('camera_filter'),
        positive_threshold=config.get('positive_threshold', 25.0),
        negative_threshold=config.get('negative_threshold', 50.0)
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Create model
    logger.info("Creating NetVLAD model...")
    model = create_netvlad_model(
        num_clusters=config['num_clusters'],
        pretrained=config['pretrained'],
        vladv2=config.get('vladv2', False)
    )
    model.to(device)
    
    # Create data loaders for clustering
    clustering_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, 
        num_workers=4, pin_memory=True
    )
    
    # Initialize NetVLAD by clustering
    cluster_centers = cluster_features(
        model=model, 
        train_loader=clustering_loader,
        num_clusters=config['num_clusters'],
        device=device,
        logger=logger
    )
    
    # Create triplet training data
    triplet_train_dataset = TripletDataset(train_dataset)
    train_loader = DataLoader(
        triplet_train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    
    # Setup training
    criterion = TripletLoss(margin=config['margin'])
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config['lr_step_size'], 
        gamma=config['lr_gamma']
    )
    
    # Training loop
    best_recall = 0.0
    patience_counter = 0
    
    for epoch in range(1, config['epochs'] + 1):
        # Train
        avg_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, logger)
        
        # Update learning rate
        scheduler.step()
        
        # Evaluate periodically
        if epoch % config['eval_every'] == 0:
            results = evaluate_model(model, train_dataset, test_dataset, device, logger)
            
            # Check if best model
            current_recall = results['recall_at_k'][5]  # Recall@5
            if current_recall > best_recall:
                best_recall = current_recall
                patience_counter = 0
                
                # Save best model
                best_checkpoint_path = os.path.join(config['output_dir'], 'best_model.pth')
                save_checkpoint(model, optimizer, epoch, avg_loss, results, best_checkpoint_path)
                logger.info(f"New best model saved with Recall@5: {current_recall:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= config['patience']:
                logger.info(f"Early stopping after {epoch} epochs")
                break
        
        # Save regular checkpoint
        if epoch % config['save_every'] == 0:
            checkpoint_path = os.path.join(config['output_dir'], f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, avg_loss, {}, checkpoint_path)
    
    logger.info("Training completed!")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train NetVLAD for Visual Place Recognition')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Train model
    train_netvlad(config)


if __name__ == "__main__":
    main()
