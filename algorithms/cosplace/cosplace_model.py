"""
CosPlace Implementation for VPR
Based on "Learning Generalized Visual Place Recognition using Neural Networks"

CosPlace uses:
- ResNet backbone for feature extraction
- Cosine similarity for place matching
- Multi-scale spatial pooling
- Contrastive learning with hard negative mining
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
import math


class MultiScalePooling(nn.Module):
    """Multi-scale spatial pooling for different spatial resolutions"""
    
    def __init__(self, input_dim: int, output_dim: int, scales: List[int] = [1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(scale) for scale in scales
        ])
        
        # Calculate total dimension after concatenation
        total_dim = input_dim * sum(scale * scale for scale in scales)
        
        # Projection layer to output dimension
        self.projection = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] feature maps
        Returns:
            pooled: [B, output_dim] multi-scale pooled features
        """
        pooled_features = []
        
        for pool in self.pools:
            pooled = pool(x)  # [B, C, scale, scale]
            pooled = pooled.view(x.size(0), -1)  # [B, C * scale * scale]
            pooled_features.append(pooled)
        
        # Concatenate all scales
        concatenated = torch.cat(pooled_features, dim=1)
        
        # Project to output dimension
        output = self.projection(concatenated)
        
        return output


class CosPlaceBackbone(nn.Module):
    """ResNet backbone with multi-scale pooling for CosPlace"""
    
    def __init__(self, backbone: str = 'resnet50', output_dim: int = 2048, 
                 pooling_scales: List[int] = [1, 2, 4], pretrained: bool = True):
        super().__init__()
        
        # Load backbone
        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            backbone_dim = 512
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            backbone_dim = 512
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            backbone_dim = 2048
        elif backbone == 'resnet101':
            resnet = models.resnet101(pretrained=pretrained)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove final layers (avgpool and fc)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Multi-scale pooling
        self.pooling = MultiScalePooling(backbone_dim, output_dim, pooling_scales)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] input images
        Returns:
            features: [B, output_dim] global descriptors
        """
        # Extract backbone features
        features = self.backbone(x)  # [B, backbone_dim, H', W']
        
        # Multi-scale pooling
        pooled = self.pooling(features)  # [B, output_dim]
        
        return pooled


class CosPlaceModel(nn.Module):
    """
    CosPlace: Learning place recognition with cosine similarity
    
    Key features:
    - Multi-scale spatial pooling
    - L2 normalized descriptors for cosine similarity
    - Contrastive learning framework
    """
    
    def __init__(self, 
                 backbone: str = 'resnet50',
                 output_dim: int = 2048,
                 pooling_scales: List[int] = [1, 2, 4],
                 pretrained: bool = True):
        super().__init__()
        
        self.output_dim = output_dim
        
        # Backbone with multi-scale pooling
        self.backbone = CosPlaceBackbone(
            backbone=backbone,
            output_dim=output_dim,
            pooling_scales=pooling_scales,
            pretrained=pretrained
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] input images
        Returns:
            descriptors: [B, output_dim] L2-normalized global descriptors
        """
        # Extract features
        features = self.backbone(x)
        
        # L2 normalize for cosine similarity
        descriptors = F.normalize(features, p=2, dim=1)
        
        return descriptors
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward method"""
        return self.forward(x)


class CosPlaceLoss(nn.Module):
    """
    CosPlace loss with cosine similarity and contrastive learning
    
    Uses cosine distance instead of Euclidean distance for triplet loss
    """
    
    def __init__(self, margin: float = 0.1, temperature: float = 0.07):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        
    def cosine_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute cosine distance (1 - cosine similarity)"""
        # Inputs are already L2 normalized, so cosine similarity is just dot product
        cosine_sim = torch.sum(x1 * x2, dim=1)
        cosine_dist = 1.0 - cosine_sim
        return cosine_dist
    
    def triplet_loss(self, anchor: torch.Tensor, positive: torch.Tensor, 
                    negative: torch.Tensor) -> torch.Tensor:
        """Triplet loss with cosine distance"""
        pos_dist = self.cosine_distance(anchor, positive)
        neg_dist = self.cosine_distance(anchor, negative)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
    
    def contrastive_loss(self, anchor: torch.Tensor, positive: torch.Tensor,
                        negative: torch.Tensor) -> torch.Tensor:
        """Contrastive loss with temperature scaling"""
        # Compute similarities
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature
        neg_sim = torch.sum(anchor * negative, dim=1) / self.temperature
        
        # Contrastive loss (InfoNCE style)
        logits = torch.stack([pos_sim, neg_sim], dim=1)
        targets = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
        
        loss = F.cross_entropy(logits, targets)
        return loss
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: torch.Tensor, loss_type: str = 'triplet') -> torch.Tensor:
        """
        CosPlace loss computation
        
        Args:
            anchor: [B, D] anchor descriptors
            positive: [B, D] positive descriptors
            negative: [B, D] negative descriptors
            loss_type: 'triplet' or 'contrastive'
        Returns:
            loss: Scalar loss value
        """
        if loss_type == 'triplet':
            return self.triplet_loss(anchor, positive, negative)
        elif loss_type == 'contrastive':
            return self.contrastive_loss(anchor, positive, negative)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")


class HardNegativeMiner(nn.Module):
    """Hard negative mining for more effective training"""
    
    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin
    
    def mine_hard_negatives(self, anchor: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        """
        Mine hard negatives from a batch of negative samples
        
        Args:
            anchor: [B, D] anchor descriptors
            negatives: [B, N, D] negative descriptors (N negatives per anchor)
        Returns:
            hard_negatives: [B, D] hardest negative for each anchor
        """
        B, N, D = negatives.shape
        
        # Compute cosine distances to all negatives
        anchor_expanded = anchor.unsqueeze(1).expand(-1, N, -1)  # [B, N, D]
        distances = 1.0 - torch.sum(anchor_expanded * negatives, dim=2)  # [B, N]
        
        # Find hardest (smallest distance) negatives
        hard_indices = torch.argmin(distances, dim=1)  # [B]
        
        # Select hard negatives
        batch_indices = torch.arange(B, device=anchor.device)
        hard_negatives = negatives[batch_indices, hard_indices]  # [B, D]
        
        return hard_negatives


def create_cosplace_model(config: Dict) -> CosPlaceModel:
    """Create CosPlace model from configuration"""
    return CosPlaceModel(
        backbone=config.get('backbone', 'resnet50'),
        output_dim=config.get('output_dim', 2048),
        pooling_scales=config.get('pooling_scales', [1, 2, 4]),
        pretrained=config.get('pretrained', True)
    )


def get_cosplace_transforms(image_size: int = 224, augment: bool = True):
    """Get image transforms for CosPlace"""
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


if __name__ == "__main__":
    # Test CosPlace model
    model = CosPlaceModel()
    model.eval()
    
    # Test input
    x = torch.randn(4, 3, 224, 224)
    
    with torch.no_grad():
        descriptors = model(x)
        
    print("CosPlace Model Test:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {descriptors.shape}")
    print(f"Descriptor norm: {torch.norm(descriptors, dim=1)}")
    print(f"Cosine similarity between first two samples: {torch.sum(descriptors[0] * descriptors[1])}")
    
    # Test loss
    criterion = CosPlaceLoss()
    anchor = descriptors[0:2]
    positive = descriptors[1:3]
    negative = descriptors[2:4]
    
    triplet_loss = criterion(anchor, positive, negative, 'triplet')
    contrastive_loss = criterion(anchor, positive, negative, 'contrastive')
    
    print(f"Triplet loss: {triplet_loss.item():.4f}")
    print(f"Contrastive loss: {contrastive_loss.item():.4f}")
