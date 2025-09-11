"""
EigenPlaces Implementation for VPR
Based on "EigenPlaces: Training Viewpoint Robust Models for Visual Place Recognition"

EigenPlaces uses:
- ResNet backbone for feature extraction
- Eigenvalue decomposition for viewpoint robustness
- Multi-scale feature aggregation
- Contrastive learning with hard negative mining
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
import math


class EigenLayer(nn.Module):
    """Eigenvalue decomposition layer for viewpoint robustness"""
    
    def __init__(self, input_dim: int, eigenvalue_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.eigenvalue_dim = eigenvalue_dim
        
        # Projection to eigenvalue space
        self.eigen_proj = nn.Linear(input_dim, eigenvalue_dim * eigenvalue_dim)
        
        # Regularization parameter
        self.register_parameter('reg_param', nn.Parameter(torch.tensor(0.01)))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply eigenvalue decomposition for viewpoint robustness
        
        Args:
            x: [B, input_dim] input features
        Returns:
            eigen_features: [B, eigenvalue_dim] eigenvalue-based features
        """
        B = x.size(0)
        
        # Project to matrix form
        matrix_features = self.eigen_proj(x)  # [B, eigenvalue_dim^2]
        matrix_features = matrix_features.view(B, self.eigenvalue_dim, self.eigenvalue_dim)
        
        # Make symmetric for stable eigenvalue decomposition
        matrix_features = (matrix_features + matrix_features.transpose(1, 2)) / 2
        
        # Add regularization for numerical stability
        identity = torch.eye(self.eigenvalue_dim, device=x.device).unsqueeze(0).expand(B, -1, -1)
        matrix_features = matrix_features + self.reg_param * identity
        
        # Eigenvalue decomposition
        eigenvals, eigenvecs = torch.linalg.eigh(matrix_features)
        
        # Use eigenvalues as features (sorted in descending order)
        eigenvals = torch.flip(eigenvals, dims=[1])  # Descending order
        
        # Apply log normalization to eigenvalues
        eigenvals = torch.log(torch.clamp(eigenvals, min=1e-8))
        
        return eigenvals


class MultiScaleEigenAggregation(nn.Module):
    """Multi-scale feature aggregation with eigenvalue decomposition"""
    
    def __init__(self, input_dim: int, output_dim: int, scales: List[int] = [1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(scale) for scale in scales
        ])
        
        # Eigenvalue layers for each scale
        eigen_dim = 32  # Eigenvalue dimension for each scale
        self.eigen_layers = nn.ModuleList([
            EigenLayer(input_dim * scale * scale, eigen_dim) for scale in scales
        ])
        
        # Combine features from all scales
        total_eigen_dim = eigen_dim * len(scales)
        self.final_proj = nn.Sequential(
            nn.Linear(total_eigen_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] feature maps
        Returns:
            aggregated: [B, output_dim] multi-scale eigen features
        """
        eigen_features = []
        
        for i, (pool, eigen_layer) in enumerate(zip(self.pools, self.eigen_layers)):
            # Pool to specific scale
            pooled = pool(x)  # [B, C, scale, scale]
            pooled_flat = pooled.view(x.size(0), -1)  # [B, C * scale * scale]
            
            # Apply eigenvalue decomposition
            eigen_feat = eigen_layer(pooled_flat)  # [B, eigen_dim]
            eigen_features.append(eigen_feat)
        
        # Concatenate all eigen features
        concatenated = torch.cat(eigen_features, dim=1)
        
        # Final projection
        output = self.final_proj(concatenated)
        
        return output


class EigenPlacesModel(nn.Module):
    """
    EigenPlaces: Training Viewpoint Robust Models for Visual Place Recognition
    
    Key components:
    - ResNet backbone for feature extraction
    - Multi-scale eigenvalue decomposition for viewpoint robustness
    - Contrastive learning framework
    """
    
    def __init__(self, 
                 backbone: str = 'resnet50',
                 output_dim: int = 2048,
                 eigen_scales: List[int] = [1, 2, 4],
                 pretrained: bool = True):
        super().__init__()
        
        self.output_dim = output_dim
        
        # Backbone network
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
        
        # Multi-scale eigen aggregation
        self.eigen_aggregation = MultiScaleEigenAggregation(
            backbone_dim, output_dim, eigen_scales
        )
        
        # Optional: Additional feature processing
        self.feature_norm = nn.BatchNorm1d(output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] input images
        Returns:
            descriptors: [B, output_dim] global descriptors with viewpoint robustness
        """
        # Extract backbone features
        features = self.backbone(x)  # [B, backbone_dim, H', W']
        
        # Multi-scale eigen aggregation
        eigen_features = self.eigen_aggregation(features)
        
        # Normalize features
        eigen_features = self.feature_norm(eigen_features)
        
        # L2 normalize for cosine similarity
        descriptors = F.normalize(eigen_features, p=2, dim=1)
        
        return descriptors
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward method"""
        return self.forward(x)


class EigenPlacesLoss(nn.Module):
    """
    EigenPlaces loss with viewpoint-aware contrastive learning
    """
    
    def __init__(self, margin: float = 0.1, temperature: float = 0.07, 
                 viewpoint_weight: float = 0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.viewpoint_weight = viewpoint_weight
        
    def viewpoint_contrastive_loss(self, anchor: torch.Tensor, 
                                  positive: torch.Tensor, 
                                  negative: torch.Tensor) -> torch.Tensor:
        """Viewpoint-aware contrastive loss"""
        # Compute similarities
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature
        neg_sim = torch.sum(anchor * negative, dim=1) / self.temperature
        
        # Contrastive loss
        logits = torch.stack([pos_sim, neg_sim], dim=1)
        targets = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
        
        base_loss = F.cross_entropy(logits, targets)
        
        # Viewpoint regularization: encourage eigenvalue diversity
        eigen_reg = self.eigenvalue_regularization(anchor, positive, negative)
        
        total_loss = base_loss + self.viewpoint_weight * eigen_reg
        return total_loss
    
    def eigenvalue_regularization(self, anchor: torch.Tensor, 
                                 positive: torch.Tensor, 
                                 negative: torch.Tensor) -> torch.Tensor:
        """Regularization to encourage eigenvalue diversity for viewpoint robustness"""
        # Compute variance of features to encourage diversity
        all_features = torch.cat([anchor, positive, negative], dim=0)
        feature_var = torch.var(all_features, dim=0).mean()
        
        # Encourage higher variance (diversity)
        reg_loss = 1.0 / (feature_var + 1e-8)
        
        return reg_loss
    
    def triplet_loss(self, anchor: torch.Tensor, positive: torch.Tensor, 
                    negative: torch.Tensor) -> torch.Tensor:
        """Standard triplet loss with cosine distance"""
        pos_dist = 1.0 - torch.sum(anchor * positive, dim=1)
        neg_dist = 1.0 - torch.sum(anchor * negative, dim=1)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: torch.Tensor, loss_type: str = 'viewpoint_contrastive') -> torch.Tensor:
        """
        EigenPlaces loss computation
        
        Args:
            anchor: [B, D] anchor descriptors
            positive: [B, D] positive descriptors
            negative: [B, D] negative descriptors
            loss_type: 'triplet' or 'viewpoint_contrastive'
        Returns:
            loss: Scalar loss value
        """
        if loss_type == 'triplet':
            return self.triplet_loss(anchor, positive, negative)
        elif loss_type == 'viewpoint_contrastive':
            return self.viewpoint_contrastive_loss(anchor, positive, negative)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")


def create_eigenplaces_model(config: Dict) -> EigenPlacesModel:
    """Create EigenPlaces model from configuration"""
    return EigenPlacesModel(
        backbone=config.get('backbone', 'resnet50'),
        output_dim=config.get('output_dim', 2048),
        eigen_scales=config.get('eigen_scales', [1, 2, 4]),
        pretrained=config.get('pretrained', True)
    )


def get_eigenplaces_transforms(image_size: int = 224, augment: bool = True):
    """Get image transforms for EigenPlaces with viewpoint augmentations"""
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),  # Viewpoint variation
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Perspective changes
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
    # Test EigenPlaces model
    model = EigenPlacesModel()
    model.eval()
    
    # Test input
    x = torch.randn(4, 3, 224, 224)
    
    with torch.no_grad():
        descriptors = model(x)
        
    print("EigenPlaces Model Test:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {descriptors.shape}")
    print(f"Descriptor norm: {torch.norm(descriptors, dim=1)}")
    print(f"Feature diversity (std): {torch.std(descriptors).item():.4f}")
    
    # Test loss
    criterion = EigenPlacesLoss()
    anchor = descriptors[0:2]
    positive = descriptors[1:3]
    negative = descriptors[2:4]
    
    triplet_loss = criterion(anchor, positive, negative, 'triplet')
    viewpoint_loss = criterion(anchor, positive, negative, 'viewpoint_contrastive')
    
    print(f"Triplet loss: {triplet_loss.item():.4f}")
    print(f"Viewpoint contrastive loss: {viewpoint_loss.item():.4f}")
