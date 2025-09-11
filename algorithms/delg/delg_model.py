"""
DELG (Deep Local and Global features) Implementation for VPR
Based on "Unifying Deep Local and Global Features for Image Search" (ECCV 2020)

Combines:
- Global features: CNN-based global descriptor
- Local features: Keypoint detection + local feature extraction
- Attention mechanism: For feature selection and aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
import numpy as np


class AttentionModule(nn.Module):
    """Attention module for keypoint selection"""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention_conv = nn.Conv2d(input_dim, 1, kernel_size=1)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W] feature maps
        Returns:
            attention_map: [B, 1, H, W] attention scores
        """
        attention = self.attention_conv(features)
        attention = torch.sigmoid(attention)
        return attention


class LocalFeatureExtractor(nn.Module):
    """Local feature extraction with keypoint detection"""
    
    def __init__(self, backbone_dim: int, local_dim: int = 128):
        super().__init__()
        self.local_dim = local_dim
        
        # Local feature projection
        self.local_proj = nn.Sequential(
            nn.Conv2d(backbone_dim, local_dim, kernel_size=1),
            nn.BatchNorm2d(local_dim),
            nn.ReLU(inplace=True)
        )
        
        # Attention for keypoint detection
        self.attention = AttentionModule(backbone_dim)
        
    def forward(self, features: torch.Tensor, num_keypoints: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [B, C, H, W] backbone features
            num_keypoints: Maximum number of keypoints to extract
        Returns:
            local_descriptors: [B, num_keypoints, local_dim] local descriptors
            keypoint_scores: [B, num_keypoints] keypoint confidence scores
        """
        B, C, H, W = features.shape
        
        # Generate attention map for keypoint detection
        attention_map = self.attention(features)  # [B, 1, H, W]
        
        # Extract local features
        local_features = self.local_proj(features)  # [B, local_dim, H, W]
        
        # Flatten spatial dimensions
        attention_flat = attention_map.view(B, -1)  # [B, H*W]
        local_features_flat = local_features.view(B, self.local_dim, -1)  # [B, local_dim, H*W]
        
        # Select top keypoints based on attention scores
        num_keypoints = min(num_keypoints, H * W)
        top_indices = torch.topk(attention_flat, num_keypoints, dim=1).indices  # [B, num_keypoints]
        
        # Extract descriptors and scores for selected keypoints
        batch_indices = torch.arange(B).unsqueeze(1).expand(-1, num_keypoints).to(features.device)
        
        # Gather local descriptors
        local_descriptors = local_features_flat[batch_indices, :, top_indices]  # [B, local_dim, num_keypoints]
        local_descriptors = local_descriptors.transpose(1, 2)  # [B, num_keypoints, local_dim]
        
        # Gather keypoint scores
        keypoint_scores = attention_flat[batch_indices, top_indices]  # [B, num_keypoints]
        
        # L2 normalize descriptors
        local_descriptors = F.normalize(local_descriptors, p=2, dim=2)
        
        return local_descriptors, keypoint_scores


class GlobalFeatureExtractor(nn.Module):
    """Global feature extraction with GeM pooling and attention"""
    
    def __init__(self, backbone_dim: int, global_dim: int = 2048):
        super().__init__()
        self.global_dim = global_dim
        
        # Global feature projection
        self.global_proj = nn.Sequential(
            nn.Conv2d(backbone_dim, global_dim, kernel_size=1),
            nn.BatchNorm2d(global_dim),
            nn.ReLU(inplace=True)
        )
        
        # GeM pooling parameter
        self.gem_p = nn.Parameter(torch.ones(1) * 3.0)
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(global_dim, global_dim),
            nn.BatchNorm1d(global_dim),
            nn.ReLU(inplace=True),
            nn.Linear(global_dim, global_dim)
        )
        
    def gem_pooling(self, features: torch.Tensor) -> torch.Tensor:
        """Generalized Mean (GeM) pooling"""
        # Clamp to avoid numerical issues
        features = torch.clamp(features, min=1e-6)
        
        # GeM pooling: (1/N * sum(x^p))^(1/p)
        pooled = F.avg_pool2d(features.pow(self.gem_p), 
                             kernel_size=features.size()[2:])
        pooled = pooled.pow(1.0 / self.gem_p)
        
        return pooled.view(features.size(0), -1)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W] backbone features
        Returns:
            global_descriptor: [B, global_dim] global descriptor
        """
        # Project features
        global_features = self.global_proj(features)
        
        # GeM pooling
        pooled = self.gem_pooling(global_features)
        
        # Final projection
        global_descriptor = self.final_proj(pooled)
        
        # L2 normalize
        global_descriptor = F.normalize(global_descriptor, p=2, dim=1)
        
        return global_descriptor


class DELGModel(nn.Module):
    """
    DELG: Deep Local and Global features for image search
    
    Combines global and local features for robust place recognition
    """
    
    def __init__(self, 
                 backbone: str = 'resnet50',
                 global_dim: int = 2048,
                 local_dim: int = 128,
                 num_keypoints: int = 1000,
                 pretrained: bool = True):
        super().__init__()
        
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.num_keypoints = num_keypoints
        
        # Backbone network
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.backbone_dim = 2048
        elif backbone == 'resnet101':
            resnet = models.resnet101(pretrained=pretrained)
            self.backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove final layers to get feature maps
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Global feature extractor
        self.global_extractor = GlobalFeatureExtractor(self.backbone_dim, global_dim)
        
        # Local feature extractor
        self.local_extractor = LocalFeatureExtractor(self.backbone_dim, local_dim)
        
    def forward(self, x: torch.Tensor, extract_local: bool = True) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] input images
            extract_local: Whether to extract local features (can be disabled for efficiency)
        Returns:
            Dictionary containing:
                - global_descriptor: [B, global_dim] global features
                - local_descriptors: [B, num_keypoints, local_dim] local features (if extract_local=True)
                - keypoint_scores: [B, num_keypoints] keypoint confidence scores (if extract_local=True)
        """
        # Extract backbone features
        features = self.backbone(x)  # [B, backbone_dim, H, W]
        
        # Extract global descriptor
        global_descriptor = self.global_extractor(features)
        
        result = {'global_descriptor': global_descriptor}
        
        # Extract local features if requested
        if extract_local:
            local_descriptors, keypoint_scores = self.local_extractor(features, self.num_keypoints)
            result.update({
                'local_descriptors': local_descriptors,
                'keypoint_scores': keypoint_scores
            })
        
        return result
    
    def extract_global_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract only global features for efficient retrieval"""
        result = self.forward(x, extract_local=False)
        return result['global_descriptor']
    
    def extract_local_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract only local features"""
        features = self.backbone(x)
        return self.local_extractor(features, self.num_keypoints)


class DELGLoss(nn.Module):
    """Combined loss for DELG training"""
    
    def __init__(self, margin: float = 0.1, alpha: float = 1.0, beta: float = 0.5):
        super().__init__()
        self.margin = margin
        self.alpha = alpha  # Weight for global loss
        self.beta = beta    # Weight for local loss
        
    def global_triplet_loss(self, global_anchor: torch.Tensor, 
                           global_positive: torch.Tensor, 
                           global_negative: torch.Tensor) -> torch.Tensor:
        """Triplet loss for global descriptors"""
        pos_dist = F.pairwise_distance(global_anchor, global_positive, p=2)
        neg_dist = F.pairwise_distance(global_anchor, global_negative, p=2)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
    
    def local_attention_loss(self, keypoint_scores: torch.Tensor) -> torch.Tensor:
        """Regularization loss for attention (encourage sparsity)"""
        # Encourage attention to be sparse and confident
        attention_reg = -torch.mean(keypoint_scores * torch.log(keypoint_scores + 1e-8))
        return attention_reg
    
    def forward(self, anchor_output: Dict[str, torch.Tensor],
                positive_output: Dict[str, torch.Tensor],
                negative_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Combined DELG loss
        
        Args:
            anchor_output: DELG output for anchor images
            positive_output: DELG output for positive images  
            negative_output: DELG output for negative images
        Returns:
            Dictionary with individual and total losses
        """
        # Global triplet loss
        global_loss = self.global_triplet_loss(
            anchor_output['global_descriptor'],
            positive_output['global_descriptor'],
            negative_output['global_descriptor']
        )
        
        # Local attention regularization
        local_loss = 0.0
        if 'keypoint_scores' in anchor_output:
            local_loss = (
                self.local_attention_loss(anchor_output['keypoint_scores']) +
                self.local_attention_loss(positive_output['keypoint_scores']) +
                self.local_attention_loss(negative_output['keypoint_scores'])
            ) / 3.0
        
        # Total loss
        total_loss = self.alpha * global_loss + self.beta * local_loss
        
        return {
            'total_loss': total_loss,
            'global_loss': global_loss,
            'local_loss': local_loss
        }


def create_delg_model(config: Dict) -> DELGModel:
    """Create DELG model from configuration"""
    return DELGModel(
        backbone=config.get('backbone', 'resnet50'),
        global_dim=config.get('global_dim', 2048),
        local_dim=config.get('local_dim', 128),
        num_keypoints=config.get('num_keypoints', 1000),
        pretrained=config.get('pretrained', True)
    )


# Transform for DELG
def get_delg_transforms(image_size: int = 224):
    """Get image transforms for DELG"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


if __name__ == "__main__":
    # Test DELG model
    model = DELGModel()
    model.eval()
    
    # Test input
    x = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        output = model(x)
        
    print("DELG Model Test:")
    print(f"Global descriptor shape: {output['global_descriptor'].shape}")
    print(f"Local descriptors shape: {output['local_descriptors'].shape}")
    print(f"Keypoint scores shape: {output['keypoint_scores'].shape}")
    print(f"Global descriptor norm: {torch.norm(output['global_descriptor'], dim=1)}")
    print(f"Local descriptors norm: {torch.norm(output['local_descriptors'], dim=2).mean()}")
