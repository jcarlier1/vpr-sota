"""
AP-GeM (Aggregated Deep Local Descriptors) implementation for Visual Place Recognition
Based on the PyTorch implementation from https://github.com/filipradenovic/cnnimageretrieval-pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from typing import Optional


class GeM(nn.Module):
    """Generalized Mean Pooling (GeM) layer"""
    
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GeM pooling
        Args:
            x: Feature tensor of shape (N, C, H, W)
        Returns:
            Pooled features of shape (N, C, 1, 1)
        """
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})'


class L2Norm(nn.Module):
    """L2 normalization layer"""
    
    def __init__(self, dim: int = 1):
        super(L2Norm, self).__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=self.dim)


class PowerLaw(nn.Module):
    """Power-law normalization layer"""
    
    def __init__(self, eps: float = 1e-6):
        super(PowerLaw, self).__init__()
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(torch.sign(x) * torch.abs(x).pow(0.5), p=2, dim=1)


class APGeMModel(nn.Module):
    """
    AP-GeM model with ResNet101 backbone
    Implements Aggregated Deep Local Descriptors with GeM pooling
    """
    
    def __init__(
        self,
        architecture: str = 'resnet101',
        pretrained: bool = True,
        gem_p: float = 3.0,
        whitening: bool = False,
        whitening_dim: Optional[int] = None,
        regional_pooling: bool = False,
        local_whitening: bool = False
    ):
        super(APGeMModel, self).__init__()
        
        self.architecture = architecture
        self.pretrained = pretrained
        self.whitening = whitening
        self.regional_pooling = regional_pooling
        self.local_whitening = local_whitening
        
        # Get output dimension based on architecture
        self.feature_dim = self._get_feature_dim(architecture)
        
        # Create backbone
        self.features = self._create_features(architecture, pretrained)
        
        # Local whitening before pooling (if enabled)
        if local_whitening:
            self.lwhiten = nn.Linear(self.feature_dim, self.feature_dim, bias=True)
            self.lwhiten.weight.data.copy_(torch.eye(self.feature_dim))
            self.lwhiten.bias.data.zero_()
        
        # Global pooling layer
        self.pool = GeM(p=gem_p)
        
        # Normalization after pooling
        self.norm = L2Norm()
        
        # Whitening layer (if enabled)
        if whitening:
            if whitening_dim is None:
                whitening_dim = self.feature_dim
            self.whiten = nn.Linear(self.feature_dim, whitening_dim, bias=True)
            self.whiten.weight.data.copy_(torch.eye(min(self.feature_dim, whitening_dim)))
            self.whiten.bias.data.zero_()
            self.output_dim = whitening_dim
        else:
            self.output_dim = self.feature_dim
    
    def _get_feature_dim(self, architecture: str) -> int:
        """Get output feature dimension for different architectures"""
        dims = {
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048,
            'resnet101': 2048,
            'resnet152': 2048,
            'vgg16': 512,
            'vgg19': 512,
        }
        return dims.get(architecture, 2048)
    
    def _create_features(self, architecture: str, pretrained: bool) -> nn.Module:
        """Create feature extraction backbone"""
        
        if architecture.startswith('resnet'):
            # Load ResNet model
            model = getattr(models, architecture)(pretrained=pretrained)
            # Remove final classification layers (avgpool, fc)
            features = list(model.children())[:-2]
            
            if pretrained:
                # Freeze early layers if using pretrained model
                for i, layer in enumerate(features[:-2]):  # Keep last 2 layers trainable
                    for param in layer.parameters():
                        param.requires_grad = False
        
        elif architecture.startswith('vgg'):
            # Load VGG model
            model = getattr(models, architecture)(pretrained=pretrained)
            # Take feature layers but remove final ReLU
            features = list(model.features.children())[:-1]
            
            if pretrained:
                # Freeze early layers if using pretrained model
                for layer in features[:-4]:  # Keep last few layers trainable
                    for param in layer.parameters():
                        param.requires_grad = False
        
        else:
            raise ValueError(f'Unsupported architecture: {architecture}')
        
        return nn.Sequential(*features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: Input images of shape (N, 3, H, W)
        Returns:
            Feature vectors of shape (N, output_dim)
        """
        # Extract convolutional features
        x = self.features(x)
        
        # Local whitening (if enabled)
        if self.local_whitening:
            # Reshape to apply linear layer across spatial dimensions
            n, c, h, w = x.size()
            x = x.view(n, c, -1).permute(0, 2, 1)  # (N, H*W, C)
            x = self.lwhiten(x)
            x = x.permute(0, 2, 1).view(n, c, h, w)  # Back to (N, C, H, W)
        
        # Global pooling
        x = self.pool(x)  # (N, C, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (N, C)
        
        # L2 normalization
        x = self.norm(x)
        
        # Whitening (if enabled)
        if self.whitening:
            x = self.whiten(x)
            x = self.norm(x)  # L2 normalize again after whitening
        
        return x


class TripletLoss(nn.Module):
    """Triplet margin loss for AP-GeM training"""
    
    def __init__(self, margin: float = 0.85):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        return self.triplet_loss(anchor, positive, negative)


class ContrastiveLoss(nn.Module):
    """Contrastive loss for AP-GeM training"""
    
    def __init__(self, margin: float = 0.85):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature vectors of shape (N, D) where N is batch size
            labels: Binary labels of shape (N,) where 1 indicates positive pair
        """
        # Compute pairwise distances
        dists = torch.pdist(x, p=2)
        
        # Convert labels to pairwise format
        n = x.size(0)
        labels_expanded = labels.unsqueeze(1) == labels.unsqueeze(0)
        labels_pairs = labels_expanded[torch.triu(torch.ones(n, n), diagonal=1) == 1]
        
        # Compute contrastive loss
        pos_loss = labels_pairs.float() * dists.pow(2)
        neg_loss = (1 - labels_pairs.float()) * F.relu(self.margin - dists).pow(2)
        
        loss = 0.5 * (pos_loss + neg_loss).mean()
        return loss


def create_apgem_model(
    architecture: str = 'resnet101',
    pretrained: bool = True,
    gem_p: float = 3.0,
    whitening: bool = False,
    whitening_dim: Optional[int] = None,
    regional_pooling: bool = False,
    local_whitening: bool = False
) -> APGeMModel:
    """
    Create AP-GeM model
    
    Args:
        architecture: Backbone architecture ('resnet101', 'resnet50', etc.)
        pretrained: Whether to use pretrained weights
        gem_p: GeM pooling parameter
        whitening: Whether to add whitening layer
        whitening_dim: Dimension of whitening layer
        regional_pooling: Whether to use regional pooling
        local_whitening: Whether to use local whitening before pooling
    
    Returns:
        AP-GeM model
    """
    return APGeMModel(
        architecture=architecture,
        pretrained=pretrained,
        gem_p=gem_p,
        whitening=whitening,
        whitening_dim=whitening_dim,
        regional_pooling=regional_pooling,
        local_whitening=local_whitening
    )


def extract_features(
    model: APGeMModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device = torch.device('cpu')
) -> np.ndarray:
    """
    Extract features from a dataset using the AP-GeM model
    
    Args:
        model: Trained AP-GeM model
        dataloader: DataLoader for the dataset
        device: Device to run inference on
    
    Returns:
        Feature matrix of shape (N, feature_dim)
    """
    model.eval()
    model.to(device)
    
    features = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                images = batch['image']
            else:
                images = batch[0]
            
            images = images.to(device)
            batch_features = model(images)
            features.append(batch_features.cpu().numpy())
    
    return np.vstack(features)


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create AP-GeM model with ResNet101
    model = create_apgem_model(
        architecture='resnet101',
        pretrained=True,
        gem_p=3.0,
        whitening=True,
        whitening_dim=2048
    )
    
    model.to(device)
    model.eval()
    
    # Test with random input
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output dim: {model.output_dim}")
    
    # Test GeM pooling separately
    gem_layer = GeM(p=3.0)
    test_features = torch.randn(batch_size, 2048, 7, 7)
    
    with torch.no_grad():
        pooled = gem_layer(test_features)
    
    print(f"GeM input shape: {test_features.shape}")
    print(f"GeM output shape: {pooled.shape}")
    
    # Test loss functions
    anchor = torch.randn(4, 2048)
    positive = torch.randn(4, 2048)
    negative = torch.randn(4, 2048)
    
    triplet_loss = TripletLoss(margin=0.85)
    loss = triplet_loss(anchor, positive, negative)
    print(f"Triplet loss: {loss.item():.4f}")
