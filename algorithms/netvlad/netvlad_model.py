"""
NetVLAD implementation for Visual Place Recognition
Based on the PyTorch implementation from https://github.com/Nanne/pytorch-NetVlad
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Tuple


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(
        self,
        num_clusters: int = 64,
        dim: int = 512,
        normalize_input: bool = True,
        vladv2: bool = False
    ):
        """
        Args:
            num_clusters: The number of clusters
            dim: Dimension of descriptors
            normalize_input: If true, descriptor-wise L2 normalization is applied to input
            vladv2: If true, use vladv2 otherwise use vladv1
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def init_params(self, clsts: np.ndarray, traindescs: np.ndarray):
        """Initialize parameters from clusters"""
        # Ensure new parameters are allocated on the same device/dtype as existing module
        current_weight = self.conv.weight
        device = current_weight.device if current_weight is not None else torch.device('cpu')
        dtype = current_weight.dtype if current_weight is not None else torch.float32

        if self.vladv2 == False:
            clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            dots = np.dot(clstsAssign, traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :]  # sort, descending

            self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts).to(device=device, dtype=dtype))
            self.conv.weight = nn.Parameter(
                torch.from_numpy(self.alpha * clstsAssign).to(device=device, dtype=dtype).unsqueeze(2).unsqueeze(3)
            )
            self.conv.bias = None
        else:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(traindescs)
            del traindescs
            dsSq = np.square(knn.kneighbors(clsts, 2)[1])
            del knn
            self.alpha = (-np.log(0.01) / np.mean(dsSq[:, 1] - dsSq[:, 0])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts).to(device=device, dtype=dtype))
            del clsts, dsSq

            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(
                -self.alpha * self.centroids.norm(dim=1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for c in range(self.num_clusters):  # slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                       self.centroids[c:c + 1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:, c:c + 1, :].unsqueeze(2)
            vlad[:, c:c + 1, :] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad


class NetVLADModel(nn.Module):
    """Complete NetVLAD model with ResNet50 backbone"""

    def __init__(
        self,
        num_clusters: int = 64,
        pretrained: bool = True,
        vladv2: bool = False
    ):
        super(NetVLADModel, self).__init__()
        
        # ResNet50 backbone
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layers (avgpool, fc)
        layers = list(resnet.children())[:-2]
        
        # If using pretrained, freeze early layers and only train later ones
        if pretrained:
            # Freeze layers except the last residual block
            for i, layer in enumerate(layers[:-1]):
                for param in layer.parameters():
                    param.requires_grad = False
        
        self.encoder = nn.Sequential(*layers)
        
        # NetVLAD pooling layer
        encoder_dim = 2048  # ResNet50 output dimension
        self.pool = NetVLAD(num_clusters=num_clusters, dim=encoder_dim, vladv2=vladv2)
        
        self.encoder_dim = encoder_dim
        self.num_clusters = num_clusters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure NetVLAD layer is on the same device as input
        if self.pool.conv.weight.device != x.device:
            self.pool.to(x.device)

        # Extract features using ResNet50
        x = self.encoder(x)
        
        # Apply NetVLAD pooling
        x = self.pool(x)
        
        return x

    def init_vlad_params(self, cluster_centers: np.ndarray, train_descriptors: np.ndarray):
        """Initialize NetVLAD parameters from clustering"""
        self.pool.init_params(cluster_centers, train_descriptors)


def create_netvlad_model(
    num_clusters: int = 64,
    pretrained: bool = True,
    vladv2: bool = False
) -> NetVLADModel:
    """Create NetVLAD model with ResNet50 backbone"""
    return NetVLADModel(
        num_clusters=num_clusters,
        pretrained=pretrained,
        vladv2=vladv2
    )


def extract_features_for_clustering(
    model: NetVLADModel,
    dataloader: torch.utils.data.DataLoader,
    num_descriptors: int = 50000,
    device: torch.device = torch.device('cpu')
) -> np.ndarray:
    """
    Extract features from the encoder (before NetVLAD) for clustering.
    """
    model.eval()
    model.to(device)
    
    descriptors = []
    total_collected = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                images = batch['image']
            else:
                images = batch[0]
            
            images = images.to(device)
            
            # Get features from encoder (before NetVLAD pooling)
            features = model.encoder(images)  # Shape: (batch, 2048, H, W)
            
            # Flatten spatial dimensions and permute
            batch_size, C, H, W = features.shape
            features = features.view(batch_size, C, -1).permute(0, 2, 1)  # (batch, H*W, C)
            
            # Sample random descriptors from each image
            for i in range(batch_size):
                if total_collected >= num_descriptors:
                    break
                
                img_descriptors = features[i]  # (H*W, C)
                num_spatial_locs = img_descriptors.shape[0]
                
                # Sample random locations
                sample_size = min(100, num_spatial_locs, num_descriptors - total_collected)
                sample_indices = np.random.choice(num_spatial_locs, sample_size, replace=False)
                
                sampled_descriptors = img_descriptors[sample_indices].cpu().numpy()
                descriptors.append(sampled_descriptors)
                total_collected += sample_size
            
            if total_collected >= num_descriptors:
                break
    
    # Concatenate all descriptors
    descriptors = np.vstack(descriptors)[:num_descriptors]
    
    return descriptors


def compute_cluster_centers(
    descriptors: np.ndarray,
    num_clusters: int = 64,
    max_iter: int = 100
) -> np.ndarray:
    """
    Compute cluster centers using K-means.
    """
    try:
        import faiss
        
        # Use FAISS for efficient clustering
        print(f"Running K-means clustering with {num_clusters} clusters...")
        d = descriptors.shape[1]
        kmeans = faiss.Kmeans(d, num_clusters, niter=max_iter, verbose=True)
        kmeans.train(descriptors.astype(np.float32))
        
        return kmeans.centroids
        
    except ImportError:
        # Fallback to sklearn
        from sklearn.cluster import KMeans
        
        print(f"Running K-means clustering with {num_clusters} clusters using sklearn...")
        kmeans = KMeans(n_clusters=num_clusters, max_iter=max_iter, random_state=42)
        kmeans.fit(descriptors)
        
        return kmeans.cluster_centers_


class TripletLoss(nn.Module):
    """Triplet loss for NetVLAD training"""
    
    def __init__(self, margin: float = 0.1):
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


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_netvlad_model(num_clusters=64, pretrained=True)
    model.to(device)
    
    # Test forward pass
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {64 * 2048})")
    
    # Test NetVLAD layer separately
    netvlad = NetVLAD(num_clusters=64, dim=2048)
    test_features = torch.randn(batch_size, 2048, 7, 7)
    
    with torch.no_grad():
        vlad_output = netvlad(test_features)
    
    print(f"NetVLAD input shape: {test_features.shape}")
    print(f"NetVLAD output shape: {vlad_output.shape}")
