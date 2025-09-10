# VPR SOTA Algorithms Implementation

This project implements multiple State-of-the-Art (SOTA) Visual Place Recognition (VPR) algorithms for evaluation on GPS-tagged datasets.

## Implemented Algorithms

### ✅ Ready for Testing
- **NetVLAD** (ResNet50 backbone) - Aggregated local descriptors with learnable clustering
- **AP-GeM** (ResNet101 backbone) - Attention-based pooling with GeM pooling

### 🚧 In Development
- **DELG** (ResNet50 backbone) - Global + local descriptors for place recognition
- **CosPlace** (ResNet backbone) - Cosine-based place recognition
- **EigenPlaces** (ResNet backbone) - Eigenvalue-based place recognition

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate vpr-sota
```

### 2. Dataset Preparation

Prepare your GPS-tagged dataset with the following structure:

```
data/
├── train_dataset.csv       # Training images with GPS coordinates
├── test_dataset.csv        # Test images with GPS coordinates
├── train_images/           # Training image directory
└── test_images/            # Test image directory
```

**CSV Format:**
```csv
image_path,latitude,longitude,camera_direction
path/to/image1.jpg,40.7128,-74.0060,front
path/to/image2.jpg,40.7580,-73.9855,rear
```

### 3. Run Experiments

#### Option 1: Run All Available Algorithms (Recommended)
```bash
# Simple one-command execution
./run_vpr_experiments.sh
```

#### Option 2: Run Specific Algorithms
```bash
# Activate environment
conda activate vpr-sota

# Run specific algorithms
python experiments/run_experiments.py \
    --config configs/base_experiment_config.yaml \
    --output-dir experiments/results \
    --algorithms netvlad ap-gem
```

#### Option 3: Run Individual Algorithms
```bash
# NetVLAD
python algorithms/netvlad/train_netvlad.py --config configs/netvlad_config.yaml

# AP-GeM
python algorithms/ap-gem/train_apgem.py --config configs/apgem_config.yaml
```

## Configuration

### Base Configuration
Edit `configs/base_experiment_config.yaml` to adjust:
- Dataset paths
- GPS thresholds (positive/negative matching distances)
- Training parameters (batch size, learning rate, epochs)
- Algorithm-specific settings

### Algorithm-Specific Configurations
Individual algorithm configs in `configs/`:
- `netvlad_config.yaml` - NetVLAD clustering and pooling settings
- `apgem_config.yaml` - AP-GeM attention and pooling parameters

## Project Structure

```
vpr-sota/
├── algorithms/              # Algorithm implementations
│   ├── netvlad/            # NetVLAD implementation
│   ├── ap-gem/             # AP-GeM implementation
│   ├── delg/               # DELG implementation (coming soon)
│   ├── cosplace/           # CosPlace implementation (coming soon)
│   └── eigenplaces/        # EigenPlaces implementation (coming soon)
├── datasets/               # Dataset loaders
│   └── gps_dataset.py      # GPS-based dataset with automatic pair generation
├── utils/                  # Utilities
│   └── evaluation.py       # VPR evaluation metrics (Recall@K, mAP)
├── configs/                # Configuration files
├── experiments/            # Experiment runner and results
├── environment.yml         # Conda environment specification
└── run_vpr_experiments.sh  # One-click experiment runner
```

## Evaluation Metrics

The framework evaluates all algorithms using standard VPR metrics:

- **Recall@K**: Percentage of queries with correct match in top-K retrievals
- **Precision@K**: Precision of top-K retrievals
- **mAP**: Mean Average Precision across all queries
- **GPS-based Ground Truth**: Uses haversine distance for positive/negative matching

## Results

After running experiments, results are saved to `experiments/results/`:

```
experiments/results/
├── comparison_report.txt    # Algorithm comparison summary
├── experiment_results.json # Detailed results in JSON format
├── logs/                   # Detailed execution logs
├── netvlad/                # NetVLAD model and results
├── ap-gem/                 # AP-GeM model and results
└── configs/                # Generated algorithm-specific configs
```

## GPU Requirements

- **Recommended**: NVIDIA GPU with 8GB+ VRAM
- **Minimum**: NVIDIA GPU with 4GB+ VRAM (reduce batch sizes)
- **CPU**: Supported but significantly slower

## Algorithm Details

### NetVLAD
- **Backbone**: ResNet50 (pretrained on ImageNet)
- **Pooling**: NetVLAD with learnable cluster centers
- **Features**: 32,768-dimensional global descriptors
- **Loss**: Triplet loss with hard negative mining

### AP-GeM
- **Backbone**: ResNet101 (pretrained on ImageNet)
- **Pooling**: Attention-weighted GeM pooling
- **Features**: 2048-dimensional global descriptors
- **Loss**: Contrastive loss with triplet mining

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in config files
2. **Dataset not found**: Check CSV file paths and image directories
3. **Environment issues**: Recreate conda environment with `conda env create -f environment.yml --force`

### Performance Tips

1. **Faster training**: Use mixed precision (`mixed_precision: true` in config)
2. **Memory optimization**: Reduce batch size and use gradient accumulation
3. **Multi-GPU**: Set `CUDA_VISIBLE_DEVICES` to use specific GPUs

## Citation

If you use this framework in your research, please cite the original papers:

```bibtex
# NetVLAD
@inproceedings{arandjelovic2016netvlad,
  title={NetVLAD: CNN architecture for weakly supervised place recognition},
  author={Arandjelovi{\'c}, Relja and Gronat, Petr and Torii, Akihiko and Pajdla, Tomas and Sivic, Josef},
  booktitle={CVPR},
  year={2016}
}

# AP-GeM
@article{revaud2019learning,
  title={Learning with average precision: Training image retrieval with a listwise loss},
  author={Revaud, Jerome and Almazan, Jon and Rezende, Rafael S and Souza, Cesar Roberto de},
  journal={ICCV},
  year={2019}
}
```

## License

This project is for research purposes. Please check individual algorithm repositories for their specific licenses.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with algorithm implementations or improvements

## Status

- ✅ **Framework Core**: Dataset loading, evaluation metrics, experiment runner
- ✅ **NetVLAD**: Complete implementation with training pipeline
- ✅ **AP-GeM**: Complete implementation with training pipeline
- 🚧 **DELG**: In development (global + local descriptors)
- 📋 **CosPlace**: Planned (cosine-based matching)
- 📋 **EigenPlaces**: Planned (eigenvalue-based features)

---

**Quick Test**: Run `./run_vpr_experiments.sh` to test the current implementations with your GPS-tagged dataset!
