#!/bin/bash

# VPR SOTA Experiment Runner Script
# Simple script to run all VPR algorithms with the conda environment

echo "=== VPR SOTA Experiment Runner ==="
echo "Starting experiments with all SOTA algorithms"
echo ""

# Activate conda environment
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vpr-sota

# Check if environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "vpr-sota" ]]; then
    echo "Error: Could not activate vpr-sota conda environment"
    echo "Please ensure the environment is created with: conda env create -f environment.yml"
    exit 1
fi

echo "Environment activated successfully!"
echo ""

# Set paths
PROJECT_DIR="/home/carlier1/Documents/vpr-sota"
CONFIG_FILE="$PROJECT_DIR/configs/base_experiment_config.yaml"
OUTPUT_DIR="$PROJECT_DIR/experiments/results"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Project directory: $PROJECT_DIR"
echo "Configuration file: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Change to project directory
cd "$PROJECT_DIR"

# Check if we have the dataset files specified in the config file
TRAIN_CSV=$(grep '^train_csv:' "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
TEST_CSV=$(grep '^test_csv:' "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')

if [[ ! -f "$TRAIN_CSV" ]]; then
    echo "Warning: Training dataset not found at $TRAIN_CSV"
    echo "Please ensure your GPS-tagged dataset is properly configured"
fi

if [[ ! -f "$TEST_CSV" ]]; then
    echo "Warning: Test dataset not found at $TEST_CSV"
    echo "Please ensure your GPS-tagged dataset is properly configured"
fi

echo ""

# Available algorithms
ALGORITHMS=("netvlad" "apgem" "delg" "cosplace" "eigenplaces")  # All implemented algorithms

# Check which algorithms are available
AVAILABLE_ALGORITHMS=()
for alg in "${ALGORITHMS[@]}"; do
    if [[ -f "algorithms/$alg/train_${alg}.py" ]]; then
        AVAILABLE_ALGORITHMS+=("$alg")
        echo "✓ $alg implementation found"
    else
        echo "✗ $alg implementation not found (skipping)"
    fi
done

echo ""

if [[ ${#AVAILABLE_ALGORITHMS[@]} -eq 0 ]]; then
    echo "Error: No algorithm implementations found!"
    exit 1
fi

# Run experiments
echo "Starting experiments with algorithms: ${AVAILABLE_ALGORITHMS[*]}"
echo ""

python experiments/run_experiments.py \
    --config "$CONFIG_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --algorithms "${AVAILABLE_ALGORITHMS[@]}"

# Check if experiments completed successfully
if [[ $? -eq 0 ]]; then
    echo ""
    echo "=== EXPERIMENTS COMPLETED SUCCESSFULLY ==="
    echo ""
    echo "Results are available in: $OUTPUT_DIR"
    echo "Comparison report: $OUTPUT_DIR/comparison_report.txt"
    echo "Detailed logs: $OUTPUT_DIR/logs/"
    echo ""
    
    # Show quick summary if report exists
    if [[ -f "$OUTPUT_DIR/comparison_report.txt" ]]; then
        echo "=== QUICK SUMMARY ==="
        head -20 "$OUTPUT_DIR/comparison_report.txt"
        echo ""
        echo "See full report at: $OUTPUT_DIR/comparison_report.txt"
    fi
else
    echo ""
    echo "=== EXPERIMENTS FAILED ==="
    echo "Check the logs for details: $OUTPUT_DIR/logs/"
    exit 1
fi
