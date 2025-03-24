#!/bin/bash

# Script for running music notation transformer training
# Add these lines just before running the command
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training_${TIMESTAMP}.log"

# Default parameters
DATA_ROOT="/homes/es314/agnostic-OMR/data"
OUTPUT_DIR="outputs"
BATCH_SIZE=1
VAL_BATCH_SIZE=1
EPOCHS=50
LEARNING_RATE=0.0001
HIDDEN_DIM=256
# VIT_MODEL="vit_small_patch16_224"
VIT_MODEL="vit_small_patch16_384" 
DECODER_LAYERS=4
GPU_ID=0
NUM_WORKERS=4
SUBSET_SIZE=0  # 0 means use all data

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    --subset)
      SUBSET_SIZE="$2"
      shift 2
      ;;
    --batch)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --lr)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --vit)
      VIT_MODEL="$2"
      shift 2
      ;;
    --workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --resume)
      RESUME="$2"
      shift 2
      ;;
    --help)
      echo "Usage: ./run.sh [options]"
      echo "Options:"
      echo "  --gpu NUM        GPU ID to use (default: 0)"
      echo "  --subset NUM     Number of samples to use (default: all)"
      echo "  --batch NUM      Batch size (default: 8)"
      echo "  --epochs NUM     Number of epochs (default: 50)"
      echo "  --lr NUM         Learning rate (default: 0.0001)"
      echo "  --vit MODEL      ViT model name (default: vit_small_patch16_224)"
      echo "  --workers NUM    Number of data loading workers (default: 4)"
      echo "  --resume PATH    Path to checkpoint to resume from"
      echo "  --help           Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run './run.sh --help' for usage information."
      exit 1
      ;;
  esac
done

# Print configuration
echo "Starting training with:" | tee -a "$LOG_FILE"
echo "  GPU ID: $GPU_ID" | tee -a "$LOG_FILE"
echo "  Data root: $DATA_ROOT" | tee -a "$LOG_FILE"
echo "  Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "  Batch size: $BATCH_SIZE" | tee -a "$LOG_FILE"
echo "  Validation batch size: $VAL_BATCH_SIZE" | tee -a "$LOG_FILE"
echo "  Epochs: $EPOCHS" | tee -a "$LOG_FILE"
echo "  Learning rate: $LEARNING_RATE" | tee -a "$LOG_FILE"
echo "  Hidden dimension: $HIDDEN_DIM" | tee -a "$LOG_FILE"
echo "  ViT model: $VIT_MODEL" | tee -a "$LOG_FILE"
echo "  Decoder layers: $DECODER_LAYERS" | tee -a "$LOG_FILE"
echo "  Number of workers: $NUM_WORKERS" | tee -a "$LOG_FILE"
if [ "$SUBSET_SIZE" -gt 0 ]; then
  echo "  Subset size: $SUBSET_SIZE" | tee -a "$LOG_FILE"
fi
if [ ! -z "$RESUME" ]; then
  echo "  Resuming from: $RESUME" | tee -a "$LOG_FILE"
fi
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Build command
CMD="python train.py --gpu_id 0 --data_root $DATA_ROOT --output_dir $OUTPUT_DIR"
CMD="$CMD --batch_size $BATCH_SIZE --val_batch_size $VAL_BATCH_SIZE --epochs $EPOCHS"
CMD="$CMD --learning_rate $LEARNING_RATE --hidden_dim $HIDDEN_DIM --vit_model $VIT_MODEL"
CMD="$CMD --decoder_layers $DECODER_LAYERS --num_workers $NUM_WORKERS"

if [ "$SUBSET_SIZE" -gt 0 ]; then
  CMD="$CMD --subset_size $SUBSET_SIZE"
fi

if [ ! -z "$RESUME" ]; then
  CMD="$CMD --resume $RESUME"
fi

# Add mixed precision if using GPU
if [ "$GPU_ID" -ge 0 ]; then
  CMD="$CMD --mixed_precision"
fi

# Print command
echo "Running command:" | tee -a "$LOG_FILE"
echo "$CMD" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run the command and redirect both stdout and stderr to the log file
# Use tee to also display output in the terminal
$CMD 2>&1 | tee -a "$LOG_FILE"

# Check if the command succeeded
if [ ${PIPESTATUS[0]} -eq 0 ]; then
  echo "" | tee -a "$LOG_FILE"
  echo "Training completed successfully!" | tee -a "$LOG_FILE"
else
  echo "" | tee -a "$LOG_FILE"
  echo "Training failed with error code ${PIPESTATUS[0]}" | tee -a "$LOG_FILE"
fi

echo "Log saved to: $LOG_FILE"