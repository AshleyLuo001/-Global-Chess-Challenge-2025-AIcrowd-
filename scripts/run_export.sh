#!/bin/bash
# Ensure to run this script from the project root directory

VERSION="v16"
# Path to the custom model saved by tokenizer_utils.py
MODEL_PATH="models/Qwen3-0.6B-custom"
# Path to the processed training CSV file
DATA_PATH="data/feature/train_v08.csv"
# Output directory for the cached dataset (versioned)
OUTPUT_DIR="data/feature/qwen3-0.6B-${VERSION}_cached_dataset"

echo "Starting dataset cache export using SWIFT..."

# Export CSV dataset to cached format optimized for Qwen3 model
swift export \
    --model $MODEL_PATH \
    --model_type qwen3 \
    --dataset $DATA_PATH \
    --split_dataset_ratio 0.0 \
    --dataset_num_proc 60 \
    --to_cached_dataset true \
    --output_dir $OUTPUT_DIR

echo "Export completed. Cached dataset saved to: $OUTPUT_DIR"