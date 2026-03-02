#!/bin/bash
# Ensure to run this script from the project root directory

VERSION="v16"
# Path to the custom Qwen3-0.6B tokenizer/model directory
MODEL_PATH="models/Qwen3-0.6B-custom"
# Path to the preprocessed cached training dataset
CACHED_DATASET="data/feature/qwen3-0.6B-${VERSION}_cached_dataset/train"
# Output directory for trained model weights (versioned)
OUTPUT_DIR="data/models/${VERSION}"

echo "Starting SFT training for Qwen-ChessOracle (Version: ${VERSION})..."

# Launch Qwen3 supervised fine-tuning (SFT) with distributed training
# NPROC_PER_NODE: Number of GPUs per node (8 GPUs)
# MASTER_PORT: Master port for distributed training communication
NPROC_PER_NODE=8 MASTER_PORT=29501 swift sft \
    --model $MODEL_PATH \                 # Path to base model/custom tokenizer
    --model_type qwen3 \                  # Model architecture type (Qwen3 series)
    --output_dir $OUTPUT_DIR \            # Directory to save trained model
    --model_revision master \             # Model revision branch (master)
    --torch_dtype bfloat16 \              # Floating point precision (bfloat16 for efficiency)
    --cached_dataset $CACHED_DATASET \    # Path to cached training dataset
    --train_type full \                   # Training type (full fine-tuning)
    --num_train_epochs 1 \                # Number of training epochs
    --max_length 4096 \                   # Maximum sequence length for input data
    --truncation_strategy delete \        # Strategy for truncating long sequences (delete excess)
    --gradient_checkpointing false \      # Disable gradient checkpointing (faster training)
    --per_device_train_batch_size 32 \    # Batch size per GPU
    --learning_rate 1e-4 \                # Learning rate for optimizer
    --gradient_accumulation_steps 1 \     # Gradient accumulation steps (no accumulation)
    --max_grad_norm 1.0 \                 # Gradient clipping norm to prevent explosion
    --warmup_ratio 0.01 \                 # Learning rate warmup ratio (1% of steps)
    --eval_steps 5000 \                   # Evaluation interval (every 5000 steps)
    --save_steps 5000 \                   # Model saving interval (every 5000 steps)
    --save_total_limit 20 \               # Maximum number of saved checkpoints to keep
    --attn_impl flash_attn \              # Attention implementation (FlashAttention for speed)
    --save_only_model true \              # Save only model weights (exclude optimizer states)
    --freeze_llm false \                  # Do NOT freeze LLM backbone (full fine-tuning)
    --freeze_vit true \                   # Freeze ViT module (not used for chess task)
    --freeze_aligner true \               # Freeze aligner module (not used for chess task)
    --split_dataset_ratio 0 \             # No validation split (train only)
    --dataloader_num_workers 20 \         # Number of workers for data loading
    --dataset_num_proc 30 \               # Number of processes for dataset preprocessing
    --acc_strategy seq \                  # Accuracy strategy (sequence-level)
    --seed 2026 \                         # Random seed for reproducibility
    --deepspeed zero2 \                   # DeepSpeed ZeRO-2 for memory optimization
    --logging_steps 100                   # Log training metrics every 100 steps

echo "Training completed! Model weights saved to $OUTPUT_DIR"