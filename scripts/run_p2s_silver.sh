#!/bin/bash
# Run the decomposed annotation (Point-to-Span) silver experiment pipeline.
#
# This implements the proposed decomposed annotation strategy from the paper.
# The pipeline has three stages:
#   Stage 1: Train a P2S model on a small set of annotated point examples (~200)
#            Then run P2S inference to generate silver NER labels on a larger unlabeled set
#   Stage 2: Train a NER model on the Stage 1 silver labels
#            Then run NER inference to generate more silver NER labels
#   Stage 3: Train a final NER model on Stage 2 silver labels, evaluate on test
#
# Required environment variables:
#   CONFIG_FILE   - Path to the experiment config JSON (see configs/example_p2s_silver_genia.json)
#   REPO_DIR      - Path to this repository root
#
# Optional environment variables:
#   NUM_GPUS      - Number of GPUs (default: 1)
#   EPOCHS        - Training epochs for stages 1 and 3 (default: 25)
#   SILVER_EPOCHS - Training epochs for stage 2 silver model (default: 60)
#   BATCH_SIZE    - Per-device batch size (default: 5)
#   MAX_LENGTH    - Max token length (default: 370)
#   LEARNING_RATE - Learning rate (default: 2e-4)
#
# Usage:
#   CONFIG_FILE=configs/example_p2s_silver_genia.json \
#   REPO_DIR=/path/to/ner-efficiency-public \
#   bash scripts/run_p2s_silver.sh

set -e

if ! command -v jq &>/dev/null; then
    echo "Error: jq is required. Install with: apt-get install jq or brew install jq"
    exit 1
fi

if [ -z "$CONFIG_FILE" ] || [ -z "$REPO_DIR" ]; then
    echo "Usage: CONFIG_FILE=<config.json> REPO_DIR=<repo_dir> bash $0"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE"
    exit 1
fi

NUM_GPUS="${NUM_GPUS:-1}"
EPOCHS="${EPOCHS:-25}"
SILVER_EPOCHS="${SILVER_EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-5}"
MAX_LENGTH="${MAX_LENGTH:-370}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"

export NCCL_P2P_DISABLE=1

find_available_port() {
    while true; do
        PORT=$((1024 + RANDOM % 64512))
        if ! lsof -i:$PORT >/dev/null 2>&1; then
            echo $PORT
            return
        fi
    done
}
export MASTER_PORT=$(find_available_port)

# Read config
model_weights=$(jq -r '.model_weights' "$CONFIG_FILE")
train_1_data_path=$(jq -r '.train1_data_path' "$CONFIG_FILE")
train_1_output_dir=$(jq -r '.train1_output_dir' "$CONFIG_FILE")
eval_1_data_path=$(jq -r '.train12_data_path' "$CONFIG_FILE")
negative_examples_data_path=$(jq -r '.negative_train_data_path' "$CONFIG_FILE")
to_overwrite_data_path=$(jq -r '.train1_ner_data_path' "$CONFIG_FILE")
train_2_data_path=$(jq -r '.silver12_data_path' "$CONFIG_FILE")
train_2_output_dir=$(jq -r '.train12_output_dir' "$CONFIG_FILE")
eval_2_data_path=$(jq -r '.train3_data_path' "$CONFIG_FILE")
train_3_data_path=$(jq -r '.silver3_data_path' "$CONFIG_FILE")
train_3_output_dir=$(jq -r '.train3_output_dir' "$CONFIG_FILE")
eval_3_data_path=$(jq -r '.test_data_path' "$CONFIG_FILE")
entity_type_map=$(jq -r '.entity_type_map' "$CONFIG_FILE")

mkdir -p "$train_1_output_dir" "$train_2_output_dir" "$train_3_output_dir"

final_result="${train_3_output_dir}/test.csv"
if [ -f "$final_result" ]; then
    echo "Final results already exist at $final_result. Exiting."
    exit 0
fi

echo "=== P2S Silver Experiment (Decomposed Annotation) ==="
echo "Config:         $CONFIG_FILE"
echo "Model:          $model_weights"
echo "Stage 1 (P2S):  $train_1_data_path"
echo "Stage 2 (NER):  $train_2_data_path"
echo "Stage 3 (NER):  $train_3_data_path"
echo "Test:           $eval_3_data_path"
echo "======================================================"

cd "$REPO_DIR"

# --- Stage 1: Train P2S model on small annotated set ---
model_checkpoint=("$train_1_output_dir"/checkpoint-*)
if [ ! -d "${model_checkpoint[0]}" ]; then
    echo "Stage 1: Training P2S model..."
    CUDA_VISIBLE_DEVICES=0 deepspeed --num_gpus=$NUM_GPUS --master_port=$MASTER_PORT src/train/train_lora.py \
        --model_name_or_path "$model_weights" \
        --data_path "$train_1_data_path" \
        --bf16 True \
        --output_dir "$train_1_output_dir" \
        --num_train_epochs $EPOCHS \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --save_strategy "epoch" \
        --save_total_limit 1 \
        --learning_rate $LEARNING_RATE \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 10 \
        --tf32 True \
        --model_max_length $MAX_LENGTH \
        --deepspeed src/train/deepspeed_config_s2.json \
        --lora_r 8 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --q_lora True \
        --max_steps 2000
else
    echo "Stage 1: Found existing checkpoint, skipping training."
fi

# --- Stage 1 eval: run P2S inference to generate silver NER labels ---
echo "Stage 1 eval: Generating P2S silver labels..."
python -m src.evaluate \
    --lora_path "$train_1_output_dir" \
    --model_path "$model_weights" \
    --tensor_parallel_size $NUM_GPUS \
    --result_filepath "${train_1_output_dir}/eval.csv" \
    --entity_type_map "$entity_type_map" \
    --silver_data_path "$eval_1_data_path" \
    --negative_data_path "$negative_examples_data_path" \
    --to_overwrite_data_path "$to_overwrite_data_path" \
    --from_p2s

# --- Stage 2: Train NER model on Stage 1 silver labels ---
model_checkpoint=("$train_2_output_dir"/checkpoint-*)
if [ ! -d "${model_checkpoint[0]}" ]; then
    echo "Stage 2: Training NER model on silver data..."
    CUDA_VISIBLE_DEVICES=0 deepspeed --num_gpus=$NUM_GPUS --master_port=$MASTER_PORT src/train/train_lora.py \
        --model_name_or_path "$model_weights" \
        --data_path "$train_2_data_path" \
        --bf16 True \
        --output_dir "$train_2_output_dir" \
        --num_train_epochs $SILVER_EPOCHS \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --save_strategy "epoch" \
        --save_total_limit 1 \
        --learning_rate $LEARNING_RATE \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 10 \
        --tf32 True \
        --model_max_length $MAX_LENGTH \
        --deepspeed src/train/deepspeed_config_s2.json \
        --lora_r 8 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --q_lora True
else
    echo "Stage 2: Found existing checkpoint, skipping training."
fi

# --- Stage 2 eval: generate more silver labels on next split ---
echo "Stage 2 eval: Generating second-round silver labels..."
python -m src.evaluate \
    --lora_path "$train_2_output_dir" \
    --model_path "$model_weights" \
    --tensor_parallel_size $NUM_GPUS \
    --result_filepath "${train_2_output_dir}/eval.csv" \
    --entity_type_map "$entity_type_map" \
    --test_data_path "$eval_3_data_path" \
    --silver_data_path "$eval_2_data_path"

# --- Stage 3: Train final NER model on Stage 2 silver labels ---
model_checkpoint=("$train_3_output_dir"/checkpoint-*)
if [ ! -d "${model_checkpoint[0]}" ]; then
    echo "Stage 3: Training final NER model..."
    CUDA_VISIBLE_DEVICES=0 deepspeed --num_gpus=$NUM_GPUS --master_port=$MASTER_PORT src/train/train_lora.py \
        --model_name_or_path "$model_weights" \
        --data_path "$train_3_data_path" \
        --bf16 True \
        --output_dir "$train_3_output_dir" \
        --num_train_epochs $EPOCHS \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --save_strategy "epoch" \
        --save_total_limit 1 \
        --learning_rate $LEARNING_RATE \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 10 \
        --tf32 True \
        --model_max_length $MAX_LENGTH \
        --deepspeed src/train/deepspeed_config_s2.json \
        --lora_r 8 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --q_lora True
else
    echo "Stage 3: Found existing checkpoint, skipping training."
fi

# --- Final eval on held-out test set ---
echo "Final eval: Evaluating on test set..."
python -m src.evaluate \
    --lora_path "$train_3_output_dir" \
    --model_path "$model_weights" \
    --tensor_parallel_size $NUM_GPUS \
    --result_filepath "$final_result" \
    --entity_type_map "$entity_type_map" \
    --test_data_path "$eval_3_data_path"

echo "Done. Results: $final_result"
