#!/usr/bin/env bash

echo "Running ALL"

metric_types=(
    "consistency"
    "coherence"
    "fluency"
    "relevance"
)

model_names=(
    #"openai/o3-mini"
    #"openai/o3-mini"
    #"openai/o3-mini"
    #"deepseek/deepseek-r1"
    "deepseek/deepseek-r1-distill-llama-70b"
    #"deepseek/deepseek-r1-distill-qwen-32b"
)

reasoning_efforts=(
    #"low"
    #"medium"
    #"high"
    #"none"
    "none"
)

for i in $(seq 0 $((${#model_names[@]} - 1))); do
    model_name=${model_names[$i]}
    reasoning_effort=${reasoning_efforts[$i]}
    echo "================================================"
    echo "Running $model_name with $reasoning_effort reasoning effort"
    echo "================================================"
    for metric_type in "${metric_types[@]}"; do
        echo "<=============>"
        echo "Running $model_name with $metric_type"
        echo "<=============>"
        python openrouter/batch.py \
            --model-name "$model_name" \
            --reasoning-effort "$reasoning_effort" \
            --metric-type "$metric_type" \
            --api-base="https://openrouter.ai/api/v1" \
            --api-key="" \
            --output-dir "evals" \
            --concurrency 100
    done
done

