export HF_HOME := "/ceph/dalarion/hf-cache"

clear-logs:
    rm -f server_sbatch/logs/*.out

echo-hf-home:
    echo "HF_HOME: $HF_HOME"

run-vllm model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B":
    echo "HF_HOME: $HF_HOME"
    echo "================================================================"
    echo "Running VLLM server for model: {{model}} on port: 11434 at $(hostname)"
    echo "================================================================"
    vllm serve \
        --host="0.0.0.0" \
        --port=11434 \
        --api-key="nllg-key-dsr1" \
        --enable-reasoning \
        --reasoning-parser="deepseek_r1" \
        --kv-cache-dtype="auto" \
        --dtype="bfloat16" \
        {{model}}