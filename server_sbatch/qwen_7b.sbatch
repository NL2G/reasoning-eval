#!/usr/bin/env bash

#SBATCH --job-name=vllm_dsr1_qwen_7b
#SBATCH --output="./server_sbatch/logs/vllm_dsr1_qwen_7b-%A.out"
#SBATCH --time=128:00:00
#SBATCH --partition=gpu-vram-32gb,gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

srun just run-vllm "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

