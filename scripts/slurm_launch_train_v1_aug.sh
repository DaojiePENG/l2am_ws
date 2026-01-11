#!/bin/bash
#SBATCH --job-name=train_ddp_aug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4                # 显式申请 4 块 GPU（强烈建议加上！）
#SBATCH --output=train_ddp_aug_%j.out
#SBATCH --error=train_ddp_aug_%j.err
#SBATCH --nodelist=4090node2


export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

PYTHON="/mnt/slurmfs-4090node1/homes/dpeng108/miniforge3/envs/env_transformer_eval/bin/python"

$PYTHON -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=29500 \
    l2am/train_v1_aug.py