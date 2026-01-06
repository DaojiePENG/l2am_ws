# set environment variables for NCCL
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# /mnt/slurmfs-4090node1/homes/dpeng108/miniforge3/envs/env_transformer_eval/bin/activate env_transformer_eval


torchrun --nproc_per_node=4 --master_port=29500 l2am/train_chunk_v3.py
