#!/bin/bash
#SBATCH --job-name=l2am_chunk_v1_m05  # 任务名称
#SBATCH -p i64m1tga800u                     # 指定GPU分区（根据HPC文档，可替换为实际可用分区如normal等）
#SBATCH --output=l2am_chunk_v1_m05_%j.out      # 主日志
#SBATCH --error=l2am_chunk_v1_m05_%j.err       # 错误日志
#SBATCH --nodes=1                        # 单节点
#SBATCH --gres=gpu:1                     # 单张GPU（80G）
#SBATCH --cpus-per-task=6               # 足够CPU核心支持并行
#SBATCH --mem=80G                        # 充足内存
#SBATCH --time=168:00:00                  # 总运行时间

# ##########################################################
# # 环境激活（使用你的miniforge环境）
# ##########################################################
# 加载运行所需环境（根据HPC实际环境调整，如conda环境、CUDA等）
# 示例：激活conda环境（若使用conda管理依赖）
# source activate your_conda_env_name  # 替换为你的conda环境名
# ----------------------For HPC3 without conda---------------------------
# source ~/envs/openvla-dual-eval/bin/activate
# ----------------------For HPC2 with miniforge3---------------------------
# CONDA_ENV_PATH="$HOME/miniforge3/envs/openvla-dual"
# echo "[$(date '+%Y-%m-%d %H:%M:%S')] 激活环境: $CONDA_ENV_PATH"
# source "$HOME/miniforge3/etc/profile.d/conda.sh"
# conda activate "$CONDA_ENV_PATH" || {
#         echo "[$(date '+%Y-%m-%d %H:%M:%S')] 错误：初始化conda后仍激活失败！路径: $CONDA_ENV_PATH" && exit 1
# }
# 示例：加载CUDA（若HPC需手动加载，版本需与PyTorch匹配）
module load cuda/11.8

export TOKENIZERS_PARALLELISM=false
# export CUDA_VISIBLE_DEVICES=0,1,2,3

PYTHON="/hpc2hdd/home/zli514/miniforge3/envs/l2am/bin/python"

# 获取当前作业的节点列表
NODE_LIST=$(scontrol show hostnames $SLURM_JOB_NODELIST)
MASTER_ADDR=$(head -n 1 <<< "$NODE_LIST")
MASTER_PORT=29500
NUM_NODES=$SLURM_JOB_NUM_NODES
GPUS_PER_NODE=1   # 必须和 --gres=gpu:6 一致

echo "Running on nodes: $NODE_LIST"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Total processes: $((NUM_NODES * GPUS_PER_NODE))"


$PYTHON -m torch.distributed.run \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=29500 \
    l2am/train_chunk_v1_m05.py