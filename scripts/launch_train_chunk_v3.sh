#!/bin/bash

# 用法: ./launch_train_chunk_v3.sh <id> [suffix]
# 例如: ./launch_train_chunk_v3.sh 01 chunk8_v3

if [ $# -lt 1 ]; then
    echo "Usage: $0 <task_id> [suffix]"
    exit 1
fi

TASK_ID="$1"
SUFFIX="${2:-default}"

LOG_FILE="task${TASK_ID}_${SUFFIX}.log"
ERR_FILE="task${TASK_ID}_${SUFFIX}.err"
PID_FILE="task${TASK_ID}_${SUFFIX}.pid"

echo "🚀 Starting task ${TASK_ID} with suffix '${SUFFIX}'"
echo "📄 Log file: ${LOG_FILE}"
echo "⚠️  Error file: ${ERR_FILE}"
echo "🆔 PID file: ${PID_FILE}"

# 启动训练（使用 nohup 确保终端退出后仍运行）
nohup bash -c '
    torchrun \
        --nproc_per_node=6 \
        --master_port=29500 \
        l2am/train_chunk_v3.py \
        > "'"$LOG_FILE"'" 2> "'"$ERR_FILE"'"
    
    # 训练结束后自动清理 PID 文件
    rm -f "'"$PID_FILE"'"
' &

# 获取 nohup 启动的 shell 进程 PID
WRAPPER_PID=$!

# 等待几秒让 torchrun 和 python 子进程启动
sleep 3

# 尝试找到实际占用 GPU 的 python 子进程 PID
PYTHON_PID=""
# 方法：查找 WRAPPER_PID 的子进程中包含 "train_chunk_v3.py" 的 python 进程
while read -r pid ppid cmd; do
    if [[ "$ppid" == "$WRAPPER_PID" ]] && [[ "$cmd" == *"python"* ]] && [[ "$cmd" == *"train_chunk_v3.py"* ]]; then
        PYTHON_PID="$pid"
        break
    fi
done < <(ps -eo pid,ppid,args)

# 如果没找到，退而求其次用 torchrun 的直接子进程（通常是 python）
if [ -z "$PYTHON_PID" ]; then
    PYTHON_PID=$(pgrep -P "$WRAPPER_PID" | head -n1)
fi

# 如果还是找不到，就用 WRAPPER_PID（不太理想，但能 kill）
if [ -z "$PYTHON_PID" ]; then
    PYTHON_PID="$WRAPPER_PID"
fi

# 写入 PID 文件（这是你应该 kill 的 PID）
echo "$PYTHON_PID" > "$PID_FILE"
echo "✅ Recorded killable PID: $PYTHON_PID (saved in $PID_FILE)"

echo "💡 To stop training and free GPU memory, run:"
echo "      kill -9 \$(cat $PID_FILE)"
echo "   Or simply:"
echo "      kill -9 $PYTHON_PID"