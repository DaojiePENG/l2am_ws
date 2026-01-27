from collections import Counter
import json
import os
import glob
from datasets import load_dataset, load_from_disk


def build_prompt_dsc(instruction, depth_patches, semantic_patches, color_patches, num_grid_r=6, num_grid_c=6):
    """
    构建带有显式位置编码的提示词（观测前置版）。进一步提升结构化信息的可读性，加换行分组（每行一个 grid row）。
    格式示例:
        Observation Grid:
        [0,0]: depth=1.23, semantic=wall; [0,1]: depth=2.10, semantic=floor, color=red; 
        [1,0]: depth=1.23, semantic=wall; [1,1]: depth=2.10, semantic=floor, color=blue; 
        ...
        Instruction: Go to the kitchen.
    """
    # 先构建观测部分（固定结构）
    # 更易读的多行格式（适合长上下文模型）
    system_str = "You are a robot that can turn left or right by a specific degree, moving forward a certain distance or stop at where you are. You need to decide which action to take based on the following Observation Grid and Task Instruction."
    observation_lines = []
    for i in range(num_grid_r):
        row_cells = []
        for j in range(num_grid_c):
            key = f"({i},{j})"
            d_val = depth_patches[key]
            s_val = semantic_patches[key]
            c_val = color_patches[key]
            row_cells.append(f"[{i},{j}]: depth={d_val:.2f}, semantic={s_val}, color={c_val}")
        observation_lines.append("; ".join(row_cells))

    observation_str = "Observation Grid:\n" + "\n".join(observation_lines)
    
    # 再拼接指令（可变长度）
    prompt = f"{system_str}\n{observation_str}\nTask Instruction is: {instruction}"
    return prompt.strip()


def prepare_text_samples_batch_chunk_dsc(batch, num_grid_r=6, num_grid_c=6, num_chunk=4):
    ''' 版本 dsc ：使用 build_prompt_dsc 构建 prompt，包含颜色信息'''
    all_prompts = []
    all_action_chunks = []  # 存储每个样本的 [a_t, a_{t+1}, ..., a_{t+3}]
    all_actions = []  # 存储单步动作 a_t，用于分析类别分布


    for i in range(len(batch["episodes"])):
        ep = batch["episodes"][i]
        instr = ep["instruction"]
        frames = ep["frames"]
        total_frames = len(frames)

        for t in range(total_frames):
            # 当前帧的 prompt（不变）
            prompt = build_prompt_dsc(
                instr,
                frames[t]["depth_patches"],
                frames[t]["semantic_patches"],
                frames[t]["color_patches"],
                num_grid_r=num_grid_r,
                num_grid_c=num_grid_c,
            )
            all_prompts.append(prompt)

            # 构造未来 NUM_CHUNK 个动作，不足则用 0 填充（action0：停止, 符合物理控制逻辑）
            chunk = []
            for k in range(num_chunk):
                if t + k < total_frames:
                    chunk.append(frames[t + k]["action"])
                else:
                    # chunk.append(-100)  # ignore index
                    chunk.append(0)  # stop index
            all_action_chunks.append(chunk)
            all_actions.append(frames[t]["action"])  # 记录当前步的单步动作

    return {
        "prompt": all_prompts,
        "action": all_actions,
        "action_chunk": all_action_chunks  # shape: (N, NUM_CHUNK)
    }


def get_or_create_dataset_chunk_dsc(data_dir, cache_dir, num_grid_r=6, num_grid_c=6, num_chunk=4):
    ''' 版本 dsc ：使用 prepare_text_samples_batch_chunk_dsc 构建数据集，包含颜色信息'''
    if os.path.exists(cache_dir):
        print(f"Loading cached dataset from {cache_dir}")
        return load_from_disk(cache_dir)

    print("No cache found. Loading and processing raw JSON files...")
    json_files = sorted(glob.glob(os.path.join(data_dir, "merged_part_*.json")))
    if not json_files:
        raise FileNotFoundError(f"No merged_part_*.json files in {data_dir}")

    print(f"Found {len(json_files)} files. Loading...")
    raw_ds = load_dataset("json", data_files=json_files, split="train")

    # 打印读取的所有.json文件路径
    print("Loaded JSON files:")
    for f in json_files:
        print(f" - {f}")

    print("Expanding episodes to frames...")
    frame_ds = raw_ds.map(
        prepare_text_samples_batch_chunk_dsc,
        # 传入额外参数
        fn_kwargs={"num_grid_r": num_grid_r, "num_grid_c": num_grid_c, "num_chunk": num_chunk},
        batched=True,
        remove_columns=raw_ds.column_names,
        desc="Building text prompts (v3)",
        num_proc=32,  # 并行加速（可选）
        load_from_cache_file=False  # ← 关键！强制重新计算
    )

    # 统计类别分布
    print(f"Total frames: {len(frame_ds)}")
    actions = [ex["action"] for ex in frame_ds]
    print("Action distribution:", Counter(actions))
    all_actions_single_step = []
    for ex in frame_ds:
        chunk = ex["action_chunk"]
        # 过滤掉 -100，只取有效动作
        valid_actions = [a for a in chunk if a != -100]
        all_actions_single_step.extend(valid_actions)

    print("Flattened action distribution:", Counter(all_actions_single_step))
    os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
    frame_ds.save_to_disk(cache_dir)
    print(f"Saved processed dataset to {cache_dir}")
    return frame_ds









# history

from collections import Counter
import json
import os
import glob
from datasets import load_dataset, load_from_disk


def build_prompt_dsc_his(instruction, history_frames, num_grid_r=6, num_grid_c=6):
    """
    构建带历史观测和显式时间步标签的提示词。
    
    Args:
        instruction (str): 任务指令
        history_frames (list): 按时间顺序排列的帧列表 [oldest, ..., newest]，
                               长度必须等于 num_his
        num_grid_r, num_grid_c: 网格尺寸
    
    Note:
        时间偏移 = index - (len(history_frames) - 1)
        例如 len=4 → indices 0,1,2,3 → time steps -3,-2,-1,0
    """
    system_str = (
        "You are a robot that can turn left or right by a specific degree, "
        "move forward a certain distance, or stop. You must decide your next action "
        "based on the following sequence of time-stamped Observation Grids and the Task Instruction."
    )

    num_his = len(history_frames)
    observation_blocks = []

    for idx, frame in enumerate(history_frames):
        time_step = idx - (num_his - 1)  # e.g., for num_his=4: 0→-3, 1→-2, 2→-1, 3→0
        lines = []
        for i in range(num_grid_r):
            row_cells = []
            for j in range(num_grid_c):
                key = f"({i},{j})"
                d_val = frame["depth_patches"][key]
                s_val = frame["semantic_patches"][key]
                c_val = frame["color_patches"][key]
                row_cells.append(f"[{i},{j}]: depth={d_val:.2f}, semantic={s_val}, color={c_val}")
            lines.append("; ".join(row_cells))
        grid_block = "Observation Grid:\n" + "\n".join(lines)
        block_with_time = f"## Time Step {time_step}:\n{grid_block}"
        observation_blocks.append(block_with_time)

    full_observation = "\n\n".join(observation_blocks)
    prompt = f"{system_str}\n\n{full_observation}\n\nTask Instruction is: {instruction}"
    return prompt.strip()


def prepare_text_samples_batch_chunk_dsc_his(batch, num_grid_r=6, num_grid_c=6, num_chunk=4, num_his=4):
    """
    版本 dsc_his：支持历史观测。
    Each sample uses `num_his` consecutive frames ending at time t (i.e., [t-num_his+1, ..., t]).
    If not enough history, pad with the earliest available frame (i.e., repeat frame at t).
    """
    all_prompts = []
    all_action_chunks = []
    all_actions = []

    for i in range(len(batch["episodes"])):
        ep = batch["episodes"][i]
        instr = ep["instruction"]
        frames = ep["frames"]
        total_frames = len(frames)

        for t in range(total_frames):
            # 收集历史帧：从 t - num_his + 1 到 t（含）
            start_idx = max(0, t - num_his + 1)
            actual_history = frames[start_idx : t + 1]

            # 如果历史不足 num_his，用当前帧（或最早帧）向前填充
            if len(actual_history) < num_his:
                # Option: pad with the first available frame (start_idx) repeated
                padding_frame = frames[start_idx]  # 或 frames[t]，这里用最早可用帧更合理
                padded_history = [padding_frame] * (num_his - len(actual_history)) + actual_history
            else:
                padded_history = actual_history

            # 构建 prompt
            prompt = build_prompt_dsc_his(
                instruction=instr,
                history_frames=padded_history,
                num_grid_r=num_grid_r,
                num_grid_c=num_grid_c
            )
            all_prompts.append(prompt)

            # 动作 chunk：从当前 t 开始预测未来 num_chunk 步
            chunk = []
            for k in range(num_chunk):
                if t + k < total_frames:
                    chunk.append(frames[t + k]["action"])
                else:
                    chunk.append(0)  # stop as padding
            all_action_chunks.append(chunk)
            all_actions.append(frames[t]["action"])

    return {
        "prompt": all_prompts,
        "action": all_actions,
        "action_chunk": all_action_chunks
    }


def get_or_create_dataset_chunk_s_his(data_dir, cache_dir, num_grid_r=6, num_grid_c=6, num_chunk=4, num_his=4):
    """
    构建带历史观测的数据集（s_his 版本）。
    Args:
        data_dir: 原始 JSON 文件目录
        cache_dir: 缓存路径（会自动创建）
        num_his: 历史观测帧数（包括当前帧），如 4 表示 [t-3, t-2, t-1, t]
    """
    if os.path.exists(cache_dir):
        print(f"Loading cached dataset from {cache_dir}")
        return load_from_disk(cache_dir)

    print("No cache found. Loading and processing raw JSON files...")
    json_files = sorted(glob.glob(os.path.join(data_dir, "merged_part_*.json")))
    if not json_files:
        raise FileNotFoundError(f"No merged_part_*.json files in {data_dir}")

    print(f"Found {len(json_files)} files. Loading...")
    raw_ds = load_dataset("json", data_files=json_files, split="train")

    print("Loaded JSON files:")
    for f in json_files:
        print(f" - {f}")

    print(f"Expanding episodes to frames with history (num_his={num_his})...")
    frame_ds = raw_ds.map(
        prepare_text_samples_batch_chunk_dsc_his,
        fn_kwargs={
            "num_grid_r": num_grid_r,
            "num_grid_c": num_grid_c,
            "num_chunk": num_chunk,
            "num_his": num_his
        },
        batched=True,
        remove_columns=raw_ds.column_names,
        desc=f"Building prompts with history (num_his={num_his})",
        num_proc=48,   # 并行加速（可选）
        load_from_cache_file=False
    )

    # 统计
    print(f"Total frames: {len(frame_ds)}")
    actions = [ex["action"] for ex in frame_ds]
    print("Action distribution (current step):", Counter(actions))

    all_actions_flat = []
    for ex in frame_ds:
        chunk = ex["action_chunk"]
        valid_actions = [a for a in chunk if a != -100]  # 虽然现在不用 -100，保留兼容
        all_actions_flat.extend(valid_actions)
    print("Flattened future action distribution:", Counter(all_actions_flat))

    os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
    frame_ds.save_to_disk(cache_dir)
    print(f"Saved processed dataset to {cache_dir}")
    return frame_ds





# 只包含semantic的历史版本
# 只包含semantic的历史版本

def build_prompt_s_his(instruction, history_frames, num_grid_r=6, num_grid_c=6):
    """
    构建带历史观测和显式时间步标签的提示词。
    
    Args:
        instruction (str): 任务指令
        history_frames (list): 按时间顺序排列的帧列表 [oldest, ..., newest]，
                               长度必须等于 num_his
        num_grid_r, num_grid_c: 网格尺寸
    
    Note:
        时间偏移 = index - (len(history_frames) - 1)
        例如 len=4 → indices 0,1,2,3 → time steps -3,-2,-1,0
    """
    system_str = (
        "You are a robot that can turn left or right by a specific degree, "
        "move forward a certain distance, or stop. You must decide your next action "
        "based on the following sequence of time-stamped Observation Grids and the Task Instruction."
    )

    num_his = len(history_frames)
    observation_blocks = []

    for idx, frame in enumerate(history_frames):
        time_step = idx - (num_his - 1)  # e.g., for num_his=4: 0→-3, 1→-2, 2→-1, 3→0
        lines = []
        for i in range(num_grid_r):
            row_cells = []
            for j in range(num_grid_c):
                key = f"({i},{j})"
                # d_val = frame["depth_patches"][key]
                s_val = frame["semantic_patches"][key]
                # c_val = frame["color_patches"][key]
                # row_cells.append(f"[{i},{j}]: depth={d_val:.2f}, semantic={s_val}, color={c_val}")
                row_cells.append(f"[{i},{j}]:semantic={s_val}")
            lines.append("; ".join(row_cells))
        grid_block = "Observation Grid:\n" + "\n".join(lines)
        block_with_time = f"## Time Step {time_step}:\n{grid_block}"
        observation_blocks.append(block_with_time)

    full_observation = "\n\n".join(observation_blocks)
    prompt = f"{system_str}\n\n{full_observation}\n\nTask Instruction is: {instruction}"
    return prompt.strip()


def prepare_text_samples_batch_chunk_s_his(batch, num_grid_r=6, num_grid_c=6, num_chunk=4, num_his=4):
    """
    版本 dsc_his：支持历史观测。
    Each sample uses `num_his` consecutive frames ending at time t (i.e., [t-num_his+1, ..., t]).
    If not enough history, pad with the earliest available frame (i.e., repeat frame at t).
    """
    all_prompts = []
    all_action_chunks = []
    all_actions = []

    for i in range(len(batch["episodes"])):
        ep = batch["episodes"][i]
        instr = ep["instruction"]
        frames = ep["frames"]
        total_frames = len(frames)

        for t in range(total_frames):
            # 收集历史帧：从 t - num_his + 1 到 t（含）
            start_idx = max(0, t - num_his + 1)
            actual_history = frames[start_idx : t + 1]

            # 如果历史不足 num_his，用当前帧（或最早帧）向前填充
            if len(actual_history) < num_his:
                # Option: pad with the first available frame (start_idx) repeated
                padding_frame = frames[start_idx]  # 或 frames[t]，这里用最早可用帧更合理
                padded_history = [padding_frame] * (num_his - len(actual_history)) + actual_history
            else:
                padded_history = actual_history

            # 构建 prompt
            prompt = build_prompt_s_his(
                instruction=instr,
                history_frames=padded_history,
                num_grid_r=num_grid_r,
                num_grid_c=num_grid_c
            )
            all_prompts.append(prompt)

            # 动作 chunk：从当前 t 开始预测未来 num_chunk 步
            chunk = []
            for k in range(num_chunk):
                if t + k < total_frames:
                    chunk.append(frames[t + k]["action"])
                else:
                    chunk.append(0)  # stop as padding
            all_action_chunks.append(chunk)
            all_actions.append(frames[t]["action"])

    return {
        "prompt": all_prompts,
        "action": all_actions,
        "action_chunk": all_action_chunks
    }


def get_or_create_dataset_chunk_s_his(data_dir, cache_dir, num_grid_r=6, num_grid_c=6, num_chunk=4, num_his=4):
    """
    构建带历史观测的数据集（s_his 版本）。
    Args:
        data_dir: 原始 JSON 文件目录
        cache_dir: 缓存路径（会自动创建）
        num_his: 历史观测帧数（包括当前帧），如 4 表示 [t-3, t-2, t-1, t]
    """
    if os.path.exists(cache_dir):
        print(f"Loading cached dataset from {cache_dir}")
        return load_from_disk(cache_dir)

    print("No cache found. Loading and processing raw JSON files...")
    json_files = sorted(glob.glob(os.path.join(data_dir, "merged_part_*.json")))
    if not json_files:
        raise FileNotFoundError(f"No merged_part_*.json files in {data_dir}")

    print(f"Found {len(json_files)} files. Loading...")
    raw_ds = load_dataset("json", data_files=json_files, split="train")

    print("Loaded JSON files:")
    for f in json_files:
        print(f" - {f}")

    print(f"Expanding episodes to frames with history (num_his={num_his})...")
    frame_ds = raw_ds.map(
        prepare_text_samples_batch_chunk_s_his,
        fn_kwargs={
            "num_grid_r": num_grid_r,
            "num_grid_c": num_grid_c,
            "num_chunk": num_chunk,
            "num_his": num_his
        },
        batched=True,
        remove_columns=raw_ds.column_names,
        desc=f"Building prompts with history (num_his={num_his})",
        num_proc=48,   # 并行加速（可选）
        load_from_cache_file=False
    )

    # 统计
    print(f"Total frames: {len(frame_ds)}")
    actions = [ex["action"] for ex in frame_ds]
    print("Action distribution (current step):", Counter(actions))

    all_actions_flat = []
    for ex in frame_ds:
        chunk = ex["action_chunk"]
        valid_actions = [a for a in chunk if a != -100]  # 虽然现在不用 -100，保留兼容
        all_actions_flat.extend(valid_actions)
    print("Flattened future action distribution:", Counter(all_actions_flat))

    os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
    frame_ds.save_to_disk(cache_dir)
    print(f"Saved processed dataset to {cache_dir}")
    return frame_ds