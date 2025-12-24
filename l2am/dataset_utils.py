# dataset_utils.py


# 定义用于构建提示词的函数
def build_prompt(instruction, depth_patches, semantic_patches, num_grid_r=6, num_grid_c=6):
    # 构建提示词的开头
    prompt = f"Instruction: {instruction}\n"
    prompt += "Observation:\n"
    
    # 遍历 num_grid_r=6 x num_grid_c=6 网格，按行拼接
    for i in range(num_grid_r):
        depth_row = []
        sem_row = []
        for j in range(num_grid_c):
            key = f"({i},{j})"
            d_val = depth_patches[key]
            s_val = semantic_patches[key]
            # 将深度值四舍五入保留2位，便于 tokenizer 处理
            depth_str = f"{d_val:.2f}"
            depth_row.append(depth_str)
            sem_row.append(s_val)
        prompt += f"Depth row {i}: {' '.join(depth_row)}\n"
        prompt += f"Semantic row {i}: {' '.join(sem_row)}\n"
    
    return prompt.strip()

# 定义用于构建带有显式位置编码的提示词的函数
def build_prompt_pos(instruction, depth_patches, semantic_patches, num_grid_r=6, num_grid_c=6):
    """
    构建带有显式位置编码的提示词。
    格式示例:
        Instruction: Go to the kitchen.
        Observation Grid:
        [0,0]: depth=1.23, sem=wall; [0,1]: depth=2.10, sem=floor; ...
    """
    prompt = f"Instruction: {instruction}\nObservation Grid:\n"
    
    cells = []
    for i in range(num_grid_r):
        for j in range(num_grid_c):
            key = f"({i},{j})"
            d_val = depth_patches[key]
            s_val = semantic_patches[key]
            # 显式位置 + 结构化字段
            cell_str = f"[{i},{j}]: depth={d_val:.2f}, sem={s_val}"
            cells.append(cell_str)
    
    # 用分号连接所有单元格，保持单行或紧凑多行（tokenizer 可处理）
    prompt += "; ".join(cells)
    return prompt.strip()

def build_prompt_pos_v1(instruction, depth_patches, semantic_patches, num_grid_r=6, num_grid_c=6):
    """
    构建带有显式位置编码的提示词（观测前置版）。
    格式示例:
        Observation Grid:
        [0,0]: depth=1.23, sem=wall; [0,1]: depth=2.10, sem=floor; ...
        Instruction: Go to the kitchen.
    """
    # 先构建观测部分（固定结构）
    cells = []
    for i in range(num_grid_r):
        for j in range(num_grid_c):
            key = f"({i},{j})"
            d_val = depth_patches[key]
            s_val = semantic_patches[key]
            cell_str = f"[{i},{j}]: depth={d_val:.2f}, sem={s_val}"
            cells.append(cell_str)
    
    observation_str = "Observation Grid:\n" + "; ".join(cells)
    
    # 再拼接指令（可变长度）
    prompt = f"{observation_str}\nInstruction: {instruction}"
    return prompt.strip()

def build_prompt_pos_v2(instruction, depth_patches, semantic_patches, num_grid_r=6, num_grid_c=6):
    """
    构建带有显式位置编码的提示词（观测前置版）。进一步提升结构化信息的可读性，加换行分组（每行一个 grid row）。
    格式示例:
        Observation Grid:
        [0,0]: depth=1.23, sem=wall; [0,1]: depth=2.10, sem=floor; 
        [1,0]: depth=1.23, sem=wall; [1,1]: depth=2.10, sem=floor; 
        ...
        Instruction: Go to the kitchen.
    """
    # 先构建观测部分（固定结构）
    # 更易读的多行格式（适合长上下文模型）
    observation_lines = []
    for i in range(num_grid_r):
        row_cells = []
        for j in range(num_grid_c):
            key = f"({i},{j})"
            d_val = depth_patches[key]
            s_val = semantic_patches[key]
            row_cells.append(f"[{i},{j}]: depth={d_val:.2f}, sem={s_val}")
        observation_lines.append("; ".join(row_cells))

    observation_str = "Observation Grid:\n" + "\n".join(observation_lines)
    
    # 再拼接指令（可变长度）
    prompt = f"{observation_str}\nInstruction: {instruction}"
    return prompt.strip()


# 定义用于处理数据集的函数
def prepare_text_samples_batch(batch):
    # batch 是 dict，每个 key 的 value 是 list（长度 = batch_size）
    all_prompts = []
    all_actions = []

    for i in range(len(batch["episodes"])):
        # ep_list = batch["episodes"][i]
        # ep = ep_list[0] if isinstance(ep_list, list) else ep_list
        ep = batch["episodes"][i]  # 直接是 dict，无需 [0] 或 isinstance 检查

        instr = ep["instruction"]
        for frame in ep["frames"]:
            prompt = build_prompt_pos_v2(instr, frame["depth_patches"], frame["semantic_patches"])
            all_prompts.append(prompt)
            all_actions.append(frame["action"])

    return {
        "prompt": all_prompts,
        "action": all_actions
    }


# ======================
# 2. 加载或预处理数据集
# ======================

import os
import glob
from datasets import load_dataset, load_from_disk

def get_or_create_dataset(data_dir, cache_dir):
    if os.path.exists(cache_dir):
        print(f"Loading cached dataset from {cache_dir}")
        return load_from_disk(cache_dir)

    print("No cache found. Loading and processing raw JSON files...")
    json_files = sorted(glob.glob(os.path.join(data_dir, "episodes_part_*.json")))
    if not json_files:
        raise FileNotFoundError(f"No episodes_part_*.json files in {data_dir}")

    print(f"Found {len(json_files)} files. Loading...")
    raw_ds = load_dataset("json", data_files=json_files, split="train")

    # 打印读取的所有.json文件路径
    print("Loaded JSON files:")
    for f in json_files:
        print(f" - {f}")

    print("Expanding episodes to frames...")
    frame_ds = raw_ds.map(
        prepare_text_samples_batch,
        batched=True,
        remove_columns=raw_ds.column_names,
        desc="Building text prompts",
        num_proc=4  # 并行加速（可选）
    )

    print(f"Total frames: {len(frame_ds)}")
    os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
    frame_ds.save_to_disk(cache_dir)
    print(f"Saved processed dataset to {cache_dir}")
    return frame_ds


# ======================
# 3. Tokenize 函数
# ======================
def tokenize_function(examples, tokenizer, max_length=1024):
    return tokenizer(
        examples["prompt"],
        truncation=True,
        max_length=max_length,  # 根据模型调整最大长度
        padding=False  # 由 DataCollator 动态 padding
    )