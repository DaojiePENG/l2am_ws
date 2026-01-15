# dataset_utils.py


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

def build_prompt_pos_v3(instruction, depth_patches, semantic_patches, color_patches, num_grid_r=6, num_grid_c=6):
    """
    构建带有显式位置编码的提示词（观测前置版）。进一步提升结构化信息的可读性，加换行分组（每行一个 grid row）。
    格式示例:
        Observation Grid:
        [0,0]: depth=1.23, sem=wall; [0,1]: depth=2.10, sem=floor, color=red; 
        [1,0]: depth=1.23, sem=wall; [1,1]: depth=2.10, sem=floor, color=blue; 
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
            c_val = color_patches[key]
            # row_cells.append(f"[{i},{j}]: depth={d_val:.2f}, sem={s_val}")
            row_cells.append(f"[{i},{j}]: depth={d_val:.2f}, sem={s_val}, color={c_val}")
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

import numpy as np
def prepare_text_samples_batch_aug(batch, aug_dict):
    all_prompts = []
    all_actions = []

    for i in range(len(batch["episodes"])):
        ep = batch["episodes"][i]
        episode_id = ep["episode_id"]
        original_instr = ep["instruction"]
        frames = ep["frames"]

        # 获取所有候选指令
        aug_entry = aug_dict.get(episode_id, {})
        candidate_instructions = [original_instr]  # 至少包含原始指令

        # 按顺序添加 augmented_instruction_1 到 augmented_instruction_10（如果存在）
        for j in range(1, 11):
            key = f"augmented_instruction_{j}"
            if key in aug_entry:
                candidate_instructions.append(aug_entry[key])
            else:
                break  # 假设是连续的；也可不 break，继续检查

        # 对每条指令，遍历所有帧
        for instr in candidate_instructions:
            for frame in frames:
                prompt = build_prompt_pos_v2(instr, frame["depth_patches"], frame["semantic_patches"])
                all_prompts.append(prompt)
                all_actions.append(frame["action"])

    return {
        "prompt": all_prompts,
        "action": all_actions
    }

def prepare_text_samples_batch_color(batch, num_grid_r=6, num_grid_c=6):
    # batch 是 dict，每个 key 的 value 是 list（长度 = batch_size）
    all_prompts = []
    all_actions = []

    for i in range(len(batch["episodes"])):
        # ep_list = batch["episodes"][i]
        # ep = ep_list[0] if isinstance(ep_list, list) else ep_list
        ep = batch["episodes"][i]  # 直接是 dict，无需 [0] 或 isinstance 检查

        instr = ep["instruction"]
        for frame in ep["frames"]:
            prompt = build_prompt_pos_v3(instr, frame["depth_patches"], frame["semantic_patches"], frame["color_patches"], num_grid_r=num_grid_r, num_grid_c=num_grid_c)
            all_prompts.append(prompt)
            all_actions.append(frame["action"])

    return {
        "prompt": all_prompts,
        "action": all_actions
    }


NUM_CHUNK = 4  # ← 新增全局常量

def prepare_text_samples_batch_chunk(batch):
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
            prompt = build_prompt_pos_v2(
                instr,
                frames[t]["depth_patches"],
                frames[t]["semantic_patches"]
            )
            all_prompts.append(prompt)

            # 构造未来 NUM_CHUNK 个动作，不足则用 -100 填充（PyTorch CrossEntropy 忽略 -100）
            chunk = []
            for k in range(NUM_CHUNK):
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


def prepare_text_samples_batch_chunk_v1(batch, num_grid_r=6, num_grid_c=6, num_chunk=4):
    ''' 版本3：使用 build_prompt_pos_v1 构建 prompt，不包含颜色信息'''
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
            prompt = build_prompt_pos_v2(
                instr,
                frames[t]["depth_patches"],
                frames[t]["semantic_patches"],
                # frames[t]["color_patches"],
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


def prepare_text_samples_batch_chunk_v3(batch, num_grid_r=6, num_grid_c=6, num_chunk=4):
    ''' 版本3：使用 build_prompt_pos_v3 构建 prompt，包含颜色信息'''
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
            prompt = build_prompt_pos_v3(
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
        num_proc=16,  # 并行加速（可选）
        load_from_cache_file=False  # ← 关键！强制重新计算
    )

    print(f"Total frames: {len(frame_ds)}")
    os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
    frame_ds.save_to_disk(cache_dir)
    print(f"Saved processed dataset to {cache_dir}")
    return frame_ds

def get_or_create_dataset_v1(data_dir, cache_dir):
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
        prepare_text_samples_batch,
        batched=True,
        remove_columns=raw_ds.column_names,
        desc="Building text prompts",
        num_proc=16,  # 并行加速（可选）
        load_from_cache_file=False  # ← 关键！强制重新计算
    )

    print(f"Total frames: {len(frame_ds)}")
    os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
    frame_ds.save_to_disk(cache_dir)
    print(f"Saved processed dataset to {cache_dir}")
    return frame_ds

import json
def get_or_create_dataset_v1_aug(data_dir, cache_dir, aug_instructions_path):
    # 加载增强指令
    with open(aug_instructions_path, 'r') as f:
        aug_instructions = json.load(f)

    # 将增强指令转换为字典形式，便于快速查找
    aug_dict = {ep["episode_id"]: ep for ep in aug_instructions["episodes"]}

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
        lambda batch: prepare_text_samples_batch_aug(batch, aug_dict),
        batched=True,
        remove_columns=raw_ds.column_names,
        desc="Building text prompts",
        num_proc=32,  # 并行加速（可选）
        load_from_cache_file=False  # ← 关键！强制重新计算
    )

    print(f"Total frames: {len(frame_ds)}")
    os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
    frame_ds.save_to_disk(cache_dir)
    print(f"Saved processed dataset to {cache_dir}")
    return frame_ds

def get_or_create_dataset_v1_color(data_dir, cache_dir):
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
        prepare_text_samples_batch_color,
        batched=True,
        remove_columns=raw_ds.column_names,
        desc="Building text prompts",
        num_proc=32,  # 并行加速（可选）
        load_from_cache_file=False  # ← 关键！强制重新计算
    )

    print(f"Total frames: {len(frame_ds)}")
    os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
    frame_ds.save_to_disk(cache_dir)
    print(f"Saved processed dataset to {cache_dir}")
    return frame_ds

from collections import Counter
def get_or_create_dataset_chunk(data_dir, cache_dir):
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
        prepare_text_samples_batch_chunk_v1,
        batched=True,
        remove_columns=raw_ds.column_names,
        desc="Building text prompts (v1)",
        num_proc=8,  # 并行加速（可选）
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

def get_or_create_dataset_chunk_v3(data_dir, cache_dir, num_grid_r=6, num_grid_c=6, num_chunk=4):
    ''' 版本3：使用 prepare_text_samples_batch_chunk_v3 构建数据集，包含颜色信息'''
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
        prepare_text_samples_batch_chunk_v3,
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

def get_or_create_dataset_chunk_v1(data_dir, cache_dir, num_grid_r=6, num_grid_c=6, num_chunk=4):
    ''' 版本3：使用 prepare_text_samples_batch_chunk_v3 构建数据集，包含颜色信息'''
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
        prepare_text_samples_batch_chunk_v1,
        # 传入额外参数
        fn_kwargs={"num_grid_r": num_grid_r, "num_grid_c": num_grid_c, "num_chunk": num_chunk},
        batched=True,
        remove_columns=raw_ds.column_names,
        desc="Building text prompts (v1)",
        num_proc=48,  # 并行加速（可选）
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

# ======================
# 3. Tokenize 函数（含基础Mask策略）
# ======================
import torch
import random
from typing import Dict, List, Any

def tokenize_function_mask_v1(
    examples: Dict[str, List[Any]],
    tokenizer,
    max_length: int = 1024,
    mask_prob: float = 0.15,
    special_token_masking: bool = False  # 是否允许 mask 特殊 token（通常设为 False）
) -> Dict[str, List[List[int]]]:
    """
    Tokenize prompts and apply random token masking as data augmentation.
    
    This is NOT MLM! The labels remain the original action_chunk.
    Masking is only applied to input_ids to simulate partial observation/noisy instruction.
    
    Args:
        examples: {"prompt": [str1, str2, ...]}
        tokenizer: HuggingFace tokenizer (e.g., BigBirdTokenizer)
        max_length: max sequence length
        mask_prob: probability of masking a non-special token
        special_token_masking: whether to allow masking [CLS], [SEP], etc. (not recommended)
    
    Returns:
        {"input_ids": [[id1, id2, ...], ...]}  # list of lists
    """
    # Step 1: Tokenize to get input_ids and attention_mask as lists
    batch_encoding = tokenizer(
        examples["prompt"],
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=False,
        return_tensors=None  # Important: return Python lists
    )
    
    input_ids_list = batch_encoding["input_ids"]
    masked_input_ids_list = []
    
    vocab_size = tokenizer.vocab_size
    mask_token_id = tokenizer.mask_token_id
    
    for input_ids in input_ids_list:
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        seq_len = input_ids.size(0)
        
        # Get special tokens mask (1 = special token)
        if hasattr(tokenizer, "get_special_tokens_mask"):
            special_tokens_mask = tokenizer.get_special_tokens_mask(
                input_ids.tolist(), already_has_special_tokens=True
            )
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            # Fallback: assume first and last are special (e.g., [CLS], [SEP])
            special_tokens_mask = torch.zeros(seq_len, dtype=torch.bool)
            if seq_len > 0:
                special_tokens_mask[0] = True
                if seq_len > 1:
                    special_tokens_mask[-1] = True
        
        # Create mask probability matrix
        mask_probs = torch.full((seq_len,), mask_prob, dtype=torch.float32)
        if not special_token_masking:
            mask_probs.masked_fill_(special_tokens_mask, 0.0)
        
        # Sample which tokens to mask
        mask_indices = torch.bernoulli(mask_probs).bool()
        
        # 80% -> [MASK], 10% -> random token, 10% -> keep original
        # But since this is data augmentation (not MLM), many just use 100% [MASK]
        # We'll follow standard BERT-style for stronger regularization
        
        # Clone to avoid modifying original
        masked_ids = input_ids.clone()
        
        # 80%: replace with [MASK]
        replace_with_mask = torch.bernoulli(torch.full((seq_len,), 0.8)).bool() & mask_indices
        masked_ids[replace_with_mask] = mask_token_id
        
        # 10%: replace with random token
        remaining = mask_indices & ~replace_with_mask
        replace_with_random = torch.bernoulli(torch.full((seq_len,), 0.5)).bool() & remaining
        random_tokens = torch.randint(0, vocab_size, (seq_len,), dtype=torch.long)
        masked_ids[replace_with_random] = random_tokens[replace_with_random]
        
        # 10%: keep original (do nothing)
        
        masked_input_ids_list.append(masked_ids.tolist())
    
    return {"input_ids": masked_input_ids_list}