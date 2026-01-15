# train_chunk_v3.py
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore", message=".*beta.*renamed.*")
warnings.filterwarnings("ignore", message=".*gamma.*renamed.*")


# ======================
# 1. 配置路径
# ======================
DATA_DIR = "data/l2am_r2r_v3/train/6"
CACHE_DIR = "data/cache/train_frames_chunk4_v1_6_m05"
VAL_DATA_DIR = "data/l2am_r2r_v3/val_seen/6"
VAL_CACHE_DIR = "data/cache/val_seen_frames_chunk4_v1_6_m05"
VAL_U_DATA_DIR = "data/l2am_r2r_v3/val_unseen/6"
VAL_U_CACHE_DIR = "data/cache/val_unseen_frames_chunk4_v1_6_m05"

NUM_GRID_R = 6
NUM_GRID_C = 6
HF_CACHE_DIR = "data/hf_model_cache"  # HF 模型缓存路径
RESUME_FROM_CHECKPOINT = None  # "outputs/l2a_longformer_action_classifier/checkpoint-500"  # 设置为某个检查点路径以从该检查点继续训练，否则为 None
# model configs
MODEL_NAME = "google/bigbird-roberta-base"  # 可替换为 roberta-base、 bert-base-uncased、allenai/longformer-base-4096、google/bigbird-roberta-base等
MAX_LENGTH = 1024  # 根据模型调整最大长度
NUM_CHUNK = 4  # 与 dataset_utils 一致

# 数据增强比例：从验证集中抽取一部分数据加入训练集
augment_ratio = 0.5  # 可调整比例

# training configs
OUTPUT_DIR = "outputs/l2a_bigbird_action_classifier_chunk4_v1_6_m05"
NUM_EPOCHS = 30
PER_DEVICE_TRAIN_BATCH_SIZE = 12
PER_DEVICE_EVAL_BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 6e-5
WARMUP_RATIO = 0.02  # 学习率预热比例
WANDB_RUN_NAME = "bigbird-action-chunk4-pred-depth-sem-v1-6-m05"  # 可选：设置 wandb 实验名称
LOGGING_STEPS = 100
EVAL_STEPS = 500
SAVE_STEPS = 500

# ======================
# 2. 加载或预处理数据集
# ======================
from dataset_utils import get_or_create_dataset_chunk_v1


# ======================
# 3. Tokenize 函数
# ======================
from dataset_utils import tokenize_function


# ======================
# 4. 主训练流程
# ======================
def main():
    # 加载分词器
    from transformers import BigBirdTokenizer

    tokenizer = BigBirdTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=HF_CACHE_DIR,
        clean_up_tokenization_spaces=True,
    )
    
    # Step 1: 获取帧级数据集
    ds = get_or_create_dataset_chunk_v1(DATA_DIR, CACHE_DIR, num_grid_r=NUM_GRID_R, num_grid_c=NUM_GRID_C, num_chunk=NUM_CHUNK)
    vds = get_or_create_dataset_chunk_v1(VAL_DATA_DIR, VAL_CACHE_DIR, num_grid_r=NUM_GRID_R, num_grid_c=NUM_GRID_C, num_chunk=NUM_CHUNK)
    vuds = get_or_create_dataset_chunk_v1(VAL_U_DATA_DIR, VAL_U_CACHE_DIR, num_grid_r=NUM_GRID_R, num_grid_c=NUM_GRID_C, num_chunk=NUM_CHUNK)

    # Step 2: 划分训练/验证集
    # ds = ds.train_test_split(test_size=0.1, seed=42)
    train_ds = ds
    eval_ds = vds

    # 将 vds 和 vuds 的一定比例数据加入训练集以增强训练
    from datasets import concatenate_datasets
    vds_sampled = vds.shuffle(seed=42).select(range(int(len(vds) * augment_ratio)))
    vuds_sampled = vuds.shuffle(seed=42).select(range(int(len(vuds) * augment_ratio)))
    train_ds = concatenate_datasets([train_ds, vds_sampled, vuds_sampled])

    # Step 3: Tokenize
    # 如果没有事先保存的数据集，则创建数据集
    if os.path.exists(os.path.join(OUTPUT_DIR, "tokenized_train")) and os.path.exists(os.path.join(OUTPUT_DIR, "tokenized_eval")):
        print("Loading tokenized datasets from disk directory:", OUTPUT_DIR)
        from datasets import load_from_disk
        tokenized_train = load_from_disk(os.path.join(OUTPUT_DIR, "tokenized_train"))
        tokenized_eval = load_from_disk(os.path.join(OUTPUT_DIR, "tokenized_eval"))
    else:
        print("Creating and tokenizing datasets...")    
        tokenized_train = train_ds.map(
            lambda x: tokenize_function(x, tokenizer, max_length=MAX_LENGTH),
            batched=True,
            remove_columns=["prompt"],
            num_proc=48  # 👈 关键！用多进程并行 tokenize
        )
        tokenized_eval = eval_ds.map(
            lambda x: tokenize_function(x, tokenizer, max_length=MAX_LENGTH),
            batched=True,
            remove_columns=["prompt"],
            num_proc=48  # 👈 关键！用多进程并行 tokenize
        )

        # 在 train.py 的 tokenization 部分：
        tokenized_train = tokenized_train.rename_column("action_chunk", "labels")
        tokenized_eval = tokenized_eval.rename_column("action_chunk", "labels")

        # 保存数据集以后训练时可直接加载
        tokenized_train.save_to_disk(os.path.join(OUTPUT_DIR, "tokenized_train"))
        tokenized_eval.save_to_disk(os.path.join(OUTPUT_DIR, "tokenized_eval"))

    # Step 4: 确定类别数
    num_labels = len(set(train_ds["action"]))
    print(f"Number of action classes: {num_labels}")

    # Step 5: 加载模型
    # 计算action权重
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    import torch
    labels = np.array(train_ds["action"])
    prompts = np.array(train_ds["prompt"])
    # 打印一个prompt示例：
    print("Example prompt:", prompts[0])
    print("Example labels chunk:", train_ds[0]["action_chunk"])
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float) # 多卡训练时放在 Trainer 里处理
    print("Class weights:", class_weights)

    from model_zoo import MultiStepWeightedClassifier
    model = MultiStepWeightedClassifier(
        MODEL_NAME,
        num_labels=num_labels,
        class_weights=class_weights,
        num_steps=NUM_CHUNK,
        cache_dir=HF_CACHE_DIR,
    )
    
    # 检查可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params / 1e6:.1f}M")
    print(f"Trainable params: {trainable_params / 1e6:.1f}M ({trainable_params == total_params})")

    # Step 6: 定义评估指标
    def compute_metrics(eval_pred):
        logits, labels = eval_pred  # logits: (N, NUM_CHUNK, num_labels), labels: (N, NUM_CHUNK)
        preds = np.argmax(logits, axis=-1)  # (N, NUM_CHUNK)

        metrics = {}
        total_acc = 0.0

        # 假设 num_labels = 4（根据你的数据）
        num_labels = logits.shape[-1]

        for step in range(NUM_CHUNK):
            # 取出当前 step 的标签和预测
            step_labels = labels[:, step]
            step_preds = preds[:, step]

            # 过滤掉 ignore_index (-100)
            valid_mask = step_labels != -100
            if not np.any(valid_mask):
                # 如果该 step 没有有效样本（比如全是 padding），跳过
                for cls_id in range(num_labels):
                    metrics[f"step{step}_recall_class_{cls_id}"] = 0.0
                    metrics[f"step{step}_f1_class_{cls_id}"] = 0.0
                metrics[f"step{step}_acc"] = 0.0
                continue

            step_labels = step_labels[valid_mask]
            step_preds = step_preds[valid_mask]

            # Accuracy
            acc = accuracy_score(step_labels, step_preds)
            metrics[f"step{step}_acc"] = acc
            total_acc += acc

            # Classification report per class
            report = classification_report(
                step_labels, step_preds,
                labels=list(range(num_labels)),  # 显式指定所有类别（即使未出现）
                output_dict=True,
                zero_division=0
            )

            for cls_id in range(num_labels):
                cls_str = str(cls_id)
                metrics[f"step{step}_recall_class_{cls_id}"] = report[cls_str]["recall"]
                metrics[f"step{step}_f1_class_{cls_id}"] = report[cls_str]["f1-score"]

        metrics["mean_step_acc"] = total_acc / NUM_CHUNK

        # 可选：保留第一步的总体指标用于兼容或对比
        if "step0_acc" in metrics:
            metrics["first_step_acc"] = metrics["step0_acc"]
            for cls_id in range(num_labels):
                metrics[f"first_step_recall_class_{cls_id}"] = metrics[f"step0_recall_class_{cls_id}"]
                metrics[f"first_step_f1_class_{cls_id}"] = metrics[f"step0_f1_class_{cls_id}"]

        return metrics


    # Step 7: 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,          # ←←← 关键修改：设置学习率
        warmup_ratio=WARMUP_RATIO,                # ←←← 关键修改：设置学习率预热比例
        weight_decay=0.01,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        # metric_for_best_model="eval_step0_f1_class_0",  # ←←← 关键修改：以罕见类F1为最佳模型选择标准
        metric_for_best_model="eval_step0_acc",  
        greater_is_better=True,
        save_total_limit=4,
        report_to="wandb",                 # ←←← 关键：启用 wandb
        run_name=WANDB_RUN_NAME,    # ← 可选：给实验命名
        # report_to="none",
        seed=42,
        dataloader_num_workers=16,
        ddp_find_unused_parameters=True,  # ←←← 添加这一行来打开ddp的unused parameter检查
        save_safetensors=False,  # ←←← 关键！禁用 safetensors以避免兼容性问题
    )

    # Step 8: 数据整理器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Step 9: 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Step 10: 开始训练
    print("🚀 Starting training...")
    # 继续之前的训练（如果有的话）
    # trainer.train(resume_from_checkpoint=True)
    if RESUME_FROM_CHECKPOINT is not None:
        print(f"Resuming training from checkpoint: {RESUME_FROM_CHECKPOINT}")
        trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
    else:
        trainer.train()


    # Step 11: 保存最终模型
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))

    print(f"✅ Training completed! Model saved to {os.path.join(OUTPUT_DIR, 'final')}")


if __name__ == "__main__":
    main()