# train.py
import os
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
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
DATA_DIR = "data/l2am_r2r_v3/train/8"
CACHE_DIR = "data/cache/train_frames_v1_8"
VAL_DATA_DIR = "data/l2am_r2r_v3/val_seen/8"
VAL_CACHE_DIR = "data/cache/val_seen_frames_v1_8"
HF_CACHE_DIR = "data/hf_model_cache"  # HF 模型缓存路径
RESUME_FROM_CHECKPOINT = None  # "outputs/l2a_longformer_action_classifier/checkpoint-500"  # 设置为某个检查点路径以从该检查点继续训练，否则为 None
# model configs
MODEL_NAME = "google/bigbird-roberta-base"  # 可替换为 roberta-base、 bert-base-uncased、allenai/longformer-base-4096、google/bigbird-roberta-base等
MAX_LENGTH = 1024  # 根据模型调整最大长度

# training configs
OUTPUT_DIR = "outputs/l2a_bigbird_action_classifier_v1_8"
NUM_EPOCHS = 30
PER_DEVICE_TRAIN_BATCH_SIZE = 10
PER_DEVICE_EVAL_BATCH_SIZE = 88
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 6e-5
WARMUP_RATIO = 0.02  # 学习率预热比例
WANDB_RUN_NAME = "bigbird-action-pred-depth-sem_v1_8"
LOGGING_STEPS = 100
EVAL_STEPS = 500
SAVE_STEPS = 500

# ======================
# 2. 加载或预处理数据集
# ======================
from dataset_utils import get_or_create_dataset_v1 as get_or_create_dataset


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
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
    #                                           cache_dir=HF_CACHE_DIR,  # ← 和 download_model.py 一致
    #                                           clean_up_tokenization_spaces=True,  # 保持当前行为（清理空格）
    #                                           )

    # Step 1: 获取帧级数据集
    ds = get_or_create_dataset(DATA_DIR, CACHE_DIR)
    vds = get_or_create_dataset(VAL_DATA_DIR, VAL_CACHE_DIR)

    # Step 2: 划分训练/验证集
    # ds = ds.train_test_split(test_size=0.05, seed=42)
    train_ds = ds
    eval_ds = vds

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
            num_proc=48,  # 👈 关键！用多进程并行 tokenize
        )
        tokenized_eval = eval_ds.map(
            lambda x: tokenize_function(x, tokenizer, max_length=MAX_LENGTH),
            batched=True,
            remove_columns=["prompt"],
            num_proc=48,  # 👈 关键！用多进程并行 tokenize
        )

        tokenized_train = tokenized_train.rename_column("action", "labels")
        tokenized_eval = tokenized_eval.rename_column("action", "labels")

        # 保存数据集以后训练时可直接加载
        tokenized_train.save_to_disk(os.path.join(OUTPUT_DIR, "tokenized_train"))
        tokenized_eval.save_to_disk(os.path.join(OUTPUT_DIR, "tokenized_eval"))

    # Step 4: 确定类别数
    num_labels = len(set(train_ds["action"]))
    print(f"Number of action classes: {num_labels}")

    # Step 5: 加载模型
    from model_zoo import WeightedSequenceClassifier
    # 计算action权重
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    import torch
    labels = np.array(train_ds["action"])
    prompts = np.array(train_ds["prompt"])
    # 打印一个prompt示例：
    print("Example prompt:", prompts[0])
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )
    # class_weights = torch.tensor(class_weights, dtype=torch.float).to("cuda")  # 或 "cpu"
    class_weights = torch.tensor(class_weights, dtype=torch.float) # 多卡训练时放在 Trainer 里处理
    print("Class weights:", class_weights)

    model = WeightedSequenceClassifier(
        MODEL_NAME,
        num_labels=num_labels,
        class_weights=class_weights,
        cache_dir=HF_CACHE_DIR,  # ← 和 download_model.py 一致
    )
    
    # 检查可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params / 1e6:.1f}M")
    print(f"Trainable params: {trainable_params / 1e6:.1f}M ({trainable_params == total_params})")

    # Step 6: 定义评估指标
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        
        # Overall
        acc = accuracy_score(labels, preds)
        
        # Per-class recall (critical for rare class)
        report = classification_report(labels, preds, output_dict=True, zero_division=0)
        metrics = {"accuracy": acc}
        for i in range(4):
            metrics[f"recall_class_{i}"] = report[str(i)]["recall"]
            metrics[f"f1_class_{i}"] = report[str(i)]["f1-score"]
        
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
        metric_for_best_model="eval_f1_class_0",  # ←←← 关键修改：以罕见类F1为最佳模型选择标准
        greater_is_better=True,
        save_total_limit=2,
        report_to="wandb",                 # ←←← 关键：启用 wandb
        run_name=WANDB_RUN_NAME,    # ← 可选：给实验命名
        # report_to="none",
        seed=42,
        dataloader_num_workers=4,
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