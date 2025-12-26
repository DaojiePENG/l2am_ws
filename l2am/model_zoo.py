# model_zoo.py


from transformers import AutoModelForSequenceClassification
from torch import nn
from transformers import AutoModel
import torch

# 自定义模型（加 weighted loss），用于解决模型action分布不均的问题
class WeightedSequenceClassifier(nn.Module):
    def __init__(self, model_name, num_labels, class_weights, cache_dir=None):
        super().__init__()
        id2label = {i: f"action_{i}" for i in range(num_labels)}
        label2id = {v: k for k, v in id2label.items()}
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            id2label=id2label, 
            label2id=label2id,
            cache_dir=cache_dir,  # ← 关键：传入 cache_dir
        )
        # self.class_weights = class_weights
        # ✅ 关键修复：注册为 buffer，随模型自动迁移到 GPU
        self.register_buffer("class_weights", class_weights)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=None  # 不让内部计算 loss
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}


class MultiStepWeightedClassifier(nn.Module):
    def __init__(self, model_name, num_labels, class_weights, num_steps=4, cache_dir=None):
        super().__init__()
        self.num_steps = num_steps
        self.num_labels = num_labels

        # 加载 backbone（不带 classification head）
        self.encoder = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )

        # 为每个 future step 创建一个分类头
        self.classifiers = nn.ModuleList([
            nn.Linear(self.encoder.config.hidden_size, num_labels)
            for _ in range(num_steps)
        ])

        # 注册 class weights（每个 step 共享相同的权重）
        self.register_buffer("class_weights", class_weights)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        # 获取 [CLS] 或 mean-pooling 表示（Longformer 默认用 [CLS]）
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # 取 [CLS] token 的表示 (batch_size, hidden_size)
        cls_repr = outputs.last_hidden_state[:, 0, :]  # shape: (B, H)

        # 每个 step 的 logits
        all_logits = []
        for clf in self.classifiers:
            logits = clf(cls_repr)  # (B, num_labels)
            all_logits.append(logits)
        # Stack to (B, num_steps, num_labels)
        logits = torch.stack(all_logits, dim=1)

        loss = None
        if labels is not None:
            # labels: (B, num_steps)
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-100)
            loss = 0.0
            for step in range(self.num_steps):
                step_loss = loss_fct(
                    logits[:, step, :].view(-1, self.num_labels),
                    labels[:, step].view(-1)
                )
                loss += step_loss
            loss = loss / self.num_steps  # 平均 loss

        return {"loss": loss, "logits": logits}