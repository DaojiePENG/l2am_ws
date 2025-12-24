# model_zoo.py


from transformers import AutoModelForSequenceClassification
from torch import nn

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