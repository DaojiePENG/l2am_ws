# inference_api.py
import os
import torch
from transformers import BigBirdTokenizer, AutoTokenizer
from safetensors.torch import load_file

from l2am.model_zoo import MultiStepWeightedClassifier


class ActionChunkPredictor:
    def __init__(
        self,
        model_checkpoint: str,
        hf_cache_dir: str = "data/hf_model_cache",
        model_name: str = "google/bigbird-roberta-base",
        num_labels: int = 4,
        num_steps: int = 4,
        max_length: int = 1024,
    ):
        self.model_checkpoint = model_checkpoint
        self.hf_cache_dir = hf_cache_dir
        self.model_name = model_name
        self.num_labels = num_labels
        self.num_steps = num_steps
        self.max_length = max_length

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ActionChunkPredictor] Using device: {self.device}")

        # Load tokenizer
        if self.model_name.startswith("google/bigbird"):
            # Load BigBird tokenizer
            self.tokenizer = BigBirdTokenizer.from_pretrained(
                model_checkpoint,
                cache_dir=hf_cache_dir,
                clean_up_tokenization_spaces=True,
            )
        else:
            # Load other model tokenizers
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_checkpoint,
                cache_dir=hf_cache_dir,  # ← 和 download_model.py 一致
                clean_up_tokenization_spaces=True,  # 保持当前行为（清理空格）
                )

        # Build model architecture
        dummy_class_weights = torch.ones(num_labels)
        self.model = MultiStepWeightedClassifier(
            model_name,
            num_labels=num_labels,
            class_weights=dummy_class_weights,
            num_steps=num_steps,
            cache_dir=hf_cache_dir,
        )

        # Load weights
        self._load_weights()

        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self):
        model_safetensors = os.path.join(self.model_checkpoint, "model.safetensors")
        model_bin = os.path.join(self.model_checkpoint, "pytorch_model.bin")

        if os.path.exists(model_safetensors):
            print(f"Loading weights from safetensors: {model_safetensors}")
            state_dict = load_file(model_safetensors, device=str(self.device))
        elif os.path.exists(model_bin):
            print(f"Loading weights from pytorch_model.bin: {model_bin}")
            state_dict = torch.load(model_bin, map_location=self.device)
        else:
            raise FileNotFoundError(
                f"Neither 'model.safetensors' nor 'pytorch_model.bin' found in {self.model_checkpoint}"
            )

        self.model.load_state_dict(state_dict, strict=True)

    def predict(self, prompt: str) -> list[int]:
        """
        Predict an action chunk from a text prompt.

        Args:
            prompt (str): Input instruction or scene description.

        Returns:
            List[int]: Predicted action sequence of length `num_steps`.
        """
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs["logits"]  # (B, num_steps, num_labels)
            pred_classes = torch.argmax(logits, dim=-1)  # (B, num_steps)

            if pred_classes.shape[0] == 1:
                pred_classes = pred_classes.squeeze(0)  # (num_steps,)

            return pred_classes.cpu().tolist()

    def predict_clean(self, prompt: str, stop_token: int = 0) -> list[int]:
        """
        Predict an action chunk from a text prompt, automatically truncating at the stop token.

        Args:
            prompt (str): Input instruction or scene description.

        Returns:
            List[int]: Predicted action sequence of length `num_steps`.
        """
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs["logits"]
            pred = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()

        # 后处理：遇到 stop_token 截断
        truncated = []
        for a in pred:
            truncated.append(a)
            if a == stop_token:
                break
        return truncated