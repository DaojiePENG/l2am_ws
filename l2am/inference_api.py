# inference_api.py
import os
import torch
from transformers import BigBirdTokenizer, AutoTokenizer
from safetensors.torch import load_file

from l2am.model_zoo import MultiStepWeightedClassifier
from l2am.model_zoo import WeightedSequenceClassifier


class L2AMActionClassifier:
    def __init__(
        self,
        model_checkpoint="data/l2a_bigbird_action_classifier1/checkpoint-1500",
        model_name="google/bigbird-roberta-base",
        hf_cache_dir="data/hf_model_cache",
        max_length=1024,
        num_labels=4,
        device=None
    ):
        """
        初始化 L2AM 动作分类器。
        
        Args:
            model_checkpoint (str): 模型 checkpoint 路径（包含 model.safetensors 或 pytorch_model.bin）
            model_name (str): HuggingFace 模型名称（如 'google/bigbird-roberta-base'）
            hf_cache_dir (str): HuggingFace 模型缓存目录
            max_length (int): 输入序列最大长度
            num_labels (int): 分类类别数（默认 4）
            device (str or torch.device): 设备（如 'cuda' 或 'cpu'），若为 None 则自动选择
        """
        self.model_checkpoint = model_checkpoint
        self.model_name = model_name
        self.hf_cache_dir = hf_cache_dir
        self.max_length = max_length
        self.num_labels = num_labels

        # 自动选择设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[L2AMActionClassifier] Using device: {self.device}")

        # 加载 tokenizer
        self.tokenizer = BigBirdTokenizer.from_pretrained(
            model_checkpoint,
            cache_dir=hf_cache_dir,
            clean_up_tokenization_spaces=True,
        )

        # 构建模型结构
        dummy_class_weights = torch.ones(self.num_labels)
        self.model = WeightedSequenceClassifier(
            model_name=self.model_name,
            num_labels=self.num_labels,
            class_weights=dummy_class_weights,
            cache_dir=self.hf_cache_dir,
        )

        # 加载权重
        self._load_weights()

        # 移动到设备并设为 eval 模式
        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self):
        """从 checkpoint 加载模型权重（支持 safetensors 或 pytorch_model.bin）"""
        model_safetensors = os.path.join(self.model_checkpoint, "model.safetensors")
        model_bin = os.path.join(self.model_checkpoint, "pytorch_model.bin")

        if os.path.exists(model_safetensors):
            print(f"[L2AMActionClassifier] Loading weights from safetensors: {model_safetensors}")
            state_dict = load_file(model_safetensors, device=str(self.device))
        elif os.path.exists(model_bin):
            print(f"[L2AMActionClassifier] Loading weights from pytorch_model.bin: {model_bin}")
            state_dict = torch.load(model_bin, map_location=self.device)
        else:
            raise FileNotFoundError(
                f"Neither 'model.safetensors' nor 'pytorch_model.bin' found in {self.model_checkpoint}"
            )

        self.model.load_state_dict(state_dict, strict=True)

    def predict(self, prompt: str) -> int:
        """
        对单个文本 prompt 进行动作预测。
        
        Args:
            prompt (str): 输入的文本提示
        
        Returns:
            int: 预测的动作类别（0 ~ num_labels-1）
        """
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs["logits"]
            pred_class = torch.argmax(logits, dim=-1).item()

        return pred_class

    def predict_batch(self, prompts: list[str]) -> list[int]:
        """
        批量预测（注意：当前未启用动态 batch，仍逐样本处理以避免 padding 浪费）
        如需高效 batch 推理，可改用 tokenizer(..., padding=True, ...) 并一次性前向。
        """
        predictions = []
        for prompt in prompts:
            pred = self.predict(prompt)
            predictions.append(pred)
        return predictions

    def predict_batch_fast(self, prompts: list[str], batch_size: int) -> list[int]:
        """
        批量预测（启用动态 batch）
        """
        self.predict("Warm up")  # 预热一次，修改模型状态（如 attention_type到 original_full）
        predictions = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            inputs = self.tokenizer(
                batch_prompts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs["logits"]
                pred_classes = torch.argmax(logits, dim=-1).cpu().tolist()
                predictions.extend(pred_classes)

        return predictions

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
            padding=True,
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
            padding=True,
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
    
    def predict_batch(self, prompts: list[str], batch_size: int = 128) -> list[list[int]]:
        """
        Predict action chunks for a batch of text prompts with controlled batch size.

        Args:
            prompts (List[str]): A list of input instructions or scene descriptions.
            batch_size (int): Maximum number of samples to process at once. Default: 8.

        Returns:
            List[List[int]]: A list of predicted action sequences, each of length `num_steps`.
        """
        if not isinstance(prompts, list):
            raise ValueError("Input 'prompts' must be a list of strings.")
        if len(prompts) == 0:
            return []

        all_predictions = []

        # Process in chunks to avoid OOM
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_prompts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs["logits"]  # (B, num_steps, num_labels)
                pred_classes = torch.argmax(logits, dim=-1)  # (B, num_steps)
                all_predictions.extend(pred_classes.cpu().tolist())

        return all_predictions