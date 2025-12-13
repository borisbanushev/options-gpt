from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Dict, Any

import torch
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from torch.utils.tensorboard import SummaryWriter

from ..configs.config import FineTuningConfig
from ..tracking.weave_tracker import WeaveTracker
from ..data.schemas import TrainingExample


@dataclass
class UnslothTrainer:
	cfg: FineTuningConfig
	tracker: WeaveTracker
	tensorboard_writer: SummaryWriter = None

	def _load_model(self):
		from peft import LoraConfig, get_peft_model, TaskType
		from glob import glob
		from huggingface_hub import snapshot_download
		
		device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
		print(f' Device: {device}.')

		# Respect local-only/offline settings to avoid re-downloading
		local_only = os.getenv("HF_LOCAL_ONLY", "").lower() in ("1", "true", "yes") or \
			os.getenv("TRANSFORMERS_OFFLINE", "").lower() in ("1", "true", "yes")
		print(f'local_only: {local_only}.')
		cache_dir = os.getenv("HF_HOME") or os.getenv("HUGGINGFACE_HUB_CACHE") or None
		model_local_path = os.getenv("MODEL_LOCAL_PATH")

		# If a local path is specified, ensure it's populated once, then always use it
		if model_local_path:
			os.makedirs(model_local_path, exist_ok=True)
			index_file = os.path.join(model_local_path, "model.safetensors.index.json")
			shards = glob(os.path.join(model_local_path, "model-*.safetensors"))
			missing_weights = not (os.path.isfile(index_file) and len(shards) > 0)
			if missing_weights and not local_only:
				# One-time download into the specified local directory
				snapshot_download(
					repo_id=self.cfg.train.model_id,
					local_dir=model_local_path,
					local_dir_use_symlinks=False,
					allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.jinja"],
				)
			# Always load from the local path afterward
			model_id_or_path = model_local_path
			local_only = True
		else:
			model_id_or_path = self.cfg.train.model_id
		
		# Tokenizer (prefer fast, with robust fallback)
		try:
			tokenizer = AutoTokenizer.from_pretrained(
				model_id_or_path,
				trust_remote_code=True,
				local_files_only=local_only,
				cache_dir=cache_dir,
				use_fast=True,
			)
		except Exception:
			# Fallback: load tokenizer.json directly if present
			from transformers import PreTrainedTokenizerFast
			tokenizer_json = os.path.join(model_id_or_path, "tokenizer.json") if os.path.isdir(model_id_or_path) else None
			if tokenizer_json and os.path.isfile(tokenizer_json):
				tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_json)
				tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
			else:
				# Re-raise original if no local tokenizer.json
				raise
		if tokenizer.pad_token is None:
			tokenizer.pad_token = tokenizer.eos_token
		
		# Base model (no bitsandbytes on Apple Silicon)
		model = AutoModelForCausalLM.from_pretrained(
			model_id_or_path,
			trust_remote_code=True,
			torch_dtype=torch.bfloat16 if device != "cpu" else None,
			local_files_only=local_only,
			cache_dir=cache_dir,
		)

		print('Model Leaded...')
		# Reduce memory: disable cache during training
		if hasattr(model, "config") and hasattr(model.config, "use_cache"):
			model.config.use_cache = False
		model.to(device)
		
		# Enable grad checkpointing if requested
		if self.cfg.train.gradient_checkpointing:
			model.gradient_checkpointing_enable()
		
		# Apply LoRA
		lora_config = LoraConfig(
			r=self.cfg.qlora.rank,
			lora_alpha=self.cfg.qlora.alpha,
			target_modules=self.cfg.qlora.target_modules,
			lora_dropout=self.cfg.qlora.dropout,
			bias=self.cfg.qlora.bias,
			task_type=TaskType.CAUSAL_LM,
		)
		model = get_peft_model(model, lora_config)
		return model, tokenizer

	def _build_dataset(self, examples: Iterable[TrainingExample]) -> Dataset:
		records = [e.model_dump() for e in examples]
		return Dataset.from_list(records)

	def _formatting_func(self, example: Dict[str, Any]) -> str:
		return f"Instruction: {example['instruction']}\nInput: {example['input']}\nOutput: {example['output']}"

	def train(self, train_examples: Iterable[TrainingExample], eval_examples: Iterable[TrainingExample] | None = None) -> None:
		os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
		
		# Initialize TensorBoard
		tensorboard_log_dir = os.path.join(self.cfg.train.output_dir, "tensorboard_logs")
		os.makedirs(tensorboard_log_dir, exist_ok=True)
		self.tensorboard_writer = SummaryWriter(log_dir=tensorboard_log_dir)
		
		model, tokenizer = self._load_model()
		train_ds = self._build_dataset(train_examples)
		if eval_examples:
			eval_ds = self._build_dataset(eval_examples)
		else:
			eval_ds = None

		from trl import SFTTrainer
		from trl.trainer.sft_config import SFTConfig
		from transformers import TrainerCallback

		# Custom callback for TensorBoard logging
		class TensorBoardCallback(TrainerCallback):
			def __init__(self, writer, tracker):
				self.writer = writer
				self.tracker = tracker
				self.step = 0

			def on_train_begin(self, args, state, control, **kwargs):
				return control

			def on_train_end(self, args, state, control, **kwargs):
				return control

			def on_log(self, args, state, control, model=None, logs=None, **kwargs):
				if logs is not None:
					# Log to TensorBoard
					for key, value in logs.items():
						if isinstance(value, (int, float)):
							self.writer.add_scalar(f"train/{key}", value, self.step)
					# Log to Weave/wandb
					self.tracker.log_metrics(logs, step=self.step)
					# Update step counter
					if "loss" in logs:
						self.step += 1

		callback = TensorBoardCallback(self.tensorboard_writer, self.tracker)

		trainer = SFTTrainer(
			model=model,
			train_dataset=train_ds,
			eval_dataset=eval_ds,
			processing_class=tokenizer,
			formatting_func=self._formatting_func,
			args=SFTConfig(
				output_dir=self.cfg.train.output_dir,
				learning_rate=self.cfg.train.learning_rate,
				per_device_train_batch_size=self.cfg.train.per_device_train_batch_size,
				gradient_accumulation_steps=self.cfg.train.gradient_accumulation_steps,
				num_train_epochs=self.cfg.train.num_train_epochs,
				warmup_steps=self.cfg.train.warmup_steps,
				logging_steps=self.cfg.train.logging_steps,
				eval_strategy=self.cfg.train.evaluation_strategy,
				eval_steps=self.cfg.train.eval_steps,
				save_strategy=self.cfg.train.save_strategy,
				save_steps=self.cfg.train.save_steps,
				fp16=False,
				bf16=self.cfg.train.bf16,
				dataloader_pin_memory=self.cfg.train.dataloader_pin_memory,
				remove_unused_columns=self.cfg.train.remove_unused_columns,
				gradient_checkpointing=self.cfg.train.gradient_checkpointing,
				dataloader_num_workers=self.cfg.train.dataloader_num_workers,
				group_by_length=self.cfg.train.group_by_length,
				weight_decay=self.cfg.train.weight_decay,
				max_grad_norm=self.cfg.train.max_grad_norm,
				seed=self.cfg.train.seed,
				report_to=["tensorboard"],
				logging_dir=tensorboard_log_dir,
				use_mps_device=True,
				max_length=self.cfg.data.max_seq_length,
				packing=False,  # Disable packing to avoid Flash Attention requirement
			),
		)

		# Add callback to trainer
		trainer.add_callback(callback)

		self.tracker.log_params({
			"model": self.cfg.train.model_id,
			"rank": self.cfg.qlora.rank,
			"alpha": self.cfg.qlora.alpha,
			"dropout": self.cfg.qlora.dropout,
			"learning_rate": self.cfg.train.learning_rate,
			"epochs": self.cfg.train.num_train_epochs,
		})

		print(f"🚀 Starting training with TensorBoard logging at: {tensorboard_log_dir}")
		print(f"📊 View training progress with: tensorboard --logdir {tensorboard_log_dir}")

		trainer.train()
		trainer.save_model(self.cfg.train.output_dir)
		self.tracker.log_artifact(self.cfg.train.output_dir, name="fine_tuned_model", type_="model")

		# Close TensorBoard writer
		if self.tensorboard_writer:
			self.tensorboard_writer.close()
			print(f"✅ Training complete! TensorBoard logs saved to: {tensorboard_log_dir}")
