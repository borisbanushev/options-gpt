from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class DataConfig:
	train_jsonl_path: str
	val_jsonl_path: Optional[str] = None
	max_seq_length: int = 2048
	shuffle: bool = True
	seed: int = 42


@dataclass
class QLoRAConfig:
	rank: int = 32
	alpha: int = 64
	dropout: float = 0.1
	bias: str = "none"
	target_modules: List[str] = field(default_factory=lambda: [
		"q_proj",
		"k_proj",
		"v_proj",
		"o_proj",
		"gate_proj",
		"up_proj",
		"down_proj",
	])
	task_type: str = "CAUSAL_LM"


@dataclass
class TrainConfig:
	model_id: str
	output_dir: str = "./artifacts/fine_tuned"
	learning_rate: float = 2e-4
	per_device_train_batch_size: int = 1
	gradient_accumulation_steps: int = 8
	num_train_epochs: int = 2
	warmup_steps: int = 100
	logging_steps: int = 10
	evaluation_strategy: str = "steps"
	eval_steps: int = 100
	save_strategy: str = "steps"
	save_steps: int = 500
	fp16: bool = False
	bf16: bool = True
	dataloader_pin_memory: bool = False
	remove_unused_columns: bool = False
	gradient_checkpointing: bool = True
	dataloader_num_workers: int = 0
	group_by_length: bool = True
	weight_decay: float = 0.01
	max_grad_norm: float = 1.0
	seed: int = 42


@dataclass
class TrackingConfig:
	project: str = "options-trading-finetuning"
	entity: Optional[str] = None
	enable_weave: bool = True
	enable_wandb: bool = True
	metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class FineTuningConfig:
	data: DataConfig
	qlora: QLoRAConfig
	train: TrainConfig
	tracking: TrackingConfig
