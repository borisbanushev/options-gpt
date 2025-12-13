from __future__ import annotations

import os
from typing import Any, Dict

import weave
import wandb


class WeaveTracker:
	def __init__(self, project: str, entity: str | None = None, enable_wandb: bool = True) -> None:
		self.project = project
		self.entity = entity
		self.enable_wandb = enable_wandb
		weave.init(project)
		if enable_wandb:
			wandb.init(project=project, entity=entity, config={"tool": "unsloth_q_lora"})

	@weave.op()
	def log_params(self, params: Dict[str, Any]) -> None:
		if self.enable_wandb:
			wandb.config.update(params, allow_val_change=True)

	@weave.op()
	def log_metrics(self, metrics: Dict[str, float], step: int | None = None) -> None:
		if self.enable_wandb:
			wandb.log(metrics, step=step)

	@weave.op()
	def log_artifact(self, path: str, name: str, type_: str = "model") -> None:
		if self.enable_wandb and os.path.exists(path):
			art = wandb.Artifact(name=name, type=type_)
			art.add_dir(path)
			wandb.log_artifact(art)
