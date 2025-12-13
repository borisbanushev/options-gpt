from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import typer
from rich import print
from dotenv import load_dotenv

from ..configs.config import DataConfig, QLoRAConfig, TrainConfig, TrackingConfig, FineTuningConfig
from ..tracking.weave_tracker import WeaveTracker
from ..validation.validator import DataValidator
from ..data.processor import LocalDataProcessor
from ..training.mlx_trainer import MLXTrainer

load_dotenv("config.env")

app = typer.Typer()


@app.command()
def run(
    model_id: str = None,
    train_jsonl: str = None,
    val_jsonl: str = None,
    output_dir: str = None,
):
    model_id = model_id or os.getenv("MODEL_ID", "openai/gpt-oss-20b")
    train_jsonl = train_jsonl or os.getenv("TRAIN_DATA_PATH", "./data/train.jsonl")
    val_jsonl = val_jsonl or os.getenv("VAL_DATA_PATH", "")
    output_dir = output_dir or os.getenv("OUTPUT_DIR", "./artifacts/fine_tuned")

    data_cfg = DataConfig(train_jsonl_path=train_jsonl, val_jsonl_path=val_jsonl or None)
    qlora_cfg = QLoRAConfig()
    train_cfg = TrainConfig(model_id=model_id, output_dir=output_dir)
    tracking_cfg = TrackingConfig()
    cfg = FineTuningConfig(data=data_cfg, qlora=qlora_cfg, train=train_cfg, tracking=tracking_cfg)

    tracker = WeaveTracker(
        project=os.getenv("WEAVE_PROJECT", "options-trading-finetuning"),
        entity=os.getenv("WANDB_ENTITY"),
        enable_wandb=bool(os.getenv("WANDB_API_KEY")),
    )

    validator = DataValidator()
    proc = LocalDataProcessor()

    train_records = [json.loads(l) for l in Path(train_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    train_stats, train_examples = validator.validate_examples(train_records)
    print({"train_stats": train_stats.model_dump()})

    if cfg.data.val_jsonl_path:
        val_records = [json.loads(l) for l in Path(cfg.data.val_jsonl_path).read_text(encoding="utf-8").splitlines() if l.strip()]
        val_stats, val_examples = validator.validate_examples(val_records)
        print({"val_stats": val_stats.model_dump()})
    else:
        val_examples = None

    trainer = MLXTrainer(cfg=cfg, tracker=tracker)
    trainer.train(train_examples, val_examples)


if __name__ == "__main__":
    app()


