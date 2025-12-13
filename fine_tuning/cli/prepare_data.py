from __future__ import annotations

import typer
from rich import print

from ..data.processor import LocalDataProcessor

app = typer.Typer()


@app.command()
def process(in_path: str, out_path: str):
	"""Validate and normalize a local JSONL into a cleaned JSONL."""
	proc = LocalDataProcessor()
	stats = proc.process_jsonl(in_path, out_path)
	print({"wrote": out_path, "stats": stats.model_dump()})


@app.command()
def split(in_path: str, train_path: str, val_path: str, val_ratio: float = 0.1):
	"""Split a JSONL file into train/validation sets."""
	proc = LocalDataProcessor()
	train_stats, val_stats = proc.split_dataset(in_path, train_path, val_path, val_ratio)
	print({
		"train": train_stats.model_dump(),
		"val": val_stats.model_dump()
	})


@app.command()
def create_sample(output_path: str, num_examples: int = 10):
	"""Create sample training data for testing."""
	proc = LocalDataProcessor()
	proc.create_sample_data(output_path, num_examples)
	print(f"Created {num_examples} sample examples at {output_path}")


if __name__ == "__main__":
	app()
