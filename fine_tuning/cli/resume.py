from __future__ import annotations

import typer
from rich import print

app = typer.Typer()


@app.command()
def run(checkpoint_dir: str):
	# Placeholder: In practice, load TrainingArguments and model from checkpoint_dir and resume
	print({"resuming_from": checkpoint_dir})


if __name__ == "__main__":
	app()
