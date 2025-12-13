from __future__ import annotations

import typer
from rich import print

from ..evaluation.evaluator import EleutherEvaluator

app = typer.Typer()


@app.command()
def run(model_id: str):
	evalr = EleutherEvaluator(model_id)
	std = evalr.run_standard()
	fin = evalr.run_custom_financial()
	print({"standard": std, "financial": fin})


if __name__ == "__main__":
	app()
