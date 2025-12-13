from __future__ import annotations

from pydantic import BaseModel, Field, validator
from typing import Literal, Optional, Dict


StrategyType = Literal["covered_calls", "leaps", "0dte"]


class ExampleMetadata(BaseModel):
	strategy_type: StrategyType
	market_condition: str
	difficulty: Literal["easy", "medium", "hard"]
	timestamp: str


class TrainingExample(BaseModel):
	instruction: str
	input: str
	output: str
	metadata: ExampleMetadata

	@validator("output")
	def ensure_disclaimer(cls, v: str) -> str:
		standard = (
			"This is not financial advice. Options trading involves significant risk of loss. "
			"Past performance does not guarantee future results. Please consult with a qualified "
			"financial advisor before making investment decisions."
		)
		if "This is not financial advice" not in v:
			return f"{v.strip()}\n\n{standard}"
		return v


class DatasetStats(BaseModel):
	num_examples: int
	num_covered_calls: int = 0
	num_leaps: int = 0
	num_0dte: int = 0
	invalid_examples: int = 0
	warnings: Dict[str, int] = Field(default_factory=dict)
