from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple

from pydantic import ValidationError

from .schemas import TrainingExample, DatasetStats


class LocalDataProcessor:
	"""Process and validate local JSONL files for options trading training data."""

	def __init__(self) -> None:
		pass

	@staticmethod
	def _iter_jsonl(path: str) -> Iterable[dict]:
		"""Iterate over JSONL file, yielding each JSON object."""
		with open(path, "r", encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				yield json.loads(line)

	def validate_and_write_jsonl(self, records: Iterable[dict], out_path: str) -> DatasetStats:
		"""Validate records and write clean JSONL output with statistics."""
		stats = DatasetStats(num_examples=0)
		Path(out_path).parent.mkdir(parents=True, exist_ok=True)
		
		with open(out_path, "w", encoding="utf-8") as out:
			for rec in records:
				try:
					obj = TrainingExample.model_validate(rec)
				except ValidationError as e:
					stats.invalid_examples += 1
					print(f"Validation error: {e}")
					continue
				
				json_str = obj.model_dump_json()
				out.write(json_str + "\n")
				stats.num_examples += 1
				
				# Count by strategy type
				if obj.metadata.strategy_type == "covered_calls":
					stats.num_covered_calls += 1
				elif obj.metadata.strategy_type == "leaps":
					stats.num_leaps += 1
				elif obj.metadata.strategy_type == "0dte":
					stats.num_0dte += 1
		
		return stats

	def process_jsonl(self, in_path: str, out_path: str) -> DatasetStats:
		"""Process and validate a JSONL file, writing cleaned output."""
		records = self._iter_jsonl(in_path)
		return self.validate_and_write_jsonl(records, out_path)

	def split_dataset(self, in_path: str, train_path: str, val_path: str, val_ratio: float = 0.1) -> Tuple[DatasetStats, DatasetStats]:
		"""Split a JSONL file into train/validation sets."""
		records = list(self._iter_jsonl(in_path))
		
		# Simple split by strategy type to ensure balanced validation
		by_strategy = {}
		for rec in records:
			try:
				obj = TrainingExample.model_validate(rec)
				strategy = obj.metadata.strategy_type
				if strategy not in by_strategy:
					by_strategy[strategy] = []
				by_strategy[strategy].append(rec)
			except ValidationError:
				continue
		
		train_records = []
		val_records = []
		
		for strategy, strategy_records in by_strategy.items():
			split_idx = int(len(strategy_records) * (1 - val_ratio))
			train_records.extend(strategy_records[:split_idx])
			val_records.extend(strategy_records[split_idx:])
		
		train_stats = self.validate_and_write_jsonl(train_records, train_path)
		val_stats = self.validate_and_write_jsonl(val_records, val_path)
		
		return train_stats, val_stats

	def create_sample_data(self, output_path: str, num_examples: int = 10) -> None:
		"""Create sample training data for testing purposes."""
		import datetime
		
		sample_data = []
		strategies = ["covered_calls", "leaps", "0dte"]
		market_conditions = ["bull", "bear", "sideways", "volatile"]
		difficulties = ["easy", "medium", "hard"]
		
		for i in range(num_examples):
			strategy = strategies[i % len(strategies)]
			market = market_conditions[i % len(market_conditions)]
			difficulty = difficulties[i % len(difficulties)]
			
			sample = {
				"instruction": f"Analyze this {strategy.replace('_', ' ')} options strategy",
				"input": f"Stock: AAPL, Price: $150, Market: {market}, Account: $10,000",
				"output": f"Based on the {market} market conditions, here's a {strategy.replace('_', ' ')} strategy recommendation...",
				"metadata": {
					"strategy_type": strategy,
					"market_condition": market,
					"difficulty": difficulty,
					"timestamp": datetime.datetime.now().isoformat()
				}
			}
			sample_data.append(sample)
		
		with open(output_path, "w", encoding="utf-8") as f:
			for sample in sample_data:
				f.write(json.dumps(sample) + "\n")
