from __future__ import annotations

from datetime import datetime
from typing import Iterable, Tuple

from pydantic import ValidationError

from ..data.schemas import TrainingExample, DatasetStats


class DataValidator:
	STANDARD_DISCLAIMER_PREFIX = "This is not financial advice"

	def __init__(self) -> None:
		pass

	def validate_examples(self, records: Iterable[dict]) -> Tuple[DatasetStats, list[TrainingExample]]:
		stats = DatasetStats(num_examples=0)
		valid: list[TrainingExample] = []
		for rec in records:
			try:
				ex = TrainingExample.model_validate(rec)
			except ValidationError:
				stats.invalid_examples += 1
				continue
			# Temporal sanity (timestamp parseable)
			try:
				datetime.fromisoformat(ex.metadata.timestamp.replace("Z", "+00:00"))
			except Exception:
				stats.warnings["bad_timestamp"] = stats.warnings.get("bad_timestamp", 0) + 1
			valid.append(ex)
			stats.num_examples += 1
			st = ex.metadata.strategy_type
			if st == "covered_calls":
				stats.num_covered_calls += 1
			elif st == "leaps":
				stats.num_leaps += 1
			elif st == "0dte":
				stats.num_0dte += 1
		return stats, valid
