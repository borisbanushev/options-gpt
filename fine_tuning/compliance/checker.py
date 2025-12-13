from __future__ import annotations

from typing import Iterable, Dict

from ..data.schemas import TrainingExample


class FinancialComplianceChecker:
	STANDARD = (
		"This is not financial advice. Options trading involves significant risk of loss. "
		"Past performance does not guarantee future results. Please consult with a qualified "
		"financial advisor before making investment decisions."
	)

	def __init__(self) -> None:
		self.violations: Dict[str, int] = {}

	def check(self, examples: Iterable[TrainingExample]) -> None:
		for ex in examples:
			if "This is not financial advice" not in ex.output:
				self.violations["missing_disclaimer"] = self.violations.get("missing_disclaimer", 0) + 1

	def inject_disclaimer(self, text: str) -> str:
		if "This is not financial advice" in text:
			return text
		return f"{text.strip()}\n\n{self.STANDARD}"
