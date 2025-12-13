from __future__ import annotations

import psutil
from dataclasses import dataclass


@dataclass
class MemoryStats:
	total_gb: float
	used_gb: float
	percent: float


def get_unified_memory_usage() -> MemoryStats:
	vm = psutil.virtual_memory()
	return MemoryStats(total_gb=vm.total / 1e9, used_gb=vm.used / 1e9, percent=vm.percent)
