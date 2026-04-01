from __future__ import annotations

from configs.schema import AppConfig


def build_benchmark_plan(config: AppConfig) -> dict[str, object]:
    return {
        "batch_size": config.evaluation.benchmark_batch_size,
        "warmup_steps": config.evaluation.warmup_steps,
        "timed_steps": config.evaluation.timed_steps,
        "measure_memory": config.evaluation.measure_memory,
        "device_preference": config.runtime.device,
    }

