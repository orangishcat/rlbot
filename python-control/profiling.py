"""Wall-clock timers for each phase of the training loop.

Phases accumulate between log dumps. `GdRecurrentPPO.dump_logs` writes them to
the logger as `profile/<phase>_s` (seconds) and `profile/<phase>_calls`, so they
land in metrics.jsonl and plot_training.py plots them like any other metric.

Nested phases overlap: `bridge/poll` runs inside `env/bridge_step`, and
`rollout/total` contains the env, policy, and checkpoint phases.
"""

import time
from collections import defaultdict
from contextlib import contextmanager


class Profiler:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.seconds: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)

    @contextmanager
    def phase(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] += time.perf_counter() - start
            self.calls[name] += 1

    def flush(self, logger) -> None:
        for name, seconds in self.seconds.items():
            logger.record(f"profile/{name}_s", seconds)
            logger.record(f"profile/{name}_calls", self.calls[name])
        self.reset()


profiler = Profiler()
