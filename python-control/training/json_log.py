"""Full-precision machine-readable twin of the human training log.

`HumanOutputFormat` prints floats with three significant digits, so the table in
`training.log` cannot be plotted without loss. This writer appends one JSON
object per `logger.dump()` instead.
"""

import json
import numbers
from pathlib import Path

from stable_baselines3.common.logger import KVWriter


class JsonlOutputFormat(KVWriter):
    """Write every numeric logged value as one JSON line per dump."""

    def __init__(self, filename: str) -> None:
        self.path = Path(filename)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # same reset semantics as HumanOutputFormat: one file per run
        self.file = self.path.open("w")

    def write(self, key_values, key_excluded, step: int = 0) -> None:
        record = {}
        for key, value in key_values.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, numbers.Integral):
                record[key] = int(value)
            elif isinstance(value, numbers.Real):
                record[key] = float(value)
        if not record:
            return
        self.file.write(json.dumps(record) + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()
