"""Live plots for Stable-Baselines3 training metrics.

Reads the JSONL sidecar (`training/results/metrics.jsonl`, full precision) or
the human table (`training/results/training.log`), follows appends, and redraws
one panel per logged metric.

    python plot_training.py
    python plot_training.py --x total_timesteps
    python plot_training.py --metrics loss entropy_loss --once

x axis defaults to `iterations`. `--metrics` keeps only metrics whose name
contains one of the given substrings.
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

RESULTS = Path(__file__).resolve().parent / "training" / "results"
DEFAULT_LOG = RESULTS / "training.log"
DEFAULT_JSONL = RESULTS / "metrics.jsonl"
SEPARATOR = re.compile(r"^-{3,}$")
ROW = re.compile(r"^\|\s*(.*?)\s*\|\s*(.*?)\s*\|$")

X_CHOICES = ("iterations", "total_timesteps", "time_elapsed", "n_updates")


def parse_block(lines):
    """Table lines -> {group/key: float}; header rows carry no value."""
    group = None
    values = {}
    for line in lines:
        match = ROW.match(line)
        if match is None:
            continue
        key, raw = match.group(1).strip(), match.group(2).strip()
        if not raw:
            if key.endswith("/"):
                group = key.rstrip("/")
                continue
            if key:
                group = key
            continue
        if group is None:
            continue
        try:
            values[f"{group}/{key}"] = float(raw)
        except ValueError:
            pass
    return values


class Tail:
    """Incremental line tailer: byte offsets, partial lines, rewrite detection."""

    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.buffer = ""
        self.tail = b""
        self.reset = False

    def _rewind(self):
        self.offset, self.buffer, self.tail = 0, "", b""
        self.reset = True

    def read_lines(self):
        try:
            with self.path.open("rb") as handle:
                size = os.fstat(handle.fileno()).st_size
                if size < self.offset:
                    self._rewind()
                elif self.tail:
                    # appends never touch the last bytes; a rewrite does
                    handle.seek(self.offset - len(self.tail))
                    if handle.read(len(self.tail)) != self.tail:
                        self._rewind()
                handle.seek(self.offset)
                chunk = handle.read()
                self.offset = handle.tell()
        except OSError:
            return []

        if chunk:
            self.tail = (self.tail + chunk)[-512:]
        self.buffer += chunk.decode("utf-8", errors="replace")
        lines = self.buffer.split("\n")
        self.buffer = lines.pop()  # last line may still be partial
        return lines


class TableReader(Tail):
    """Yields one dict per completed `HumanOutputFormat` table."""

    def __init__(self, path):
        super().__init__(path)
        self.block = []

    def _rewind(self):
        super()._rewind()
        self.block = []

    def read(self):
        blocks = []
        for line in self.read_lines():
            line = line.strip()
            if not line:
                continue
            if SEPARATOR.match(line):
                if self.block:
                    values = parse_block(self.block)
                    if values:
                        blocks.append(values)
                    self.block = []
                continue
            self.block.append(line)
        return blocks


class JsonlReader(Tail):
    """Yields one dict per JSON line written by JsonlOutputFormat."""

    def read(self):
        records = []
        for line in self.read_lines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            values = {
                key: float(value)
                for key, value in record.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            if values:
                records.append(values)
        return records


def open_reader(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return JsonlReader(path)
    try:
        head = path.read_bytes()[:4096].lstrip()
    except OSError:
        head = b""
    if head.startswith(b"{"):
        return JsonlReader(path)
    return TableReader(path)


class Plotter:
    def __init__(self, path, x_metric, filters):
        self.tail = open_reader(path)
        self.x_key = f"time/{x_metric}"
        self.x_label = x_metric.replace("_", " ")
        self.filters = filters
        self.series = {}  # metric -> [(x, y), ...]
        self.order = []
        self.axes = {}
        self.fig = plt.figure(figsize=(13, 8), layout="constrained")
        self.fig.suptitle(f"{Path(path).name} — live ({self.x_label})")
        self._empty_note = None

    def _wanted(self, name):
        if name == self.x_key:
            return False
        return not self.filters or any(f in name for f in self.filters)

    def ingest(self):
        blocks = self.tail.read()
        added = False
        if self.tail.reset:
            # file was truncated/rewritten; blocks cover it from the start
            self.tail.reset = False
            self.series.clear()
            self.order.clear()
            self.axes.clear()
            added = True
        for block in blocks:
            x = block.get(self.x_key)
            if x is None:
                continue
            for name, value in block.items():
                if not self._wanted(name):
                    continue
                if name not in self.series:
                    self.series[name] = []
                    self.order.append(name)
                    added = True
                self.series[name].append((x, value))
        return added

    def _rebuild(self):
        self.fig.clf()
        self.axes = {}
        names = self.order
        if not names:
            self._empty_note = self.fig.text(
                0.5, 0.5, "waiting for the first training block…", ha="center", va="center"
            )
            return
        self._empty_note = None
        cols = min(4, max(1, math.ceil(math.sqrt(len(names)))))
        rows = math.ceil(len(names) / cols)
        axes = self.fig.subplots(rows, cols, squeeze=False)
        for ax in axes.flat:
            ax.remove()  # place axes explicitly in discovery order
        for i, name in enumerate(names):
            ax = self.fig.add_subplot(rows, cols, i + 1)
            ax.set_title(name, fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            if i // cols == rows - 1:
                ax.set_xlabel(self.x_label, fontsize=8)
            (line,) = ax.plot([], [], linewidth=1)
            self.axes[name] = (ax, line)
        self.fig.suptitle(f"{Path(self.tail.path).name} — live ({self.x_label})")

    def refresh(self):
        added = self.ingest()
        if added or (not self.axes and self.order):
            self._rebuild()
        for name, (ax, line) in self.axes.items():
            points = self.series[name]
            line.set_data([p[0] for p in points], [p[1] for p in points])
            ax.relim()
            ax.autoscale_view()
        return bool(self.order)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_JSONL if DEFAULT_JSONL.exists() else DEFAULT_LOG,
        help="metrics.jsonl (full precision) or training.log (table)",
    )
    parser.add_argument("--x", default="iterations", choices=X_CHOICES)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        metavar="SUBSTRING",
        help="only plot metrics whose name contains one of these",
    )
    parser.add_argument("--interval", default=1000, type=int, help="poll period in ms")
    parser.add_argument("--once", action="store_true", help="plot what exists, then exit")
    parser.add_argument("--save", type=Path, help="write a PNG and exit")
    args = parser.parse_args()

    if not args.path.exists() and not args.save:
        print(f"waiting for {args.path} to appear…", file=sys.stderr)

    plotter = Plotter(args.path, args.x, args.metrics)

    if args.once or args.save:
        plotter.refresh()
        plotter.fig.savefig(args.save or "training.png", dpi=140)
        if args.save:
            print(f"wrote {args.save}")
            return
        plt.show()
        return

    animation = FuncAnimation(
        plotter.fig, lambda _: plotter.refresh(), interval=args.interval, cache_frame_data=False
    )
    plt.show()
    del animation


if __name__ == "__main__":
    main()
