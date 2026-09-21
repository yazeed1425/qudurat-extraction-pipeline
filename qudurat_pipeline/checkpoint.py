"""Simple per-skill checkpoint file so a run can resume after interruption.

progress_<skill>.txt holds one completed page number per line. A page is
appended only after it has been fully extracted, embedded, and inserted, so a
crash mid-page never marks it done.
"""

from __future__ import annotations

from pathlib import Path


class Checkpoint:
    def __init__(self, skill: str, directory: Path = Path(".")):
        self.path = directory / f"progress_{skill}.txt"
        self._completed: set[int] = set()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.isdigit():
                    self._completed.add(int(line))

    def is_done(self, page_number: int) -> bool:
        return page_number in self._completed

    def mark_done(self, page_number: int) -> None:
        self._completed.add(page_number)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(f"{page_number}\n")
