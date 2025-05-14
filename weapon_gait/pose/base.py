from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

class PoseExtractor(ABC):
    """Abstract interface every pose back‑end must implement."""

    @abstractmethod
    def extract(self, video_path: Path) -> np.ndarray:  # (T, N, 3)
        """Return landmark tensor; NaNs for missing frames."""

    @property
    @abstractmethod
    def name(self) -> str:  # friendly identifier
        ...
