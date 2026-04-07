from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

Polygon = list[tuple[int, int]]

@dataclass
class OCRBox:
    polygon: Polygon
    bbox: tuple[int, int, int, int]
    text: str = ""
    confidence: float = 0.0

@dataclass
class TextGroup:
    group_id: str
    box_indices: list[int]
    bbox: tuple[int, int, int, int]
    text: str

@dataclass
class OCRResult:
    boxes: list[OCRBox]
    groups: list[TextGroup]
    provider: str
    degraded: bool
    text_reliable: bool
    debug: dict[str, Any]

@dataclass
class RegionDecision:
    region_id: int
    bbox: tuple[int, int, int, int]
    complexity: str
    method: str
    accepted: bool
    reasons: list[str] = field(default_factory=list)

@dataclass
class PipelineResult:
    output_rgb: np.ndarray
    overlay_rgb: np.ndarray
    diagnostics: dict[str, Any]
    selection_mask: np.ndarray | None = None
    execution_mask: np.ndarray | None = None
    protect_mask: np.ndarray | None = None
