"""Wall/opening expert wrapper.

This is a scaffold around the existing hard_wall/door/window graph-node models.
The model-specific checkpoint loading stays outside this lightweight package.
"""

from __future__ import annotations

from .base import PassthroughExpert


class WallOpeningExpert(PassthroughExpert):
    def __init__(self) -> None:
        super().__init__(name="wall_opening", family="boundary")
        self.default_label = "hard_wall"
