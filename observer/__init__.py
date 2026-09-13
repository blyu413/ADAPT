"""Standalone momentum observer; Torch is needed only for batched training."""

from .momo import MomentumObserver

__all__ = ["MomentumObserver"]
