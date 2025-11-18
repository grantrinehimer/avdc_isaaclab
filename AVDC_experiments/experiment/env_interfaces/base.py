from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


class EnvAdapter:
    """Abstract adapter that exposes the signals required by MyPolicy_CL."""

    def __init__(self, *, name: str | None = None):
        self.name = name or self.__class__.__name__

    @property
    def raw_env(self):
        """Return the wrapped environment instance (gym.Env, Isaac env, etc.)."""
        raise NotImplementedError("Sub-classes must expose the underlying environment.")

    def reset(self, seed: int | None = None):
        """Reset the underlying environment."""
        raise NotImplementedError

    def step(self, action):
        """Forward an action to the environment."""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Perception hooks
    # -------------------------------------------------------------------------
    def fetch_rgbd(
        self,
        camera_name: str,
        resolution: Tuple[int, int],
        *,
        depth: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return an (rgb, depth) tuple for the specified camera."""
        raise NotImplementedError

    def fetch_intrinsics(
        self,
        camera_name: str,
        resolution: Tuple[int, int],
    ) -> np.ndarray:
        """Return the 3x4 projection matrix used by get_transforms."""
        raise NotImplementedError

    def fetch_segmentation(
        self,
        camera_name: str,
        resolution: Tuple[int, int],
        seg_ids: Iterable[int],
    ) -> np.ndarray:
        """Return a binary segmentation mask aligned with fetch_rgbd."""
        raise NotImplementedError

    def close(self):
        """Close any simulator handles (optional)."""
        try:
            env = self.raw_env
        except NotImplementedError:
            env = None
        if env is not None and hasattr(env, "close"):
            env.close()

