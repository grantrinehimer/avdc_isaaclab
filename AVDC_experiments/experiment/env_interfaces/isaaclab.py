from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Tuple

from .base import EnvAdapter

LOGGER = logging.getLogger(__name__)


@dataclass
class IsaacAppHandles:
    """Container for Isaac Sim handles so the adapter can clean up gracefully."""

    app_launcher: object
    simulation_app: object


class IsaacLabEnvAdapter(EnvAdapter):
    """Skeleton adapter that will expose RGB-D data once the sensors are ready."""

    def __init__(
        self,
        *,
        env,
        handles: IsaacAppHandles,
        camera_setup: Dict[str, Tuple[int, int]] | None = None,
        task: str | None = None,
    ):
        super().__init__(name=task or "isaaclab")
        self._env = env
        self._handles = handles
        self._camera_setup = camera_setup or {}
        self._task = task

    # ------------------------------------------------------------------ #
    # EnvAdapter interface
    # ------------------------------------------------------------------ #
    @property
    def raw_env(self):
        return self._env

    def reset(self, seed: int | None = None):
        if seed is not None and hasattr(self._env, "seed"):
            self._env.seed(seed)
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)

    def fetch_rgbd(self, camera_name, resolution, *, depth: bool = True):
        raise NotImplementedError(
            "IsaacLab RGB-D capture is not wired yet. "
            "Add camera sensors and route their outputs through this adapter."
        )

    def fetch_intrinsics(self, camera_name, resolution):
        raise NotImplementedError(
            "Camera intrinsics are undefined. Export the projection matrices from cs6758_custom."
        )

    def fetch_segmentation(self, camera_name, resolution, seg_ids):
        raise NotImplementedError(
            "Segmentation masks are not available yet. Provide instance IDs once the sensors exist."
        )

    def close(self):
        super().close()
        if self._handles.simulation_app is not None:
            LOGGER.info("Closing Isaac Sim app.")
            self._handles.simulation_app.close()


def build_env_adapter(
    *,
    task: str,
    camera_setup: Dict[str, Tuple[int, int]] | None = None,
    device: str | None = None,
    num_envs: int | None = None,
    disable_fabric: bool = False,
    app_launcher_kwargs: Dict | None = None,
):
    """Launch Isaac Sim, build the gym env, and wrap it in an adapter.

    Parameters
    ----------
    task: str
        Fully-qualified Isaac task name (e.g. ``avdc_isaaclab::cs6758_custom/FrankaBasketCubeEnv``).
    camera_setup: dict
        Mapping ``camera_name -> (width, height)`` for convenience defaults.
    device: str
        Torch device string forwarded to ``parse_env_cfg``.
    num_envs: int
        Number of vectorized envs. MyPolicy_CL currently assumes ``1``.
    disable_fabric: bool
        Mirrors the CLI flag from existing scripts.
    app_launcher_kwargs: dict
        Keyword dict converted into a ``SimpleNamespace`` that mimics the argparse
        Namespace consumed by ``AppLauncher`` (e.g. ``{\"headless\": True}``).
    """

    try:
        from isaaclab.app import AppLauncher
        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        import avdc_isaaclab.tasks  # noqa: F401
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError(
            "IsaacLab is not available in this environment. "
            "Install Isaac Sim and ensure isaaclab is on PYTHONPATH."
        ) from exc

    app_launcher_kwargs = app_launcher_kwargs or {}
    app_cli = SimpleNamespace(**app_launcher_kwargs)
    app_launcher = AppLauncher(app_cli)
    simulation_app = app_launcher.app

    env_cfg = parse_env_cfg(
        task,
        device=device,
        num_envs=num_envs,
        use_fabric=not disable_fabric,
    )
    env = gym.make(task, cfg=env_cfg)

    handles = IsaacAppHandles(app_launcher=app_launcher, simulation_app=simulation_app)
    adapter = IsaacLabEnvAdapter(
        env=env,
        handles=handles,
        camera_setup=camera_setup,
        task=task,
    )
    return adapter

