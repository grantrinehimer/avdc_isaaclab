import os
from pathlib import Path
from typing import Optional

from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_tasks.utils import get_checkpoint_path


def resolve_expert_checkpoint(
    experiment_name: str = "franka_lift",
    log_root: str | os.PathLike[str] = "logs/rsl_rl",
    load_run: Optional[str] = None,
    load_checkpoint: Optional[int] = None,
) -> str:
    """Return an absolute path to a suitable expert checkpoint for the lift_custom task.

    The function first attempts to find a locally trained checkpoint under
    ``{log_root}/{experiment_name}`` using Isaac Lab's ``get_checkpoint_path`` helper.
    If no checkpoint can be located, it falls back to Isaac Lab's published checkpoints.
    """

    log_root = Path(log_root).expanduser().resolve()
    experiment_dir = log_root / experiment_name

    if experiment_dir.exists():
        try:
            checkpoint_path = get_checkpoint_path(
                str(experiment_dir), load_run=load_run, load_checkpoint=load_checkpoint
            )
            if checkpoint_path:
                return checkpoint_path
        except (FileNotFoundError, RuntimeError):
            pass

    published_checkpoint = get_published_pretrained_checkpoint("rsl_rl", experiment_name)
    if published_checkpoint:
        return published_checkpoint

    raise FileNotFoundError(
        "Unable to locate a trained checkpoint. "
        "Please run `scripts/rsl_rl/train.py --task Isaac-Lift-Cube-Franka-Custom-IK-Rel-v0` "
        "or provide a --checkpoint path explicitly."
    )

