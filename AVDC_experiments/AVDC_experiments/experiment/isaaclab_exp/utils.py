from __future__ import annotations

from typing import Iterable, Sequence

import cv2
import numpy as np
import torch


def _resolve_scene(env):
    """Walk common wrappers until we find the IsaacLab scene object."""
    visited = set()
    stack = [env]
    while stack:
        current = stack.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        scene = getattr(current, "scene", None)
        if scene is not None:
            return scene
        for attr in ("unwrapped", "env", "_env", "_wrapped_env"):
            if hasattr(current, attr):
                child = getattr(current, attr)
                if child is not current:
                    stack.append(child)
    raise RuntimeError("Unable to locate an IsaacLab scene on the provided environment.")


def _get_camera_sensor(env, camera_name: str):
    scene = _resolve_scene(env)
    sensors = getattr(scene, "sensors", None)
    if not sensors or camera_name not in sensors:
        raise ValueError(f"Camera sensor '{camera_name}' is not registered on the scene.")
    return sensors[camera_name]


def _to_numpy(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    if isinstance(tensor, np.ndarray):
        return tensor
    return np.asarray(tensor)


def get_camera_frame(env, camera_name: str, env_index: int = 0):
    """Return RGB, depth, segmentation, and intrinsics for a given IsaacLab camera."""
    sensor = _get_camera_sensor(env, camera_name)
    data = sensor.data
    outputs = data.output

    if "rgb" not in outputs:
        raise ValueError(f"Camera '{camera_name}' does not publish RGB data.")
    rgb = _to_numpy(outputs["rgb"][env_index])

    depth_tensor = outputs.get("depth")
    if depth_tensor is None:
        raise ValueError(f"Camera '{camera_name}' does not publish depth data.")
    depth = _to_numpy(depth_tensor[env_index])[..., 0]

    if "instance_segmentation_fast" not in outputs:
        raise ValueError(f"Camera '{camera_name}' does not publish instance segmentation data.")
    seg = _to_numpy(outputs["instance_segmentation_fast"][env_index])[..., 0]

    intrinsics = _to_numpy(data.intrinsic_matrices[env_index])
    position = _to_numpy(data.pos_w[env_index])
    quat_world = _to_numpy(data.quat_w_world[env_index])

    seg_info = None
    if data.info and len(data.info) > env_index:
        seg_info = data.info[env_index].get("instance_segmentation_fast")

    return {
        "rgb": rgb,
        "depth": depth,
        "segmentation": seg,
        "intrinsics": intrinsics,
        "position": position,
        "quat_world": quat_world,
        "seg_info": seg_info,
    }


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert (w, x, y, z) quaternion into a rotation matrix."""
    w, x, y, z = quat
    n = w * w + x * x + y * y + z * z
    if n < 1e-8:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def get_cmat(env, camera_name: str, env_index: int = 0):
    """Return a 3x4 camera matrix that maps world coordinates to image coordinates."""
    frame = get_camera_frame(env, camera_name, env_index)
    intrinsics = frame["intrinsics"]
    position = frame["position"]
    quat_world = frame["quat_world"]

    rotation_wc = _quat_to_matrix(quat_world)
    rotation_cw = rotation_wc.T
    translation_cw = -rotation_cw @ position.reshape(3, 1)
    extrinsic = np.concatenate([rotation_cw, translation_cw], axis=1)
    return intrinsics @ extrinsic


def _flatten_labels(label_entry) -> list[str]:
    if label_entry is None:
        return []
    if isinstance(label_entry, dict):
        values = []
        for key in ("semanticLabel", "semantic_label", "label", "class", "name", "primPath", "prim_path"):
            if key in label_entry and label_entry[key]:
                values.append(str(label_entry[key]))
        if "labels" in label_entry:
            values.extend(_flatten_labels(label_entry["labels"]))
        if not values:
            values.append(str(label_entry))
        return values
    if isinstance(label_entry, (list, tuple, set)):
        values = []
        for item in label_entry:
            values.extend(_flatten_labels(item))
        return values
    return [str(label_entry)]


def _match_instance_ids(seg_info, match_terms: Iterable[str]) -> set[int]:
    if not match_terms or not seg_info:
        return set()

    def _iter_mappings():
        if isinstance(seg_info, dict):
            if "idToLabels" in seg_info:
                yield seg_info["idToLabels"]
            if "info" in seg_info and isinstance(seg_info["info"], dict):
                yield seg_info["info"]

    matched = set()
    lowered_terms = [term.lower() for term in match_terms if term]
    if not lowered_terms:
        return matched

    for mapping in _iter_mappings():
        if not isinstance(mapping, dict):
            continue
        for key, labels in mapping.items():
            label_strings = _flatten_labels(labels)
            if any(any(term in lbl.lower() for term in lowered_terms) for lbl in label_strings):
                try:
                    matched.add(int(key))
                except (TypeError, ValueError):
                    continue
    return matched


def get_seg(
    env,
    camera_name: str,
    resolution: tuple[int, int] | None = None,
    seg_ids: Sequence[int] | None = None,
    target_terms: Sequence[str] = ("Object",),
    env_index: int = 0,
) -> np.ndarray:
    """Return a binary mask (uint8, 0/255) for the specified camera frame."""
    frame = get_camera_frame(env, camera_name, env_index)
    seg = frame["segmentation"]
    seg_info = frame["seg_info"]

    if seg_ids:
        target_ids = set(seg_ids)
    else:
        target_ids = _match_instance_ids(seg_info, target_terms)

    if target_ids:
        mask = np.isin(seg, list(target_ids))
    else:
        # Fallback to any non-zero segmentation id.
        mask = seg != 0

    mask = mask.astype(np.uint8) * 255
    if resolution is not None:
        width, height = resolution
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    mask = cv2.medianBlur(mask, 3)
    return mask


def collect_camera_frame(env, camera_name: str, env_index: int = 0):
    """Convenience wrapper that only returns RGB + depth as numpy arrays."""
    frame = get_camera_frame(env, camera_name, env_index)
    return frame["rgb"], frame["depth"]


def sample_n_frames(frames, n):
    new_vid_ind = [int(i * len(frames) / (n - 1)) for i in range(n - 1)] + [len(frames) - 1]
    return np.array([frames[i] for i in new_vid_ind])


def get_scene(env):
    """Public access to the resolved IsaacLab scene."""
    return _resolve_scene(env)


def get_camera_sensor(env, camera_name: str):
    """Public helper that returns the live camera sensor instance."""
    return _get_camera_sensor(env, camera_name)