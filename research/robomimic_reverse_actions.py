from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


class ReverseStatus(str, Enum):
    EXACT = "exact"
    SATURATED = "saturated"


@dataclass(frozen=True)
class OSCActionContract:
    input_min: np.ndarray
    input_max: np.ndarray
    output_min: np.ndarray
    output_max: np.ndarray
    goal_update_mode: str = "achieved"

    def __post_init__(self):
        input_min = np.asarray(self.input_min, dtype=float)
        input_max = np.asarray(self.input_max, dtype=float)
        output_min = np.asarray(self.output_min, dtype=float)
        output_max = np.asarray(self.output_max, dtype=float)
        for name, value in (
            ("input_min", input_min),
            ("input_max", input_max),
            ("output_min", output_min),
            ("output_max", output_max),
        ):
            if value.shape != (6,):
                raise ValueError(f"{name} must have shape (6,)")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
        if np.any(input_max <= input_min) or np.any(output_max <= output_min):
            raise ValueError("action scaling ranges must be strictly ordered")
        if self.goal_update_mode not in {"achieved", "desired"}:
            raise ValueError("goal_update_mode must be achieved or desired")
        object.__setattr__(self, "input_min", input_min)
        object.__setattr__(self, "input_max", input_max)
        object.__setattr__(self, "output_min", output_min)
        object.__setattr__(self, "output_max", output_max)


@dataclass(frozen=True)
class ReverseActionCertificate:
    status: ReverseStatus
    native_delta_action: np.ndarray
    physical_delta: np.ndarray
    reconstructed_absolute_action: np.ndarray
    absolute_residual: np.ndarray
    residual_norm: float
    representable: bool
    saturation_mask: np.ndarray
    reason: str


def scale_native_delta(native_delta: np.ndarray, contract: OSCActionContract) -> np.ndarray:
    native = np.asarray(native_delta, dtype=float)
    if native.shape != (6,):
        raise ValueError("native_delta must have shape (6,)")
    clipped = np.clip(native, contract.input_min, contract.input_max)
    input_mid = 0.5 * (contract.input_max + contract.input_min)
    output_mid = 0.5 * (contract.output_max + contract.output_min)
    scale = (contract.output_max - contract.output_min) / (
        contract.input_max - contract.input_min
    )
    return (clipped - input_mid) * scale + output_mid


def inverse_scale_physical_delta(
    physical_delta: np.ndarray,
    contract: OSCActionContract,
) -> tuple[np.ndarray, bool, np.ndarray]:
    physical = np.asarray(physical_delta, dtype=float)
    if physical.shape != (6,):
        raise ValueError("physical_delta must have shape (6,)")

    representable_mask = (physical >= contract.output_min) & (
        physical <= contract.output_max
    )
    clipped = np.clip(physical, contract.output_min, contract.output_max)
    input_mid = 0.5 * (contract.input_max + contract.input_min)
    output_mid = 0.5 * (contract.output_max + contract.output_min)
    inverse_scale = (contract.input_max - contract.input_min) / (
        contract.output_max - contract.output_min
    )
    native = (clipped - output_mid) * inverse_scale + input_mid
    native = np.clip(native, contract.input_min, contract.input_max)
    return native, bool(np.all(representable_mask)), ~representable_mask


def compose_delta_with_pose(
    baseline_position: np.ndarray,
    baseline_orientation: np.ndarray,
    physical_delta: np.ndarray,
) -> np.ndarray:
    pos = np.asarray(baseline_position, dtype=float)
    ori = np.asarray(baseline_orientation, dtype=float)
    delta = np.asarray(physical_delta, dtype=float)
    if pos.shape != (3,) or ori.shape != (3, 3) or delta.shape != (6,):
        raise ValueError("pose/delta dimensions are invalid")

    goal_pos = pos + delta[:3]
    R_delta = Rotation.from_rotvec(delta[3:6]).as_matrix()
    goal_ori = R_delta @ ori
    return np.concatenate(
        [goal_pos, Rotation.from_matrix(goal_ori).as_rotvec()]
    )


def physical_delta_from_absolute_pose(
    absolute_action: np.ndarray,
    baseline_position: np.ndarray,
    baseline_orientation: np.ndarray,
) -> np.ndarray:
    absolute = np.asarray(absolute_action, dtype=float)
    pos = np.asarray(baseline_position, dtype=float)
    ori = np.asarray(baseline_orientation, dtype=float)
    if absolute.shape != (6,) or pos.shape != (3,) or ori.shape != (3, 3):
        raise ValueError("pose dimensions are invalid")

    goal_pos = absolute[:3]
    goal_ori = Rotation.from_rotvec(absolute[3:6]).as_matrix()
    delta_pos = goal_pos - pos

    # robosuite OSC composes orientation as R_goal = R_delta @ R_baseline.
    R_delta = goal_ori @ ori.T
    delta_ori = Rotation.from_matrix(R_delta).as_rotvec()
    return np.concatenate([delta_pos, delta_ori])


def absolute_pose_to_delta_action(
    absolute_action: np.ndarray,
    *,
    achieved_position: np.ndarray,
    achieved_orientation: np.ndarray,
    contract: OSCActionContract,
    desired_position: np.ndarray | None = None,
    desired_orientation: np.ndarray | None = None,
) -> ReverseActionCertificate:
    if contract.goal_update_mode == "desired":
        if desired_position is None or desired_orientation is None:
            raise ValueError(
                "desired goal-update mode requires desired_position and desired_orientation"
            )
        baseline_pos = np.asarray(desired_position, dtype=float)
        baseline_ori = np.asarray(desired_orientation, dtype=float)
    else:
        baseline_pos = np.asarray(achieved_position, dtype=float)
        baseline_ori = np.asarray(achieved_orientation, dtype=float)

    physical = physical_delta_from_absolute_pose(
        absolute_action,
        baseline_pos,
        baseline_ori,
    )
    native, representable, saturation_mask = inverse_scale_physical_delta(
        physical,
        contract,
    )
    reconstructed = compose_delta_with_pose(
        baseline_pos,
        baseline_ori,
        scale_native_delta(native, contract),
    )

    absolute = np.asarray(absolute_action, dtype=float)
    target_ori = Rotation.from_rotvec(absolute[3:6]).as_matrix()
    reconstructed_ori = Rotation.from_rotvec(reconstructed[3:6]).as_matrix()

    pos_residual = reconstructed[:3] - absolute[:3]
    ori_residual = Rotation.from_matrix(
        reconstructed_ori @ target_ori.T
    ).as_rotvec()
    residual = np.concatenate([pos_residual, ori_residual])
    residual_norm = float(np.linalg.norm(residual))

    status = ReverseStatus.EXACT if representable else ReverseStatus.SATURATED
    reason = (
        "absolute OSC goal is representable by the delta controller action chart"
        if representable
        else "absolute OSC goal requires a delta outside the controller output range"
    )
    return ReverseActionCertificate(
        status=status,
        native_delta_action=native,
        physical_delta=physical,
        reconstructed_absolute_action=reconstructed,
        absolute_residual=residual,
        residual_norm=residual_norm,
        representable=representable,
        saturation_mask=saturation_mask,
        reason=reason,
    )


def convert_robot_absolute_action(
    absolute_robot_action: np.ndarray,
    *,
    achieved_position: np.ndarray,
    achieved_orientation: np.ndarray,
    contract: OSCActionContract,
    desired_position: np.ndarray | None = None,
    desired_orientation: np.ndarray | None = None,
) -> tuple[np.ndarray, ReverseActionCertificate]:
    """Convert [abs_pos, abs_rotvec, remainder...] and preserve remainder."""
    action = np.asarray(absolute_robot_action, dtype=float)
    if action.ndim != 1 or action.shape[0] < 6:
        raise ValueError("absolute_robot_action must have at least 6 values")

    certificate = absolute_pose_to_delta_action(
        action[:6],
        achieved_position=achieved_position,
        achieved_orientation=achieved_orientation,
        contract=contract,
        desired_position=desired_position,
        desired_orientation=desired_orientation,
    )
    converted = np.concatenate([certificate.native_delta_action, action[6:]])
    return converted, certificate


def extract_osc_contract(controller: Any) -> OSCActionContract:
    """Extract the scaling/update contract from a robosuite OSC controller."""
    required = ("input_min", "input_max", "output_min", "output_max")
    missing = [field for field in required if not hasattr(controller, field)]
    if missing:
        raise TypeError(f"controller missing OSC scaling fields: {missing}")

    goal_update_mode = getattr(controller, "_goal_update_mode", "achieved")
    return OSCActionContract(
        input_min=np.asarray(controller.input_min, dtype=float),
        input_max=np.asarray(controller.input_max, dtype=float),
        output_min=np.asarray(controller.output_min, dtype=float),
        output_max=np.asarray(controller.output_max, dtype=float),
        goal_update_mode=goal_update_mode,
    )


def extract_osc_baseline_pose(controller: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract achieved OSC baseline for robosuite <1.5 and >=1.5.

    For >=1.5, the absolute action lives in the controller input reference
    frame.  For <1.5, absolute OSC goals use the world-frame ee pose.
    """
    if hasattr(controller, "update"):
        controller.update(force=True)

    if hasattr(controller, "input_ref_frame"):
        frame = controller.input_ref_frame
        if frame == "base":
            position = np.asarray(
                controller.world_to_origin_frame(controller.ref_pos),
                dtype=float,
            )
            orientation = np.asarray(
                controller.goal_origin_to_eef_pose()[:3, :3],
                dtype=float,
            )
        elif frame == "world":
            position = np.asarray(controller.ref_pos, dtype=float)
            orientation = np.asarray(controller.ref_ori_mat, dtype=float)
        else:
            raise ValueError(f"unsupported OSC input_ref_frame: {frame}")
    else:
        position = np.asarray(controller.ee_pos, dtype=float)
        orientation = np.asarray(controller.ee_ori_mat, dtype=float)

    return position, orientation
