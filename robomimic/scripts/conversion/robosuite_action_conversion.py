import numpy as np
from scipy.spatial.transform import Rotation


def inverse_scale_action(
    physical_delta,
    input_min,
    input_max,
    output_min,
    output_max,
):
    """Invert robosuite Controller.scale_action and expose saturation."""
    physical_delta = np.asarray(physical_delta, dtype=float)
    input_min = np.asarray(input_min, dtype=float)
    input_max = np.asarray(input_max, dtype=float)
    output_min = np.asarray(output_min, dtype=float)
    output_max = np.asarray(output_max, dtype=float)

    if not (
        physical_delta.shape
        == input_min.shape
        == input_max.shape
        == output_min.shape
        == output_max.shape
    ):
        raise ValueError("action scaling arrays must have matching shapes")
    if np.any(input_max <= input_min) or np.any(output_max <= output_min):
        raise ValueError("action scaling ranges must be strictly ordered")

    representable_mask = (physical_delta >= output_min) & (
        physical_delta <= output_max
    )
    clipped = np.clip(physical_delta, output_min, output_max)

    input_mid = (input_max + input_min) / 2.0
    output_mid = (output_max + output_min) / 2.0
    inverse_scale = (input_max - input_min) / (output_max - output_min)
    native = (clipped - output_mid) * inverse_scale + input_mid
    native = np.clip(native, input_min, input_max)
    return native, representable_mask


def scale_action(native_action, input_min, input_max, output_min, output_max):
    """Reference implementation of robosuite's affine action scaling."""
    native_action = np.asarray(native_action, dtype=float)
    input_min = np.asarray(input_min, dtype=float)
    input_max = np.asarray(input_max, dtype=float)
    output_min = np.asarray(output_min, dtype=float)
    output_max = np.asarray(output_max, dtype=float)

    clipped = np.clip(native_action, input_min, input_max)
    input_mid = (input_max + input_min) / 2.0
    output_mid = (output_max + output_min) / 2.0
    scale = (output_max - output_min) / (input_max - input_min)
    return (clipped - input_mid) * scale + output_mid


def physical_delta_from_absolute_pose(
    absolute_pose,
    baseline_position,
    baseline_orientation,
):
    """Invert robosuite OSC's achieved-pose delta composition."""
    absolute_pose = np.asarray(absolute_pose, dtype=float)
    baseline_position = np.asarray(baseline_position, dtype=float)
    baseline_orientation = np.asarray(baseline_orientation, dtype=float)

    if absolute_pose.shape != (6,):
        raise ValueError("absolute_pose must have shape (6,)")
    if baseline_position.shape != (3,):
        raise ValueError("baseline_position must have shape (3,)")
    if baseline_orientation.shape != (3, 3):
        raise ValueError("baseline_orientation must have shape (3, 3)")

    goal_position = absolute_pose[:3]
    goal_orientation = Rotation.from_rotvec(absolute_pose[3:6]).as_matrix()
    position_delta = goal_position - baseline_position

    # robosuite OSC uses R_goal = R_delta @ R_baseline.
    orientation_delta = goal_orientation @ baseline_orientation.T
    rotation_delta = Rotation.from_matrix(orientation_delta).as_rotvec()
    return np.concatenate([position_delta, rotation_delta])


def compose_delta_with_pose(
    baseline_position,
    baseline_orientation,
    physical_delta,
):
    """Forward OSC pose composition, used for round-trip verification."""
    baseline_position = np.asarray(baseline_position, dtype=float)
    baseline_orientation = np.asarray(baseline_orientation, dtype=float)
    physical_delta = np.asarray(physical_delta, dtype=float)

    goal_position = baseline_position + physical_delta[:3]
    delta_orientation = Rotation.from_rotvec(
        physical_delta[3:6]
    ).as_matrix()
    goal_orientation = delta_orientation @ baseline_orientation
    return np.concatenate(
        [
            goal_position,
            Rotation.from_matrix(goal_orientation).as_rotvec(),
        ]
    )
