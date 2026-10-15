"""Regression coverage for lossless single-/dual-robot action extraction."""
import h5py
import numpy as np
import pytest

from robomimic.scripts.extract_action_dict import _action_fields, extract_action_dict


def _actions(steps, arms, width):
    matrix = np.zeros((steps, arms, width), dtype=np.float32)
    for robot_idx in range(arms):
        for step in range(steps):
            matrix[step, robot_idx, :3] = [robot_idx * 10 + step, step + 0.25, robot_idx + 0.5]
            matrix[step, robot_idx, 3:6] = [0.0, 0.0, 0.1 * (robot_idx + 1)]
            matrix[step, robot_idx, 6] = robot_idx * 2 - 1
            if width == 8:
                matrix[step, robot_idx, 7] = robot_idx + 5
    return matrix.reshape(steps, arms * width)


@pytest.mark.parametrize("num_robots,per_arm_width", [(1, 7), (1, 8), (2, 7), (2, 8)])
@pytest.mark.parametrize("include_abs", [False, True])
def test_hdf5_action_extraction_preserves_every_arm(tmp_path, num_robots, per_arm_width, include_abs):
    path = tmp_path / "robot-actions.hdf5"
    original = _actions(5, num_robots, per_arm_width)
    absolute = original.reshape(5, num_robots, per_arm_width).copy()
    # Only pose goals change; gripper and base-mode controls are invariant
    # under robomimic's delta-to-absolute conversion.
    absolute[:, :, :6] += 0.25
    absolute = absolute.reshape(original.shape)
    with h5py.File(path, "w") as f:
        demo = f.create_group("data").create_group("demo_0")
        demo.create_dataset("actions", data=original)
        if include_abs:
            demo.create_dataset("actions_abs", data=absolute)

    extract_action_dict(str(path), add_absolute_actions=include_abs)
    with h5py.File(path, "r") as f:
        group = f["data/demo_0/action_dict"]
        assert group.attrs["num_robots"] == num_robots
        for robot_idx in range(num_robots):
            owner = f"robot{robot_idx}_" if num_robots > 1 else ""
            for name, data in (("rel_", original), ("abs_", absolute)):
                if name == "abs_" and not include_abs:
                    continue
                arm = data.reshape(5, num_robots, per_arm_width)[:, robot_idx]
                np.testing.assert_array_equal(group[owner + name + "pos"][:], arm[:, :3])
                np.testing.assert_array_equal(
                    group[owner + name + "rot_axis_angle"][:], arm[:, 3:6]
                )
                six_d = group[owner + name + "rot_6d"][:]
                assert six_d.shape == (5, 6)
                assert np.isfinite(six_d).all()
            np.testing.assert_array_equal(
                group[owner + "gripper"][:],
                original.reshape(5, num_robots, per_arm_width)[:, robot_idx, 6:7],
            )
            if per_arm_width == 8:
                np.testing.assert_array_equal(
                    group[owner + "base_mode"][:],
                    original.reshape(5, num_robots, per_arm_width)[:, robot_idx, 7:8],
                )
        assert ("rel_pos" in group) == (num_robots == 1)
        assert ("gripper" in group) == (num_robots == 1)

    # Repeated extraction must be deterministic and idempotent for existing keys.
    extract_action_dict(str(path), add_absolute_actions=include_abs)


def test_stale_single_robot_action_fields_are_removed_on_dual_robot_data(tmp_path):
    path = tmp_path / "old-format.hdf5"
    with h5py.File(path, "w") as f:
        demo = f.create_group("data").create_group("demo_0")
        demo.create_dataset("actions", data=_actions(3, 2, 7))
        group = demo.create_group("action_dict")
        group.create_dataset("rel_pos", data=np.ones((3, 3)))
        group.create_dataset("gripper", data=np.ones((3, 1)))
        group.create_dataset("custom_annotation", data=np.ones(3))
    extract_action_dict(str(path), add_absolute_actions=False)
    with h5py.File(path, "r") as f:
        group = f["data/demo_0/action_dict"]
        assert "rel_pos" not in group
        assert "gripper" not in group
        assert "custom_annotation" in group
        assert "robot0_rel_pos" in group and "robot1_rel_pos" in group


@pytest.mark.parametrize("width", [0, 6, 9, 10, 13, 15, 17, 21])
def test_ambiguous_action_dimensions_fail_closed(width):
    with pytest.raises(ValueError, match="Unsupported action width"):
        _action_fields(np.zeros((3, width), dtype=np.float32), is_absolute=False)


def test_malformed_rank_and_mismatched_absolute_arm_count(tmp_path):
    with pytest.raises(ValueError, match="Expected action matrix"):
        _action_fields(np.zeros((7,), dtype=np.float32), is_absolute=True)
    path = tmp_path / "mismatched.hdf5"
    with h5py.File(path, "w") as f:
        demo = f.create_group("data").create_group("demo_0")
        demo.create_dataset("actions", data=_actions(4, 2, 7))
        demo.create_dataset("actions_abs", data=_actions(4, 1, 7))
    with pytest.raises(ValueError, match="arm counts disagree"):
        extract_action_dict(str(path), add_absolute_actions=True)
    with h5py.File(path, "r") as f:
        assert "action_dict" not in f["data/demo_0"]


@pytest.mark.parametrize("num_robots,per_arm_width", [(1, 7), (1, 8), (2, 7), (2, 8)])
def test_inconsistent_absolute_shared_controls_are_rejected_without_writes(
    tmp_path, num_robots, per_arm_width
):
    path = tmp_path / "inconsistent-shared-controls.hdf5"
    original = _actions(3, num_robots, per_arm_width)
    mismatched = original.reshape(3, num_robots, per_arm_width).copy()
    mismatched[:, -1, 6] += 0.5
    with h5py.File(path, "w") as f:
        demo = f.create_group("data").create_group("demo_0")
        demo.create_dataset("actions", data=original)
        demo.create_dataset("actions_abs", data=mismatched.reshape(original.shape))
    with pytest.raises(ValueError, match="shared control"):
        extract_action_dict(str(path), add_absolute_actions=True)
    with h5py.File(path, "r") as f:
        assert "action_dict" not in f["data/demo_0"]
