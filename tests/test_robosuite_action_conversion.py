import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robomimic.scripts.extract_action_dict import extract_action_dict
from robomimic.scripts.conversion.robosuite_add_delta_actions import (
    _absolute_pose_to_delta,
    _controller_achieved_pose,
    _inverse_scale_action,
)


class DummyController:
    input_min = np.full(6, -1.0)
    input_max = np.full(6, 1.0)
    output_min = np.array([-0.05, -0.05, -0.05, -0.5, -0.5, -0.5])
    output_max = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
    _goal_update_mode = "achieved"

    def __init__(self, position, orientation):
        self.input_ref_frame = "world"
        self.ref_pos = np.asarray(position, dtype=float)
        self.ref_ori_mat = np.asarray(orientation, dtype=float)

    def update(self, force=False):
        assert force


def scale_action(controller, native):
    native = np.clip(native, controller.input_min, controller.input_max)
    scale = abs(controller.output_max - controller.output_min) / abs(
        controller.input_max - controller.input_min
    )
    output_transform = (controller.output_max + controller.output_min) / 2.0
    input_transform = (controller.input_max + controller.input_min) / 2.0
    return (native - input_transform) * scale + output_transform


def compose_absolute(position, orientation, physical_delta):
    goal_pos = position + physical_delta[:3]
    goal_ori = Rotation.from_rotvec(physical_delta[3:6]).as_matrix() @ orientation
    return np.concatenate([goal_pos, Rotation.from_matrix(goal_ori).as_rotvec()])


def test_inverse_scale_action_round_trip():
    controller = DummyController(np.zeros(3), np.eye(3))
    rng = np.random.default_rng(270)
    for _ in range(1000):
        native = rng.uniform(-1.0, 1.0, size=6)
        physical = scale_action(controller, native)
        recovered, representable = _inverse_scale_action(controller, physical)
        assert representable
        np.testing.assert_allclose(recovered, native, atol=1e-12)


def test_absolute_pose_to_delta_inverts_robosuite_orientation_composition():
    rng = np.random.default_rng(3312)
    for _ in range(500):
        position = rng.uniform(-0.3, 0.3, size=3)
        orientation = Rotation.random(random_state=rng).as_matrix()
        controller = DummyController(position, orientation)
        native = rng.uniform(-0.8, 0.8, size=6)
        physical = scale_action(controller, native)
        absolute = compose_absolute(position, orientation, physical)

        recovered, representable, recovered_physical = _absolute_pose_to_delta(
            controller, absolute
        )

        assert representable
        np.testing.assert_allclose(recovered, native, atol=1e-10)
        np.testing.assert_allclose(recovered_physical, physical, atol=1e-10)


def test_out_of_range_goal_is_reported_as_nonrepresentable():
    controller = DummyController(np.zeros(3), np.eye(3))
    absolute = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])

    native, representable, physical = _absolute_pose_to_delta(controller, absolute)

    assert not representable
    assert physical[0] == pytest.approx(0.2)
    assert native[0] == pytest.approx(1.0)


def test_desired_goal_mode_is_refused_without_controller_memory():
    controller = DummyController(np.zeros(3), np.eye(3))
    controller._goal_update_mode = "desired"

    with pytest.raises(NotImplementedError, match="achieved-state"):
        _absolute_pose_to_delta(controller, np.zeros(6))


class BaseFrameController(DummyController):
    def __init__(self, position, orientation):
        super().__init__(position, orientation)
        self.input_ref_frame = "base"

    def world_to_origin_frame(self, position):
        return np.asarray(position) - np.array([1.0, 2.0, 3.0])

    def goal_origin_to_eef_pose(self):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec([0.1, 0.2, -0.1]).as_matrix()
        return pose


def test_current_robosuite_base_frame_pose_extraction():
    controller = BaseFrameController([2.0, 4.0, 6.0], np.eye(3))
    position, orientation = _controller_achieved_pose(controller)
    np.testing.assert_allclose(position, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        orientation,
        Rotation.from_rotvec([0.1, 0.2, -0.1]).as_matrix(),
        atol=1e-12,
    )


class LegacyController:
    input_min = np.full(6, -1.0)
    input_max = np.full(6, 1.0)
    output_min = np.full(6, -0.5)
    output_max = np.full(6, 0.5)
    ee_pos = np.array([0.2, -0.1, 0.4])
    ee_ori_mat = Rotation.from_rotvec([0.2, 0.1, 0.0]).as_matrix()

    def update(self, force=False):
        assert force


def test_legacy_robosuite_world_frame_pose_extraction():
    controller = LegacyController()
    position, orientation = _controller_achieved_pose(controller)
    np.testing.assert_allclose(position, controller.ee_pos)
    np.testing.assert_allclose(orientation, controller.ee_ori_mat)



def test_action_dict_labels_absolute_input_and_generated_delta(tmp_path):
    dataset = tmp_path / "actions.hdf5"
    absolute = np.array(
        [
            [0.2, 0.3, 0.4, 0.0, 0.0, 0.0, -1.0],
            [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    delta = np.array(
        [
            [0.1, 0.0, -0.1, 0.0, 0.0, 0.0, -1.0],
            [0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    with h5py.File(dataset, "w") as f_out:
        demo = f_out.create_group("data").create_group("demo_0")
        demo.create_dataset("actions", data=absolute)
        demo.create_dataset("actions_delta", data=delta)

    extract_action_dict(
        str(dataset),
        add_absolute_actions=False,
        add_delta_actions=True,
        actions_are_absolute=True,
    )

    with h5py.File(dataset, "r") as f_in:
        action_dict = f_in["data/demo_0/action_dict"]
        np.testing.assert_allclose(action_dict["abs_pos"][:], absolute[:, :3])
        np.testing.assert_allclose(action_dict["rel_pos"][:], delta[:, :3])
        np.testing.assert_allclose(action_dict["gripper"][:], delta[:, 6:7])
