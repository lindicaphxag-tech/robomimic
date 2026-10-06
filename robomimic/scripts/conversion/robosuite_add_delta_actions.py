"""
Convert robosuite datasets with absolute OSC pose actions to delta actions.

This mirrors robosuite_add_absolute_actions.py in the opposite direction.
For each recorded state, it reconstructs the robosuite OSC action that realizes
the dataset's absolute end-effector goal, while preserving the remainder of the
robot action, for example the gripper command.

The conversion is exact only when the required physical delta lies inside the
configured controller output range. Saturated steps are reported explicitly.
"""

import argparse
import copy
import multiprocessing
import pathlib

import h5py
import numpy as np
import robosuite
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory


def _inverse_scale_action(controller, physical_delta):
    """Invert robosuite Controller.scale_action."""
    physical_delta = np.asarray(physical_delta, dtype=float)
    input_min = np.asarray(controller.input_min, dtype=float)
    input_max = np.asarray(controller.input_max, dtype=float)
    output_min = np.asarray(controller.output_min, dtype=float)
    output_max = np.asarray(controller.output_max, dtype=float)

    representable_mask = (physical_delta >= output_min) & (physical_delta <= output_max)
    clipped = np.clip(physical_delta, output_min, output_max)

    input_mid = (input_max + input_min) / 2.0
    output_mid = (output_max + output_min) / 2.0
    inverse_scale = (input_max - input_min) / (output_max - output_min)
    native = (clipped - output_mid) * inverse_scale + input_mid
    native = np.clip(native, input_min, input_max)
    return native, bool(np.all(representable_mask))


def _controller_achieved_pose(controller):
    """Return the achieved OSC pose in the absolute-action reference frame."""
    controller.update(force=True)

    # robosuite >= 1.5 exposes the controller input reference frame explicitly.
    if hasattr(controller, "input_ref_frame"):
        if controller.input_ref_frame == "base":
            position = controller.world_to_origin_frame(controller.ref_pos)
            orientation = controller.goal_origin_to_eef_pose()[:3, :3]
        elif controller.input_ref_frame == "world":
            position = controller.ref_pos
            orientation = controller.ref_ori_mat
        else:
            raise ValueError(f"Unsupported OSC input_ref_frame: {controller.input_ref_frame}")
    # robosuite <= 1.4.1 absolute OSC actions use the world-frame ee pose.
    else:
        position = controller.ee_pos
        orientation = controller.ee_ori_mat

    return np.asarray(position), np.asarray(orientation)


def _absolute_pose_to_delta(controller, absolute_pose):
    """Convert absolute [pos, rotvec] to a policy-native OSC delta action."""
    absolute_pose = np.asarray(absolute_pose)
    if absolute_pose.shape != (6,):
        raise ValueError("absolute_pose must have shape (6,)")

    # A single recorded simulator state does not contain arbitrary controller
    # target memory. Refuse desired-goal semantics rather than synthesizing a
    # silently incorrect delta.
    goal_update_mode = getattr(controller, "_goal_update_mode", "achieved")
    if goal_update_mode != "achieved":
        raise NotImplementedError(
            "absolute->delta conversion currently requires achieved-state OSC goal updates"
        )

    baseline_pos, baseline_ori = _controller_achieved_pose(controller)
    goal_pos = absolute_pose[:3]
    goal_ori = Rotation.from_rotvec(absolute_pose[3:6]).as_matrix()

    physical_pos_delta = goal_pos - baseline_pos

    # robosuite OSC composes orientation as:
    #   R_goal = R_delta @ R_baseline
    # so the inverse is:
    #   R_delta = R_goal @ R_baseline.T
    delta_ori = goal_ori @ baseline_ori.T
    physical_ori_delta = Rotation.from_matrix(delta_ori).as_rotvec()

    physical_delta = np.concatenate([physical_pos_delta, physical_ori_delta])
    native_delta, representable = _inverse_scale_action(controller, physical_delta)
    return native_delta, representable, physical_delta


class RobomimicDeltaActionConverter:
    """Convert an absolute-action robomimic dataset to delta OSC actions."""

    def __init__(self, dataset_path, algo_name="bc"):
        config = config_factory(algo_name=algo_name)
        ObsUtils.initialize_obs_utils_with_config(config)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        delta_env_meta = copy.deepcopy(env_meta)

        if robosuite.__version__ < "1.5":
            delta_env_meta["env_kwargs"]["controller_configs"]["control_delta"] = True
        else:
            controller_config = delta_env_meta["env_kwargs"]["controller_configs"]["body_parts"]["right"]
            controller_config["control_delta"] = True
            if "input_type" in controller_config:
                controller_config["input_type"] = "delta"

        env = EnvUtils.create_env_from_metadata(
            env_meta=delta_env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=False,
        )
        assert len(env.env.robots) in (1, 2)

        for robot in env.env.robots:
            controller = (
                robot.controller
                if robosuite.__version__ < "1.5"
                else robot.part_controllers["right"]
            )
            if robosuite.__version__ < "1.5":
                assert controller.use_delta
            else:
                assert controller.input_type == "delta"
            if getattr(controller, "impedance_mode", "fixed") != "fixed":
                raise NotImplementedError(
                    "absolute->delta conversion currently supports fixed-impedance OSC only"
                )
            if getattr(controller, "_goal_update_mode", "achieved") != "achieved":
                raise NotImplementedError(
                    "absolute->delta conversion currently supports achieved-state OSC goal updates only"
                )

        self.env = env
        self.file = h5py.File(dataset_path, "r")

    def get_demo_keys(self):
        return list(self.file["data"].keys())

    def convert_actions(self, states, actions, initial_state):
        env = self.env
        d_a = len(env.env.robots[0].action_limits[0])

        # (N, 14) -> (N, 2, 7), or (N, 7) -> (N, 1, 7).
        stacked_actions = actions.reshape(*actions.shape[:-1], -1, d_a)
        stacked_delta_actions = np.array(stacked_actions, copy=True)

        saturation_count = 0
        max_physical_excess = 0.0

        for i in range(len(states)):
            if i == 0:
                env.reset_to(initial_state)
            else:
                env.reset_to({"states": states[i]})

            for idx, robot in enumerate(env.env.robots):
                controller = (
                    robot.controller
                    if robosuite.__version__ < "1.5"
                    else robot.part_controllers["right"]
                )
                native_delta, representable, physical_delta = _absolute_pose_to_delta(
                    controller,
                    stacked_actions[i, idx, :6],
                )
                stacked_delta_actions[i, idx, :6] = native_delta

                if not representable:
                    saturation_count += 1
                    output_min = np.asarray(controller.output_min)
                    output_max = np.asarray(controller.output_max)
                    excess = np.maximum(
                        np.maximum(output_min - physical_delta, 0.0),
                        np.maximum(physical_delta - output_max, 0.0),
                    )
                    max_physical_excess = max(max_physical_excess, float(np.max(excess)))

                # Values after the OSC pose command are intentionally preserved
                # (e.g. gripper or mobile-base mode).

        return stacked_delta_actions.reshape(actions.shape), {
            "saturation_count": saturation_count,
            "max_physical_excess": max_physical_excess,
        }

    def convert_demo(self, demo_key):
        demo = self.file[f"data/{demo_key}"]
        states = demo["states"][:]
        actions = demo["actions"][:]
        initial_state = dict(states=states[0])
        initial_state["model"] = demo.attrs["model_file"]
        initial_state["ep_meta"] = demo.attrs.get("ep_meta", None)
        return self.convert_actions(states, actions, initial_state=initial_state)


def worker(x):
    path, demo_key = x
    converter = RobomimicDeltaActionConverter(path)
    return converter.convert_demo(demo_key)


def add_delta_actions_to_dataset(dataset, num_workers):
    """Add an actions_delta dataset to every demonstration."""
    dataset = pathlib.Path(dataset).expanduser()
    assert dataset.is_file()

    converter = RobomimicDeltaActionConverter(dataset)
    demo_keys = converter.get_demo_keys()
    del converter

    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(worker, [(dataset, demo_key) for demo_key in demo_keys])

    total_saturation = 0
    max_physical_excess = 0.0
    with h5py.File(dataset, "r+") as out_file:
        for i in tqdm(range(len(results)), desc="Writing to output"):
            delta_actions, info = results[i]
            demo = out_file[f"data/{demo_keys[i]}"]
            if "actions_delta" not in demo:
                demo.create_dataset("actions_delta", data=np.asarray(delta_actions))
            else:
                demo["actions_delta"][:] = delta_actions

            total_saturation += info["saturation_count"]
            max_physical_excess = max(max_physical_excess, info["max_physical_excess"])

    if total_saturation:
        print(
            "Warning: absolute->delta conversion saturated "
            f"{total_saturation} arm-step actions; max physical excess was "
            f"{max_physical_excess:.6g}. Those steps cannot exactly reproduce "
            "the absolute goal under the configured delta-action limits."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=10)
    args = parser.parse_args()

    add_delta_actions_to_dataset(
        dataset=args.dataset,
        num_workers=args.num_workers,
    )
