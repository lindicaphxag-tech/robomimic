"""Add policy-native delta OSC actions to an absolute-action robomimic dataset."""

import argparse
import copy
import multiprocessing
import pathlib
import re

import h5py
import numpy as np
import robosuite
from tqdm import tqdm

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory
from robomimic.scripts.conversion.robosuite_action_conversion import (
    inverse_scale_action,
    physical_delta_from_absolute_pose,
)


def _robosuite_version_tuple():
    parts = [int(x) for x in re.findall(r"\d+", robosuite.__version__)[:2]]
    while len(parts) < 2:
        parts.append(0)
    return tuple(parts)


def _legacy_robosuite():
    return _robosuite_version_tuple() < (1, 5)


def _arm_controller(robot):
    if _legacy_robosuite():
        return robot.controller
    return robot.part_controllers["right"]


def _controller_config(env_meta):
    if _legacy_robosuite():
        return env_meta["env_kwargs"]["controller_configs"]
    return env_meta["env_kwargs"]["controller_configs"]["body_parts"]["right"]


def _controller_is_delta(config):
    if "input_type" in config:
        return config["input_type"] == "delta"
    if "control_delta" in config:
        return bool(config["control_delta"])
    raise ValueError(
        "controller metadata does not declare input_type or control_delta"
    )


def _set_delta_controller(config):
    if "input_type" in config:
        config["input_type"] = "delta"
    # Match robomimic's existing >=1.5 metadata compatibility field.
    config["control_delta"] = True


def _controller_achieved_pose(controller):
    """Return achieved OSC pose in the same chart as absolute controller input."""
    controller.update(force=True)

    if hasattr(controller, "input_ref_frame"):
        if controller.input_ref_frame == "base":
            position = controller.world_to_origin_frame(controller.ref_pos)
            orientation = controller.goal_origin_to_eef_pose()[:3, :3]
        elif controller.input_ref_frame == "world":
            position = controller.ref_pos
            orientation = controller.ref_ori_mat
        else:
            raise ValueError(
                "unsupported OSC input_ref_frame: "
                f"{controller.input_ref_frame}"
            )
    else:
        # robosuite <= 1.4.1 absolute OSC inputs use world-frame ee pose.
        position = controller.ee_pos
        orientation = controller.ee_ori_mat

    return np.asarray(position), np.asarray(orientation)


def _absolute_pose_to_delta(controller, absolute_pose):
    if getattr(controller, "impedance_mode", "fixed") != "fixed":
        raise NotImplementedError(
            "absolute-to-delta conversion currently supports fixed "
            "impedance OSC controllers only"
        )
    if getattr(controller, "_goal_update_mode", "achieved") != "achieved":
        raise NotImplementedError(
            "desired goal-update mode requires previous controller target "
            "history, which is not stored in the dataset state"
        )

    baseline_position, baseline_orientation = _controller_achieved_pose(
        controller
    )
    physical_delta = physical_delta_from_absolute_pose(
        absolute_pose,
        baseline_position,
        baseline_orientation,
    )
    native_delta, representable_mask = inverse_scale_action(
        physical_delta,
        controller.input_min,
        controller.input_max,
        controller.output_min,
        controller.output_max,
    )
    return native_delta, physical_delta, representable_mask


class RobomimicDeltaActionConverter:
    """Convert absolute OSC pose actions into robosuite-native delta actions."""

    def __init__(self, dataset_path, algo_name="bc"):
        config = config_factory(algo_name=algo_name)
        ObsUtils.initialize_obs_utils_with_config(config)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        source_controller_config = _controller_config(env_meta)
        if _controller_is_delta(source_controller_config):
            raise ValueError(
                "absolute-to-delta conversion requires an absolute-action "
                "source controller"
            )

        delta_env_meta = copy.deepcopy(env_meta)
        _set_delta_controller(_controller_config(delta_env_meta))

        env = EnvUtils.create_env_from_metadata(
            env_meta=delta_env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=False,
        )
        assert len(env.env.robots) in (1, 2)

        for robot in env.env.robots:
            controller = _arm_controller(robot)
            if _legacy_robosuite():
                assert controller.use_delta
            elif hasattr(controller, "input_type"):
                assert controller.input_type == "delta"

        self.env = env
        self.file = h5py.File(dataset_path, "r")

    def get_demo_keys(self):
        return list(self.file["data"].keys())

    def convert_actions(self, states, actions, initial_state):
        """Convert one absolute-action episode while preserving action remainder."""
        env = self.env
        d_a = len(env.env.robots[0].action_limits[0])
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
                # reset_to restores MuJoCo state but does not update each arm
                # controller's reference frame. In robosuite >=1.5 the OSC
                # base-frame pose depends on this state-derived origin.
                if not _legacy_robosuite():
                    robot.composite_controller.update_state()
                controller = _arm_controller(robot)
                native_delta, physical_delta, representable_mask = (
                    _absolute_pose_to_delta(
                        controller,
                        stacked_actions[i, idx, :6],
                    )
                )
                stacked_delta_actions[i, idx, :6] = native_delta

                if not np.all(representable_mask):
                    saturation_count += 1
                    output_min = np.asarray(controller.output_min)
                    output_max = np.asarray(controller.output_max)
                    excess = np.maximum(
                        np.maximum(output_min - physical_delta, 0.0),
                        np.maximum(physical_delta - output_max, 0.0),
                    )
                    max_physical_excess = max(
                        max_physical_excess,
                        float(np.max(excess)),
                    )

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
        return self.convert_actions(
            states,
            actions,
            initial_state=initial_state,
        )


def worker(x):
    path, demo_key = x
    converter = RobomimicDeltaActionConverter(path)
    return converter.convert_demo(demo_key)


def add_delta_actions_to_dataset(dataset, num_workers):
    """Add an actions_delta dataset to each demonstration."""
    dataset = pathlib.Path(dataset).expanduser()
    assert dataset.is_file()

    converter = RobomimicDeltaActionConverter(dataset)
    demo_keys = converter.get_demo_keys()
    del converter

    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(
            worker,
            [(dataset, demo_key) for demo_key in demo_keys],
        )

    total_saturation = 0
    max_physical_excess = 0.0
    with h5py.File(dataset, "r+") as out_file:
        for i in tqdm(range(len(results)), desc="Writing to output"):
            delta_actions, info = results[i]
            demo = out_file[f"data/{demo_keys[i]}"]
            if "actions_delta" not in demo:
                demo.create_dataset(
                    "actions_delta",
                    data=np.asarray(delta_actions),
                )
            else:
                demo["actions_delta"][:] = delta_actions

            total_saturation += info["saturation_count"]
            max_physical_excess = max(
                max_physical_excess,
                info["max_physical_excess"],
            )

    if total_saturation:
        print(
            "Warning: absolute-to-delta conversion saturated "
            f"{total_saturation} arm-step actions; max physical excess "
            f"was {max_physical_excess:.6g}. These steps cannot exactly "
            "reproduce the absolute goal under the configured delta limits."
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
