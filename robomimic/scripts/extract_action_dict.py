"""
This script extracts action dictionaries from a dataset in HDF5 format.
It reads the actions from the dataset, processes them to extract position,
rotation (both as axis-angle and 6D representation), and gripper state,
and saves these as new datasets within the HDF5 file under the "action_dict"
group for each demonstration.
"""

import argparse
import pathlib
import sys
import tqdm
import h5py
import numpy as np
import torch
import os

import robomimic.utils.torch_utils as TorchUtils

def _action_fields(in_action, is_absolute):
    """
    Convert per-step action vectors to named action fields.

    Single-arm 7/8-D datasets retain the historical field names unchanged.
    Dual-arm 14/16-D datasets receive explicit robot0_/robot1_ prefixes:
    silently treating 14-D commands as a 7-D single arm drops half the action.

    Each arm is encoded as xyz position, 3-D rotation vector, gripper, and
    optionally a mobile-base mode flag.
    """
    if in_action.ndim != 2:
        raise ValueError(f"Expected action matrix [steps, features], got {in_action.shape}")
    num_steps, width = in_action.shape
    if width in (7, 8):
        num_robots, arm_width = 1, width
    elif width in (14, 16):
        num_robots, arm_width = 2, width // 2
    else:
        raise ValueError(
            f"Unsupported action width {width}; expected 7/8 per arm and 1 or 2 arms."
        )

    arms = in_action.reshape(num_steps, num_robots, arm_width)
    representation = "abs_" if is_absolute else "rel_"
    fields = {}
    for arm_idx in range(num_robots):
        arm = arms[:, arm_idx]
        owner = f"robot{arm_idx}_" if num_robots > 1 else ""
        arm_rotation = arm[:, 3:6].astype(np.float32)
        rot_6d = TorchUtils.axis_angle_to_rot_6d(
            axis_angle=torch.from_numpy(arm_rotation)
        ).numpy().astype(np.float32)
        fields[owner + representation + "pos"] = arm[:, :3].astype(np.float32)
        fields[owner + representation + "rot_axis_angle"] = arm_rotation
        fields[owner + representation + "rot_6d"] = rot_6d
        fields[owner + "gripper"] = arm[:, 6:7].astype(np.float32)
        if arm_width == 8:
            fields[owner + "base_mode"] = arm[:, 7:8].astype(np.float32)

    return fields, num_robots


def extract_action_dict(dataset, add_absolute_actions=True):
    with h5py.File(os.path.expanduser(dataset), mode="r+") as f:
        specs = [("actions", False)]
        if add_absolute_actions:
            specs.append(("actions_abs", True))

        for demo in f["data"].values():
            action_dict_group = demo.require_group("action_dict")
            num_robots = None
            for action_key, is_absolute in specs:
                if action_key not in demo:
                    raise KeyError(f"Missing {action_key} in demonstration {demo.name}")
                fields, count = _action_fields(demo[action_key][:], is_absolute)
                if num_robots is not None and count != num_robots:
                    raise ValueError(f"Relative/absolute action arm counts disagree in {demo.name}")
                num_robots = count
                for key, data in fields.items():
                    if key in action_dict_group:
                        del action_dict_group[key]
                    action_dict_group.create_dataset(key, data=data)
            if num_robots > 1:
                # Remove stale legacy fields from datasets previously processed
                # by the single-arm-only extractor. They represent just arm 0.
                for key in (
                    "rel_pos", "rel_rot_axis_angle", "rel_rot_6d",
                    "abs_pos", "abs_rot_axis_angle", "abs_rot_6d",
                    "gripper", "base_mode",
                ):
                    if key in action_dict_group:
                        del action_dict_group[key]
            action_dict_group.attrs["num_robots"] = num_robots


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True
    )
    
    args = parser.parse_args()
    extract_action_dict(args.dataset)
    
if __name__ == "__main__":
    main()
