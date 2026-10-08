"""Synthetic HDF5 demo through the real robomimic convert_demo method.

Generates a small episode using REAL Panda OSC / Lift simulator, stores
states and corresponding absolute goals in HDF5, then runs the actual
RobomimicDeltaActionConverter.convert_demo method against state restoration.

This is not a public dataset or task-success demonstration.
"""
import importlib.util
import pathlib
import sys
import types
from unittest.mock import patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("robosuite")
h5py = pytest.importorskip("h5py")
import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)
from robomimic.scripts.conversion.robosuite_action_conversion import (
    compose_delta_with_pose,
    scale_action,
)


def _delta_env():
    cfg = load_composite_controller_config(controller=None, robot="Panda")
    arm = cfg["body_parts"]["right"]
    arm["type"] = "OSC_POSE"
    arm["input_type"] = "delta"
    arm["input_ref_frame"] = "base"
    arm["impedance_mode"] = "fixed"
    return suite.make(
        "Lift", robots="Panda", controller_configs=cfg,
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
        control_freq=20, horizon=40,
    )


class _EnvRestoreAdapter:
    """Minimal robomimic EnvRobosuite.reset_to shape over a real simulator."""

    def __init__(self, real_env):
        self.env = real_env

    def reset_to(self, state):
        self.env.sim.set_state_from_flattened(
            np.asarray(state["states"], dtype=float)
        )
        self.env.sim.forward()
        return None


def _load_real_converter_method_without_heavy_training_dependencies():
    # The conversion script's module-level env/obs/config imports are for
    # converter __init__, which this test intentionally bypasses. This test
    # checks the REAL convert_demo and convert_actions implementations, not
    # the robomimic metadata constructor or multiprocessing writer.
    import robomimic.utils as robomimic_utils

    fake = {}
    for modname in (
        "robomimic.utils.env_utils",
        "robomimic.utils.file_utils",
        "robomimic.utils.obs_utils",
    ):
        fake[modname] = types.ModuleType(modname)
    cfg = types.ModuleType("robomimic.config")
    cfg.config_factory = lambda *args, **kwargs: None
    fake["robomimic.config"] = cfg

    path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "robomimic/scripts/conversion/robosuite_add_delta_actions.py"
    )
    spec = importlib.util.spec_from_file_location(
        "robomimic_delta_converter_hdf5_test", path
    )
    mod = importlib.util.module_from_spec(spec)
    # Import statement semantics also resolve the child as a parent-package
    # attribute, so stub both sys.modules and robomimic.utils attributes.
    saved_attrs = {}
    for fullname in (
        "robomimic.utils.env_utils",
        "robomimic.utils.file_utils",
        "robomimic.utils.obs_utils",
    ):
        leaf = fullname.rsplit(".", 1)[-1]
        saved_attrs[leaf] = getattr(robomimic_utils, leaf, None)
        setattr(robomimic_utils, leaf, fake[fullname])
    try:
        with patch.dict(sys.modules, fake):
            spec.loader.exec_module(mod)
    finally:
        for leaf, previous in saved_attrs.items():
            if previous is None:
                delattr(robomimic_utils, leaf)
            else:
                setattr(robomimic_utils, leaf, previous)
    return mod.RobomimicDeltaActionConverter


def test_generated_hdf5_episode_roundtrips_real_osc_deltas(tmp_path):
    src = _delta_env()
    replay = _delta_env()
    try:
        src.reset()
        replay.reset()
        controller = src.robots[0].part_controllers["right"]
        states = []
        abs_actions = []
        native_actions = []
        rng = np.random.default_rng(270)
        for _ in range(8):
            src.robots[0].composite_controller.update_state()
            controller.update(force=True)
            state = np.asarray(src.sim.get_state().flatten(), dtype=float).copy()
            native = rng.uniform(-0.2, 0.2, size=6)
            physical = scale_action(
                native, controller.input_min, controller.input_max,
                controller.output_min, controller.output_max,
            )
            pos = controller.world_to_origin_frame(controller.ref_pos)
            ori = controller.goal_origin_to_eef_pose()[:3, :3]
            goal = compose_delta_with_pose(pos, ori, physical)
            states.append(state)
            abs_actions.append(np.r_[goal, 0.0])
            native_actions.append(np.r_[native, 0.0])
            action = np.zeros_like(src.action_spec[0])
            arm_idx = src.robots[0].composite_controller._action_split_indexes["right"]
            action[arm_idx[0]:arm_idx[1]] = native
            src.step(action)

        dataset = tmp_path / "synthetic_osc_episode.hdf5"
        with h5py.File(dataset, "w") as out:
            demo = out.create_group("data/demo_0")
            demo.create_dataset("states", data=np.asarray(states))
            demo.create_dataset("actions", data=np.asarray(abs_actions))
            demo.attrs["model_file"] = ""
            demo.attrs["ep_meta"] = ""

        converter_type = _load_real_converter_method_without_heavy_training_dependencies()
        converter = object.__new__(converter_type)
        converter.env = _EnvRestoreAdapter(replay)
        converter.file = h5py.File(dataset, "r")
        try:
            recovered, info = converter.convert_demo("demo_0")
        finally:
            converter.file.close()

        expected = np.asarray(native_actions)
        error = float(np.max(np.abs(recovered - expected)))
        print(
            f"ROBOMIMIC_HDF5_REAL_OSC_PASS steps={len(states)} "
            f"native_error_max={error:.6e} saturation_count={info['saturation_count']}"
        )
        assert error < 1e-8
        assert info["saturation_count"] == 0
        np.testing.assert_array_equal(recovered[:, 6:], expected[:, 6:])
    finally:
        src.close()
        replay.close()
