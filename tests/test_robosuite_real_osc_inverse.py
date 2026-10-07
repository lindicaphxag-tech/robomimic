"""Real robosuite controller integration: fixed-impedance OSC delta round trip.

CPU-only, no dataset download and no renderer. Exercises robosuite's own
controller goal generator in addition to the pure SO(3)/scale equations.
"""
import numpy as np
from scipy.spatial.transform import Rotation

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config
from robomimic.scripts.conversion.robosuite_action_conversion import (
    compose_delta_with_pose,
    inverse_scale_action,
    physical_delta_from_absolute_pose,
    scale_action,
)


def test_real_robosuite_osc_goal_and_reverse_native_action():
    cfg = load_composite_controller_config(controller=None, robot="Panda")
    arm_cfg = cfg["body_parts"]["right"]
    arm_cfg["type"] = "OSC_POSE"
    arm_cfg["input_type"] = "delta"
    arm_cfg["input_ref_frame"] = "base"
    arm_cfg["impedance_mode"] = "fixed"
    arm_cfg["goal_update_mode"] = "achieved"

    env = suite.make(
        "Lift",
        robots="Panda",
        controller_configs=cfg,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=40,
    )
    try:
        env.reset()
        robot = env.robots[0]
        osc = robot.part_controllers["right"]
        assert osc.input_type == "delta"
        assert osc._goal_update_mode == "achieved"

        rng = np.random.default_rng(270)
        max_native_error = 0.0
        max_position_goal_error = 0.0
        max_orientation_goal_error = 0.0
        for _ in range(10):
            robot.composite_controller.update_state()
            osc.update(force=True)
            baseline_pos = np.asarray(osc.world_to_origin_frame(osc.ref_pos), dtype=float)
            baseline_ori = np.asarray(osc.goal_origin_to_eef_pose()[:3, :3], dtype=float)

            native = rng.uniform(-0.25, 0.25, size=6)
            physical = scale_action(
                native, osc.input_min, osc.input_max,
                osc.output_min, osc.output_max
            )
            expected_abs = compose_delta_with_pose(
                baseline_pos, baseline_ori, physical
            )
            recovered_delta = physical_delta_from_absolute_pose(
                expected_abs, baseline_pos, baseline_ori
            )
            recovered_native, mask = inverse_scale_action(
                recovered_delta,
                osc.input_min, osc.input_max,
                osc.output_min, osc.output_max
            )
            assert np.all(mask)
            max_native_error = max(
                max_native_error,
                float(np.max(np.abs(recovered_native - native)))
            )

            # Verify actual robosuite controller goal semantics rather than
            # just comparing the two algebraic helper implementations.
            osc.set_goal(native)
            max_position_goal_error = max(
                max_position_goal_error,
                float(np.linalg.norm(osc.goal_pos - expected_abs[:3]))
            )
            target_R = Rotation.from_rotvec(expected_abs[3:]).as_matrix()
            max_orientation_goal_error = max(
                max_orientation_goal_error,
                float(Rotation.from_matrix(osc.goal_ori @ target_R.T).magnitude())
            )

            action = np.zeros_like(env.action_spec[0], dtype=float)
            start, stop = robot.composite_controller._action_split_indexes["right"]
            action[start:stop] = native
            env.step(action)

        print(
            f"REAL_OSC_PASS steps=10 native_max={max_native_error:.3e} "
            f"goal_pos_max={max_position_goal_error:.3e} "
            f"goal_rot_max={max_orientation_goal_error:.3e}"
        )
        assert max_native_error < 1e-9
        assert max_position_goal_error < 1e-9
        # SO(3) matrix-to-axis-angle conversions have an absolute numeric
        # error floor near 1e-8 rad when recovering tiny relative rotations.
        assert max_orientation_goal_error < 1e-7
    finally:
        env.close()
