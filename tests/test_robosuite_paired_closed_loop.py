"""Paired closed-loop OSC rollout: frozen action source vs controller swap.

Start two real robosuite Panda/Lift instances from the same physics state.
Compare native delta OSC execution with the exact absolute OSC goal compiled
from that delta action, without modifying either environment's dynamics.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("robosuite")
import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config
from robomimic.scripts.conversion.robosuite_action_conversion import (
    compose_delta_with_pose,
    scale_action,
)


def _env(input_type):
    cfg = load_composite_controller_config(controller=None, robot="Panda")
    arm = cfg["body_parts"]["right"]
    arm["type"] = "OSC_POSE"
    arm["input_type"] = input_type
    arm["input_ref_frame"] = "base"
    arm["impedance_mode"] = "fixed"
    return suite.make(
        "Lift",
        robots="Panda",
        controller_configs=cfg,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=25,
    )


def _action(env, arm_action):
    robot = env.robots[0]
    action = np.zeros_like(env.action_spec[0], dtype=float)
    start, end = robot.composite_controller._action_split_indexes["right"]
    assert end - start == 6
    action[start:end] = arm_action
    return action


def test_two_real_osc_controllers_track_same_physics_under_compiled_actions():
    delta_env = _env("delta")
    abs_env = _env("absolute")
    try:
        delta_env.reset()
        abs_env.reset()

        initial_state = delta_env.sim.get_state().flatten()
        abs_env.sim.set_state_from_flattened(initial_state)
        abs_env.sim.forward()
        np.testing.assert_allclose(
            delta_env.sim.data.qpos,
            abs_env.sim.data.qpos,
            atol=1e-12,
        )
        delta_robot = delta_env.robots[0]
        abs_robot = abs_env.robots[0]
        delta_osc = delta_robot.part_controllers["right"]
        abs_osc = abs_robot.part_controllers["right"]
        assert delta_osc.input_type == "delta"
        assert abs_osc.input_type == "absolute"

        # Diagnose controller state that is NOT encoded in MuJoCo's flattened
        # physics state. Copying qpos/qvel alone can leave controller-owned
        # nullspace targets or reference memory inconsistent.
        for attr in ("initial_joint", "kp", "kd", "origin_pos", "origin_ori"):
            a = getattr(delta_osc, attr, None)
            b = getattr(abs_osc, attr, None)
            if a is not None and b is not None:
                print(
                    "CONTROLLER_STATE", attr,
                    "maxdiff", float(np.max(np.abs(np.asarray(a) - np.asarray(b)))),
                    "source", np.asarray(a).tolist(),
                    "target", np.asarray(b).tolist(),
                )
        print(
            "INITIAL_PHYSICS",
            float(np.max(np.abs(
                delta_env.sim.get_state().flatten() -
                abs_env.sim.get_state().flatten()
            ))),
        )
        rng = np.random.default_rng(270)
        qpos_errors = []
        goal_errors = []
        for _ in range(8):
            delta_robot.composite_controller.update_state()
            abs_robot.composite_controller.update_state()
            delta_osc.update(force=True)
            abs_osc.update(force=True)

            base_pos = np.asarray(
                delta_osc.world_to_origin_frame(delta_osc.ref_pos),
                dtype=float,
            )
            base_ori = np.asarray(
                delta_osc.goal_origin_to_eef_pose()[:3, :3],
                dtype=float,
            )
            native = rng.uniform(-0.20, 0.20, size=6)
            physical = scale_action(
                native,
                delta_osc.input_min,
                delta_osc.input_max,
                delta_osc.output_min,
                delta_osc.output_max,
            )
            absolute = compose_delta_with_pose(base_pos, base_ori, physical)

            delta_env.step(_action(delta_env, native))
            abs_env.step(_action(abs_env, absolute))

            source_q = np.asarray(delta_env.sim.data.qpos, dtype=float).copy()
            target_q = np.asarray(abs_env.sim.data.qpos, dtype=float).copy()
            qpos_errors.append(float(np.max(np.abs(source_q - target_q))))
            if len(qpos_errors) == 1:
                print(
                    "FIRST_STEP",
                    "source_qpos", source_q.tolist(),
                    "target_qpos", target_q.tolist(),
                    "ctrl_maxdiff", float(np.max(np.abs(
                        np.asarray(delta_env.sim.data.ctrl) -
                        np.asarray(abs_env.sim.data.ctrl)
                    ))),
                )

            p_error = float(np.linalg.norm(delta_osc.goal_pos - abs_osc.goal_pos))
            R_error = float(
                Rotation.from_matrix(
                    delta_osc.goal_ori @ abs_osc.goal_ori.T
                ).magnitude()
            )
            goal_errors.append(max(p_error, R_error))

        max_qpos = max(qpos_errors)
        max_goal = max(goal_errors)
        print(
            f"PAIRED_OSC_REAL_ROLLOUT steps=8 "
            f"max_qpos={max_qpos:.6e} max_goal={max_goal:.6e} "
            f"qpos_errors={qpos_errors}"
        )
        assert max_goal < 1e-6
        assert max_qpos < 1e-4
    finally:
        delta_env.close()
        abs_env.close()
