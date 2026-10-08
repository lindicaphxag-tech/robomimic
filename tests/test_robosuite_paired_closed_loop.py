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


@pytest.mark.parametrize("model_alignment", ["none", "xml"])
def test_two_real_osc_controllers_track_same_physics_under_compiled_actions(model_alignment):
    delta_env = _env("delta")
    abs_env = _env("absolute")
    try:
        delta_env.reset()
        abs_env.reset()
        if model_alignment == "xml":
            # MJCF alignment eliminates independently sampled object geometry
            # and mass differences before testing controller migration.
            abs_env.reset_from_xml_string(delta_env.sim.model.get_xml())

        initial_state = delta_env.sim.get_state().flatten()
        abs_env.sim.set_state_from_flattened(initial_state)
        abs_env.sim.forward()
        if hasattr(delta_env.sim.data, "qacc_warmstart"):
            abs_env.sim.data.qacc_warmstart[...] = delta_env.sim.data.qacc_warmstart
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
        # Controller-state handshake: the simulator snapshot captures qpos,
        # qvel and time, but does not carry the OSC nullspace posture target.
        # Its run_controller() adds a nullspace torque around initial_joint.
        # A controller swap must migrate that target as well as physical state.
        abs_osc.update_initial_joints(np.asarray(delta_osc.initial_joint).copy())
        print(
            "HANDSHAKE_INITIAL_JOINT_MAXDIFF",
            float(np.max(np.abs(
                np.asarray(delta_osc.initial_joint) -
                np.asarray(abs_osc.initial_joint)
            ))),
        )

        rng = np.random.default_rng(270)
        qpos_errors = []
        arm_errors = []
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
            # Diagnose arm kinematics separately from free object dynamics.
            joint_indices = np.asarray(delta_robot._ref_joint_pos_indexes, dtype=int)
            arm_errors.append(
                float(np.max(np.abs(source_q[joint_indices] - target_q[joint_indices])))
            )
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
        max_arm = max(arm_errors)
        max_goal = max(goal_errors)
        print(
            f"PAIRED_OSC_REAL_ROLLOUT steps=8 model_alignment={model_alignment} "
            f"max_qpos={max_qpos:.6e} max_arm={max_arm:.6e} "
            f"max_goal={max_goal:.6e} "
            f"full_qpos_errors={qpos_errors} arm_errors={arm_errors}"
        )
        assert max_goal < 1e-6
        # Full-state equivalence is deliberately NOT asserted; the free
        # object belongs to the complete model dynamics, not this joint test.
        if max_qpos >= 1e-4:
            print("FULL_STATE_UNCERTIFIED: object/other qpos mismatch remains")
        assert max_arm < 1e-4
        if model_alignment == "xml":
            # Stronger but still FINITE-HORIZON / SCENE-SPECIFIC claim.
            assert max_qpos < 1e-5
    finally:
        delta_env.close()
        abs_env.close()


def test_closed_loop_controller_state_handshake_causal_torque_ablation():
    """Same physical state/goals; alter only target OSC nullspace memory."""
    delta_env = _env("delta")
    absolute_env = _env("absolute")
    try:
        delta_env.reset()
        absolute_env.reset()
        snapshot = delta_env.sim.get_state().flatten()
        absolute_env.sim.set_state_from_flattened(snapshot)
        absolute_env.sim.forward()
        src = delta_env.robots[0].part_controllers["right"]
        tgt = absolute_env.robots[0].part_controllers["right"]
        delta_env.robots[0].composite_controller.update_state()
        absolute_env.robots[0].composite_controller.update_state()
        src.update(force=True)
        tgt.update(force=True)
        native = np.array([0.08, -0.15, 0.12, 0.06, 0.09, -0.04])
        baseline_pos = src.world_to_origin_frame(src.ref_pos)
        baseline_ori = src.goal_origin_to_eef_pose()[:3, :3]
        physical = scale_action(
            native, src.input_min, src.input_max,
            src.output_min, src.output_max,
        )
        target_abs = compose_delta_with_pose(
            baseline_pos, baseline_ori, physical,
        )
        src.set_goal(native)
        tgt.set_goal(target_abs)
        assert np.linalg.norm(src.goal_pos - tgt.goal_pos) < 1e-9
        source_torque = np.asarray(src.run_controller(), dtype=float).copy()
        original_target_reference = np.asarray(tgt.initial_joint).copy()

        # Bad state transfer: action chart correct, nullspace reference stale.
        perturbation = np.array([0.04, -0.03, 0.02, -0.01, 0.03, -0.02, 0.01])
        tgt.initial_joint = np.asarray(src.initial_joint).copy() + perturbation
        bad_torque = np.asarray(tgt.run_controller(), dtype=float).copy()

        # Correct state transfer changes only controller-owned posture memory.
        tgt.initial_joint = np.asarray(src.initial_joint).copy()
        good_torque = np.asarray(tgt.run_controller(), dtype=float).copy()
        tgt.initial_joint = original_target_reference

        bad_error = float(np.max(np.abs(bad_torque - source_torque)))
        good_error = float(np.max(np.abs(good_torque - source_torque)))

        # Frozen physical state + fixed controller goals: 32 independent
        # state-only interventions probe how sensitive low-level torque is
        # to memory mismatch. These are synthetic interventions, not task
        # success rates or independently sampled robot trajectories.
        rng = np.random.default_rng(429)
        intervention_errors = []
        for _ in range(32):
            disturbance = rng.normal(0.0, 0.03, size=7)
            tgt.initial_joint = np.asarray(src.initial_joint).copy() + disturbance
            torque = np.asarray(tgt.run_controller(), dtype=float).copy()
            intervention_errors.append(
                float(np.max(np.abs(torque - source_torque)))
            )
        tgt.initial_joint = original_target_reference
        p10, median, p90 = np.percentile(intervention_errors, [10, 50, 90])
        print(
            f"OSC_MEMORY_ABLATION n=32 p10={p10:.6e} "
            f"median={median:.6e} p90={p90:.6e} "
            f"matched_state={good_error:.6e}"
        )
        assert median > 0.05
        assert p90 > 0.1
        assert median > 1000 * good_error
        print(
            f"OSC_CAUSAL_HANDSHAKE bad_torque_max={bad_error:.6e} "
            f"good_torque_max={good_error:.6e} "
            f"improvement_ratio={bad_error/max(good_error,1e-15):.3f}"
        )
        assert bad_error > 1e-3
        assert good_error < 1e-5
        assert good_error * 100 < bad_error
    finally:
        delta_env.close()
        absolute_env.close()
