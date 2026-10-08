"""Negative control: two SAME OSC controller interfaces, separate simulators.

If identical-controller paired simulation shows free-object drift despite
identical physics state snapshots and motor commands, this estimates simulator
confounding rather than action-adapter error.
"""
import numpy as np
import pytest

pytest.importorskip("robosuite")
import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)


def _delta_env():
    cfg = load_composite_controller_config(controller=None, robot="Panda")
    c = cfg["body_parts"]["right"]
    c["type"] = "OSC_POSE"
    c["input_type"] = "delta"
    c["input_ref_frame"] = "base"
    c["impedance_mode"] = "fixed"
    return suite.make(
        "Lift", robots="Panda", controller_configs=cfg,
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, control_freq=20, horizon=25,
    )


def test_identical_controller_instances_baseline_scene_drift():
    a = _delta_env()
    b = _delta_env()
    try:
        a.reset()
        b.reset()
        b.sim.set_state_from_flattened(a.sim.get_state().flatten())
        b.sim.forward()
        ca = a.robots[0].part_controllers["right"]
        cb = b.robots[0].part_controllers["right"]
        cb.update_initial_joints(np.asarray(ca.initial_joint).copy())
        np.testing.assert_allclose(
            a.sim.get_state().flatten(),
            b.sim.get_state().flatten(),
            atol=1e-12,
        )

        for name in ("geom_size", "geom_friction", "body_pos", "dof_damping"):
            ma = getattr(a.sim.model, name, None)
            mb = getattr(b.sim.model, name, None)
            if ma is not None and mb is not None:
                xa, xb = np.asarray(ma), np.asarray(mb)
                if xa.shape == xb.shape:
                    print(
                        "MODEL_NEG_CONTROL", name,
                        float(np.max(np.abs(xa - xb)))
                    )

        robot_joints = np.asarray(a.robots[0]._ref_joint_pos_indexes, dtype=int)
        rng = np.random.default_rng(270)
        full_errors = []
        arm_errors = []
        ctrl_errors = []
        for _ in range(8):
            native = rng.uniform(-0.2, 0.2, size=6)
            controls = []
            for env in (a, b):
                robot = env.robots[0]
                action = np.zeros_like(env.action_spec[0])
                start, stop = robot.composite_controller._action_split_indexes["right"]
                action[start:stop] = native
                env.step(action)
                controls.append(np.asarray(env.sim.data.ctrl).copy())
            q_a = np.asarray(a.sim.data.qpos).copy()
            q_b = np.asarray(b.sim.data.qpos).copy()
            full_errors.append(float(np.max(np.abs(q_a - q_b))))
            arm_errors.append(
                float(np.max(np.abs(q_a[robot_joints] - q_b[robot_joints])))
            )
            ctrl_errors.append(
                float(np.max(np.abs(controls[0] - controls[1])))
            )

        print(
            "IDENTICAL_CONTROLLER_NEG_CONTROL",
            "steps=8",
            "full_qpos_max", max(full_errors),
            "arm_qpos_max", max(arm_errors),
            "ctrl_max", max(ctrl_errors),
        )
        # Robot motor control should match nearly exactly. No full-scene
        # equivalence is asserted; that is the hypothesis under inspection.
        assert max(arm_errors) < 1e-5
        assert max(ctrl_errors) < 1e-3
    finally:
        a.close()
        b.close()
