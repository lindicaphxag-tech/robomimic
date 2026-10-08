"""Real controller online desired-OSC target memory transport.

No video, no hardware, no learned policy. Exercises controller-owned
previous-goal memory in Panda/Lift using real MuJoCo dynamics.
"""
import numpy as np
import pytest

pytest.importorskip("robosuite")
import robosuite as suite
from scipy.spatial.transform import Rotation
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)

from research.desired_osc_online_transport import (
    OnlineTransportStatus,
    compile_live_desired_osc_action,
)
from research.executable_osc_migration import (
    MigrationOutcome,
    compile_and_apply_osc_posture_handshake,
    digest_mjcf,
)
from robomimic.scripts.conversion.robosuite_action_conversion import (
    compose_delta_with_pose,
    scale_action,
)


def _make_env(mode):
    cfg = load_composite_controller_config(controller=None, robot="Panda")
    c = cfg["body_parts"]["right"]
    c["type"] = "OSC_POSE"
    c["input_type"] = mode
    c["input_ref_frame"] = "world"
    c["impedance_mode"] = "fixed"
    return suite.make(
        "Lift", robots="Panda", controller_configs=cfg,
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, control_freq=20, horizon=25,
    )


def _full_action(env, arm):
    x = np.zeros_like(env.action_spec[0], dtype=float)
    a,b = env.robots[0].composite_controller._action_split_indexes["right"]
    x[a:b] = arm
    return x


@pytest.mark.parametrize("seed", [42, 270])
def test_real_online_desired_goal_memory_transport(seed):
    src = _make_env("delta")
    dst = _make_env("absolute")
    try:
        src.reset()
        dst.reset()
        mjcf = src.sim.model.get_xml()
        dst.reset_from_xml_string(mjcf)
        dst.sim.set_state_from_flattened(src.sim.get_state().flatten())
        dst.sim.forward()
        if hasattr(src.sim.data, "qacc_warmstart"):
            dst.sim.data.qacc_warmstart[...] = src.sim.data.qacc_warmstart

        cs = src.robots[0].part_controllers["right"]
        ct = dst.robots[0].part_controllers["right"]
        for env in (src, dst):
            env.robots[0].composite_controller.update_state()
            env.robots[0].part_controllers["right"].update(force=True)

        sha = digest_mjcf(mjcf)
        handshake=compile_and_apply_osc_posture_handshake(
            cs,ct,source_mjcf_sha256=sha,
            target_mjcf_sha256=sha,tolerance=1e-7,
        )
        assert handshake.outcome is MigrationOutcome.APPLIED

        # First state-handshake was done in achieved mode. Now intentionally
        # switch SOURCE to desired mode, making previous-goal memory causal.
        cs.set_goal_update_mode("desired")

        # Align the INITIAL desired goal to the measured pose IN THE
        # declared input reference frame. Otherwise reset_goal() may leave
        # a stale coordinate-chart target and confound the history effect.
        cs.update(force=True)
        if cs.input_ref_frame == "base":
            initial_goal_pos = cs.world_to_origin_frame(cs.ref_pos)
            initial_goal_ori = cs.goal_origin_to_eef_pose()[:3, :3]
        elif cs.input_ref_frame == "world":
            initial_goal_pos = cs.ref_pos
            initial_goal_ori = cs.ref_ori_mat
        else:
            raise AssertionError("unsupported reference frame")
        cs.goal_pos = np.asarray(initial_goal_pos, dtype=float).copy()
        cs.goal_ori = np.asarray(initial_goal_ori, dtype=float).copy()
        ct.goal_pos = cs.goal_pos.copy()
        ct.goal_ori = cs.goal_ori.copy()
        np.testing.assert_allclose(cs.goal_pos, initial_goal_pos, atol=1e-12)
        print(
            "DESIRED_INITIAL_GOAL_ALIGNMENT",
            cs.input_ref_frame,
            np.asarray(cs.goal_pos).tolist(),
            np.asarray(initial_goal_pos).tolist(),
        )
        rng = np.random.default_rng(seed)
        full_errors=[]
        goal_errors=[]
        stale_goal_mistakes=[]
        for _ in range(8):
            src.robots[0].composite_controller.update_state()
            dst.robots[0].composite_controller.update_state()
            cs.update(force=True)
            ct.update(force=True)
            native=rng.uniform(-0.18,0.18,size=6)
            result=compile_live_desired_osc_action(cs,native)
            assert result.status is OnlineTransportStatus.EXECUTABLE_WITH_STEP_HOOK,result.reason

            # Deliberate invalid shortcut: achieved-state baseline ignores
            # controller-owned previous desired target; measure separately.
            achieved_pos=cs.world_to_origin_frame(cs.ref_pos)
            achieved_ori=cs.goal_origin_to_eef_pose()[:3,:3]
            physical=scale_action(
                native,cs.input_min,cs.input_max,
                cs.output_min,cs.output_max,
            )
            naive=compose_delta_with_pose(achieved_pos,achieved_ori,physical)
            stale_goal_mistakes.append(float(np.linalg.norm(naive[:3]-result.absolute_action[:3])))

            src.step(_full_action(src,native))
            dst.step(_full_action(dst,result.absolute_action))
            pos_err=float(np.linalg.norm(cs.goal_pos-ct.goal_pos))
            ori_err=float(Rotation.from_matrix(cs.goal_ori @ ct.goal_ori.T).magnitude())
            goal_errors.append(max(pos_err,ori_err))
            full_errors.append(float(np.max(np.abs(src.sim.data.qpos-dst.sim.data.qpos))))

        print(
            "CST_REAL_DESIRED_STREAM",
            {"seed":seed,"steps":8,
             "full_scene_max":max(full_errors),
             "goal_max":max(goal_errors),
             "naive_achieved_baseline_discrepancy_max":max(stale_goal_mistakes)}
        )
        assert max(goal_errors) < 1e-6
        assert max(full_errors) < 2e-5
        # This must arise from accumulated desired-goal memory after
        # matching the initial target to measured pose, not reset artifacts.
        assert max(stale_goal_mistakes[1:]) > 1e-3
    finally:
        src.close()
        dst.close()
