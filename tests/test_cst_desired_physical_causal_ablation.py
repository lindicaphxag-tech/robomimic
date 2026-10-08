"""Counterfactual physical rollout: live desired-goal memory vs achieved-only adapter.

All three physics models and initial qpos/qvel are matched by exact source
MJCF/state transfer, and OSC nullspace posture is synchronized.
Actions come from a deterministic stream, NOT a learned policy.
"""
import numpy as np
import pytest

pytest.importorskip("robosuite")
import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)
from research.desired_osc_online_transport import (
    OnlineTransportStatus, compile_live_desired_osc_action,
)
from research.executable_osc_migration import (
    MigrationOutcome, compile_and_apply_osc_posture_handshake, digest_mjcf,
)
from robomimic.scripts.conversion.robosuite_action_conversion import (
    compose_delta_with_pose, scale_action,
)


def make_env(mode):
    cfg=load_composite_controller_config(controller=None,robot="Panda")
    c=cfg["body_parts"]["right"]
    c.update(type="OSC_POSE",input_type=mode,input_ref_frame="world",
             impedance_mode="fixed",goal_update_mode="achieved")
    return suite.make("Lift",robots="Panda",controller_configs=cfg,
                      has_renderer=False,has_offscreen_renderer=False,
                      use_camera_obs=False,control_freq=20,horizon=35)


def action(env, native):
    x=np.zeros_like(env.action_spec[0],dtype=float)
    start,stop=env.robots[0].composite_controller._action_split_indexes["right"]
    x[start:stop]=native
    return x


@pytest.mark.parametrize("seed",[42,270,429])
def test_stateful_vs_achieved_only_absolute_compiler_real_rollout(seed):
    source=make_env("delta")
    good=make_env("absolute")
    bad=make_env("absolute")
    try:
        source.reset()
        mjcf=source.sim.model.get_xml()
        snapshot=source.sim.get_state().flatten()
        good.reset()
        bad.reset()
        for e in (good,bad):
            e.reset_from_xml_string(mjcf)
            e.sim.set_state_from_flattened(snapshot)
            e.sim.forward()
            if hasattr(source.sim.data,"qacc_warmstart"):
                e.sim.data.qacc_warmstart[...] = source.sim.data.qacc_warmstart

        cs=source.robots[0].part_controllers["right"]
        cg=good.robots[0].part_controllers["right"]
        cb=bad.robots[0].part_controllers["right"]
        for e in (source,good,bad):
            e.robots[0].composite_controller.update_state()
            e.robots[0].part_controllers["right"].update(force=True)
        digest=digest_mjcf(mjcf)
        for target in (cg,cb):
            cert=compile_and_apply_osc_posture_handshake(
                cs,target,
                source_mjcf_sha256=digest,target_mjcf_sha256=digest,
                tolerance=1e-7)
            assert cert.outcome is MigrationOutcome.APPLIED, cert.reason
        cs.set_goal_update_mode("desired")
        # Anchor the source's initial goal to the actual achieved pose to
        # rule out reset_goal coordinate-chart inconsistencies.
        cs.goal_pos=np.asarray(cs.ref_pos,dtype=float).copy()
        cs.goal_ori=np.asarray(cs.ref_ori_mat,dtype=float).copy()
        for tgt in (cg,cb):
            tgt.goal_pos=cs.goal_pos.copy()
            tgt.goal_ori=cs.goal_ori.copy()
        np.testing.assert_allclose(source.sim.data.qpos,good.sim.data.qpos,atol=0,rtol=0)
        np.testing.assert_allclose(source.sim.data.qpos,bad.sim.data.qpos,atol=0,rtol=0)

        rng=np.random.default_rng(seed)
        good_state=[]
        bad_state=[]
        good_goal=[]
        bad_goal=[]
        for step in range(16):
            for e in (source,good,bad):
                e.robots[0].composite_controller.update_state()
                e.robots[0].part_controllers["right"].update(force=True)

            native=rng.uniform(-0.2,0.2,size=6)
            correct=compile_live_desired_osc_action(cs,native)
            assert correct.status is OnlineTransportStatus.EXECUTABLE_WITH_STEP_HOOK

            # Naive adapter uses the target's *currently achieved* world
            # pose and cannot access the SOURCE controller's previous goal.
            physical=scale_action(native,cs.input_min,cs.input_max,
                                  cs.output_min,cs.output_max)
            naive=compose_delta_with_pose(
                np.asarray(cb.ref_pos,dtype=float),
                np.asarray(cb.ref_ori_mat,dtype=float),
                physical)

            source.step(action(source,native))
            good.step(action(good,correct.absolute_action))
            bad.step(action(bad,naive))
            qref=np.asarray(source.sim.data.qpos,dtype=float)
            good_state.append(float(np.max(np.abs(qref-good.sim.data.qpos))))
            bad_state.append(float(np.max(np.abs(qref-bad.sim.data.qpos))))
            good_goal.append(float(np.linalg.norm(cs.goal_pos-cg.goal_pos)))
            bad_goal.append(float(np.linalg.norm(cs.goal_pos-cb.goal_pos)))

        report={
            "seed":seed,"steps":16,
            "max_correct_full_scene_qpos":max(good_state),
            "max_naive_full_scene_qpos":max(bad_state),
            "max_correct_goal_pos":max(good_goal),
            "max_naive_goal_pos":max(bad_goal),
            "correct_trace":good_state,
            "naive_trace":bad_state,
        }
        print("CST_DESIRED_PHYSICAL_CAUSAL_ABLATION",report)
        assert max(good_goal)<1e-6
        assert max(good_state)<2e-5
        # Do NOT preset a "naive must fail" threshold: report exact effect.
    finally:
        source.close()
        good.close()
        bad.close()
