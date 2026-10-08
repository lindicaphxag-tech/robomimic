"""Independent robot-family CPU test of transactional OSC state/action transport.

Sawyer (7 DOF) and UR5e (6 DOF) use genuine robosuite controllers
and physics; the original Panda proof-of-concept was deliberately
not reused as the only robot. This is a finite 8-step learned-free test.
"""
import numpy as np
import pytest
import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config

from research.executable_osc_migration import (
    MigrationOutcome, compile_and_apply_osc_posture_handshake, digest_mjcf,
)
from robomimic.scripts.conversion.robosuite_action_conversion import (
    compose_delta_with_pose, scale_action,
)


def make_env(robot, mode, task):
    controller=load_composite_controller_config(controller=None, robot=robot)
    right=controller["body_parts"]["right"]
    right.update(type="OSC_POSE", input_type=mode,
                 input_ref_frame="base", impedance_mode="fixed",
                 goal_update_mode="achieved")
    return suite.make(task, robots=robot, controller_configs=controller,
                      has_renderer=False, has_offscreen_renderer=False,
                      use_camera_obs=False, control_freq=20, horizon=25)


def native_command(env, arm_command):
    action=np.zeros_like(env.action_spec[0],dtype=float)
    start,end=env.robots[0].composite_controller._action_split_indexes["right"]
    action[start:end]=arm_command
    return action


@pytest.mark.parametrize("robot",["Sawyer","UR5e"])
@pytest.mark.parametrize("task",["Lift","Stack"])
@pytest.mark.parametrize("action_seed",[42,429])
def test_same_model_different_action_charts_two_other_robots(robot,task,action_seed):
    source=make_env(robot, "delta", task)
    target=make_env(robot, "absolute", task)
    try:
        source.reset()
        target.reset()
        model=source.sim.model.get_xml()
        target.reset_from_xml_string(model)
        target.sim.set_state_from_flattened(source.sim.get_state().flatten())
        target.sim.forward()
        if hasattr(source.sim.data,"qacc_warmstart"):
            target.sim.data.qacc_warmstart[...]=source.sim.data.qacc_warmstart
        np.testing.assert_array_equal(source.sim.data.qpos,target.sim.data.qpos)
        np.testing.assert_array_equal(source.sim.data.qvel,target.sim.data.qvel)
        src=source.robots[0].part_controllers["right"]
        dst=target.robots[0].part_controllers["right"]
        for e in (source,target):
            e.robots[0].composite_controller.update_state()
            e.robots[0].part_controllers["right"].update(force=True)
        hash=digest_mjcf(model)
        cert=compile_and_apply_osc_posture_handshake(
            src,dst,source_mjcf_sha256=hash,target_mjcf_sha256=hash,
            tolerance=1e-7)
        assert cert.outcome is MigrationOutcome.APPLIED,cert.reason

        rng=np.random.default_rng(action_seed)
        errors=[]
        goal_errors=[]
        for step in range(8):
            for e in (source,target):
                e.robots[0].composite_controller.update_state()
                e.robots[0].part_controllers["right"].update(force=True)
            native=rng.uniform(-0.18,0.18,size=6)
            physical=scale_action(native,src.input_min,src.input_max,
                                  src.output_min,src.output_max)
            pos=src.world_to_origin_frame(src.ref_pos)
            ori=src.goal_origin_to_eef_pose()[:3,:3]
            absolute=compose_delta_with_pose(pos,ori,physical)
            source.step(native_command(source,native))
            target.step(native_command(target,absolute))
            errors.append(float(np.max(np.abs(source.sim.data.qpos-target.sim.data.qpos))))
            goal_errors.append(float(max(
                np.max(np.abs(src.goal_pos-dst.goal_pos)),
                np.max(np.abs(src.goal_ori-dst.goal_ori)),
            )))
        result={"robot":robot,"task":task,"action_seed":action_seed,
                "source_model_sha256":hash,"controller_joint_dimension":len(src.initial_joint),
                "max_full_scene_qpos":max(errors),"max_goal_matrix":max(goal_errors),
                "per_step_scene_qpos":errors}
        print("CST_CROSS_ROBOT_OSC",result)
        assert max(goal_errors)<1e-6
        assert max(errors)<2e-5
    finally:
        source.close()
        target.close()


def test_reject_cross_robot_state_without_mutating_destination():
    source=make_env("Sawyer","delta","Lift")
    target=make_env("UR5e","absolute","Lift")
    try:
        source.reset()
        target.reset()
        src=source.robots[0].part_controllers["right"]
        dst=target.robots[0].part_controllers["right"]
        old=np.asarray(dst.initial_joint).copy()
        sha1=digest_mjcf(source.sim.model.get_xml())
        sha2=digest_mjcf(target.sim.model.get_xml())
        assert sha1!=sha2
        cert=compile_and_apply_osc_posture_handshake(
            src,dst,source_mjcf_sha256=sha1,
            target_mjcf_sha256=sha2)
        assert cert.outcome is MigrationOutcome.REFUSED
        np.testing.assert_array_equal(np.asarray(dst.initial_joint),old)
        print("CST_CROSS_ROBOT_REFUSAL",{"status":cert.outcome.value,"reason":cert.reason})
    finally:
        source.close()
        target.close()
