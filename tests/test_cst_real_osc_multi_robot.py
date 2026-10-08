"""Cross-task real-simulator verification of a frozen action interface swap.

The policy is a deterministic scripted native-action stream, NOT a learned
frozen checkpoint. Claims are finite-horizon controller-semantic transport
only. Two additional robosuite robot embodiments use identical MJCF and a
state-complete OSC posture handshake.
"""
import numpy as np
import pytest

pytest.importorskip("robosuite")
import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
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


def _sim(task, mode, robot):
    config=load_composite_controller_config(controller=None, robot=robot)
    c=config["body_parts"]["right"]
    c.update(
        type="OSC_POSE",
        input_type=mode,
        input_ref_frame="base",
        impedance_mode="fixed",
        goal_update_mode="achieved",
    )
    return suite.make(
        task, robots=robot, controller_configs=config,
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, control_freq=20, horizon=25,
    )


def _full_action(env, native):
    cmd=np.zeros_like(env.action_spec[0],dtype=float)
    arm=env.robots[0].composite_controller._action_split_indexes["right"]
    cmd[arm[0]:arm[1]]=native
    return cmd


@pytest.mark.parametrize("robot", ["UR5e", "IIWA"])
@pytest.mark.parametrize("task", ["Lift"])
@pytest.mark.parametrize("seed", [42, 270])
def test_state_complete_osc_transport_across_robot_kinematics(task, seed, robot):
    a=_sim(task, "delta", robot)
    b=_sim(task, "absolute", robot)
    try:
        a.reset()
        b.reset()
        mjcf=a.sim.model.get_xml()
        b.reset_from_xml_string(mjcf)
        b.sim.set_state_from_flattened(a.sim.get_state().flatten())
        b.sim.forward()
        if hasattr(a.sim.data,"qacc_warmstart"):
            b.sim.data.qacc_warmstart[...] = a.sim.data.qacc_warmstart

        ca=a.robots[0].part_controllers["right"]
        cb=b.robots[0].part_controllers["right"]
        for e in (a,b):
            e.robots[0].composite_controller.update_state()
            e.robots[0].part_controllers["right"].update(force=True)
        identity=digest_mjcf(mjcf)
        cert=compile_and_apply_osc_posture_handshake(
            ca,cb,source_mjcf_sha256=identity,
            target_mjcf_sha256=identity,tolerance=1e-7,
        )
        assert cert.outcome is MigrationOutcome.APPLIED,cert.reason

        rng=np.random.default_rng(seed)
        arm_errors=[]
        scene_errors=[]
        goal_errors=[]
        for _ in range(8):
            ra=a.robots[0]
            rb=b.robots[0]
            ra.composite_controller.update_state()
            rb.composite_controller.update_state()
            ca.update(force=True)
            cb.update(force=True)

            native=rng.uniform(-0.18,0.18,size=6)
            physical=scale_action(
                native,ca.input_min,ca.input_max,
                ca.output_min,ca.output_max,
            )
            pos=ca.world_to_origin_frame(ca.ref_pos)
            ori=ca.goal_origin_to_eef_pose()[:3,:3]
            abs_cmd=compose_delta_with_pose(pos,ori,physical)
            a.step(_full_action(a,native))
            b.step(_full_action(b,abs_cmd))
            qa=np.asarray(a.sim.data.qpos,dtype=float)
            qb=np.asarray(b.sim.data.qpos,dtype=float)
            scene_errors.append(float(np.max(np.abs(qa-qb))))
            arm_idx=np.asarray(ra._ref_joint_pos_indexes,dtype=int)
            arm_errors.append(float(np.max(np.abs(qa[arm_idx]-qb[arm_idx]))))
            # Position and full matrix match: not raw axis-angle subtraction.
            dp=float(np.max(np.abs(ca.goal_pos-cb.goal_pos)))
            dr=float(np.max(np.abs(ca.goal_ori-cb.goal_ori)))
            goal_errors.append(max(dp,dr))

        values={
            "task":task, "seed":seed,"robot":robot,"horizon":8,
            "max_full_scene_qpos":max(scene_errors),
            "max_arm_qpos":max(arm_errors),
            "max_goal_matrix":max(goal_errors),
        }
        print("CST_MULTI_ROBOT_OSC",values)
        assert max(goal_errors) < 1e-6
        assert max(arm_errors) < 2e-5
        assert max(scene_errors) < 2e-5
    finally:
        a.close()
        b.close()
