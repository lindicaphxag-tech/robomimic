"""Verify executable CST handshake against REAL robosuite OSC controllers."""
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


def _make_env(mode):
    cfg = load_composite_controller_config(controller=None, robot="Panda")
    arm = cfg["body_parts"]["right"]
    arm["type"] = "OSC_POSE"
    arm["input_type"] = mode
    arm["input_ref_frame"] = "base"
    arm["impedance_mode"] = "fixed"
    arm["goal_update_mode"] = "achieved"
    return suite.make(
        "Lift", robots="Panda", controller_configs=cfg,
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, control_freq=20, horizon=25,
    )


def test_runtime_osc_state_handshake_with_identical_mjcf():
    source = _make_env("delta")
    target = _make_env("absolute")
    try:
        source.reset()
        target.reset()
        common_xml = source.sim.model.get_xml()
        target.reset_from_xml_string(common_xml)
        target.sim.set_state_from_flattened(
            source.sim.get_state().flatten()
        )
        target.sim.forward()
        for env in (source, target):
            robot=env.robots[0]
            robot.composite_controller.update_state()
            robot.part_controllers["right"].update(force=True)

        src=source.robots[0].part_controllers["right"]
        dst=target.robots[0].part_controllers["right"]
        model_hash=digest_mjcf(common_xml)
        cert=compile_and_apply_osc_posture_handshake(
            src,dst,
            source_mjcf_sha256=model_hash,
            target_mjcf_sha256=model_hash,
            tolerance=1e-7,
        )
        print("OSC_RUNTIME_HANDSHAKE",cert)
        assert cert.outcome is MigrationOutcome.APPLIED
        np.testing.assert_allclose(src.initial_joint,dst.initial_joint,atol=0)

        wrong_hash=digest_mjcf("<mujoco different='yes'/>")
        bad=compile_and_apply_osc_posture_handshake(
            src,dst,
            source_mjcf_sha256=model_hash,
            target_mjcf_sha256=wrong_hash,
        )
        assert bad.outcome is MigrationOutcome.REFUSED
        assert "MJCF" in bad.reason
        print("OSC_RUNTIME_REFUSAL",bad.reason)
    finally:
        source.close()
        target.close()
