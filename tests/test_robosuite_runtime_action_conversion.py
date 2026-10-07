import numpy as np
import robosuite as suite

from robomimic.scripts.conversion.robosuite_action_conversion import (
    compose_delta_with_pose,
)
from robomimic.scripts.conversion.robosuite_add_delta_actions import (
    _absolute_pose_to_delta,
    _controller_achieved_pose,
)


def test_real_robosuite_osc_runtime_round_trip():
    controller_config = suite.load_composite_controller_config(
        controller="BASIC",
        robot="Panda",
    )
    env = suite.make(
        "Lift",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
    )
    try:
        env.reset()
        controller = env.robots[0].part_controllers["right"]
        assert controller.input_type == "delta"
        assert getattr(controller, "impedance_mode", "fixed") == "fixed"

        baseline_position, baseline_orientation = _controller_achieved_pose(
            controller
        )
        native_delta = np.array([0.2, -0.1, 0.3, 0.15, -0.2, 0.1])
        physical_delta = controller.scale_action(native_delta)
        absolute_pose = compose_delta_with_pose(
            baseline_position,
            baseline_orientation,
            physical_delta,
        )

        recovered, recovered_physical, representable = _absolute_pose_to_delta(
            controller,
            absolute_pose,
        )

        assert np.all(representable)
        np.testing.assert_allclose(
            recovered_physical,
            physical_delta,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            recovered,
            native_delta,
            atol=1e-8,
        )
    finally:
        env.close()
