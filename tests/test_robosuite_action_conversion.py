import numpy as np
from scipy.spatial.transform import Rotation

from robomimic.scripts.conversion.robosuite_action_conversion import (
    compose_delta_with_pose,
    inverse_scale_action,
    physical_delta_from_absolute_pose,
    scale_action,
)


def _limits():
    input_min = np.full(6, -1.0)
    input_max = np.full(6, 1.0)
    output_min = np.array([-0.05, -0.05, -0.05, -0.5, -0.5, -0.5])
    output_max = -output_min
    return input_min, input_max, output_min, output_max


def test_inverse_scale_round_trip_inside_range():
    input_min, input_max, output_min, output_max = _limits()
    rng = np.random.default_rng(270)

    for _ in range(1000):
        native = rng.uniform(-1.0, 1.0, size=6)
        physical = scale_action(
            native,
            input_min,
            input_max,
            output_min,
            output_max,
        )
        recovered, mask = inverse_scale_action(
            physical,
            input_min,
            input_max,
            output_min,
            output_max,
        )
        assert np.all(mask)
        np.testing.assert_allclose(recovered, native, atol=1e-12)


def test_absolute_pose_inverse_matches_robosuite_left_composition():
    input_min, input_max, output_min, output_max = _limits()
    rng = np.random.default_rng(20261008)

    for _ in range(500):
        baseline_position = rng.uniform(-0.3, 0.3, size=3)
        baseline_orientation = Rotation.random(random_state=rng).as_matrix()
        native = rng.uniform(-0.8, 0.8, size=6)
        physical = scale_action(
            native,
            input_min,
            input_max,
            output_min,
            output_max,
        )
        absolute = compose_delta_with_pose(
            baseline_position,
            baseline_orientation,
            physical,
        )

        recovered_physical = physical_delta_from_absolute_pose(
            absolute,
            baseline_position,
            baseline_orientation,
        )
        recovered_native, mask = inverse_scale_action(
            recovered_physical,
            input_min,
            input_max,
            output_min,
            output_max,
        )

        assert np.all(mask)
        np.testing.assert_allclose(recovered_native, native, atol=1e-10)


def test_inverse_scale_exposes_saturation_instead_of_hiding_it():
    input_min, input_max, output_min, output_max = _limits()
    physical = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])

    native, mask = inverse_scale_action(
        physical,
        input_min,
        input_max,
        output_min,
        output_max,
    )

    assert not mask[0]
    assert np.all(mask[1:])
    assert native[0] == input_max[0]


def test_noncommuting_orientation_uses_group_inverse_not_rotvec_subtraction():
    baseline_orientation = Rotation.from_rotvec(
        [0.4, -0.2, 0.3]
    ).as_matrix()
    physical_delta = np.array([0.0, 0.0, 0.0, -0.3, 0.25, 0.2])
    absolute = compose_delta_with_pose(
        np.zeros(3),
        baseline_orientation,
        physical_delta,
    )

    recovered = physical_delta_from_absolute_pose(
        absolute,
        np.zeros(3),
        baseline_orientation,
    )

    np.testing.assert_allclose(recovered, physical_delta, atol=1e-10)
    naive = absolute[3:6] - Rotation.from_matrix(
        baseline_orientation
    ).as_rotvec()
    assert np.linalg.norm(naive - physical_delta[3:6]) > 1e-3
