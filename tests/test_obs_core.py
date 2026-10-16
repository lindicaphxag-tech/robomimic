import numpy as np
import torch

from robomimic.models.obs_core import CropRandomizer


def test_crop_randomizer_eval_preserves_position_encoding():
    randomizer = CropRandomizer(
        input_shape=(3, 8, 10),
        crop_height=4,
        crop_width=6,
        num_crops=2,
        pos_enc=True,
    )
    inputs = torch.zeros(2, 3, 8, 10)

    out = randomizer._forward_in_eval(inputs)

    assert tuple(out.shape) == (2, 5, 4, 6)

    # Center crop starts at (2, 2). Position channels must encode source
    # coordinates, not crop-local coordinates.
    np.testing.assert_allclose(out[0, 3].numpy(), np.arange(2, 6)[:, None] / 8.0)
    np.testing.assert_allclose(out[0, 4].numpy(), np.arange(2, 8)[None, :] / 10.0)


def test_crop_randomizer_eval_without_position_encoding_is_unchanged():
    randomizer = CropRandomizer(
        input_shape=(3, 8, 10),
        crop_height=4,
        crop_width=6,
        num_crops=2,
        pos_enc=False,
    )
    inputs = torch.randn(2, 3, 8, 10)

    out = randomizer._forward_in_eval(inputs)

    assert tuple(out.shape) == (2, 4, 6)
