"""Checkpoint-provided model names are data, never executable code."""
import pytest

from robomimic.models.base_nets import ResNet18Conv, SpatialMeanPool, SpatialSoftmax
from robomimic.models.obs_core import (
    VisualCore,
    _registered_backbone_class,
    _registered_pool_class,
)


def test_standard_registered_checkpoint_networks_still_resolve():
    assert _registered_backbone_class("ResNet18Conv") is ResNet18Conv
    assert _registered_pool_class("SpatialSoftmax") is SpatialSoftmax
    assert _registered_pool_class("SpatialMeanPool") is SpatialMeanPool


@pytest.mark.parametrize("bad", [
    "SpatialSoftmax.__class__",
    "SpatialMeanPool.__mro__",
    "NotARegisteredPoolingLayer",
    "__import__",
])
def test_checkpoint_name_expressions_and_unknown_classes_fail_closed(bad):
    with pytest.raises(ValueError, match="Unknown or invalid registered pool"):
        _registered_pool_class(bad)


def test_checkpoint_names_must_be_strings_not_objects():
    with pytest.raises(TypeError, match="pool_class"):
        _registered_pool_class(SpatialSoftmax)
    with pytest.raises(TypeError, match="backbone_class"):
        _registered_backbone_class(ResNet18Conv)


def test_checkpoint_backbone_expressions_are_not_evaluated():
    with pytest.raises(ValueError, match="Unknown or invalid registered backbone"):
        _registered_backbone_class("ResNet18Conv.__class__")


def test_visualcore_rejects_unregistered_backbone_even_before_construction():
    with pytest.raises(ValueError, match="Unknown or invalid registered backbone"):
        VisualCore(input_shape=(3, 64, 64), backbone_class="ResNet18Conv.__class__")


def test_visualcore_still_supports_builtin_mean_pooling():
    model = VisualCore(
        input_shape=(3, 64, 64),
        backbone_class="ResNet18Conv",
        backbone_kwargs={"pretrained": False},
        pool_class="SpatialMeanPool",
        feature_dimension=None,
    )
    assert isinstance(model.pool, SpatialMeanPool)


def test_visualcore_still_supports_builtin_spatial_softmax():
    model = VisualCore(
        input_shape=(3, 64, 64),
        backbone_class="ResNet18Conv",
        backbone_kwargs={"pretrained": False},
        pool_class="SpatialSoftmax",
        pool_kwargs={"num_kp": 4},
        feature_dimension=None,
    )
    assert isinstance(model.pool, SpatialSoftmax)


def test_visualcore_fails_closed_on_pool_expression_after_valid_backbone():
    with pytest.raises(ValueError, match="Unknown or invalid registered pool"):
        VisualCore(
            input_shape=(3, 64, 64),
            backbone_kwargs={"pretrained": False},
            pool_class="SpatialSoftmax.__class__",
            feature_dimension=None,
        )
