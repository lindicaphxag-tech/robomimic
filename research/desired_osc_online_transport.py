"""Live controller-goal memory transport for OSC desired-update semantics.

Only a runtime step hook can supply the previous desired target. In
contrast, a dataset of independent physical states cannot generally recover
this controller-owned memory. This module does not edit a robot policy.
"""
from dataclasses import dataclass
from enum import Enum

import numpy as np

from research.robomimic_reverse_actions import (
    OSCActionContract,
    compose_delta_with_pose,
    scale_native_delta,
)


class OnlineTransportStatus(str, Enum):
    EXECUTABLE_WITH_STEP_HOOK = "executable_with_step_hook"
    REFUSE_MISSING_RUNTIME_STATE = "refuse_missing_runtime_state"


@dataclass(frozen=True)
class DesiredOSCActionResult:
    status: OnlineTransportStatus
    absolute_action: np.ndarray | None
    previous_goal_position: np.ndarray | None
    previous_goal_orientation: np.ndarray | None
    reason: str


def _refuse(reason):
    return DesiredOSCActionResult(
        status=OnlineTransportStatus.REFUSE_MISSING_RUNTIME_STATE,
        absolute_action=None,
        previous_goal_position=None,
        previous_goal_orientation=None,
        reason=reason,
    )


def compile_live_desired_osc_action(controller, native_delta):
    """Compute next absolute OSC action from *current runtime goal memory*.

    Supported: fixed-impedance, pose-controlled, desired goal-update, native
    bounded 6D deltas, no interpolation or variable controller gains.

    This compiler intentionally does not change the controller state. Its
    caller must query it BEFORE source.set_goal/action step, then execute the
    source and absolute target actions in lock-step.
    """
    if getattr(controller, "input_type", None) != "delta":
        return _refuse("source controller must use delta OSC native actions")
    if getattr(controller, "_goal_update_mode", None) != "desired":
        return _refuse("source is not in desired-target update mode")
    if getattr(controller, "impedance_mode", None) != "fixed":
        return _refuse("variable impedance is not covered")
    if getattr(controller, "use_ori", False) is not True:
        return _refuse("position-only OSC is outside this pose contract")
    if (
        getattr(controller, "interpolator_pos", None) is not None
        or getattr(controller, "interpolator_ori", None) is not None
    ):
        return _refuse("interpolation adds unhandled controller memory")

    goal_pos = getattr(controller, "goal_pos", None)
    goal_ori = getattr(controller, "goal_ori", None)
    if goal_pos is None or goal_ori is None:
        return _refuse("previous desired pose is not present in the live controller")

    try:
        pos = np.asarray(goal_pos, dtype=float)
        ori = np.asarray(goal_ori, dtype=float)
        native = np.asarray(native_delta, dtype=float)
        if pos.shape != (3,) or ori.shape != (3, 3) or native.shape != (6,):
            return _refuse("wrong controller-goal/action memory dimensions")
        if not (
            np.all(np.isfinite(pos))
            and np.all(np.isfinite(ori))
            and np.all(np.isfinite(native))
        ):
            return _refuse("nonfinite controller-goal/action memory")

        contract = OSCActionContract(
            input_min=np.asarray(controller.input_min, dtype=float),
            input_max=np.asarray(controller.input_max, dtype=float),
            output_min=np.asarray(controller.output_min, dtype=float),
            output_max=np.asarray(controller.output_max, dtype=float),
            goal_update_mode="desired",
        )
        if np.any(native < contract.input_min) or np.any(native > contract.input_max):
            return _refuse("native action outside controller input limits; clipping is not exact")
        physical_delta = scale_native_delta(native, contract)
        absolute = compose_delta_with_pose(pos, ori, physical_delta)
        return DesiredOSCActionResult(
            status=OnlineTransportStatus.EXECUTABLE_WITH_STEP_HOOK,
            absolute_action=absolute,
            previous_goal_position=pos.copy(),
            previous_goal_orientation=ori.copy(),
            reason="used live desired controller target memory",
        )
    except (AttributeError, TypeError, ValueError, FloatingPointError) as exc:
        return _refuse(f"runtime controller contract invalid: {exc}")
