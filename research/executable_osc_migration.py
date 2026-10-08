"""Fail-closed executable controller-state migration for fixed-impedance OSC.

Research prototype, not a replacement for robosuite's controller or a global
robot safety guarantee. A caller must provide a common MJCF identity witness.
The compiler deliberately rejects controller pairs whose hidden state is not
covered by its narrow, tested handshake contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import re

import numpy as np


class OSCMigrationRollbackError(RuntimeError):
    """Raised when a failed state transfer cannot be safely undone."""


class MigrationOutcome(str, Enum):
    APPLIED = "applied"
    REFUSED = "refused"


@dataclass(frozen=True)
class OSCHandshakeCertificate:
    outcome: MigrationOutcome
    reason: str
    common_model_sha256: str | None
    source_reference_before: tuple[float, ...] | None
    target_reference_before: tuple[float, ...] | None
    target_reference_after: tuple[float, ...] | None
    maximum_reference_residual: float | None


def digest_mjcf(xml_bytes: str | bytes) -> str:
    """Compute a provenance digest of the *same source MJCF bytes*.

    This is a reproducibility fingerprint, not a proof that MuJoCo
    world states or solver internal states are identical.
    """
    data = xml_bytes.encode("utf-8") if isinstance(xml_bytes, str) else bytes(xml_bytes)
    return hashlib.sha256(data).hexdigest()


def _reject(reason: str) -> OSCHandshakeCertificate:
    return OSCHandshakeCertificate(
        outcome=MigrationOutcome.REFUSED,
        reason=reason,
        common_model_sha256=None,
        source_reference_before=None,
        target_reference_before=None,
        target_reference_after=None,
        maximum_reference_residual=None,
    )


def _vector(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite nonempty 1-D vector")
    return array


def compile_and_apply_osc_posture_handshake(
    source,
    target,
    *,
    source_mjcf_sha256: str,
    target_mjcf_sha256: str,
    tolerance: float = 1e-10,
) -> OSCHandshakeCertificate:
    """Apply narrow OSC nullspace-posture state transfer, or refuse.

    Preconditions:
    - identical *provenance* of source/target MJCF (caller-calculated);
    - achieved-pose update mode, fixed impedance, no interpolation memory;
    - matching OSC gain/damping and reference-frame semantics;
    - same number of controller joints and matching joint state at the point
      the handshake runs. This is a static point-in-time local obligation.

    Not proved: complete physics-state identity, arbitrary controller
    equivalence, arbitrary contact dynamics or future successful episodes.
    """
    if not (np.isfinite(tolerance) and tolerance >= 0):
        return _reject("invalid numeric comparison tolerance")
    if (
        not isinstance(source_mjcf_sha256, str)
        or not isinstance(target_mjcf_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_mjcf_sha256) is None
        or source_mjcf_sha256 != target_mjcf_sha256
    ):
        return _reject("source and target do not share a verified MJCF provenance digest")

    required = (
        "initial_joint", "goal_pos", "goal_ori",
        "kp", "kd", "joint_pos", "joint_vel",
        "update_initial_joints", "input_type", "input_ref_frame",
        "_goal_update_mode", "impedance_mode",
    )
    for role, controller in (("source", source), ("target", target)):
        missing = [name for name in required if not hasattr(controller, name)]
        if missing:
            return _reject(f"{role} OSC controller missing required state: {missing}")

    if (source.input_type, target.input_type) != ("delta", "absolute"):
        return _reject("only delta OSC -> absolute OSC target handshake is supported")
    if source._goal_update_mode != "achieved" or target._goal_update_mode != "achieved":
        return _reject("desired-goal memory is unsupported by this handshake")
    if source.impedance_mode != "fixed" or target.impedance_mode != "fixed":
        return _reject("variable impedance adds uncontrolled controller state")
    if source.input_ref_frame != target.input_ref_frame:
        return _reject("source and target use different OSC reference frames")
    for role, ctrl in (("source", source), ("target", target)):
        if getattr(ctrl, "interpolator_pos", None) is not None or getattr(ctrl, "interpolator_ori", None) is not None:
            return _reject(f"{role} contains unsupported interpolation memory")

    try:
        src_initial = _vector(source.initial_joint, "source.initial_joint").copy()
        dst_initial = _vector(target.initial_joint, "target.initial_joint").copy()

        # robosuite OSC.update_initial_joints() also calls reset_goal(), so
        # controller-owned target memory MUST be part of the transaction.
        src_goal_pos = _vector(source.goal_pos, "source.goal_pos").copy()
        dst_goal_pos = _vector(target.goal_pos, "target.goal_pos").copy()
        src_goal_ori = np.asarray(source.goal_ori, dtype=float).copy()
        dst_goal_ori = np.asarray(target.goal_ori, dtype=float).copy()
        if (
            src_goal_pos.shape != (3,)
            or dst_goal_pos.shape != (3,)
            or src_goal_ori.shape != (3, 3)
            or dst_goal_ori.shape != (3, 3)
            or not np.all(np.isfinite(src_goal_ori))
            or not np.all(np.isfinite(dst_goal_ori))
        ):
            return _reject("OSC target pose memory is incomplete or nonfinite")
        original_mode = target._goal_update_mode

        if src_initial.shape != dst_initial.shape:
            return _reject("source and target have different nullspace joint dimensions")

        for name in ("kp", "kd", "joint_pos", "joint_vel"):
            first = _vector(getattr(source, name), "source." + name)
            second = _vector(getattr(target, name), "target." + name)
            if first.shape != second.shape or not np.allclose(first, second, atol=tolerance, rtol=0):
                return _reject(f"source/target {name} contract differs")

    except (ValueError, AttributeError, TypeError) as exc:
        return _reject(f"state extraction failed before mutation: {exc}")

    # The model and gain checks are necessary but insufficient.  A rejected
    # handshake must not leave even INDIRECT reset_goal side effects behind.
    try:
        target.update_initial_joints(src_initial.copy())
        # update_initial_joints resets goal in the real OSC implementation:
        # explicitly synchronize the controller-owned goal memory as well.
        target.goal_pos = src_goal_pos.copy()
        target.goal_ori = src_goal_ori.copy()
        target._goal_update_mode = "achieved"

        transferred = _vector(target.initial_joint, "target.initial_joint")
        if transferred.shape != src_initial.shape:
            raise ValueError("post-transfer nullspace dimensions differ")
        error = float(np.max(np.abs(src_initial - transferred)))
        if error > tolerance:
            raise ValueError("target did not retain transferred posture reference")
        if (
            not np.array_equal(np.asarray(target.goal_pos), src_goal_pos)
            or not np.array_equal(np.asarray(target.goal_ori), src_goal_ori)
            or target._goal_update_mode != "achieved"
        ):
            raise ValueError("target did not retain synchronized OSC goal memory")
    except Exception as exc:
        # A failed write may partially apply and reset additional state.
        # Restore and verify *all* mutable fields covered by this contract.
        try:
            target.update_initial_joints(dst_initial.copy())
            target.goal_pos = dst_goal_pos.copy()
            target.goal_ori = dst_goal_ori.copy()
            target._goal_update_mode = original_mode
            restored = _vector(target.initial_joint, "rollback.initial_joint")
            if (
                restored.shape != dst_initial.shape
                or not np.array_equal(restored, dst_initial)
                or not np.array_equal(np.asarray(target.goal_pos), dst_goal_pos)
                or not np.array_equal(np.asarray(target.goal_ori), dst_goal_ori)
                or target._goal_update_mode != original_mode
            ):
                raise OSCMigrationRollbackError(
                    "controller posture/goal memory was not restored exactly"
                )
        except Exception as rollback_exc:
            raise OSCMigrationRollbackError(
                "controller state transfer failed AND full-state rollback could not "
                "be verified; target controller must be quarantined from further execution"
            ) from rollback_exc
        return _reject(f"controller mutation rolled back after failure: {exc}")

    return OSCHandshakeCertificate(
        outcome=MigrationOutcome.APPLIED,
        reason="OSC nullspace and goal memory synchronized within declared scope",
        common_model_sha256=source_mjcf_sha256,
        source_reference_before=tuple(map(float, src_initial)),
        target_reference_before=tuple(map(float, dst_initial)),
        target_reference_after=tuple(map(float, transferred)),
        maximum_reference_residual=error,
    )
