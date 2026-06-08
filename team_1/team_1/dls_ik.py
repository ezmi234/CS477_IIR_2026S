"""Damped Least Squares and singularity-aware IK helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np


DEFAULT_JOINT_LIMITS = [(-2.0 * math.pi, 2.0 * math.pi)] * 6


@dataclass
class IKDiagnostics:
    ik_method: str
    sigma_min: float | None = None
    condition_number: float | None = None
    damping_lambda: float = 0.0
    joint_delta_norm: float = 0.0
    max_joint_delta: float = 0.0
    pose_error: float | None = None
    singularity_warning: bool = False
    accepted: bool = False
    reject_reason: str | None = None
    score: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ik_method": self.ik_method,
            "sigma_min": self.sigma_min,
            "condition_number": self.condition_number,
            "lambda": float(self.damping_lambda),
            "joint_delta_norm": float(self.joint_delta_norm),
            "max_joint_delta": float(self.max_joint_delta),
            "pose_error": self.pose_error,
            "singularity_warning": bool(self.singularity_warning),
            "accepted": bool(self.accepted),
            "reject_reason": self.reject_reason,
            "score": self.score,
            "extra": self.extra,
        }


def normalize_angle(angle: float) -> float:
    value = float(angle)
    while value > math.pi:
        value -= 2.0 * math.pi
    while value < -math.pi:
        value += 2.0 * math.pi
    return value


def angle_difference(a: float, b: float) -> float:
    return normalize_angle(float(a) - float(b))


def joint_delta(a, b) -> np.ndarray:
    aa = np.asarray(a, dtype=float).flatten()
    bb = np.asarray(b, dtype=float).flatten()
    return np.asarray([angle_difference(x, y) for x, y in zip(aa, bb)], dtype=float)


def compute_jacobian(kin, joints) -> np.ndarray | None:
    q = np.asarray(joints, dtype=float).flatten()
    for call in (
        lambda: kin.jacobian(q),
        lambda: kin.jacobian(q.tolist()),
    ):
        try:
            jacobian = call()
            arr = np.asarray(jacobian, dtype=float)
            if arr.shape[0] >= 6 and arr.shape[1] >= q.shape[0]:
                return arr[:6, : q.shape[0]]
            if arr.shape[1] >= 6 and arr.shape[0] >= q.shape[0]:
                return arr.T[:6, : q.shape[0]]
        except Exception:
            continue
    return None


def jacobian_metrics(
    kin,
    joints,
    *,
    sigma_threshold: float = 0.02,
    condition_threshold: float = 500.0,
    lambda_base: float = 0.04,
) -> tuple[np.ndarray | None, IKDiagnostics]:
    jacobian = compute_jacobian(kin, joints)
    diag = IKDiagnostics(ik_method="jacobian")
    if jacobian is None:
        diag.reject_reason = "jacobian_unavailable"
        return None, diag
    try:
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
    except Exception as exc:
        diag.reject_reason = f"svd_failed:{exc}"
        return jacobian, diag
    sigma_min = float(np.min(singular_values)) if singular_values.size else 0.0
    sigma_max = float(np.max(singular_values)) if singular_values.size else 0.0
    condition = float("inf") if sigma_min <= 1e-9 else sigma_max / sigma_min
    if sigma_min < float(sigma_threshold):
        lambda_sq = float(lambda_base) ** 2 * (1.0 + (float(sigma_threshold) - sigma_min) / float(sigma_threshold))
    else:
        lambda_sq = float(lambda_base) ** 2
    diag.sigma_min = sigma_min
    diag.condition_number = condition
    diag.damping_lambda = math.sqrt(max(0.0, lambda_sq))
    diag.singularity_warning = bool(sigma_min < float(sigma_threshold) or condition > float(condition_threshold))
    return jacobian, diag


def joint_limit_margin(joints, joint_limits=DEFAULT_JOINT_LIMITS) -> float:
    margins = []
    for value, (lower, upper) in zip(np.asarray(joints, dtype=float).flatten(), joint_limits):
        margins.append(min(float(value) - float(lower), float(upper) - float(value)))
    return float(min(margins or [0.0]))


def within_joint_limits(joints, joint_limits=DEFAULT_JOINT_LIMITS) -> bool:
    for value, (lower, upper) in zip(np.asarray(joints, dtype=float).flatten(), joint_limits):
        if float(value) < float(lower) or float(value) > float(upper):
            return False
    return True


def score_candidate(
    kin,
    candidate,
    current,
    home,
    *,
    joint_limits=DEFAULT_JOINT_LIMITS,
    sigma_threshold: float = 0.02,
    condition_threshold: float = 500.0,
    max_joint_delta: float = 2.75,
    max_wrist_flip: float = math.pi,
) -> IKDiagnostics:
    q = np.asarray(candidate, dtype=float).flatten()
    current_q = np.asarray(current, dtype=float).flatten()
    home_q = np.asarray(home, dtype=float).flatten()
    _jacobian, diag = jacobian_metrics(
        kin,
        q,
        sigma_threshold=sigma_threshold,
        condition_threshold=condition_threshold,
    )
    diag.ik_method = "kdl"
    delta_current = joint_delta(q, current_q)
    delta_home = joint_delta(q, home_q)
    diag.joint_delta_norm = float(np.linalg.norm(delta_current))
    diag.max_joint_delta = float(np.max(np.abs(delta_current))) if delta_current.size else 0.0
    wrist_flip = abs(float(delta_current[5])) if delta_current.size >= 6 else 0.0
    margin = joint_limit_margin(q, joint_limits)
    diag.extra = {
        "joint_limit_margin": margin,
        "wrist_flip_delta": wrist_flip,
        "joint_distance_from_home": float(np.linalg.norm(delta_home)),
    }
    if not within_joint_limits(q, joint_limits):
        diag.reject_reason = "joint_limits"
    elif diag.singularity_warning:
        diag.reject_reason = "singularity_risk"
    elif diag.max_joint_delta > float(max_joint_delta):
        diag.reject_reason = "joint_jump"
    elif wrist_flip > float(max_wrist_flip):
        diag.reject_reason = "wrist_flip"
    else:
        diag.accepted = True
    singularity_penalty = 0.0
    if diag.sigma_min is None:
        singularity_penalty = 1.0
    elif diag.sigma_min < sigma_threshold:
        singularity_penalty = (sigma_threshold - diag.sigma_min) / max(1e-6, sigma_threshold)
    condition_penalty = 0.0
    if diag.condition_number is None or not np.isfinite(diag.condition_number):
        condition_penalty = 1.0
    elif diag.condition_number > condition_threshold:
        condition_penalty = min(2.0, diag.condition_number / max(1.0, condition_threshold) - 1.0)
    limit_penalty = 0.0 if margin > 0.20 else (0.20 - margin) * 4.0
    diag.score = float(
        1.00 * np.linalg.norm(delta_current)
        + 0.20 * np.linalg.norm(delta_home)
        + 3.00 * singularity_penalty
        + 0.60 * condition_penalty
        + 1.00 * limit_penalty
        + 0.35 * max(0.0, wrist_flip - 1.2)
    )
    return diag


def quaternion_normalize(q) -> np.ndarray:
    arr = np.asarray(q, dtype=float).flatten()
    if arr.shape[0] != 4:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)
    return arr / norm


def quaternion_multiply(q1, q2) -> np.ndarray:
    x1, y1, z1, w1 = quaternion_normalize(q1)
    x2, y2, z2, w2 = quaternion_normalize(q2)
    return quaternion_normalize(np.asarray([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]))


def quaternion_inverse(q) -> np.ndarray:
    x, y, z, w = quaternion_normalize(q)
    return np.asarray([-x, -y, -z, w], dtype=float)


def quaternion_error_vector(target_q, current_q) -> np.ndarray:
    q_err = quaternion_multiply(target_q, quaternion_inverse(current_q))
    if q_err[3] < 0.0:
        q_err *= -1.0
    return 2.0 * q_err[:3]


def pose_to_pos_quat(pose) -> tuple[np.ndarray, np.ndarray]:
    p = pose.position
    q = pose.orientation
    return (
        np.asarray([float(p.x), float(p.y), float(p.z)], dtype=float),
        quaternion_normalize([float(q.x), float(q.y), float(q.z), float(q.w)]),
    )


def fk_pos_quat(kin, joints) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        from hrl_geom.pose_converter import PoseConv

        mat = kin.forward(np.asarray(joints, dtype=float).flatten())
        pos, quat = PoseConv.to_pos_quat(mat)
        return np.asarray(pos, dtype=float), quaternion_normalize(quat)
    except Exception:
        return None


def pose_error_vector(kin, joints, target_pose) -> np.ndarray | None:
    fk = fk_pos_quat(kin, joints)
    if fk is None:
        return None
    current_pos, current_q = fk
    target_pos, target_q = pose_to_pos_quat(target_pose)
    pos_error = target_pos - current_pos
    rot_error = quaternion_error_vector(target_q, current_q)
    return np.concatenate([pos_error, rot_error])


def dls_step(jacobian: np.ndarray, error: np.ndarray, lambda_sq: float) -> np.ndarray:
    rows = jacobian.shape[0]
    return jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + float(lambda_sq) * np.eye(rows),
        error,
    )


def solve_dls_ik(
    kin,
    target_pose,
    seed,
    current,
    home,
    *,
    joint_limits=DEFAULT_JOINT_LIMITS,
    max_iterations: int = 80,
    position_tolerance: float = 0.006,
    rotation_tolerance: float = 0.060,
    max_step: float = 0.12,
    sigma_threshold: float = 0.02,
    condition_threshold: float = 500.0,
    lambda_base: float = 0.04,
) -> tuple[np.ndarray | None, IKDiagnostics]:
    q = np.asarray(seed, dtype=float).flatten()
    current_q = np.asarray(current, dtype=float).flatten()
    best_q = q.copy()
    best_error_norm = float("inf")
    last_diag = IKDiagnostics(ik_method="dls", reject_reason="not_started")
    for iteration in range(max(1, int(max_iterations))):
        error = pose_error_vector(kin, q, target_pose)
        if error is None:
            last_diag.reject_reason = "fk_unavailable"
            return None, last_diag
        pos_norm = float(np.linalg.norm(error[:3]))
        rot_norm = float(np.linalg.norm(error[3:]))
        error_norm = pos_norm + 0.25 * rot_norm
        if error_norm < best_error_norm:
            best_error_norm = error_norm
            best_q = q.copy()
        jacobian, diag = jacobian_metrics(
            kin,
            q,
            sigma_threshold=sigma_threshold,
            condition_threshold=condition_threshold,
            lambda_base=lambda_base,
        )
        diag.ik_method = "dls"
        diag.pose_error = float(error_norm)
        diag.extra["iteration"] = iteration
        if pos_norm <= position_tolerance and rot_norm <= rotation_tolerance:
            final_diag = score_candidate(
                kin,
                q,
                current_q,
                home,
                joint_limits=joint_limits,
                sigma_threshold=sigma_threshold,
                condition_threshold=condition_threshold,
            )
            final_diag.ik_method = "dls"
            final_diag.pose_error = float(error_norm)
            final_diag.damping_lambda = diag.damping_lambda
            if final_diag.accepted or final_diag.reject_reason == "singularity_risk":
                # DLS is allowed to operate near a mild singularity if the step
                # is small and the final pose error is good.
                if final_diag.max_joint_delta <= 1.60 and pos_norm <= position_tolerance:
                    final_diag.accepted = True
                    final_diag.reject_reason = None
            return q, final_diag
        if jacobian is None:
            diag.reject_reason = "jacobian_unavailable"
            return None, diag
        lambda_sq = float(diag.damping_lambda) ** 2
        try:
            dq = dls_step(jacobian, error, lambda_sq)
        except Exception as exc:
            diag.reject_reason = f"dls_solve_failed:{exc}"
            return None, diag
        dq = np.asarray(dq, dtype=float).flatten()
        max_abs = float(np.max(np.abs(dq))) if dq.size else 0.0
        if max_abs > max_step:
            dq *= float(max_step) / max_abs
        q = q + dq
        for idx, (lower, upper) in enumerate(joint_limits[: q.shape[0]]):
            q[idx] = min(max(q[idx], float(lower)), float(upper))
        last_diag = diag

    final_diag = score_candidate(
        kin,
        best_q,
        current_q,
        home,
        joint_limits=joint_limits,
        sigma_threshold=sigma_threshold,
        condition_threshold=condition_threshold,
    )
    final_diag.ik_method = "dls"
    final_diag.pose_error = None if not np.isfinite(best_error_norm) else float(best_error_norm)
    if best_error_norm < 0.020 and final_diag.max_joint_delta <= 1.80:
        final_diag.accepted = True
        final_diag.reject_reason = None
        return best_q, final_diag
    final_diag.accepted = False
    final_diag.reject_reason = final_diag.reject_reason or "dls_no_convergence"
    final_diag.extra["last_iteration"] = last_diag.to_dict()
    return None, final_diag
