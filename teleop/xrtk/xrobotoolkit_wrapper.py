"""XRoboToolkit SDK adapter that returns the same TeleData shape as TeleVuerWrapper.

Controller/headset data is supported. Hand tracking is mapped from the SDK's
26-joint OpenXR-style layout to the old 25-joint WebXR-style layout by dropping
joint 0, which is typically the palm joint.
"""

import numpy as np

from teleop.televuer.tv_wrapper import (
    CONST_HEAD_POSE,
    CONST_LEFT_ARM_POSE,
    CONST_RIGHT_ARM_POSE,
    T_OPENXR_ROBOT,
    T_ROBOT_OPENXR,
    T_TO_UNITREE_HAND,
    T_TO_UNITREE_HUMANOID_LEFT_ARM,
    T_TO_UNITREE_HUMANOID_RIGHT_ARM,
    TeleData,
    fast_mat_inv,
    safe_mat_update,
)


def pose7_to_mat4(pose7):
    """Convert SDK pose [x, y, z, qx, qy, qz, qw] to a 4x4 matrix."""
    pose = np.asarray(pose7, dtype=float)
    if pose.shape != (7,):
        raise ValueError(f"Expected pose shape (7,), got {pose.shape}")

    x, y, z, qx, qy, qz, qw = pose
    norm = np.linalg.norm([qx, qy, qz, qw])
    if not np.isfinite(norm) or norm < 1e-8:
        rot = np.eye(3)
    else:
        qx, qy, qz, qw = np.array([qx, qy, qz, qw], dtype=float) / norm
        rot = np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ],
            dtype=float,
        )

    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = [x, y, z]
    return mat


def axis_to_vuer(axis):
    """Map XRoboToolkit thumbstick axis to the Vuer/WebXR TeleData convention."""
    mapped = np.asarray(axis, dtype=float).copy()
    if mapped.shape != (2,):
        return np.zeros(2)
    # XRTK's Y axis is opposite to the Vuer path on the tested PICO controller.
    mapped[1] *= -1.0
    return mapped


class XRoboToolkitWrapper:
    """Read XR data from xrobotoolkit_sdk and expose TeleVuerWrapper-like API."""

    def __init__(self, use_hand_tracking=False):
        import xrobotoolkit_sdk as xrt

        self.xrt = xrt
        self.use_hand_tracking = use_hand_tracking
        self.xrt.init()

    def _read_pose_mat(self, getter, fallback):
        try:
            pose = pose7_to_mat4(getter())
        except Exception:
            return fallback, False
        return safe_mat_update(fallback, pose)

    def get_tele_data(self):
        """Return XR data in the same TeleData format as tv_wrapper.py."""
        # SDK pose is assumed to use the same OpenXR basis as the old Vuer path.
        # If the robot moves in a mirrored or rotated direction, this assumption is the first thing to verify.
        Bxr_world_head, _ = self._read_pose_mat(self.xrt.get_headset_pose, CONST_HEAD_POSE)
        Brobot_world_head = T_ROBOT_OPENXR @ Bxr_world_head @ T_OPENXR_ROBOT

        if self.use_hand_tracking:
            return self._get_hand_tracking_tele_data(Brobot_world_head)
        return self._get_controller_tele_data(Brobot_world_head)

    def _get_controller_tele_data(self, Brobot_world_head):
        """Build TeleData for controller tracking mode."""
        left_IPunitree_Bxr_world_arm, _ = self._read_pose_mat(self.xrt.get_left_controller_pose, CONST_LEFT_ARM_POSE)
        right_IPunitree_Bxr_world_arm, _ = self._read_pose_mat(self.xrt.get_right_controller_pose, CONST_RIGHT_ARM_POSE)

        # Same controller branch transform as TeleVuerWrapper.get_tele_data().
        left_IPunitree_Brobot_world_arm = T_ROBOT_OPENXR @ left_IPunitree_Bxr_world_arm @ T_OPENXR_ROBOT
        right_IPunitree_Brobot_world_arm = T_ROBOT_OPENXR @ right_IPunitree_Bxr_world_arm @ T_OPENXR_ROBOT

        left_IPunitree_Brobot_head_arm = left_IPunitree_Brobot_world_arm.copy()
        right_IPunitree_Brobot_head_arm = right_IPunitree_Brobot_world_arm.copy()
        left_IPunitree_Brobot_head_arm[0:3, 3] -= Brobot_world_head[0:3, 3]
        right_IPunitree_Brobot_head_arm[0:3, 3] -= Brobot_world_head[0:3, 3]

        # Move origin from head to the approximate waist/IK origin, matching tv_wrapper.py.
        left_IPunitree_Brobot_wrist_arm = left_IPunitree_Brobot_head_arm.copy()
        right_IPunitree_Brobot_wrist_arm = right_IPunitree_Brobot_head_arm.copy()
        left_IPunitree_Brobot_wrist_arm[0, 3] += 0.15
        right_IPunitree_Brobot_wrist_arm[0, 3] += 0.15
        left_IPunitree_Brobot_wrist_arm[2, 3] += 0.45
        right_IPunitree_Brobot_wrist_arm[2, 3] += 0.45

        left_trigger = float(self.xrt.get_left_trigger())
        right_trigger = float(self.xrt.get_right_trigger())
        left_grip = float(self.xrt.get_left_grip())
        right_grip = float(self.xrt.get_right_grip())

        return TeleData(
            head_pose=Brobot_world_head,
            left_wrist_pose=left_IPunitree_Brobot_wrist_arm,
            right_wrist_pose=right_IPunitree_Brobot_wrist_arm,
            left_ctrl_trigger=left_trigger > 0.0,
            left_ctrl_triggerValue=10.0 - left_trigger * 10.0,
            left_ctrl_squeeze=left_grip > 0.0,
            left_ctrl_squeezeValue=left_grip,
            left_ctrl_aButton=bool(self.xrt.get_X_button()),
            left_ctrl_bButton=bool(self.xrt.get_Y_button()),
            left_ctrl_thumbstick=bool(self.xrt.get_left_axis_click()),
            left_ctrl_thumbstickValue=axis_to_vuer(self.xrt.get_left_axis()),
            right_ctrl_trigger=right_trigger > 0.0,
            right_ctrl_triggerValue=10.0 - right_trigger * 10.0,
            right_ctrl_squeeze=right_grip > 0.0,
            right_ctrl_squeezeValue=right_grip,
            right_ctrl_aButton=bool(self.xrt.get_A_button()),
            right_ctrl_bButton=bool(self.xrt.get_B_button()),
            right_ctrl_thumbstick=bool(self.xrt.get_right_axis_click()),
            right_ctrl_thumbstickValue=axis_to_vuer(self.xrt.get_right_axis()),
        )

    def _get_hand_tracking_tele_data(self, Brobot_world_head):
        """Build TeleData for hand tracking mode.

        XRoboToolkit hand state is 26 x [x, y, z, qx, qy, qz, qw]. OpenXR hand
        joints usually include palm at index 0 and wrist at index 1. The old
        WebXR/Vuer path uses 25 joints without palm, so we drop index 0.
        """
        left_hand26 = np.asarray(self.xrt.get_left_hand_tracking_state(), dtype=float)
        right_hand26 = np.asarray(self.xrt.get_right_hand_tracking_state(), dtype=float)
        left_active = bool(self.xrt.get_left_hand_is_active())
        right_active = bool(self.xrt.get_right_hand_is_active())

        if left_active and left_hand26.shape == (26, 7):
            left_IPxr_Bxr_world_arm = pose7_to_mat4(left_hand26[1])
            left_hand25 = left_hand26[1:26]
        else:
            left_IPxr_Bxr_world_arm = CONST_LEFT_ARM_POSE
            left_hand25 = np.zeros((25, 7))

        if right_active and right_hand26.shape == (26, 7):
            right_IPxr_Bxr_world_arm = pose7_to_mat4(right_hand26[1])
            right_hand25 = right_hand26[1:26]
        else:
            right_IPxr_Bxr_world_arm = CONST_RIGHT_ARM_POSE
            right_hand25 = np.zeros((25, 7))

        # Same hand-tracking arm-pose transform as TeleVuerWrapper.get_tele_data().
        left_IPxr_Brobot_world_arm = T_ROBOT_OPENXR @ left_IPxr_Bxr_world_arm @ T_OPENXR_ROBOT
        right_IPxr_Brobot_world_arm = T_ROBOT_OPENXR @ right_IPxr_Bxr_world_arm @ T_OPENXR_ROBOT
        left_IPunitree_Brobot_world_arm = left_IPxr_Brobot_world_arm @ (
            T_TO_UNITREE_HUMANOID_LEFT_ARM if left_active else np.eye(4)
        )
        right_IPunitree_Brobot_world_arm = right_IPxr_Brobot_world_arm @ (
            T_TO_UNITREE_HUMANOID_RIGHT_ARM if right_active else np.eye(4)
        )

        left_IPunitree_Brobot_head_arm = left_IPunitree_Brobot_world_arm.copy()
        right_IPunitree_Brobot_head_arm = right_IPunitree_Brobot_world_arm.copy()
        left_IPunitree_Brobot_head_arm[0:3, 3] -= Brobot_world_head[0:3, 3]
        right_IPunitree_Brobot_head_arm[0:3, 3] -= Brobot_world_head[0:3, 3]

        left_IPunitree_Brobot_wrist_arm = left_IPunitree_Brobot_head_arm.copy()
        right_IPunitree_Brobot_wrist_arm = right_IPunitree_Brobot_head_arm.copy()
        left_IPunitree_Brobot_wrist_arm[0, 3] += 0.15
        right_IPunitree_Brobot_wrist_arm[0, 3] += 0.15
        left_IPunitree_Brobot_wrist_arm[2, 3] += 0.45
        right_IPunitree_Brobot_wrist_arm[2, 3] += 0.45

        if left_active and right_active:
            left_IPxr_Bxr_world_hand_pos = np.concatenate(
                [left_hand25[:, :3].T, np.ones((1, left_hand25.shape[0]))],
                axis=0,
            )
            right_IPxr_Bxr_world_hand_pos = np.concatenate(
                [right_hand25[:, :3].T, np.ones((1, right_hand25.shape[0]))],
                axis=0,
            )

            left_IPxr_Brobot_world_hand_pos = T_ROBOT_OPENXR @ left_IPxr_Bxr_world_hand_pos
            right_IPxr_Brobot_world_hand_pos = T_ROBOT_OPENXR @ right_IPxr_Bxr_world_hand_pos

            left_IPxr_Brobot_arm_hand_pos = fast_mat_inv(left_IPxr_Brobot_world_arm) @ left_IPxr_Brobot_world_hand_pos
            right_IPxr_Brobot_arm_hand_pos = fast_mat_inv(right_IPxr_Brobot_world_arm) @ right_IPxr_Brobot_world_hand_pos

            left_IPunitree_Brobot_arm_hand_pos = (T_TO_UNITREE_HAND @ left_IPxr_Brobot_arm_hand_pos)[0:3, :].T
            right_IPunitree_Brobot_arm_hand_pos = (T_TO_UNITREE_HAND @ right_IPxr_Brobot_arm_hand_pos)[0:3, :].T
        else:
            left_IPunitree_Brobot_arm_hand_pos = np.zeros((25, 3))
            right_IPunitree_Brobot_arm_hand_pos = np.zeros((25, 3))

        left_pinch_value = self._estimate_pinch_value(left_IPunitree_Brobot_arm_hand_pos, left_active)
        right_pinch_value = self._estimate_pinch_value(right_IPunitree_Brobot_arm_hand_pos, right_active)

        return TeleData(
            head_pose=Brobot_world_head,
            left_wrist_pose=left_IPunitree_Brobot_wrist_arm,
            right_wrist_pose=right_IPunitree_Brobot_wrist_arm,
            left_hand_pos=left_IPunitree_Brobot_arm_hand_pos,
            right_hand_pos=right_IPunitree_Brobot_arm_hand_pos,
            # Match tv_wrapper.py's scale: pinchValue is distance in centimeters.
            left_hand_pinch=left_pinch_value < 3.0,
            left_hand_pinchValue=left_pinch_value,
            left_hand_squeeze=False,
            left_hand_squeezeValue=0.0,
            right_hand_pinch=right_pinch_value < 3.0,
            right_hand_pinchValue=right_pinch_value,
            right_hand_squeeze=False,
            right_hand_squeezeValue=0.0,
            # Keep controller buttons available even in hand-tracking mode.
            left_ctrl_aButton=bool(self.xrt.get_X_button()),
            left_ctrl_bButton=bool(self.xrt.get_Y_button()),
            left_ctrl_thumbstick=bool(self.xrt.get_left_axis_click()),
            left_ctrl_thumbstickValue=np.asarray(self.xrt.get_left_axis(), dtype=float),
            right_ctrl_aButton=bool(self.xrt.get_A_button()),
            right_ctrl_bButton=bool(self.xrt.get_B_button()),
            right_ctrl_thumbstick=bool(self.xrt.get_right_axis_click()),
            right_ctrl_thumbstickValue=np.asarray(self.xrt.get_right_axis(), dtype=float),
        )

    @staticmethod
    def _estimate_pinch_value(hand_pos, active):
        """Estimate thumb-index pinch distance in centimeters.

        In the 25-joint WebXR-compatible layout, thumb tip is index 4 and
        index-finger tip is index 9. tv_wrapper.py returns pinchValue scaled by
        100, so this helper also returns centimeters.
        """
        hand = np.asarray(hand_pos, dtype=float)
        if not active or hand.shape != (25, 3):
            return 10.0
        return float(np.linalg.norm(hand[4] - hand[9]) * 100.0)

    def render_to_xr(self, img):
        """Compatibility no-op: XRoboToolkit data path does not render camera frames here."""
        return None

    def set_display_image(self, img):
        """Compatibility no-op for older test code that used set_display_image()."""
        return self.render_to_xr(img)

    def close(self):
        self.xrt.close()
