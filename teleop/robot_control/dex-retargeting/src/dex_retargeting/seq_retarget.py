import time
from typing import Optional

import numpy as np
from pytransform3d import rotations

from .constants import OPERATOR2MANO, HandType
from .optimizer import Optimizer
from .optimizer_utils import LPFilter


class SeqRetargeting:
    """带时间连续性的 retargeting 包装器。

    Optimizer 只关心“给定 ref_value，求一次最优 qpos”。
    SeqRetargeting 额外负责：
    - 把上一帧 qpos 作为本帧优化初值，减少跳变并加速收敛；
    - 施加关节限位；
    - 把优化关节和固定关节合并回完整 robot_qpos；
    - 可选做低通滤波。
    """
    def __init__(
        self,
        optimizer: Optimizer,
        has_joint_limits=True,
        lp_filter: Optional[LPFilter] = None,
    ):
        self.optimizer = optimizer
        robot = self.optimizer.robot

        # 关节限位：默认使用 URDF 中的 lower/upper limit；关闭时用极大范围近似无限制。
        self.has_joint_limits = has_joint_limits
        joint_limits = np.ones_like(robot.joint_limits)
        joint_limits[:, 0] = -1e4  # a large value is equivalent to no limit
        joint_limits[:, 1] = 1e4
        if has_joint_limits:
            joint_limits[:] = robot.joint_limits[:]
            self.optimizer.set_joint_limit(joint_limits[self.optimizer.idx_pin2target])
        self.joint_limits = joint_limits[self.optimizer.idx_pin2target]

        # last_qpos 是下一帧优化的初值。初始值取各关节限位中点。
        self.last_qpos = joint_limits.mean(1)[self.optimizer.idx_pin2target].astype(np.float32)
        self.accumulated_time = 0
        self.num_retargeting = 0

        # retarget 后可选低通滤波，进一步压制手部动作抖动。
        self.filter = lp_filter

        # warm_start 主要用于带 6D free joint 的 position retargeting，本项目手部遥操作通常不用。
        self.is_warm_started = False

    def warm_start(
        self,
        wrist_pos: np.ndarray,
        wrist_quat: np.ndarray,
        hand_type: HandType = HandType.right,
        is_mano_convention: bool = False,
    ):
        """
        Initialize the wrist joint pose using analytical computation instead of retargeting optimization.
        This function is specifically for position retargeting with the flying robot hand, i.e. has 6D free joint
        You are not expected to use this function for vector retargeting, e.g. when you are working on teleoperation

        Args:
            wrist_pos: position of the hand wrist, typically from human hand pose
            wrist_quat: quaternion of the hand wrist, the same convention as the operator frame definition if not is_mano_convention
            hand_type: hand type, used to determine the operator2mano matrix
            is_mano_convention: whether the wrist_quat is in mano convention
        """
        # This function can only be used when the first joints of robot are free joints

        if len(wrist_pos) != 3:
            raise ValueError(f"Wrist pos: {wrist_pos} is not a 3-dim vector.")
        if len(wrist_quat) != 4:
            raise ValueError(f"Wrist quat: {wrist_quat} is not a 4-dim vector.")

        operator2mano = OPERATOR2MANO[hand_type] if is_mano_convention else np.eye(3)
        robot = self.optimizer.robot
        target_wrist_pose = np.eye(4)
        target_wrist_pose[:3, :3] = rotations.matrix_from_quaternion(wrist_quat) @ operator2mano.T
        target_wrist_pose[:3, 3] = wrist_pos

        name_list = [
            "dummy_x_translation_joint",
            "dummy_y_translation_joint",
            "dummy_z_translation_joint",
            "dummy_x_rotation_joint",
            "dummy_y_rotation_joint",
            "dummy_z_rotation_joint",
        ]
        wrist_link_id = robot.get_joint_parent_child_frames(name_list[5])[1]

        # Set the dummy joints angles to zero
        old_qpos = robot.q0
        new_qpos = old_qpos.copy()
        for num, joint_name in enumerate(self.optimizer.target_joint_names):
            if joint_name in name_list:
                new_qpos[num] = 0

        robot.compute_forward_kinematics(new_qpos)
        root2wrist = robot.get_link_pose_inv(wrist_link_id)
        target_root_pose = target_wrist_pose @ root2wrist

        euler = rotations.euler_from_matrix(target_root_pose[:3, :3], 0, 1, 2, extrinsic=False)
        pose_vec = np.concatenate([target_root_pose[:3, 3], euler])

        # Find the dummy joints
        for num, joint_name in enumerate(self.optimizer.target_joint_names):
            if joint_name in name_list:
                index = name_list.index(joint_name)
                self.last_qpos[num] = pose_vec[index]

        self.is_warm_started = True

    def retarget(self, ref_value, fixed_qpos=np.array([])):
        """执行一帧重定向。

        ref_value 的形状由 Optimizer 决定：
        - position: (N, 3)，N 个目标点位置；
        - vector/dexpilot: (N, 3)，N 条人手参考向量。

        返回值是完整机器人手的 qpos，顺序为 robot.dof_joint_names。
        调用方通常还会再用关节名映射，把它重排成硬件消息顺序。
        """
        tic = time.perf_counter()

        # 把上一帧结果作为初值传给 nlopt，且先裁剪到关节限位内。
        qpos = self.optimizer.retarget(
            ref_value=ref_value.astype(np.float32),
            fixed_qpos=fixed_qpos.astype(np.float32),
            last_qpos=np.clip(self.last_qpos, self.joint_limits[:, 0], self.joint_limits[:, 1]),
        )
        self.accumulated_time += time.perf_counter() - tic
        self.num_retargeting += 1
        self.last_qpos = qpos

        # Optimizer 只返回 target_joint_names 对应的关节。
        # 这里把 target 关节和 fixed 关节拼回完整机器人关节向量。
        robot_qpos = np.zeros(self.optimizer.robot.dof)
        robot_qpos[self.optimizer.idx_pin2fixed] = fixed_qpos
        robot_qpos[self.optimizer.idx_pin2target] = qpos

        if self.optimizer.adaptor is not None:
            # 如果 URDF 里有 mimic joint，这里根据主动关节补齐从动关节位置。
            robot_qpos = self.optimizer.adaptor.forward_qpos(robot_qpos)

        if self.filter is not None:
            # 低通滤波发生在完整 robot_qpos 上。
            robot_qpos = self.filter.next(robot_qpos)
        return robot_qpos

    def set_qpos(self, robot_qpos: np.ndarray):
        target_qpos = robot_qpos[self.optimizer.idx_pin2target]
        self.last_qpos = target_qpos

    def get_qpos(self, fixed_qpos: Optional[np.ndarray] = None):
        robot_qpos = np.zeros(self.optimizer.robot.dof)
        robot_qpos[self.optimizer.idx_pin2target] = self.last_qpos
        if fixed_qpos is not None:
            robot_qpos[self.optimizer.idx_pin2fixed] = fixed_qpos
        return robot_qpos

    def verbose(self):
        min_value = self.optimizer.opt.last_optimum_value()
        print(f"Retargeting {self.num_retargeting} times takes: {self.accumulated_time}s")
        print(f"Last distance: {min_value}")

    def reset(self):
        self.last_qpos = self.joint_limits.mean(1).astype(np.float32)
        self.num_retargeting = 0
        self.accumulated_time = 0

    @property
    def joint_names(self):
        return self.optimizer.robot.dof_joint_names
