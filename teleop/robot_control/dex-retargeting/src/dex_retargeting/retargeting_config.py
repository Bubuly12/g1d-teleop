from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from typing import Union

import numpy as np
import yaml
import os

from . import yourdfpy as urdf
from .kinematics_adaptor import MimicJointKinematicAdaptor
from .optimizer_utils import LPFilter
from .robot_wrapper import RobotWrapper
from .seq_retarget import SeqRetargeting
from .yourdfpy import DUMMY_JOINT_NAMES


@dataclass
class RetargetingConfig:
    """一份 YAML 配置在代码里的结构化表示。

    retargeting 的关键不是把人手每个关节角直接映射到机器人关节角，而是定义一组
    “要匹配的任务空间目标”，再通过优化求机器人关节角。

    三种 type 的输入含义：
    - position: 让机器人某些 link 的 3D 位置追踪人手某些 landmark。
    - vector: 让机器人 link_a -> link_b 的向量追踪人手 landmark_i -> landmark_j 的向量。
    - dexpilot: 也是向量追踪，但会额外处理指尖靠近时的抓取稳定性。
    """
    type: str
    urdf_path: str
    target_joint_names: Optional[List[str]] = None

    # Whether to add free joint to the root of the robot. Free joint enable the robot hand move freely in the space
    add_dummy_free_joint: bool = False

    # DexPilot 使用手腕/掌根 link 作为基准，再组合各指尖之间、掌根到指尖的向量。
    # 这个 link 对应人手 wrist/palm 基准点。
    wrist_link_name: Optional[str] = None
    # 机器人手的指尖 link 名称，例如 thumb_tip/index_tip。
    finger_tip_link_names: Optional[List[str]] = None
    # (2, N) 的人手 landmark 索引，表示 N 条参考向量：human[row1] - human[row0]。
    target_link_human_indices_dexpilot: Optional[np.ndarray] = None

    # Position 模式：这些机器人 link 直接追踪人手对应 landmark 的 3D 位置。
    target_link_names: Optional[List[str]] = None
    target_link_human_indices_position: Optional[np.ndarray] = None

    # Vector 模式：每个 origin/task link 对应一条机器人向量 task - origin。
    # 它会和 target_link_human_indices_vector 定义的人手向量一一匹配。
    target_origin_link_names: Optional[List[str]] = None
    target_task_link_names: Optional[List[str]] = None
    target_link_human_indices_vector: Optional[np.ndarray] = None

    # Scaling factor for vector retargeting only
    # For example, Allegro is 1.6 times larger than normal human hand, then this scaling factor should be 1.6
    scaling_factor: float = 1.0

    # Low pass filter
    low_pass_alpha: float = 0.1

    # Optimization parameters
    normal_delta: float = 4e-3
    huber_delta: float = 2e-2

    # DexPilot optimizer parameters
    project_dist: float = 0.03
    escape_dist: float = 0.05

    # Joint limit tag
    has_joint_limits: bool = True

    # Mimic joint tag
    ignore_mimic_joint: bool = False

    _TYPE = ["vector", "position", "dexpilot"]
    _DEFAULT_URDF_DIR = "./"

    def __post_init__(self):
        # Retargeting type check
        self.type = self.type.lower()
        if self.type not in self._TYPE:
            raise ValueError(f"Retargeting type must be one of {self._TYPE}")

        # Vector retargeting requires: target_origin_link_names + target_task_link_names
        # Position retargeting requires: target_link_names
        if self.type == "vector":
            if self.target_origin_link_names is None or self.target_task_link_names is None:
                raise ValueError(f"Vector retargeting requires: target_origin_link_names + target_task_link_names")
            if len(self.target_task_link_names) != len(self.target_origin_link_names):
                raise ValueError(f"Vector retargeting origin and task links dim mismatch")
            if self.target_link_human_indices_vector.shape != (2, len(self.target_origin_link_names)):
                raise ValueError(f"Vector retargeting link names and link indices dim mismatch")
            if self.target_link_human_indices_vector is None:
                raise ValueError(f"Vector retargeting requires: target_link_human_indices_vector")

        elif self.type == "position":
            if self.target_link_names is None:
                raise ValueError(f"Position retargeting requires: target_link_names")
            self.target_link_human_indices_position = self.target_link_human_indices_position.squeeze()
            if self.target_link_human_indices_position.shape != (len(self.target_link_names),):
                raise ValueError(f"Position retargeting link names and link indices dim mismatch")
            if self.target_link_human_indices_position is None:
                raise ValueError(f"Position retargeting requires: target_link_human_indices_position")

        elif self.type == "dexpilot":
            if self.finger_tip_link_names is None or self.wrist_link_name is None:
                raise ValueError(f"Position retargeting requires: finger_tip_link_names + wrist_link_name")
            if self.target_link_human_indices_dexpilot is not None:
                print(
                    "\033[33m",
                    "Target link human indices is provided in the DexPilot retargeting config, which is uncommon.\n"
                    "If you do not know exactly how it is used, please leave it to None for default.\n"
                    "\033[00m",
                )

        # URDF path check
        urdf_path = Path(self.urdf_path)
        if not urdf_path.is_absolute():
            urdf_path = self._DEFAULT_URDF_DIR / urdf_path
            urdf_path = urdf_path.absolute()
        if not urdf_path.exists():
            raise ValueError(f"URDF path {urdf_path} does not exist")
        self.urdf_path = str(urdf_path)

    @classmethod
    def set_default_urdf_dir(cls, urdf_dir: Union[str, Path]):
        path = Path(urdf_dir)
        if not path.exists():
            raise ValueError(f"URDF dir {urdf_dir} not exists.")
        cls._DEFAULT_URDF_DIR = urdf_dir

    @classmethod
    def load_from_file(cls, config_path: Union[str, Path], override: Optional[Dict] = None):
        path = Path(config_path)
        if not path.is_absolute():
            path = path.absolute()

        with path.open("r") as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            cfg = yaml_config["retargeting"]
            return cls.from_dict(cfg, override)

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any], override: Optional[Dict] = None):
        if "target_link_human_indices_position" in cfg:
            cfg["target_link_human_indices_position"] = np.array(cfg["target_link_human_indices_position"])
        if "target_link_human_indices_vector" in cfg:
            cfg["target_link_human_indices_vector"] = np.array(cfg["target_link_human_indices_vector"])
        if "target_link_human_indices_dexpilot" in cfg:
            cfg["target_link_human_indices_dexpilot"] = np.array(cfg["target_link_human_indices_dexpilot"])

        if override is not None:
            for key, value in override.items():
                cfg[key] = value
        config = RetargetingConfig(**cfg)
        return config

    def build(self) -> SeqRetargeting:
        """根据配置创建 SeqRetargeting。

        构建顺序：
        1. 读 URDF，创建 Pinocchio RobotWrapper。
        2. 根据 type 选择 Position/Vector/DexPilot Optimizer。
        3. 设置低通滤波、关节限位和 mimic joint 适配。

        build() 返回的 SeqRetargeting 是控制循环每一帧调用 retarget(ref_value) 的对象。
        """
        from .optimizer import (
            VectorOptimizer,
            PositionOptimizer,
            DexPilotOptimizer,
        )
        import tempfile

        # 先用 yourdfpy 处理 URDF，主要是修正 mesh/路径，并可选插入 dummy free joints。
        robot_urdf = urdf.URDF.load(
            self.urdf_path, add_dummy_free_joints=self.add_dummy_free_joint, build_scene_graph=False
        )
        urdf_name = self.urdf_path.split(os.path.sep)[-1]
        temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
        temp_path = f"{temp_dir}/{urdf_name}"
        robot_urdf.write_xml_file(temp_path)

        # Pinocchio 负责正运动学和雅可比，优化器每次评估目标函数都会调用它。
        robot = RobotWrapper(temp_path)

        # 如果给手掌根部加了 6D 自由关节，也把这些虚拟关节加入优化变量。
        if self.add_dummy_free_joint and self.target_joint_names is not None:
            self.target_joint_names = DUMMY_JOINT_NAMES + self.target_joint_names
        joint_names = self.target_joint_names if self.target_joint_names is not None else robot.dof_joint_names

        if self.type == "position":
            # PositionOptimizer: 目标是最小化机器人 link 位置和人手 landmark 位置之间的误差。
            optimizer = PositionOptimizer(
                robot,
                joint_names,
                target_link_names=self.target_link_names,
                target_link_human_indices=self.target_link_human_indices_position,
                norm_delta=self.normal_delta,
                huber_delta=self.huber_delta,
            )
        elif self.type == "vector":
            # VectorOptimizer: 目标是最小化机器人 link 向量和人手 landmark 向量之间的误差。
            optimizer = VectorOptimizer(
                robot,
                joint_names,
                target_origin_link_names=self.target_origin_link_names,
                target_task_link_names=self.target_task_link_names,
                target_link_human_indices=self.target_link_human_indices_vector,
                scaling=self.scaling_factor,
                norm_delta=self.normal_delta,
                huber_delta=self.huber_delta,
            )
        elif self.type == "dexpilot":
            # DexPilotOptimizer: 在 Vector 的基础上，对指尖靠近时的抓取向量做投影/加权。
            optimizer = DexPilotOptimizer(
                robot,
                joint_names,
                finger_tip_link_names=self.finger_tip_link_names,
                wrist_link_name=self.wrist_link_name,
                target_link_human_indices=self.target_link_human_indices_dexpilot,
                scaling=self.scaling_factor,
                project_dist=self.project_dist,
                escape_dist=self.escape_dist,
            )
        else:
            raise RuntimeError()

        if 0 <= self.low_pass_alpha <= 1:
            lp_filter = LPFilter(self.low_pass_alpha)
        else:
            lp_filter = None

        # mimic joint 是 URDF 里的“从动关节”。适配器会在优化主关节后补齐 mimic 关节，
        # 并把雅可比梯度从完整关节空间折回优化变量空间。
        has_mimic_joints, source_names, mimic_names, multipliers, offsets = parse_mimic_joint(robot_urdf)
        if has_mimic_joints and not self.ignore_mimic_joint:
            adaptor = MimicJointKinematicAdaptor(
                robot,
                target_joint_names=joint_names,
                source_joint_names=source_names,
                mimic_joint_names=mimic_names,
                multipliers=multipliers,
                offsets=offsets,
            )
            optimizer.set_kinematic_adaptor(adaptor)
            print(
                "\033[34m",
                "Mimic joint adaptor enabled. The mimic joint tags in the URDF will be considered during retargeting.\n"
                "To disable mimic joint adaptor, consider setting ignore_mimic_joint=True in the configuration.",
                "\033[39m",
            )

        retargeting = SeqRetargeting(
            optimizer,
            has_joint_limits=self.has_joint_limits,
            lp_filter=lp_filter,
        )
        return retargeting


def get_retargeting_config(config_path: Union[str, Path]) -> RetargetingConfig:
    config = RetargetingConfig.load_from_file(config_path)
    return config


def parse_mimic_joint(robot_urdf: urdf.URDF) -> Tuple[bool, List[str], List[str], List[float], List[float]]:
    mimic_joint_names = []
    source_joint_names = []
    multipliers = []
    offsets = []
    for name, joint in robot_urdf.joint_map.items():
        if joint.mimic is not None:
            mimic_joint_names.append(name)
            source_joint_names.append(joint.mimic.joint)
            multipliers.append(joint.mimic.multiplier)
            offsets.append(joint.mimic.offset)

    return len(mimic_joint_names) > 0, source_joint_names, mimic_joint_names, multipliers, offsets
