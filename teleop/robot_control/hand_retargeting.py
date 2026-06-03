from dex_retargeting import RetargetingConfig
from pathlib import Path
import yaml
from enum import Enum
import logging_mp
logger_mp = logging_mp.getLogger(__name__)

class HandType(Enum):
    """不同末端执行器对应的重定向配置文件路径。"""
    INSPIRE_HAND = "../assets/inspire_hand/inspire_hand.yml"
    INSPIRE_HAND_Unit_Test = "../../assets/inspire_hand/inspire_hand.yml"
    UNITREE_DEX3 = "../assets/unitree_hand/unitree_dex3.yml"
    UNITREE_DEX3_Unit_Test = "../../assets/unitree_hand/unitree_dex3.yml"
    BRAINCO_HAND = "../assets/brainco_hand/brainco.yml"
    BRAINCO_HAND_Unit_Test = "../../assets/brainco_hand/brainco.yml"

class HandRetargeting:
    def __init__(self, hand_type: HandType):
        """加载左右手 retargeting 配置，并建立“优化输出顺序 -> 硬件关节顺序”的映射。

        这层本身不直接做优化，它做三件事：
        1. 读取 YAML，告诉 dex_retargeting 应该使用哪种算法、哪份 URDF、哪些机器人关节可动。
        2. 暴露 left_indices/right_indices，控制循环会用它们从 XR 25 个手部点里取点对。
        3. 建立 retargeting 输出关节顺序到真实硬件消息顺序的重排表。

        每帧真正的重定向流程在 robot_hand_*.py 里触发：
            human_points = XR 手部 25 点坐标
            ref_value = human_points[indices[1]] - human_points[indices[0]]
            q = self.left_retargeting.retarget(ref_value)

        其中 retarget() 会进入 dex-retargeting 的 SeqRetargeting/Optimizer：
        用 Pinocchio 正运动学计算机器人手 link 位置，再用 nlopt 优化机器人关节角，
        让机器人手的指尖/手掌相对向量尽量接近 ref_value。
        """
        # dex_retargeting 需要知道 URDF 根目录；单元测试从更深一级目录运行，所以路径不同。
        if hand_type == HandType.UNITREE_DEX3:
            RetargetingConfig.set_default_urdf_dir('../assets')
        elif hand_type == HandType.UNITREE_DEX3_Unit_Test:
            RetargetingConfig.set_default_urdf_dir('../../assets')
        elif hand_type == HandType.INSPIRE_HAND:
            RetargetingConfig.set_default_urdf_dir('../assets')
        elif hand_type == HandType.INSPIRE_HAND_Unit_Test:
            RetargetingConfig.set_default_urdf_dir('../../assets')
        elif hand_type == HandType.BRAINCO_HAND:
            RetargetingConfig.set_default_urdf_dir('../assets')
        elif hand_type == HandType.BRAINCO_HAND_Unit_Test:
            RetargetingConfig.set_default_urdf_dir('../../assets')

        config_file_path = Path(hand_type.value)

        try:
            # YAML 里分别保存 left/right 的 retargeting 参数。
            # 以 DexPilot 为例，target_link_human_indices_dexpilot 是一个 (2, N) 数组：
            #   第 0 行是每条人手参考向量的起点 landmark id
            #   第 1 行是每条人手参考向量的终点 landmark id
            # 控制循环里会计算 human[终点] - human[起点]，得到 N 条 3D 向量。
            with config_file_path.open('r') as f:
                self.cfg = yaml.safe_load(f)
                
            if 'left' not in self.cfg or 'right' not in self.cfg:
                raise ValueError("Configuration file must contain 'left' and 'right' keys.")

            left_retargeting_config = RetargetingConfig.from_dict(self.cfg['left'])
            right_retargeting_config = RetargetingConfig.from_dict(self.cfg['right'])
            # build() 后得到真正用于每帧计算的 retargeting 对象。
            self.left_retargeting = left_retargeting_config.build()
            self.right_retargeting = right_retargeting_config.build()

            # target_link_human_indices 告诉控制代码从 25 个 XR 手部点里取哪些点对构造参考向量。
            # 注意：这里的 indices 会随 YAML 的 type 不同而不同：
            #   position: 直接追踪某些人手点的 3D 位置；
            #   vector: 追踪若干人手点对向量；
            #   DexPilot: 追踪指尖之间、手腕到指尖之间的向量，并对近距离抓取做特殊投影。
            self.left_retargeting_joint_names = self.left_retargeting.joint_names
            self.right_retargeting_joint_names = self.right_retargeting.joint_names
            self.left_indices = self.left_retargeting.optimizer.target_link_human_indices
            self.right_indices = self.right_retargeting.optimizer.target_link_human_indices

            if hand_type == HandType.UNITREE_DEX3 or hand_type == HandType.UNITREE_DEX3_Unit_Test:
                # Unitree Dex3 硬件消息里的关节顺序和 retargeting 输出顺序不一定一致，需要重排。
                self.left_dex3_api_joint_names  = [ 'left_hand_thumb_0_joint', 'left_hand_thumb_1_joint', 'left_hand_thumb_2_joint',
                                                    'left_hand_middle_0_joint', 'left_hand_middle_1_joint', 
                                                    'left_hand_index_0_joint', 'left_hand_index_1_joint' ]
                self.right_dex3_api_joint_names = [ 'right_hand_thumb_0_joint', 'right_hand_thumb_1_joint', 'right_hand_thumb_2_joint',
                                                    'right_hand_middle_0_joint', 'right_hand_middle_1_joint',
                                                    'right_hand_index_0_joint', 'right_hand_index_1_joint' ]
                self.left_dex_retargeting_to_hardware = [ self.left_retargeting_joint_names.index(name) for name in self.left_dex3_api_joint_names]
                self.right_dex_retargeting_to_hardware = [ self.right_retargeting_joint_names.index(name) for name in self.right_dex3_api_joint_names]

            elif hand_type == HandType.INSPIRE_HAND or hand_type == HandType.INSPIRE_HAND_Unit_Test:
                # Inspire 手也按官方电机顺序重排，保证写入 MotorCmds_ 时索引正确。
                self.left_inspire_api_joint_names  = [ 'L_pinky_proximal_joint', 'L_ring_proximal_joint', 'L_middle_proximal_joint',
                                                       'L_index_proximal_joint', 'L_thumb_proximal_pitch_joint', 'L_thumb_proximal_yaw_joint' ]
                self.right_inspire_api_joint_names = [ 'R_pinky_proximal_joint', 'R_ring_proximal_joint', 'R_middle_proximal_joint',
                                                       'R_index_proximal_joint', 'R_thumb_proximal_pitch_joint', 'R_thumb_proximal_yaw_joint' ]
                self.left_dex_retargeting_to_hardware = [ self.left_retargeting_joint_names.index(name) for name in self.left_inspire_api_joint_names]
                self.right_dex_retargeting_to_hardware = [ self.right_retargeting_joint_names.index(name) for name in self.right_inspire_api_joint_names]
            
            elif hand_type == HandType.BRAINCO_HAND or hand_type == HandType.BRAINCO_HAND_Unit_Test:
                # BrainCo 的驱动 ID 顺序同样需要从 retargeting 关节名映射到硬件顺序。
                self.left_brainco_api_joint_names  = [ 'left_thumb_metacarpal_joint', 'left_thumb_proximal_joint', 'left_index_proximal_joint',
                                                       'left_middle_proximal_joint', 'left_ring_proximal_joint', 'left_pinky_proximal_joint' ]
                self.right_brainco_api_joint_names = [ 'right_thumb_metacarpal_joint', 'right_thumb_proximal_joint', 'right_index_proximal_joint',
                                                       'right_middle_proximal_joint', 'right_ring_proximal_joint', 'right_pinky_proximal_joint' ]
                self.left_dex_retargeting_to_hardware = [ self.left_retargeting_joint_names.index(name) for name in self.left_brainco_api_joint_names]
                self.right_dex_retargeting_to_hardware = [ self.right_retargeting_joint_names.index(name) for name in self.right_brainco_api_joint_names]
        
        except FileNotFoundError:
            logger_mp.warning(f"Configuration file not found: {config_file_path}")
            raise
        except yaml.YAMLError as e:
            logger_mp.warning(f"YAML error while reading {config_file_path}: {e}")
            raise
        except Exception as e:
            logger_mp.error(f"An error occurred: {e}")
            raise
