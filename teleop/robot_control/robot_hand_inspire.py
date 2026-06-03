from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_                           # idl
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
import numpy as np
from enum import IntEnum
import threading
import time
from multiprocessing import Process, Array

import logging_mp
logger_mp = logging_mp.getLogger(__name__)

Inspire_Num_Motors = 6
kTopicInspireDFXCommand = "rt/inspire/cmd"
kTopicInspireDFXState = "rt/inspire/state"

class Inspire_Controller_DFX:
    def __init__(self, left_hand_array, right_hand_array, dual_hand_data_lock = None, dual_hand_state_array = None,
                       dual_hand_action_array = None, fps = 100.0, Unit_Test = False, simulation_mode = False):
        """Inspire DFX 灵巧手控制器：XR 手部骨架 -> retargeting -> DFX 电机命令。"""
        logger_mp.info("Initialize Inspire_Controller_DFX...")
        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode
        if not self.Unit_Test:
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND)
        else:
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND_Unit_Test)

        if self.simulation_mode:
            ChannelFactoryInitialize(1)
        else:
            ChannelFactoryInitialize(0)

        # DFX 左右手共用一个 MotorCmds_ topic，状态也从同一个 topic 读。
        self.HandCmb_publisher = ChannelPublisher(kTopicInspireDFXCommand, MotorCmds_)
        self.HandCmb_publisher.Init()

        self.HandState_subscriber = ChannelSubscriber(kTopicInspireDFXState, MotorStates_)
        self.HandState_subscriber.Init()

        # 缓存左右手 6 个电机的归一化/实际状态，用于控制进程和数据记录。
        self.left_hand_state_array  = Array('d', Inspire_Num_Motors, lock=True)  
        self.right_hand_state_array = Array('d', Inspire_Num_Motors, lock=True)

        # 订阅线程持续刷新当前手部状态。
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while True:
            if any(self.right_hand_state_array): # any(self.left_hand_state_array) and 
                break
            time.sleep(0.01)
            logger_mp.warning("[Inspire_Controller_DFX] Waiting to subscribe dds...")
        logger_mp.info("[Inspire_Controller_DFX] Subscribe dds ok.")

        hand_control_process = Process(target=self.control_process, args=(left_hand_array, right_hand_array,  self.left_hand_state_array, self.right_hand_state_array,
                                                                          dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array))
        hand_control_process.daemon = True
        hand_control_process.start()

        logger_mp.info("Initialize Inspire_Controller_DFX OK!")

    def _subscribe_hand_state(self):
        """订阅 Inspire DFX 状态，并按官方电机顺序拆成左右手。"""
        while True:
            hand_msg  = self.HandState_subscriber.Read()
            if hand_msg is not None:
                for idx, id in enumerate(Inspire_Left_Hand_JointIndex):
                    self.left_hand_state_array[idx] = hand_msg.states[id].q
                for idx, id in enumerate(Inspire_Right_Hand_JointIndex):
                    self.right_hand_state_array[idx] = hand_msg.states[id].q
            time.sleep(0.002)

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """
        将左右手目标值写入同一个 MotorCmds_ 消息并发布。
        """
        for idx, id in enumerate(Inspire_Left_Hand_JointIndex):             
            self.hand_msg.cmds[id].q = left_q_target[idx]         
        for idx, id in enumerate(Inspire_Right_Hand_JointIndex):             
            self.hand_msg.cmds[id].q = right_q_target[idx] 

        self.HandCmb_publisher.Write(self.hand_msg)
        # logger_mp.debug("hand ctrl publish ok.")
    
    def control_process(self, left_hand_array, right_hand_array, left_hand_state_array, right_hand_state_array,
                              dual_hand_data_lock = None, dual_hand_state_array = None, dual_hand_action_array = None):
        """固定频率读取 XR 手部骨架，计算 Inspire DFX 的 0~1 开合命令。"""
        self.running = True

        # Inspire 官方定义：1.0 接近张开，0.0 接近闭合，因此初值设为张开。
        left_q_target  = np.full(Inspire_Num_Motors, 1.0)
        right_q_target = np.full(Inspire_Num_Motors, 1.0)

        # 初始化左右手共 12 个电机命令。
        self.hand_msg  = MotorCmds_()
        self.hand_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(len(Inspire_Right_Hand_JointIndex) + len(Inspire_Left_Hand_JointIndex))]

        for idx, id in enumerate(Inspire_Left_Hand_JointIndex):
            self.hand_msg.cmds[id].q = 1.0
        for idx, id in enumerate(Inspire_Right_Hand_JointIndex):
            self.hand_msg.cmds[id].q = 1.0

        try:
            while self.running:
                start_time = time.time()
                # 从共享内存读取 XR 手部 25 点坐标。
                with left_hand_array.get_lock():
                    left_hand_data  = np.array(left_hand_array[:]).reshape(25, 3).copy()
                with right_hand_array.get_lock():
                    right_hand_data = np.array(right_hand_array[:]).reshape(25, 3).copy()

                # 当前硬件状态用于输出记录。
                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))

                if not np.all(right_hand_data == 0.0) and not np.all(left_hand_data[4] == np.array([-1.13, 0.3, 0.15])): # if hand data has been initialized.
                    # retargeting 输入是人手关键点对构成的相对向量，减少全局手部位置影响。
                    # Inspire YAML 使用 5 指 DexPilot，除了拇指/食指/中指/无名指/小指之间的向量，
                    # 还包含掌根到各指尖的向量，用来保留整体手型。
                    ref_left_value = left_hand_data[self.hand_retargeting.left_indices[1,:]] - left_hand_data[self.hand_retargeting.left_indices[0,:]]
                    ref_right_value = right_hand_data[self.hand_retargeting.right_indices[1,:]] - right_hand_data[self.hand_retargeting.right_indices[0,:]]

                    # retarget() 先求 URDF 关节角，再按 Inspire 官方电机顺序重排。
                    left_q_target  = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[self.hand_retargeting.left_dex_retargeting_to_hardware]
                    right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]

                    # retargeting 输出是弧度，而 DFX 硬件接口要 0~1：
                    # 0.0 表示全闭，1.0 表示全开，所以用 (max - value) / range 做反向归一化。
                    def normalize(val, min_val, max_val):
                        return np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)

                    for idx in range(Inspire_Num_Motors):
                        if idx <= 3:
                            left_q_target[idx]  = normalize(left_q_target[idx], 0.0, 1.7)
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 1.7)
                        elif idx == 4:
                            left_q_target[idx]  = normalize(left_q_target[idx], 0.0, 0.5)
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 0.5)
                        elif idx == 5:
                            left_q_target[idx]  = normalize(left_q_target[idx], -0.1, 1.3)
                            right_q_target[idx] = normalize(right_q_target[idx], -0.1, 1.3)

                # action_data 保存最终发给硬件的 0~1 命令。
                action_data = np.concatenate((left_q_target, right_q_target))    
                if dual_hand_state_array and dual_hand_action_array:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data

                self.ctrl_dual_hand(left_q_target, right_q_target)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Inspire_Controller_DFX has been closed.")



kTopicInspireFTPLeftCommand   = "rt/inspire_hand/ctrl/l"
kTopicInspireFTPRightCommand  = "rt/inspire_hand/ctrl/r"
kTopicInspireFTPLeftState  = "rt/inspire_hand/state/l"
kTopicInspireFTPRightState = "rt/inspire_hand/state/r"

class Inspire_Controller_FTP:
    def __init__(self, left_hand_array, right_hand_array, dual_hand_data_lock = None, dual_hand_state_array = None,
                       dual_hand_action_array = None, fps = 100.0, Unit_Test = False, simulation_mode = False):
        """Inspire FTP 灵巧手控制器：接口使用 inspire_sdkpy，命令角度范围为 0~1000。"""
        logger_mp.info("Initialize Inspire_Controller_FTP...")
        from inspire_sdkpy import inspire_dds, inspire_hand_defaut # lazy import
        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode
        if not self.Unit_Test:
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND)
        else:
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND_Unit_Test)

        if self.simulation_mode:
            ChannelFactoryInitialize(1)
        else:
            ChannelFactoryInitialize(0)

        # FTP 左右手使用独立命令 topic。
        self.LeftHandCmd_publisher = ChannelPublisher(kTopicInspireFTPLeftCommand, inspire_dds.inspire_hand_ctrl)
        self.LeftHandCmd_publisher.Init()
        self.RightHandCmd_publisher = ChannelPublisher(kTopicInspireFTPRightCommand, inspire_dds.inspire_hand_ctrl)
        self.RightHandCmd_publisher.Init()

        # 左右手状态也分别订阅。
        self.LeftHandState_subscriber = ChannelSubscriber(kTopicInspireFTPLeftState, inspire_dds.inspire_hand_state)
        self.LeftHandState_subscriber.Init() # Consider using callback if preferred: Init(callback_func, period_ms)
        self.RightHandState_subscriber = ChannelSubscriber(kTopicInspireFTPRightState, inspire_dds.inspire_hand_state)
        self.RightHandState_subscriber.Init()

        # 状态 angle_act 原始范围是 0~1000，这里统一缓存为 0~1。
        self.left_hand_state_array  = Array('d', Inspire_Num_Motors, lock=True)
        self.right_hand_state_array = Array('d', Inspire_Num_Motors, lock=True)

        # 后台订阅线程读取 FTP hand state。
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        # 等待首次状态消息，超时后继续运行，方便调试没有手部硬件的场景。
        wait_count = 0
        while not (any(self.left_hand_state_array) or any(self.right_hand_state_array)):
            if wait_count % 100 == 0: # Print every second
                logger_mp.info(f"[Inspire_Controller_FTP] Waiting to subscribe to hand states from DDS (L: {any(self.left_hand_state_array)}, R: {any(self.right_hand_state_array)})...")
            time.sleep(0.01)
            wait_count += 1
            if wait_count > 500: # Timeout after 5 seconds
                logger_mp.warning("[Inspire_Controller_FTP] Warning: Timeout waiting for initial hand states. Proceeding anyway.")
                break
        logger_mp.info("[Inspire_Controller_FTP] Initial hand states received or timeout.")

        hand_control_process = Process(target=self.control_process, args=(left_hand_array, right_hand_array, self.left_hand_state_array, self.right_hand_state_array,
                                                                          dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array))
        hand_control_process.daemon = True
        hand_control_process.start()

        logger_mp.info("Initialize Inspire_Controller_FTP OK!\n")

    def _subscribe_hand_state(self):
        """订阅 FTP 左右手状态，并把 0~1000 的 angle_act 缩放到 0~1。"""
        logger_mp.info("[Inspire_Controller_FTP] Subscribe thread started.")
        while True:
            # Left Hand
            left_state_msg = self.LeftHandState_subscriber.Read()
            if left_state_msg is not None:
                if hasattr(left_state_msg, 'angle_act') and len(left_state_msg.angle_act) == Inspire_Num_Motors:
                    with self.left_hand_state_array.get_lock():
                        for i in range(Inspire_Num_Motors):
                            self.left_hand_state_array[i] = left_state_msg.angle_act[i] / 1000.0
                else:
                    logger_mp.warning(f"[Inspire_Controller_FTP] Received left_state_msg but attributes are missing or incorrect. Type: {type(left_state_msg)}, Content: {str(left_state_msg)[:100]}")
            # Right Hand
            right_state_msg = self.RightHandState_subscriber.Read()
            if right_state_msg is not None:
                if hasattr(right_state_msg, 'angle_act') and len(right_state_msg.angle_act) == Inspire_Num_Motors:
                    with self.right_hand_state_array.get_lock():
                        for i in range(Inspire_Num_Motors):
                            self.right_hand_state_array[i] = right_state_msg.angle_act[i] / 1000.0
                else:
                    logger_mp.warning(f"[Inspire_Controller_FTP] Received right_state_msg but attributes are missing or incorrect. Type: {type(right_state_msg)}, Content: {str(right_state_msg)[:100]}")

            time.sleep(0.002)

    def _send_hand_command(self, left_angle_cmd_scaled, right_angle_cmd_scaled):
        """
        向左右 FTP 手发送 0~1000 的角度命令。
        """
        # 左手命令。
        left_cmd_msg = inspire_hand_defaut.get_inspire_hand_ctrl()
        left_cmd_msg.angle_set = left_angle_cmd_scaled
        left_cmd_msg.mode = 0b0001 # Mode 1: Angle control
        self.LeftHandCmd_publisher.Write(left_cmd_msg)

        # 右手命令。
        right_cmd_msg = inspire_hand_defaut.get_inspire_hand_ctrl()
        right_cmd_msg.angle_set = right_angle_cmd_scaled
        right_cmd_msg.mode = 0b0001 # Mode 1: Angle control
        self.RightHandCmd_publisher.Write(right_cmd_msg)

    def control_process(self, left_hand_array, right_hand_array, left_hand_state_array, right_hand_state_array,
                              dual_hand_data_lock = None, dual_hand_state_array = None, dual_hand_action_array = None):
        """读取 XR 手部骨架，生成 FTP 所需的 0~1000 整数角度命令。"""
        logger_mp.info("[Inspire_Controller_FTP] Control process started.")
        self.running = True

        left_q_target  = np.full(Inspire_Num_Motors, 1.0)
        right_q_target = np.full(Inspire_Num_Motors, 1.0)

        try:
            while self.running:
                start_time = time.time()
                # 读取 XR 手部 25 点坐标。
                with left_hand_array.get_lock():
                    left_hand_data  = np.array(left_hand_array[:]).reshape(25, 3).copy()
                with right_hand_array.get_lock():
                    right_hand_data = np.array(right_hand_array[:]).reshape(25, 3).copy()

                # 当前硬件状态用于外部记录。
                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))

                if not np.all(right_hand_data == 0.0) and not np.all(left_hand_data[4] == np.array([-1.13, 0.3, 0.15])): # if hand data has been initialized.
                    # 使用相对向量做 retargeting，避免受手腕整体平移影响。
                    # 这里和 DFX 版本同样走 DexPilot/Vector 优化，只是最后发送给 FTP SDK 前要缩放到 0~1000。
                    ref_left_value = left_hand_data[self.hand_retargeting.left_indices[1,:]] - left_hand_data[self.hand_retargeting.left_indices[0,:]]
                    ref_right_value = right_hand_data[self.hand_retargeting.right_indices[1,:]] - right_hand_data[self.hand_retargeting.right_indices[0,:]]

                    left_q_target  = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[self.hand_retargeting.left_dex_retargeting_to_hardware]
                    right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]

                    def normalize(val, min_val, max_val):
                        return np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)

                    for idx in range(Inspire_Num_Motors):
                        if idx <= 3:
                            left_q_target[idx]  = normalize(left_q_target[idx], 0.0, 1.7)
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 1.7)
                        elif idx == 4:
                            left_q_target[idx]  = normalize(left_q_target[idx], 0.0, 0.5)
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 0.5)
                        elif idx == 5:
                            left_q_target[idx]  = normalize(left_q_target[idx], -0.1, 1.3)
                            right_q_target[idx] = normalize(right_q_target[idx], -0.1, 1.3)

                # FTP SDK 发送整数刻度，因此把 0~1 命令缩放到 0~1000。
                scaled_left_cmd = [int(np.clip(val * 1000, 0, 1000)) for val in left_q_target]
                scaled_right_cmd = [int(np.clip(val * 1000, 0, 1000)) for val in right_q_target]

                # 记录未缩放的 0~1 action，和 DFX 控制器保持一致。
                action_data = np.concatenate((left_q_target, right_q_target))
                if dual_hand_state_array and dual_hand_action_array:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data

                self._send_hand_command(scaled_left_cmd, scaled_right_cmd)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Inspire_Controller_FTP has been closed.")

# Inspire 官方电机顺序如下。控制时必须按这个顺序写入 MotorCmds_。
# Update hand state, according to the official documentation:
# 1. https://support.unitree.com/home/en/G1_developer/inspire_dfx_dexterous_hand
# 2. https://support.unitree.com/home/en/G1_developer/inspire_ftp_dexterity_hand
# the state sequence is as shown in the table below
# ┌──────┬───────┬──────┬────────┬────────┬────────────┬────────────────┬───────┬──────┬────────┬────────┬────────────┬────────────────┐
# │ Id   │   0   │  1   │   2    │   3    │     4      │       5        │   6   │  7   │   8    │   9    │    10      │       11       │
# ├──────┼───────┼──────┼────────┼────────┼────────────┼────────────────┼───────┼──────┼────────┼────────┼────────────┼────────────────┤
# │      │                    Right Hand                                │                   Left Hand                                  │
# │Joint │ pinky │ ring │ middle │ index  │ thumb-bend │ thumb-rotation │ pinky │ ring │ middle │ index  │ thumb-bend │ thumb-rotation │
# └──────┴───────┴──────┴────────┴────────┴────────────┴────────────────┴───────┴──────┴────────┴────────┴────────────┴────────────────┘
class Inspire_Right_Hand_JointIndex(IntEnum):
    kRightHandPinky = 0
    kRightHandRing = 1
    kRightHandMiddle = 2
    kRightHandIndex = 3
    kRightHandThumbBend = 4
    kRightHandThumbRotation = 5

class Inspire_Left_Hand_JointIndex(IntEnum):
    kLeftHandPinky = 6
    kLeftHandRing = 7
    kLeftHandMiddle = 8
    kLeftHandIndex = 9
    kLeftHandThumbBend = 10
    kLeftHandThumbRotation = 11
