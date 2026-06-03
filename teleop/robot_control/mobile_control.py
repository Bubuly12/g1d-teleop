from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Point32_ # idl
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Twist_
from unitree_sdk2py.idl.default import geometry_msgs_msg_dds__Point32_,geometry_msgs_msg_dds__Twist_
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_
from unitree_sdk2py.idl.default import nav_msgs_msg_dds__Odometry_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
import numpy as np
from enum import IntEnum
import threading
import time
from multiprocessing import Process, Array
import os
import sys

import logging_mp
logger_mp = logging_mp.getLogger(__name__)

parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)



kTopicHeightCmd = "rt/cmd_hispeed"
kTopicHeightState = "rt/hispeed_state"
kTopicG1MoveCmd = "rt/cmd_vel_no_limit"
kTopicG1MoveState = "rt/slamware_ros_sdk_server_node/odom"



kTopicUnitreeHandle = "rt/wirelesscontroller"

class G1_Mobile_Lift_Controller:
    def __init__(self, base_type,r3_controller, fps = 30.0, Unit_Test = False, simulation_mode = False, filter_alpha=0.2, init_timeout=5.0):
        """
        初始化 G1 移动底盘和升降控制器。
        
        Args:
            base_type: 只升降或移动+升降；当前代码里 "mobile_lift" 表示同时控制移动底盘。
            r3_controller: 是否额外订阅 Unitree 手柄状态。
            fps: 控制进程发布频率。
        """
        logger_mp.info("Initialize G1_Mobile_Lift_Controller...")
        self.init_state = True
        # 防止 fps <= 0 导致控制循环里除零。
        if fps <= 0:
            logger_mp.warning(f"Invalid fps value: {fps}, using default value 30.0")
            self.fps = 30.0
        else:
            self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode
        self.base_type = base_type 
        self.r3_controller = r3_controller
        self.init_timeout = init_timeout
        # 初始化阶段用这些标志确认 DDS 状态数据已经收到。
        self.height_data_received = False
        self.move_data_received = False

        # 输入缓冲：主进程写入目标升降高度和移动速度，控制子进程读取并发布。
        self.g1_height_action_array_in = Array('d', 1, lock = True) 
        self.g1_move_action_array_in = Array('d', 2, lock = True)


        # 输出缓冲：订阅线程写入当前高度/速度，外部可读取并保存到数据集。
        self.g1_height_state_array_out  = Array('d', 2, lock=True)  
        self.g1_height_action_array_out = Array('d', 1, lock=True)  # For receiving published height action values, ready to save to dataset
        self.g1_move_state_array_out = None
        self.g1_move_action_array_out = None  # For receiving published movement action values, ready to save to dataset

        self.unitree_handle_state_array_out = None

        # DDS Domain: 仿真使用 1，真实机器人一般使用 0。
        if self.simulation_mode:
            ChannelFactoryInitialize(1)
        else:
            ChannelFactoryInitialize(0)
        # 升降控制：Point32.z 保存目标高度。
        self.HeightCmb_publisher = ChannelPublisher(kTopicHeightCmd, Point32_)
        self.HeightCmb_publisher.Init()
        # Height state subscriber
        self.HeightState_subscriber = ChannelSubscriber(kTopicHeightState, Point32_)
        self.HeightState_subscriber.Init()
        # Height action subscriber


        self.g1_height_msg = geometry_msgs_msg_dds__Point32_()

        # mobile_lift 模式下额外发布 cmd_vel，并订阅里程计速度。
        if self.base_type == "mobile_lift":
            self.g1_move_state_array_out = Array('d', 2, lock=True)
            self.g1_move_action_array_out = Array('d', 2, lock=True)  # For receiving published movement action values, ready to save to dataset
            # Movement control publisher
            self.G1MoveCmb_publisher = ChannelPublisher(kTopicG1MoveCmd, Twist_)
            self.G1MoveCmb_publisher.Init()
            self.g1_move_msg = geometry_msgs_msg_dds__Twist_()
            # Movement state subscriber
            self.G1MoveState_subscriber = ChannelSubscriber(kTopicG1MoveState, Odometry_)
            self.G1MoveState_subscriber.Init()


        self.subscribe_g1_mobilebase_state_thread = threading.Thread(target=self._subscribe_g1_mobilebase_state)
        self.subscribe_g1_mobilebase_state_thread.daemon = True
        self.subscribe_g1_mobilebase_state_thread.start()

        init_start_time = time.time()
        last_wait_log_time = 0.0
        while True:
            if self.base_type == "mobile_lift":
                # 移动升降模式需要同时收到高度和底盘速度数据，才认为初始化成功。
                if self.height_data_received and self.move_data_received:
                    self.init_state = False
                    logger_mp.info("[Initialization] Received height and movement data")
                    break
                else:
                    now = time.time()
                    if now - last_wait_log_time >= 1.0:
                        status = f"[Initialization] Waiting for DDS data... Height: {self.height_data_received}, Movement: {self.move_data_received}"
                        logger_mp.info(status)
                        last_wait_log_time = now
            else:
                # 只控制升降时，只要求高度状态 ready。
                if self.height_data_received:
                    self.init_state = False
                    logger_mp.info("[Initialization] Received height data")
                    break
                else:
                    now = time.time()
                    if now - last_wait_log_time >= 1.0:
                        logger_mp.info(f"[Initialization] Waiting for height data...")
                        last_wait_log_time = now

            if time.time() - init_start_time >= self.init_timeout:
                # 这里主动超时抛错，比一直卡住更容易定位 DDS 连接问题。
                error_msg = (
                    f"[Initialization] DDS wait timeout after {self.init_timeout:.1f}s. "
                    f"Height: {self.height_data_received}, Movement: {self.move_data_received}."
                )
                logger_mp.error(error_msg)
                raise RuntimeError(error_msg)
            time.sleep(0.02)
        
        logger_mp.info("[G1_Mobile_Lift_Controller] Subscribe dds ok.")
        # 使用 R3/Unitree 手柄时，单独订阅手柄摇杆和按键状态。
        if self.r3_controller:
            self.unitree_handle_state_array_out = Array('d', 5, lock=True)
            self.UnitreeHandleState_subscriber = ChannelSubscriber(kTopicUnitreeHandle, WirelessController_)
            self.UnitreeHandleState_subscriber.Init()
            self.subscribe_unitree_handle_state_thread = threading.Thread(target=self._subscribe_unitree_handle_state)
            self.subscribe_unitree_handle_state_thread.daemon = True
            self.subscribe_unitree_handle_state_thread.start()
        self.running = True
        # 控制发布放到子进程里，避免主进程图像/遥操作逻辑阻塞发布频率。
        mobile_control_process = Process(target=self.control_process, args=(self.base_type,))
        mobile_control_process.daemon = True
        mobile_control_process.start()

        logger_mp.info("Initialize G1_Mobile_Lift_Controller OK!\n")
    def _subscribe_unitree_handle_state(self):
        """订阅 Unitree 无线手柄，缓存左右摇杆和按键状态。"""
        while True:
            try:
                unitree_handle_msg = self.UnitreeHandleState_subscriber.Read()
                if unitree_handle_msg is not None:
                    self.unitree_handle_state_array_out[1] = unitree_handle_msg.lx
                    self.unitree_handle_state_array_out[0] = unitree_handle_msg.ly
                    self.unitree_handle_state_array_out[2] = unitree_handle_msg.rx
                    self.unitree_handle_state_array_out[3] = unitree_handle_msg.ry
                    self.unitree_handle_state_array_out[4] = unitree_handle_msg.keys
            except Exception as e:
                logger_mp.info(f"[_subscribe_unitree_handle_state] Exception: {e}")
                time.sleep(0.1)
            time.sleep(0.01)

    def _subscribe_g1_mobilebase_state(self):
        """订阅升降高度和底盘里程计速度。"""
        while True:
            try:
                height_msg = self.HeightState_subscriber.Read()
                if height_msg is not None:
                    # y/z 的具体含义由底层 topic 定义；这里保留两个通道供外部记录。
                    self.g1_height_state_array_out[0] = height_msg.y  # in meters
                    self.g1_height_state_array_out[1] = height_msg.z
                    
                    if not self.height_data_received:
                        self.height_data_received = True
                        
                if self.base_type == "mobile_lift":
                    move_msg = self.G1MoveState_subscriber.Read()
                    if move_msg is not None:
                        # 只取线速度 x 和角速度 z，正好对应 cmd_vel 的两个输入维度。
                        self.g1_move_state_array_out[0] = move_msg.twist.twist.linear.x
                        self.g1_move_state_array_out[1] = move_msg.twist.twist.angular.z
                        
                        if not self.move_data_received:
                            self.move_data_received = True
                        
                        # Apply deadzone (set to 0 when below threshold to eliminate jitter at rest)
                        # DEADZONE_THRESHOLD = 0.015
                        # if abs(self.g1_move_state_array_out[0]) < DEADZONE_THRESHOLD:
                        #     self.g1_move_state_array_out[0] = 0.0
                        # if abs(self.g1_move_state_array_out[1]) < DEADZONE_THRESHOLD:
                        #     self.g1_move_state_array_out[1] = 0.0
                time.sleep(0.01)
                
            except Exception as e:
                logger_mp.info(f"[_subscribe_g1_mobilebase_state] Exception: {e}")
                time.sleep(0.1) 
    def ctrl_g1_height(self, g1_height_target):
        """发布升降高度目标。"""
        self.g1_height_msg.z = g1_height_target
        self.HeightCmb_publisher.Write(self.g1_height_msg)
    

    def ctrl_g1_move(self, g1_move_target):
        """发布底盘运动目标：[线速度 x, 角速度 z]。"""
        self.g1_move_msg.linear.x = g1_move_target[0]
        self.g1_move_msg.angular.z = g1_move_target[1]
        self.G1MoveCmb_publisher.Write(self.g1_move_msg)
    def control_process(self, base_type):
        """固定频率发布升降和可选底盘速度命令。"""
        try:
            while self.running:
                start_time = time.time()
                # 从共享内存读取最新目标；外部只需更新数组，不直接触碰 DDS publisher。
                target_height = self.g1_height_action_array_in[0]
                self.ctrl_g1_height(target_height)
                
                if base_type == "mobile_lift":
                    g1_move_target = self.g1_move_action_array_in
                    self.ctrl_g1_move(g1_move_target)
                
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("G1_Mobilebase_Height_Controller has been closed.")


def _set_mobile_test_command(mobile_ctrl, base_type, height=0.0, move_x=0.0, yaw=0.0):
    """Write safe test commands into the controller shared arrays."""
    mobile_ctrl.g1_height_action_array_in[0] = height
    if base_type == "mobile_lift":
        mobile_ctrl.g1_move_action_array_in[0] = move_x
        mobile_ctrl.g1_move_action_array_in[1] = yaw


def _log_mobile_test_status(mobile_ctrl, base_type, label):
    height_state = list(mobile_ctrl.g1_height_state_array_out)
    height_action = list(mobile_ctrl.g1_height_action_array_in)
    if base_type == "mobile_lift":
        move_state = list(mobile_ctrl.g1_move_state_array_out)
        move_action = list(mobile_ctrl.g1_move_action_array_in)
    else:
        move_state = None
        move_action = None
    logger_mp.info(
        f"[mobile test] {label}; "
        f"height_state={height_state}; height_action={height_action}; "
        f"move_state={move_state}; move_action={move_action}"
    )


def _run_mobile_test_step(mobile_ctrl, base_type, label, duration, height=0.0, move_x=0.0, yaw=0.0, log_interval=0.5):
    _set_mobile_test_command(mobile_ctrl, base_type, height=height, move_x=move_x, yaw=yaw)
    end_time = time.time() + max(duration, 0.0)
    last_log_time = 0.0
    while time.time() < end_time:
        current_time = time.time()
        if current_time - last_log_time >= log_interval:
            last_log_time = current_time
            _log_mobile_test_status(mobile_ctrl, base_type, label)
        time.sleep(0.02)


def _apply_deadzone(value, deadzone=0.08):
    value = float(value)
    if abs(value) < deadzone:
        return 0.0
    return value


def _run_controller_mobile_test(
    mobile_ctrl,
    base_type,
    tv_wrapper,
    max_forward_speed,
    max_yaw_speed,
    max_height_speed,
):
    """Use XR controller thumbsticks to test mobile base/lift DDS control."""
    last_log_time = 0.0
    logger_mp.info("[mobile controller test] left stick: base forward/yaw; right stick up/down: lift; right B: stop.")
    while True:
        tele_data = tv_wrapper.get_tele_data()
        left_stick = np.asarray(tele_data.left_ctrl_thumbstickValue, dtype=float)
        right_stick = np.asarray(tele_data.right_ctrl_thumbstickValue, dtype=float)

        # Match teleop_hand_and_arm.py's controller convention after XR wrapper normalization.
        move_x = _apply_deadzone(-left_stick[1]) * max_forward_speed
        yaw = _apply_deadzone(-left_stick[0]) * max_yaw_speed
        height = _apply_deadzone(-right_stick[1]) * max_height_speed

        _set_mobile_test_command(mobile_ctrl, base_type, height=height, move_x=move_x, yaw=yaw)

        current_time = time.time()
        if current_time - last_log_time >= 0.5:
            last_log_time = current_time
            _log_mobile_test_status(
                mobile_ctrl,
                base_type,
                (
                    "controller "
                    f"left_stick={np.array2string(left_stick, precision=3, suppress_small=True)} "
                    f"right_stick={np.array2string(right_stick, precision=3, suppress_small=True)}"
                ),
            )

        if tele_data.right_ctrl_bButton:
            logger_mp.info("[mobile controller test] right B pressed, stopping.")
            break

        time.sleep(0.01)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Standalone safe DDS test for G1 mobile base and lift.")
    parser.add_argument("--base-type", type=str, choices=["lift", "mobile_lift"], default="mobile_lift")
    parser.add_argument("--test-mode", type=str, choices=["move", "lift", "both", "controller"], default="move")
    parser.add_argument("--sim", action="store_true", help="Use simulation DDS domain")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--init-timeout", type=float, default=5.0)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--forward-speed", type=float, default=0.05, help="Safe base linear.x test speed")
    parser.add_argument("--yaw-speed", type=float, default=0.15, help="Safe base angular.z test speed")
    parser.add_argument("--height-speed", type=float, default=0.15, help="Safe lift command value")
    parser.add_argument("--move-duration", type=float, default=1.0)
    parser.add_argument("--lift-duration", type=float, default=1.0)
    parser.add_argument("--stop-duration", type=float, default=1.0)
    args = parser.parse_args()
    logger_mp.info(f"args: {args}")

    if args.test_mode == "move" and args.base_type != "mobile_lift":
        raise ValueError("--test-mode move requires --base-type mobile_lift")
    if args.test_mode == "controller" and args.base_type != "mobile_lift":
        raise ValueError("--test-mode controller requires --base-type mobile_lift")

    mobile_ctrl = G1_Mobile_Lift_Controller(
        args.base_type,
        r3_controller=False,
        fps=args.fps,
        simulation_mode=args.sim,
        init_timeout=args.init_timeout,
    )

    user_input = input("Please enter the start signal (enter 's' to start the mobile/lift test):\n")
    if user_input.lower() != "s":
        _set_mobile_test_command(mobile_ctrl, args.base_type)
        logger_mp.info("Start signal not received. Exit without sending motion test commands.")
        raise SystemExit(0)

    try:
        if args.test_mode == "controller":
            from teleop.xrtk.xrobotoolkit_wrapper import XRoboToolkitWrapper

            tv_wrapper = XRoboToolkitWrapper(use_hand_tracking=False)
            _run_controller_mobile_test(
                mobile_ctrl,
                args.base_type,
                tv_wrapper,
                max_forward_speed=args.forward_speed,
                max_yaw_speed=args.yaw_speed,
                max_height_speed=args.height_speed,
            )
        else:
            for cycle in range(max(args.cycles, 1)):
                logger_mp.info(f"[mobile test] cycle {cycle + 1}/{max(args.cycles, 1)}")
                if args.test_mode in ("move", "both"):
                    _run_mobile_test_step(
                        mobile_ctrl,
                        args.base_type,
                        "base forward",
                        args.move_duration,
                        move_x=args.forward_speed,
                    )
                    _run_mobile_test_step(mobile_ctrl, args.base_type, "base stop", args.stop_duration)
                    _run_mobile_test_step(
                        mobile_ctrl,
                        args.base_type,
                        "base yaw",
                        args.move_duration,
                        yaw=args.yaw_speed,
                    )
                    _run_mobile_test_step(mobile_ctrl, args.base_type, "base stop", args.stop_duration)

                if args.test_mode in ("lift", "both"):
                    _run_mobile_test_step(
                        mobile_ctrl,
                        args.base_type,
                        "lift up command",
                        args.lift_duration,
                        height=args.height_speed,
                    )
                    _run_mobile_test_step(mobile_ctrl, args.base_type, "lift stop", args.stop_duration)
                    _run_mobile_test_step(
                        mobile_ctrl,
                        args.base_type,
                        "lift down command",
                        args.lift_duration,
                        height=-args.height_speed,
                    )
                    _run_mobile_test_step(mobile_ctrl, args.base_type, "lift stop", args.stop_duration)
    except KeyboardInterrupt:
        logger_mp.info("[mobile test] interrupted by user.")
    finally:
        _set_mobile_test_command(mobile_ctrl, args.base_type)
        _log_mobile_test_status(mobile_ctrl, args.base_type, "final stop")
        time.sleep(0.2)
