import logging_mp
import os
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logging_mp.basicConfig(level=logging_mp.INFO, 
                       file=True, 
                       file_path=os.path.join(REPO_ROOT, "logs"),
                       backup_count=100,
                       max_file_size=50*1024*1024)
logger_mp = logging_mp.getLogger(__name__)
import time
import argparse
from multiprocessing import Value, Array, Lock
import threading
import numpy as np
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from teleop.xrtk.xrobotoolkit_wrapper import XRoboToolkitWrapper
from teleop.robot_control.robot_arm import G1_29_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller, Dex1_1_Gripper_Controller
from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX, Inspire_Controller_FTP
from teleop.robot_control.robot_hand_brainco import Brainco_Controller_hand, Brainco_Controller_ctrl
from teleop.robot_control.mobile_control import G1_Mobile_Lift_Controller
from teleop.utils.instruction_map import ControlDataMapper, HandleInstruction

from teleop.teleimager.src.teleimager.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.ipc import IPC_Server
# from teleop.utils.motion_switcher import MotionSwitcher
from sshkeyboard import listen_keyboard, stop_listening

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_


def start_pico_video_sender(img_server_ip: str, teleimager_port: int):
    """Start XRoboToolkit video sender as a sidecar process."""
    sender_path = os.path.join(
        REPO_ROOT,
        "third_party",
        "XRoboToolkit-Orin-Video-Sender",
        "TeleimagerVideoSender",
    )
    if not os.path.exists(sender_path):
        raise FileNotFoundError(
            f"TeleimagerVideoSender not found: {sender_path}. "
            "Build it with `make teleimager` first."
        )

    cmd = [
        sender_path,
        "--teleimager-host",
        img_server_ip,
        "--teleimager-port",
        str(teleimager_port),
    ]
    logger_mp.info(f"Starting Pico video sender: {' '.join(cmd)}")

    # The sender is a system-linked C++ binary. When launched from the conda
    # Python process it may inherit conda's libffi/libssl paths and break
    # system GStreamer libraries, so keep its dynamic loader path system-only.
    env = os.environ.copy()
    conda_prefix = env.get("CONDA_PREFIX")
    ld_library_path = env.get("LD_LIBRARY_PATH", "")
    if conda_prefix and ld_library_path:
        paths = [
            path for path in ld_library_path.split(os.pathsep)
            if path and not path.startswith(conda_prefix)
        ]
        if paths:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(paths)
        else:
            env.pop("LD_LIBRARY_PATH", None)

    return subprocess.Popen(cmd, env=env)


def stop_pico_video_sender(process):
    if process is None:
        return
    if process.poll() is not None:
        return

    logger_mp.info("Stopping Pico video sender...")
    process.terminate()
    try:
        process.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        logger_mp.warning("Pico video sender did not stop after terminate; killing it.")
        process.kill()
        process.wait(timeout=1.0)


def publish_reset_category(category: int, publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

# state transition
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
EPISODE_ID     = 0      # Episode ID (int) for IPC communication
#  -------        ---------                -----------                -----------            ---------
#   state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
#  -------        ---------      |         -----------      |         -----------      |     ---------
#   START           True         |manual      True          |manual      True          |        True
#   READY           True         |set         False         |set         False         |auto    True
#   RECORD_RUNNING  False        |to          True          |to          False         |        False
#                                ∨                          ∨                          ∨
#   RECORD_TOGGLE   False       True          False        True          False                  False
#  -------        ---------                -----------                 -----------            ---------
#  ==> manual: when READY is True, set RECORD_TOGGLE=True to transition.
#  --> auto  : Auto-transition after saving data.

def on_press(key, episode_id=None):
    global STOP, START, RECORD_TOGGLE, EPISODE_ID
    if key == 'r':
        START = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 's' and START == True:
        if episode_id is not None:
            EPISODE_ID = episode_id
        RECORD_TOGGLE = True
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, READY
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire_ftp', 'inspire_dfx', 'brainco'], help='Select end effector controller')
    
    # mobile base, elevation and waist control
    parser.add_argument('--base-type', type=str, choices=['mobile_lift', 'lift','legs'], default='mobile_lift', help='Select lower body type')
    parser.add_argument('--use-waist', action = 'store_true', help = 'Enable waist control')

    # mode flags
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server')
    parser.add_argument('--enable-pico-video', action='store_true', help='Start XRoboToolkit Teleimager video sender for Pico')
    parser.add_argument('--pico-video-teleimager-port', type=int, default=55555, help='Teleimager ZMQ camera port used by Pico video sender')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording mode')
    parser.add_argument('--task-dir', type = str, default = '/home/unitree/unitree_eai_environment/data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'task description', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording at json file')

    args = parser.parse_args()
    logger_mp.info(f"args: {args}")

    try:
        pico_video_sender = None
        if args.enable_pico_video:
            pico_video_sender = start_pico_video_sender(args.img_server_ip, args.pico_video_teleimager_port)

        # ipc communication mode. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press,get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication mode
        else:
            listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                                      kwargs={"on_press": on_press, "until": None, "sequential": False,}, 
                                                      daemon=True)
            listen_keyboard_thread.start()

        img_client = None
        camera_config = {
            "head_camera": {"enable_zmq": False, "binocular": False, "image_shape": [0, 0]},
            "left_wrist_camera": {"enable_zmq": False},
            "right_wrist_camera": {"enable_zmq": False},
        }
        if not args.headless:
            img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
            camera_config = img_client.get_cam_config()
            logger_mp.debug(f"Camera config: {camera_config}")

        # XR data wrapper: obtain hand/controller pose data from XRoboToolkit SDK.
        # Keep the variable name tv_wrapper so the rest of the control loop can keep using get_tele_data().
        tv_wrapper = XRoboToolkitWrapper(use_hand_tracking=args.input_mode == "hand")

        # Enter debug mode
        # motion_switcher = MotionSwitcher()
        # status, result = motion_switcher.Enter_Debug_Mode()
        # logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")
        
        # arm
        arm_ik = G1_29_ArmIK()
        arm_ctrl = G1_29_ArmController(simulation_mode=args.sim, use_waist=args.use_waist)
        arm_ctrl.ctrl_waist_go_home()


        # end-effector
        if args.ee == "dex3":
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
            hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                          dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "dex1":
            left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
            gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, 
                                                     dual_gripper_state_array, dual_gripper_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_dfx":
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_DFX(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_ftp":
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_FTP(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "brainco" and args.input_mode == "hand":
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller_hand(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                           dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "brainco" and args.input_mode == "controller":
            left_gripper_trigger_in = Value('d', 10.0, lock=True)  # [input]
            left_gripper_squeeze_in = Value('d', 0.0, lock=True)   # [input]
            right_gripper_trigger_in = Value('d', 10.0, lock=True) # [input]
            right_gripper_squeeze_in = Value('d', 0.0, lock=True)  # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller_ctrl(left_gripper_trigger_in, left_gripper_squeeze_in, right_gripper_trigger_in, right_gripper_squeeze_in,
                                                dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        else:
            pass

        # For mobile base and elevation control
        if args.base_type != "legs":
            # args.input_mode == "hand" use unitree R3-controller, otherwise use XR controller
            try:
                mobile_ctrl = G1_Mobile_Lift_Controller(args.base_type, args.input_mode == "hand", simulation_mode=args.sim)
            except Exception as e:
                STOP = True
                logger_mp.error(f"Failed to initialize mobile base/lift controller: {e}")
                
                raise
        else:
            mobile_ctrl=None
        control_data_mapper = ControlDataMapper(arm_ctrl)
        handle_instruction = HandleInstruction(args.input_mode == "hand", tv_wrapper, mobile_ctrl)

        # simulation mode
        if args.sim:
            reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            reset_pose_publisher.Init()
            from teleop.utils.sim_state_topic import start_sim_state_subscribe
            sim_state_subscriber = start_sim_state_subscribe()

        # record + headless / non-headless mode
        if args.record:
            recorder = EpisodeWriter(task_dir = os.path.join(args.task_dir, args.task_name),
                                     task_goal = args.task_goal,
                                     task_desc = args.task_desc,
                                     task_steps = args.task_steps,
                                     frequency = args.frequency, 
                                     rerun_log = not args.headless)

        logger_mp.info("Ready. Press Pico left X button or keyboard 'r' to start/stop teleoperation.")
        READY = True                  # now ready to (1) enter START state
        last_x_button = False
        last_start_state = False

        # main loop. When START is False, keep reading XR buttons but do not command the robot to follow.
        while not STOP:
            start_time = time.time()
            tele_data = tv_wrapper.get_tele_data()

            # Pico left X button is exposed as left_ctrl_aButton by XRoboToolkitWrapper.
            # Edge trigger: one press toggles START once, holding the button will not repeatedly toggle.
            x_button = bool(getattr(tele_data, "left_ctrl_aButton", False))
            if x_button and not last_x_button:
                if START:
                    logger_mp.info("[Pico X] Stop teleoperation and return arms to home.")
                    START = False
                    if args.record and RECORD_RUNNING:
                        RECORD_TOGGLE = True
                    try:
                        arm_ctrl.ctrl_dual_arm_go_home()
                    except Exception as e:
                        logger_mp.error(f"Failed to return arms home after Pico X stop: {e}")
                else:
                    logger_mp.info("[Pico X] Start teleoperation.")
                    START = True
            last_x_button = x_button

            if START and not last_start_state:
                logger_mp.info("---------------------start teleoperation-------------------------")
                arm_ctrl.speed_gradual_max()
            last_start_state = START

            # record mode
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    if START and recorder.create_episode(episode_id=EPISODE_ID):
                        RECORD_RUNNING = True
                    elif not START:
                        logger_mp.warning("Ignoring record start request because teleoperation is not START.")
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    recorder.save_episode()
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)

            if not START:
                sleep_time = max(0, (1 / args.frequency) - (time.time() - start_time))
                time.sleep(sleep_time)
                continue

            # get image
            if img_client is not None and camera_config['head_camera']['enable_zmq']:
                if args.record:
                    head_img = img_client.get_head_frame()
            if img_client is not None and camera_config['left_wrist_camera']['enable_zmq']:
                if args.record:
                    left_wrist_img = img_client.get_left_wrist_frame()
            if img_client is not None and camera_config['right_wrist_camera']['enable_zmq']:
                if args.record:
                    right_wrist_img = img_client.get_right_wrist_frame()

            # logger_mp.info(f"tele_data: {tele_data}")
            if args.ee in ("dex3", "inspire_ftp", "inspire_dfx", "brainco") and args.input_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif args.ee == "brainco" and args.input_mode == "controller":
                with left_gripper_trigger_in.get_lock():
                    left_gripper_trigger_in.value = tele_data.left_ctrl_triggerValue
                with left_gripper_squeeze_in.get_lock():
                    left_gripper_squeeze_in.value = tele_data.left_ctrl_squeezeValue
                with right_gripper_trigger_in.get_lock():
                    right_gripper_trigger_in.value = tele_data.right_ctrl_triggerValue
                with right_gripper_squeeze_in.get_lock():
                    right_gripper_squeeze_in.value = tele_data.right_ctrl_squeezeValue
            elif args.ee == "dex1" and args.input_mode == "controller":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_ctrl_triggerValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_ctrl_triggerValue
            elif args.ee == "dex1" and args.input_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue
            else:
                pass
            
            # get current robot state data.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()
            left_arm_pose_state, right_arm_pose_state = arm_ik.solve_fk(current_lr_arm_q)
            # solve ik using motor data and wrist pose, then use ik results to control arms.
            time_ik_start = time.time()
            sol_q, sol_tauff  = arm_ik.solve_ik(tele_data.left_wrist_pose, tele_data.right_wrist_pose, current_lr_arm_q, current_lr_arm_dq)
            left_arm_pose_action, right_arm_pose_action = arm_ik.solve_fk(sol_q)
            time_ik_end = time.time()
            logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
            # For mobile base and elevation control
            height_state = None
            height_action = [0.0]
            move_state = None
            move_action = [0.0, 0.0]
            waist_state = None
            waist_action = None
            if  mobile_ctrl is not None:
                height_state = mobile_ctrl.g1_height_state_array_out
                handle_instruction_data = handle_instruction.get_instruction()
                
                vel_data = control_data_mapper.update(ry=handle_instruction_data['ry'])
                height_action = np.array([vel_data['g1_height']]).tolist()
                mobile_ctrl.g1_height_action_array_in[0] = height_action[0]  
                if args.base_type == "mobile_lift":
                    move_state = mobile_ctrl.g1_move_state_array_out
                    handle_instruction_data = handle_instruction.get_instruction()
                    vel_data = control_data_mapper.update(lx=handle_instruction_data['lx'], ly=handle_instruction_data['ly'])
                    move_action = np.array([vel_data['mobile_x_vel'], vel_data['mobile_yaw_vel']]).tolist()
                    mobile_ctrl.g1_move_action_array_in[0] = move_action[0]  
                    mobile_ctrl.g1_move_action_array_in[1] = move_action[1] 

            if args.use_waist:
                handle_instruction_data = handle_instruction.get_instruction()
                waist_state = arm_ctrl.get_current_waist_q()
                gravity_feedforward_data = arm_ctrl.get_gravity_feedforward_data(waist_state[1])

                vel_data = control_data_mapper.update(
                    rx=handle_instruction_data['rx'],
                    rbutton_A=None, #handle_instruction_data['rbutton_A'],
                    rbutton_B=None, #handle_instruction_data['rbutton_B'],
                    current_waist_yaw=waist_state[0], 
                    current_waist_pitch=waist_state[1]
                )
                waist_action = np.array([vel_data['waist_yaw_pos'], vel_data['waist_pitch_pos']])
                
                sol_q = np.concatenate([sol_q, [waist_action[0]]])
            else:
                gravity_feedforward_data=0
            try:   
                arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff, gravity_feedforward_data)
            except Exception as e:
                logger_mp.error(f"Error 2: {e}")
                raise e
            # record data
            if args.record:
                READY = recorder.is_ready() # now ready to (2) enter RECORD_RUNNING state
                # dex hand or gripper
                if args.ee == "dex3" and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "hand":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "controller":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                elif (args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = []
                        current_body_action = []
                elif (args.ee == "brainco" and args.input_mode == "controller"):
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []

                # arm state and action
                left_arm_state  = current_lr_arm_q[:7]
                right_arm_state = current_lr_arm_q[-7:]
                left_arm_action = sol_q[:7]
                right_arm_action = sol_q[7:7+7]
                if RECORD_RUNNING:
                    colors = {}
                    depths = {}
                    if camera_config['head_camera']['binocular']:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr[:, :camera_config['head_camera']['image_shape'][1]//2]
                            colors[f"color_{1}"] = head_img.bgr[:, camera_config['head_camera']['image_shape'][1]//2:]
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{2}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{3}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    else:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{1}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{2}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    states = {
                        "left_arm": {                                                                    
                            "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                            "qvel":   [],                          
                            "torque": [],                        
                        }, 
                        "right_arm": {                                                                    
                            "qpos":   right_arm_state.tolist(),       
                            "qvel":   [],                          
                            "torque": [],                         
                        },       
                        "left_arm_pose": {
                            "qpos": left_arm_pose_state.tolist(),
                            "qvel": [],
                            "torque": [],
                        },
                        "right_arm_pose": {
                            "qpos": right_arm_pose_state.tolist(),
                            "qvel": [],
                            "torque": [],
                        },                  
                        "left_ee": {                                                                    
                            "qpos":   left_ee_state,           
                            "qvel":   [],                           
                            "torque": [],                          
                        }, 
                        "right_ee": {                                                                    
                            "qpos":   right_ee_state,       
                            "qvel":   [],                           
                            "torque": [],  
                        }, 
                        "body": {
                            "qpos": current_body_state,
                        }, 

                    }
                    actions = {
                        "left_arm": {                                   
                            "qpos":   left_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],      
                        }, 
                        "right_arm": {                                   
                            "qpos":   right_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],       
                        },     
                        "left_arm_pose": {
                            "qpos": left_arm_pose_action.tolist(),
                            "qvel": [],
                            "torque": [],
                        },
                        "right_arm_pose": {
                            "qpos": right_arm_pose_action.tolist(),
                            "qvel": [],
                            "torque": [],
                        },                     
                        "left_ee": {                                   
                            "qpos":   left_hand_action,       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_ee": {                                   
                            "qpos":   right_hand_action,       
                            "qvel":   [],       
                            "torque": [], 
                        }, 
                        "body": {
                            "qpos": current_body_action,
                        }, 

                        
                    }
                    if mobile_ctrl != None:
                        states["torso"] = {
                            "height": np.array(height_state[0]).tolist(),
                            "qvel": np.array(height_state[1]).tolist()
                        }
                        actions["torso"] = {
                            "qvel": np.array(height_action[0]).tolist()
                        }
                        if args.base_type == "mobile_lift":
                            states["chassis"] = {
                                "qvel": np.array(move_state).tolist()  # [x_vel, yaw_vel]
                            }
                            actions["chassis"] = {
                                "qvel": np.array(move_action).tolist()   # [x_vel, yaw_vel]
                            }
                    if args.use_waist and waist_state is not None and waist_action is not None:
                        states["waist"] = {
                            "qpos": np.array(waist_state).tolist(),  # [yaw, pitch]
                        }
                        actions["waist"] = {
                            "qpos": np.array(waist_action).tolist(),  # [yaw, pitch]
                        }

                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()            
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state)
                    else:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("KeyboardInterrupt, exiting program...")
    except Exception as e:
        logger_mp.error(f"Error: {e}")
    finally:
        try:
            if "arm_ctrl" in locals():
                arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")
        
        try:
            if args.ipc:
                ipc_server.stop()
            else:
                stop_listening()
                listen_keyboard_thread.join()
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")
        
        try:
            if img_client is not None:
                img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close image client: {e}")

        try:
            stop_pico_video_sender(pico_video_sender if "pico_video_sender" in locals() else None)
        except Exception as e:
            logger_mp.error(f"Failed to stop Pico video sender: {e}")

        try:
            tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close televuer wrapper: {e}")

        # try:
        #     if not args.motion:
        #         status, result = motion_switcher.Exit_Debug_Mode()
        #         logger_mp.info(f"Exit debug mode: {'Success' if status == 3104 else 'Failed'}")
        # except Exception as e:
        #     logger_mp.error(f"Failed to exit debug mode: {e}")

        try:
            if args.sim:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")
        
        try:
            if args.record:
                if RECORD_RUNNING:
                    logger_mp.info("Recording is still running on exit; saving episode before closing recorder.")
                    recorder.save_episode()
                    RECORD_RUNNING = False
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")
        logger_mp.info("Finally, exiting program.")
