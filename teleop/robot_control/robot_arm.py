import numpy as np
import threading
import time
import os
from enum import IntEnum

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import ( LowCmd_  as hg_LowCmd, LowState_ as hg_LowState) # idl for g1-d
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC

from teleop.utils.gravity_feedforward import GravityFeedforward
import logging_mp
logger_mp = logging_mp.getLogger(__name__)

kTopicLowCommand_Debug  = "rt/lowcmd"
kTopicLowState = "rt/lowstate"

G1_29_Num_Motors = 35
 

class MotorState:
    """Keep only joint position q and velocity dq as a lightweight DDS LowState cache."""
    def __init__(self):
        self.q = None
        self.dq = None

class G1_29_LowState:
    """Cache the low-level state of all 35 motors."""
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(G1_29_Num_Motors)]

class DataBuffer:
    """Thread-safe data buffer: subscription threads write, control/main threads read."""
    def __init__(self):
        self.data = None
        self.lock = threading.Lock()

    def GetData(self):
        with self.lock:
            return self.data

    def SetData(self, data):
        with self.lock:
            self.data = data

class G1_29_ArmController:
    def __init__(self, simulation_mode = False,use_waist=False):
        """Dual-arm low-level control detail.

        Dual-arm low-level control detail.
        """
        logger_mp.info("Initialize G1_29_ArmController...")
        self.q_target = np.zeros(16)
        self.tauff_target = np.zeros(16)
        self.simulation_mode = simulation_mode

        self.kp_high = 300.0
        self.kd_high = 3.0
        self.kp_low = 80.0
        self.kd_low = 3.0
        self.kp_wrist = 40.0
        self.kd_wrist = 1.5
        self.kp_waist_pitch=800
        self.all_motor_q = None
        self.arm_velocity_limit = 20.0
        self.control_dt = 1.0 / 250.0
        self.tauf = -8.0509 
        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None
        self.use_waist = use_waist
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.gravity_feedforward = GravityFeedforward(
            urdf_path=os.path.join(repo_root, "assets", "g1_D", "g1_d.urdf"),
            joint_names=["waist_pitch_joint"]
        )


        # Dual-arm low-level control detail.
        if self.simulation_mode:
            ChannelFactoryInitialize(1)
        else:
            ChannelFactoryInitialize(0)

        # Dual-arm low-level control detail.
        self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()

        # Dual-arm low-level control detail.
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        while not self.lowstate_buffer.GetData():
            time.sleep(0.1)
            logger_mp.warning("[G1_29_ArmController] Waiting to subscribe dds...")
        logger_mp.info("[G1_29_ArmController] Subscribe dds ok.")

        # Dual-arm low-level control detail.
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()

        self.all_motor_q = self.get_current_motor_q()
        logger_mp.info(f"Current all body motor state q:\n{self.all_motor_q} \n")
        logger_mp.info(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        logger_mp.info("Lock all joints except two arms...")
        arm_indices = set(member.value for member in G1_29_JointArmIndex)
        for id in G1_29_JointIndex:
            # Dual-arm low-level control detail.
            self.msg.motor_cmd[id].mode = 1
            if id.value in arm_indices:
                # Dual-arm low-level control detail.
                if self._Is_wrist_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_wrist
                    self.msg.motor_cmd[id].kd = self.kd_wrist
                else:
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
            else:
                # Dual-arm low-level control detail.
                if self._Is_weak_motor(id):
                    if self._Is_waistPitch(id):
                        self.msg.motor_cmd[id].kp = 500
                        self.msg.motor_cmd[id].kd = 3.0 
                    elif self._Is_waistYaw(id):
                        self.msg.motor_cmd[id].kp = 100
                        self.msg.motor_cmd[id].kd = 2.0
                    else:
                        self.msg.motor_cmd[id].kp = self.kp_low
                        self.msg.motor_cmd[id].kd = self.kd_low
                else:
                    self.msg.motor_cmd[id].kp = self.kp_high
                    self.msg.motor_cmd[id].kd = self.kd_high
            self.msg.motor_cmd[id].q  = self.all_motor_q[id]
            
            # Dual-arm low-level control detail.
            if id.value == G1_29_JointWaistIndex.kWaistYaw.value:
                self.q_target[-2] = self.msg.motor_cmd[id].q 
                self.tauff_target[-2] = 0
            elif id.value == G1_29_JointWaistIndex.kWaistPitch.value:
                self.q_target[-1] = self.msg.motor_cmd[id].q
                self.tauff_target[-1] = -10
        logger_mp.info("Lock OK!\n")
        self.waist_state = self.get_current_waist_q()
        # Dual-arm low-level control detail.
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        logger_mp.info("Initialize G1_29_ArmController OK!")

    def _subscribe_motor_state(self):
        """Dual-arm low-level control detail."""
        while True:
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = G1_29_LowState()
                for id in range(G1_29_Num_Motors):
                    lowstate.motor_state[id].q  = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                self.lowstate_buffer.SetData(lowstate)
            time.sleep(0.002)

    def clip_arm_q_target(self, target_q, velocity_limit):
        """Dual-arm low-level control detail."""
        current_q = self.get_current_arm_waist_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_q_target

    def _ctrl_motor_state(self):
        """Dual-arm low-level control detail."""
        while True:
            start_time = time.time()

            with self.ctrl_lock:
                q_target     = self.q_target
                tauff_target = self.tauff_target
            if self.simulation_mode:
                cliped_q_target = q_target
            else:
                # Dual-arm low-level control detail.
                cliped_q_target = self.clip_arm_q_target(q_target, velocity_limit = self.arm_velocity_limit)
                    
            for idx, id in enumerate(G1_29_JointArmWaistIndex):
                self.msg.motor_cmd[id].q = cliped_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                if id == G1_29_JointWaistIndex.kWaistPitch:
                    # Dual-arm low-level control detail.
                    self.msg.motor_cmd[id].tau = self.tauf
                else:
                    if idx < len(tauff_target):
                        self.msg.motor_cmd[id].tau = tauff_target[idx]

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)

            if self._speed_gradual_max is True:
                # Dual-arm low-level control detail.
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)

    def ctrl_dual_arm(self, q_target, tauff_target, gravity_feedforward_tau=None):
        '''Dual-arm low-level control detail.'''
        if gravity_feedforward_tau is None:
            self.tauf = 0.0
        else:
            self.tauf = gravity_feedforward_tau
        q_arr = np.atleast_1d(q_target)

        if q_arr.shape[0] == 14:
            # Dual-arm low-level control detail.
            with self.ctrl_lock:
                self.q_target[:14] = q_arr[:14]
                self.tauff_target[:14] = tauff_target[:14]
        elif q_arr.shape[0] == 15:
            # Dual-arm low-level control detail.
            with self.ctrl_lock:
                self.q_target[:15] = q_arr[:15]
                self.tauff_target[:14] = tauff_target[:14]
        elif q_arr.shape[0] == 16:
            # Dual-arm low-level control detail.
            with self.ctrl_lock:
                self.q_target[:16] = q_arr[:16]
                self.tauff_target[:14] = tauff_target[:14]
        else:
            raise ValueError(f"Invalid q_target shape: {q_arr.shape}")

    def get_gravity_feedforward_data(self,waist_pitch_pos):
        """Dual-arm low-level control detail."""
        return self.gravity_feedforward.compute(np.array([waist_pitch_pos]))
    def get_mode_machine(self):
        '''Dual-arm low-level control detail.'''
        return self.lowstate_subscriber.Read().mode_machine
    
    def get_current_motor_q(self):
        '''Return current state q of all body motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_29_JointIndex])
    
    def get_current_dual_arm_q(self):
        '''Return current state q of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_29_JointArmIndex])
    def get_current_waist_q(self):
        '''Return current state q of the waist motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_29_JointWaistIndex])
    def get_current_arm_waist_q(self):
        '''Return current state q of the left and right arm and waist motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_29_JointArmWaistIndex])
    def get_current_dual_arm_dq(self):
        '''Return current state dq of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in G1_29_JointArmIndex])

    def ctrl_waist_go_home(self):
        """Dual-arm low-level control detail."""
        logger_mp.info("[G1_29_ArmController] Moving waist to home position...")
        tolerance = 0.015  # Dual-arm low-level control detail.
        if self.use_waist:
            waist_current = self.get_current_waist_q()[0]
            current_arm_q = self.get_current_dual_arm_q()
            self.q_target[:14] =current_arm_q
            self.tauf = 0 #self.get_gravity_feedforward_data(waist_current[1])
            
            # Dual-arm low-level control detail.
            if not np.all(np.abs(waist_current) < tolerance):
                # Dual-arm low-level control detail.
                distance = float(np.linalg.norm(waist_current))
                step_size = 0.01  # rad per step equivalent
                num_steps = max(150, int(np.ceil(distance / step_size)))
                waist_start = waist_current.copy()
                
                for step in range(num_steps):
                    # Dual-arm low-level control detail.
                    alpha = (step + 1) / num_steps  # 0 to 1
                    waist_target = waist_start * (1 - alpha)  # Gradually decrease to 0
                    with self.ctrl_lock:
                        self.q_target[-2] = waist_target
                    # Check if waist has reached home position
                    waist_current = self.get_current_waist_q()[0]
                    if np.all(np.abs(waist_current) < tolerance):
                        logger_mp.info("[G1_29_ArmController] Waist has reached home position.")
                        break
                    time.sleep(0.02)
                # Final check
                waist_current = self.get_current_waist_q()[0]
                if not np.all(np.abs(waist_current) < tolerance):
                    logger_mp.warning("[G1_29_ArmController] Waist did not fully reach home position.")
            else:
                logger_mp.info("[G1_29_ArmController] Waist already at home position.")
    def ctrl_dual_arm_go_home(self):
        '''Dual-arm low-level control detail.'''
        logger_mp.info("[G1_29_ArmController] ctrl_dual_arm_go_home start...")
        tolerance = 0.02  # Dual-arm low-level control detail.
        time.sleep(0.1)
        self.ctrl_waist_go_home()
        # Dual-arm low-level control detail.
        logger_mp.info("[G1_29_ArmController] Moving arms to home position...")
        max_attempts = 100
        current_attempts = 0
        with self.ctrl_lock:
            self.q_target = np.zeros(16)
        while current_attempts < max_attempts:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                logger_mp.info("[G1_29_ArmController] Both arms and waist have reached the home position.")
                break
            current_attempts += 1
            time.sleep(0.05)
        
        if current_attempts >= max_attempts:
            logger_mp.warning("[G1_29_ArmController] Arms did not reach home position within timeout.")

    def speed_gradual_max(self, t = 5.0):
        '''Dual-arm low-level control detail.'''
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        '''Dual-arm low-level control detail.'''
        self.arm_velocity_limit = 30.0

    def _Is_weak_motor(self, motor_index):
        """Dual-arm low-level control detail."""
        weak_motors = [
            G1_29_JointIndex.kLeftAnklePitch.value,
            G1_29_JointIndex.kRightAnklePitch.value,
            # Left arm
            G1_29_JointIndex.kLeftShoulderPitch.value,
            G1_29_JointIndex.kLeftShoulderRoll.value,
            G1_29_JointIndex.kLeftShoulderYaw.value,
            G1_29_JointIndex.kLeftElbow.value,
            # Right arm
            G1_29_JointIndex.kRightShoulderPitch.value,
            G1_29_JointIndex.kRightShoulderRoll.value,
            G1_29_JointIndex.kRightShoulderYaw.value,
            G1_29_JointIndex.kRightElbow.value,

            # Waist
            G1_29_JointWaistIndex.kWaistYaw.value,
            G1_29_JointWaistIndex.kWaistPitch.value,
        ]
        return motor_index.value in weak_motors
    
    def _Is_wrist_motor(self, motor_index):
        """Dual-arm low-level control detail."""
        wrist_motors = [
            G1_29_JointIndex.kLeftWristRoll.value,
            G1_29_JointIndex.kLeftWristPitch.value,
            G1_29_JointIndex.kLeftWristyaw.value,
            G1_29_JointIndex.kRightWristRoll.value,
            G1_29_JointIndex.kRightWristPitch.value,
            G1_29_JointIndex.kRightWristYaw.value,
        ]
        return motor_index.value in wrist_motors
    def _Is_waistPitch(self, motor_index):
        """Dual-arm low-level control detail."""
        waist_motors = [
            G1_29_JointWaistIndex.kWaistPitch.value
        ]
        return motor_index.value in waist_motors
    def _Is_waistYaw(self, motor_index):
        """Dual-arm low-level control detail."""
        waist_motors = [
            G1_29_JointWaistIndex.kWaistYaw.value
        ]
        return motor_index.value in waist_motors
class G1_29_JointArmIndex(IntEnum):
    """Dual-arm low-level control detail."""
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28

class G1_29_JointWaistIndex(IntEnum):
    """Dual-arm low-level control detail."""
    kWaistYaw = 12
    kWaistPitch = 13

class G1_29_JointArmWaistIndex(IntEnum):
    """Dual-arm low-level control detail."""
        # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28

    # Waist
    kWaistYaw = 12
    kWaistPitch = 13

    
class G1_29_JointIndex(IntEnum):
    """Dual-arm low-level control detail."""
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12
    kWaistRoll = 13
    kWaistPitch = 14

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28
    
    # not used
    kNotUsedJoint0 = 29
    kNotUsedJoint1 = 30
    kNotUsedJoint2 = 31
    kNotUsedJoint3 = 32
    kNotUsedJoint4 = 33
    kNotUsedJoint5 = 34
