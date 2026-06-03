import numpy as np
import time
from numpy.random import f
import pinocchio as pin
import logging_mp
logger_mp = logging_mp.getLogger(__name__)

class HandleInstruction:
    def __init__(self,r3_controller,tv_wrapper,mobile_ctrl):
        self.r3_controller = r3_controller
        self.tv_wrapper = tv_wrapper
        self.mobile_ctrl = mobile_ctrl
    def get_instruction(self):
        if self.r3_controller and self.mobile_ctrl is not None:
            lx = self.mobile_ctrl.unitree_handle_state_array_out[0]
            ly = -self.mobile_ctrl.unitree_handle_state_array_out[1]
            rx = -self.mobile_ctrl.unitree_handle_state_array_out[2]
            ry = -self.mobile_ctrl.unitree_handle_state_array_out[3]
            rbutton_A = True if int(self.mobile_ctrl.unitree_handle_state_array_out[4]) == 256 else False
            rbutton_B = True if int(self.mobile_ctrl.unitree_handle_state_array_out[4]) == 512 else False
        else:
            lx = -self.tv_wrapper.get_tele_data().left_ctrl_thumbstickValue[1]
            ly = -self.tv_wrapper.get_tele_data().left_ctrl_thumbstickValue[0]
            rx = -self.tv_wrapper.get_tele_data().right_ctrl_thumbstickValue[0]
            ry = -self.tv_wrapper.get_tele_data().right_ctrl_thumbstickValue[1]
            rbutton_A = self.tv_wrapper.get_tele_data().right_ctrl_aButton
            rbutton_B = self.tv_wrapper.get_tele_data().right_ctrl_bButton
        return {'lx': lx, 'ly': ly, 'rx': rx, 'ry': ry, 'rbutton_A': rbutton_A, 'rbutton_B': rbutton_B}

class LowPassFilter:
    """Low-pass filter for smoothing data"""
    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self._value = 0.0
        self._last_value = 0.0

    def update(self, new_value, max_accel=1.5):
        delta = new_value - self._last_value
        delta = np.clip(delta, -max_accel, max_accel)
        filtered = self.alpha * (self._last_value + delta) + (1 - self.alpha) * self._value
        self._last_value = filtered
        self._value = filtered
        return self._value


class ControlDataMapper:
    """
    Control data mapper for mobile base and elevation
    """
    def __init__(self, arm_ctrl=None):
        # Velocity filters
        self._filters = {
            'mobile_x_vel': LowPassFilter(alpha=0.15),
            'mobile_yaw_vel': LowPassFilter(alpha=0.15)
        }
        
        # Height accumulated value (remains unchanged after release)
        self.height_speed_value = 0
        self.mobile_x_vel = 0
        self.mobile_yaw_vel = 0
        self.arm_ctrl = arm_ctrl

        self.PITCH_MAX = 2.36
        self.PITCH_MIN = -0.035   
        self.last_timestamp = time.perf_counter()  
        waist_state = self.arm_ctrl.get_current_waist_q()
        self.waist_pitch_pos =   waist_state[1]
        self.waist_yaw_pos = waist_state[0]  
    def update(self, lx=None, ly=None, rx=None, ry=None, rbutton_A=None, rbutton_B=None, 
               current_waist_yaw=None,current_waist_pitch=None):
        """
        Update and map control parameters
        
        Args:
            lx: Left joystick X raw value (-1 to 1)
            ly: Left joystick Y raw value (-1 to 1)
            rx: Right joystick X raw value (-1 to 1)
            ry: Right joystick Y raw value (-1 to 1)
            rbutton_A: Right button A raw value (0 or 1)
            rbutton_B: Right button B raw value (0 or 1)
            current_waist_yaw: Current waist yaw position (rad)
            current_waist_pitch: Current waist pitch position (rad)
        Returns:
            dict: Dictionary containing mobile velocities and waist positions
        """
        if lx is not None:
            # Map forward velocity 
            raw = self._map_forward_velocity(lx)
            mobile_x_vel = self._filters['mobile_x_vel'].update(raw, max_accel=1.0)
            self.mobile_x_vel = mobile_x_vel
        else:
            mobile_x_vel = self.mobile_x_vel
        if ly is not None:
            raw = self._map_lateral_velocity(ly)
            mobile_yaw_vel = self._filters['mobile_yaw_vel'].update(raw, max_accel=1.0)
            self.mobile_yaw_vel = mobile_yaw_vel
        else:
            mobile_yaw_vel = self.mobile_yaw_vel
        # # Update waist yaw position based on joystick input and current position
        if rx is not None and current_waist_yaw is not None:
            waist_yaw_pos = self._update_waist_position(rx, current_waist_yaw,max_velocity=0.05,min_position=-2.5,max_position=2.5)
        elif current_waist_yaw is not None:
            waist_yaw_pos = self.waist_yaw_pos
        else:
            waist_yaw_pos = 0.0

        if ry is not None:
            self._update_height(ry)
        else:
            self.height_speed_value = 0
        if rbutton_A or rbutton_B:
            waist_pitch_pos = self._update_waist_picth_button(rbutton_A,rbutton_B,max_velocity=8.0,min_position=-0.02,max_position=2.3)
        else:
            waist_pitch_pos = self.waist_pitch_pos
        return {
            'mobile_x_vel': mobile_x_vel,
            'mobile_yaw_vel': mobile_yaw_vel,
            'waist_yaw_pos': waist_yaw_pos,
            'waist_pitch_pos': waist_pitch_pos,
            'g1_height': self.height_speed_value,
        }


    def _update_waist_position(self, raw_value, current_position, max_velocity,min_position,max_position):
        """
        Update waist position based on joystick input and current position
        Position increases/decreases based on joystick direction
        
        Args:
            raw_value: Raw joystick value (-1 to 1)
            current_position: Current waist position (rad)
            
        Returns:
            float: Updated waist position
        """
        deadzone = 0.5
        max_velocity = max_velocity  # Maximum position change per update (adjust for smooth control)
        min_position = min_position
        max_position = max_position
        
        if abs(raw_value) < deadzone:
            # Joystick in deadzone, maintain current position
            return self.waist_yaw_pos
        else:
            # Calculate velocity based on joystick input
            # Positive joystick -> increase position
            # Negative joystick -> decrease position
            sign = 1 if raw_value > 0 else -1
            intensity = (abs(raw_value) - deadzone) / (1.0 - deadzone)
            smooth = 6*intensity**5 - 15*intensity**4 + 10*intensity**3
            velocity = sign * smooth * max_velocity
            
            # Calculate new position based on current position
            self.waist_yaw_pos = current_position + velocity
            
            # Clamp to valid range
            self.waist_yaw_pos = np.clip(self.waist_yaw_pos, min_position, max_position)
            
            return self.waist_yaw_pos
    def _update_waist_picth_button(self,rbutton_A,rbutton_B,max_velocity=8.0,min_position=-0.02,max_position=2.3):
        
        current_timestamp = time.perf_counter()
        delta_t = current_timestamp - self.last_timestamp
        self.last_timestamp = current_timestamp
        if delta_t <= 0 or delta_t > 0.1: 
            return self.waist_pitch_pos
        delta_q = 0.0
        if rbutton_A and self.waist_pitch_pos < max_position:
            delta_q = delta_t * ((max_position - min_position) / max_velocity)

        
        elif rbutton_B and self.waist_pitch_pos > min_position:  
            delta_q = -delta_t * ((max_position - min_position) / max_velocity)
        
        if delta_q != 0:
            self.waist_pitch_pos += delta_q
            self.waist_pitch_pos = min(max_position, max(min_position, self.waist_pitch_pos))
        return self.waist_pitch_pos

    def _update_height(self, raw_value):
        """
        Update height value
        Height remains unchanged when joystick is released (won't drop down)
        
        Args:
            raw_value: Raw value (-1 to 1)
        """
        deadzone = 0.05
        max_range = 1.0 
        
        if abs(raw_value) < deadzone:
            self.height_speed_value = 0.0
        else:
            sign = 1 if raw_value > 0 else -1
            intensity = (abs(raw_value) - deadzone) / (1.0 - deadzone)
            smooth = 6*intensity**5 - 15*intensity**4 + 10*intensity**3
            height_value = sign * smooth * max_range
            self.height_speed_value = height_value
    
    def _map_forward_velocity(self, value):
        return self._smooth_map(value, -0.2, 0.2)
    
    def _map_lateral_velocity(self, value):
        return self._smooth_map(value, -0.6, 0.6)
    
    def _map_yaw_velocity(self, value,min_value,max_value):
        return self._smooth_map(value,min_value,max_value)
    
    def _smooth_map(self, value, out_min, out_max, deadzone=0.05):
        """
        Smooth mapping function
        Maps input value to output range using deadzone and smooth curve
        
        Args:
            value: Input value (-1 to 1)
            out_min: Output minimum value
            out_max: Output maximum value
            deadzone: Deadzone size
        """
        if abs(value) < deadzone:
            return 0.0
        t = (abs(value) - deadzone) / (1.0 - deadzone)
        t = np.clip(t, 0.0, 1.0)
        smooth = 6 * t**5 - 15 * t**4 + 10 * t**3
        return smooth * (out_max if value > 0 else out_min)
    
    def reset_height(self, height=0.0):
        """Reset height value"""
        self._height_value = height
    
    def get_current_height(self):
        """Get current height value"""
        return self._height_value
    