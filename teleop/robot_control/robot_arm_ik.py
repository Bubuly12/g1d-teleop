import casadi                                                                       
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin                             
import pickle
from pinocchio import casadi as cpin    
from pinocchio.visualize import MeshcatVisualizer   
import os
import sys
import logging_mp
logger_mp = logging_mp.getLogger(__name__)
parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)
from scipy.spatial.transform import Rotation as R
from teleop.utils.weighted_moving_filter import WeightedMovingFilter

class G1_29_ArmIK:
    def __init__(self, Unit_Test = False, Visualization = False):
        """Dual-arm IK solver for the G1 29-DoF model.

        Inputs are 4x4 target wrist poses for both arms; outputs are the 14 arm joint angles and the corresponding dynamics feedforward torques.
        """
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.Unit_Test = Unit_Test
        self.Visualization = Visualization

        # Building the Pinocchio model from URDF is slow, so cache it and reload it directly next time.
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.cache_path = os.path.join(repo_root, "teleop", "robot_control", "g1_29_model_cache.pkl")

        if not self.Unit_Test:
            self.urdf_path = os.path.join(repo_root, "assets", "g1", "g1_body29_hand14.urdf")
            self.model_dir = os.path.join(repo_root, "assets", "g1")
        else:
            self.urdf_path = os.path.join(repo_root, "assets", "g1", "g1_body29_hand14.urdf")
            self.model_dir = os.path.join(repo_root, "assets", "g1")

        # Prefer the cached model; only rebuild the full robot model from URDF when the cache is missing.
        if os.path.exists(self.cache_path):
            logger_mp.info(f"[G1_29_ArmIK] >>> Loading cached robot model: {self.cache_path}")
            self.robot, self.reduced_robot = self.load_cache()
        else:
            logger_mp.info("[G1_29_ArmIK] >>> Cache not found. Loading URDF (slow)...")
            self.robot = pin.RobotWrapper.BuildFromURDF(self.urdf_path, self.model_dir)

            # IK only solves the two arms, so lock legs, waist, and finger joints to reduce the number of optimization variables.
            self.mixed_jointsToLockIDs = [
                                            "left_hip_pitch_joint" ,
                                            "left_hip_roll_joint" ,
                                            "left_hip_yaw_joint" ,
                                            "left_knee_joint" ,
                                            "left_ankle_pitch_joint" ,
                                            "left_ankle_roll_joint" ,
                                            "right_hip_pitch_joint" ,
                                            "right_hip_roll_joint" ,
                                            "right_hip_yaw_joint" ,
                                            "right_knee_joint" ,
                                            "right_ankle_pitch_joint" ,
                                            "right_ankle_roll_joint" ,
                                            "waist_yaw_joint" ,
                                            "waist_roll_joint" ,
                                            "waist_pitch_joint" ,
                                            
                                            "left_hand_thumb_0_joint" ,
                                            "left_hand_thumb_1_joint" ,
                                            "left_hand_thumb_2_joint" ,
                                            "left_hand_middle_0_joint" ,
                                            "left_hand_middle_1_joint" ,
                                            "left_hand_index_0_joint" ,
                                            "left_hand_index_1_joint" ,
                                            
                                            "right_hand_thumb_0_joint" ,
                                            "right_hand_thumb_1_joint" ,
                                            "right_hand_thumb_2_joint" ,
                                            "right_hand_index_0_joint" ,
                                            "right_hand_index_1_joint" ,
                                            "right_hand_middle_0_joint",
                                            "right_hand_middle_1_joint"
                                        ]

            self.reduced_robot = self.robot.buildReducedRobot(
                list_of_joints_to_lock=self.mixed_jointsToLockIDs,
                reference_configuration=np.array([0.0] * self.robot.model.nq),
            )

            # The URDF wrist joint is not always exactly at the tool center, so add virtual end-effector frames after the wrist yaw joints.
            self.reduced_robot.model.addFrame(
                pin.Frame('L_ee',
                          self.reduced_robot.model.getJointId('left_wrist_yaw_joint'),
                          pin.SE3(np.eye(3),
                                  np.array([0.05,0,0]).T),
                          pin.FrameType.OP_FRAME)
            )
            self.reduced_robot.model.addFrame(
                pin.Frame('R_ee',
                          self.reduced_robot.model.getJointId('right_wrist_yaw_joint'),
                          pin.SE3(np.eye(3),
                                  np.array([0.05,0,0]).T),
                          pin.FrameType.OP_FRAME)
            )
            # Cache only after the reduced model and virtual end-effector frames are fully built.
            self.save_cache()
            logger_mp.info(f"[G1_29_ArmIK]>>> Cache saved to {self.cache_path}")

        # The CasADi Pinocchio model is used to build differentiable forward-kinematics error functions.
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # q is the optimized joint vector; tf_l/tf_r are the external target poses for the left and right end effectors.
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1) 
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Fetch left/right end-effector frame ids and wrap translation/rotation errors as CasADi Functions.
        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3,3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3,3]
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3,:3].T),
                    cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3,:3].T)
                )
            ],
        )

        # The optimization variable is the dual-arm joint vector; var_q_last penalizes large jumps from the previous frame for smoother teleoperation.
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # for smooth
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Joint positions must stay within the limits defined by the URDF.
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        # Translation is weighted more than rotation because hand position usually matters more for teleoperation usability.
        # Regularization prevents excessive posture drift, and smooth_cost suppresses frame-to-frame jitter.
        self.opti.minimize(50 * self.translational_cost + self.rotation_cost + 0.02 * self.regularization_cost + 0.1 * self.smooth_cost)

        opts = {
            # CasADi-level options
            'expand': True, 
            'detect_simple_bounds': True,
            'calc_lam_p': False,  # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
            'print_time':False,   # print or not
            # IPOPT solver options
            'ipopt.sb': 'yes',    # disable Ipopt's license message
            'ipopt.print_level': 0,
            'ipopt.max_iter': 30, 
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 5e-4,
            'ipopt.acceptable_iter': 5,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.derivative_test': 'none',
            'ipopt.jacobian_approximation': 'exact',
            # 'ipopt.hessian_approximation': 'limited-memory',
        }
        self.opti.solver("ipopt", opts)

        # init_data is both the solver initial guess and the reference for the next frame's smoothness term.
        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 14)
        self.vis = None

        if self.Visualization:
            # Meshcat visualizes the actual and target end-effector frames, mainly for debugging IK tracking.
            self.vis = MeshcatVisualizer(self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
            self.vis.initViewer(open=True) 
            self.vis.loadViewerModel("pinocchio") 
            self.vis.displayFrames(True, frame_ids=[107, 108], axis_length = 0.15, axis_width = 5)
            self.vis.display(pin.neutral(self.reduced_robot.model))

            # Target frames are shown as colored XYZ axes so they can be compared with the robot end-effector poses.
            frame_viz_names = ['L_ee_target', 'R_ee_target']
            FRAME_AXIS_POSITIONS = (
                np.array([[0, 0, 0], [1, 0, 0],
                          [0, 0, 0], [0, 1, 0],
                          [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
            )
            FRAME_AXIS_COLORS = (
                np.array([[1, 0, 0], [1, 0.6, 0],
                          [0, 1, 0], [0.6, 1, 0],
                          [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
            )
            axis_length = 0.1
            axis_width = 20
            for frame_viz_name in frame_viz_names:
                self.vis.viewer[frame_viz_name].set_object(
                    mg.LineSegments(
                        mg.PointsGeometry(
                            position=axis_length * FRAME_AXIS_POSITIONS,
                            color=FRAME_AXIS_COLORS,
                        ),
                        mg.LineBasicMaterial(
                            linewidth=axis_width,
                            vertexColors=True,
                        ),
                    )
                )

    def save_cache(self):
        """Save the full and reduced models to avoid parsing the URDF again on the next startup."""
        data = {
            "robot_model": self.robot.model,
            "reduced_model": self.reduced_robot.model,
        }

        with open(self.cache_path, "wb") as f:
            pickle.dump(data, f)

    def load_cache(self):
        """Restore Pinocchio models from pickle and recreate runtime data objects."""
        with open(self.cache_path, "rb") as f:
            data = pickle.load(f)

        robot = pin.RobotWrapper()
        robot.model = data["robot_model"]
        robot.data = robot.model.createData()

        reduced_robot = pin.RobotWrapper()
        reduced_robot.model = data["reduced_model"]
        reduced_robot.data = reduced_robot.model.createData()

        return robot, reduced_robot
    
    def scale_arms(self, human_left_pose, human_right_pose, human_arm_length=0.60, robot_arm_length=0.75):
        """Scale target positions by the human/robot arm-length ratio to account for different arm lengths."""
        scale_factor = robot_arm_length / human_arm_length
        robot_left_pose = human_left_pose.copy()
        robot_right_pose = human_right_pose.copy()
        robot_left_pose[:3, 3] *= scale_factor
        robot_right_pose[:3, 3] *= scale_factor
        return robot_left_pose, robot_right_pose

    def solve_ik(self, left_wrist, right_wrist, current_lr_arm_motor_q = None, current_lr_arm_motor_dq = None):
        """Solve dual-arm joint angles for the left and right wrist target poses.

        left_wrist/right_wrist: 4x4 homogeneous transformation matrices.
        Dual-arm IK solver detail.
        """
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        # Dual-arm IK solver detail.
        # left_wrist, right_wrist = self.scale_arms(left_wrist, right_wrist)
        if self.Visualization:
            self.vis.viewer['L_ee_target'].set_transform(left_wrist)   # for visualization
            self.vis.viewer['R_ee_target'].set_transform(right_wrist)  # for visualization

        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            sol = self.opti.solve()
            # sol = self.opti.solve_limited()

            sol_q = self.opti.value(self.var_q)
            # Dual-arm IK solver detail.
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            # Dual-arm IK solver detail.
            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            # Dual-arm IK solver detail.
            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            return sol_q, sol_tauff
        
        except Exception as e:
            logger_mp.error(f"ERROR in convergence, plotting debug info.{e}")

            # Dual-arm IK solver detail.
            sol_q = self.opti.debug.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            logger_mp.error(f"sol_q:{sol_q} \nmotorstate: \n{current_lr_arm_motor_q} \nleft_pose: \n{left_wrist} \nright_pose: \n{right_wrist}")
            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            # Dual-arm IK solver detail.
            # return sol_q, sol_tauff
            return current_lr_arm_motor_q, np.zeros(self.reduced_robot.model.nv)
    def matrix_to_xyzrpy(self, T):
        """
        Dual-arm IK solver detail.
        """
        assert T.shape == (4, 4)
        xyz = T[:3, 3]
        rpy = R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=False)
        return np.concatenate([xyz, rpy])
    
    def solve_fk(self, q_full):
        """Dual-arm IK solver detail."""
        assert q_full.shape == (self.reduced_robot.model.nq,), f"Expected shape ({self.reduced_robot.model.nq},), got {q_full.shape}"
        
        try:
            pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q_full)
            
            # Dual-arm IK solver detail.
            L_ee_id = self.L_hand_id
            R_ee_id = self.R_hand_id
            
            # Dual-arm IK solver detail.
            ee_pose_l_se3 = pin.updateFramePlacement(self.reduced_robot.model, self.reduced_robot.data, L_ee_id)
            ee_pose_r_se3 = pin.updateFramePlacement(self.reduced_robot.model, self.reduced_robot.data, R_ee_id)
            
            # Copy the homogeneous matrix data immediately to avoid reference issues
            ee_pose_l = ee_pose_l_se3.homogeneous.copy()
            ee_pose_r = ee_pose_r_se3.homogeneous.copy()
            
            ee_xyzrpy_l = self.matrix_to_xyzrpy(ee_pose_l)
            ee_xyzrpy_r = self.matrix_to_xyzrpy(ee_pose_r)
            
            return ee_xyzrpy_l, ee_xyzrpy_r
        except Exception as e:
            logger_mp.error(f"Error in solve_fk: {e}")
            logger_mp.error(f"q_full shape: {q_full.shape}, model.nq: {self.reduced_robot.model.nq}")
            logger_mp.error(f"L_hand_id: {self.L_hand_id}, R_hand_id: {self.R_hand_id}")
            logger_mp.error(f"oMf length: {len(self.reduced_robot.data.oMf)}, nframes: {self.reduced_robot.model.nframes}")
            raise
