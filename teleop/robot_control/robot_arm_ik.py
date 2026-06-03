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
        """G1 29 自由度模型的双臂 IK 求解器。

        输入是左右手腕目标位姿 4x4 矩阵，输出是 14 个双臂关节角和对应的动力学前馈力矩。
        """
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.Unit_Test = Unit_Test
        self.Visualization = Visualization

        # Pinocchio 从 URDF 构建模型比较慢，缓存后下次可以直接反序列化加载。
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.cache_path = os.path.join(repo_root, "teleop", "robot_control", "g1_29_model_cache.pkl")

        if not self.Unit_Test:
            self.urdf_path = os.path.join(repo_root, "assets", "g1", "g1_body29_hand14.urdf")
            self.model_dir = os.path.join(repo_root, "assets", "g1")
        else:
            self.urdf_path = os.path.join(repo_root, "assets", "g1", "g1_body29_hand14.urdf")
            self.model_dir = os.path.join(repo_root, "assets", "g1")

        # 优先加载缓存；没有缓存时才从 URDF 创建完整机器人模型。
        if os.path.exists(self.cache_path):
            logger_mp.info(f"[G1_29_ArmIK] >>> Loading cached robot model: {self.cache_path}")
            self.robot, self.reduced_robot = self.load_cache()
        else:
            logger_mp.info("[G1_29_ArmIK] >>> Cache not found. Loading URDF (slow)...")
            self.robot = pin.RobotWrapper.BuildFromURDF(self.urdf_path, self.model_dir)

            # IK 只求双臂，因此把腿、腰和手指关节锁住，减少优化变量数量。
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

            # URDF 里的手腕关节不一定正好是工具中心点，这里在左右腕 yaw 后方添加虚拟末端执行器 frame。
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
            # reduced model 和虚拟末端 frame 都构建完成后再缓存。
            self.save_cache()
            logger_mp.info(f"[G1_29_ArmIK]>>> Cache saved to {self.cache_path}")

        # CasADi 版本的 Pinocchio 模型用于构造可微的正运动学误差函数。
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # q 是待优化关节角；tf_l/tf_r 是外部传入的左右末端目标位姿。
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1) 
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # 取左右末端 frame id，并把平移误差、旋转误差封装成 CasADi Function。
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

        # 优化变量是整条双臂的关节角；var_q_last 用来惩罚与上一帧相差过大，提升遥操作平滑度。
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # for smooth
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # 关节角必须保持在 URDF 定义的上下限内。
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        # 平移权重大于旋转：遥操作中手的位置通常比姿态更影响可用性。
        # regularization 防止姿态漂到过大角度，smooth_cost 防止相邻帧抖动。
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

        # init_data 既是求解初值，也是下一帧平滑项的参考。
        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 14)
        self.vis = None

        if self.Visualization:
            # Meshcat 可视化真实末端 frame 和目标末端 frame，主要用于调试 IK 是否跟踪正确。
            self.vis = MeshcatVisualizer(self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
            self.vis.initViewer(open=True) 
            self.vis.loadViewerModel("pinocchio") 
            self.vis.displayFrames(True, frame_ids=[107, 108], axis_length = 0.15, axis_width = 5)
            self.vis.display(pin.neutral(self.reduced_robot.model))

            # 目标 frame 用彩色三轴线段显示，方便和机器人当前末端位姿对比。
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
        """保存完整模型和 reduced 模型，避免下次启动重复解析 URDF。"""
        data = {
            "robot_model": self.robot.model,
            "reduced_model": self.reduced_robot.model,
        }

        with open(self.cache_path, "wb") as f:
            pickle.dump(data, f)

    def load_cache(self):
        """从 pickle 里恢复 Pinocchio 模型，并重新创建运行时 data。"""
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
        """按人臂/机器人臂长比例缩放目标位置，适配两者臂长不同的情况。"""
        scale_factor = robot_arm_length / human_arm_length
        robot_left_pose = human_left_pose.copy()
        robot_right_pose = human_right_pose.copy()
        robot_left_pose[:3, 3] *= scale_factor
        robot_right_pose[:3, 3] *= scale_factor
        return robot_left_pose, robot_right_pose

    def solve_ik(self, left_wrist, right_wrist, current_lr_arm_motor_q = None, current_lr_arm_motor_dq = None):
        """求解左右手腕目标位姿对应的双臂关节角。

        left_wrist/right_wrist: 4x4 齐次变换矩阵。
        current_lr_arm_motor_q: 当前关节角，可作为优化初值，提高连续帧求解稳定性。
        """
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        # 如需要把人手空间缩放到机器人臂长，可打开下面这行。
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
            # 优化结果再过一层加权滑动平均，减少 IK 数值抖动传到电机。
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            # 当前实现把速度置零，只计算静态重力/惯性项下的 rnea 力矩。
            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            # rnea 根据 q、v、a 计算逆动力学力矩，这里 a=0。
            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            return sol_q, sol_tauff
        
        except Exception as e:
            logger_mp.error(f"ERROR in convergence, plotting debug info.{e}")

            # 求解失败时取 CasADi debug 中最后一次迭代值，便于定位目标位姿或关节限位问题。
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

            # 实机安全优先：失败时保持当前关节角，力矩置零，避免把异常解发送给机械臂。
            # return sol_q, sol_tauff
            return current_lr_arm_motor_q, np.zeros(self.reduced_robot.model.nv)
    def matrix_to_xyzrpy(self, T):
        """
        将 4x4 齐次变换矩阵转换为 [x, y, z, roll, pitch, yaw]。
        """
        assert T.shape == (4, 4)
        xyz = T[:3, 3]
        rpy = R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=False)
        return np.concatenate([xyz, rpy])
    
    def solve_fk(self, q_full):
        """根据双臂关节角计算左右末端的正运动学位姿。"""
        assert q_full.shape == (self.reduced_robot.model.nq,), f"Expected shape ({self.reduced_robot.model.nq},), got {q_full.shape}"
        
        try:
            pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q_full)
            
            # 使用 __init__ 里缓存的 frame id，避免每帧通过名字查找。
            L_ee_id = self.L_hand_id
            R_ee_id = self.R_hand_id
            
            # 单独更新末端 frame 位姿；返回值是内部数据引用，所以立刻 copy 成独立矩阵。
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
