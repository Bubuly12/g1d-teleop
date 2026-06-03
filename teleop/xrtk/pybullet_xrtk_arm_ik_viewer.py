#!/usr/bin/env python3
"""Visualize XRTK wrist-pose IK results in PyBullet.

This script does not connect to the robot DDS and does not publish motor
commands. It reads XRoboToolkit controller poses, solves the same G1 arm IK
used by teleop_hand_and_arm.py, and displays the resulting 14 arm joint angles
on the G1 URDF in PyBullet.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
TELEOP_DIR = REPO_ROOT / "teleop"
URDF_PATH = REPO_ROOT / "assets" / "g1" / "g1_body29_hand14.urdf"
sys.path.append(str(REPO_ROOT))


ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def import_pybullet():
    try:
        import pybullet as p
    except ImportError as exc:
        print("Failed to import pybullet.")
        print("Install it in your current conda env first:")
        print("  pip install pybullet")
        raise exc
    return p


def get_joint_name_to_index(pybullet, body_id):
    mapping = {}
    for joint_index in range(pybullet.getNumJoints(body_id)):
        joint_info = pybullet.getJointInfo(body_id, joint_index)
        joint_name = joint_info[1].decode("utf-8")
        mapping[joint_name] = joint_index
    return mapping


def set_arm_q(pybullet, body_id, joint_name_to_index, q):
    q = np.asarray(q, dtype=float)
    for name, value in zip(ARM_JOINT_NAMES, q):
        joint_index = joint_name_to_index.get(name)
        if joint_index is None:
            raise KeyError(f"Joint {name} not found in PyBullet URDF.")
        pybullet.resetJointState(body_id, joint_index, float(value))


def offset_pose_for_display(pose, base_z):
    """Add PyBullet display-only base height without changing IK coordinates."""
    display_pose = np.asarray(pose, dtype=float).copy()
    display_pose[2, 3] += base_z
    return display_pose


def draw_pose_axes(pybullet, pose, label, axis_length=0.12):
    origin = np.asarray(pose[:3, 3], dtype=float)
    rot = np.asarray(pose[:3, :3], dtype=float)
    colors = ([1, 0, 0], [0, 1, 0], [0, 0.2, 1])
    for axis_index, color in enumerate(colors):
        endpoint = origin + rot[:, axis_index] * axis_length
        pybullet.addUserDebugLine(origin, endpoint, color, lineWidth=3, lifeTime=0.0)
    pybullet.addUserDebugText(label, origin + np.array([0.0, 0.0, axis_length]), [1, 1, 1], textSize=1.1)


def setup_pybullet(gui=True, base_z=0.75):
    p = import_pybullet()
    client_mode = p.GUI if gui else p.DIRECT
    p.connect(client_mode)
    p.setAdditionalSearchPath(str(URDF_PATH.parent))
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    plane_collision = p.createCollisionShape(p.GEOM_PLANE)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=plane_collision, basePosition=[0, 0, -0.02])
    robot_id = p.loadURDF(str(URDF_PATH), [0, 0, base_z], useFixedBase=True)

    if gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=1.8,
            cameraYaw=135,
            cameraPitch=-20,
            cameraTargetPosition=[0.15, 0.0, base_z + 0.2],
        )

    return p, robot_id


def main():
    parser = argparse.ArgumentParser(description="PyBullet viewer for XRTK wrist pose IK dry-run.")
    parser.add_argument("--iterations", "-n", type=int, default=1000)
    parser.add_argument("--interval", "-t", type=float, default=0.05)
    parser.add_argument("--direct", action="store_true", help="Run PyBullet without GUI.")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--base-z", type=float, default=0.75, help="Display-only PyBullet pelvis/base height")
    args = parser.parse_args()

    # robot_arm_ik.py uses relative URDF/cache paths.
    os.chdir(TELEOP_DIR)

    from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
    from teleop.xrtk.xrobotoolkit_wrapper import XRoboToolkitWrapper

    print("Initializing IK model...")
    arm_ik = G1_29_ArmIK()
    q_current = np.zeros(arm_ik.reduced_robot.model.nq)
    dq_current = np.zeros(arm_ik.reduced_robot.model.nv)

    print("Initializing PyBullet...")
    pybullet, robot_id = setup_pybullet(gui=not args.direct, base_z=args.base_z)
    joint_name_to_index = get_joint_name_to_index(pybullet, robot_id)
    set_arm_q(pybullet, robot_id, joint_name_to_index, q_current)

    print("Showing arm zero/home pose. Enter 's' to start XRTK + IK following.")
    user_input = input("Please enter the start signal (enter 's' to start):\n")
    if user_input.lower() != "s":
        print("Start signal not received. Exit without starting XRTK.")
        pybullet.disconnect()
        raise SystemExit(0)

    print("Initializing XRoboToolkitWrapper...")
    tv_wrapper = XRoboToolkitWrapper(use_hand_tracking=False)

    try:
        for index in range(args.iterations):
            tele_data = tv_wrapper.get_tele_data()
            left_wrist = np.asarray(tele_data.left_wrist_pose, dtype=float)
            right_wrist = np.asarray(tele_data.right_wrist_pose, dtype=float)

            sol_q, _ = arm_ik.solve_ik(left_wrist, right_wrist, q_current, dq_current)
            q_current = np.asarray(sol_q, dtype=float)
            dq_current = np.zeros_like(q_current)

            set_arm_q(pybullet, robot_id, joint_name_to_index, q_current)

            pybullet.removeAllUserDebugItems()
            draw_pose_axes(pybullet, offset_pose_for_display(left_wrist, args.base_z), "L target")
            draw_pose_axes(pybullet, offset_pose_for_display(right_wrist, args.base_z), "R target")

            if args.print_every > 0 and index % args.print_every == 0:
                left_fk, right_fk = arm_ik.solve_fk(q_current)
                print("=" * 88)
                print(f"Iteration {index + 1}/{args.iterations}")
                print(f"left target xyz:  {np.array2string(left_wrist[:3, 3], precision=5, suppress_small=True)}")
                print(f"left fk xyz:      {np.array2string(np.asarray(left_fk[:3]), precision=5, suppress_small=True)}")
                print(f"right target xyz: {np.array2string(right_wrist[:3, 3], precision=5, suppress_small=True)}")
                print(f"right fk xyz:     {np.array2string(np.asarray(right_fk[:3]), precision=5, suppress_small=True)}")
                print(f"sol_q left[:7]:   {np.array2string(q_current[:7], precision=5, suppress_small=True)}")
                print(f"sol_q right[-7:]: {np.array2string(q_current[-7:], precision=5, suppress_small=True)}")

            pybullet.stepSimulation()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print("Closing XRoboToolkitWrapper and PyBullet...")
        if "tv_wrapper" in locals():
            tv_wrapper.close()
        pybullet.disconnect()


if __name__ == "__main__":
    main()
