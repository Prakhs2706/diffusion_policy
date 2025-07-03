import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
import os

# Forward Kinematics
def get_tf_mat(i, joint_angles):
    dh_params = np.array([
        [0, 0.333, 0, joint_angles[0]],
        [0, 0, -np.pi / 2, joint_angles[1]],
        [0, 0.316, np.pi / 2, joint_angles[2]],
        [0.0825, 0, np.pi / 2, joint_angles[3]],
        [-0.0825, 0.384, -np.pi / 2, joint_angles[4]],
        [0, 0, np.pi / 2, joint_angles[5]],
        [0.088, 0, np.pi / 2, joint_angles[6]],
        [0, 0.107, 0, 0],             # wrist → flange
        [0, 0, 0, -np.pi / 4],        # tool rotation offset
        [0.0, 0.1034, 0, 0]           # TCP offset
    ], dtype=np.float64)

    a, d, alpha, theta = dh_params[i]
    q = theta  # either joint-dependent or fixed

    return np.array([
        [np.cos(q), -np.sin(q), 0, a],
        [np.sin(q)*np.cos(alpha), np.cos(q)*np.cos(alpha), -np.sin(alpha), -np.sin(alpha)*d],
        [np.sin(q)*np.sin(alpha), np.cos(q)*np.sin(alpha), np.cos(alpha), np.cos(alpha)*d],
        [0, 0, 0, 1]
    ])

def forward_kinematics(joint_angles):
    T = np.eye(4)
    for i in range(10):
        T = T @ get_tf_mat(i, joint_angles)
    return T


# File paths
src_path = "/home/ksaha/visualplanning/visplanWM/src/visplan/demos/motionplanning/BlockStacking-v1/optimal_250_stacking.h5"
dst_path = src_path.replace(".h5", "_with_ee.h5")
os.system(f"cp {src_path} {dst_path}")

# Insert EE data
with h5py.File(dst_path, "r+") as f:
    for traj_name in f:
        if not traj_name.startswith("traj_"):
            continue
        print(f"Processing {traj_name}...")

        agent_group = f[f"{traj_name}/obs/agent"]
        qpos = agent_group["qpos"][:]  # (T, 9)
        T = qpos.shape[0]

        ee_positions = np.zeros((T, 3))
        ee_quaternions = np.zeros((T, 4))
        gripper_widths = np.zeros((T,), dtype=np.float32)

        for t in range(T):
            joint_angles = qpos[t]  # ✅ now using all 9
            T_ee =forward_kinematics(joint_angles)
            ee_positions[t] = T_ee[:3, 3]
            ee_quaternions[t] = R.from_matrix(T_ee[:3, :3]).as_quat() # [x, y, z, w]
            gripper_widths[t] = qpos[t][7] + qpos[t][8]  # ✅ fixed gripper width

        # Save datasets
        for key in ["ee_pos", "ee_quat", "gripper_width"]:
            if key in agent_group:
                del agent_group[key]

        agent_group.create_dataset("ee_pos", data=ee_positions)
        agent_group.create_dataset("ee_quat", data=ee_quaternions)
        agent_group.create_dataset("gripper_width", data=gripper_widths)

print("✅ Done. Modified file saved at:", dst_path)