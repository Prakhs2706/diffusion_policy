import h5py
import zarr
import numpy as np
import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to axis-angle (rotation vector)."""
    rot = R.from_quat(quat)
    return rot.as_rotvec()

# ---- Paths ----
h5_path = "/home/ksaha/visualplanning/visplanWM/src/visplan/demos/motionplanning/BlockStacking-v1/optimal_250_stacking_with_ee_augmented.h5"
zarr_path = "/home/ksaha/Downloads/block_stacking.zarr"

os.makedirs(zarr_path, exist_ok=True)
root = zarr.open(zarr_path, mode='w')

with h5py.File(h5_path, "r") as f:
    for i, traj_name in enumerate(tqdm(list(f.keys()))):
        obs = f[f"{traj_name}/obs"]
        agent = obs["agent"]
        sensor_data = obs["sensor_data"]
        
        # (1) Images
        rgb = sensor_data["base_camera"]["rgb"][:]       # (T, H, W, 3)

        # (2) State
        ee_pos = agent["ee_pos"][:]                      # (T, 3)
        ee_quat = agent["ee_quat"][:]                    # (T, 4)
        ee_axis_angle = quat_to_axis_angle(ee_quat)      # (T, 3)
        gripper_width = agent["gripper_width"][:]        # (T,)
        gripper_width = gripper_width[:, None]           # (T, 1)

        state = np.concatenate([ee_pos, ee_axis_angle, gripper_width], axis=1)  # (T, 7)

        # (3) Action
        ee_delta_pose = agent["ee_delta_pose"][:]        # (T, 3)
        ee_delta_axis = agent["ee_delta_axis"][:]        # (T, 3)
        delta_gripper_width = np.diff(gripper_width, axis=0, prepend=gripper_width[0:1])  # (T, 1)

        action = np.concatenate([ee_delta_pose, ee_delta_axis, delta_gripper_width], axis=1)  # (T, 7)

        # (4) Write to Zarr
        traj_group = root.create_group(str(i))  # Must use str keys
        traj_group.create_dataset("img", data=rgb.astype(np.uint8))         # (T, H, W, 3)
        traj_group.create_dataset("state", data=state.astype(np.float32))   # (T, 7)
        traj_group.create_dataset("action", data=action.astype(np.float32)) # (T, 7)
