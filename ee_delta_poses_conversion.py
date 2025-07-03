import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

input_path = "/home/ksaha/visualplanning/visplanWM/src/visplan/demos/motionplanning/BlockStacking-v1/optimal_250_stacking_with_ee.h5"
output_path = "/home/ksaha/visualplanning/visplanWM/src/visplan/demos/motionplanning/BlockStacking-v1/optimal_250_stacking_with_ee_augmented.h5"

with h5py.File(input_path, "r") as f_in, h5py.File(output_path, "w") as f_out:
    for traj_key in f_in.keys():
        print(f"Processing {traj_key}")
        obs_in = f_in[f"{traj_key}/obs"]
        obs_out = f_out.create_group(f"{traj_key}/obs")
        
        # Copy non-agent groups
        for k in ["sensor_data", "sensor_param", "extra"]:
            f_in.copy(obs_in[k], obs_out)
        
        agent_in = obs_in["agent"]
        agent_out = obs_out.create_group("agent")

        # Copy existing agent data
        for k in agent_in.keys():
            agent_out.create_dataset(k, data=agent_in[k])

        # Load data
        ee_pos = agent_in["ee_pos"][:]        # (T, 3)
        ee_quat = agent_in["ee_quat"][:]      # (T, 4)
        gripper = agent_in["gripper_width"][:]  # (T,)
        T = ee_pos.shape[0]

        # === Compute outputs with T-shape (zero-padded at end) === #

        # Quaternion rotations
        r_t = R.from_quat(ee_quat)

        # --- ee_delta_pose: Δposition only ---
        ee_delta_pose = ee_pos[1:] - ee_pos[:-1]                         # (T-1, 3)
        ee_delta_pose = np.vstack([ee_delta_pose, np.zeros((1, 3))])    # (T, 3)

        # --- ee_axis_angles: absolute orientation at each t ---
        ee_axis_angles = r_t.as_rotvec()                                # (T, 3)

        # --- ee_delta_axis: relative rotation (t → t+1) as axis-angle ---
        ee_delta_axis = (r_t[:-1].inv() * r_t[1:]).as_rotvec()          # (T-1, 3)
        ee_delta_axis = np.vstack([ee_delta_axis, np.zeros((1, 3))])    # (T, 3)

        # --- ee_gripper_width (binary: open=1, closed=0) ---
        ee_gripper_width = (gripper >= 0.07).astype(np.float32)         # (T,)

        # --- delta_ee_gripper_width: discrete change ---
        delta_ee_gripper_width = ee_gripper_width[1:] - ee_gripper_width[:-1]  # (T-1,)
        delta_ee_gripper_width = np.concatenate([delta_ee_gripper_width, [0]])  # (T,)

        # === Save all new datasets ===
        agent_out.create_dataset("ee_delta_pose", data=ee_delta_pose)
        agent_out.create_dataset("ee_axis_angles", data=ee_axis_angles)
        agent_out.create_dataset("ee_delta_axis", data=ee_delta_axis)
        agent_out.create_dataset("ee_gripper_width", data=ee_gripper_width)
        agent_out.create_dataset("delta_ee_gripper_width", data=delta_ee_gripper_width)

print(f"\n✅ All trajectories processed and saved to:\n{output_path}")