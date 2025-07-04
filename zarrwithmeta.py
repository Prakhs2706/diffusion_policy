import os
import zarr
import numpy as np
import shutil

def convert_episodes_to_flat_data_zarr(
    src_zarr_path,
    dst_zarr_path,
    keys=('img', 'state', 'action')  # keys per episode to include
):
    if os.path.exists(dst_zarr_path):
        print(f"Removing existing: {dst_zarr_path}")
        shutil.rmtree(dst_zarr_path)

    src = zarr.open(src_zarr_path, mode='r')
    dst = zarr.open(dst_zarr_path, mode='w')

    episode_keys = sorted([k for k in src.group_keys() if k.isdigit()], key=int)
    print(f"Found {len(episode_keys)} episodes.")

    data_group = dst.create_group("data")
    meta_group = dst.create_group("meta")

    stacked = {k: [] for k in keys}
    episode_ends = []
    total_len = 0

    for k in episode_keys:
        ep = src[k]
        print(f"Processing episode {k}...")
        ep_len = None
        for key in keys:
            arr = ep[key][:]
            stacked[key].append(arr)
            if ep_len is None:
                ep_len = arr.shape[0]
            else:
                assert ep_len == arr.shape[0]
        total_len += ep_len
        episode_ends.append(total_len)

    for key in keys:
        arr = np.concatenate(stacked[key], axis=0)
        data_group.array(key, data=arr, chunks=(min(1000, arr.shape[0]),) + arr.shape[1:])

    episode_ends = np.array(episode_ends, dtype=np.int64)
    meta_group.array("episode_ends", data=episode_ends, chunks=episode_ends.shape)
    meta_group.attrs["n_episodes"] = len(episode_ends)
    meta_group.attrs["frame_count"] = int(episode_ends[-1])

    print(f"✅ Done. New ReplayBuffer-compatible Zarr at: {dst_zarr_path}")
    print(f"→ Total frames: {episode_ends[-1]}  | Episodes: {len(episode_ends)}")

# === USAGE ===
src = "/home/ksaha/Downloads/block_stacking_with_meta.zarr"
dst = "/home/ksaha/Downloads/block_stacking_replay.zarr"
convert_episodes_to_flat_data_zarr(src, dst)
