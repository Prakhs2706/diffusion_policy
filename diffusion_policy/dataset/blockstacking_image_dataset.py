from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class BlockStackingImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'state', 'action'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'], #del x, dely, delz, deltheta_x, deltheta_y, deltheta_z, gripper_width
            'agent_pos': self.replay_buffer['state']  # x, y, z, theta_x, theta_y, theta_z, gripper_width
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        normalizer['goal_image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample, idx):
        agent_pos = sample['state'].astype(np.float32) 
        image = np.moveaxis(sample['img'], -1, 1) / 255.0  # (T, 3, H, W)

        # Use episode index of the first timestep in the sample
        sample_start_idx = self.sampler.indices[idx][0]
        episode_idxs = self.replay_buffer.get_episode_idxs()
        episode_id = episode_idxs[sample_start_idx]

        # Get last frame index for that episode
        goal_idx = self.replay_buffer.episode_ends[episode_id] - 1
        goal_img = self.replay_buffer['img'][goal_idx]           # (H, W, 3)
        goal_img = np.moveaxis(goal_img, -1, 0) / 255.0           # (3, H, W)

        # Repeat goal image along time axis
        goal_image = np.repeat(goal_img[None], repeats=image.shape[0], axis=0)  # (T, 3, H, W)

        data = {
            'obs': {
                'image': image.astype(np.float32),
                'agent_pos': agent_pos,
                'goal_image': goal_image.astype(np.float32),
            },
            'action': sample['action'].astype(np.float32)
        }
        return data

    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample, idx)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def test():
    import os
    zarr_path = os.path.expanduser('/home/prakhar/Downloads/20250721_161332_with_ee_augmented_sh.zarr')
    dataset = BlockStackingImageDataset(zarr_path, horizon=16)

