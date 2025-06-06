#!/usr/bin/env python

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from minizero.network.py.create_network import create_network
import gym

# Usage: python extract_latent_dataset.py game_type training_dir conf_file model_file start_iter end_iter output_file
# Example: python extract_latent_dataset.py atari /path/to/train_dir /path/to/conf.cfg weight_iter_10000.pkl 0 99 latent_dataset.npz

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs, flush=True)

if len(sys.argv) != 8:
    eprint("Usage: python extract_latent_dataset.py game_type training_dir conf_file model_file start_iter end_iter output_file")
    sys.exit(1)

game_type = sys.argv[1]
training_dir = sys.argv[2]
conf_file_name = sys.argv[3]
model_file = sys.argv[4]
start_iter = int(sys.argv[5])
end_iter = int(sys.argv[6])
output_file = sys.argv[7]

# Import pybind library for environment/game config
temp_mod = __import__(f'build.{game_type}', globals(), locals(), ['minizero_py'], 0)
py = temp_mod.minizero_py

py.load_config_file(conf_file_name)
data_loader = None
model = None

class AtariEnv:
    def __init__(self, game_name, gym_game_name, seed=None):
        self.game_name = game_name
        self.env = gym.make(gym_game_name, frameskip=4, repeat_action_probability=0.25, full_action_space=True, render_mode="rgb_array")
        self.env.metadata['render_fps'] = 60
        self.reset(seed)
        self.frames = []
        self.rewards = []  # Store rewards for each frame
        self.actions = []  # Store actions for each frame
        self.total_reward = 0

    def reset(self, seed):
        self.seed = seed
        self.done = False
        self.total_reward = 0
        self.env.ale.setInt("random_seed", seed)
        self.env.ale.loadROM(f"/opt/atari57/{self.game_name}.bin")
        self.observation, self.info = self.env.reset()
        self.frames = [self.env.render()]
        self.rewards = [0]  # Initial reward is 0
        self.actions = [0]  # Initial action is NOOP

    def act(self, action_id):
        self.observation, reward, self.done, terminated, self.info = self.env.step(action_id)
        self.frames.append(self.env.render())
        self.total_reward += reward
        self.rewards.append(reward)  # Store the immediate reward, not the total
        self.actions.append(action_id)
        return self.done or terminated

    def play_through_game(self, record):
        """Play through the game using the actions from the training record"""
        self.reset(self.seed)
        current_pos = 0
        
        # Extract actions from the record
        actions = record.split("B[")[1:]
        
        if not actions:
            return len(self.frames)
        
        # Play through the game using the actual actions
        for action in actions:
            if self.done:
                break
            try:
                action_id = int(action.split("|")[0].split(']')[0])
                # Map action to Atari environment
                atari_action = self.env.get_action_meanings().index(ACTION_MEANING[action_id])
                self.act(atari_action)
                current_pos += 1
            except (ValueError, IndexError) as e:
                continue
        
        return len(self.frames)

def get_diverse_frames(frames, rewards, num_samples=5, reward_counts=None):
    """Select diverse frames based on position and reward, ensuring reward diversity"""
    if len(frames) <= num_samples:
        return list(range(len(frames)))
    
    # Convert rewards to numpy array for easier manipulation
    rewards = np.array(rewards)
    unique_rewards = np.unique(rewards)
    non_zero_rewards = unique_rewards[unique_rewards > 0]
    
    # Initialize reward_counts if not provided
    if reward_counts is None:
        reward_counts = {r: 0 for r in unique_rewards}
    
    selected_indices = []
    
    # First, try to select one frame from each non-zero reward value
    # Prioritize rewards that have been sampled less frequently
    if len(non_zero_rewards) > 0:
        # Sort non-zero rewards by their current count (ascending)
        sorted_rewards = sorted(non_zero_rewards, key=lambda r: reward_counts.get(r, 0))
        
        for reward in sorted_rewards:
            # Get all frames with this reward
            reward_indices = np.where(rewards == reward)[0]
            if len(reward_indices) > 0:
                # Select the middle frame for this reward value
                mid_idx = len(reward_indices) // 2
                selected_indices.append(reward_indices[mid_idx])
                reward_counts[reward] = reward_counts.get(reward, 0) + 1
    
    # Fill remaining slots with frames that are well-distributed in time
    remaining_slots = num_samples - len(selected_indices)
    if remaining_slots > 0:
        # Get indices of frames we haven't selected yet
        remaining_indices = list(set(range(len(frames))) - set(selected_indices))
        
        # Calculate ideal positions for remaining frames
        if len(remaining_indices) > 0:
            # Distribute remaining frames evenly across the game
            positions = np.linspace(0, len(remaining_indices)-1, remaining_slots, dtype=int)
            for pos in positions:
                selected_indices.append(remaining_indices[pos])
                reward = rewards[remaining_indices[pos]]
                reward_counts[reward] = reward_counts.get(reward, 0) + 1
    
    # Sort the indices to maintain temporal order
    selected_indices = sorted(selected_indices)
    
    return selected_indices, reward_counts

def get_sgf_iteration(sgf_path):
    # Extract iteration number from SGF path
    # Example: atari_ms_pacman_mz_2bx64_n50-78660b-dirty/sgf/300.sgf -> 300
    filename = os.path.basename(sgf_path)
    return os.path.splitext(filename)[0]

def main():
    global data_loader, model
    data_loader = MinizeroDadaLoader(conf_file_name)
    model = Model()
    model.load_model(training_dir, model_file)

    # Prepare storage
    latent_vectors = []
    images = []
    rewards = []
    ids = []
    
    # Load data files
    data_loader.load_data(training_dir, start_iter, end_iter)
    
    # Get total number of game records from SGF files
    total_records = 0
    for sgf_path in data_loader.data_list:
        with open(sgf_path, 'r') as fin:
            records = fin.readlines()
            total_records += len(records)
    
    # Track reward distribution across all samples
    reward_counts = {}
    
    # Process each game record
    record_count = 0
    for sgf_path in data_loader.data_list:
        with open(sgf_path, 'r') as fin:
            records = fin.readlines()
            for record in records:
                env_id = record_count
                
                # Create environment and play through game
                env = AtariEnv("ms_pacman", "ALE/MsPacman-v5", seed=env_id)
                num_frames = env.play_through_game(record)
                
                if num_frames <= 1:
                    record_count += 1
                    continue
                
                # Get diverse frame indices, passing current reward distribution
                frame_indices, reward_counts = get_diverse_frames(env.frames, env.rewards, num_samples=5, reward_counts=reward_counts)
                
                # Print sampled rewards for this game
                sampled_rewards = [env.rewards[i] for i in frame_indices]
                print(f"Game {env_id}: Sampled rewards {sampled_rewards} (Total samples: {len(latent_vectors) + len(frame_indices)})")
                
                # Sample a batch to get latent vectors
                features, action_features, label_policy, label_value, label_reward, loss_scale, sampled_index = data_loader.sample_data(model.device)
                
                with torch.no_grad():
                    network_output = model.network(features)
                    
                    # Store selected frames and their latent vectors
                    for frame_idx in frame_indices:
                        unique_id = f"{env_id}:{frame_idx}"
                        latent = network_output["hidden_state"][0].cpu().numpy().flatten()
                        reward = env.rewards[frame_idx]
                        
                        latent_vectors.append(latent)
                        images.append(env.frames[frame_idx])
                        rewards.append(reward)
                        ids.append(unique_id)
                
                record_count += 1
                
                # Print current reward distribution every 100 games
                if record_count % 100 == 0:
                    print("\nCurrent reward distribution:")
                    for reward, count in sorted(reward_counts.items()):
                        percentage = (count / len(rewards)) * 100
                        print(f"Reward {reward}: {count} samples ({percentage:.1f}%)")
                    print(f"Total samples collected: {len(rewards)}")
                    print()

    if not latent_vectors:
        print("Error: No valid frames were collected")
        sys.exit(1)

    # Save as .npz
    latent_vectors = np.stack(latent_vectors)
    images = np.stack(images)
    rewards = np.array(rewards)
    ids = np.array(ids)
    
    # Print final statistics
    print("\nFinal dataset statistics:")
    print(f"Total samples: {len(latent_vectors)}")
    print(f"Unique rewards: {len(np.unique(rewards))}")
    print(f"Reward range: [{min(rewards)}, {max(rewards)}]")
    print(f"Image shape: {images.shape}")
    
    # Print final reward distribution
    print("\nFinal reward distribution:")
    for reward, count in sorted(reward_counts.items()):
        percentage = (count / len(rewards)) * 100
        print(f"Reward {reward}: {count} samples ({percentage:.1f}%)")
    
    np.savez_compressed(output_file, latent=latent_vectors, image=images, reward=rewards, id=ids)
    print(f"\nSaved dataset to {output_file}")

# Add ACTION_MEANING dictionary at the top of the file
ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}

# --- Required classes from train.py ---
class MinizeroDadaLoader:
    def __init__(self, conf_file_name):
        self.data_loader = py.DataLoader(conf_file_name)
        self.data_loader.initialize()
        self.data_list = []
        self.sampled_index = np.zeros(py.get_batch_size() * 2, dtype=np.int32)
        self.features = np.zeros(py.get_batch_size() * py.get_nn_num_input_channels() * py.get_nn_input_channel_height() * py.get_nn_input_channel_width(), dtype=np.float32)
        self.loss_scale = np.zeros(py.get_batch_size(), dtype=np.float32)
        self.value_accumulator = np.ones(1) if py.get_nn_discrete_value_size() == 1 else np.arange(-int(py.get_nn_discrete_value_size() / 2), int(py.get_nn_discrete_value_size() / 2) + 1)
        if py.get_nn_type_name() == "alphazero":
            self.action_features = None
            self.policy = np.zeros(py.get_batch_size() * py.get_nn_action_size(), dtype=np.float32)
            self.value = np.zeros(py.get_batch_size() * py.get_nn_discrete_value_size(), dtype=np.float32)
            self.reward = None
        else:
            self.action_features = np.zeros(py.get_batch_size() * py.get_muzero_unrolling_step() * py.get_nn_num_action_feature_channels()
                                            * py.get_nn_hidden_channel_height() * py.get_nn_hidden_channel_width(), dtype=np.float32)
            self.policy = np.zeros(py.get_batch_size() * (py.get_muzero_unrolling_step() + 1) * py.get_nn_action_size(), dtype=np.float32)
            self.value = np.zeros(py.get_batch_size() * (py.get_muzero_unrolling_step() + 1) * py.get_nn_discrete_value_size(), dtype=np.float32)
            self.reward = np.zeros(py.get_batch_size() * py.get_muzero_unrolling_step() * py.get_nn_discrete_value_size(), dtype=np.float32)

    def load_data(self, training_dir, start_iter, end_iter):
        for i in range(start_iter, end_iter + 1):
            file_name = f"{training_dir}/sgf/{i}.sgf"
            if file_name in self.data_list:
                continue
            self.data_loader.load_data_from_file(file_name)
            self.data_list.append(file_name)
            if len(self.data_list) > py.get_zero_replay_buffer():
                self.data_list.pop(0)

    def sample_data(self, device='cpu'):
        self.data_loader.sample_data(self.features, self.action_features, self.policy, self.value, self.reward, self.loss_scale, self.sampled_index)
        features = torch.FloatTensor(self.features).view(py.get_batch_size(), py.get_nn_num_input_channels(), py.get_nn_input_channel_height(), py.get_nn_input_channel_width()).to(device)
        action_features = None if self.action_features is None else torch.FloatTensor(self.action_features).view(py.get_batch_size(),
                                                                                                                 -1,
                                                                                                                 py.get_nn_num_action_feature_channels(),
                                                                                                                 py.get_nn_hidden_channel_height(),
                                                                                                                 py.get_nn_hidden_channel_width()).to(device)
        policy = torch.FloatTensor(self.policy).view(py.get_batch_size(), -1, py.get_nn_action_size()).to(device)
        value = torch.FloatTensor(self.value).view(py.get_batch_size(), -1, py.get_nn_discrete_value_size()).to(device)
        reward = None if self.reward is None else torch.FloatTensor(self.reward).view(py.get_batch_size(), -1, py.get_nn_discrete_value_size()).to(device)
        loss_scale = torch.FloatTensor(self.loss_scale / np.amax(self.loss_scale)).to(device)
        sampled_index = self.sampled_index
        return features, action_features, policy, value, reward, loss_scale, sampled_index

class Model:
    def __init__(self):
        self.training_step = 0
        self.network = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.optimizer = None
        self.scheduler = None
    def load_model(self, training_dir, model_file):
        self.training_step = 0
        self.network = create_network(py.get_game_name(),
                                      py.get_nn_num_input_channels(),
                                      py.get_nn_input_channel_height(),
                                      py.get_nn_input_channel_width(),
                                      py.get_nn_num_hidden_channels(),
                                      py.get_nn_hidden_channel_height(),
                                      py.get_nn_hidden_channel_width(),
                                      py.get_nn_num_action_feature_channels(),
                                      py.get_nn_num_blocks(),
                                      py.get_nn_action_size(),
                                      py.get_nn_num_value_hidden_channels(),
                                      py.get_nn_discrete_value_size(),
                                      py.get_nn_type_name())
        self.network.to(self.device)
        # for multi-gpu
        self.network = nn.DataParallel(self.network)

if __name__ == "__main__":
    main()