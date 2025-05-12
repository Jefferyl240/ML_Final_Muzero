import os
import sys
import subprocess
import math
import random
import configparser
import glob
import re

# Install required packages
def install_requirements():
    """Install required packages for Colab"""
    # First install ale-py
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ale-py'])
    
    # Then install other packages
    packages = [
        'opencv-python',
        'numpy',
        'torch'
    ]
    
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    
    # Create roms directory if it doesn't exist
    roms_dir = os.path.join(os.getcwd(), 'roms')
    os.makedirs(roms_dir, exist_ok=True)
    
    # Download Ms Pacman ROM if it doesn't exist
    rom_path = os.path.join(roms_dir, 'ms_pacman.bin')
    if not os.path.exists(rom_path):
        try:
            import urllib.request
            print("Downloading Ms Pacman ROM...")
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/mgbellemare/Arcade-Learning-Environment/master/roms/ms_pacman.bin",
                rom_path
            )
            print(f"ROM downloaded to: {rom_path}")
        except Exception as e:
            print(f"Error downloading ROM: {str(e)}")
            print("Please download ms_pacman.bin manually and place it in the ./roms directory")
            return False
    
    # Verify ROM can be loaded
    try:
        from ale_py import ALEInterface
        ale = ALEInterface()
        ale.loadROM(rom_path)
        print(f"ROM loaded successfully from: {rom_path}")
        return True
    except Exception as e:
        print(f"Error loading ROM: {str(e)}")
        return False

# Install requirements if running in Colab
if 'google.colab' in sys.modules:
    install_requirements()

import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from google.colab import files, output
import base64
from pathlib import Path
import cv2
from collections import deque
from ale_py import ALEInterface

# Constants from training environment
ATARI_FEATURE_HISTORY_SIZE = 8
ATARI_RESOLUTION = 96
ATARI_FRAME_SKIP = 4
ATARI_REPEAT_ACTION_PROBABILITY = 0.25
ATARI_ACTION_SIZE = 18  # Number of possible actions

class MCTSNode:
    def __init__(self):
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}
        self.hidden_state = None
        self.reward = 0.0
        self.prior = 0.0
        self.value = 0.0
        self.hidden_state_index = -1
        self.virtual_loss = 0.0

    def expanded(self):
        return len(self.children) > 0

    def get_mean(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def get_value(self):
        return self.reward + self.get_mean()

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """Add exploration noise to the prior probabilities."""
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        for a, n in zip(actions, noise):
            self.children[a].prior = (
                self.children[a].prior * (1 - exploration_fraction) 
                + n * exploration_fraction
            )

    def ucb_score(self, parent_visit_count, ucb_base, ucb_init, reward_discount):
        """Calculate the UCB score for this node."""
        if self.visit_count == 0:
            return float('inf')
        
        prior_score = (
            ucb_init * self.prior * 
            math.sqrt(parent_visit_count) / 
            (1 + self.visit_count)
        )
        value_score = -self.get_value()  # Negative because we want to maximize
        return value_score + prior_score

class ModelPlayer:
    def __init__(self):
        self.model_dirs = {
            'Ms. Pac-Man (MuZero, n=50)': 'mz_50',
            'Ms. Pac-Man (MuZero, n=5)': 'mz_5',
            'Ms. Pac-Man (MuZero, n=1)': 'mz_1'
        }
        self.ale = None
        self.model = None
        self.video_output = widgets.Output()
        self.frames = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mcts_config = None
        
    def load_config(self, model_dir):
        """Load network and MCTS configuration from the model's config file"""
        try:
            config_path = Path(model_dir) / 'config.cfg'
            if not config_path.exists():
                print(f"No config file found at {config_path}")
                return False
                
            # Read config file as text
            with open(config_path, 'r') as f:
                config_lines = f.readlines()
            
            # Parse config values
            config_values = {}
            for line in config_lines:
                # Skip comments and empty lines
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                # Parse key-value pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.split('#')[0].strip()  # Remove comments
                    config_values[key] = value
            
            # Network parameters
            self.network_config = {
                'num_input_channels': int(config_values.get('nn_num_input_channels', '32')),
                'input_channel_height': int(config_values.get('nn_input_channel_height', '96')),
                'input_channel_width': int(config_values.get('nn_input_channel_width', '96')),
                'num_hidden_channels': int(config_values.get('nn_num_hidden_channels', '256')),
                'hidden_channel_height': int(config_values.get('nn_hidden_channel_height', '6')),
                'hidden_channel_width': int(config_values.get('nn_hidden_channel_width', '6')),
                'num_blocks': int(config_values.get('nn_num_blocks', '10')),
                'action_size': int(config_values.get('nn_action_size', '18')),
                'num_value_hidden_channels': int(config_values.get('nn_num_value_hidden_channels', '256')),
                'discrete_value_size': int(config_values.get('nn_discrete_value_size', '601')),
                'num_action_feature_channels': int(config_values.get('nn_num_action_feature_channels', '18'))
            }
            
            # MCTS parameters
            self.mcts_config = {
                'num_simulations': int(config_values.get('actor_num_simulation', '50')),
                'puct_base': int(config_values.get('actor_mcts_puct_base', '19652')),
                'puct_init': float(config_values.get('actor_mcts_puct_init', '1.25')),
                'reward_discount': float(config_values.get('actor_mcts_reward_discount', '0.997')),
                'temperature': float(config_values.get('actor_select_action_softmax_temperature', '1.0')),
                'dirichlet_alpha': float(config_values.get('actor_mcts_dirichlet_alpha', '0.3')),
                'dirichlet_fraction': float(config_values.get('actor_mcts_dirichlet_fraction', '0.25'))
            }
            
            print(f"Loaded network config: {self.network_config}")
            print(f"Loaded MCTS config: {self.mcts_config}")
            return True
            
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            return False
        
    def setup_ui(self):
        """Create the user interface widgets"""
        self.model_dropdown = widgets.Dropdown(
            options=list(self.model_dirs.keys()),
            description='Select Model:',
            style={'description_width': 'initial'}
        )
        
        self.play_button = widgets.Button(
            description='Play Game',
            button_style='success'
        )
        
        self.play_button.on_click(self.play_game)
        
        # Display widgets
        display(HTML("<h2>MiniZero Model Player</h2>"))
        display(self.model_dropdown)
        display(self.play_button)
        display(self.video_output)
        
    def load_model(self, model_path):
        """Load a model from the given path"""
        try:
            model_path = os.path.join(os.getcwd(), model_path, 'model', 'model.pt')
            model = torch.jit.load(model_path)
            model.eval()
            model.to(self.device)
            
            # Check if this is a MuZeroAtariNetwork
            if hasattr(model, 'get_type_name'):
                model_type = model.get_type_name()
                print(f"Loaded model type: {model_type}")
                if model_type != "muzero_atari":
                    print(f"Error: Expected muzero_atari model, got {model_type}")
                    return None
            else:
                print("Error: Model does not have get_type_name method")
                return None
                
            return model
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
            
    def preprocess_frame(self, frame):
        """Preprocess frame for model input"""
        # Resize to match training resolution
        frame = cv2.resize(frame, (ATARI_RESOLUTION, ATARI_RESOLUTION))
        # Convert to RGB if grayscale
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        return frame

    def create_state_tensor(self, frame_buffer, action_buffer, state_tensor):
        """Create state tensor from frame and action buffers"""
        # Fill state tensor with action and RGB features
        for i in range(ATARI_FEATURE_HISTORY_SIZE):
            # Action feature (1 channel)
            action_id = action_buffer[i]
            state_tensor[0, i * 4] = action_id / ATARI_ACTION_SIZE
            
            # RGB features (3 channels)
            frame = frame_buffer[i]
            state_tensor[0, i * 4 + 1:i * 4 + 4] = torch.from_numpy(frame.transpose(2, 0, 1))
        
        return state_tensor

    def convert_value_logits(self, value_logits):
        """Convert value logits to scalar value using expected value"""
        # Create value bins from -300 to 300
        bins = torch.linspace(-300, 300, 601, device=value_logits.device)
        # Apply softmax to get probabilities
        probs = torch.softmax(value_logits, dim=-1)
        # Calculate expected value
        value = (bins * probs).sum().item()
        return value

    def run_mcts(self, state, root):
        """Run Monte Carlo Tree Search from the given state"""
        if not self.mcts_config:
            print("MCTS config not loaded")
            return

        # Initial inference for root node
        with torch.no_grad():
            model_output = self.model(state)
        
        # Convert policy to probabilities
        policy = model_output['policy'][0].cpu().numpy()
        
        # Convert value logits to scalar
        value_logits = model_output['value'][0]
        value = self.convert_value_logits(value_logits)
        
        # Store hidden state
        root.hidden_state = model_output['hidden_state']
        
        # Expand root node
        for action, prob in enumerate(policy):
            if prob > 0:
                root.children[action] = MCTSNode()
                root.children[action].prior = prob
        
        # Add exploration noise to root
        root.add_exploration_noise(
            dirichlet_alpha=0.3,  # From the MuZero paper
            exploration_fraction=0.25
        )
        
        # Batch size for parallel MCTS
        batch_size = 8
        
        # Run simulations in batches
        for sim_idx in range(0, self.mcts_config['num_simulations'], batch_size):
            current_batch_size = min(batch_size, self.mcts_config['num_simulations'] - sim_idx)
            
            # Prepare batch tensors
            hidden_states = []
            action_planes = []
            search_paths = []
            
            # Selection phase for each simulation in batch
            for _ in range(current_batch_size):
                node = root
                search_path = [node]
                
                while node.expanded():
                    best_action = None
                    best_ucb = float('-inf')
                    
                    for action, child in node.children.items():
                        ucb = child.ucb_score(
                            node.visit_count,
                            self.mcts_config['puct_base'],
                            self.mcts_config['puct_init'],
                            self.mcts_config['reward_discount']
                        )
                        if ucb > best_ucb:
                            best_action = action
                            best_ucb = ucb
                    
                    node = node.children[best_action]
                    search_path.append(node)
                
                # Store search path and prepare tensors
                search_paths.append(search_path)
                hidden_states.append(search_path[-2].hidden_state)
                
                # Create action plane for the selected action
                action_plane = torch.zeros(
                    (1, ATARI_ACTION_SIZE, self.network_config['hidden_channel_height'], self.network_config['hidden_channel_width']),
                    dtype=torch.float32,
                    device=self.device
                )
                action = list(search_path[-2].children.keys())[list(search_path[-2].children.values()).index(search_path[-1])]
                action_plane[0, action] = 1.0
                action_planes.append(action_plane)
            
            # Batch inference
            with torch.no_grad():
                # Stack tensors for batch processing
                hidden_states_batch = torch.cat(hidden_states, dim=0)
                action_planes_batch = torch.cat(action_planes, dim=0)
                
                # Run batch inference
                model_outputs = self.model(hidden_states_batch, action_planes_batch)
                
                # Process results for each simulation
                for i in range(current_batch_size):
                    node = search_paths[i][-1]
                    parent = search_paths[i][-2]
                    
                    # Get outputs for this simulation
                    policy = model_outputs['policy'][i].cpu().numpy()
                    value_logits = model_outputs['value'][i]
                    value = self.convert_value_logits(value_logits)
                    node.hidden_state = model_outputs['hidden_state'][i:i+1]  # Keep batch dimension
                    
                    # Expand node
                    for action, prob in enumerate(policy):
                        if prob > 0:
                            node.children[action] = MCTSNode()
                            node.children[action].prior = prob
                    
                    # Backpropagation
                    for n in reversed(search_paths[i]):
                        n.value_sum += value
                        n.visit_count += 1
                        value = n.reward + self.mcts_config['reward_discount'] * value

    def select_action(self, root, temperature=1.0):
        """Select action using visit count distribution."""
        visit_counts = np.array([child.visit_count for child in root.children.values()])
        actions = list(root.children.keys())
        
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        else:
            # Apply temperature
            visit_count_dist = visit_counts ** (1.0 / temperature)
            visit_count_dist = visit_count_dist / np.sum(visit_count_dist)
            action = np.random.choice(actions, p=visit_count_dist)
        
        return action
        
    def play_game(self, b):
        """Play the game using the selected model"""
        try:
            # Clear previous output
            self.video_output.clear_output(wait=True)
            
            # Get selected model
            selected_model = self.model_dirs[self.model_dropdown.value]
            print(f"Selected model: {selected_model}")
            
            # Load model
            self.model = self.load_model(selected_model)
            if self.model is None:
                with self.video_output:
                    print("Failed to load model")
                return
                
            print("Model loaded successfully")
            
            # Load MCTS config
            if not self.load_config(selected_model):
                with self.video_output:
                    print("Failed to load MCTS config")
                return
            print("MCTS config loaded successfully")
                
            # Create ALE environment
            self.ale = ALEInterface()
            # Use a random seed for each game
            random_seed = np.random.randint(0, 1000000)
            self.ale.setInt('random_seed', random_seed)
            self.ale.setInt('frame_skip', ATARI_FRAME_SKIP)
            self.ale.setFloat('repeat_action_probability', ATARI_REPEAT_ACTION_PROBABILITY)
            
            # Load ROM from local path
            rom_path = os.path.join(os.getcwd(), 'roms', 'ms_pacman.bin')
            if not os.path.exists(rom_path):
                with self.video_output:
                    print(f"ROM not found at {rom_path}")
                    print("Please run install_requirements() first")
                return
                
            self.ale.loadROM(rom_path)
            print("ROM loaded successfully")
            
            # Initialize frame buffer and action buffer
            frame_buffer = deque(maxlen=ATARI_FEATURE_HISTORY_SIZE)
            action_buffer = deque(maxlen=ATARI_FEATURE_HISTORY_SIZE)
            
            # Initialize with zeros
            for _ in range(ATARI_FEATURE_HISTORY_SIZE):
                frame_buffer.append(np.zeros((ATARI_RESOLUTION, ATARI_RESOLUTION, 3), dtype=np.float32))
                action_buffer.append(0)
            
            self.ale.reset_game()
            screen = self.ale.getScreenRGB()
            frame_buffer.append(self.preprocess_frame(screen))
            
            # Play game
            done = False
            total_reward = 0
            self.frames = []
            step = 0
            
            # Pre-allocate tensors for better performance
            state = torch.zeros((1, 32, ATARI_RESOLUTION, ATARI_RESOLUTION), dtype=torch.float32, device=self.device)
            print("Model Playing Game...")
            while not done:
                
                # Create state tensor
                self.create_state_tensor(frame_buffer, action_buffer, state)
                
                # Initialize MCTS root
                root = MCTSNode()
                
                # Run MCTS
                self.run_mcts(state, root)
                
                # Select action
                action = self.select_action(root)
                
                # Take action
                reward = self.ale.act(action)
                total_reward += reward
                done = self.ale.game_over()
                
                # Update buffers
                action_buffer.append(action)
                screen = self.ale.getScreenRGB()
                frame_buffer.append(self.preprocess_frame(screen))
                
                # Store frame for video
                self.frames.append(screen)
                
                step += 1
                
            # Create video
            self.create_video()
            
            with self.video_output:
                print(f"Game finished! Total reward: {total_reward}")
                print(f"Random seed used: {random_seed}")
                
        except Exception as e:
            with self.video_output:
                print(f"Error playing game: {str(e)}")
                import traceback
                traceback.print_exc()
                
    import matplotlib.animation as animation

    def create_video(self):
        """Create and display video from frames"""
        try:
            # Create temporary video file
            temp_video_path = "/content/gameplay.mp4"
            height, width, _ = self.frames[0].shape
            
            # Use matplotlib for video creation (more reliable in Colab)
            plt.figure(figsize=(8, 6))
            img = plt.imshow(self.frames[0])
            plt.axis('off')
            plt.tight_layout()
            
            # Create animation with 15 FPS to match training videos
            video = FuncAnimation(
                plt.gcf(),
                lambda i: img.set_data(self.frames[i]),
                frames=len(self.frames),
                interval=1000/15  # 15 FPS to match training videos
            )
            
            # Save video using matplotlib's FFMpegWriter
            video.save(temp_video_path, writer=matplotlib.animation.FFMpegWriter(fps=15))  # 15 FPS
            plt.close()  # Close the figure to free memory
            
            # Read video and convert to base64
            with open(temp_video_path, 'rb') as f:
                video_data = f.read()
            video_b64 = base64.b64encode(video_data).decode()
            
            # Create HTML5 video element
            video_html = f'''
            <video width="640" height="480" controls>
                <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            '''
            
            with self.video_output:
              clear_output(wait=True)
              display(HTML(video_html))
                
        except Exception as e:
            with self.video_output:
                print(f"Error creating video: {str(e)}")
                import traceback
                traceback.print_exc()







def main():
    player = ModelPlayer()
    player.setup_ui()

if __name__ == "__main__":
    main() 