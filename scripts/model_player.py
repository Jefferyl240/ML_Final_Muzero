import os
import sys
import subprocess
import math
import random
import configparser

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
    def __init__(self, prior=0.0):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0.0
        self.children = {}
        self.hidden_state = None
        self.reward = 0.0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

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
        """Load MCTS configuration from the model's config file"""
        try:
            config_path = Path(model_dir) / 'config.cfg'
            if not config_path.exists():
                print(f"No config file found at {config_path}")
                return False
                
            config = configparser.ConfigParser()
            config.read(config_path)
            
            # Load MCTS parameters
            self.mcts_config = {
                'num_simulations': config.getint('actor', 'num_simulation'),
                'puct_base': config.getint('actor', 'mcts_puct_base'),
                'puct_init': config.getfloat('actor', 'mcts_puct_init'),
                'reward_discount': config.getfloat('actor', 'mcts_reward_discount'),
                'temperature': config.getfloat('actor', 'select_action_softmax_temperature')
            }
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
        
    def load_model(self, model_dir):
        """Load the selected model"""
        try:
            # Load config first
            if not self.load_config(model_dir):
                return None
                
            # Try to load .pt file first
            model_path = Path(model_dir) / 'model' / 'model.pt'
            if not model_path.exists():
                # If .pt doesn't exist, try .pkl
                model_path = Path(model_dir) / 'model' / 'model.pkl'
                if not model_path.exists():
                    print(f"No model found at {model_dir}/model/")
                    return None
            
            # Load model based on file extension
            if model_path.suffix == '.pt':
                # Try loading as TorchScript model first
                try:
                    model = torch.jit.load(str(model_path), map_location=self.device)
                except:
                    # If that fails, try loading as regular PyTorch model
                    model = torch.load(str(model_path), map_location=self.device)
            else:  # .pkl file
                import pickle
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                    if isinstance(model, torch.nn.Module):
                        model = model.to(self.device)
            
            # Ensure model is in eval mode
            if hasattr(model, 'eval'):
                model.eval()
            
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

    def create_state_tensor(self, frame_buffer, action_buffer):
        """Create state tensor from frame and action buffers"""
        state = np.zeros((1, ATARI_FEATURE_HISTORY_SIZE * 4, ATARI_RESOLUTION, ATARI_RESOLUTION), dtype=np.float32)
        
        # Fill state tensor with action and RGB features
        for i in range(ATARI_FEATURE_HISTORY_SIZE):
            # Action feature (1 channel)
            action_id = action_buffer[i]
            state[0, i * 4] = action_id / ATARI_ACTION_SIZE
            
            # RGB features (3 channels)
            frame = frame_buffer[i]
            state[0, i * 4 + 1:i * 4 + 4] = frame.transpose(2, 0, 1)
        
        return torch.FloatTensor(state).to(self.device)

    def run_mcts(self, state, root):
        """Run MCTS from the given state"""
        if not self.mcts_config:
            print("MCTS config not loaded")
            return
            
        for _ in range(self.mcts_config['num_simulations']):
            node = root
            search_path = [node]
            
            # Selection
            while node.expanded():
                action, node = self.select_child(node)
                search_path.append(node)
            
            # Expansion and evaluation
            parent = search_path[-2]
            action = list(parent.children.keys())[list(parent.children.values()).index(node)]
            
            # Get model prediction
            with torch.no_grad():
                model_output = self.model(state)
                if isinstance(model_output, dict):
                    policy = model_output.get('policy', model_output.get('policy_logit'))
                    value = model_output.get('value', torch.tensor([0.0]))
                    hidden_state = model_output.get('hidden_state')
                else:
                    policy = model_output
                    value = torch.tensor([0.0])
                    hidden_state = None
                
                # Ensure policy is 2D tensor [batch_size, num_actions]
                if len(policy.shape) == 1:
                    policy = policy.unsqueeze(0)
                
                # Convert policy to probabilities if needed
                if policy.shape[1] != ATARI_ACTION_SIZE:
                    policy = torch.softmax(policy, dim=1)
            
            # Expand node
            for action_idx in range(ATARI_ACTION_SIZE):
                prior = policy[0, action_idx].item()
                node.children[action_idx] = MCTSNode(prior)
            
            node.hidden_state = hidden_state
            
            # Backup
            self.backpropagate(search_path, value.item(), self.model)
            
            # Update state for next simulation
            if hidden_state is not None:
                state = hidden_state

    def select_child(self, node):
        """Select child node using PUCT formula"""
        ucb_scores = []
        for action, child in node.children.items():
            ucb_score = self.ucb_score(node, child)
            ucb_scores.append((action, child, ucb_score))
        
        # Select action with highest UCB score
        action, child, _ = max(ucb_scores, key=lambda x: x[2])
        return action, child

    def ucb_score(self, parent, child):
        """Calculate UCB score for a child node"""
        # PUCT formula from config
        prior_score = self.mcts_config['puct_init'] * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        value_score = child.value()
        return value_score + prior_score

    def backpropagate(self, search_path, value, model):
        """Backpropagate value through the search path"""
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = node.reward + self.mcts_config['reward_discount'] * value

    def select_action(self, root):
        """Select action using softmax over visit counts"""
        if not root.children:
            return 0  # Default to action 0 if no children
            
        visit_counts = np.array([child.visit_count for child in root.children.values()])
        if self.mcts_config['temperature'] == 0:
            action = np.argmax(visit_counts)
        else:
            # Apply temperature
            visit_counts = visit_counts ** (1.0 / self.mcts_config['temperature'])
            visit_counts = visit_counts / np.sum(visit_counts)
            action = np.random.choice(len(visit_counts), p=visit_counts)
        return action
        
    def play_game(self, b):
        """Play the game using the selected model"""
        try:
            # Clear previous output
            self.video_output.clear_output(wait=True)
            
            # Get selected model
            selected_model = self.model_dirs[self.model_dropdown.value]
            
            # Load model
            self.model = self.load_model(selected_model)
            if self.model is None:
                with self.video_output:
                    print("Failed to load model")
                return
                
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
            
            while not done:
                # Create state tensor
                state = self.create_state_tensor(frame_buffer, action_buffer)
                
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
                
            # Create video
            self.create_video()
            
            with self.video_output:
                print(f"Game finished! Total reward: {total_reward}")
                print(f"Random seed used: {random_seed}")
                
        except Exception as e:
            with self.video_output:
                print(f"Error playing game: {str(e)}")
                
    def create_video(self):
        """Create and display video from frames"""
        try:
            # Create temporary video file
            temp_video_path = "/content/gameplay.mp4"
            height, width, _ = self.frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, 30.0, (width, height))
            
            for frame in self.frames:
                out.write(frame)
            out.release()
            
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
                display(HTML(video_html))
                
        except Exception as e:
            with self.video_output:
                print(f"Error creating video: {str(e)}")

def main():
    player = ModelPlayer()
    player.setup_ui()

if __name__ == "__main__":
    main() 