import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from google.colab import files, output
import base64
from pathlib import Path
import gym
import cv2
from collections import deque

class ModelPlayer:
    def __init__(self):
        self.model_dirs = {
            'Ms. Pac-Man (MuZero, n=50)': 'mz_50',
            'Ms. Pac-Man (MuZero, n=5)': 'mz_5',
            'Ms. Pac-Man (MuZero, n=1)': 'mz_1'
        }
        self.env = None
        self.model = None
        self.video_output = widgets.Output()
        self.frames = []
        
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
                model = torch.load(model_path)
            else:  # .pkl file
                import pickle
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            
            # Ensure model is in eval mode
            if hasattr(model, 'eval'):
                model.eval()
            
            return model
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
            
    def preprocess_frame(self, frame):
        """Preprocess frame for model input"""
        # Resize to 84x84 and convert to grayscale
        frame = cv2.resize(frame, (84, 84))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return frame
        
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
                
            # Create environment
            self.env = gym.make('ALE/MsPacman-v5')
            self.env = gym.wrappers.AtariPreprocessing(self.env, 
                                                     frame_skip=4,
                                                     grayscale_obs=True,
                                                     scale_obs=True)
            
            # Initialize frame buffer
            frame_buffer = deque(maxlen=4)
            obs = self.env.reset()
            frame_buffer.append(obs)
            
            # Play game
            done = False
            total_reward = 0
            self.frames = []
            
            while not done:
                # Get current state
                state = np.stack(frame_buffer)
                
                # Get model prediction
                with torch.no_grad():
                    action = self.model(state)
                
                # Take action
                obs, reward, done, info = self.env.step(action)
                frame_buffer.append(obs)
                total_reward += reward
                
                # Store frame for video
                self.frames.append(self.env.render(mode='rgb_array'))
                
            # Create video
            self.create_video()
            
            with self.video_output:
                print(f"Game finished! Total reward: {total_reward}")
                
        except Exception as e:
            with self.video_output:
                print(f"Error playing game: {str(e)}")
        finally:
            if self.env is not None:
                self.env.close()
                
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