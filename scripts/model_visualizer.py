import os
import sys
import subprocess
from IPython.display import display, HTML, Video
import ipywidgets as widgets
from google.colab import files
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class ModelVisualizer:
    def __init__(self):
        self.model_dirs = {
            'Ms. Pac-Man (MuZero, n=50)': 'atari_ms_pacman_mz_2bx64_n50-78660b-dirty',
            'Ms. Pac-Man (MuZero, n=5)': 'atari_ms_pacman_mz_2bx64_n5-105390',
            'Ms. Pac-Man (MuZero, n=1)': 'atari_ms_pacman_mz_2bx64_n1-78660b-dirty'
        }
        # Create a mapping of step numbers to iteration numbers
        self.steps_to_iterations = {
            200: 1,
            10000: 50,
            20000: 100,
            30000: 150,
            40000: 200,
            50000: 250,
            60000: 300
        }
        
    def setup_ui(self):
        """Create the user interface widgets"""
        self.model_dropdown = widgets.Dropdown(
            options=list(self.model_dirs.keys()),
            description='Select Model:',
            style={'description_width': 'initial'}
        )
        
        self.visualization_type = widgets.RadioButtons(
            options=['Video', 'Performance Graph'],
            description='Visualization Type:',
            style={'description_width': 'initial'}
        )
        
        self.steps_dropdown = widgets.Dropdown(
            options=list(self.steps_to_iterations.keys()),
            description='Select Steps:',
            style={'description_width': 'initial'}
        )
        
        self.run_button = widgets.Button(
            description='Run Visualization',
            button_style='success'
        )
        
        self.run_button.on_click(self.run_visualization)
        
        # Display widgets
        display(HTML("<h2>MiniZero Model Visualizer</h2>"))
        display(self.model_dropdown)
        display(self.visualization_type)
        display(self.steps_dropdown)
        display(self.run_button)
        
    def run_visualization(self, b):
        """Handle visualization based on user selection"""
        selected_model = self.model_dirs[self.model_dropdown.value]
        viz_type = self.visualization_type.value
        selected_steps = self.steps_dropdown.value
        selected_iteration = self.steps_to_iterations[selected_steps]
        
        if viz_type == 'Video':
            self.show_video(selected_model, selected_iteration)
        else:
            self.show_performance_graph(selected_model)
    
    def show_video(self, model_dir, iteration):
        """Display video of model gameplay for specific iteration"""
        try:
            # Construct the video path using the correct structure
            video_path = model_dir / 'result_video' / f'ms_pacman-{iteration}itr.mp4'
            
            if not video_path.exists():
                print(f"No video found at {video_path}")
                return
                
            display(Video(str(video_path)))
            
        except Exception as e:
            print(f"Error displaying video: {str(e)}")
    
    def show_performance_graph(self, model_dir):
        """Display performance graph of the model"""
        try:
            # Find the performance graph
            graph_path = Path('minizero') / model_dir / 'analysis' / 'Return.png'
            if not graph_path.exists():
                print(f"No performance graph found in {model_dir}")
                return
                
            plt.figure(figsize=(10, 6))
            img = plt.imread(str(graph_path))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'Performance Graph - {self.model_dropdown.value}')
            plt.show()
            
        except Exception as e:
            print(f"Error displaying performance graph: {str(e)}")

def main():
    visualizer = ModelVisualizer()
    visualizer.setup_ui()

if __name__ == "__main__":
    main() 