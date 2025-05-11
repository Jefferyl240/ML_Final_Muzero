import os
import sys
import subprocess
from IPython.display import display, HTML, Video, clear_output
import ipywidgets as widgets
from google.colab import files, output
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import base64

class ModelVisualizer:
    def __init__(self):
        self.model_dirs = {
            'Ms. Pac-Man (MuZero, n=50)': 'mz_50',
            'Ms. Pac-Man (MuZero, n=5)': 'mz_5',
            'Ms. Pac-Man (MuZero, n=1)': 'mz_1'
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
        self.video_output = widgets.Output()
        
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
        display(self.video_output)
        
    def run_visualization(self, b):
        """Handle visualization based on user selection"""
        selected_model = self.model_dirs[self.model_dropdown.value]
        viz_type = self.visualization_type.value
        selected_steps = self.steps_dropdown.value
        selected_iteration = self.steps_to_iterations[selected_steps]
        
        # Clear previous output
        self.video_output.clear_output(wait=True)
        
        if viz_type == 'Video':
            self.show_video(selected_model, selected_iteration)
        else:
            self.show_performance_graph(selected_model)
    
    def show_video(self, model_dir, iteration):
        """Display video of model gameplay for specific iteration (Colab-compatible)"""
        try:
            # Construct the video path
            video_path = Path(model_dir) / 'result_video' / f'ms_pacman-{iteration}itr.mp4'
            
            if not video_path.exists():
                with self.video_output:
                    self.video_output.clear_output(wait=True)
                    print(f"No video found at {video_path}")
                return

            # Move video to /content to ensure Colab can access it via relative path
            temp_video_path = f"/content/{video_path.name}"
            os.system(f"cp {video_path} {temp_video_path}")

            with self.video_output:
                                # Read the video file
                with open(video_path, 'rb') as f:
                    video_data = f.read()
                
                # Convert to base64
                video_b64 = base64.b64encode(video_data).decode()
                
                # Create HTML5 video element
                video_html = f'''
                <video width="640" height="480" controls>
                    <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                '''
                display(HTML(video_html))

        except Exception as e:
            with self.video_output:
                self.video_output.clear_output(wait=True)
                print(f"Error displaying video: {str(e)}")

    
    def show_performance_graph(self, model_dir):
        """Display performance graph of the model"""
        try:
            # Find the performance graph
            graph_path = Path(model_dir) / 'analysis' / 'Return.png'
            if not graph_path.exists():
                with self.video_output:
                    print(f"No performance graph found in {model_dir}")
                return
            
            with self.video_output:
                plt.figure(figsize=(10, 6))
                img = plt.imread(str(graph_path))
                plt.imshow(img)
                plt.axis('off')
                plt.title(f'Performance Graph - {self.model_dropdown.value}')
                plt.show()
            
        except Exception as e:
            with self.video_output:
                print(f"Error displaying performance graph: {str(e)}")

def main():
    visualizer = ModelVisualizer()
    visualizer.setup_ui()

if __name__ == "__main__":
    main() 