import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle, Arrow, Circle
import matplotlib.colors as mcolors
import argparse
import math
from typing import List, Dict, Tuple, Optional

class VRPSolutionVisualizer:
    """
    Advanced visualizer for VRP solutions that creates fancy, publication-quality images
    showing customer positions, demands, multiple depots, routes with arrows,
    time windows, and comprehensive legends.
    """
    
    def __init__(self, results_dir: str, output_dir: str, dpi: int = 300):
        """Initialize the visualizer with path to results directory and output settings"""
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.dpi = dpi
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Color settings
        self.palette = plt.cm.tab10
        self.depot_color = '#FF5733'  # Bright orange for depots
        self.customer_color = '#3498DB'  # Blue for customers
        self.tw_color = 'green'  # Green for time windows
        self.appear_color = 'purple'  # Purple for appear times
        
        # Style settings
        plt.style.use('seaborn-v0_8-whitegrid')
        self.node_size_base = 100  # Base size for nodes
        self.node_size_factor = 5  # Factor to scale nodes based on demand
        self.linewidth_base = 1.5  # Base width for route lines
        
        # Load all result files
        self.results = self._load_results()
    
    def _load_results(self) -> Dict:
        """Load all result files from the directory"""
        results = {}
        result_files = glob.glob(os.path.join(self.results_dir, "*.json"))
        
        print(f"Found {len(result_files)} result files")
        if not result_files:
            print(f"No result files found in {self.results_dir}")
            return results
        
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    result = json.load(f)
                    
                instance_id = result.get('instance', os.path.basename(file_path).replace('.json', ''))
                results[instance_id] = result
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return results
    
    def visualize_all_solutions(self, max_instances: int = None):
        """
        Visualize all solutions in the results directory
        
        Args:
            max_instances: Maximum number of instances to visualize (None for all)
        """
        instances = list(self.results.keys())
        if max_instances is not None:
            instances = instances[:max_instances]
        
        for instance_id in instances:
            try:
                result = self.results[instance_id]
                problem_name = result.get('problem', f'Instance {instance_id}')
                solver_name = result.get('solver', 'Unknown')
                
                output_file = os.path.join(
                    self.output_dir, 
                    f"solution_{problem_name}_{solver_name}_{instance_id}.png"
                )
                
                self.visualize_solution(result, output_file)
                print(f"Visualized solution for instance {instance_id}: {output_file}")
            
            except Exception as e:
                print(f"Error visualizing instance {instance_id}: {e}")
    
    def visualize_solution(self, result: Dict, output_file: str):
        """
        Visualize a single VRP solution with all requested elements
        
        Args:
            result: The solution result dictionary
            output_file: Path to save the output image
        """
        # Extract solution data
        routes = result.get('routes', [])
        locations = result.get('locations', [])
        depots = result.get('depots', [0])
        customers = result.get('customers', [i for i in range(len(locations)) if i not in depots])
        
        # Extract demands if available
        demands = result.get('demands', [1] * len(locations))
        if not isinstance(demands, list):
            demands = demands.tolist() if hasattr(demands, 'tolist') else [1] * len(locations)
        
        # Extract time windows if available
        time_windows = result.get('time_windows', {})
        appear_times = result.get('appear_times', {})
        
        # Normalize appear times and time windows to same format
        tw_dict = self._normalize_time_data(time_windows, len(locations))
        appear_dict = self._normalize_time_data(appear_times, len(locations))
        
        # Compute bounding box and other settings
        x_coords = [loc[0] for loc in locations]
        y_coords = [loc[1] for loc in locations]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add some padding to the bounding box
        padding = max(x_max - x_min, y_max - y_min) * 0.15
        x_min, x_max = x_min - padding, x_max + padding
        y_min, y_max = y_min - padding, y_max + padding
        
        # Create figure with a single axis (no separate legend)
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Get problem info for the title
        problem_name = result.get('problem', 'VRP Solution')
        solver_name = result.get('solver', 'Unknown')
        
        # Set title and axis limits
        ax.set_title(f"Solution for {problem_name}\nSolver: {solver_name}", fontsize=14)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_xlabel("X Coordinate", fontsize=12)
        ax.set_ylabel("Y Coordinate", fontsize=12)
        
        # Draw the routes with arrows and different colors
        route_lengths = []
        for i, route in enumerate(routes):
            if len(route) <= 1:
                continue
                
            color = self.palette(i % 10)
            linewidth = self.linewidth_base
            
            # Calculate route length
            route_length = 0
            for j in range(len(route) - 1):
                if route[j] < len(locations) and route[j+1] < len(locations):
                    loc1 = locations[route[j]]
                    loc2 = locations[route[j+1]]
                    route_length += math.sqrt((loc2[0] - loc1[0])**2 + (loc2[1] - loc1[1])**2)
            
            route_lengths.append(route_length)
            
            # Draw route segments with arrows
            for j in range(len(route) - 1):
                if route[j] < len(locations) and route[j+1] < len(locations):
                    self._draw_route_segment(
                        ax, 
                        locations[route[j]], 
                        locations[route[j+1]], 
                        color, 
                        linewidth
                    )
        
        # Draw customer nodes with demands
        max_demand = max(demands) if demands else 1
        min_size = self.node_size_base
        size_range = self.node_size_base * self.node_size_factor
        
        for i in customers:
            if i < len(locations):
                demand = demands[i] if i < len(demands) else 1
                # Scale node size by demand
                size = min_size + (demand / max_demand) * size_range
                
                # Draw customer node
                ax.scatter(
                    locations[i][0], 
                    locations[i][1], 
                    s=size, 
                    c=self.customer_color,
                    alpha=0.7, 
                    edgecolor='black', 
                    linewidth=1, 
                    zorder=10
                )
                
                # Add node ID and demand as text
                ax.text(
                    locations[i][0], 
                    locations[i][1], 
                    f"{i}\n({demand})", 
                    ha='center', 
                    va='center', 
                    fontsize=8, 
                    fontweight='bold', 
                    color='black', 
                    zorder=11,
                    bbox=dict(facecolor='white', alpha=0.5, pad=1, edgecolor='none')
                )
                
                # Draw time window if available
                if i in tw_dict:
                    self._draw_time_window(
                        ax, 
                        locations[i], 
                        tw_dict[i],
                        self.tw_color,
                        size
                    )
                
                # Draw appear time if available
                if i in appear_dict:
                    self._draw_appear_time(
                        ax, 
                        locations[i], 
                        appear_dict[i],
                        self.appear_color, 
                        size
                    )
        
        # Draw depot nodes
        for i in depots:
            if i < len(locations):
                # Draw depot with diamond shape
                ax.scatter(
                    locations[i][0], 
                    locations[i][1], 
                    s=self.node_size_base * 2, 
                    c=self.depot_color, 
                    marker='D', 
                    edgecolor='black', 
                    linewidth=1.5, 
                    zorder=20
                )
                
                # Add depot ID as text
                ax.text(
                    locations[i][0], 
                    locations[i][1], 
                    f"D{i}", 
                    ha='center', 
                    va='center', 
                    fontsize=10, 
                    fontweight='bold', 
                    color='white', 
                    zorder=21
                )
        
        # Add a small legend directly on the main plot (top-right corner)
        self._add_mini_legend(ax)
        
        # Save figure without using tight_layout()
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _normalize_time_data(self, time_data, num_locations: int) -> Dict[int, Tuple]:
        """Convert time window or appear time data to a standardized format"""
        result = {}
        
        if isinstance(time_data, dict):
            # Already in dictionary format
            for node_id, time_value in time_data.items():
                node_id = int(node_id)
                if isinstance(time_value, (list, tuple)) and len(time_value) == 2:
                    # Time window format: (start, end)
                    result[node_id] = tuple(time_value)
                else:
                    # Appear time format: single value
                    result[node_id] = time_value
        
        elif isinstance(time_data, list):
            # List format
            for i, time_value in enumerate(time_data):
                if time_value:  # Skip zero or empty values
                    if isinstance(time_value, (list, tuple)) and len(time_value) == 2:
                        # Time window format: (start, end)
                        result[i] = tuple(time_value)
                    else:
                        # Appear time format: single value
                        result[i] = time_value
        
        return result
    
    def _draw_route_segment(self, ax, start_loc, end_loc, color, linewidth):
        """Draw a route segment with an arrow indicating direction"""
        # Draw the line segment
        ax.plot(
            [start_loc[0], end_loc[0]], 
            [start_loc[1], end_loc[1]], 
            color=color, 
            linewidth=linewidth, 
            alpha=0.7, 
            zorder=5
        )
        
        # Calculate the midpoint for the arrow
        mid_x = (start_loc[0] + end_loc[0]) / 2
        mid_y = (start_loc[1] + end_loc[1]) / 2
        
        # Calculate the direction vector
        dx = end_loc[0] - start_loc[0]
        dy = end_loc[1] - start_loc[1]
        
        # Normalize the vector
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            dx = dx / length
            dy = dy / length
        
        # Arrow size - scale with figure
        arrow_scale = 0.05 * length
        
        # Draw an arrow at the midpoint
        arrow = ax.arrow(
            mid_x - dx * arrow_scale/2,  # x position
            mid_y - dy * arrow_scale/2,  # y position
            dx * arrow_scale,  # dx
            dy * arrow_scale,  # dy
            head_width=arrow_scale*0.8,  # head width
            head_length=arrow_scale,  # head length
            fc=color,  # face color
            ec=color,  # edge color
            linewidth=linewidth,
            zorder=6
        )
    
    def _draw_time_window(self, ax, location, time_window, color, node_size):
        """Draw a time window indicator for a customer"""
        if isinstance(time_window, (list, tuple)) and len(time_window) == 2:
            start_time, end_time = time_window
            
            # Normalize the time to be in range [0, 24]
            start_hour = min(24, max(0, start_time / 60))
            end_hour = min(24, max(0, end_time / 60))
            
            # Draw a small clock-like visualizer for time window
            radius = math.sqrt(node_size) / 3
            center_x = location[0] + radius * 2.2
            center_y = location[1] + radius * 2.2
            
            # Draw the clock face
            circle = Circle(
                (center_x, center_y), 
                radius, 
                color='white', 
                edgecolor='black', 
                linewidth=1, 
                alpha=0.8, 
                zorder=15
            )
            ax.add_patch(circle)
            
            # Convert hours to angles in radians (0h = top, clockwise)
            start_angle = (start_hour / 24) * 2 * math.pi - math.pi/2
            end_angle = (end_hour / 24) * 2 * math.pi - math.pi/2
            
            # Draw the time window arc
            arc_points_x = [center_x]
            arc_points_y = [center_y]
            
            # Generate points along the arc
            for angle in np.linspace(start_angle, end_angle, 20):
                arc_points_x.append(center_x + radius * math.cos(angle))
                arc_points_y.append(center_y + radius * math.sin(angle))
            
            # Close the arc to the center
            arc_points_x.append(center_x)
            arc_points_y.append(center_y)
            
            # Draw the time window arc
            ax.fill(
                arc_points_x, 
                arc_points_y, 
                color=color, 
                alpha=0.6, 
                zorder=16
            )
            
            # Add start and end times as small text
            text_distance = radius * 1.5
            start_text_x = center_x + text_distance * math.cos(start_angle)
            start_text_y = center_y + text_distance * math.sin(start_angle)
            end_text_x = center_x + text_distance * math.cos(end_angle)
            end_text_y = center_y + text_distance * math.sin(end_angle)
            
            ax.text(
                start_text_x, 
                start_text_y, 
                f"{int(start_hour)}h", 
                fontsize=6, 
                ha='center', 
                va='center', 
                color='black', 
                zorder=17
            )
            
            ax.text(
                end_text_x, 
                end_text_y, 
                f"{int(end_hour)}h", 
                fontsize=6, 
                ha='center', 
                va='center', 
                color='black', 
                zorder=17
            )
    
    def _draw_appear_time(self, ax, location, appear_time, color, node_size):
        """Draw an appear time indicator for a customer"""
        # Convert to hours
        appear_hour = min(24, max(0, appear_time / 60)) if appear_time is not None else 0
        
        # Draw a small clock-like visualizer for appear time
        radius = math.sqrt(node_size) / 3
        center_x = location[0] - radius * 2.2
        center_y = location[1] - radius * 2.2
        
        # Draw the clock face
        circle = Circle(
            (center_x, center_y), 
            radius, 
            color='white', 
            edgecolor='black', 
            linewidth=1, 
            alpha=0.8, 
            zorder=15
        )
        ax.add_patch(circle)
        
        # Convert hour to angle in radians (0h = top, clockwise)
        appear_angle = (appear_hour / 24) * 2 * math.pi - math.pi/2
        
        # Draw the appear time hand
        ax.plot(
            [center_x, center_x + radius * math.cos(appear_angle)],
            [center_y, center_y + radius * math.sin(appear_angle)],
            color=color,
            linewidth=2,
            zorder=16
        )
        
        # Add appear time as small text
        text_distance = radius * 1.5
        text_x = center_x + text_distance * math.cos(appear_angle)
        text_y = center_y + text_distance * math.sin(appear_angle)
        
        ax.text(
            text_x,
            text_y,
            f"{int(appear_hour)}h",
            fontsize=6,
            ha='center',
            va='center',
            color='black',
            zorder=17
        )
    
    def _add_mini_legend(self, ax):
        """Add a mini legend to the top right corner of the main plot"""
        # Create legend items
        legend_items = [
            (self.depot_color, 'D', "Depot"),
            (self.customer_color, 'o', "Customer (ID/Demand)")
        ]
        
        # Create legend handles and labels
        handles = []
        labels = []
        
        for color, marker, label in legend_items:
            handle = ax.scatter([], [], color=color, marker=marker, s=100, 
                               edgecolor='black', linewidth=1)
            handles.append(handle)
            labels.append(label)
        
        # Create legend with background
        legend = ax.legend(handles, labels, loc='upper right', 
                          framealpha=0.9, fontsize=10)
        
        # Set the legend to have a white background with border
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('gray')
        

def main():
    """Main function for the VRP solution visualizer"""
    parser = argparse.ArgumentParser(description='Generate fancy visualizations of VRP solutions')
    parser.add_argument('--results_dir', default='rl_results', help='Directory containing result JSON files')
    parser.add_argument('--output_dir', default='vrp_visualizations', help='Directory to store visualization images')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for output images')
    parser.add_argument('--max_instances', type=int, default=None, help='Maximum number of instances to visualize')
    
    args = parser.parse_args()
    
    # Create visualizer and generate images
    visualizer = VRPSolutionVisualizer(args.results_dir, args.output_dir, args.dpi)
    visualizer.visualize_all_solutions(args.max_instances)
    
    print(f"\nVisualization complete. Images saved to {args.output_dir}")


if __name__ == "__main__":
    main()