import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

def predict_and_visualize(model, test_loader, device, edge_index, subset_ratio=0.3, seed=100, save_path=None):
    """
    Make predictions on test data and visualize the results.
    
    Args:
        model (nn.Module): Trained model
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to run the model on
        edge_index (torch.Tensor): Edge index for graph connections
        subset_ratio (float): Ratio of test samples to visualize
        seed (int): Random seed for reproducibility
    """
    model.eval()
    
    y_cts_true, y_cts_pred = [], []
    
    with torch.no_grad():
        for batch_idx, (data, jcd, labels) in enumerate(test_loader):
            data = data.to(device)
            jcd = jcd.to(device)
            labels = labels.to(device)

            outputs = model(data, edge_index.to(device), jcd)
            out_cts = outputs[:, 0]  # For single target (cTS)
            
            y_cts_true.extend((labels[:, 0].cpu().numpy() * 50))
            y_cts_pred.extend((out_cts.cpu().numpy() * 50))
    
    # Set seed for reproducible subset selection
    np.random.seed(seed)
    
    # Select a random subset of data for display
    subset_size = int(len(y_cts_true) * subset_ratio)
    indices = np.random.choice(len(y_cts_true), subset_size, replace=False)
    indices = sorted(indices)  # Sort indices for clearer visualization
    
    y_cts_true_subset = [y_cts_true[i] for i in indices]
    y_cts_pred_subset = [y_cts_pred[i] for i in indices]
    
    # Compute errors
    errors = [abs(y_true - y_pred) for y_true, y_pred in zip(y_cts_true_subset, y_cts_pred_subset)]
    
    # Visualize predictions
    visualize_predictions(y_cts_true_subset, y_cts_pred_subset, errors, save_path=save_path)
    
    # Return metrics for the subset
    return calculate_subset_metrics(y_cts_true_subset, y_cts_pred_subset)

def visualize_predictions(y_true, y_pred, errors=None, save_path=None):
    """
    Visualize comparison between true and predicted values.
    
    Args:
        y_true (list): Ground truth values
        y_pred (list): Predicted values
        errors (list, optional): Absolute errors between true and predicted values
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot predictions vs ground truth
    x = np.arange(len(y_true))
    ax1.plot(x, y_true, label="True cTS", color='b', marker='o', linestyle='dashed', alpha=0.7)
    ax1.plot(x, y_pred, label="Predicted cTS", color='r', marker='x', linestyle='dotted')
    ax1.set_title('Comparison of True vs Predicted cTS Values')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('cTS Value')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot errors if provided
    if errors is not None:
        ax2.bar(x, errors, color='purple', alpha=0.6)
        ax2.set_title('Absolute Error')
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Error')
        ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction visualization saved to {save_path}")
    
    plt.show()
    plt.close()

def calculate_subset_metrics(y_true, y_pred):
    """
    Calculate metrics for a subset of data.
    
    Args:
        y_true (list): Ground truth values
        y_pred (list): Predicted values
        
    Returns:
        dict: Dictionary containing metrics (MAD, RMSE, MAPE)
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate absolute error
    abs_error = np.abs(y_true - y_pred)
    
    # Calculate squared error
    square_error = (y_true - y_pred) ** 2
    
    # Mean Absolute Deviation (MAD)
    mad = np.mean(abs_error)
    
    # Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean(square_error))
    
    # Mean Absolute Percentage Error (MAPE)
    # Adding small epsilon to avoid division by zero
    mape = np.mean(abs_error / (y_true + 1e-8)) * 100
    
    return {
        'MAD': mad,
        'RMSE': rmse,
        'MAPE': mape
    }

def visualize_skeleton(position_data, frame_idx=0, sample_idx=0, save_path=None):
    """
    Visualize skeleton data for a specific frame and sample.
    
    Args:
        position_data (torch.Tensor): Position data with shape [batch, frame, joints, 3]
        frame_idx (int): Frame index to visualize
        sample_idx (int): Sample index to visualize
    """
    # Extract skeleton data for the specified sample and frame
    skeleton = position_data[sample_idx, frame_idx].cpu().numpy()
    
    # Kinect v2 skeleton connections
    connections = [
        (0, 1), (1, 2), (2, 3),  # Spine to Head
        (2, 4), (4, 5), (5, 6), (6, 7),  # Left Arm
        (2, 8), (8, 9), (9, 10), (10, 11),  # Right Arm
        (0, 12), (12, 13), (13, 14), (14, 15),  # Left Leg
        (0, 16), (16, 17), (17, 18), (18, 19),  # Right Leg
        (20, 4), (20, 8),  # Shoulders
        (7, 21), (6, 22),  # Left Hand Details
        (11, 23), (10, 24)  # Right Hand Details
    ]
    
    # Set up 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot joints
    ax.scatter(skeleton[:, 0], skeleton[:, 1], skeleton[:, 2], c='b', marker='o', s=50)
    
    # Plot connections
    for connection in connections:
        ax.plot([skeleton[connection[0], 0], skeleton[connection[1], 0]],
                [skeleton[connection[0], 1], skeleton[connection[1], 1]],
                [skeleton[connection[0], 2], skeleton[connection[1], 2]], 'r-')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Skeleton Visualization (Sample {sample_idx}, Frame {frame_idx})')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Skeleton visualization saved to {save_path}")
    
    plt.show()
    plt.close()

def create_skeleton_animation(position_data, sample_idx=0, output_file=None):
    """
    Create an animation of skeleton movement.
    
    Args:
        position_data (torch.Tensor): Position data with shape [batch, frame, joints, 3]
        sample_idx (int): Sample index to animate
        output_file (str, optional): Output file path for saving the animation
    """
    # Extract skeleton data for the specified sample
    skeleton_seq = position_data[sample_idx].cpu().numpy()
    
    # Kinect v2 skeleton connections
    connections = [
        (0, 1), (1, 2), (2, 3),  # Spine to Head
        (2, 4), (4, 5), (5, 6), (6, 7),  # Left Arm
        (2, 8), (8, 9), (9, 10), (10, 11),  # Right Arm
        (0, 12), (12, 13), (13, 14), (14, 15),  # Left Leg
        (0, 16), (16, 17), (17, 18), (18, 19),  # Right Leg
        (20, 4), (20, 8),  # Shoulders
        (7, 21), (6, 22),  # Left Hand Details
        (11, 23), (10, 24)  # Right Hand Details
    ]
    
    # Set up 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize plots
    joint_plot = ax.scatter([], [], [], c='b', marker='o', s=50)
    line_plots = [ax.plot([], [], [], 'r-')[0] for _ in connections]
    
    # Find global min and max for consistent scaling
    x_min, x_max = np.min(skeleton_seq[:, :, 0]), np.max(skeleton_seq[:, :, 0])
    y_min, y_max = np.min(skeleton_seq[:, :, 1]), np.max(skeleton_seq[:, :, 1])
    z_min, z_max = np.min(skeleton_seq[:, :, 2]), np.max(skeleton_seq[:, :, 2])
    
    # Set limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Skeleton Animation (Sample {sample_idx})')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Animation update function
    def update(frame):
        skeleton = skeleton_seq[frame]
        
        # Update joint positions
        joint_plot._offsets3d = (skeleton[:, 0], skeleton[:, 1], skeleton[:, 2])
        
        # Update connection lines
        for i, connection in enumerate(connections):
            line_plots[i].set_data([skeleton[connection[0], 0], skeleton[connection[1], 0]],
                                  [skeleton[connection[0], 1], skeleton[connection[1], 1]])
            line_plots[i].set_3d_properties([skeleton[connection[0], 2], skeleton[connection[1], 2]])
        
        ax.set_title(f'Skeleton Animation (Sample {sample_idx}, Frame {frame})')
        return [joint_plot] + line_plots
    
    # Create animation
    frames = min(100, len(skeleton_seq))  # Limit to 100 frames for performance
    ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
    
    # Save animation if output_file is provided
    if output_file:
        ani.save(output_file, writer='pillow', fps=20)
    
    plt.tight_layout()
    plt.show()
    
    return ani