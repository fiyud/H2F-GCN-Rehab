import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

def predict_and_visualize(model, test_loader, device, edge_index, subset_ratio=0.3, seed=100):
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
    
    np.random.seed(seed)
    
    subset_size = int(len(y_cts_true) * subset_ratio)
    indices = np.random.choice(len(y_cts_true), subset_size, replace=False)
    indices = sorted(indices)
    
    y_cts_true_subset = [y_cts_true[i] for i in indices]
    y_cts_pred_subset = [y_cts_pred[i] for i in indices]
    
    errors = [abs(y_true - y_pred) for y_true, y_pred in zip(y_cts_true_subset, y_cts_pred_subset)]
    
    visualize_predictions(y_cts_true_subset, y_cts_pred_subset, errors)
    
    return calculate_subset_metrics(y_cts_true_subset, y_cts_pred_subset)

def visualize_predictions(y_true, y_pred, errors=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    x = np.arange(len(y_true))
    ax1.plot(x, y_true, label="True cTS", color='b', marker='o', linestyle='dashed', alpha=0.7)
    ax1.plot(x, y_pred, label="Predicted cTS", color='r', marker='x', linestyle='dotted')
    ax1.set_title('Comparison of True vs Predicted cTS Values')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('cTS Value')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    if errors is not None:
        ax2.bar(x, errors, color='purple', alpha=0.6)
        ax2.set_title('Absolute Error')
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Error')
        ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def calculate_subset_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    abs_error = np.abs(y_true - y_pred)
    square_error = (y_true - y_pred) ** 2
    
    mad = np.mean(abs_error)
    rmse = np.sqrt(np.mean(square_error))

    mape = np.mean(abs_error / (y_true + 1e-8)) * 100
    
    return {
        'MAD': mad,
        'RMSE': rmse,
        'MAPE': mape
    }

def visualize_skeleton(position_data, frame_idx=0, sample_idx=0):
    skeleton = position_data[sample_idx, frame_idx].cpu().numpy()
    
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
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(skeleton[:, 0], skeleton[:, 1], skeleton[:, 2], c='b', marker='o', s=50)
    
    for connection in connections:
        ax.plot([skeleton[connection[0], 0], skeleton[connection[1], 0]],
                [skeleton[connection[0], 1], skeleton[connection[1], 1]],
                [skeleton[connection[0], 2], skeleton[connection[1], 2]], 'r-')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Skeleton Visualization (Sample {sample_idx}, Frame {frame_idx})')
    
    ax.set_box_aspect([1, 1, 1])
    
    plt.show()

def create_skeleton_animation(position_data, sample_idx=0, output_file=None):
    skeleton_seq = position_data[sample_idx].cpu().numpy()
    
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
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    joint_plot = ax.scatter([], [], [], c='b', marker='o', s=50)
    line_plots = [ax.plot([], [], [], 'r-')[0] for _ in connections]
    
    x_min, x_max = np.min(skeleton_seq[:, :, 0]), np.max(skeleton_seq[:, :, 0])
    y_min, y_max = np.min(skeleton_seq[:, :, 1]), np.max(skeleton_seq[:, :, 1])
    z_min, z_max = np.min(skeleton_seq[:, :, 2]), np.max(skeleton_seq[:, :, 2])
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Skeleton Animation (Sample {sample_idx})')
    
    ax.set_box_aspect([1, 1, 1])
    
    def update(frame):
        skeleton = skeleton_seq[frame]
        
        joint_plot._offsets3d = (skeleton[:, 0], skeleton[:, 1], skeleton[:, 2])
        
        for i, connection in enumerate(connections):
            line_plots[i].set_data([skeleton[connection[0], 0], skeleton[connection[1], 0]],
                                  [skeleton[connection[0], 1], skeleton[connection[1], 1]])
            line_plots[i].set_3d_properties([skeleton[connection[0], 2], skeleton[connection[1], 2]])
        
        ax.set_title(f'Skeleton Animation (Sample {sample_idx}, Frame {frame})')
        return [joint_plot] + line_plots
    
    frames = min(100, len(skeleton_seq))  # Limit to 100 frames for performance
    ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
    
    if output_file:
        ani.save(output_file, writer='pillow', fps=20)
    
    plt.tight_layout()
    plt.show()
    
    return ani