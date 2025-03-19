#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from data.data_loader import load_kimore_data, preprocess_merged_data
from data.dataset import CustomDataset
from data.preprocessing import preprocess_data_and_labels, get_JCD
from models.H2F_GCN import ThreeStreamGCN_ModelvB
from models.four_stream_gcn import FourStreamGCN_Model
from utils.seed import set_seed
from utils.metrics import compute_metrics
from utils.visualization import (
    predict_and_visualize, 
    visualize_skeleton, 
    create_skeleton_animation, 
    calculate_subset_metrics
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate pre-trained Kimore GCN Model')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='Kimore', help='Path to the Kimore dataset')
    parser.add_argument('--exercise', type=int, default=5, choices=[1, 2, 3, 4, 5], 
                        help='Exercise number to use (1-5)')
    
    parser.add_argument('--model', type=str, default='three_stream', 
                        choices=['three_stream', 'four_stream'], 
                        help='Model architecture to use')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GRU layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.15, help='Dropout rate')
    
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--chunk_size', type=int, default=50, help='Chunk size for sequences')
    parser.add_argument('--seed', type=int, default=100, help='Random seed for reproducibility')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set ratio')
    parser.add_argument('--model_path', type=str, default='best_model.pth', 
                        help='Path to the pre-trained model weights')
    
    parser.add_argument('--output_folder', type=str, default='eval_output',
                        help='Folder to save visualization outputs')
    parser.add_argument('--vis_ratio', type=float, default=0.5, 
                        help='Ratio of test samples to visualize')
    parser.add_argument('--detailed_metrics', action='store_true', default=True,
                        help='Generate detailed performance metrics')
    parser.add_argument('--confusion_matrix', action='store_true', default=True,
                        help='Generate confusion matrix for predictions')
    
    parser.add_argument('--device', type=str, default='', 
                        help='Device to use (leave empty for auto-detection)')
    
    args = parser.parse_args()
    
    if not args.device:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)
    
    return args

def load_test_data(args):
    print(f"Loading Kimore dataset from {args.data_path}...")
    data = load_kimore_data(args.data_path)
    
    import pandas as pd
    df = pd.DataFrame(data)
    
    cols_to_convert = [col for col in df.columns if '-p' in col or '-o' in col]
    df[cols_to_convert] = df[cols_to_convert].applymap(lambda x: np.array(x) if isinstance(x, list) else x)
    
    if 2 in df.index and 'Group' in df.columns:
        df.at[2, 'Group'] = "E"
        df.at[3, 'Group'] = "E"
        df.at[4, 'Group'] = "E"
    

    df = df.dropna().reset_index(drop=True)    
    df_ex = df[df['Exercise'] == args.exercise].reset_index(drop=True)
    print(f"Using exercise {args.exercise} data: {len(df_ex)} samples")
    
    df_merged = preprocess_merged_data(df_ex)
    data_tensor, labels_tensor = preprocess_data_and_labels(df_merged, args.chunk_size)
    position_data = data_tensor[:, :, :, 4:] 
    
    JCD = get_JCD(position_data)
    
    train_data, test_data, train_jcd, test_jcd, train_labels, test_labels = train_test_split(
        data_tensor, JCD, labels_tensor, test_size=args.test_size, random_state=args.seed
    )
    
    test_dataset = CustomDataset(test_data, test_jcd, test_labels)
    
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        generator=g,
        worker_init_fn=lambda worker_id: np.random.seed(args.seed + worker_id)
    )
    
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), 
        (4, 5), (5, 6), (6, 7), 
        (8, 9), (9, 10), (10, 11), 
        (0, 12), (12, 13), (13, 14), (14, 15), 
        (0, 16), (16, 17), (17, 18), (18, 19), 
        (2, 20), (7, 21), (6, 22), 
        (11, 23), (10, 24)
    ]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    return test_loader, edge_index, test_data.shape, JCD.shape

def load_model(args, input_shape, jcd_shape):
    num_joints = input_shape[2]  # Shape: [batch, time, joints, features]
    num_features = input_shape[3]
    output_dim = 1  # For cTS prediction
    feat_d = jcd_shape[-1]  # JCD feature dimension
    
    print(f"Creating {args.model} model...")
    if args.model == 'three_stream':
        model = ThreeStreamGCN_ModelvB(
            num_joints=num_joints,
            num_features=num_features,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            output_dim=output_dim,
            feat_d=feat_d,
            nhead=args.num_heads,
            dropout=args.dropout
        )
    else:  # four_stream
        model = FourStreamGCN_Model(
            num_joints=num_joints,
            num_features=num_features,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            output_dim=output_dim,
            feat_d=feat_d,
            nhead=args.num_heads,
            dropout=args.dropout
        )
    
    print(f"Loading model weights from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.to(args.device)
    model.eval()
    
    return model

def evaluate_model(model, test_loader, edge_index, device):
    model.eval()
    test_mad, test_mape, test_rmse = 0.0, 0.0, 0.0
    all_true_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, (data, jcd, labels) in enumerate(test_loader):
            data = data.to(device)
            jcd = jcd.to(device)
            labels = labels.to(device)

            outputs = model(data, edge_index.to(device), jcd)
            
            # Calculate metrics
            mad, mape, rmse = compute_metrics(labels, outputs)
            test_mad += mad
            test_mape += mape
            test_rmse += rmse
            
            # Store labels and predictions
            all_true_labels.extend(labels.cpu().numpy() * 50)  # Scale back to original range
            all_predictions.extend(outputs.cpu().numpy() * 50)
    
    # Average metrics across batches
    avg_test_mad = test_mad / len(test_loader)
    avg_test_mape = test_mape / len(test_loader)
    avg_test_rmse = test_rmse / len(test_loader)
    
    print("\n" + "=" * 40)
    print("EVALUATION RESULTS:")
    print(f"- MAD: {avg_test_mad:.4f}")
    print(f"- RMSE: {avg_test_rmse:.4f}")
    print(f"- MAPE: {avg_test_mape:.2f}%")
    print("=" * 40)
    
    return all_true_labels, all_predictions, {
        'MAD': avg_test_mad,
        'RMSE': avg_test_rmse,
        'MAPE': avg_test_mape
    }

def generate_error_histogram(true_labels, predictions, save_path):
    errors = np.array(predictions) - np.array(true_labels)
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=20, alpha=0.7, color='blue')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=1)
    plt.title('Histogram of Prediction Errors')
    plt.xlabel('Error (Predicted - True)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    plt.text(0.98, 0.95, f'Mean Error: {mean_error:.2f}\nStd Dev: {std_error:.2f}',
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Error histogram saved to {save_path}")
    plt.close()

def generate_scatter_plot(true_labels, predictions, save_path):
    true_labels = np.squeeze(true_labels)
    predictions = np.squeeze(predictions)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(true_labels, predictions, alpha=0.5)
    
    min_val = min(np.min(true_labels), np.min(predictions))
    max_val = max(np.max(true_labels), np.max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    z = np.polyfit(true_labels, predictions, 1)
    p = np.poly1d(z)
    plt.plot(true_labels, p(true_labels), "g-", alpha=0.5)
    
    plt.title('True vs Predicted Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    
    from scipy.stats import pearsonr
    r_squared = pearsonr(true_labels, predictions)[0]**2
    plt.text(0.02, 0.95, f'R² = {r_squared:.4f}',
             horizontalalignment='left',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Scatter plot saved to {save_path}")
    plt.close()

def main():
    args = parse_args()
    
    set_seed(args.seed)
    
    os.makedirs(args.output_folder, exist_ok=True)
    print(f"Evaluation outputs will be saved to {args.output_folder}")
    
    # Load test data
    test_loader, edge_index, input_shape, jcd_shape = load_test_data(args)
    
    model = load_model(args, input_shape, jcd_shape)
    true_labels, predictions, metrics = evaluate_model(model, test_loader, edge_index, args.device)
    
    print("Generating visualizations...")
    
    with open(os.path.join(args.output_folder, f"metrics_ex{args.exercise}.txt"), 'w') as f:
        f.write("EVALUATION METRICS\n")
        f.write("=================\n")
        f.write(f"Exercise: {args.exercise}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Model path: {args.model_path}\n\n")
        f.write(f"MAD: {metrics['MAD']:.4f}\n")
        f.write(f"RMSE: {metrics['RMSE']:.4f}\n")
        f.write(f"MAPE: {metrics['MAPE']:.2f}%\n")
    
    if args.detailed_metrics:
        generate_error_histogram(
            true_labels, 
            predictions, 
            os.path.join(args.output_folder, f"error_histogram_ex{args.exercise}.png")
        )
        
        generate_scatter_plot(
            true_labels, 
            predictions, 
            os.path.join(args.output_folder, f"scatter_plot_ex{args.exercise}.png")
        )
    
    print("Visualizing model predictions...")
    predict_and_visualize(
        model, 
        test_loader, 
        args.device, 
        edge_index, 
        args.vis_ratio, 
        args.seed,
        save_path=os.path.join(args.output_folder, f"predictions_ex{args.exercise}.png")
    )
    
    print("Visualizing skeleton data...")
    for batch_idx, (data, jcd, labels) in enumerate(test_loader):
        position_data = data[:, :, :, 4:7].to('cpu')  # Assuming positions are in indices 4-6
        
        for sample_idx in range(min(3, len(data))):  # Visualize up to 3 samples
            for frame_idx in [0, int(data.shape[1]/2), data.shape[1]-1]:  # First, middle, and last frames
                vis_path = os.path.join(args.output_folder, f"skeleton_ex{args.exercise}_sample{sample_idx}_frame{frame_idx}.png")
                visualize_skeleton(
                    position_data, 
                    frame_idx=frame_idx, 
                    sample_idx=sample_idx,
                    save_path=vis_path
                )
        
        print("Creating skeleton animations...")
        for sample_idx in range(min(2, len(data))):  # Create animations for up to 2 samples
            anim_path = os.path.join(args.output_folder, f"animation_ex{args.exercise}_sample{sample_idx}.gif")
            create_skeleton_animation(
                position_data,
                sample_idx=sample_idx,
                output_file=anim_path
            )
        
        break
    
    print(f"\nEvaluation complete. Results saved to {args.output_folder}")

if __name__ == "__main__":
    main()