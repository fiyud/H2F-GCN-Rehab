#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import argparse

from utils.seed import set_seed
from utils.eval_utils import *
from utils.visualization import (
    predict_and_visualize, 
    visualize_skeleton, 
    create_skeleton_animation, 
)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate H2F-GCN Model')
    
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

def main():
    args = parse_args()
    
    set_seed(args.seed)
    
    os.makedirs(args.output_folder, exist_ok=True)
    print(f"Evaluation outputs will be saved to {args.output_folder}")
    
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