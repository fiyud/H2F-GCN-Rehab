#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import argparse
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pickle
from data.data_loader import load_kimore_data, preprocess_merged_data
from data.dataset import CustomDataset
from models.H2F_GCN import ThreeStreamGCN_ModelvB
from models.H2F_Shift_GCN import ThreeStreamShift_GCN_ModelvB
from models.four_stream_gcn import FourStreamGCN_Model
from models.H2f_Shift_GCN_ConvGRU import ThreeStreamShift_GCN_ModelvB_ConvGRU
from utils.seed import set_seed
from utils.metrics import compute_metrics
from utils.visualization import predict_and_visualize, visualize_skeleton, create_skeleton_animation
from data.preprocessing import preprocess_data_and_labels, get_JCD

def parse_args():
    parser = argparse.ArgumentParser(description='H2F-GCN')
    
    parser.add_argument('--data_path', type=str, default='Kimore', help='Path to the Kimore dataset')
    parser.add_argument('--exercise', type=int, default=5, choices=[1, 2, 3, 4, 5], 
                        help='Exercise number to use (1-5)')
    
    parser.add_argument('--model', type=str, default='three_stream', 
                        choices=['three_stream', 'four_stream', 'three_stream_shift_gcn', 'three_stream_shift_gcn_gru'], 
                        help='Model architecture to use')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GRU layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.15, help='Dropout rate')
    
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--chunk_size', type=int, default=50, help='Chunk size for sequences')
    parser.add_argument('--seed', type=int, default=100, help='Random seed for reproducibility')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set ratio')
    
    parser.add_argument('--device', type=str, default='', 
                        help='Device to use (leave empty for auto-detection)')
    parser.add_argument('--save_model', action='store_true', help='Save best model')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to save/load model')
    parser.add_argument('--visualize', action='store_true', default=True, help='Visualize predictions')
    parser.add_argument('--vis_ratio', type=float, default=0.3, 
                        help='Ratio of test samples to visualize')
    parser.add_argument('--visualize_skeleton', action='store_true', default=True,
                        help='Visualize skeleton data')
    parser.add_argument('--create_animation', action='store_true', default=True,
                        help='Create skeleton animations')
    parser.add_argument('--output_folder', type=str, default='visualization_output',
                        help='Folder to save visualization outputs')
    
    args = parser.parse_args()
    
    if not args.device:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)
    
    return args

def compute_pareto_front(metrics_list):
    pareto = []
    for candidate in metrics_list:
        dominated = False
        for other in metrics_list:
            # Compare MAD, RMSE, and MAPE (lower is better)
            if (other[1] <= candidate[1] and 
                other[2] <= candidate[2] and 
                other[3] <= candidate[3] and
                (other[1] < candidate[1] or other[2] < candidate[2] or other[3] < candidate[3])):
                dominated = True
                break
        if not dominated:
            pareto.append(candidate)
    return pareto

def main():
    args = parse_args()
    glb_model_save_path = os.path.join(args.output_folder, f"best_model_exercise{args.exercise}_shiftgcn.pth")

    checkpoint_path = os.path.join(args.output_folder, f"checkpoint_ex{args.exercise}.pth")
    
    os.makedirs(args.output_folder, exist_ok=True)
    set_seed(args.seed)
    
    print("Using device:", args.device)
    print(f"Loading Kimore dataset from {args.data_path}...")
    
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)
    
    df_ex = data['ex' + str(args.exercise)]
    
    data_tensor, labels_tensor = preprocess_data_and_labels(df_ex, args.chunk_size)
    position_data = data_tensor[:, :, :, 4:] 
    
    JCD = get_JCD(position_data)
    
    train_data, test_data, train_jcd, test_jcd, train_labels, test_labels = train_test_split(
        data_tensor, JCD, labels_tensor, test_size=args.test_size, random_state=args.seed
    )
    
    train_dataset = CustomDataset(train_data, train_jcd, train_labels)
    test_dataset = CustomDataset(test_data, test_jcd, test_labels)
    
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        generator=g,
        worker_init_fn=lambda worker_id: np.random.seed(args.seed + worker_id)
    )
    
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
    
    num_joints = 25
    num_features = 7
    output_dim = 1  # For cTS prediction
    
    if args.model == 'three_stream':
        model = ThreeStreamGCN_ModelvB(
            num_joints=num_joints,
            num_features=num_features,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            output_dim=output_dim,
            feat_d=JCD.size(-1),
            nhead=args.num_heads,
            dropout=args.dropout
        )
    elif args.model == 'three_stream_shift_gcn':
        model = ThreeStreamShift_GCN_ModelvB(
            num_joints=num_joints,
            num_features=num_features,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            output_dim=output_dim,
            feat_d=JCD.size(-1),
            nhead=args.num_heads,
            dropout=args.dropout
        )
    elif args.model == 'three_stream_shift_gcn_gru':
        model = ThreeStreamShift_GCN_ModelvB_ConvGRU(
            num_joints=num_joints,
            num_features=num_features,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            output_dim=output_dim,
            feat_d=JCD.size(-1),
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
            feat_d=JCD.size(-1),
            nhead=args.num_heads,
            dropout=args.dropout
        )
    
    model.to(args.device)
    
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_rmse = float('inf')
    best_mad = float('inf')
    best_mape = float('inf')
    best_epoch = 0
    best_model_state = None

    # List to store [epoch, MAD, RMSE, MAPE]
    epoch_metrics = []
    
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # ----------------- Training -----------------
        model.train()
        train_loss = 0.0
    
        for batch_idx, (data, jcd, labels) in enumerate(train_loader):
            # Move data to device
            data = data.to(args.device)
            jcd = jcd.to(args.device)
            labels = labels.to(args.device)
    
            optimizer.zero_grad()
            outputs = model(data, edge_index.to(args.device), jcd)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            train_loss += loss.item()
    
        avg_train_loss = train_loss / len(train_loader)
    
        # ----------------- Evaluation -----------------
        model.eval()
        test_loss = 0.0
        test_mad, test_mape, test_rmse = 0.0, 0.0, 0.0
    
        with torch.no_grad():
            for batch_idx, (data, jcd, labels) in enumerate(test_loader):
                data = data.to(args.device)
                jcd = jcd.to(args.device)
                labels = labels.to(args.device)
    
                outputs = model(data, edge_index.to(args.device), jcd)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
    
                mad, mape, rmse = compute_metrics(labels, outputs)
                test_mad += mad
                test_mape += mape
                test_rmse += rmse
    
        # Average metrics across batches
        avg_test_loss = test_loss / len(test_loader)
        avg_test_mad = test_mad / len(test_loader)
        avg_test_mape = test_mape / len(test_loader)
        avg_test_rmse = test_rmse / len(test_loader)
    
        # Save metrics for current epoch
        epoch_metrics.append([epoch+1, avg_test_mad, avg_test_rmse, avg_test_mape])
    
        is_best = False
        if avg_test_rmse < best_rmse:
            best_rmse = avg_test_rmse
            best_mad = avg_test_mad
            best_mape = avg_test_mape
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            # Overwrite the best model checkpoint
            torch.save(best_model_state, glb_model_save_path)
            is_best = True
    
        print(f"Epoch [{epoch+1}/{args.epochs}]")
        print(f"Train Loss: {avg_train_loss:.6f}")
        print(f"Test Loss: {avg_test_loss:.6f}")
        print(f"  - MAD: {avg_test_mad:.4f}, RMSE: {avg_test_rmse:.4f}, MAPE: {avg_test_mape:.2f}")
        
        if is_best:
            print(f"  ✓ New best result!")
        print("-" * 80)
        
        # Autosave checkpoint (this file gets overwritten each epoch)
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
            'metrics': {'mad': avg_test_mad, 'rmse': avg_test_rmse, 'mape': avg_test_mape}
        }
        torch.save(checkpoint, checkpoint_path)
    
    # Save the Pareto front of metrics
    pareto_front = compute_pareto_front(epoch_metrics)
    pareto_front = np.array(pareto_front)
    np.save(os.path.join(args.output_folder, f"pareto_front_ex{args.exercise}.npy"), pareto_front)
    print("Pareto front saved.")
    
    print("\n" + "=" * 40)
    print(f"BEST RESULT (Epoch {best_epoch}):")
    print(f"- MAD: {best_mad:.4f}")
    print(f"- RMSE: {best_rmse:.4f}")
    print(f"- MAPE: {best_mape:.2f}%")
    print("=" * 40)
    
    print(f"\nSaving visualizations to {args.output_folder}...")
    
    print("Visualizing predictions...")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    predict_and_visualize(
        model, 
        test_loader, 
        args.device, 
        edge_index, 
        args.vis_ratio, 
        args.seed,
        save_path=os.path.join(args.output_folder, f"predictions_ex{args.exercise}.png")
    )
    
    if args.visualize_skeleton:
        print("Visualizing skeleton data...")
        for batch_idx, (data, jcd, labels) in enumerate(test_loader):
            position_data = data[:, :, :, 4:7].to('cpu')
            
            for sample_idx in range(min(3, len(data))):
                for frame_idx in [0, int(data.shape[1]/2), data.shape[1]-1]:
                    vis_path = os.path.join(args.output_folder, f"skeleton_ex{args.exercise}_sample{sample_idx}_frame{frame_idx}.png")
                    visualize_skeleton(
                        position_data, 
                        frame_idx=frame_idx, 
                        sample_idx=sample_idx,
                        save_path=vis_path
                    )
            
            if args.create_animation:
                print("Creating skeleton animations...")
                for sample_idx in range(min(2, len(data))):
                    anim_path = os.path.join(args.output_folder, f"animation_ex{args.exercise}_sample{sample_idx}.gif")
                    create_skeleton_animation(
                        position_data,
                        sample_idx=sample_idx,
                        output_file=anim_path
                    )
            
            break

if __name__ == "__main__":
    main()
