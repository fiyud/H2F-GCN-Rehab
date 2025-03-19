#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import argparse
import numpy as np
import pandas as pd
from torch import optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data.data_loader import load_kimore_data, preprocess_merged_data
from data.dataset import CustomDataset
from models.three_stream_gcn import ThreeStreamGCN_ModelvB
from models.four_stream_gcn import FourStreamGCN_Model
from utils.seed import set_seed
from utils.metrics import compute_metrics
from utils.visualization import predict_and_visualize
from data.preprocessing import preprocess_data_and_labels, get_JCD

def parse_args():
    parser = argparse.ArgumentParser(description='H2F-GCN')
    
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
    
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--chunk_size', type=int, default=50, help='Chunk size for sequences')
    parser.add_argument('--seed', type=int, default=100, help='Random seed for reproducibility')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set ratio')
    
    parser.add_argument('--device', type=str, default='', 
                        help='Device to use (leave empty for auto-detection)')
    parser.add_argument('--save_model', action='store_true', help='Save best model')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to save/load model')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    parser.add_argument('--vis_ratio', type=float, default=0.3, 
                        help='Ratio of test samples to visualize')
    
    args = parser.parse_args()
    
    if not args.device:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)
    
    return args

def main():
    args = parse_args()

    set_seed(args.seed)

    print(f"Loading Kimore dataset from {args.data_path}...")
    data = load_kimore_data(args.data_path)
    
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
    else:
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
    
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # ----------------- Training -----------------
        model.train()
        train_loss = 0.0
    
        for batch_idx, (data, jcd, labels) in enumerate(train_loader):
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
    
        is_best = False
        if avg_test_rmse < best_rmse:
            best_rmse = avg_test_rmse
            best_mad = avg_test_mad
            best_mape = avg_test_mape
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            is_best = True
    
        print(f"Epoch [{epoch+1}/{args.epochs}]")
        print(f"Train Loss: {avg_train_loss:.6f}")
        print(f"Test Loss: {avg_test_loss:.6f}")
        print(f"  - MAD: {avg_test_mad:.4f}, RMSE: {avg_test_rmse:.4f}, MAPE: {avg_test_mape:.2f}")
        
        if is_best:
            print(f"  ✓ New best result!")
        print("-" * 80)
    
    if args.save_model and best_model_state is not None:
        print(f"Saving best model to {args.model_path}")
        torch.save(best_model_state, args.model_path)
    
    print("\n" + "=" * 40)
    print(f"BEST RESULT (Epoch {best_epoch}):")
    print(f"- MAD: {best_mad:.4f}")
    print(f"- RMSE: {best_rmse:.4f}")
    print(f"- MAPE: {best_mape:.2f}%")
    print("=" * 40)
    
    if args.visualize:
        print("Visualizing predictions...")
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        predict_and_visualize(model, test_loader, args.device, edge_index, args.vis_ratio, args.seed)

if __name__ == "__main__":
    main()