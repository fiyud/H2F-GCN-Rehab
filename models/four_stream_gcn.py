import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from models.modules.DDHG import HyperComputeModule

class FourStreamGCN_Model(nn.Module):
    def __init__(self, num_joints, num_features, hidden_dim, num_layers, output_dim, feat_d, max_time_step=150, nhead=4, dropout=0.1):
        super(FourStreamGCN_Model, self).__init__()

        # Store dimensions for later use
        self.hidden_dim = hidden_dim
        self.num_joints = num_joints

        # STREAM 1: Skeleton-based stream
        self.skeleton_gcn1 = GCNConv(num_features, hidden_dim)
        self.skeleton_gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        self.skeleton_hyper_module = HyperComputeModule(
            c1=hidden_dim,
            c2=hidden_dim,
            reduction=16,
            edge_dim=3,
            hidden_dim=32
        )
        
        self.skeleton_transformer_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout
        )
        self.skeleton_transformer = TransformerEncoder(self.skeleton_transformer_layer, num_layers=1)
        
        # STREAM 2: JCD-based stream
        self.jcd_gcn1 = GCNConv(feat_d, hidden_dim)
        self.jcd_gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        # STREAM 3: Temporal Frequency Stream 
        # temporal convolutions
        self.temp_conv1 = nn.Conv1d(num_joints * num_features, hidden_dim, kernel_size=3, padding=1)
        self.temp_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.temp_pool = nn.AdaptiveAvgPool1d(1)
        
        # STREAM 4: Spatial Frequency Stream 
        self.spatial_gcn1 = GCNConv(num_features, hidden_dim)
        self.spatial_gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        concat_size = num_joints * hidden_dim + hidden_dim + hidden_dim + hidden_dim
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(concat_size, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index, jcd, jcd_edge_index=None):
        batch_size, time_step, num_joints, num_features = x.shape
        
        if jcd_edge_index is None:
            jcd_edge_index = edge_index
        
        #------- STREAM 1: Process skeleton features -------#
        skeleton_x = x.reshape(-1, num_features)  # [batch_size * time_step * num_joints, num_features]
        
        skeleton_x = F.relu(self.skeleton_gcn1(skeleton_x, edge_index))
        skeleton_x = F.relu(self.skeleton_gcn2(skeleton_x, edge_index))
        skeleton_x = skeleton_x.view(batch_size * time_step, num_joints, -1)  # [batch_size * time_step, num_joints, hidden_dim]
        
        hyper_input = skeleton_x.permute(0, 2, 1).unsqueeze(-1)  # [batch_size * time_step, hidden_dim, num_joints, 1]
        hyper_output = self.skeleton_hyper_module(hyper_input)
        skeleton_x = hyper_output.squeeze(-1).permute(0, 2, 1)  # [batch_size * time_step, num_joints, hidden_dim]
        
        transformer_input = skeleton_x.permute(1, 0, 2).contiguous()  # [num_joints, batch_size * time_step, hidden_dim]
        transformer_output = self.skeleton_transformer(transformer_input)
        skeleton_features = transformer_output.permute(1, 0, 2).contiguous()  # [batch_size * time_step, num_joints, hidden_dim]
        
        skeleton_features = skeleton_features.reshape(batch_size, time_step, num_joints * self.hidden_dim)
        
        #------- STREAM 2: Process JCD features -------#
        jcd_flat = jcd.reshape(-1, jcd.size(-1))  # [batch_size * time_step, feat_d]
        
        jcd_features = F.relu(self.jcd_gcn1(jcd_flat, jcd_edge_index))
        jcd_features = F.relu(self.jcd_gcn2(jcd_features, jcd_edge_index))
        
        # Reshape
        jcd_features = jcd_features.reshape(batch_size, time_step, -1)  # [batch_size, time_step, hidden_dim]
        
        #------- STREAM 3: Temporal Frequency Stream -------#
        temp_x = x.reshape(batch_size, time_step, -1)  # [batch_size, time_step, num_joints * num_features]
        temp_x = temp_x.permute(0, 2, 1)  # [batch_size, num_joints * num_features, time_step]
        
        temp_features = F.relu(self.temp_conv1(temp_x))  # [batch_size, hidden_dim, time_step]
        temp_features = F.relu(self.temp_conv2(temp_features))  # [batch_size, hidden_dim, time_step]
        
        temp_features = temp_features.permute(0, 2, 1)  # [batch_size, time_step, hidden_dim]
        
        #------- STREAM 4: Spatial Frequency Stream -------#
        # Create a fully connected spatial graph for frequency analysis
        num_nodes = num_joints
        spatial_edge_index = torch.zeros((2, num_nodes * num_nodes), dtype=torch.long, device=edge_index.device)
        
        idx = 0
        for i in range(num_nodes):
            for j in range(num_nodes):
                spatial_edge_index[0, idx] = i
                spatial_edge_index[1, idx] = j
                idx += 1
        
        spatial_x = x.reshape(-1, num_features)  # [batch_size * time_step * num_joints, num_features]
        spatial_features = F.relu(self.spatial_gcn1(spatial_x, spatial_edge_index))
        spatial_features = F.relu(self.spatial_gcn2(spatial_features, spatial_edge_index))

        # take mean across joints to get a global spatial representation
        spatial_features = spatial_features.reshape(batch_size, time_step, num_joints, -1)
        spatial_features = torch.mean(spatial_features, dim=2)  # [batch_size, time_step, hidden_dim]
        
        combined_features = torch.cat([
            skeleton_features,            # [batch_size, time_step, num_joints * hidden_dim]
            jcd_features,                 # [batch_size, time_step, hidden_dim]
            temp_features,                # [batch_size, time_step, hidden_dim]
            spatial_features              # [batch_size, time_step, hidden_dim]
        ], dim=2)

        fused_features = self.fusion_layer(combined_features)  # [batch_size, time_step, hidden_dim]
        
        gru_out, _ = self.gru(fused_features)  # [batch_size, time_step, hidden_dim]
        
        out = self.fc(gru_out[:, -1, :])  # [batch_size, output_dim]
        
        return out