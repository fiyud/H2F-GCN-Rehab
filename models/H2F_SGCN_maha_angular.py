import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from models.modules.DDHG import HyperComputeModule
from models.modules.shift_gcn import Shift_gcn

class MahalanobisDistanceModule(nn.Module):
    def __init__(self, coord_dim=3):
        super(MahalanobisDistanceModule, self).__init__()
        self.L = nn.Linear(coord_dim, coord_dim, bias=False)
        
    def forward_diff(self, diff):
        diff_transformed = self.L(diff)
        return torch.norm(diff_transformed, p=2, dim=-1)
    
    def forward(self, p_i, p_j):
        diff = p_i - p_j
        return self.forward_diff(diff)

class MahalanobisDistanceModule(nn.Module):
    def __init__(self, coord_dim=3):
        super(MahalanobisDistanceModule, self).__init__()
        self.L = nn.Linear(coord_dim, coord_dim, bias=False)
        
    def forward_diff(self, diff):
        diff_transformed = self.L(diff)
        return torch.norm(diff_transformed, p=2, dim=-1)
    
    def forward(self, p_i, p_j):
        diff = p_i - p_j
        return self.forward_diff(diff)

def JCD_mahalanobis_angular(p, distance_module, temporal_weight=0.3, angular_weight=0.4):
    batch_size, time_step, num_joints, coord_dim = p.shape
    
    # Initialize output tensor
    num_pairs = (num_joints * (num_joints - 1)) // 2
    JCD = torch.zeros((batch_size, time_step, num_pairs), device=p.device)
    
    # Get indices for upper triangular pairs
    indices = torch.triu_indices(num_joints, num_joints, offset=1, device=p.device)
    
    # Get all pairs of joints differences
    p_i = p.unsqueeze(3)
    p_j = p.unsqueeze(2)
    spatial_diff = p_i - p_j
    spatial_diff_pairs = spatial_diff[:, :, indices[0], indices[1], :]
    
    # Compute the Mahalanobis distance for spatial component
    spatial_diff_flat = spatial_diff_pairs.reshape(-1, coord_dim)
    spatial_dist_flat = distance_module.forward_diff(spatial_diff_flat)
    spatial_dist = spatial_dist_flat.reshape(batch_size, time_step, -1)
    
    # Combine with temporal and angular components
    for b in range(batch_size):
        for t in range(time_step):
            if t > 0:  # For frames after the first one
                # Calculate velocities
                vel = p[b, t] - p[b, t-1]  # Shape: [num_joints, coord_dim]
                
                # Prepare velocity differences for all pairs
                vel_i = vel[indices[0]].unsqueeze(1)  # Shape: [num_pairs, 1, coord_dim]
                vel_j = vel[indices[1]].unsqueeze(1)  # Shape: [num_pairs, 1, coord_dim]
                
                # Temporal component - difference in velocities
                temporal_diff = vel_i - vel_j  # Shape: [num_pairs, 1, coord_dim]
                temporal_dist = distance_module.forward_diff(temporal_diff.squeeze(1))
                
                # Angular component - angle between velocity vectors
                vel_i_norm = vel_i / (torch.norm(vel_i, dim=-1, keepdim=True) + 1e-8)
                vel_j_norm = vel_j / (torch.norm(vel_j, dim=-1, keepdim=True) + 1e-8)
                
                cos_sim = torch.sum(vel_i_norm * vel_j_norm, dim=-1).squeeze(1)
                cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
                angular_dist = torch.acos(cos_sim) / torch.pi
                
                # Combine all components with weights
                spatial_w = 1.0 - temporal_weight - angular_weight
                JCD[b, t] = (
                    spatial_w * spatial_dist[b, t] +
                    temporal_weight * temporal_dist +
                    angular_weight * angular_dist
                )
            else:
                # For first frame, only use spatial information
                JCD[b, t] = spatial_dist[b, t]
    
    # Normalize JCD per batch
    for b in range(batch_size):
        b_min = JCD[b].min()
        b_max = JCD[b].max()
        if b_max > b_min:  # Avoid division by zero
            JCD[b] = (JCD[b] - b_min) / (b_max - b_min)
    
    return JCD

class ThreeStreamMaha_GCN_ModelvB(nn.Module):
    def __init__(self, num_joints, num_features, hidden_dim, num_layers, output_dim, feat_d, max_time_step=150, nhead=4, dropout=0.1):
        super(ThreeStreamMaha_GCN_ModelvB, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_joints = num_joints

        self.skeleton_gcn1 = Shift_gcn(in_channels=num_features, out_channels=hidden_dim, num_nodes=num_joints)
        self.skeleton_gcn2 = Shift_gcn(in_channels=hidden_dim, out_channels=hidden_dim, num_nodes=num_joints)
        
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
        
        self.jcd_gcn1 = Shift_gcn(in_channels=feat_d, out_channels=hidden_dim, num_nodes=1)
        self.jcd_gcn2 = Shift_gcn(in_channels=hidden_dim, out_channels=hidden_dim, num_nodes=1)
        
        self.spatial_gcn1 = Shift_gcn(in_channels=num_features, out_channels=hidden_dim, num_nodes=num_joints)
        self.spatial_gcn2 = Shift_gcn(in_channels=hidden_dim, out_channels=hidden_dim, num_nodes=num_joints)
        
        concat_size = num_joints * hidden_dim + hidden_dim + hidden_dim
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
        
        self.mah_distance_module = MahalanobisDistanceModule(coord_dim=3)
        
    def forward(self, x, edge_index, jcd=None, jcd_edge_index=None):
        batch_size, time_step, num_joints, num_features = x.shape
        
        if jcd_edge_index is None:
            jcd_edge_index = edge_index
        
        skeleton_x = F.relu(self.skeleton_gcn1(x, edge_index))
        skeleton_x = skeleton_x.permute(0, 2, 3, 1).contiguous()
        skeleton_x = F.relu(self.skeleton_gcn2(skeleton_x, edge_index))
        skeleton_x = skeleton_x.permute(0, 2, 3, 1).contiguous()
        skeleton_x = skeleton_x.view(batch_size * time_step, num_joints, -1)
        
        hyper_input = skeleton_x.permute(0, 2, 1).unsqueeze(-1)
        hyper_output = self.skeleton_hyper_module(hyper_input)
        skeleton_x = hyper_output.squeeze(-1).permute(0, 2, 1)
        
        transformer_input = skeleton_x.permute(1, 0, 2).contiguous()
        transformer_output = self.skeleton_transformer(transformer_input)
        skeleton_features = transformer_output.permute(1, 0, 2).contiguous()
        skeleton_features = skeleton_features.reshape(batch_size, time_step, num_joints * self.hidden_dim)
        
        pos_data = x[:, :, :, :3]
        jcd = JCD_mahalanobis_angular(pos_data, self.mah_distance_module, temporal_weight=0.3, angular_weight=0.4)
        jcd_flat = jcd.unsqueeze(2)
        jcd_feat = F.relu(self.jcd_gcn1(jcd_flat))
        jcd_feat = jcd_feat.permute(0, 2, 3, 1).contiguous()
        jcd_feat = F.relu(self.jcd_gcn2(jcd_feat))
        jcd_feat = jcd_feat.permute(0, 2, 3, 1).contiguous() 
        jcd_features = jcd_feat.reshape(batch_size, time_step, -1)
        
        num_nodes = num_joints
        spatial_edge_index = torch.zeros((2, num_nodes * num_nodes), dtype=torch.long, device=edge_index.device)
        idx = 0
        for i in range(num_nodes):
            for j in range(num_nodes):
                spatial_edge_index[0, idx] = i
                spatial_edge_index[1, idx] = j
                idx += 1
        
        spatial_features = F.relu(self.spatial_gcn1(x, spatial_edge_index))
        spatial_features = spatial_features.permute(0, 2, 3, 1).contiguous()
        spatial_features = F.relu(self.spatial_gcn2(spatial_features, spatial_edge_index))
        spatial_features = spatial_features.permute(0, 2, 3, 1).contiguous()
        spatial_features = spatial_features.reshape(batch_size, time_step, num_joints, -1)
        spatial_features = torch.mean(spatial_features, dim=2)
        
        combined_features = torch.cat([
            skeleton_features,
            jcd_features,
            spatial_features
        ], dim=2)
        
        fused_features = self.fusion_layer(combined_features)
        gru_out, _ = self.gru(fused_features)
        out = self.fc(gru_out[:, -1, :])
        
        return out
