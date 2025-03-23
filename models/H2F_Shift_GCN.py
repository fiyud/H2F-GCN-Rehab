import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from models.modules.DDHG import HyperComputeModule
from models.modules.shift_gcn import Shift_gcn

class ThreeStreamShift_GCN_ModelvB(nn.Module):
    def __init__(self, num_joints, num_features, hidden_dim, num_layers, output_dim, feat_d, 
                 max_time_step=150, nhead=4, dropout=0.1):
        super(ThreeStreamShift_GCN_ModelvB, self).__init__()

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
        
    def forward(self, x, edge_index, jcd):

            batch_size, time_step, num_joints, num_features = x.shape
            skeleton_x = self.skeleton_gcn1(x)         # output: (n, hidden_dim, t, num_joints)
            skeleton_x = F.relu(skeleton_x)
            skeleton_x = skeleton_x.permute(0, 2, 3, 1).contiguous()
            skeleton_x = self.skeleton_gcn2(skeleton_x)  # output: (n, hidden_dim, t, num_joints)
            skeleton_x = F.relu(skeleton_x)
            skeleton_x = skeleton_x.permute(0, 2, 3, 1).contiguous()
            skeleton_x_reshape = skeleton_x.reshape(batch_size * time_step, num_joints, self.hidden_dim)
            hyper_input = skeleton_x_reshape.permute(0, 2, 1).unsqueeze(-1)
            hyper_output = self.skeleton_hyper_module(hyper_input)
            skeleton_x_proc = hyper_output.squeeze(-1).permute(0, 2, 1).contiguous()
            transformer_input = skeleton_x_proc.permute(1, 0, 2).contiguous()  # (num_joints, n*t, hidden_dim)
            transformer_output = self.skeleton_transformer(transformer_input)
            skeleton_features = transformer_output.permute(1, 0, 2).contiguous()
            skeleton_features = skeleton_features.reshape(batch_size, time_step, num_joints * self.hidden_dim)
            
            # --- STREAM 2: JCD-based stream ---
            jcd_input = jcd.unsqueeze(2)  # (n, t, 1, feat_d)
            jcd_feat = self.jcd_gcn1(jcd_input)        # output: (n, hidden_dim, t, 1)
            jcd_feat = F.relu(jcd_feat)
            jcd_feat = jcd_feat.permute(0, 2, 3, 1).contiguous()  # (n, t, 1, hidden_dim)
            jcd_feat = self.jcd_gcn2(jcd_feat)           # output: (n, hidden_dim, t, 1)
            jcd_feat = F.relu(jcd_feat)
            jcd_feat = jcd_feat.permute(0, 2, 3, 1).contiguous()  # (n, t, 1, hidden_dim)
            jcd_features = jcd_feat.reshape(batch_size, time_step, -1)  # (n, t, hidden_dim)
            
            # --- STREAM 3: Spatial Frequency Stream ---
            spatial_x = self.spatial_gcn1(x)         # output: (n, hidden_dim, t, num_joints)
            spatial_x = F.relu(spatial_x)
            spatial_x = spatial_x.permute(0, 2, 3, 1).contiguous()  # (n, t, num_joints, hidden_dim)
            spatial_x = self.spatial_gcn2(spatial_x)  # output: (n, hidden_dim, t, num_joints)
            spatial_x = F.relu(spatial_x)
            spatial_x = spatial_x.permute(0, 2, 3, 1).contiguous()  # (n, t, num_joints, hidden_dim)
            spatial_features = torch.mean(spatial_x, dim=2)  # (n, t, hidden_dim)
            combined_features = torch.cat([
                skeleton_features,  # (n, t, num_joints * hidden_dim)
                jcd_features,       # (n, t, hidden_dim)
                spatial_features    # (n, t, hidden_dim)
            ], dim=2)
            
            fused_features = self.fusion_layer(combined_features)  # (n, t, hidden_dim)
            gru_out, _ = self.gru(fused_features)  # (n, t, hidden_dim)
            out = self.fc(gru_out[:, -1, :])         # (n, output_dim)
            
            return out
