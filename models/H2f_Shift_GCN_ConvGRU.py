import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.autograd import Variable

from models.modules.DDHG import HyperComputeModule
from models.modules.shift_gcn import Shift_gcn
from models.modules.ConvGRU import ConvGRU

class ThreeStreamShift_GCN_ModelvB_ConvGRU(nn.Module):
    def __init__(self, num_joints, num_features, hidden_dim, num_layers, output_dim, feat_d, 
                 max_time_step=150, nhead=4, dropout=0.1):
        super(ThreeStreamShift_GCN_ModelvB_ConvGRU, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        
        # Replace GRU with ConvGRU with proper device handling
        self.convgru = ConvGRU(
            input_size=(1, 1),
            input_dim=hidden_dim,
            hidden_dim=[hidden_dim] * num_layers,
            kernel_size=(3, 3),
            num_layers=num_layers,
            dtype=torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor,
            batch_first=True,
            bias=True,
            return_all_layers=False
        ).to(self.device)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Move entire model to appropriate device
        self.to(self.device)

    def forward(self, x, edge_index, jcd):
        # Ensure inputs are on the correct device
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        jcd = jcd.to(self.device)
        
        batch_size, time_step, num_joints, num_features = x.shape
        
        # --- STREAM 1: Skeleton Stream ---
        skeleton_x = self.skeleton_gcn1(x)
        skeleton_x = F.relu(skeleton_x)
        skeleton_x = skeleton_x.permute(0, 2, 3, 1).contiguous()
        skeleton_x = self.skeleton_gcn2(skeleton_x)
        skeleton_x = F.relu(skeleton_x)
        skeleton_x = skeleton_x.permute(0, 2, 3, 1).contiguous()
        skeleton_x_reshape = skeleton_x.reshape(batch_size * time_step, num_joints, self.hidden_dim)
        hyper_input = skeleton_x_reshape.permute(0, 2, 1).unsqueeze(-1)
        hyper_output = self.skeleton_hyper_module(hyper_input)
        skeleton_x_proc = hyper_output.squeeze(-1).permute(0, 2, 1).contiguous()
        transformer_input = skeleton_x_proc.permute(1, 0, 2).contiguous()
        transformer_output = self.skeleton_transformer(transformer_input)
        skeleton_features = transformer_output.permute(1, 0, 2).contiguous()
        skeleton_features = skeleton_features.reshape(batch_size, time_step, num_joints * self.hidden_dim)
        
        # --- STREAM 2: JCD-based stream ---
        jcd_input = jcd.unsqueeze(2)
        jcd_feat = self.jcd_gcn1(jcd_input)
        jcd_feat = F.relu(jcd_feat)
        jcd_feat = jcd_feat.permute(0, 2, 3, 1).contiguous()
        jcd_feat = self.jcd_gcn2(jcd_feat)
        jcd_feat = F.relu(jcd_feat)
        jcd_feat = jcd_feat.permute(0, 2, 3, 1).contiguous()
        jcd_features = jcd_feat.reshape(batch_size, time_step, -1)
        
        # --- STREAM 3: Spatial Frequency Stream ---
        spatial_x = self.spatial_gcn1(x)
        spatial_x = F.relu(spatial_x)
        spatial_x = spatial_x.permute(0, 2, 3, 1).contiguous()
        spatial_x = self.spatial_gcn2(spatial_x)
        spatial_x = F.relu(spatial_x)
        spatial_x = spatial_x.permute(0, 2, 3, 1).contiguous()
        spatial_features = torch.mean(spatial_x, dim=2)
        
        # Combine features
        combined_features = torch.cat([
            skeleton_features,
            jcd_features,
            spatial_features
        ], dim=2)
        
        fused_features = self.fusion_layer(combined_features)
        
        # Reshape for ConvGRU
        convgru_input = fused_features.unsqueeze(3).unsqueeze(4)
        
        # Pass through ConvGRU
        layer_output_list, last_state_list = self.convgru(convgru_input)
        convgru_out = layer_output_list[0]
        
        # Prepare output
        convgru_out = convgru_out.squeeze(3).squeeze(3)
        out = self.fc(convgru_out[:, -1, :])
        
        return out