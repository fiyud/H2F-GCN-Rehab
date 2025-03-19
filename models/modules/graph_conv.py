import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConvCustom(MessagePassing):
    """
    Custom implementation of Graph Convolutional Network layer
    """
    def __init__(self, in_channels, out_channels):
        super(GCNConvCustom, self).__init__(aggr='add')  # "Add" aggregation
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        # Add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Linearly transform node feature matrix
        x = self.lin(x)

        # Compute normalization
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Apply normalization to edge weights
        if edge_weight is not None:
            norm = norm * edge_weight

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # Normalize node features
        return norm.view(-1, 1) * x_j

class ResGCNBlock(nn.Module):
    """
    Residual GCN block with improved stability
    """
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(ResGCNBlock, self).__init__()
        self.gcn = GCNConvCustom(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection with projection if needed
        self.has_proj = in_channels != out_channels
        if self.has_proj:
            self.proj = nn.Linear(in_channels, out_channels)
            
    def forward(self, x, edge_index, edge_weight=None):
        identity = x
        out = self.gcn(x, edge_index, edge_weight)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.norm(out)
        
        # Apply projection for residual if needed
        if self.has_proj:
            identity = self.proj(identity)
            
        # Add residual connection
        out = out + identity
        
        return out