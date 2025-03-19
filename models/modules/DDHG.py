import torch
import torch.nn as nn

class MessageAgg(nn.Module):
    def __init__(self, agg_method="mean"):
        super().__init__()
        self.agg_method = agg_method
        
    def forward(self, X, path):
        """
            X: [n_node, dim]
            path: col(source) -> row(target)
        """
        X = torch.matmul(path, X)
        if self.agg_method == "mean":
            norm_out = 1 / torch.sum(path, dim=2, keepdim=True)
            norm_out[torch.isinf(norm_out)] = 0
            X = norm_out * X
            return X
        elif self.agg_method == "sum":
            pass
        return X

class HyPConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc = nn.Linear(c1, c2)
        self.v2e = MessageAgg(agg_method="mean")
        self.e2v = MessageAgg(agg_method="mean")
        
    def forward(self, x, H):
        x = self.fc(x)
        # v -> e
        E = self.v2e(x, H.transpose(1, 2).contiguous())
        # e -> v
        x = self.e2v(E, H)
        return x

class AdaptiveThresholdModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        mid_channels = max(in_channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c = x.shape[:2]
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        feat = torch.cat([avg_out, max_out], dim=1)
        threshold = self.mlp(feat).view(b, 1, 1, 1)
        return threshold

class DynamicEdgeFeatures(nn.Module):
    def __init__(self, edge_dim=3, hidden_dim=32):
        super().__init__()
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
   
    def forward(self, distance):
        edge_features = torch.stack([
            distance,
            torch.cos(distance),
            torch.exp(-distance)
        ], dim=-1)
       
        edge_weights = self.edge_proj(edge_features)
        return edge_weights.squeeze(-1)

class HyperComputeModule(nn.Module):
    def __init__(self, c1, c2, reduction=16, edge_dim=3, hidden_dim=32):
        super().__init__()
        self.hgconv = HyPConv(c1, c2)  # c1=c2
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        
        self.adaptive_threshold = AdaptiveThresholdModule(c1, reduction=reduction)        
        self.dynamic_edge = DynamicEdgeFeatures(edge_dim=edge_dim, hidden_dim=hidden_dim)
        
    def forward(self, x):
        b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.view(b, c, -1).transpose(1, 2).contiguous()
        feature = x.clone()
        
        # Calculate distances between nodes
        distance = torch.cdist(feature, feature)
        
        threshold = self.adaptive_threshold(x.transpose(1, 2).contiguous().view(b, c, h, w))
        threshold = threshold.view(b, 1, 1)
        
        hg = distance < threshold
        
        edge_weights = self.dynamic_edge(distance)
        
        # Apply weights to hypergraph connections
        hg = (hg.float() * edge_weights).to(x.device).to(x.dtype)
        
        x = self.hgconv(x, hg).to(x.device).to(x.dtype) + x
        
        x = x.transpose(1, 2).contiguous().view(b, c, h, w)
        x = self.act(self.bn(x))
        
        return x