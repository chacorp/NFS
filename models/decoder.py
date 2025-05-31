import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseDecoder(nn.Module):
    """
    Decoder from NFR
    """
    def __init__(self, 
                 in_dim, 
                 hid_dim=128, 
                 num_gn=32, 
                 num_layer=6,
                 out_shape=9, 
                 act='relu'):
        super(BaseDecoder, self).__init__()
        
        if act == 'none':
            self.act = lambda x: x
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'lrelu':
            self.act = nn.LeakyReLU()
        
        self.linears = [nn.Linear(in_dim, hid_dim, bias=False)]
        for _ in range(num_layer):
            self.linears.append(nn.Linear(hid_dim, hid_dim, bias=False))

        # Linear layers
        self.linears = nn.ModuleList(self.linears)
        
        # GROUP NORMs ## Input: (N,C,∗) 
        self.gns = [nn.GroupNorm(num_gn, hid_dim) for _ in range(len(self.linears))]
        self.gns = nn.ModuleList(self.gns)
        
        self.linear_out = nn.Linear(hid_dim, out_shape)

    def forward(self, x):
        """
            x (torch.tensor) [B, N, C]: input
            out (torch.tensor)
        """
        out = x
        for i in range(len(self.linears)):
            out = self.linears[i](out)
            out = torch.transpose(self.act(self.gns[i](torch.transpose(out, -1, -2))), -1, -2)
        out = self.linear_out(out)
        return out
    
class SkinningDecoder(nn.Module):
    def __init__(self, 
                 in_dim,
                 id_dim=100,
                 exp_dim=128,
                 seg_dim=20,
                 hid_dim=256, 
                 num_layer=6,
                 num_gn=32, 
                 out_shape=9, 
                 act='relu',
                 exclude_MLP=False,
                ):
        super().__init__()
        self.out_shape = out_shape
        self.in_dim = in_dim

        self.seg_dim = seg_dim
        self.exclude_MLP = exclude_MLP
        
        if act == 'none':
            self.act = lambda x: x
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'lrelu':
            self.act = nn.LeakyReLU()
        
        self.linears = [nn.Linear(in_dim + exp_dim + id_dim, hid_dim, bias=False)]
        for _ in range(num_layer):
            self.linears.append(nn.Linear(hid_dim, hid_dim, bias=False))
                    
        # Linear layers
        self.linears = nn.ModuleList(self.linears)
        
        # GROUP NORMs
        self.gns = [nn.GroupNorm(num_gn, hid_dim) for _ in range(len(self.linears))]
        self.gns = nn.ModuleList(self.gns)
        
        self.linear_out = nn.Linear(hid_dim, out_shape)
        
        skinning_layer = [
            nn.Linear(seg_dim, hid_dim, bias=False),
            self.act,
            nn.Linear(hid_dim, hid_dim, bias=False),
            self.act,
            nn.Linear(hid_dim, exp_dim, bias=False),
        ]
        self.skinning_layer = nn.Sequential(*skinning_layer)

    ## skinning
    def apply_skinning(self, exp_code, seg_code):
        skinning_weight = self.skinning_layer(seg_code) # [B, V, exp]
        
        # masking and amplify relevent expression 
        focused_exp = skinning_weight * exp_code
        return focused_exp
    
    def forward(self, x, exp_code, id_code, seg_code, eps=1e-12):
        """batch is 1, V = number of vertices

        Args
        ----
            x: (torch.tensor) [B, V, C]: input mesh vertex/face features
            exp_code: (torch.tensor) [B, exp]
            id_code: (torch.tensor) [B, 1, ID]
            seg_code: (torch.tensor) [1, V, S]: input mesh vertex/face segments

        Returns
        -------
            out: (torch.tensor)
        """
        
        focused_exp = self.apply_skinning(exp_code, seg_code) # [B, V, exp]
        
        out = torch.cat([x, focused_exp, id_code], dim=-1) #[B, V, C+exp+ID]
        
        for i in range(len(self.linears)):
            tmp = self.linears[i](out)
            out = torch.transpose(self.act(self.gns[i](torch.transpose(tmp, -1, -2))), -1, -2)
        out = self.linear_out(out)
        
        return out