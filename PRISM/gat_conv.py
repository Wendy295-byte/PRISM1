import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union, Tuple, Optional

def softmax(src: Tensor, index: Tensor, num_nodes: Optional[int] = None) -> Tensor:
    """
    A pure PyTorch implementation of softmax for graph data (scatter-based).
    """
    if num_nodes is None:
        num_nodes = int(index.max()) + 1 if index.numel() > 0 else 0
    
    # Subtract max for numerical stability
    out = src - src.max()
    out = out.exp()
    
    # Sum exp values for each destination node
    sum_exp = torch.zeros((num_nodes, *out.shape[1:]), device=out.device, dtype=out.dtype)
    sum_exp.index_add_(0, index, out)
    
    # Divide by sum
    out = out / (sum_exp[index] + 1e-16)
    return out

class GATConv(nn.Module):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        super(GATConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        # Linear transformation
        # If in_channels is a tuple, it's (in_channels_src, in_channels_dst)
        if isinstance(in_channels, int):
            self.lin_src = nn.Parameter(torch.Tensor(in_channels, out_channels))
            self.lin_dst = self.lin_src
        else:
            self.lin_src = nn.Parameter(torch.Tensor(in_channels[0], out_channels))
            self.lin_dst = nn.Parameter(torch.Tensor(in_channels[1], out_channels))

        # Attention parameters
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))

        if bias:
            if concat:
                self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
            else:
                self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.attentions = None
        self._alpha = None

    def reset_parameters(self):
        nn.init.xavier_normal_(self.lin_src, gain=1.414)
        if self.lin_dst is not self.lin_src:
            nn.init.xavier_normal_(self.lin_dst, gain=1.414)
        nn.init.xavier_normal_(self.att_src, gain=1.414)
        nn.init.xavier_normal_(self.att_dst, gain=1.414)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Union[Tensor, Tuple[Tensor, Optional[Tensor]]], 
                edge_index: Tensor, size: Optional[Tuple[int, int]] = None, 
                return_attention_weights=None, attention=True, tied_attention=None):
        
        H, C = self.heads, self.out_channels

        # 1. Linear Transformation
        if isinstance(x, Tensor):
            x_src = x_dst = torch.matmul(x, self.lin_src).view(-1, H, C)
        else:
            x_src, x_dst = x
            x_src = torch.matmul(x_src, self.lin_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = torch.matmul(x_dst, self.lin_dst).view(-1, H, C)
            else:
                x_dst = x_src

        # 2. Add Self-loops if needed
        if self.add_self_loops:
            num_nodes = x_src.size(0)
            # Simple self-loop addition for Tensor edge_index
            # Remove existing self-loops first
            mask = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, mask]
            # Add new self-loops
            loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
            loop_index = loop_index.unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, loop_index], dim=1)

        # 3. Attention calculation
        if attention:
            if tied_attention is None:
                # Compute attention coefficients
                # alpha_src: [N, H], alpha_dst: [N, H]
                alpha_src = (x_src * self.att_src).sum(dim=-1)
                alpha_dst = (x_dst * self.att_dst).sum(dim=-1)
                
                edge_src, edge_dst = edge_index[0], edge_index[1]
                
                # e = LeakyReLU(a^T [Wh_i || Wh_j])
                # This is equivalent to a_src^T Wh_i + a_dst^T Wh_j
                alpha = alpha_src[edge_src] + alpha_dst[edge_dst]
                alpha = F.leaky_relu(alpha, self.negative_slope)
                
                # Softmax over neighbors
                alpha = softmax(alpha, edge_dst, num_nodes=x_dst.size(0))
                self.attentions = (alpha_src, alpha_dst)
            else:
                # Use provided tied attention
                # tied_attention is (alpha_src, alpha_dst)
                alpha_src, alpha_dst = tied_attention
                edge_src, edge_dst = edge_index[0], edge_index[1]
                alpha = alpha_src[edge_src] + alpha_dst[edge_dst]
                alpha = F.leaky_relu(alpha, self.negative_slope)
                alpha = softmax(alpha, edge_dst, num_nodes=x_dst.size(0))

            self._alpha = alpha
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            
            # Weighted aggregation
            # out_i = sum_j alpha_ij * Wh_j
            weighted_x = x_src[edge_src] * alpha.unsqueeze(-1) # [E, H, C]
            
            out = torch.zeros((x_dst.size(0), H, C), device=x_src.device, dtype=x_src.dtype)
            out.index_add_(0, edge_dst, weighted_x)
        else:
            # Simple aggregation (mean) if attention is False
            edge_src, edge_dst = edge_index[0], edge_index[1]
            out = torch.zeros((x_dst.size(0), H, C), device=x_src.device, dtype=x_src.dtype)
            out.index_add_(0, edge_dst, x_src[edge_src])
            
            # Divide by degree
            degree = torch.zeros(x_dst.size(0), device=x_src.device)
            degree.index_add_(0, edge_dst, torch.ones(edge_src.size(0), device=x_src.device))
            out = out / (degree.view(-1, 1, 1) + 1e-16)

        # 4. Final output processing
        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            return out, (edge_index, self._alpha)
        return out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
