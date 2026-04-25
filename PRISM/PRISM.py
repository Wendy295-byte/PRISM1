import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .gat_conv import GATConv
from sklearn.decomposition import PCA

# Hyperparameters for PRISM
batch = 64
pca_components = 8

class TransformerEncoder(nn.Module):
    """Transformer Encoder block with Multi-head Attention and MLP."""
    def __init__(self, embedding_dim, nhead, mlp_dim):
        super(TransformerEncoder, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embedding_dim, nhead)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embedding_dim)
        )

    def forward(self, x):
        # Attention + Residual + Norm
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.norm1(x + attn_output)
        # MLP + Residual + Norm
        mlp_output = self.mlp(x)
        x = self.norm2(x + mlp_output)
        return x

class PRISM(nn.Module):
    """
    PRISM Model for multi-modal embedding and prediction.
    Features GAT-based encoding/decoding and Transformer-based feature fusion.
    """
    def __init__(self, src_hidden_dims, tgt_hidden_dims, src_out, tgt_out):
        super(PRISM, self).__init__()
        self.batch_size = batch
        self.n_components = pca_components
        self.pca = PCA(self.n_components)
        self.relu = nn.ReLU()

        # Input dimension parsing
        [src_in_dim, src_num_hidden, src_out_dim] = src_hidden_dims
        [tgt_in_dim, tgt_num_hidden, tgt_out_dim] = tgt_hidden_dims
        
        # --- Source Modality (e.g., RNA) GAT Blocks ---
        self.en_src_conv1 = GATConv(src_in_dim, src_num_hidden, heads=1, concat=False, add_self_loops=False, bias=False)
        self.en_src_conv2 = GATConv(src_num_hidden, src_out_dim, heads=1, concat=False, add_self_loops=False, bias=False)
        
        self.de_src_conv1 = GATConv(src_out_dim, src_num_hidden, heads=1, concat=False, add_self_loops=False, bias=False)
        self.de_src_conv2 = GATConv(src_num_hidden, src_in_dim, heads=1, concat=False, add_self_loops=False, bias=False)

        # --- Target Modality (e.g., Protein) GAT Blocks ---
        self.en_tgt_conv1 = GATConv(tgt_in_dim, tgt_num_hidden, heads=1, concat=False, add_self_loops=False, bias=False)
        self.en_tgt_conv2 = GATConv(tgt_num_hidden, tgt_out_dim, heads=1, concat=False, add_self_loops=False, bias=False)
        
        self.de_tgt_conv1 = GATConv(tgt_out_dim, tgt_num_hidden, heads=1, concat=False, add_self_loops=False, bias=False)
        self.de_tgt_conv2 = GATConv(tgt_num_hidden, tgt_in_dim, heads=1, concat=False, add_self_loops=False, bias=False)

        # --- Fusion Components ---
        # Learnable token for unrigistered cells/spots imputation
        self.token = nn.Parameter(torch.randn(src_out_dim)) 
        nn.init.normal_(self.token, mean=0.0, std=1.0)

        # Multi-modal Transformer
        fusion_dim = src_out_dim + tgt_out_dim
        self.transformer = TransformerEncoder(embedding_dim=fusion_dim, nhead=8, mlp_dim=64)
        
        # Projection heads for fused features
        self.proj_src = nn.Sequential(
            nn.Linear(fusion_dim, src_num_hidden),
            nn.ReLU(),
            nn.Linear(src_num_hidden, src_out_dim)
        )
        self.proj_tgt = nn.Sequential(
            nn.Linear(fusion_dim, tgt_num_hidden),
            nn.ReLU(),
            nn.Linear(tgt_num_hidden, tgt_out_dim)
        )

        # Prediction Heads: share parameters
        self.de_src_pre1 = self.de_src_conv1
        self.de_src_pre2 = self.de_src_conv2

        self.de_tgt_pre1 = self.de_tgt_conv1
        self.de_tgt_pre2 = self.de_tgt_conv2

    #def forward(self, src_x, src_edge, tgt_x, tgt_edge, mask, valid_distances, top_k_indices, non_missing_indices):
    def forward(self, src_x, src_edge, tgt_x, tgt_edge, top_k_indices, non_missing_indices):

        # 1. Source Encoding
        s1 = F.elu(self.en_src_conv1(src_x, src_edge))
        s2 = self.en_src_conv2(s1, src_edge, attention=False)

        # Tie weights for source reconstruction
        self.de_src_conv1.lin_src.data = self.en_src_conv2.lin_src.transpose(0, 1)
        self.de_src_conv1.lin_dst.data = self.en_src_conv2.lin_dst.transpose(0, 1)
        self.de_src_conv2.lin_src.data = self.en_src_conv1.lin_src.transpose(0, 1)
        self.de_src_conv2.lin_dst.data = self.en_src_conv1.lin_dst.transpose(0, 1)
        
        s_recon_h = F.elu(self.de_src_conv1(s2, src_edge, attention=True, tied_attention=self.en_src_conv1.attentions))
        s_recon = self.de_src_conv2(s_recon_h, src_edge, attention=False)

        # 2. Target Encoding
        t1 = F.elu(self.en_tgt_conv1(tgt_x, tgt_edge))
        t2 = self.en_tgt_conv2(t1, tgt_edge, attention=False)

        # Tie weights for target reconstruction
        self.de_tgt_conv1.lin_src.data = self.en_tgt_conv2.lin_src.transpose(0, 1)
        self.de_tgt_conv1.lin_dst.data = self.en_tgt_conv2.lin_dst.transpose(0, 1)
        self.de_tgt_conv2.lin_src.data = self.en_tgt_conv1.lin_src.transpose(0, 1)
        self.de_tgt_conv2.lin_dst.data = self.en_tgt_conv1.lin_dst.transpose(0, 1)

        t_recon_h = F.elu(self.de_tgt_conv1(t2, tgt_edge, attention=True, tied_attention=self.en_tgt_conv1.attentions))
        t_recon = self.de_tgt_conv2(t_recon_h, tgt_edge, attention=False)

        # 3. Covet Similarity Tensors
        # Source features with top-k neighbors
        s2_tensor = torch.zeros((src_x.shape[0], top_k_indices.shape[1] + 1, s2.shape[1]), device=s2.device)
        for i in range(s2.shape[0]):
            s2_tensor[i] = torch.cat((s2[i].unsqueeze(0), s2[non_missing_indices[top_k_indices[i]]]), dim=0)

        # Target features with token for missing parts and top-k neighbors
        t2_tensor = torch.zeros((tgt_x.shape[0], top_k_indices.shape[1] + 1, s2.shape[1]), device=t2.device)
        for i in range(t2.shape[0]):
            t2_tensor[i] = torch.cat((self.token.unsqueeze(0), t2[non_missing_indices[top_k_indices[i]]]), dim=0)

        # 4. Transformer Fusion
        fusion_input = torch.cat([s2_tensor, t2_tensor], dim=2)
        all_outputs = []
        for i in range(0, fusion_input.size(0), self.batch_size):
            batch_out = self.transformer(fusion_input[i : i + self.batch_size])
            all_outputs.append(batch_out)
        
        fused_cat = torch.cat(all_outputs, dim=0).mean(1)
        
        # Branch projection
        tf_src = self.proj_src(fused_cat)
        tf_tgt = self.proj_tgt(fused_cat)

        # 5. Interaction Features (PCA-based)
        # Using detach().cpu() for NumPy compatibility, then moving back to GPU
        s_np = tf_src.detach().cpu().numpy()
        t_np = tf_tgt.detach().cpu().numpy()
        
        # Batch-wise outer product to avoid memory issues and speed up
        # s_np: [N, D1], t_np: [N, D2] -> [N, D1*D2]
        int_features = (s_np[:, :, None] * t_np[:, None, :]).reshape(s_np.shape[0], -1)
        
        int_pca = self.pca.fit_transform(int_features)
        int_pca_tensor = torch.tensor(int_pca, device=src_x.device, dtype=torch.float32)

        # Final Embeddings
        emb = torch.cat((tf_src, tf_tgt), dim=-1)
        prism_emb = torch.cat((tf_src, tf_tgt, int_pca_tensor), dim=-1)

        # 6. Prediction Heads
        s_pred_h = F.elu(self.de_src_pre1(tf_src, src_edge, attention=True, tied_attention=self.en_src_conv1.attentions))
        src_pred = self.de_src_pre2(s_pred_h, src_edge, attention=False)

        t_pred_h = F.elu(self.de_tgt_pre1(tf_tgt, tgt_edge))
        tgt_pred = self.de_tgt_pre2(t_pred_h, tgt_edge, attention=False)

        return s_recon, t_recon, emb, prism_emb, src_pred, tgt_pred