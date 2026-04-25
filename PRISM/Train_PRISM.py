import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import os
import random
import matplotlib.pyplot as plt

# Import custom modules
from .PRISM import PRISM
from .utils import Transfer_pytorch_Data

def plot_loss_curve(loss_history, save_path):
    """
    Visualize and save the training loss curve with a unique name.
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(loss_history['total']) + 1)
    
    plt.plot(epochs, loss_history['total'], label='Total Loss', color='black', linewidth=2)
    plt.plot(epochs, loss_history['src_recon'], label='Source Recon Loss', linestyle='--')
    plt.plot(epochs, loss_history['tgt_recon'], label='Target Recon Loss', linestyle='--')
    plt.plot(epochs, loss_history['src_pred'], label='Source Pred Loss', linestyle=':')
    plt.plot(epochs, loss_history['tgt_pred'], label='Target Pred Loss', linestyle=':')
    
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Log Scale)')
    plt.title(f'PRISM Training Loss Curve ({os.path.basename(save_path)})')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() # Close to free up memory

def prepare_similarity_subset(distance_matrix, missing_indices, non_missing_indices, k_top):
    """Prepare subsets of the distance matrix for similarity-based prediction."""
    missing_indices = torch.tensor(missing_indices, device=distance_matrix.device)
    non_missing_indices = torch.tensor(non_missing_indices, device=distance_matrix.device)
    valid_distances = distance_matrix[:, non_missing_indices]
    top_k_indices = torch.argsort(valid_distances, dim=1)[:, :k_top]
    # return valid_distances, top_k_indices, non_missing_indices
    return top_k_indices, non_missing_indices

def train_PRISM(adata_source, adata_target, distance_matrix, k_top=5, hidden_dims=[512, 32], 
                n_epochs=1000, lr=0.001, key_added='PRISM', gradient_clipping=5., 
                weight_decay=0.0001, verbose=True, random_seed=2024,
                save_loss=True, save_reconstruction=False, 
                output_dir="Default", file_prefix="Default",
                device=None, patience=50, min_epochs=200):
    """
    Train PRISM with automated file prefixing and directory management.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")

    # Set random seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    adata_target = adata_target.copy()
    adata_source = adata_source.copy()

    # Data handling
    if not isinstance(distance_matrix, torch.Tensor):
        distance_matrix = torch.tensor(distance_matrix, device=device, dtype=torch.float32)
    else:
        distance_matrix = distance_matrix.to(device)

    adata_source.X = sp.csr_matrix(adata_source.X)
    adata_target.X = sp.csr_matrix(adata_target.X)

    # Impute missing indicators
    missing_indices = np.where(adata_target.obs['missing'] == '0')[0]
    non_missing_indices = np.where(adata_target.obs['missing'] == '1')[0]
    adata_target.X[missing_indices, :] = 0

    adata_src_vars = adata_source[:, adata_source.var['highly_variable']] if 'highly_variable' in adata_source.var.columns else adata_source
    adata_tgt_vars = adata_target[:, adata_target.var['highly_variable']] if 'highly_variable' in adata_target.var.columns else adata_target

    source_data = Transfer_pytorch_Data(adata_src_vars).to(device)
    target_data = Transfer_pytorch_Data(adata_tgt_vars).to(device)
    mask_target = torch.tensor(adata_target.obsm['protein_mask'], device=device)

    top_k_indices, non_missing_indices = prepare_similarity_subset(
        distance_matrix, missing_indices, non_missing_indices, k_top
    )

    # Initialize Model
    model = PRISM(
        src_hidden_dims=[source_data.x.shape[1]] + hidden_dims,
        tgt_hidden_dims=[target_data.x.shape[1]] + hidden_dims,
        src_out=source_data.x.shape[1],
        tgt_out=target_data.x.shape[1]
    ).to(device)

    # Learnable multi-task loss weights
    initial_weights = torch.ones(4, device=device)
    w_params = torch.nn.Parameter(initial_weights.clone())

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': [w_params], 'lr': lr * 0.1}
    ], lr=lr, weight_decay=weight_decay)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )

    loss_history = {'total': [], 'src_recon': [], 'tgt_recon': [], 'src_pred': [], 'tgt_pred': []}
    best_loss = float('inf')
    stop_counter = 0
    best_model_state, best_outs = None, None
    target_labels = target_data.x

    # --- Training Loop ---
    pbar = tqdm(range(1, n_epochs + 1), desc=f"PRISM [{file_prefix}]")
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        # Forward pass
        s_rec, t_rec, emb, p_emb, s_pre, t_pre = model(
            source_data.x, source_data.edge_index, 
            target_data.x, target_data.edge_index,
            #mask=mask_target, valid_distances=valid_distances, 
            top_k_indices=top_k_indices, non_missing_indices=non_missing_indices
        )

        # Losses
        l_s_rec = F.mse_loss(source_data.x, s_rec)
        l_t_rec = (F.mse_loss(target_labels, t_rec, reduction='none') * mask_target).sum() / mask_target.sum()
        l_s_pre = F.mse_loss(source_data.x, s_pre)
        l_t_pre = (F.mse_loss(target_labels, t_pre, reduction='none') * mask_target).sum() / mask_target.sum()

        norm_w = torch.exp(w_params) / torch.sum(torch.exp(w_params))
        total_loss = norm_w[0]*l_s_rec + norm_w[1]*l_t_rec + norm_w[2]*l_s_pre + norm_w[3]*l_t_pre

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()
        
        current_loss = total_loss.item()
        scheduler.step(current_loss)

        # History tracking
        loss_history['total'].append(current_loss)
        loss_history['src_recon'].append(l_s_rec.item())
        loss_history['tgt_recon'].append(l_t_rec.item())
        loss_history['src_pred'].append(l_s_pre.item())
        loss_history['tgt_pred'].append(l_t_pre.item())

        if current_loss < best_loss:
            best_loss = current_loss
            best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_outs = (emb.detach(), p_emb.detach(), s_pre.detach(), t_pre.detach())
            stop_counter = 0
        else:
            if epoch >= min_epochs:
                stop_counter += 1

        if stop_counter >= patience:
            if verbose:
                print(f"\n>>> Early stopping at epoch {epoch}.")
            break

    # ---------- save ----------
    plot_loss_curve(loss_history, os.path.join(output_dir, f"{file_prefix}_loss_curve.png"))

    if best_model_state is not None:
        # copy to CPU once
        best_model_state_cpu = {k: v.cpu() for k, v in best_model_state.items()}
        model_name = os.path.join(output_dir, f"{file_prefix}_model.pth")
        torch.save({'state_dict': best_model_state_cpu, 'loss': best_loss}, model_name)

        emb, p_emb, s_pre, t_pre = best_outs
        save_results(
            adata_source, adata_target, emb, p_emb, s_pre, t_pre,
            best_loss, save_loss, save_reconstruction,
            output_dir, file_prefix
        )

    return adata_source, adata_target

def save_results(adata_src, adata_tgt, emb, p_emb, s_pre, t_pre, loss, save_loss, save_recon, output_dir, file_prefix):
    """Save results with unique file names based on the dataset prefix."""
    emb_np = p_emb.cpu().numpy()
    
    # Update AnnData
    adata_src.obsm['PRISM_emb'] = emb_np
    adata_tgt.obsm['PRISM_emb'] = emb_np

    # Save CSVs
    target_csv = os.path.join(output_dir, f"{file_prefix}_pre.csv")
    embed_csv = os.path.join(output_dir, f"{file_prefix}_emb.csv")
    
    pd.DataFrame(t_pre.cpu().numpy(), index=adata_tgt.obs_names, columns=adata_tgt.var_names).to_csv(target_csv)
    pd.DataFrame(emb_np, index=adata_tgt.obs_names).to_csv(embed_csv)

    print(f"Results for '{file_prefix}' saved: {target_csv}, {embed_csv}")

    if save_loss: adata_src.uns['PRISM_loss'] = float(loss)
    if save_recon: adata_src.layers['PRISM_recon'] = emb_np.clip(min=0)