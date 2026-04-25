# PRISM/Validation.py
import numpy as np
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Union

# 1 compute_metrics for task1 ('domain identification')
from sklearn.metrics import adjusted_mutual_info_score, v_measure_score, mutual_info_score, homogeneity_score, normalized_mutual_info_score, adjusted_rand_score

def evaluate(true_labels, pred_labels):
    ami = adjusted_mutual_info_score(true_labels, pred_labels)
    v_measure = v_measure_score(true_labels, pred_labels)
    mutual_info = mutual_info_score(true_labels, pred_labels)
    homogeneity = homogeneity_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    return ami, v_measure, mutual_info, homogeneity, nmi, ari


# 2 compute_metrics_each_pair for task2('spatial omics imptutaion')
def compute_metrics_each_pair(adata_omics2, distance_matrix, top_n=10, verbose=False):
    """
    For each spot i:
      - Find the top_n most similar spots (excluding itself)
      - Record the distances to these similar spots
      - Calculate RMSE/PCC/SPCC with the protein expression vectors of these similar spots
      - Return the mean(RMSE/PCC/SPCC) + details (including distances) for each spot
    
    Parameters:
    adata_omics2 : AnnData object containing the protein expression data
    distance_matrix : scipy sparse or dense matrix representing the pairwise distance between spots
    top_n : Number of most similar spots to consider for each spot (default is 10)
    verbose : Flag to print error messages if any issues occur during PCC/SPCC calculation (default is False)
    
    Returns:
    rmse_values_all : Array of RMSE mean values for each spot
    pcc_values_all : Array of Pearson correlation coefficients for each spot
    spcc_values_all : Array of Spearman correlation coefficients for each spot
    details : List of dictionaries containing detailed metrics and similar spots for each spot
    """
    # Extract the protein expression data (dense array format)
    Xp = adata_omics2.X
    protein_data = Xp.toarray() if sp.issparse(Xp) else np.asarray(Xp)

    # Ensure the dimensions of the distance matrix match the number of spots in adata_omics2
    n_spots = protein_data.shape[0]
    if distance_matrix.shape[0] != n_spots:
        raise ValueError(f"Distance matrix size mismatch: distance_matrix.shape[0]={distance_matrix.shape[0]} "
                         f"does not match adata_omics2.n_obs={n_spots}")

    # Initialize lists to store metrics for all spots
    rmse_values_all, pcc_values_all, spcc_values_all = [], [], []
    details = []

    is_sparse = sp.issparse(distance_matrix)  # Check if the distance matrix is sparse

    # Loop over each spot in the dataset
    for i in range(n_spots):
        # -------- Find top_n most similar spots (excluding itself) --------
        if is_sparse:
            # If the distance matrix is sparse, get the row corresponding to the current spot
            row = distance_matrix.getrow(i)
            nbr_idx = row.indices  # Indices of neighbor spots
            nbr_dist = row.data  # Distances to neighbor spots

            # Remove self (self will be included as the smallest distance)
            mask = (nbr_idx != i)
            nbr_idx = nbr_idx[mask]
            nbr_dist = nbr_dist[mask]

            if nbr_idx.size == 0:
                similar_spots = np.array([], dtype=int)
                similar_dists = np.array([], dtype=np.float32)
            else:
                # Sort the neighbors by distance (ascending), and select the top_n
                order = np.argsort(nbr_dist)  # Sort by increasing distance
                top = order[:top_n]
                similar_spots = nbr_idx[top]
                similar_dists = nbr_dist[top].astype(np.float32)

        else:
            # If the distance matrix is dense, sort the entire row
            order = np.argsort(distance_matrix[i, :])
            order = order[order != i]  # Exclude itself
            similar_spots = order[:top_n]
            similar_dists = distance_matrix[i, similar_spots].astype(np.float32)

        # -------- Calculate RMSE/PCC/SPCC for each of the similar spots --------
        spot_expr = protein_data[i, :]  # Protein expression vector for the current spot
        rmse_list, pcc_list, spcc_list = [], [], []  # Lists to store metrics for each similar spot

        for j in similar_spots:
            nbr_expr = protein_data[j, :]  # Protein expression vector for the neighbor spot

            # Calculate RMSE (Root Mean Squared Error) between the current spot and the neighbor
            rmse = np.sqrt(mean_squared_error(spot_expr, nbr_expr))
            rmse_list.append(rmse)

            # Initialize PCC and SPCC with NaN in case of errors
            pcc = np.nan
            spcc_v = np.nan
            try:
                if np.std(spot_expr) > 0 and np.std(nbr_expr) > 0:
                    # Compute Pearson Correlation Coefficient (PCC)
                    pcc, _ = pearsonr(spot_expr, nbr_expr)
                    # Compute Spearman Rank Correlation Coefficient (SPCC)
                    spcc_v, _ = spearmanr(spot_expr, nbr_expr)
            except Exception as e:
                if verbose:
                    print(f"Error in PCC/SPCC calculation for spot {i} and spot {j}: {e}")

            pcc_list.append(pcc)
            spcc_list.append(spcc_v)

        # Compute mean metrics for the current spot
        rmse_mean = np.nanmean(rmse_list) if len(rmse_list) else np.nan
        pcc_mean = np.nanmean(pcc_list) if len(pcc_list) else np.nan
        spcc_mean = np.nanmean(spcc_list) if len(spcc_list) else np.nan

        # Append the mean metrics to the overall lists
        rmse_values_all.append(rmse_mean)
        pcc_values_all.append(pcc_mean)
        spcc_values_all.append(spcc_mean)

        # Store detailed metrics for the current spot
        details.append({
            "spot_index": i,
            "similar_spots": similar_spots,
            "similar_dists": similar_dists,  # Top_n neighbors and their distances
            "rmse_values": rmse_list,
            "pcc_values": pcc_list,
            "spcc_values": spcc_list,
            "rmse_mean": rmse_mean,
            "pcc_mean": pcc_mean,
            "spcc_mean": spcc_mean,
        })

    # Convert final metric lists to numpy arrays and return the results
    return (np.array(rmse_values_all), np.array(pcc_values_all), np.array(spcc_values_all), details)



#--------------------------------------------------#

def evaluate_protein_prediction(protein_true, protein_pred, missing_indices):
    """
    Calculate evaluation metrics for protein prediction in missing parts: RMSE, PCC, and SPCC.
    
    Parameters:
    -----------
    protein_true : np.ndarray
        Ground truth protein data of shape (N_cells x N_proteins).
    protein_pred : np.ndarray
        Predicted protein data of shape (N_cells x N_proteins).
    missing_indices : list or np.ndarray
        Row indices corresponding to the missing data points.
    
    Returns:
    --------
    metrics : dict
        A dictionary containing overall RMSE, PCC, SPCC, and detailed per-protein metrics.
    """
    
    # Dimension validation
    if protein_true.shape != protein_pred.shape:
        raise ValueError(f"Dimension mismatch: True {protein_true.shape}, Pred {protein_pred.shape}")

    # Extract subset data for the missing parts
    y_true = protein_true[missing_indices, :]
    y_pred = protein_pred[missing_indices, :]
    num_proteins = y_true.shape[1]

    # Calculate RMSE for each protein: RMSE = sqrt(mean((y_true - y_pred)^2))
    rmse_per_protein = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    
    pcc_per_protein = []
    spcc_per_protein = []
    
    for i in range(num_proteins):
        # Handle constant input to avoid NaN results or errors in correlation calculations
        if np.std(y_true[:, i]) == 0 or np.std(y_pred[:, i]) == 0:
            pcc_per_protein.append(0.0)
            spcc_per_protein.append(0.0)
        else:
            # Calculate Pearson Correlation Coefficient (PCC)
            pcc, _ = pearsonr(y_true[:, i], y_pred[:, i])
            # Calculate Spearman Rank Correlation Coefficient (SPCC)
            spcc, _ = spearmanr(y_true[:, i], y_pred[:, i])
            pcc_per_protein.append(pcc)
            spcc_per_protein.append(spcc)

    # Calculate overall mean values across all proteins
    #overall_rmse = np.mean(rmse_per_protein)
    overall_pcc = np.mean(pcc_per_protein)
    overall_spcc = np.mean(spcc_per_protein)

    # Organize results into a structured dictionary
    results = {
        'overall': {
            #'RMSE': overall_rmse,
            'PCC': overall_pcc,
            'SPCC': overall_spcc
        },
        'per_protein': {
            #'RMSE': rmse_per_protein,
            'PCC': pcc_per_protein,
            'SPCC': spcc_per_protein
        }
    }
    
    return results
#--------------------------------------------------#


#3 Unified evaluation and saving function for task2 ("spatial omics imputation")

# =========================
# PRISM/Validation.py  (APPEND BELOW YOUR EXISTING CODE)
# - Requires existing: evaluate_protein_prediction(true, pred, missing_indices)
# =========================


def _topk_feature_view(metrics_dict, var_names, k=800, rank_by="PCC"):
    """
    From evaluate_protein_prediction() output, pick top-k features by rank_by (PCC/SPCC),
    then recompute overall PCC/SPCC using only those features.

    Returns a dict:
      {
        "overall": {"PCC":..., "SPCC":...},
        "topk_idx": np.ndarray,
        "topk_names": list[str],
        "k": int,
        "rank_by": str
      }
    """
    if rank_by not in ("PCC", "SPCC"):
        raise ValueError("rank_by must be 'PCC' or 'SPCC'.")

    scores = np.asarray(metrics_dict["per_protein"][rank_by], dtype=np.float32)

    # rank: NaN -> -inf so they go to the end
    scores_rank = np.where(np.isfinite(scores), scores, -np.inf)

    k = int(min(k, scores_rank.size))
    if k <= 0:
        raise ValueError("k must be > 0.")

    topk_idx = np.argsort(scores_rank)[::-1][:k]  # descending

    pcc_all = np.asarray(metrics_dict["per_protein"]["PCC"], dtype=np.float32)
    spcc_all = np.asarray(metrics_dict["per_protein"]["SPCC"], dtype=np.float32)

    topk_overall_pcc = float(np.nanmean(pcc_all[topk_idx]))
    topk_overall_spcc = float(np.nanmean(spcc_all[topk_idx]))

    topk_names = list(np.asarray(var_names, dtype=str)[topk_idx])

    return {
        "overall": {"PCC": topk_overall_pcc, "SPCC": topk_overall_spcc},
        "topk_idx": topk_idx,
        "topk_names": topk_names,
        "k": k,
        "rank_by": rank_by,
    }

def _to_dense_f32(X):
    """Convert sparse/dense matrix to dense float32 numpy array."""
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _inverse_log1p_norm_to_raw(pred_log, raw_total_counts, target_sum=1e4):
    """
    Inverse of: normalize_total(target_sum) + log1p
    pred_log: (n, d) in log1p(normalized) space
    raw_total_counts: (n,) from raw counts
    """
    pred_norm = np.expm1(pred_log)  # inverse log1p -> normalized counts
    scale = (raw_total_counts.astype(np.float32) / float(target_sum))[:, None]
    return pred_norm * scale


def prism_eval_and_save(
    adata,
    save_path: str,
    first_name: str,
    missing_indices,
    *,
    target_sum: float = 1e4,
    space: str = "both",        # "processed" | "raw" | "both"
    metric: str = "both",       # "PCC" | "SPCC" | "both"
    save_files: bool = True,
    save_raw_imputed_all: bool = False,
    processed_true_key: str = "true_data",
    raw_layer_key: str = "raw_data",
    raw_total_key: str = "raw_total_counts",
    strict_shape_check: bool = True,
    verbose: bool = True,

    # Top-K option
    topk_features: Optional[int] = None,     # set None to disable
    topk_rank_by: str = "auto",             # "auto" | "PCC" | "SPCC"
    topk_space: str = "auto",               # "auto" | "processed" | "raw" | "both"
    topk_only: bool = True,                 # True: overwrite overall with topk overall

    # keep TopK computation but DO NOT save summary file
    save_topk_summary: bool = False,        # default False per your request
):
    """
    Unified evaluator for both protein/RNA tasks.

    New behaviors:
      1) TopK is capped by number of imputable features: k_eff = min(topk_features, adata.n_vars)
      2) If TopK is enabled (topk_features is not None) and save_files=True:
           - save TopK missing true/pred CSVs (only TopK columns)
           - (optional) save TopK summary CSV (disabled by default via save_topk_summary=False)
    """
    if space not in ("processed", "raw", "both"):
        raise ValueError("space must be 'processed', 'raw', or 'both'.")
    if metric not in ("PCC", "SPCC", "both"):
        raise ValueError("metric must be 'PCC', 'SPCC', or 'both'.")

    miss = np.asarray(missing_indices, dtype=int)

    # --- resolve topk_space ---
    eff_topk_space = topk_space
    if eff_topk_space == "auto":
        eff_topk_space = space
    if eff_topk_space not in ("processed", "raw", "both"):
        raise ValueError("topk_space must be 'auto', 'processed', 'raw', or 'both'.")

    # --- resolve topk_rank_by ---
    eff_topk_rank_by = topk_rank_by
    if eff_topk_rank_by == "auto":
        if metric == "SPCC":
            eff_topk_rank_by = "SPCC"
        else:
            eff_topk_rank_by = "PCC"
    if eff_topk_rank_by not in ("PCC", "SPCC"):
        raise ValueError("topk_rank_by must be 'auto', 'PCC', or 'SPCC'.")

    # -----------------------
    # load prediction (processed space)
    # -----------------------
    pred_path = os.path.join(save_path, first_name + "_pre.csv")
    pred_df = pd.read_csv(pred_path, index_col=0)
    pred = pred_df.values.astype(np.float32)  # STRICT: no reindex/align

    n = adata.n_obs
    d = adata.n_vars
    if strict_shape_check and pred.shape != (n, d):
        raise ValueError(
            f"Prediction shape {pred.shape} != (adata.n_obs, adata.n_vars)=({n},{d}). "
            f"To keep strict ordering, do not reindex. Please ensure PRISM output matches adata."
        )

    # -----------------------
    # TopK cap (Requirement #1)
    # -----------------------
    topk_k_eff = None
    if topk_features is not None:
        topk_k_eff = int(topk_features)
        if topk_k_eff <= 0:
            topk_k_eff = None
        else:
            topk_k_eff = min(topk_k_eff, d)  # cap by imputable feature count

    results = {"processed": None, "raw": None, "saved": {}}
    var_names = adata.var.index
    miss_obs_names = adata.obs.index[miss]

    # helper: write TopK files (NO summary by default)
    def _save_topk_files(tag: str, true_mat: np.ndarray, pred_mat: np.ndarray, metrics_dict: dict, topk_view: dict):
        k = int(topk_view["k"])
        idx = np.asarray(topk_view["topk_idx"], dtype=int)
        names = np.asarray(var_names, dtype=str)[idx]

        # ---- optional summary csv (disabled by default) ----
        if save_topk_summary:
            pcc_all = np.asarray(metrics_dict["per_protein"]["PCC"], dtype=np.float32)[idx]
            spcc_all = np.asarray(metrics_dict["per_protein"]["SPCC"], dtype=np.float32)[idx]
            rank_scores = np.asarray(metrics_dict["per_protein"][topk_view["rank_by"]], dtype=np.float32)[idx]

            summary_df = pd.DataFrame({
                "feature": names,
                "rank_by": topk_view["rank_by"],
                "rank_score": rank_scores,
                "PCC": pcc_all,
                "SPCC": spcc_all,
            }).sort_values("rank_score", ascending=False)

            summary_csv = os.path.join(save_path, f"{first_name}_topk_{tag}_{topk_view['rank_by']}_k{k}.csv")
            summary_df.to_csv(summary_csv, index=False)
            results["saved"][f"topk_{tag}_summary"] = summary_csv

        # ---- topk missing true/pred csv (what you actually want) ----
        true_topk_missing = true_mat[np.ix_(miss, idx)]
        pred_topk_missing = pred_mat[np.ix_(miss, idx)]

        true_topk_df = pd.DataFrame(true_topk_missing, index=miss_obs_names, columns=names)
        pred_topk_df = pd.DataFrame(pred_topk_missing, index=miss_obs_names, columns=names)

        true_topk_csv = os.path.join(save_path, f"{first_name}_unreg_{tag}_true_topk{k}.csv")
        pred_topk_csv = os.path.join(save_path, f"{first_name}_unreg_{tag}_pred_topk{k}.csv")

        true_topk_df.to_csv(true_topk_csv)
        pred_topk_df.to_csv(pred_topk_csv)

        results["saved"][f"topk_{tag}_true_missing"] = true_topk_csv
        results["saved"][f"topk_{tag}_pred_missing"] = pred_topk_csv

    # -----------------------
    # Processed-space
    # -----------------------
    if space in ("processed", "both"):
        true_proc = adata.uns.get(processed_true_key, None)
        if true_proc is None:
            raise KeyError(f"adata.uns['{processed_true_key}'] not found.")
        true_proc = _to_dense_f32(true_proc)

        metrics_proc = evaluate_protein_prediction(true_proc, pred, miss)
        results["processed"] = metrics_proc

        # Top-K on processed
        if topk_k_eff is not None and eff_topk_space in ("processed", "both"):
            topk_view = _topk_feature_view(metrics_proc, var_names, k=topk_k_eff, rank_by=eff_topk_rank_by)
            results["processed_topk"] = topk_view
            if topk_only:
                results["processed"]["overall"]["PCC"] = topk_view["overall"]["PCC"]
                results["processed"]["overall"]["SPCC"] = topk_view["overall"]["SPCC"]

        if verbose:
            print("===== Processed-space Evaluation Results =====")
            if metric in ("PCC", "both"):
                print(f"PCC : {results['processed']['overall']['PCC']:.4f}")
            if metric in ("SPCC", "both"):
                print(f"SPCC: {results['processed']['overall']['SPCC']:.4f}")
            if topk_k_eff is not None and eff_topk_space in ("processed", "both") and "processed_topk" in results:
                tv = results["processed_topk"]
                print(f"[Top-{tv['k']} by {tv['rank_by']}]")

        # Save processed (base)
        if save_files:
            processed_true_missing_df = pd.DataFrame(true_proc[miss, :], index=miss_obs_names, columns=var_names)
            processed_pred_missing_df = pd.DataFrame(pred[miss, :], index=miss_obs_names, columns=var_names)

            processed_true_csv = os.path.join(save_path, first_name + "_unreg_processed_true.csv")
            processed_pred_csv = os.path.join(save_path, first_name + "_unreg_processed_pre.csv")

            processed_true_missing_df.to_csv(processed_true_csv)
            processed_pred_missing_df.to_csv(processed_pred_csv)

            results["saved"]["processed_true_missing"] = processed_true_csv
            results["saved"]["processed_pred_missing"] = processed_pred_csv

            # Save processed TopK (no summary by default)
            if topk_k_eff is not None and eff_topk_space in ("processed", "both") and "processed_topk" in results:
                _save_topk_files("processed", true_proc, pred, metrics_proc, results["processed_topk"])

    # -----------------------
    # RAW-space
    # -----------------------
    if space in ("raw", "both"):
        raw_true = adata.layers.get(raw_layer_key, None)
        if raw_true is None:
            raise KeyError(f"adata.layers['{raw_layer_key}'] not found. Please save raw before normalize/log1p.")
        raw_true = _to_dense_f32(raw_true)

        if raw_total_key not in adata.obs.columns:
            raise KeyError(f"adata.obs['{raw_total_key}'] not found. Please save raw_total_counts before normalize/log1p.")
        raw_total_counts = np.clip(adata.obs[raw_total_key].to_numpy().astype(np.float32), 1.0, None)

        pred_raw = _inverse_log1p_norm_to_raw(pred, raw_total_counts, target_sum=target_sum)

        metrics_raw = evaluate_protein_prediction(raw_true, pred_raw, miss)
        results["raw"] = metrics_raw

        # Top-K on raw
        if topk_k_eff is not None and eff_topk_space in ("raw", "both"):
            topk_view = _topk_feature_view(metrics_raw, var_names, k=topk_k_eff, rank_by=eff_topk_rank_by)
            results["raw_topk"] = topk_view
            if topk_only:
                results["raw"]["overall"]["PCC"] = topk_view["overall"]["PCC"]
                results["raw"]["overall"]["SPCC"] = topk_view["overall"]["SPCC"]

        if verbose:
            print("\n===== RAW (unprocessed counts) Evaluation Results =====")
            if metric in ("PCC", "both"):
                print(f"PCC : {results['raw']['overall']['PCC']:.4f}")
            if metric in ("SPCC", "both"):
                print(f"SPCC: {results['raw']['overall']['SPCC']:.4f}")
            if topk_k_eff is not None and eff_topk_space in ("raw", "both") and "raw_topk" in results:
                tv = results["raw_topk"]
                print(f"[Top-{tv['k']} by {tv['rank_by']}]")

        # Save raw (base)
        if save_files:
            raw_true_missing_df = pd.DataFrame(raw_true[miss, :], index=miss_obs_names, columns=var_names)
            raw_pred_missing_df = pd.DataFrame(pred_raw[miss, :], index=miss_obs_names, columns=var_names)

            raw_true_csv = os.path.join(save_path, first_name + "_unreg_raw_true.csv")
            raw_pred_csv = os.path.join(save_path, first_name + "_unreg_raw_pred.csv")

            raw_true_missing_df.to_csv(raw_true_csv)
            raw_pred_missing_df.to_csv(raw_pred_csv)

            results["saved"]["raw_true_missing"] = raw_true_csv
            results["saved"]["raw_pred_missing"] = raw_pred_csv

            if save_raw_imputed_all:
                raw_imputed = raw_true.copy()
                raw_imputed[miss, :] = pred_raw[miss, :]
                raw_imputed_all_csv = os.path.join(save_path, first_name + "_raw_imputed_all.csv")
                pd.DataFrame(raw_imputed, index=adata.obs.index, columns=var_names).to_csv(raw_imputed_all_csv)
                results["saved"]["raw_imputed_all"] = raw_imputed_all_csv

            # Save raw TopK (no summary by default)
            if topk_k_eff is not None and eff_topk_space in ("raw", "both") and "raw_topk" in results:
                _save_topk_files("raw", raw_true, pred_raw, metrics_raw, results["raw_topk"])

    if verbose and save_files and len(results["saved"]) > 0:
        print("\nSaved files:")
        for k, p in results["saved"].items():
            print(f" - {k}: {p}")

    return results

# 4  Representative visualization of the imputation object
from typing import Optional, Union, Tuple

def plot_prism_imputation_spatial(
    dataset_file: str,
    save_path: str,
    first_name: str,
    split1_indices,
    *,
    h5ad_name: str = "adata_ADT.h5ad",
    feature: Union[int, str] = 22,             # int index OR feature name (e.g., "HLA-DRA")
    plot_space: str = "raw",                   # "raw" | "processed"
    adata_processed=None,                      # required for processed true (and for raw inverse scaling)
    processed_true_key: str = "true_data",     # adata_processed.uns[processed_true_key]
    target_sum: float = 1e4,                   # must match normalize_total target_sum
    # marker / style
    point_size: float = 15,
    highlight_missing: bool = False,
    highlight_color: str = "red",
    highlight_linewidth: float = 0.2,
    # colorbar position: [left, bottom, width, height] in figure coords (0~1)
    cbar_pos=(0.92, 0.05, 0.025, 0.8),
    cbar_ticksize: int = 12,
    figsize=(6, 5),
    dpi: int = 300,
    cmap_name: str = "viridis",
    title_true: Optional[str] = None,
    title_pred: Optional[str] = None,
    verbose: bool = False,
    clip_quantiles: Tuple[float, float] = (0.02, 0.98),
):
    """
    Plot True vs PRISM prediction in one row with ONE shared colorbar.
    Always plots the FULL slice (no missing-only mode).

    Prediction source:
      - processed: {first_name}_pre.csv (full)
      - raw:
          1) if exists: {first_name}_raw_pred_full.csv (full raw)
          2) else: inverse-transform {first_name}_pre.csv -> raw using adata_processed.obs['raw_total_counts']

    True source:
      - processed: adata_processed.uns[processed_true_key] (recommended), fallback to adata_processed.X
      - raw: adata_vis.X (raw h5ad matrix)

    NOTE: For plot_space='processed' you MUST pass adata_processed.
          For plot_space='raw' we also recommend passing adata_processed to enable inverse scaling when needed.
    """
    import os
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sp
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    # -----------------------
    # 1) Load reference h5ad (raw true + spatial coords)
    # -----------------------
    adata_vis = sc.read_h5ad(os.path.join(dataset_file, h5ad_name))
    adata_vis.var_names_make_unique()

    if "spatial" not in adata_vis.obsm:
        raise KeyError("adata_vis.obsm['spatial'] not found.")

    coords = np.asarray(adata_vis.obsm["spatial"])
    x = coords[:, 0]
    y = coords[:, 1]

    X_raw_true = adata_vis.X.toarray() if sp.issparse(adata_vis.X) else np.asarray(adata_vis.X)
    X_raw_true = X_raw_true.astype(np.float32)

    # -----------------------
    # 2) Resolve feature index/name
    # -----------------------
    def _resolve_feature(adata, feat):
        if isinstance(feat, int):
            idx = int(feat)
            if idx < 0 or idx >= adata.n_vars:
                raise IndexError(f"feature index {idx} out of range (0..{adata.n_vars-1}).")
            return idx, str(adata.var_names[idx])

        if isinstance(feat, str):
            if feat in adata.var_names:
                idx = int(np.where(adata.var_names == feat)[0][0])
                return idx, feat
            lower_map = {str(v).lower(): i for i, v in enumerate(adata.var_names)}
            key = feat.strip().lower()
            if key in lower_map:
                idx = int(lower_map[key])
                return idx, str(adata.var_names[idx])
            raise KeyError(f"Feature '{feat}' not found in adata.var_names.")
        raise TypeError("feature must be int or str.")

    feat_idx, feat_name = _resolve_feature(adata_vis, feature)

    # missing indices only for highlighting
    missing_indices = np.asarray(split1_indices, dtype=int)

    # -----------------------
    # 3) Load FULL processed prediction (always)
    # -----------------------
    pre_path = os.path.join(save_path, first_name + "_pre.csv")
    if not os.path.exists(pre_path):
        raise FileNotFoundError(f"Processed prediction not found: {pre_path}")

    pred_proc_df = pd.read_csv(pre_path, index_col=0)

    # Fill processed pred into full matrix by name matching (robust)
    X_pred_proc_full = np.full((adata_vis.n_obs, adata_vis.n_vars), np.nan, dtype=np.float32)

    row_pos = adata_vis.obs_names.get_indexer(pred_proc_df.index.astype(str))
    if np.any(row_pos < 0):
        bad = pred_proc_df.index[row_pos < 0][:5].tolist()
        raise ValueError(f"Some predicted rows not found in adata.obs_names, e.g.: {bad}")

    common_features = pred_proc_df.columns.intersection(adata_vis.var_names)
    if len(common_features) == 0:
        raise ValueError("No overlapping features between prediction CSV columns and adata.var_names.")
    col_pos = adata_vis.var_names.get_indexer(common_features)

    X_pred_proc_full[np.ix_(row_pos, col_pos)] = pred_proc_df[common_features].values.astype(np.float32)
    v_pred_processed = X_pred_proc_full[:, feat_idx]

    # -----------------------
    # 4) Decide plot space: true + pred vectors
    # -----------------------
    plot_space_clean = str(plot_space).strip().lower()
    if plot_space_clean not in ("raw", "processed"):
        raise ValueError("plot_space must be 'raw' or 'processed'.")

    # ---- TRUE ----
    if plot_space_clean == "raw":
        v_true = X_raw_true[:, feat_idx]
    else:
        if adata_processed is None:
            raise ValueError("plot_space='processed' requires adata_processed (your in-memory processed AnnData).")

        if feat_name not in adata_processed.var_names:
            raise KeyError(f"Feature '{feat_name}' not found in adata_processed.var_names.")
        feat_idx_proc = int(np.where(adata_processed.var_names == feat_name)[0][0])

        if processed_true_key in getattr(adata_processed, "uns", {}):
            Xp = adata_processed.uns[processed_true_key]
            v_true = (Xp[:, feat_idx_proc].toarray().ravel() if sp.issparse(Xp) else np.asarray(Xp[:, feat_idx_proc]).ravel())
        else:
            Xp = adata_processed.X
            v_true = (Xp[:, feat_idx_proc].toarray().ravel() if sp.issparse(Xp) else np.asarray(Xp[:, feat_idx_proc]).ravel())

        v_true = np.asarray(v_true, dtype=np.float32)

        # align obs if needed
        if not np.array_equal(adata_processed.obs_names, adata_vis.obs_names):
            idx_map = adata_processed.obs_names.get_indexer(adata_vis.obs_names)
            if np.any(idx_map < 0):
                raise ValueError("adata_processed.obs_names and adata_vis.obs_names do not align.")
            v_true = v_true[idx_map]

    # ---- PRED ----
    if plot_space_clean == "processed":
        v_pred = v_pred_processed.astype(np.float32)
    else:
        # raw pred: prefer saved full raw pred; otherwise inverse from processed pre.csv
        raw_full_path = os.path.join(save_path, first_name + "_raw_pred_full.csv")

        if os.path.exists(raw_full_path):
            raw_df = pd.read_csv(raw_full_path, index_col=0)
            # align by obs/var names
            raw_df = raw_df.reindex(index=adata_vis.obs_names, columns=adata_vis.var_names)
            v_pred = raw_df.iloc[:, feat_idx].to_numpy(dtype=np.float32)

        else:
            if adata_processed is None:
                raise ValueError(
                    "plot_space='raw' needs either:\n"
                    "  - saved full raw pred file: {first_name}_raw_pred_full.csv\n"
                    "or\n"
                    "  - adata_processed with obs['raw_total_counts'] to inverse-transform from pre.csv."
                )

            if "raw_total_counts" not in adata_processed.obs.columns:
                raise KeyError("adata_processed.obs['raw_total_counts'] not found (needed for raw inverse scaling).")

            raw_tc = np.clip(
                adata_processed.obs["raw_total_counts"].to_numpy().astype(np.float32),
                a_min=1.0,
                a_max=None
            )

            # align tc if needed
            if not np.array_equal(adata_processed.obs_names, adata_vis.obs_names):
                idx_map = adata_processed.obs_names.get_indexer(adata_vis.obs_names)
                if np.any(idx_map < 0):
                    raise ValueError("adata_processed.obs_names and adata_vis.obs_names do not align.")
                raw_tc = raw_tc[idx_map]

            # inverse: raw = expm1(log_norm) * (raw_total_counts / target_sum)
            v_pred_norm = np.expm1(np.maximum(v_pred_processed, 0)).astype(np.float32)
            v_pred = (v_pred_norm * (raw_tc / float(target_sum))).astype(np.float32)

    # -----------------------
    # 5) Robust color scaling (quantile clip)
    # -----------------------
    combo = np.concatenate([v_true[np.isfinite(v_true)], v_pred[np.isfinite(v_pred)]])
    if combo.size == 0:
        raise ValueError("No finite values to plot.")

    q_low, q_high = clip_quantiles
    vmin = float(np.quantile(combo, q_low))
    vmax = float(np.quantile(combo, q_high))
    if vmax <= vmin:
        vmin = float(combo.min())
        vmax = float(combo.max())
        if vmax <= vmin:
            vmax = vmin + 1.0

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    # -----------------------
    # 6) Plot (always FULL slice)
    # -----------------------
    def draw_panel(ax, values, title):
        ax.scatter(
            x, y,
            c=values,
            cmap=cmap,
            norm=norm,
            s=point_size,
            linewidths=0.0
        )

        if highlight_missing and len(missing_indices) > 0:
            ax.scatter(
                x[missing_indices], y[missing_indices],
                facecolors="none",
                edgecolors=highlight_color,
                s=point_size,
                linewidths=highlight_linewidth
            )

        ax.set_title(title, fontsize=13)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_frame_on(False)

    if title_true is None:
        title_true = f"True ({plot_space_clean}) {feat_name}"
    if title_pred is None:
        title_pred = f"PRISM ({plot_space_clean}) {feat_name}"

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")

    draw_panel(axes[0], v_true, title_true)
    draw_panel(axes[1], v_pred, title_pred)

    plt.subplots_adjust(wspace=0.08, left=0.02, right=0.90, top=0.90, bottom=0.02)

    cax = fig.add_axes(list(cbar_pos))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=cbar_ticksize)

    if verbose:
        print(f"[Plot] space={plot_space_clean}, feature={feat_name} (idx={feat_idx})")
        print(f"[Plot] color scale clip {clip_quantiles}: vmin={vmin:.4g}, vmax={vmax:.4g}")
        if plot_space_clean == "raw" and os.path.exists(os.path.join(save_path, first_name + "_raw_pred_full.csv")):
            print(f"[Plot] using raw full pred file: {first_name}_raw_pred_full.csv")
        else:
            print(f"[Plot] using processed pre.csv: {first_name}_pre.csv")

    plt.show()
    return feat_name




# ==========================================================
# Task2 (For REAL missing): 3-panel visualization
#   (1) unaligned   (raw or processed)
#   (2) aligned     (raw or processed)
#   (3) PRISM imputed (raw or processed)
# ==========================================================

def _get_feature_index(adata, feature):
    """feature can be int or str (exact or case-insensitive). Returns (idx, name)."""
    if isinstance(feature, int):
        if feature < 0 or feature >= adata.n_vars:
            raise IndexError(f"feature index {feature} out of range (0..{adata.n_vars-1}).")
        return int(feature), str(adata.var_names[feature])

    if isinstance(feature, str):
        if feature in adata.var_names:
            idx = int(np.where(adata.var_names == feature)[0][0])
            return idx, feature
        lower_map = {str(v).lower(): i for i, v in enumerate(adata.var_names)}
        key = feature.strip().lower()
        if key in lower_map:
            idx = int(lower_map[key])
            return idx, str(adata.var_names[idx])
        raise KeyError(f"Feature '{feature}' not found in adata.var_names.")
    raise TypeError("feature must be int or str.")


def _get_1d_from_X(adata, feat_idx):
    X = adata.X
    if sp.issparse(X):
        return np.asarray(X[:, feat_idx].toarray()).ravel().astype(np.float32)
    return np.asarray(X[:, feat_idx]).ravel().astype(np.float32)


def _get_1d_from_layer(adata, layer_key, feat_idx):
    X = adata.layers[layer_key]
    if sp.issparse(X):
        return np.asarray(X[:, feat_idx].toarray()).ravel().astype(np.float32)
    return np.asarray(X[:, feat_idx]).ravel().astype(np.float32)


def _estimate_missing_total_counts_knn(
    coords: np.ndarray,
    total_counts: np.ndarray,
    missing_idx: np.ndarray,
    observed_idx: np.ndarray,
    *,
    k: int = 30,
    eps: float = 1e-6,
):
    """
    Estimate missing total_counts via spatial kNN weighted average among observed spots.
    weights = 1/(distance + eps)
    """
    coords = np.asarray(coords, dtype=np.float32)
    tc = np.asarray(total_counts, dtype=np.float32)

    obs_coords = coords[observed_idx]
    miss_coords = coords[missing_idx]

    if obs_coords.shape[0] == 0:
        return np.zeros(len(missing_idx), dtype=np.float32)

    k_eff = min(k, obs_coords.shape[0])
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(obs_coords)
    dist, nbr = nn.kneighbors(miss_coords, return_distance=True)

    w = 1.0 / (dist + eps)                           # (m, k)
    nbr_tc = tc[observed_idx[nbr]]                   # (m, k)
    est = (w * nbr_tc).sum(axis=1) / np.clip(w.sum(axis=1), a_min=eps, a_max=None)
    return est.astype(np.float32)


def _make_raw_total_counts_safe(
    adata_aligned,
    missing_indices: np.ndarray,
    observed_indices: np.ndarray,
    *,
    target_sum: float = 1e4,
    scale_method: str = "knn_tc",   # "median" or "knn_tc"
    knn_k: int = 30,
    knn_eps: float = 1e-6,
    raw_total_key: str = "raw_total_counts",
    raw_layer_key: str = "raw_data",
):
    """
    Return a per-spot total_counts vector for inverse normalize_total.
    Missing spots with tc<=0 will be estimated.
    """
    if raw_total_key in adata_aligned.obs.columns:
        tc = adata_aligned.obs[raw_total_key].to_numpy().astype(np.float32)
    else:
        if raw_layer_key not in adata_aligned.layers:
            raise KeyError(f"Need adata.obs['{raw_total_key}'] or adata.layers['{raw_layer_key}'] to invert to raw.")
        raw_X = adata_aligned.layers[raw_layer_key]
        tc = (np.asarray(raw_X.sum(axis=1)).ravel().astype(np.float32)
              if sp.issparse(raw_X) else np.asarray(raw_X).sum(axis=1).astype(np.float32))

    tc_safe = tc.copy()
    bad = (~np.isfinite(tc_safe)) | (tc_safe <= 0)

    if np.any(bad):
        obs_tc = tc_safe[observed_indices]
        med = float(np.median(obs_tc[obs_tc > 0])) if np.any(obs_tc > 0) else float(target_sum)

        if scale_method == "median":
            tc_safe[bad] = med

        elif scale_method == "knn_tc":
            coords = np.asarray(adata_aligned.obsm["spatial"])
            miss = np.asarray(missing_indices, dtype=int)

            est_missing = _estimate_missing_total_counts_knn(
                coords=coords,
                total_counts=tc_safe,
                missing_idx=miss,
                observed_idx=np.asarray(observed_indices, dtype=int),
                k=knn_k,
                eps=knn_eps,
            )

            # only replace missing spots that are bad
            tc_safe[miss] = np.where(tc_safe[miss] > 0, tc_safe[miss], est_missing)

            # any remaining bad -> fallback median
            bad2 = (~np.isfinite(tc_safe)) | (tc_safe <= 0)
            if np.any(bad2):
                tc_safe[bad2] = med

        else:
            raise ValueError("scale_method must be 'median' or 'knn_tc'.")

    return tc_safe


def _robust_vmin_vmax(values_list, clip_quantiles=(0.02, 0.98)):
    vals = np.concatenate([v[np.isfinite(v)] for v in values_list]).astype(np.float32)
    if vals.size == 0:
        raise ValueError("No finite values for plotting.")
    q_low, q_high = clip_quantiles
    vmin = float(np.quantile(vals, q_low))
    vmax = float(np.quantile(vals, q_high))
    if vmax <= vmin:
        vmin = float(vals.min())
        vmax = float(vals.max())
        if vmax <= vmin:
            vmax = vmin + 1.0
    return vmin, vmax


def _style_ax(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spn in ax.spines.values():
        spn.set_visible(False)
    ax.set_frame_on(False)


# =========================
# Main API
# =========================
def plot_task2_real_three_panel(
    *,
    adata_aligned,          # your aligned MSI (with missing labels & spatial)
    adata_unaligned_raw,    # your raw unaligned MSI (with spatial)
    save_path: str,
    first_name: str,
    feature,                # str or int (based on adata_aligned.var_names)
    plot_space: str = "raw",             # "raw" | "processed"
    target_sum: float = 1e4,
    clip_quantiles=(0.02, 0.98),
    point_size: float = 10,
    alpha: float = 0.9,
    highlight_missing: bool = False,
    highlight_color: str = "red",
    highlight_linewidth: float = 0.01,
    figsize=(12, 4),
    dpi: int = 300,
    cmap_name: str = "viridis",
    # shared colorbar
    cbar_pos=(0.90, 0.08, 0.02, 0.77),
    cbar_ticksize: int = 10,
    # missing & raw inversion options
    missing_key: str = "missing",
    missing_value: str = "0",
    observed_value: str = "1",
    raw_layer_key: str = "raw_data",     # aligned raw stored here
    processed_true_key: str = "true_data",# aligned processed true stored here (uns)
    raw_total_key: str = "raw_total_counts",
    scale_method: str = "knn_tc",        # "median" or "knn_tc"
    knn_k: int = 30,
    knn_eps: float = 1e-6,
    # unaligned processed fallback
    unaligned_processed_fallback: str = "log1p",  # "log1p" or "none"
):
    """
    3-panel visualization for REAL missing (no GT):
      (1) unaligned raw (or processed fallback)
      (2) aligned raw/processed
      (3) PRISM prediction shown in raw/processed

    PRISM prediction file required:
      save_path/{first_name}_pre.csv  (processed space: log1p(norm))
    """
    if missing_key not in adata_aligned.obs.columns:
        raise KeyError(f"adata_aligned.obs['{missing_key}'] not found.")

    # indices
    miss_mask = (adata_aligned.obs[missing_key].astype(str).to_numpy() == str(missing_value))
    obs_mask = (adata_aligned.obs[missing_key].astype(str).to_numpy() == str(observed_value))
    missing_indices = np.flatnonzero(miss_mask)
    observed_indices = np.flatnonzero(obs_mask)

    # feature idx/name (based on aligned)
    feat_idx_aligned, feat_name = _get_feature_index(adata_aligned, feature)

    # ensure feature exists in unaligned raw
    if feat_name not in adata_unaligned_raw.var_names:
        raise KeyError(f"Feature '{feat_name}' not found in adata_unaligned_raw.var_names.")
    feat_idx_unaligned, _ = _get_feature_index(adata_unaligned_raw, feat_name)

    # --------------------------------------------------
    # Panel 1: unaligned
    # --------------------------------------------------
    if plot_space == "raw":
        v_unaligned = _get_1d_from_X(adata_unaligned_raw, feat_idx_unaligned)
    elif plot_space == "processed":
        if unaligned_processed_fallback == "log1p":
            v0 = _get_1d_from_X(adata_unaligned_raw, feat_idx_unaligned)
            v_unaligned = np.log1p(np.maximum(v0, 0)).astype(np.float32)
        else:
            raise ValueError("No processed data for unaligned; set unaligned_processed_fallback='log1p'.")
    else:
        raise ValueError("plot_space must be 'raw' or 'processed'.")

    # --------------------------------------------------
    # Panel 2: aligned
    # --------------------------------------------------
    if plot_space == "raw":
        if raw_layer_key not in adata_aligned.layers:
            raise KeyError(f"adata_aligned.layers['{raw_layer_key}'] not found.")
        v_aligned = _get_1d_from_layer(adata_aligned, raw_layer_key, feat_idx_aligned)
    else:
        if processed_true_key not in adata_aligned.uns:
            raise KeyError(f"adata_aligned.uns['{processed_true_key}'] not found.")
        Xp = adata_aligned.uns[processed_true_key]
        if sp.issparse(Xp):
            v_aligned = np.asarray(Xp[:, feat_idx_aligned].toarray()).ravel().astype(np.float32)
        else:
            v_aligned = np.asarray(Xp[:, feat_idx_aligned]).ravel().astype(np.float32)

    # --------------------------------------------------
    # Panel 3: PRISM prediction (csv is processed)
    # --------------------------------------------------
    pred_path = os.path.join(save_path, first_name + "_pre.csv")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"PRISM prediction not found: {pred_path}")

    pred_df = pd.read_csv(pred_path, index_col=0)
    # robust alignment for plotting
    pred_df = pred_df.reindex(index=adata_aligned.obs_names, columns=adata_aligned.var_names)
    pred_log = pred_df.values.astype(np.float32)
    v_pred_processed = pred_log[:, feat_idx_aligned].copy()

    if plot_space == "processed":
        v_pred = v_pred_processed
    else:
        tc_safe = _make_raw_total_counts_safe(
            adata_aligned,
            missing_indices=missing_indices,
            observed_indices=observed_indices,
            target_sum=target_sum,
            scale_method=scale_method,
            knn_k=knn_k,
            knn_eps=knn_eps,
            raw_total_key=raw_total_key,
            raw_layer_key=raw_layer_key,
        )
        v_pred_norm = np.expm1(v_pred_processed)
        v_pred = (v_pred_norm * (tc_safe / float(target_sum))).astype(np.float32)

    # --------------------------------------------------
    # Shared color scaling
    # --------------------------------------------------
    vmin, vmax = _robust_vmin_vmax([v_unaligned, v_aligned, v_pred], clip_quantiles=clip_quantiles)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")

    # panel 1 coords
    coords0 = np.asarray(adata_unaligned_raw.obsm["spatial"])
    x0, y0 = coords0[:, 0], coords0[:, 1]
    axes[0].scatter(x0, y0, c=v_unaligned, cmap=cmap, norm=norm, s=point_size, alpha=alpha, linewidths=0)
    axes[0].set_title(f"{plot_space.upper()} (unaligned)\n{feat_name}", fontsize=12)
    _style_ax(axes[0])

    # panel 2 coords
    coords1 = np.asarray(adata_aligned.obsm["spatial"])
    x1, y1 = coords1[:, 0], coords1[:, 1]
    axes[1].scatter(x1, y1, c=v_aligned, cmap=cmap, norm=norm, s=point_size, alpha=alpha, linewidths=0)
    if highlight_missing and len(missing_indices) > 0:
        axes[1].scatter(
            x1[missing_indices], y1[missing_indices],
            facecolors="none", edgecolors=highlight_color,
            s=point_size * 1.2, linewidths=highlight_linewidth
        )
    axes[1].set_title(f"{plot_space.upper()} (aligned)\n{feat_name}", fontsize=12)
    _style_ax(axes[1])

    # panel 3 (pred)
    axes[2].scatter(x1, y1, c=v_pred, cmap=cmap, norm=norm, s=point_size, alpha=alpha, linewidths=0)
    if highlight_missing and len(missing_indices) > 0:
        axes[2].scatter(
            x1[missing_indices], y1[missing_indices],
            facecolors="none", edgecolors=highlight_color,
            s=point_size * 1.2, linewidths=highlight_linewidth
        )
    axes[2].set_title(f"{plot_space.upper()} (PRISM)\n{feat_name}", fontsize=12)
    _style_ax(axes[2])

    plt.subplots_adjust(wspace=0.06, left=0.02, right=0.90, top=0.88, bottom=0.05)

    # shared colorbar
    cax = fig.add_axes(list(cbar_pos))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=cbar_ticksize)

    plt.show()

    return {
        "feature_name": feat_name,
        "missing_count": int(len(missing_indices)),
        "observed_count": int(len(observed_indices)),
        "vmin": vmin,
        "vmax": vmax,
    }