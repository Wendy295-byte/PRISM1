import os
import scipy
import anndata
import sklearn
import torch
import random
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Optional
import scipy.sparse as sp
from torch.backends import cudnn
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph 
    
def construct_neighbor_graph(adata_omics1, adata_omics2, datatype='SPOTS', n_neighbors=3): 
    """
    Construct neighbor graphs, including feature graph and spatial graph. 
    Feature graph is based expression data while spatial graph is based on cell/spot spatial coordinates.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors.

    Returns
    -------
    data : dict
        AnnData objects with preprossed data for different omics.

    """

    # construct spatial neighbor graphs
    ################# spatial graph #################
    if datatype in ['Stereo-CITE-seq', 'Spatial-epigenome-transcriptome']:
       n_neighbors=6 
    # omics1
    cell_position_omics1 = adata_omics1.obsm['spatial']
    adj_omics1 = construct_graph_by_coordinate(cell_position_omics1, n_neighbors=n_neighbors)
    adata_omics1.uns['adj_spatial'] = adj_omics1
    
    # omics2
    cell_position_omics2 = adata_omics2.obsm['spatial']
    adj_omics2 = construct_graph_by_coordinate(cell_position_omics2, n_neighbors=n_neighbors)
    adata_omics2.uns['adj_spatial'] = adj_omics2
    
    ################# feature graph #################
    feature_graph_omics1, feature_graph_omics2 = construct_graph_by_feature(adata_omics1, adata_omics2)
    adata_omics1.obsm['adj_feature'], adata_omics2.obsm['adj_feature'] = feature_graph_omics1, feature_graph_omics2
    
    data = {'adata_omics1': adata_omics1, 'adata_omics2': adata_omics2}
    
    return data

def pca(adata, use_reps=None, n_comps=10):
    
    """Dimension reduction with PCA algorithm"""
    
    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
    pca = PCA(n_components=n_comps)
    if use_reps is not None:
       feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else: 
       if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
          feat_pca = pca.fit_transform(adata.X.toarray()) 
       else:   
          feat_pca = pca.fit_transform(adata.X)
    
    return feat_pca

def clr_normalize_each_cell(adata, inplace=True):
    
    """Normalize count vector for each cell, i.e. for each row of .X"""

    import numpy as np
    import scipy

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()
    
    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata     

def construct_graph_by_feature(adata_omics1, adata_omics2, k=20, mode= "connectivity", metric="correlation", include_self=False):
    
    """Constructing feature neighbor graph according to expresss profiles"""
    
    feature_graph_omics1=kneighbors_graph(adata_omics1.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self)
    feature_graph_omics2=kneighbors_graph(adata_omics2.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self)

    return feature_graph_omics1, feature_graph_omics2

def construct_graph_by_coordinate(cell_position, n_neighbors=3):
    #print('n_neighbor:', n_neighbors)
    """Constructing spatial neighbor graph according to spatial coordinates."""
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(cell_position)  
    _ , indices = nbrs.kneighbors(cell_position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    adj = pd.DataFrame(columns=['x', 'y', 'value'])
    adj['x'] = x
    adj['y'] = y
    adj['value'] = np.ones(x.size)
    return adj

def transform_adjacent_matrix(adjacent):
    n_spot = adjacent['x'].max() + 1
    adj = coo_matrix((adjacent['value'], (adjacent['x'], adjacent['y'])), shape=(n_spot, n_spot))
    return adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# ====== Graph preprocessing
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def adjacent_matrix_preprocessing(adata_omics1, adata_omics2):
    """Converting dense adjacent matrix to sparse adjacent matrix"""
    
    ######################################## construct spatial graph ########################################
    adj_spatial_omics1 = adata_omics1.uns['adj_spatial']
    adj_spatial_omics1 = transform_adjacent_matrix(adj_spatial_omics1)
    adj_spatial_omics2 = adata_omics2.uns['adj_spatial']
    adj_spatial_omics2 = transform_adjacent_matrix(adj_spatial_omics2)
    
    adj_spatial_omics1 = adj_spatial_omics1.toarray()   # To ensure that adjacent matrix is symmetric
    adj_spatial_omics2 = adj_spatial_omics2.toarray()
    
    adj_spatial_omics1 = adj_spatial_omics1 + adj_spatial_omics1.T
    adj_spatial_omics1 = np.where(adj_spatial_omics1>1, 1, adj_spatial_omics1)
    adj_spatial_omics2 = adj_spatial_omics2 + adj_spatial_omics2.T
    adj_spatial_omics2 = np.where(adj_spatial_omics2>1, 1, adj_spatial_omics2)
    
    # convert dense matrix to sparse matrix
    adj_spatial_omics1 = preprocess_graph(adj_spatial_omics1) # sparse adjacent matrix corresponding to spatial graph
    adj_spatial_omics2 = preprocess_graph(adj_spatial_omics2)
    
    ######################################## construct feature graph ########################################
    adj_feature_omics1 = torch.FloatTensor(adata_omics1.obsm['adj_feature'].copy().toarray())
    adj_feature_omics2 = torch.FloatTensor(adata_omics2.obsm['adj_feature'].copy().toarray())
    
    adj_feature_omics1 = adj_feature_omics1 + adj_feature_omics1.T
    adj_feature_omics1 = np.where(adj_feature_omics1>1, 1, adj_feature_omics1)
    adj_feature_omics2 = adj_feature_omics2 + adj_feature_omics2.T
    adj_feature_omics2 = np.where(adj_feature_omics2>1, 1, adj_feature_omics2)
    
    # convert dense matrix to sparse matrix
    adj_feature_omics1 = preprocess_graph(adj_feature_omics1) # sparse adjacent matrix corresponding to feature graph
    adj_feature_omics2 = preprocess_graph(adj_feature_omics2)
    
    adj = {'adj_spatial_omics1': adj_spatial_omics1,
           'adj_spatial_omics2': adj_spatial_omics2,
           'adj_feature_omics1': adj_feature_omics1,
           'adj_feature_omics2': adj_feature_omics2,
           }
    
    return adj

def lsi(
        adata: anndata.AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
       ) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    #X = adata_use.X
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    #adata.obsm["X_lsi"] = X_lsi
    adata.obsm["X_lsi"] = X_lsi[:,1:]

def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf   
    
def fix_seed(seed):
    #seed = 2023
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'    





### plotting functions for spatial matching results visualization

#fig1: Matched vs Unmatched
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

def plot_matching_summary_bar(
    mapping,
    *,
    save_dir: str = ".",
    file_prefix: str = "Matched vs Unmatched",
    save: bool = True,
    dpi: int = 300,
    figsize=(6, 5),
    palette=None,
    font_family: str = "Arial",
    font_size: int = 15,
):
    """
    Bar chart summary for matching result.
    mapping: array-like, -1 means unmatched, otherwise matched.
    """
    mapping = np.asarray(mapping, dtype=int)
    matched_count = int(np.sum(mapping != -1))
    unmatched_count = int(np.sum(mapping == -1))
    total_cells = int(mapping.size)

    if palette is None:
        palette = {"Matched": "#5A6E8C", "Unmatched": "#9268AD"}

    plt.rcParams["font.family"] = font_family
    plt.rcParams["font.size"] = font_size
    sns.set_style("ticks")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    categories = ["Matched", "Unmatched"]
    counts = [matched_count, unmatched_count]

    bars = ax.bar(
        categories,
        counts,
        color=[palette[c] for c in categories],
        width=0.6,
        linewidth=1,
        alpha=0.7,
        edgecolor="black",
    )

    # number labels
    y_offset = max(1, int(0.01 * max(counts)))  # small adaptive offset
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_offset,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=font_size + 1,
        )

    ax.set_ylabel("Number of Cells")
    ax.set_xlabel("")

    # spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["left"].set_linewidth(1)
    ax.spines["bottom"].set_color("black")
    ax.spines["bottom"].set_linewidth(1)

    ax.tick_params(axis="x", colors="black", width=1, length=5, direction="out")
    ax.tick_params(axis="y", colors="black", width=1, length=5, direction="out")
    ax.grid(False)

    # legend
    handles = [
        Patch(facecolor=palette["Matched"], edgecolor="none"),
        Patch(facecolor=palette["Unmatched"], edgecolor="none"),
    ]
    ax.legend(
        handles,
        ["Matched", "Unmatched"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=2,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.2,
    )

    if save:
        os.makedirs(save_dir, exist_ok=True)
        for ext in ("png", "pdf"):
            out_path = os.path.join(save_dir, f"{file_prefix}.{ext}")
            plt.savefig(out_path, dpi=dpi, bbox_inches="tight")

    plt.show()

    # print stats
    if total_cells > 0:
        print(f"Total cells: {total_cells}")
        print(f"Matched cells: {matched_count} ({matched_count/total_cells*100:.1f}%)")
        print(f"Unmatched cells: {unmatched_count} ({unmatched_count/total_cells*100:.1f}%)")



#fig2: RNA vs MSI spatial overlay
# fig2: RNA vs MSI spatial overlay (no real matching lines)
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_spatial_overlay(
    omics1_spatial,
    omics2_spatial,
    *,
    label1: str = "RNA",
    label2: str = "MSI",
    color1: str = "blue",
    color2: str = "red",
    s: float = 5,                       # dot size for real points
    dpi: int = 300,
    figsize=(10, 10),
    font_family: str = "Arial",
    font_size: int = 18,
    margin: float = 1000.0,             # axis margin (same logic as your original)
    # legend placement (keep your original look)
    legend_marker_size: float = 20,
    legend_bbox=(0.385, 1.10),
    # "Matched" demo icon placement
    demo_ex_y: float = 1.02,
    demo_ex_x: float = 0.61,
    demo_s: float = 80,                 # demo dot size (points^2)
    demo_line_color: str = "gray",
    demo_linewidth: float = 1.2,
    demo_text: str = "Matched",
    demo_text_dx: float = 0.02,
    # saving
    save_dir: str = ".",
    file_prefix: str = "D5-1-spatial_matching_result",
    save: bool = True,
):
    """
    Spatial overlay plot for two modalities (RNA vs MSI) without drawing real matching lines.

    Notes:
    - This function DOES NOT plot the true matching lines between points.
    - It keeps your original figure size, font, legend layout, axis style, and the "Matched" demo icon.
    - Axis limits follow the same padding rule: use RNA spatial range + margin.
    """
    omics1_spatial = np.asarray(omics1_spatial, dtype=float)
    omics2_spatial = np.asarray(omics2_spatial, dtype=float)

    # ---- global style ----
    plt.rcParams.update({"font.family": font_family, "font.size": font_size})
    plt.rcParams.update({"xtick.labelsize": font_size, "ytick.labelsize": font_size})
    plt.style.use("default")  # white background

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")

    # ---- scatter points ----
    ax.scatter(omics1_spatial[:, 0], omics1_spatial[:, 1], c=color1, s=s, label=label1)
    ax.scatter(omics2_spatial[:, 0], omics2_spatial[:, 1], c=color2, s=s, label=label2)

    # ---- legend (match your original) ----
    ax.legend(
        loc="upper center",
        fontsize=font_size,
        markerscale=legend_marker_size / 5,
        ncol=2,
        columnspacing=1.5,
        labelspacing=0.1,
        borderpad=1.5,
        handletextpad=-0.1,
        bbox_to_anchor=legend_bbox,
        frameon=False,
    )

    # ---- remove spines/ticks (match your original) ----
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(
        axis="both", which="both",
        bottom=False, left=False, top=False, right=False,
        labelbottom=False, labelleft=False
    )

    # ---- "Matched" demo icon (keep your original behavior) ----
    # make sure bbox is available for accurate conversion
    fig.canvas.draw()
    r_pts = np.sqrt(demo_s / np.pi)          # radius in points
    dx_px = r_pts * fig.dpi / 72.0           # points -> pixels
    bbox = ax.get_window_extent()
    dx_axes = dx_px / bbox.width             # pixels -> axes fraction

    xL, xR = demo_ex_x - dx_axes / 2, demo_ex_x + dx_axes / 2

    # small demo connecting line (NOT real matching line)
    ax.plot(
        [xL, xR], [demo_ex_y, demo_ex_y],
        transform=ax.transAxes,
        color=demo_line_color,
        linewidth=demo_linewidth,
        clip_on=False
    )
    ax.scatter([xL], [demo_ex_y], transform=ax.transAxes, c=color1, s=demo_s, clip_on=False, zorder=3)
    ax.scatter([xR], [demo_ex_y], transform=ax.transAxes, c=color2, s=demo_s, clip_on=False, zorder=3)

    ax.text(
        xR + demo_text_dx, demo_ex_y, demo_text,
        transform=ax.transAxes,
        va="center", ha="left",
        fontsize=font_size
    )

    # ---- axis limits (same padding rule as your original) ----
    x_min = float(np.min(omics1_spatial[:, 0])) - margin
    x_max = float(np.max(omics1_spatial[:, 0])) + margin
    y_min = float(np.min(omics1_spatial[:, 1])) - margin
    y_max = float(np.max(omics1_spatial[:, 1])) + margin
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_aspect("equal")

    # ---- save ----
    if save:
        os.makedirs(save_dir, exist_ok=True)
        for ext in ("pdf", "png"):
            out_path = os.path.join(save_dir, f"{file_prefix}.{ext}")
            plt.savefig(out_path, dpi=dpi, bbox_inches="tight")

    plt.show()
    return fig, ax