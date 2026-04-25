# prism/covet_aot.py
# COVET + AOT (Haviv et al. Nat Biotech 2024) reproduction utilities
# - Spatial kNN (exclude self)         [sklearn CPU]
# - Shifted covariance (global mean)   [torch GPU if device=cuda]
# - Matrix sqrt via eigh              [torch GPU if device=cuda]
# - AOT-kNN graph in sqrt-COVET space:
#     - sklearn CPU backend
#     - torch GPU exact backend (no faiss needed)
#
# IMPORTANT: AOT distances stored are SQUARED distances:
#   aot_dist(i,j) = || vec(sqrt(Sigma_i)) - vec(sqrt(Sigma_j)) ||_2^2
#                 = || sqrt(Sigma_i) - sqrt(Sigma_j) ||_F^2

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors

ArrayLike = Union[np.ndarray, sp.spmatrix]
DeviceLike = Optional[Union[str, int, torch.device]]


def _normalize_device_string(s: str) -> str:
    return s.strip().lower().replace(" ", "")


def _get_device(device: DeviceLike = None, *, verbose: bool = False) -> torch.device:
    """
    Robust device chooser.

    Supported inputs:
      - None: auto -> cuda:0 if available else cpu
      - "cpu": cpu
      - "cuda" / "gpu": cuda:0 if available else cpu
      - "cuda:N" / "gpu:N": choose cuda:N (fallback to cuda:0 or cpu)
      - int N: same as "cuda:N"
      - torch.device: validated (fallback if invalid cuda index / cuda unavailable)
    """
    # Auto
    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # int -> cuda:int
    if isinstance(device, int):
        device = f"cuda:{device}"

    # torch.device
    if isinstance(device, torch.device):
        if device.type != "cuda":
            return device
        # cuda device requested
        if not torch.cuda.is_available():
            if verbose:
                print("[device] CUDA not available, fallback to cpu")
            return torch.device("cpu")
        # validate index (cuda without index treated as cuda:0 by torch, but we normalize)
        idx = device.index if device.index is not None else 0
        n = torch.cuda.device_count()
        if idx < 0 or idx >= n:
            if verbose:
                print(f"[device] Requested cuda:{idx} but only {n} GPU(s) available; fallback to cuda:0")
            return torch.device("cuda:0") if n > 0 else torch.device("cpu")
        return torch.device(f"cuda:{idx}")

    # string
    if isinstance(device, str):
        s = _normalize_device_string(device)

        if s == "cpu":
            return torch.device("cpu")

        if s in {"cuda", "gpu"}:
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            if verbose:
                print("[device] CUDA not available, fallback to cpu")
            return torch.device("cpu")

        # "cuda:N" or "gpu:N"
        if s.startswith("cuda:") or s.startswith("gpu:"):
            # accept both prefixes
            try:
                idx = int(s.split(":", 1)[1])
            except Exception:
                raise ValueError(f"Invalid device string: {device!r}. Use 'cpu', 'cuda', or 'cuda:N'.")

            if not torch.cuda.is_available():
                if verbose:
                    print("[device] CUDA not available, fallback to cpu")
                return torch.device("cpu")

            n = torch.cuda.device_count()
            if idx < 0 or idx >= n:
                if verbose:
                    print(f"[device] Requested cuda:{idx} but only {n} GPU(s) available; fallback to cuda:0")
                return torch.device("cuda:0") if n > 0 else torch.device("cpu")
            return torch.device(f"cuda:{idx}")

        # pass-through for other devices (e.g. "mps")
        dev = torch.device(device)
        if dev.type == "cuda":
            # validate cuda availability/index
            if not torch.cuda.is_available():
                if verbose:
                    print("[device] CUDA not available, fallback to cpu")
                return torch.device("cpu")
            idx = dev.index if dev.index is not None else 0
            n = torch.cuda.device_count()
            if idx < 0 or idx >= n:
                if verbose:
                    print(f"[device] Requested cuda:{idx} but only {n} GPU(s) available; fallback to cuda:0")
                return torch.device("cuda:0") if n > 0 else torch.device("cpu")
            return torch.device(f"cuda:{idx}")
        return dev

    raise TypeError(f"Unsupported device type: {type(device)}")


def _to_torch_dense(X: ArrayLike, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    """AnnData.X (dense or sparse) -> torch dense tensor on device."""
    if sp.issparse(X):
        X = X.toarray()
    return torch.as_tensor(X, dtype=dtype, device=device)


def select_genes_idx(
    adata,
    genes: Union[str, np.ndarray, list, tuple] = "all",
    n_hvg: int = 350,
    gene_key: str = "highly_variable",
) -> np.ndarray:
    """
    genes:
      - "all": all genes
      - "hvg": use adata.var[gene_key] if exists else top n_hvg by variance
      - list/tuple of gene names
      - np.ndarray bool mask or int indices
    """
    G = adata.n_vars

    if genes == "all":
        return np.arange(G, dtype=int)

    if genes == "hvg":
        if hasattr(adata, "var") and gene_key in getattr(adata, "var").columns:
            mask = np.asarray(adata.var[gene_key].values).astype(bool)
            if mask.sum() > 0:
                return np.where(mask)[0]

        X = adata.X
        if sp.issparse(X):
            mean = np.asarray(X.mean(axis=0)).ravel()
            mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
            var = mean_sq - mean**2
        else:
            var = np.var(np.asarray(X), axis=0)

        n = min(n_hvg, G)
        return np.argsort(var)[-n:]

    # gene name list
    if isinstance(genes, (list, tuple)) and len(genes) > 0 and isinstance(genes[0], str):
        name_to_idx = {g: i for i, g in enumerate(adata.var_names)}
        idx = [name_to_idx[g] for g in genes if g in name_to_idx]
        if len(idx) == 0:
            raise ValueError("None of the provided gene names found in adata.var_names.")
        return np.array(idx, dtype=int)

    genes = np.asarray(genes)
    if genes.dtype == bool:
        if genes.shape[0] != G:
            raise ValueError("Boolean mask length must equal adata.n_vars.")
        return np.where(genes)[0]
    if np.issubdtype(genes.dtype, np.integer):
        return genes.astype(int)

    raise ValueError("Unsupported genes argument.")


@dataclass
class CovetConfig:
    k_spatial: int = 8
    spatial_key: str = "spatial"
    genes: Union[str, np.ndarray, list, tuple] = "hvg"
    n_hvg: int = 64
    include_self: bool = False
    eps: float = 1e-6

    # device:
    #   - None: auto (cuda:0 if available else cpu)
    #   - "cpu": cpu
    #   - "cuda"/"gpu": cuda:0
    #   - "cuda:N"/"gpu:N": choose cuda:N (fallback to cuda:0 if N invalid)
    #   - int N: same as "cuda:N"
    device: DeviceLike = None

    store_prefix: str = "covet"
    return_sqrt_full: bool = False
    verbose: bool = True

    # chunking/progress
    use_chunking: bool = True
    chunk_size: int = 1000
    log_every: int = 1000


@torch.no_grad()
def compute_covet(adata, cfg: CovetConfig = CovetConfig()):
    """
    Compute sqrt-COVET embedding (upper triangle of sqrt covariance) and store into adata.obsm.

    Output:
      adata.obsm[f"{store_prefix}_sqrt_ut"]    : (n, G*(G+1)/2)
      adata.obsm[f"{store_prefix}_sqrt_full"] : (n,G,G) optional
      adata.uns[f"{store_prefix}_gene_idx"]    : gene indices used
    """
    if cfg.spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{cfg.spatial_key}'] not found.")

    device = _get_device(cfg.device, verbose=cfg.verbose)
    coords = np.asarray(adata.obsm[cfg.spatial_key])
    n = adata.n_obs

    gene_idx = select_genes_idx(adata, genes=cfg.genes, n_hvg=cfg.n_hvg)
    G = len(gene_idx)

    if cfg.verbose:
        print(
            f"[COVET] n_obs={n}, G={G}, k_spatial={cfg.k_spatial}, include_self={cfg.include_self}, device={device}"
        )
        print(f"[COVET] use_chunking={cfg.use_chunking}, chunk_size={cfg.chunk_size}, log_every={cfg.log_every}")

    # --- spatial kNN on CPU (exclude self) ---
    k_query = cfg.k_spatial + (0 if cfg.include_self else 1)
    nn = NearestNeighbors(n_neighbors=k_query, metric="euclidean", algorithm="auto")
    nn.fit(coords)
    _, neigh = nn.kneighbors(coords)
    if not cfg.include_self:
        neigh = neigh[:, 1 : cfg.k_spatial + 1]
    else:
        neigh = neigh[:, : cfg.k_spatial]

    # --- X on GPU/CPU ---
    X = adata.X[:, gene_idx]
    X_t = _to_torch_dense(X, device=device, dtype=torch.float32)  # (n,G)

    xbar = X_t.mean(dim=0, keepdim=True)  # (1,G)

    out_dim = G * (G + 1) // 2
    ut_out = np.empty((n, out_dim), dtype=np.float32)
    sqrt_full_out = np.empty((n, G, G), dtype=np.float32) if cfg.return_sqrt_full else None

    I = torch.eye(G, device=device, dtype=torch.float32).unsqueeze(0)  # (1,G,G)
    tri = torch.triu_indices(G, G, device=device)

    if not cfg.use_chunking:
        idx_t = torch.as_tensor(neigh, device=device, dtype=torch.long)  # (n,k)
        E = X_t[idx_t]  # (n,k,G)
        D = E - xbar  # (n,k,G)
        Sigma = torch.matmul(D.transpose(1, 2), D) / float(cfg.k_spatial)  # (n,G,G)
        Sigma = Sigma + cfg.eps * I

        evals, evecs = torch.linalg.eigh(Sigma)
        evals = torch.clamp(evals, min=0.0)
        sqrt_Sigma = evecs @ torch.diag_embed(torch.sqrt(evals)) @ evecs.transpose(-1, -2)

        ut_out[:] = sqrt_Sigma[:, tri[0], tri[1]].detach().cpu().numpy().astype(np.float32)
        if cfg.return_sqrt_full:
            sqrt_full_out[:] = sqrt_Sigma.detach().cpu().numpy().astype(np.float32)

        if cfg.verbose:
            print(f"[COVET] processed {n}/{n}")

    else:
        chunk = max(1, int(cfg.chunk_size))
        log_every = max(1, int(cfg.log_every))
        processed = 0

        for start in range(0, n, chunk):
            end = min(n, start + chunk)

            idx_t = torch.as_tensor(neigh[start:end], device=device, dtype=torch.long)  # (b,k)
            E = X_t[idx_t]  # (b,k,G)
            D = E - xbar  # (b,k,G)
            Sigma = torch.matmul(D.transpose(1, 2), D) / float(cfg.k_spatial)  # (b,G,G)
            Sigma = Sigma + cfg.eps * I

            evals, evecs = torch.linalg.eigh(Sigma)
            evals = torch.clamp(evals, min=0.0)
            sqrt_Sigma = evecs @ torch.diag_embed(torch.sqrt(evals)) @ evecs.transpose(-1, -2)

            ut_out[start:end] = sqrt_Sigma[:, tri[0], tri[1]].detach().cpu().numpy().astype(np.float32)
            if cfg.return_sqrt_full:
                sqrt_full_out[start:end] = sqrt_Sigma.detach().cpu().numpy().astype(np.float32)

            processed = end
            if cfg.verbose and (processed % log_every == 0 or processed == n):
                print(f"[COVET] processed {processed}/{n}")

    adata.obsm[f"{cfg.store_prefix}_sqrt_ut"] = ut_out
    if cfg.return_sqrt_full:
        adata.obsm[f"{cfg.store_prefix}_sqrt_full"] = sqrt_full_out
    adata.uns[f"{cfg.store_prefix}_gene_idx"] = gene_idx

    if cfg.verbose:
        print(f"[COVET] saved obsm['{cfg.store_prefix}_sqrt_ut'] shape = {ut_out.shape}")

    return adata


@dataclass
class AotGraphConfig:
    covet_ut_key: str = "covet_sqrt_ut"
    k_env: int = 30
    metric: str = "euclidean"
    symmetrize: bool = False   #default False
    store_prefix: str = "aot"
    verbose: bool = True

    # chunking/progress
    use_chunking: bool = True
    chunk_size: int = 1000
    log_every: int = 1000

    # backend:
    #   - "sklearn": CPU NearestNeighbors
    #   - "torch":   GPU exact kNN via chunked distance + topk (no faiss required)
    knn_backend: str = "torch"

    # knn_device supports same formats as CovetConfig.device
    knn_device: DeviceLike = None


def build_aot_knn_graph(adata, cfg: AotGraphConfig = AotGraphConfig()):
    """
    Build AOT kNN graph using SQUARED distances in sqrt-COVET space.

    Stores:
      adata.obsp[f"{store_prefix}_distances"] : csr (n,n) squared euclidean distances
    """
    if cfg.covet_ut_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{cfg.covet_ut_key}'] not found. Run compute_covet() first.")
    if cfg.metric != "euclidean":
        raise ValueError("Only metric='euclidean' supported here (AOT eq. uses Euclidean).")

    F_np = np.asarray(adata.obsm[cfg.covet_ut_key], dtype=np.float32)
    n, d = F_np.shape

    if cfg.verbose:
        print(f"[AOT-kNN] n={n}, d={d}, k_env={cfg.k_env}, backend={cfg.knn_backend}")
        print(f"[AOT-kNN] use_chunking={cfg.use_chunking}, chunk_size={cfg.chunk_size}, log_every={cfg.log_every}")
        print("[AOT-kNN] storing SQUARED distances (matches paper AOT definition)")

    # -------------------------
    # Backend: sklearn (CPU)
    # -------------------------
    if cfg.knn_backend.lower() == "sklearn":
        nn = NearestNeighbors(n_neighbors=cfg.k_env + 1, metric="euclidean", algorithm="auto")
        nn.fit(F_np)

        if not cfg.use_chunking:
            dist, idx = nn.kneighbors(F_np, return_distance=True)
            dist = dist[:, 1:]  # drop self
            idx = idx[:, 1:]
            dist2 = (dist**2).astype(np.float32)

            rows = np.repeat(np.arange(n), cfg.k_env)
            cols = idx.reshape(-1)
            data = dist2.reshape(-1)

            D = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
            if cfg.symmetrize:
                D = 0.5 * (D + D.T)

            adata.obsp[f"{cfg.store_prefix}_distances"] = D
            if cfg.verbose:
                print(f"[AOT-kNN] processed {n}/{n}")
                print(f"[AOT-kNN] saved obsp['{cfg.store_prefix}_distances'] nnz={D.nnz}, k_env={cfg.k_env}")
            return adata

        chunk = max(1, int(cfg.chunk_size))
        log_every = max(1, int(cfg.log_every))

        rows_all, cols_all, data_all = [], [], []
        processed = 0
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            dist, idx = nn.kneighbors(F_np[start:end], return_distance=True)
            dist = dist[:, 1:]
            idx = idx[:, 1:]
            dist2 = (dist**2).astype(np.float32)

            rows = np.repeat(np.arange(start, end), cfg.k_env)
            cols = idx.reshape(-1)
            data = dist2.reshape(-1)

            rows_all.append(rows)
            cols_all.append(cols)
            data_all.append(data)

            processed = end
            if cfg.verbose and (processed % log_every == 0 or processed == n):
                print(f"[AOT-kNN] processed {processed}/{n}")

        rows = np.concatenate(rows_all)
        cols = np.concatenate(cols_all)
        data = np.concatenate(data_all)

        D = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        if cfg.symmetrize:
            D = 0.5 * (D + D.T)

        adata.obsp[f"{cfg.store_prefix}_distances"] = D
        if cfg.verbose:
            print(f"[AOT-kNN] saved obsp['{cfg.store_prefix}_distances'] nnz={D.nnz}, k_env={cfg.k_env}")
        return adata

    # -------------------------
    # Backend: torch (GPU exact kNN)
    # -------------------------
    if cfg.knn_backend.lower() != "torch":
        raise ValueError("cfg.knn_backend must be 'sklearn' or 'torch'.")

    device = _get_device(cfg.knn_device, verbose=cfg.verbose)
    if cfg.verbose:
        print(f"[AOT-kNN/torch] device={device} (CUDA recommended)")

    X = torch.as_tensor(F_np, device=device, dtype=torch.float32)  # (n,d)
    x2 = (X * X).sum(dim=1)  # (n,)

    chunk = n if not cfg.use_chunking else max(1, int(cfg.chunk_size))
    log_every = max(1, int(cfg.log_every))

    rows_all, cols_all, data_all = [], [], []
    processed = 0

    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        Q = X[start:end]  # (b,d)
        q2 = (Q * Q).sum(dim=1, keepdim=True)  # (b,1)

        # squared distances (b,n)
        dist2 = q2 + x2.unsqueeze(0) - 2.0 * (Q @ X.t())

        # exclude self
        ar = torch.arange(end - start, device=device)
        dist2[ar, start + ar] = float("inf")

        # take smallest k_env (already squared distances)
        vals2, idx = torch.topk(dist2, k=cfg.k_env, dim=1, largest=False, sorted=True)

        vals2_np = torch.clamp(vals2, min=0.0).detach().cpu().numpy().astype(np.float32)
        idx_np = idx.detach().cpu().numpy().astype(np.int64)

        rows = np.repeat(np.arange(start, end), cfg.k_env)
        cols = idx_np.reshape(-1)
        data = vals2_np.reshape(-1)

        rows_all.append(rows)
        cols_all.append(cols)
        data_all.append(data)

        processed = end
        if cfg.verbose and (processed % log_every == 0 or processed == n):
            print(f"[AOT-kNN/torch] processed {processed}/{n}")

        del dist2, vals2, idx

    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    data = np.concatenate(data_all)

    D = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    if cfg.symmetrize:
        D = 0.5 * (D + D.T)

    adata.obsp[f"{cfg.store_prefix}_distances"] = D
    if cfg.verbose:
        print(f"[AOT-kNN/torch] saved obsp['{cfg.store_prefix}_distances'] nnz={D.nnz}, k_env={cfg.k_env}")

    return adata


@torch.no_grad()
def aot_distance_matrix_full(
    adata,
    covet_ut_key: str = "covet_sqrt_ut",
    device: DeviceLike = None,
    squared: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """
    OPTIONAL: full n×n AOT distance matrix via torch.cdist on GPU/CPU.
    Warning: O(n^2) memory/time.
    """
    if covet_ut_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{covet_ut_key}'] not found.")

    dev = _get_device(device, verbose=verbose)
    F = torch.as_tensor(adata.obsm[covet_ut_key], device=dev, dtype=torch.float32)

    if verbose:
        print(f"[AOT-full] computing n×n distances: n={F.shape[0]}, d={F.shape[1]}, device={dev}")

    D = torch.cdist(F, F, p=2)
    if squared:
        D = D.pow(2)
    return D.detach().cpu().numpy().astype(np.float32)