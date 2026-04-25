import numpy as np
import matplotlib.pyplot as plt


def simulate_missing_sliding(
    adata,
    spatial_key: str = "spatial",
    direction: str = "h", 
    missing_width: float = 0.5,
    step_ratio: float = 0.1,
    step_id: int = 1,
    label_key: str = "missing",
    lock_at_end: bool = True,
    plot: bool = True,
    figsize=(6, 5),
):
    """
    Simulate missing (unregistered) region by selecting a contiguous window after sorting cells by spatial coordinate.

    Parameters
    ----------
    adata : AnnData
        Target AnnData (e.g., adata_omics2).
    spatial_key : str
        Key in adata.obsm for spatial coordinates (default 'spatial').
        Expected shape: (n_obs, 2), with [row(y), col(x)] or [y, x].
    direction : str
        H is "horizonta, 'horizontal' -> sort by x (spatial[:, 1])
        V is "vertical", 'vertical'   -> sort by y (spatial[:, 0])
    missing_width : float
        Missing window width ratio (e.g., 0.5 means always 50% missing).
    step_ratio : float
        Step size ratio for moving the window (e.g., 0.1 means move by 10% of total cells per step).
    step_id : int
        Which step position to use. start = step_id * step_size.
        - step_id=0 -> window starts at 0
        - step_id=1 -> window starts at 1*step_size (your current behavior)
    label_key : str
        Column name in adata.obs to store missing labels ('1' non-missing, '0' missing).
    lock_at_end : bool
        If True, when window exceeds bounds, lock it at the end while keeping width fixed.
        If False, simply clip end (width may shrink).
    plot : bool
        Whether to visualize missing vs non-missing distribution.
    figsize : tuple
        Matplotlib figure size.

    Returns
    -------
    split1_indices : np.ndarray
        Indices (in original adata order) that are marked as missing (label '0').
    """
    if spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{spatial_key}'] not found.")

    coords = np.asarray(adata.obsm[spatial_key])
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"adata.obsm['{spatial_key}'] must be a (n_obs, 2) array.")

    # Extract spatial coordinates
    x_coords = coords[:, 1]  # X-axis coordinates: horizontal
    y_coords = coords[:, 0]  # Y-axis coordinates: vertical

    # Choose sorting direction

    direction_clean = str(direction).strip().upper()
    if direction_clean == "H":
        sorted_indices = np.argsort(x_coords)
    elif direction_clean == "V":
        sorted_indices = np.argsort(y_coords)
    else:
        raise ValueError(f"direction must be 'H' (horizontal) or 'V' (vertical). Got: '{direction}'")
    
    # ------------------
    # Validate ratios
    if not (0.0 < missing_width <= 1.0):
        raise ValueError("missing_width must be in (0, 1].")
    if not (0.0 <= step_ratio <= 1.0):
        raise ValueError("step_ratio must be in [0, 1].")
    if step_id < 0:
        raise ValueError("step_id must be >= 0.")

    N = int(adata.shape[0])
    window = int(N * missing_width)
    step_size = int(N * step_ratio)
    start = step_id * step_size
    end = start + window

    # Handle boundaries
    if end > N:
        if lock_at_end:
            end = N
            start = max(0, N - window)
        else:
            end = N  # width may shrink

    split1_indices = sorted_indices[start:end].astype(int)

    # Label Missing and Non-missing Data
    adata.obs[label_key] = "1"  # Initially, mark all data as non-missing ('1')
    adata.obs.iloc[split1_indices, adata.obs.columns.get_loc(label_key)] = "0"  # Mark missing as '0'

    # Visualize
    if plot:
        plt.figure(figsize=figsize)

        plt.scatter(
            y_coords[adata.obs[label_key] == "0"],
            x_coords[adata.obs[label_key] == "0"],
            c="gray",
            s=10,
            alpha=0.7,
            label="Simulation missing (label=0)",
        )
        plt.scatter(
            y_coords[adata.obs[label_key] == "1"],
            x_coords[adata.obs[label_key] == "1"],
            c="blue",
            s=10,
            alpha=0.7,
            label="Non-missing (label=1)",
        )

        plt.title("Simulated Data Visualization")
        plt.xlabel("Y Coordinate")
        plt.ylabel("X Coordinate")
        plt.legend(bbox_to_anchor=(1, 0.6))
        plt.show()

    return split1_indices


import numpy as np
import matplotlib.pyplot as plt

def show_real_missing(
    adata,
    *,
    spatial_key: str = "spatial",
    label_key: str = "missing",
    missing_value="0",                 # what value means "missing"
    observed_value="1",                # what value means "observed" (optional)
    normalize_to_str: bool = True,     # robust comparison
    plot: bool = True,
    figsize=(6, 5),
    s: float = 10,
    alpha: float = 0.7,
    legend_loc=(1, 0.6),
    title: str = "Real Missing Visualization",
):
    """
    Visualize REAL missing distribution based on adata.obs[label_key].
    Does NOT modify adata.

    Returns
    -------
    missing_indices : np.ndarray
    observed_indices : np.ndarray
    """
    if spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{spatial_key}'] not found.")
    if label_key not in adata.obs.columns:
        raise KeyError(f"adata.obs['{label_key}'] not found.")

    coords = np.asarray(adata.obsm[spatial_key])
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"adata.obsm['{spatial_key}'] must be a (n_obs, 2) array.")

    # keep your plotting convention: y=coords[:,0], x=coords[:,1]
    x_coords = coords[:, 1]
    y_coords = coords[:, 0]

    labels = adata.obs[label_key].to_numpy()

    if normalize_to_str:
        labels = labels.astype(str)
        miss_mask = (labels == str(missing_value))
        obs_mask = (labels == str(observed_value)) if observed_value is not None else ~miss_mask
    else:
        miss_mask = (labels == missing_value)
        obs_mask = (labels == observed_value) if observed_value is not None else ~miss_mask

    missing_indices = np.flatnonzero(miss_mask)
    observed_indices = np.flatnonzero(obs_mask)

    if plot:
        plt.figure(figsize=figsize)

        plt.scatter(
            y_coords[missing_indices],
            x_coords[missing_indices],
            c="gray",
            s=s,
            alpha=alpha,
            label=f"Missing ({label_key}={missing_value})",
        )
        plt.scatter(
            y_coords[observed_indices],
            x_coords[observed_indices],
            c="blue",
            s=s,
            alpha=alpha,
            label=f"Observed ({label_key}={observed_value})" if observed_value is not None else "Observed",
        )

        plt.title(title)
        plt.xlabel("Y Coordinate")
        plt.ylabel("X Coordinate")
        plt.legend(bbox_to_anchor=legend_loc)
        plt.show()

    return missing_indices, observed_indices