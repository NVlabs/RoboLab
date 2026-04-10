# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Demonstration of Mixed Neural Posterior Estimation (MNPE) using the sbi library.

MNPE is designed for inference with MIXED PARAMETERS (continuous + discrete/categorical).
Documentation: https://sbi.readthedocs.io/en/latest/reference/_autosummary/sbi.inference.MNPE.html

Key Features:
- Handles categorical parameters natively (no continuous approximations)
- Discrete parameters are proper integers
- Works on CPU and GPU
- Fast MCMC sampling

Usage:
    # Default demo with generated data:
    python posterior_inference_mnpe.py

    # With real data from CSV:
    python posterior_inference_mnpe.py --use-real-data \\
        --csv-file data.csv \\
        --param-columns lighting camera_angle table_material \\
        --categorical-param-columns table_material \\
        --obs-columns success_rate task_duration

    # With filtering:
    python posterior_inference_mnpe.py --use-real-data \\
        --csv-file data.csv \\
        --param-columns lighting table_material \\
        --categorical-param-columns table_material \\
        --obs-columns success_rate \\
        --filter-policy pi0 \\
        --filter-task BananaInBowlTableTask

    # Query: "What parameters lead to success?"
    python posterior_inference_mnpe.py --use-real-data \\
        --csv-file data.csv \\
        --param-columns lighting table_material \\
        --categorical-param-columns table_material \\
        --obs-columns duration success \\
        --obs-values 30.0 1.0

IMPORTANT: For MNPE, continuous parameters MUST come first, discrete parameters AFTER.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sbi.inference import MNPE
from sbi.utils import BoxUniform


def simple_simulator(theta):
    """
    Simple simulator with mixed parameters for demonstration.

    Args:
        theta: shape (batch, n_params)
            First dim_continuous params are continuous
            Remaining params are discrete (integers)

    Returns:
        x: observations, shape (batch, n_obs)
    """
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)

    batch_size = theta.shape[0]

    # Example: 2 continuous + 1 discrete parameter
    if theta.shape[1] >= 3:
        lighting = theta[:, 0]  # Continuous [0, 5000]
        camera = theta[:, 1]    # Continuous [0, 180]
        material = theta[:, 2].long()  # Discrete {0, 1, 2}

        # Normalize
        lighting_norm = (lighting - 2500) / 2500
        camera_norm = (camera - 90) / 90

        # Different materials have different performance
        material_base = torch.tensor([0.7, 0.5, 0.3])[material]

        # Observation 1: success rate
        success_logit = (torch.logit(material_base, eps=0.01) +
                          lighting_norm * 0.5 +
                          camera_norm * 0.2 +
                          torch.randn(batch_size) * 0.3)
        success_rate = torch.sigmoid(success_logit)

        # Observation 2: task duration
        duration = (30 + material.float() * 10 -
                    lighting_norm * 5 +
                    torch.randn(batch_size) * 3)
        duration = torch.clamp(duration, 15, 60)

    else:
        # Fallback for 2 parameters
        p1 = theta[:, 0]
        p2 = theta[:, 1] if theta.shape[1] > 1 else torch.zeros_like(p1)

        success_rate = torch.sigmoid(p1 * 0.001 + torch.randn(batch_size) * 0.3)
        duration = 30 + p2 * 0.01 + torch.randn(batch_size) * 5
        duration = torch.clamp(duration, 15, 60)

    x = torch.stack([success_rate, duration], dim=1)
    return x


def load_data_from_csv(csv_path):
    """Load data from CSV file."""
    print(f"\nLoading data from: {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")

    return df


def prepare_data_for_mnpe(df, param_columns, obs_columns,
                          categorical_param_columns=None,
                          filter_policy=None, filter_task=None, filter_experiment_name=None):
    """
    Prepare data for MNPE inference.

    IMPORTANT: Reorders parameters so continuous come first, discrete after.

    Args:
        df: DataFrame with data
        param_columns: List of parameter column names
        obs_columns: List of observation column names
        categorical_param_columns: List of categorical parameter columns
        filter_policy: Optional policy value to filter on
        filter_task: Optional task value to filter on

    Returns:
        theta: shape (n, n_params) - continuous first, discrete after
        x: shape (n, n_obs)
        param_info: dict with parameter metadata
    """
    print("\nPreparing data for MNPE...")

    # Apply filters
    if filter_policy is not None:
        if 'policy' in df.columns:
            df = df[df['policy'] == filter_policy].copy()
            print(f"  Filtered to policy={filter_policy}: {len(df)} rows")
        else:
            print(f"  Warning: 'policy' column not found, skipping filter")

    if filter_task is not None:
        if 'env_name' in df.columns:
            # Only compare first 10 characters of env_name
            env_prefix = filter_task[:10]
            df = df[df['env_name'].astype(str).str[:10] == env_prefix].copy()
            print(f"  Filtered to env_name starting with '{env_prefix}': {len(df)} rows")
        else:
            print(f"  Warning: 'env_name' column not found, skipping filter")

    if filter_experiment_name is not None:
        if 'experiment_name' in df.columns:
            df = df[df['experiment_name'] == filter_experiment_name].copy()
            print(f"  Filtered to experiment_name={filter_experiment_name}: {len(df)} rows")
        else:
            print(f"  Warning: 'experiment_name' column not found, skipping filter")

    if len(df) == 0:
        raise ValueError("No data remaining after filtering!")

    # Special handling: Extract distance from pose columns
    # Handle both banana_initial_pose and bowl_initial_pose
    pose_columns = ['banana_initial_pose', 'bowl_initial_pose']

    def parse_pose_distance(pose_str):
        """
        Parse pose string like '[px;py;pz;...]' and compute distance to origin.
        Returns: sqrt(px^2 + py^2 + pz^2)
        """
        try:
            # Remove brackets and split by semicolon
            pose_str = str(pose_str).strip('[]')
            coords = pose_str.split(';')

            # Extract x, y, z (first 3 coordinates)
            if len(coords) >= 3:
                px = float(coords[0])
                py = float(coords[1])
                pz = float(coords[2])

                # Compute Euclidean distance to origin
                distance = np.sqrt(px**2 + py**2 + pz**2)
                return distance
            else:
                return np.nan
        except (ValueError, AttributeError, IndexError):
            return np.nan

    for pose_col in pose_columns:
        if pose_col in param_columns:
            print(f"\n  Extracting distance from {pose_col}...")

            # Create new column with distance
            distance_col = f'{pose_col}_distance'
            df[distance_col] = df[pose_col].apply(parse_pose_distance)

            # Replace pose column with distance column in param_columns
            param_columns = [col if col != pose_col else distance_col
                            for col in param_columns]

            # Remove from categorical if present (distance is continuous)
            if categorical_param_columns and pose_col in categorical_param_columns:
                categorical_param_columns = [col for col in categorical_param_columns
                                            if col != pose_col]

            print(f"    Created {distance_col} column")
            print(f"    Range: [{df[distance_col].min():.3f}, "
                  f"{df[distance_col].max():.3f}]")
            print(f"    Mean: {df[distance_col].mean():.3f}")

    # Separate continuous and categorical parameters
    if categorical_param_columns is None:
        categorical_param_columns = []

    continuous_param_cols = [c for c in param_columns
                            if c not in categorical_param_columns]
    categorical_param_cols = [c for c in param_columns
                             if c in categorical_param_columns]

    # IMPORTANT: Reorder - continuous first, discrete after
    ordered_param_columns = continuous_param_cols + categorical_param_cols

    print(f"\n  Parameters (total: {len(ordered_param_columns)}):")
    print(f"    Continuous ({len(continuous_param_cols)}): {continuous_param_cols}")
    print(f"    Categorical ({len(categorical_param_cols)}): {categorical_param_cols}")

    # Extract parameters (explicit copy to avoid SettingWithCopyWarning)
    theta_df = df[ordered_param_columns].copy(deep=True)

    # Special handling: Convert lighting_intensity to categorical labels
    if 'lighting_intensity' in theta_df.columns:
        print("\n  Converting lighting_intensity to categorical labels...")
        # Define thresholds

        def intensity_to_label(val):
            val = pd.to_numeric(val, errors='coerce')
            if pd.isna(val):
                return 'unknown'
            elif val < 100:
                return 'low'
            elif val < 2500:
                return 'medium'
            else:
                return 'high'

        theta_df['lighting_intensity'] = theta_df['lighting_intensity'].apply(intensity_to_label)
        print(f"    Unique labels: {theta_df['lighting_intensity'].unique()}")

        # Move lighting_intensity to categorical columns
        if 'lighting_intensity' in continuous_param_cols:
            continuous_param_cols = [c for c in continuous_param_cols if c != 'lighting_intensity']
            categorical_param_cols.append('lighting_intensity')
            ordered_param_columns = continuous_param_cols + categorical_param_cols
            categorical_param_columns.append('lighting_intensity')
            print(f"    Moved lighting_intensity to categorical parameters")

    # Convert remaining continuous columns to numeric (in case they're strings)
    for col in continuous_param_cols:
        theta_df[col] = pd.to_numeric(theta_df[col], errors='coerce')

    # Drop rows with NaN in categorical parameters BEFORE encoding
    # This prevents issues with NaN→'nan' string conversion
    before_drop = len(theta_df)
    for col in categorical_param_cols:
        mask = theta_df[col].notna()
        theta_df = theta_df[mask]

    if len(theta_df) < before_drop:
        print(f"  ⚠️  Dropped {before_drop - len(theta_df)} rows with NaN in categorical parameters")

    if len(theta_df) == 0:
        raise ValueError(
            f"No data remaining after removing NaN values in categorical parameters: {categorical_param_cols}. "
            "Check your data - these columns may have all NaN values."
        )

    # Encode categorical parameters as integers
    param_encoders = {}
    categorical_info = []

    for col in categorical_param_cols:
        # Get unique values (NaN already dropped)
        unique_vals = theta_df[col].astype(str).unique()
        unique_vals = sorted(unique_vals)

        val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
        param_encoders[col] = {'val_to_idx': val_to_idx,
                              'idx_to_val': {idx: val for val, idx in val_to_idx.items()}}

        # Convert to string before mapping
        theta_df[col] = theta_df[col].astype(str).map(val_to_idx)

        categorical_info.append({
            'name': col,
            'num_categories': len(unique_vals),
            'categories': unique_vals
        })

        print(f"      {col}: {len(unique_vals)} categories - {unique_vals}")

    # Extract observations (use same indices as theta_df after dropping NaN)
    obs_df = df.loc[theta_df.index, obs_columns].copy()

    # Debug: Check what we have in observations
    print(f"\n  Debug: Observation data after filtering:")
    for col in obs_columns:
        unique_vals_with_nan = obs_df[col].unique()
        non_nan_count = obs_df[col].notna().sum()
        print(f"    {col}: {non_nan_count}/{len(obs_df)} non-NaN rows, unique values: {unique_vals_with_nan}")

    # Convert boolean/categorical observations to numeric
    for col in obs_columns:
        unique_vals = set(obs_df[col].dropna().unique())

        # Check for 0/1 FIRST (before True/False, since 0==False and 1==True in Python!)
        if unique_vals.issubset({0, 1, '0', '1', 0.0, 1.0}):
            print(f"      {col} already binary (0/1), keeping as numeric")
            obs_df[col] = pd.to_numeric(obs_df[col], errors='coerce')
        # Check if column is boolean (True/False strings)
        elif (obs_df[col].dtype == 'bool' or
              unique_vals.issubset({'True', 'False', 'true', 'false'})):
            print(f"      Converting {col} (boolean) to binary: False→0, True→1")
            obs_df[col] = obs_df[col].astype(str).str.lower().map({'true': 1, 'false': 0})
        else:
            # Convert to numeric
            obs_df[col] = pd.to_numeric(obs_df[col], errors='coerce')

    # Drop rows with NaN values in either parameters or observations
    combined_df = pd.concat([theta_df, obs_df], axis=1)
    before_count = len(combined_df)

    # Check which columns have NaN before dropping
    nan_counts = combined_df.isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]

    if len(cols_with_nan) > 0:
        print(f"\n  Columns with NaN values:")
        for col, count in cols_with_nan.items():
            print(f"    {col}: {count}/{before_count} rows ({100*count/before_count:.1f}%)")

    combined_df = combined_df.dropna()
    after_count = len(combined_df)

    if before_count != after_count:
        print(f"\n  ⚠️  Dropped {before_count - after_count} rows with NaN values")

    if len(combined_df) == 0:
        raise ValueError(
            f"\n❌ No valid data remaining after removing NaN values!\n"
            f"   Started with {before_count} rows after filtering.\n"
            f"   All rows contained NaN in at least one column.\n"
            f"   Columns with NaN: {list(cols_with_nan.keys())}\n\n"
            f"   Suggestions:\n"
            f"   1. Remove columns with many NaN values from --param-columns or --obs-columns\n"
            f"   2. Use --print-table to inspect your data\n"
            f"   3. Try different filtering (--filter-policy, --filter-task) to keep more data\n"
        )

    # Split back into theta and obs (make copies to avoid SettingWithCopyWarning)
    theta_df = combined_df[ordered_param_columns].copy()
    obs_df = combined_df[obs_columns].copy()

    # IMPORTANT FOR MNPE/NPE: Normalize continuous parameters to [0, 1] range
    # This helps neural network training and automatic type detection
    normalization_info = {}
    for col in continuous_param_cols:
        col_min = theta_df[col].min()
        col_max = theta_df[col].max()
        if col_max > col_min:  # Avoid division by zero
            theta_df[col] = (theta_df[col] - col_min) / (col_max - col_min)
            normalization_info[col] = {'min': col_min, 'max': col_max}
            print(f"      Normalized {col}: [{col_min:.2f}, {col_max:.2f}] → [0, 1]")
        else:
            print(f"      Warning: {col} has no variation (all values = {col_min})")
            normalization_info[col] = {'min': col_min, 'max': col_min}

    theta = torch.tensor(theta_df.values, dtype=torch.float32)
    x = torch.tensor(obs_df.values, dtype=torch.float32)

    print(f"\n  Observations ({len(obs_columns)}): {obs_columns}")
    print(f"\n  Final shapes:")
    print(f"    theta: {theta.shape}")
    print(f"    x: {x.shape}")

    param_info = {
        'all_columns': ordered_param_columns,
        'continuous_columns': continuous_param_cols,
        'categorical_columns': categorical_param_cols,
        'categorical_info': categorical_info,
        'encoders': param_encoders,
        'normalization': normalization_info,
        'num_continuous': len(continuous_param_cols),
        'num_categorical': len(categorical_param_cols)
    }

    # Build uniform prior using create_uniform_prior()
    # This ensures consistency between real data and generated data paths.
    #
    # For real data:
    #   - Continuous parameters are normalized to [0, 1] during data prep
    #   - Categorical parameters are encoded as integers [0, num_categories-1]
    #   - Prior is uniform over these transformed spaces
    #   - Results are automatically denormalized back to original scales

    # Build continuous bounds (all normalized to [0, 1])
    continuous_bounds = [(0.0, 1.0) for _ in continuous_param_cols]

    # Build categorical category counts
    num_categories_list = [cat_info['num_categories'] for cat_info in categorical_info]

    # Create uniform prior using the same function as generated data
    # We only need the prior object, not the sampler (data already exists)
    _, prior = create_uniform_prior(continuous_bounds, num_categories_list)

    return theta, x, param_info, prior


def estimate_empirical_prior(theta_tensor, continuous_indices, categorical_indices,
                             categorical_info, continuous_bounds):
    """
    Estimate the empirical prior distribution from training data.

    Used for importance sampling to correct MNPE posteriors when training data
    has a biased distribution (e.g., 70% of one category, 10% of others).

    For continuous: Use Gaussian approximation (mean, std from data)
    For categorical: Use empirical frequencies

    Args:
        theta_tensor: Training data parameters (N x D)
        continuous_indices: Indices of continuous parameters
        categorical_indices: Indices of categorical parameters
        categorical_info: Info about categorical parameters
        continuous_bounds: Bounds for continuous parameters (not used, kept for API consistency)

    Returns:
        log_prob_function: Function that computes log_prob for empirical distribution
    """
    # For categorical: compute empirical frequencies
    categorical_probs = {}
    for i, cat_idx in enumerate(categorical_indices):
        values = theta_tensor[:, cat_idx].long()
        n_categories = categorical_info[i]['num_categories']
        counts = torch.bincount(values, minlength=n_categories).float()
        probs = counts / counts.sum()
        # Add small epsilon to avoid log(0)
        probs = probs + 1e-10
        probs = probs / probs.sum()
        categorical_probs[cat_idx] = probs

    # For continuous: compute mean and std (simplified Gaussian approximation)
    continuous_params = {}
    for cont_idx in continuous_indices:
        values = theta_tensor[:, cont_idx]
        mean = values.mean()
        std = values.std()
        # Add small epsilon to avoid division by zero
        std = max(std, 1e-6)
        continuous_params[cont_idx] = (mean, std)

    def log_prob_empirical(samples):
        """
        Compute log probability under empirical distribution.

        Args:
            samples: (N x D) tensor of parameter values

        Returns:
            log_probs: (N,) tensor of log probabilities
        """
        batch_size = samples.shape[0]
        log_probs = torch.zeros(batch_size)

        # Continuous components (Gaussian KDE approximation)
        for cont_idx in continuous_indices:
            mean, std = continuous_params[cont_idx]
            values = samples[:, cont_idx]
            # Gaussian log prob
            log_probs += -0.5 * ((values - mean) / std) ** 2 - torch.log(std * torch.tensor(2.506628))  # 2.506628 ≈ sqrt(2π)

        # Categorical components
        for cat_idx, probs in categorical_probs.items():
            categories = samples[:, cat_idx].long()
            # Handle out-of-bounds categories
            valid_mask = (categories >= 0) & (categories < len(probs))
            log_probs[~valid_mask] = float('-inf')
            if valid_mask.any():
                log_probs[valid_mask] += torch.log(probs[categories[valid_mask]])

        return log_probs

    return log_prob_empirical


def create_uniform_prior(continuous_bounds, num_categories_list):
    """
    Create uniform priors for MNPE with mixed parameters.

    MNPE doesn't use an explicit prior object like NPE. Instead, you define
    uniform priors by:
    - Continuous: Sample from BoxUniform(low, high)
    - Categorical: Sample uniformly from {0, 1, ..., K-1} using torch.randint

    Args:
        continuous_bounds: List of (low, high) tuples for each continuous param
            Example: [(0, 1000), (0, 180)] for 2 continuous params
        num_categories_list: List of category counts for each categorical param
            Example: [3, 4] for 2 categorical params with 3 and 4 categories

    Returns:
        Tuple of (sample_function, prior_object)
        - sample_function: Function that samples from uniform prior
        - prior_object: BoxUniform prior for all parameters (for build_posterior)

    Example:
        >>> prior_sampler, prior = create_uniform_prior(
        ...     continuous_bounds=[(0, 5000), (0, 180)],
        ...     num_categories_list=[3]
        ... )
        >>> theta = prior_sampler(1000)  # Sample 1000 parameter sets
        >>> posterior = inference.build_posterior(density_estimator, prior=prior)
    """
    # Build bounds for all parameters (continuous + categorical)
    all_lows = []
    all_highs = []

    # Add continuous parameter bounds
    if continuous_bounds:
        for low, high in continuous_bounds:
            all_lows.append(float(low))
            all_highs.append(float(high))

    # Add categorical parameter bounds (as [0, num_categories-1])
    for num_categories in num_categories_list:
        all_lows.append(0.0)
        all_highs.append(float(num_categories - 1))

    # Create BoxUniform prior for all parameters
    prior = BoxUniform(
        low=torch.tensor(all_lows),
        high=torch.tensor(all_highs)
    )

    def sample_prior(num_samples):
        samples = []

        # Sample continuous parameters (uniform over bounds)
        if continuous_bounds:
            lows = torch.tensor([low for low, high in continuous_bounds])
            highs = torch.tensor([high for low, high in continuous_bounds])
            continuous_prior = BoxUniform(low=lows, high=highs)
            continuous_samples = continuous_prior.sample((num_samples,))
            samples.append(continuous_samples)

        # Sample categorical parameters (uniform over categories)
        for num_categories in num_categories_list:
            # torch.randint samples uniformly from {0, 1, ..., num_categories-1}
            categorical_sample = torch.randint(
                0, num_categories, (num_samples, 1)
            ).float()
            samples.append(categorical_sample)

        # Concatenate: continuous first, then discrete
        return torch.cat(samples, dim=1)

    return sample_prior, prior


def main(use_real_data=False, csv_file=None,
         param_columns=None, obs_columns=None, obs_values=None,
         categorical_param_columns=None,
         filter_policy=None, filter_task=None, filter_experiment_name=None,
         print_table=False, max_rows=20,
         max_epochs=50, num_simulations=2000, device="cpu",
         use_importance_sampling=False):
    """
    Main MNPE demonstration.

    Args:
        use_real_data: Use real data from CSV instead of generated
        csv_file: Path to CSV file
        param_columns: List of parameter column names
        obs_columns: List of observation column names
        obs_values: Manual observation values for inference query
        categorical_param_columns: List of categorical parameter columns
        filter_policy: Filter to specific policy value
        filter_task: Filter to specific task value
        filter_experiment_name: Filter to specific experiment name
        print_table: Print loaded data for debugging
        max_rows: Max rows to print when debugging
        max_epochs: Training epochs
        num_simulations: Number of simulations for generated data
        device: 'cpu' or 'cuda'
        use_importance_sampling: Apply importance sampling to correct for biased training data
    """
    print("="*70)
    print("MNPE: Mixed Neural Posterior Estimation")
    print("Inference with Mixed Continuous + Discrete Parameters")
    print("="*70)

    if use_real_data:
        # ==== REAL DATA MODE ====
        print("\nMode: Real Data from CSV")

        if csv_file is None:
            raise ValueError("Must provide --csv-file when using --use-real-data")

        # Load data
        df = load_data_from_csv(csv_file)

        # Print table if requested
        if print_table:
            print(f"\nData table (first {max_rows} rows):")
            print(df.head(max_rows).to_string())
            print(f"\nAvailable columns: {list(df.columns)}")
            print(f"Total rows: {len(df)}")

        # Check if columns specified
        if param_columns is None or obs_columns is None:
            print("\n" + "="*70)
            print("Column specification required for inference!")
            print("="*70)
            print("\nAvailable columns:")
            for col in df.columns:
                print(f"  - {col}")
            print("\nTo run inference, provide:")
            print("  --param-columns <column_names...>")
            print("  --obs-columns <column_names...>")
            print("  --categorical-param-columns <column_names...> (optional)")
            return

        # Prepare data
        theta, x, param_info, target_prior = prepare_data_for_mnpe(
            df, param_columns, obs_columns,
            categorical_param_columns,
            filter_policy, filter_task, filter_experiment_name
        )

    else:
        # ==== GENERATED DATA MODE ====
        print("\nMode: Generated Demo Data")
        print("  Simulating robotics task with mixed parameters:")
        print("    - lighting (continuous: 0-5000)")
        print("    - camera_angle (continuous: 0-180)")
        print("    - table_material (discrete: 0=Oak, 1=Walnut, 2=Bamboo)")

        # Sample training data for MNPE
        # IMPORTANT: Continuous first, discrete after

        # Create uniform prior for mixed parameters
        # Continuous: 2 params with uniform priors
        #   - lighting: uniform over [0, 5000]
        #   - camera_angle: uniform over [0, 180]
        # Categorical: 1 param with 3 categories (uniform probability)
        #   - table_material: {0, 1, 2} with equal probability (1/3 each)
        prior_sampler, target_prior = create_uniform_prior(
            continuous_bounds=[(0.0, 5000.0), (0.0, 180.0)],
            num_categories_list=[3]
        )

        # Sample from uniform prior
        theta = prior_sampler(num_simulations)

        print(f"\n  Generated {num_simulations} parameter sets:")
        print(f"    theta shape: {theta.shape}")
        print(f"    First few samples:")
        print(f"      {'Lighting':<10} {'Camera':<10} {'Material':<10}")
        for i in range(min(5, num_simulations)):
            mat_name = ['Oak', 'Walnut', 'Bamboo'][int(theta[i, 2])]
            print(f"      {theta[i, 0]:<10.1f} {theta[i, 1]:<10.1f} {int(theta[i, 2]):<10} ({mat_name})")

        # Run simulator
        x = simple_simulator(theta)

        print(f"\n  Generated observations:")
        print(f"    x shape: {x.shape}")
        print(f"    Success rate range: [{x[:, 0].min():.3f}, {x[:, 0].max():.3f}]")
        print(f"    Duration range: [{x[:, 1].min():.1f}, {x[:, 1].max():.1f}]")

        param_info = {
            'all_columns': ['lighting', 'camera_angle', 'table_material'],
            'continuous_columns': ['lighting', 'camera_angle'],
            'categorical_columns': ['table_material'],
            'categorical_info': [{
                'name': 'table_material',
                'num_categories': 3,
                'categories': ['Oak', 'Walnut', 'Bamboo']
            }],
            'normalization': {},  # No normalization for generated data
            'num_continuous': 2,
            'num_categorical': 1
        }

    # Move to device
    theta = theta.to(device)
    x = x.to(device)

    # ==== TRAIN MNPE ====
    print("\n" + "="*70)
    print("Training MNPE")
    print("="*70)
    print(f"\n  Device: {device}")
    print(f"  Training samples: {len(theta)}")
    print(f"  Epochs: {max_epochs}")

    # Debug: Show theta statistics
    print(f"\n  Theta statistics (for debugging):")
    for i in range(theta.shape[1]):
        param_name = param_info['all_columns'][i]
        is_categorical = i >= param_info['num_continuous']
        param_type = "categorical" if is_categorical else "continuous"
        print(f"    Column {i} ({param_name}, {param_type}):")
        print(f"      Min: {theta[:, i].min():.2f}, Max: {theta[:, i].max():.2f}")
        print(f"      Unique values: {len(theta[:, i].unique())}")
        if len(theta[:, i].unique()) <= 10:
            print(f"      Values: {sorted(theta[:, i].unique().tolist())}")

    # Choose inference method based on parameter types
    n_categorical = param_info['num_categorical']

    if n_categorical > 0:
        # Use MNPE for mixed continuous + categorical parameters
        print(f"  Using MNPE (mixed parameters: {param_info['num_continuous']} continuous + {n_categorical} categorical)")
        inference = MNPE(device=device)
    else:
        # Use regular NPE for continuous-only parameters
        from sbi.inference import NPE
        print(f"  Using NPE (continuous-only parameters: {param_info['num_continuous']})")
        print(f"  Note: MNPE requires at least one categorical parameter. Using NPE instead.")
        inference = NPE(device=device)

    inference.append_simulations(theta, x)

    print("\n  Training neural network...")
    density_estimator = inference.train(
        max_num_epochs=max_epochs,
        show_train_summary=True
    )
    print("  ✓ Training complete!")

    # ==== BUILD POSTERIOR ====
    print("\n" + "="*70)
    print("Building Posterior")
    print("="*70)

    posterior = inference.build_posterior(density_estimator, prior=target_prior)
    print("  ✓ Posterior built with uniform prior!")

    # ==== PERFORM INFERENCE ====
    print("\n" + "="*70)
    print("Posterior Inference")
    print("="*70)

    # Determine observation values for inference query
    if obs_values is not None:
        # Validate obs_values
        n_obs = x.shape[1]
        if len(obs_values) != n_obs:
            obs_cols = obs_columns if use_real_data else ['success_rate', 'task_duration']
            raise ValueError(
                f"Number of --obs-values ({len(obs_values)}) must match number of observation columns ({n_obs}).\n"
                f"Expected order: {obs_cols}"
            )
        x_o = torch.tensor([obs_values], dtype=torch.float32)
        print(f"\n  Query: What parameters produce these observations?")
    else:
        # Use mean observation as example
        x_o = x.mean(dim=0, keepdim=True)
        print(f"\n  Query: What parameters produce the AVERAGE observations?")

    print(f"\n  Observed data:")
    for i, col in enumerate(obs_columns if use_real_data else ['success_rate', 'task_duration']):
        print(f"    {col}: {x_o[0, i]:.3f}")

    print(f"\n  Sampling from posterior...")
    num_samples = 5000
    samples = posterior.sample((num_samples,), x=x_o, show_progress_bars=True)
    print(f"  ✓ Got {len(samples)} posterior samples")
    print(f"     (Each sample represents one possible parameter combination)")

    # ==== IMPORTANCE SAMPLING (OPTIONAL) ====
    sample_weights = None
    if use_importance_sampling and use_real_data:
        print("\n" + "="*70)
        print("Applying Importance Sampling")
        print("="*70)
        print("\n  MNPE learns from the empirical (biased) training distribution.")
        print("  Importance sampling corrects samples to match the target uniform prior.")

        # Estimate empirical prior from training data
        continuous_indices = list(range(param_info['num_continuous']))
        categorical_indices = list(range(param_info['num_continuous'],
                                        param_info['num_continuous'] + param_info['num_categorical']))

        print(f"\n  Estimating empirical prior from {len(theta)} training samples...")
        log_prob_empirical = estimate_empirical_prior(
            theta.cpu(), continuous_indices, categorical_indices,
            param_info['categorical_info'], []
        )

        # Calculate importance weights
        print(f"  Calculating importance weights...")
        samples_cpu = samples.cpu()

        # Target prior log probability (uniform)
        log_p_target = target_prior.log_prob(samples_cpu)

        # Empirical prior log probability (from training data)
        log_p_empirical = log_prob_empirical(samples_cpu)

        # Log weights = log(target) - log(empirical)
        log_weights = log_p_target - log_p_empirical

        # Check for invalid weights
        valid_mask = torch.isfinite(log_weights)
        n_valid = valid_mask.sum().item()
        n_total = len(log_weights)

        if n_valid < n_total:
            print(f"  ⚠️  {n_total - n_valid}/{n_total} samples have invalid weights (will be excluded)")

        if n_valid == 0:
            print(f"  ❌ No valid weights! Skipping importance sampling.")
            use_importance_sampling = False
        else:
            # Keep only valid samples and weights
            samples = samples_cpu[valid_mask]
            log_weights_valid = log_weights[valid_mask]

            # Normalize weights (for numerical stability, subtract max before exp)
            log_weights_normalized = log_weights_valid - torch.max(log_weights_valid)
            sample_weights = torch.exp(log_weights_normalized)
            sample_weights = sample_weights / sample_weights.sum()

            # Calculate effective sample size
            ess = (1 / (sample_weights**2).sum()).item()

            print(f"\n  ✓ Importance sampling applied!")
            print(f"    Valid samples: {n_valid}/{n_total}")
            print(f"    Effective sample size: {ess:.1f}")
            print(f"    Weight statistics:")
            print(f"      Min:  {sample_weights.min().item():.6f}")
            print(f"      Max:  {sample_weights.max().item():.6f}")
            print(f"      Mean: {sample_weights.mean().item():.6f}")

            if ess < 100:
                print(f"\n  ⚠️  Warning: Low effective sample size ({ess:.1f})!")
                print(f"      This means the training data is very different from the target uniform prior.")
                print(f"      Consider collecting more balanced training data.")

    # ==== ANALYZE RESULTS ====
    print("\n" + "="*70)
    print("Posterior Analysis")
    print("="*70)

    # Print effective sample size
    if sample_weights is not None:
        ess = (1 / (sample_weights**2).sum()).item()
        print(f"\n  Effective Sample Size: {ess:.1f} / {len(samples)}")
        print(f"  (Using importance-weighted samples)")
    else:
        print(f"\n  Total Samples: {len(samples)}")

    n_continuous = param_info['num_continuous']
    n_categorical = param_info['num_categorical']

    # Continuous parameters
    if n_continuous > 0:
        print(f"\n  Continuous Parameters:")
        for i in range(n_continuous):
            param_name = param_info['continuous_columns'][i]
            param_samples_norm = samples[:, i]

            # Denormalize if using real data
            if use_real_data and 'normalization' in param_info:
                if param_name in param_info['normalization']:
                    norm_info = param_info['normalization'][param_name]
                    param_samples = param_samples_norm * (norm_info['max'] - norm_info['min']) + norm_info['min']
                else:
                    param_samples = param_samples_norm
            else:
                param_samples = param_samples_norm

            print(f"\n    {param_name}:")

            # Weighted or unweighted statistics
            if sample_weights is not None:
                # Weighted statistics
                weighted_mean = (param_samples * sample_weights).sum()
                weighted_var = ((param_samples - weighted_mean)**2 * sample_weights).sum()
                weighted_std = torch.sqrt(weighted_var)

                print(f"      Mean: {weighted_mean:.2f} (importance weighted)")
                print(f"      Std: {weighted_std:.2f} (importance weighted)")

                # For weighted quantiles, we'd need to sort and cumsum weights (complex)
                # For now, just show unweighted quantiles with a note
                print(f"      95% CI: [{torch.quantile(param_samples, 0.025):.2f}, "
                      f"{torch.quantile(param_samples, 0.975):.2f}] (unweighted)")
            else:
                # Unweighted statistics
                print(f"      Mean: {param_samples.mean():.2f}")
                print(f"      Std: {param_samples.std():.2f}")
                print(f"      95% CI: [{torch.quantile(param_samples, 0.025):.2f}, "
                      f"{torch.quantile(param_samples, 0.975):.2f}]")

    # Categorical parameters (already integers!)
    if n_categorical > 0:
        print(f"\n  Categorical Parameters:")
        weighting_note = " (importance weighted)" if sample_weights is not None else ""
        print(f"     (Showing % of {len(samples)} posterior samples with each value{weighting_note})")
        for i in range(n_categorical):
            cat_idx = n_continuous + i
            cat_info = param_info['categorical_info'][i]
            param_name = cat_info['name']
            param_samples = samples[:, cat_idx].long()

            print(f"\n    {param_name}:")

            # Calculate probabilities (weighted or unweighted)
            probs = []
            for j in range(cat_info['num_categories']):
                if sample_weights is not None:
                    # Weighted probability
                    mask = (param_samples == j)
                    prob = sample_weights[mask].sum().item()
                else:
                    # Unweighted probability
                    count = (param_samples == j).sum().item()
                    prob = count / len(param_samples)

                probs.append(prob)
                cat_name = cat_info['categories'][j] if use_real_data else f"Category {j}"

                # Show count for unweighted, just percentage for weighted
                if sample_weights is not None:
                    print(f"      {cat_name}: {prob:6.1%}")
                else:
                    count = int(prob * len(param_samples))
                    print(f"      {cat_name}: {prob:6.1%} ({count}/{len(param_samples)})")

            # Find most likely category
            mode_idx = np.argmax(probs)
            mode_name = cat_info['categories'][mode_idx] if use_real_data else f"Category {mode_idx}"
            print(f"      → Most likely: {mode_name}")

    # ==== VISUALIZATION ====
    print("\n" + "="*70)
    print("Creating Visualizations")
    print("="*70)

    n_params = samples.shape[1]
    n_plots = min(n_params, 6)  # Limit to 6 plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(n_plots):
        if i < n_continuous:
            # Continuous parameter
            param_name = param_info['continuous_columns'][i]
            param_samples_norm = samples[:, i]

            # Denormalize for visualization
            if use_real_data and 'normalization' in param_info:
                if param_name in param_info['normalization']:
                    norm_info = param_info['normalization'][param_name]
                    param_samples = param_samples_norm * (norm_info['max'] - norm_info['min']) + norm_info['min']
                else:
                    param_samples = param_samples_norm
            else:
                param_samples = param_samples_norm

            # Create histogram with percentage y-axis
            data = param_samples.cpu().numpy()

            # Apply importance sampling weights if available
            if sample_weights is not None:
                hist_weights = (sample_weights * 100).cpu().numpy()
                weighted_mean = (param_samples * sample_weights).sum()
                mean_label = f"Mean: {weighted_mean:.1f} (weighted)"
            else:
                hist_weights = 100 * np.ones_like(data) / len(data)
                weighted_mean = param_samples.mean()
                mean_label = f"Mean: {weighted_mean:.1f}"

            axes[i].hist(data, bins=50, weights=hist_weights,
                        alpha=0.7, edgecolor='black')
            axes[i].axvline(weighted_mean.cpu(), color='red',
                           linestyle='--', label=mean_label)
            axes[i].set_xlabel(param_name)
            axes[i].set_ylabel('Percentage (%)')
            axes[i].set_title(f'{param_name} Posterior')
            axes[i].legend()
        else:
            # Categorical parameter
            cat_idx_in_list = i - n_continuous
            if cat_idx_in_list < n_categorical:
                cat_info = param_info['categorical_info'][cat_idx_in_list]
                param_samples = samples[:, i].long()

                # Calculate percentages (weighted or unweighted)
                percentages = []
                for j in range(cat_info['num_categories']):
                    if sample_weights is not None:
                        # Weighted percentage
                        mask = (param_samples == j)
                        pct = sample_weights[mask].sum().item() * 100
                    else:
                        # Unweighted percentage
                        count = (param_samples == j).sum().item()
                        pct = 100 * count / len(param_samples)
                    percentages.append(pct)

                labels = (cat_info['categories'] if use_real_data
                         else [f"Cat {j}" for j in range(cat_info['num_categories'])])

                axes[i].bar(range(cat_info['num_categories']), percentages,
                           tick_label=labels, alpha=0.7, edgecolor='black')
                axes[i].set_xlabel(cat_info['name'])
                axes[i].set_ylabel('Percentage (%)')

                title_suffix = " (importance weighted)" if sample_weights is not None else ""
                axes[i].set_title(f'{cat_info["name"]} Posterior{title_suffix}')
                axes[i].tick_params(axis='x', rotation=45)

    # Hide unused subplots
    for i in range(n_plots, 6):
        axes[i].axis('off')

    # Build title with observations and filters
    title_parts = []

    # Add filter information if present
    if use_real_data and (filter_experiment_name or filter_policy or filter_task):
        filter_parts = []
        if filter_experiment_name:
            filter_parts.append(f"experiment={filter_experiment_name}")
        if filter_policy:
            filter_parts.append(f"policy={filter_policy}")
        if filter_task:
            filter_parts.append(f"task={filter_task}")
        title_parts.append(" | ".join(filter_parts))

    # Add observation values
    obs_cols = obs_columns if use_real_data else ['success_rate', 'task_duration']
    obs_text = "Observed: " + ", ".join([f"{col}={x_o[0, i].item():.3f}"
                                          for i, col in enumerate(obs_cols)])
    title_parts.append(obs_text)

    # Combine all title parts
    full_title = "\n".join(title_parts)
    fig.suptitle(full_title, fontsize=13, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for suptitle

    # Build filename with filter and observation information
    filename_parts = ['mnpe_posterior_results']

    # Add filter information
    if use_real_data and filter_experiment_name:
        exp_short = filter_experiment_name.replace(' ', '_')
        filename_parts.append(f"exp_{exp_short}")
    if use_real_data and filter_policy:
        filename_parts.append(f"policy_{filter_policy}")
    if use_real_data and filter_task:
        task_short = filter_task[:].replace(' ', '_')
        filename_parts.append(f"task_{task_short}")

    # Add observation information
    if use_real_data:
        obs_cols = obs_columns
        for i, col in enumerate(obs_cols):
            obs_val = x_o[0, i].item()
            # Format value: remove decimal if integer, limit precision
            if obs_val == int(obs_val):
                val_str = str(int(obs_val))
            else:
                val_str = f"{obs_val:.2f}".rstrip('0').rstrip('.')
            filename_parts.append(f"{col}_{val_str}")

    # Join parts with underscores and add extension
    filename = '_'.join(filename_parts) + '.png'
    output_file = f'/home/framos/Code/robolab/analysis/{filename}'

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Saved visualization: {filename}")

    # ==== SUMMARY ====
    print("\n" + "="*70)
    print("SUCCESS: MNPE Inference Complete!")
    print("="*70)
    print("\nKey Points:")
    print("  ✓ MNPE handles mixed continuous + discrete parameters natively")
    print("  ✓ Discrete parameters are proper integers (no rounding needed)")
    print("  ✓ Works on CPU and GPU")
    print("  ✓ Fast MCMC sampling")
    if use_importance_sampling and sample_weights is not None:
        ess = (1 / (sample_weights**2).sum()).item()
        print("  ✓ Importance sampling applied to correct for biased training data")
        print(f"    → Effective Sample Size: {ess:.1f} / {len(samples)}")
        if ess < 100:
            print(f"    ⚠️  Low ESS - consider collecting more balanced training data")
    print("\nThis is the recommended approach for categorical parameters in SBI!")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MNPE posterior inference (mixed continuous + discrete parameters)"
    )

    # Data source
    parser.add_argument("--use-real-data", action="store_true",
                       help="Use real data from CSV instead of generated data")
    parser.add_argument("--csv-file", type=str,
                       help="Path to CSV file with real data")

    # Column specification
    parser.add_argument("--param-columns", type=str, nargs="*",
                       help="List of parameter column names")
    parser.add_argument("--obs-columns", type=str, nargs="*",
                       help="List of observation column names")
    parser.add_argument("--obs-values", type=float, nargs="*",
                       help="Manual observation values for inference query (must match --obs-columns order). "
                            "Example: --obs-values 30.0 1.0 for duration=30.0, success=1.0")
    parser.add_argument("--categorical-param-columns", type=str, nargs="*", default=[],
                       help="List of categorical parameter columns (subset of --param-columns)")

    # Filtering
    parser.add_argument("--filter-experiment-name", type=str,
                       help="Filter data to specific experiment name")
    parser.add_argument("--filter-policy", type=str,
                       help="Filter data to specific policy value")
    parser.add_argument("--filter-task", type=str,
                       help="Filter data to tasks starting with this prefix (compares first 10 characters)")

    # Debugging
    parser.add_argument("--print-table", action="store_true",
                       help="Print loaded data table for debugging")
    parser.add_argument("--max-rows", type=int, default=20,
                       help="Max rows to print when debugging")

    # Training
    parser.add_argument("--max-epochs", type=int, default=50,
                       help="Maximum training epochs")
    parser.add_argument("--num-simulations", type=int, default=2000,
                       help="Number of simulations for generated data")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device to use for training")

    # Advanced options
    parser.add_argument("--use-importance-sampling", action="store_true",
                       help="Apply importance sampling to correct for biased training data. "
                            "Useful when training data has non-uniform distribution (e.g., 70%% one category, 10%% others). "
                            "Only applies to real data.")

    args = parser.parse_args()

    main(
        use_real_data=args.use_real_data,
        csv_file=args.csv_file,
        param_columns=args.param_columns,
        obs_columns=args.obs_columns,
        obs_values=args.obs_values,
        categorical_param_columns=args.categorical_param_columns,
        filter_experiment_name=args.filter_experiment_name,
        filter_policy=args.filter_policy,
        filter_task=args.filter_task,
        print_table=args.print_table,
        max_rows=args.max_rows,
        max_epochs=args.max_epochs,
        num_simulations=args.num_simulations,
        device=args.device,
        use_importance_sampling=args.use_importance_sampling
    )
