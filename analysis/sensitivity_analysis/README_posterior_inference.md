# Posterior Inference with MNPE

This script implements **Mixed Neural Posterior Estimation (MNPE)** for Bayesian inference with mixed continuous and categorical parameters.

## Table of Contents

1. [Overview](#overview)
2. [What is MNPE?](#what-is-mnpe)
3. [Key Features](#key-features)
4. [Quick Start](#quick-start)
5. [Usage Examples](#usage-examples)
6. [Command-Line Arguments](#command-line-arguments)
7. [Advanced Features](#advanced-features)
8. [Output and Visualization](#output-and-visualization)
9. [Troubleshooting](#troubleshooting)
10. [Technical Details](#technical-details)

---

## Overview

**Goal**: Answer questions like *"What parameter values lead to successful outcomes?"*

Given experimental or simulation data, this script learns the relationship between:
- **Parameters (θ)**: Settings you control (e.g., lighting intensity, camera pose, table material)
- **Observations (x)**: Outcomes you measure (e.g., success rate, task duration)

The script trains a neural network to approximate the **posterior distribution** p(θ|x), which tells you which parameter values are most likely given observed outcomes.

---

## What is MNPE?

**Mixed Neural Posterior Estimation (MNPE)** is a simulation-based inference method that:

1. **Handles mixed data types**: Both continuous (e.g., position, angle) and categorical (e.g., material type, lighting color) parameters
2. **Learns from data**: Trains a neural network on your experimental/simulation data
3. **Performs inference**: Answers queries like "What parameters give success=1?"
4. **Is model-agnostic**: Doesn't require analytical likelihood functions

**Key Advantage**: Unlike traditional methods, MNPE natively handles categorical parameters without approximations or rounding.

**Reference**: [sbi documentation](https://sbi.readthedocs.io/en/latest/reference/_autosummary/sbi.inference.MNPE.html)

---

## Key Features

### Core Capabilities

✅ **Mixed Parameter Types**: Continuous and categorical parameters  
✅ **Uniform Priors**: Non-informative priors for unbiased inference  
✅ **Importance Sampling**: Corrects for biased training data  
✅ **GPU Acceleration**: ~5-10x faster training on CUDA  
✅ **Flexible Queries**: Manual observation values or dataset averages  

### Automatic Data Processing

✅ **Camera Pose Distances**: Auto-computes weighted pose distances for `*_cam_initial_pose` columns  
✅ **Object Pose Distances**: Extracts distance-to-origin from `*_initial_pose` columns  
✅ **Categorical Encoding**: Handles string categories, boolean values, and numeric codes  
✅ **Data Filtering**: Filter by experiment, policy, and task  

### Visualization

✅ **Mixed Pairplots**: Intelligently combines scatter plots, KDE plots, and heatmaps  
✅ **KDE Plots**: Smooth density visualization for continuous-only parameters  
✅ **Marginal Distributions**: Shows posterior for each parameter  
✅ **Interactive Filtering**: Displays filter and observation values in figure titles  

---

## Quick Start

### Basic Demo (Generated Data)

```bash
python posterior_inference.py
```

This runs a demo with synthetic data to verify installation.

### With Real Data

```bash
python posterior_inference.py \
    --use-real-data \
    --csv-file data/my_results.csv \
    --param-columns lighting_intensity table_material \
    --categorical-param-columns table_material \
    --obs-columns success \
    --filter-policy pi0
```

---

## Usage Examples

### Example 1: Query "What parameters lead to success?"

```bash
python posterior_inference.py \
    --use-real-data \
    --csv-file data/poses_fixed.csv \
    --param-columns lighting_intensity lighting_color \
    --categorical-param-columns lighting_intensity lighting_color \
    --obs-columns success \
    --obs-values 1.0 \
    --filter-policy pi0 \
    --filter-task BananaInBowlUniformInitPose20cmTask
```

**Interpretation**: Shows which lighting combinations are most likely when success=1.

### Example 2: Continuous Parameters with Camera Poses

```bash
python posterior_inference.py \
    --use-real-data \
    --csv-file data/poses_fixed.csv \
    --param-columns external_cam_initial_pose wrist_cam_initial_pose \
    --obs-columns success \
    --obs-values 1.0 \
    --filter-policy pi0 \
    --use-pairplot \
    --pose-distance-beta 1.5
```

**Features used**:
- Auto-computes weighted distances from reference poses
- Generates KDE pairplot (continuous-only)
- Custom rotation weight (beta=1.5)

### Example 3: Object Positions and Mixed Parameters

```bash
python posterior_inference.py \
    --use-real-data \
    --csv-file data/poses_fixed.csv \
    --param-columns banana_initial_pose bowl_initial_pose lighting_intensity \
    --categorical-param-columns lighting_intensity \
    --obs-columns success \
    --obs-values 1.0 \
    --filter-experiment-name experiment_001 \
    --use-pairplot
```

**Features used**:
- Extracts distance-to-origin from pose strings
- Mixed visualization (scatter + heatmap)
- Filters by experiment name

### Example 4: Importance Sampling for Biased Data

```bash
python posterior_inference.py \
    --use-real-data \
    --csv-file data/poses_fixed.csv \
    --param-columns lighting_intensity \
    --categorical-param-columns lighting_intensity \
    --obs-columns success \
    --obs-values 1.0 \
    --use-importance-sampling
```

**When to use**: Your training data has non-uniform distribution (e.g., 70% one category, 10% others) but you want posterior under uniform prior.

### Example 5: Fast Debugging

```bash
python posterior_inference.py \
    --use-real-data \
    --csv-file data/poses_fixed.csv \
    --param-columns lighting_intensity \
    --categorical-param-columns lighting_intensity \
    --obs-columns success \
    --max-epochs 5 \
    --print-table \
    --max-rows 30
```

**Debug features**:
- `--print-table`: Shows first 30 rows
- `--max-epochs 5`: Quick training for testing

---

## Command-Line Arguments

### Data Source

| Argument | Type | Description |
|----------|------|-------------|
| `--use-real-data` | flag | Use CSV data instead of generated demo data |
| `--csv-file` | path | Path to CSV file (required with `--use-real-data`) |

### Column Specification

| Argument | Type | Description |
|----------|------|-------------|
| `--param-columns` | list | Parameter column names (space-separated) |
| `--obs-columns` | list | Observation column names (space-separated) |
| `--obs-values` | list | Manual observation values for query (must match `--obs-columns` order) |
| `--categorical-param-columns` | list | Categorical parameter columns (subset of `--param-columns`) |

**Example**:
```bash
--param-columns lighting camera_pose table_material \
--categorical-param-columns table_material \
--obs-columns success duration \
--obs-values 1.0 30.0
```

### Data Filtering

| Argument | Type | Description |
|----------|------|-------------|
| `--filter-experiment-name` | str | Filter to specific experiment name |
| `--filter-policy` | str | Filter to specific policy value |
| `--filter-task` | str | Filter to tasks starting with prefix (first 10 chars) |

### Training Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max-epochs` | int | 50 | Maximum training epochs |
| `--num-simulations` | int | 2000 | Number of simulations (generated data only) |
| `--device` | str | cpu | Device: `cpu` or `cuda` |

### Visualization Options

| Argument | Type | Description |
|----------|------|-------------|
| `--use-pairplot` | flag | Generate mixed pairplot (scatter/KDE + heatmaps) |
| `--annotate-heatmap` | flag | Show percentages in heatmap cells (requires `--use-pairplot`) |

### Advanced Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use-importance-sampling` | flag | | Correct for biased training data distribution |
| `--pose-distance-beta` | float | 1.0 | Rotation weight in camera pose distance metric |

### Debugging Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--print-table` | flag | | Print loaded data table |
| `--max-rows` | int | 20 | Max rows to print in debug mode |

---

## Advanced Features

### 1. Camera Pose Distance Metric

**Automatic Detection**: Columns ending with `_cam_initial_pose` are automatically processed.

**How it works**:
1. Detects 7D pose format: `[x;y;z;qw;qx;qy;qz]`
2. Computes weighted distance from reference pose:
   ```
   distance = ||t₁ - t₂|| + β × geodesic_distance(q₁, q₂)
   ```
3. Creates new continuous parameter: `{column_name}_distance`

**Supported columns**:
- `external_cam_initial_pose` → reference: `[0.05, 0.57, 0.66, 0.195, -0.393, 0.805, -0.399]`
- `wrist_cam_initial_pose` → reference: `[0.286, -0.031, 0.461, 0.114, 0.704, 0.692, 0.110]`

**Control rotation sensitivity** with `--pose-distance-beta`:
- `beta=0.0`: Only translation matters
- `beta=1.0`: Equal weight (default)
- `beta=2.0`: Rotation dominates

**Example**:
```bash
--param-columns external_cam_initial_pose wrist_cam_initial_pose \
--pose-distance-beta 1.5
```

### 2. Object Pose Distance Extraction

**Automatic Detection**: Columns matching `banana_initial_pose` or `bowl_initial_pose`.

**How it works**:
1. Parses 3D position: `[px;py;pz;...]`
2. Computes Euclidean distance to origin: `√(px² + py² + pz²)`
3. Creates continuous parameter: `{object}_distance`

**Example**: `banana_initial_pose: [0.1;0.2;0.3;...]` → `banana_distance: 0.374`

### 3. Importance Sampling

**Problem**: Training data has biased distribution (e.g., 70% success, 30% failure) but you want posterior under uniform prior.

**Solution**: Importance sampling reweights posterior samples to correct the bias.

**When to use**:
- Training data is imbalanced
- You want to know "true" distribution under uniform prior
- You see warnings about low Effective Sample Size (ESS)

**How to use**:
```bash
--use-importance-sampling
```

**Output includes**:
```
✓ Importance sampling applied
  → Effective Sample Size: 311.5 / 1805
  ⚠️  Low ESS - consider collecting more balanced training data
```

**Interpretation**:
- ESS > 500: Good reweighting efficiency
- ESS 100-500: Acceptable, results valid
- ESS < 100: Poor efficiency, collect more balanced data

### 4. Mixed Pairplots with Intelligent Plot Selection

**Feature**: Automatically selects appropriate plot types based on parameter types.

**Plot Types**:

| Parameter Combination | Diagonal | Off-Diagonal |
|----------------------|----------|--------------|
| Continuous only | KDE density | 2D KDE contours |
| Continuous + Categorical | Histogram | Scatter or Binned Heatmap |
| Categorical only | Bar plot | Heatmap |

**Usage**:
```bash
--use-pairplot              # Generate mixed pairplot
--annotate-heatmap          # Show percentages in heatmaps
```

**Examples**:

*Continuous only (2 camera poses)*:
```bash
--param-columns external_cam_initial_pose wrist_cam_initial_pose --use-pairplot
```
→ KDE density curves + filled contours (like `seaborn.pairplot(kind="kde")`)

*Mixed (1 continuous + 1 categorical)*:
```bash
--param-columns external_cam_initial_pose lighting_intensity \
--categorical-param-columns lighting_intensity --use-pairplot
```
→ Histograms + scatter plots + binned heatmaps

### 5. Uniform Priors

**Default Behavior**: The script uses uniform (non-informative) priors:

- **Continuous parameters**: `BoxUniform` over observed data range [min, max]
- **Categorical parameters**: Equal probability (1/K) for each category

**Interpretation**: All parameter values are equally likely *a priori*. The posterior is driven entirely by observed data.

**Why uniform?**: 
- Unbiased: No preference for any parameter value
- Conservative: Lets data speak for itself
- Standard in scientific inference

**Implementation**:
```python
# Continuous: uniform [0, 1] (normalized internally)
cont_prior = BoxUniform(low=torch.zeros(n_cont), high=torch.ones(n_cont))

# Categorical: uniform over K categories
cat_idx = torch.randint(0, K, (batch_size,))  # Equal probability
```

---

## Output and Visualization

### Console Output

**Training Progress**:
```
======================================================================
Training Neural Posterior Estimator
======================================================================
  Using 547 data samples
  Training on: cpu
  
  Epoch 10/50: loss = 0.234
  Epoch 20/50: loss = 0.189
  ...
  ✓ Training complete!
```

**Posterior Analysis**:
```
======================================================================
Posterior Analysis
======================================================================

  Total Samples: 5000
     (Each sample represents one possible parameter combination)

  Continuous Parameters:

    external_cam_initial_pose_distance:
      Mean: 1.48
      Std: 0.90
      95% CI: [0.05, 3.15]

  Categorical Parameters:
     (Showing % of 5000 posterior samples with each value)

    lighting_intensity:
      10:  19.6% (980/5000)
      100:   7.3% (364/5000)
      5000:  69.5% (3477/5000)
      → Most likely: 5000
```

**Interpretation**:
- **Mean/Std/95% CI**: For continuous parameters, shows central tendency and uncertainty
- **Percentages**: For categorical parameters, shows probability of each category
- **5000 samples**: Total posterior samples drawn (fixed)
- **Most likely**: Mode of categorical distribution

### Generated Files

1. **Main visualization**: `mnpe_posterior_results_policy_{policy}_task_{task}_success_{obs}.png`
   - 1D marginal distributions for each parameter
   - Filter values and observations in title

2. **Mixed pairplot** (if `--use-pairplot`): `mixed_pairplot_posterior_policy_{policy}_task_{task}_success_{obs}.png`
   - Diagonal: KDE/histogram/bar plots
   - Off-diagonal: Scatter/KDE contours/heatmaps
   - Shows parameter correlations

### Interpreting Results

**Question**: *"What lighting leads to success?"*

**If you see**:
```
lighting_intensity:
  10:  5.2%
  100: 12.3%
  5000: 82.5%
  → Most likely: 5000
```

**Interpretation**: 
- Given success=1, lighting=5000 is most probable (82.5% posterior probability)
- Lighting=10 is least probable (5.2%)
- Strong evidence that high lighting improves success

**For continuous parameters**:
```
banana_distance:
  Mean: 0.25
  95% CI: [0.10, 0.45]
```

**Interpretation**:
- Given success=1, banana distance is most likely ~0.25m from origin
- 95% credible interval: [0.10, 0.45]m
- Tighter CI = more certain

---

## Troubleshooting

### Issue: "No valid data remaining after removing NaN values"

**Cause**: Missing values or parsing errors in your CSV.

**Solution**:
1. Check for missing values:
   ```bash
   --print-table
   ```
2. Look for columns with NaN in debug output
3. Verify data format:
   - Boolean columns: Use `1`/`0` or `True`/`False`
   - Categorical: Use strings or integers
   - Pose columns: Format as `[x;y;z;...]`

### Issue: "Number of --obs-values must match --obs-columns"

**Cause**: Mismatch between number of observation values and columns.

**Solution**: Count your arguments:
```bash
# Wrong: 2 columns, 1 value
--obs-columns success duration --obs-values 1.0

# Correct: 2 columns, 2 values
--obs-columns success duration --obs-values 1.0 30.0
```

### Issue: Training takes too long

**Solutions**:
1. **Reduce epochs** (for debugging):
   ```bash
   --max-epochs 5
   ```

2. **Use GPU**:
   ```bash
   --device cuda
   ```
   → ~5-10x faster

3. **Filter data**:
   ```bash
   --filter-policy pi0 --filter-task BananaInBowl
   ```
   → Smaller dataset trains faster

### Issue: "Low ESS" warning with importance sampling

**Cause**: Training data distribution is very different from uniform prior.

**Interpretation**: Results are still valid but less efficient (fewer "effective" samples).

**Solutions**:
1. **Acceptable** (ESS > 100): Use results as-is
2. **Low** (ESS < 100): 
   - Collect more balanced training data
   - Or disable importance sampling: remove `--use-importance-sampling`
   - Results may reflect training data bias

### Issue: "RuntimeError: Expected all tensors to be on same device"

**Cause**: Mixed CPU/GPU tensors.

**Solution**: The script handles this automatically. If you see this error:
1. Make sure you're using the latest version
2. Try explicitly setting device: `--device cpu`

### Issue: Posterior looks like prior (uninformative)

**Possible causes**:
1. **Insufficient training**: Increase `--max-epochs` (try 100-200)
2. **Too little data**: Need more CSV rows (ideally 500+)
3. **Weak parameter-observation relationship**: Parameters may not affect observations

**Diagnostic**:
```bash
--print-table  # Check data quality and size
--max-epochs 100  # Train longer
```

---

## Technical Details

### Algorithm

**MNPE (Mixed Neural Posterior Estimation)**:

1. **Prior**: Define uniform prior over parameters
   ```python
   p(θ) = Uniform[θ_min, θ_max]  # for continuous
   p(θ) = 1/K                     # for categorical
   ```

2. **Likelihood**: Learn neural approximation
   ```python
   q(θ|x) ≈ p(θ|x)  # neural network
   ```

3. **Posterior**: Sample from learned distribution
   ```python
   θ_samples ~ q(θ|x_observed)  # MCMC sampling
   ```

### Architecture

- **Density Estimator**: Mixed neural density estimator (continuous + discrete)
- **Training**: Amortized inference (train once, query many times)
- **Sampling**: MCMC with slice sampling for discrete parameters

### Data Requirements

**Minimum**:
- 100+ rows (for debugging)
- 500+ rows (for reasonable results)
- 2000+ rows (for publication-quality)

**Format**:
- CSV with header row
- No missing values in selected columns
- Categorical: strings or integers
- Continuous: floats
- Pose columns: `[x;y;z;...]` format (semicolon-separated)

### Computational Cost

**Training Time** (approximate):

| Data Size | CPU (50 epochs) | GPU (50 epochs) |
|-----------|-----------------|-----------------|
| 100 rows | 30 sec | 10 sec |
| 500 rows | 2 min | 30 sec |
| 2000 rows | 10 min | 2 min |
| 5000 rows | 30 min | 5 min |

**Memory**:
- CPU: ~1-2 GB
- GPU: ~2-4 GB VRAM

### Normalization

**Internal normalization** (transparent to user):
1. Continuous parameters normalized to [0, 1] during training
2. Denormalized back to original scale for visualization
3. You always see and input original units

---

## Best Practices

### 1. Start Simple

```bash
# First, inspect your data
python posterior_inference.py --use-real-data \
    --csv-file data/my_data.csv \
    --print-table

# Then, test with minimal parameters
python posterior_inference.py --use-real-data \
    --csv-file data/my_data.csv \
    --param-columns lighting \
    --obs-columns success \
    --max-epochs 5
```

### 2. Use Filtering

Don't analyze all data at once. Filter by condition:

```bash
# Compare policies separately
for policy in pi0 pi0_fast paligemma; do
    python posterior_inference.py \
        --filter-policy $policy \
        ...
done
```

### 3. Validate Results

Cross-check with domain knowledge:
- Do posterior means make sense?
- Are high-probability regions plausible?
- Do results change with more training epochs?

### 4. Choose Appropriate Beta

For camera pose distances:
- `beta=0.5`: Translation-focused (position matters more)
- `beta=1.0`: Balanced (default)
- `beta=2.0`: Orientation-focused (rotation matters more)

Test different values to see what matters for your task.

### 5. Visualize Relationships

Always use `--use-pairplot` with 2+ parameters:
- Reveals parameter correlations
- Shows joint posterior structure
- Identifies interactions

---

## References

1. **MNPE Paper**: Flexible statistical inference for mechanistic models of neural dynamics (Greenberg et al., 2019)
2. **sbi Library**: [https://sbi.readthedocs.io/](https://sbi.readthedocs.io/)
3. **Simulation-Based Inference**: Cranmer et al., "The frontier of simulation-based inference" (2020)

---

## Getting Help

**If you encounter issues**:

1. Run with `--print-table` to inspect data
2. Try `--max-epochs 5` for fast debugging
3. Check console output for error messages
4. Verify CSV format and column names
5. Start with minimal example and add complexity gradually

**Common workflow**:
```bash
# 1. Inspect
python posterior_inference.py --use-real-data --csv-file data.csv --print-table

# 2. Debug
python posterior_inference.py --use-real-data --csv-file data.csv \
    --param-columns X --obs-columns Y --max-epochs 5

# 3. Production
python posterior_inference.py --use-real-data --csv-file data.csv \
    --param-columns X Y Z --categorical-param-columns Z \
    --obs-columns A B --obs-values 1.0 30.0 \
    --filter-policy pi0 --use-pairplot --device cuda
```

---

**Version**: 2.0  
**Last Updated**: January 2026  
**License**: MIT
