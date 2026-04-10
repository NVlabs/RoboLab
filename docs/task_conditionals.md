# Task Conditionals

## Run a demo

To run the conditionals test suite:

```bash
python examples/test_conditionals.py --headless
```
The test will use the following test scene:

<img src="images/conditionals_scene.gif" alt="Conditionals scene" style="max-width:600px;">

## Conditionals:
See [`robolab/robolab/core/task/conditionals.py`](../robolab/core/task/conditionals.py) for implementation details.

## Frames in spatial conditions

Spatial conditions (`object_right_of`, `object_left_of`, `object_in_front_of`, `object_behind`) support different frame of reference modes:

- **`frame_of_reference="robot"`** (default): Uses the robot's egocentric perspective
  - X-axis: robot's forward direction
  - Y-axis: robot's left direction
- **`frame_of_reference="world"`**: Uses global world coordinates

The **`mirrored=False`** (default) uses the robot's natural perspective. Set **`mirrored=True`** for a flipped XY perspective, as if viewing the scene from across the robot.

<img src="images/conditionals_frame_overlay.png" alt="Frame of Reference Overlay" style="max-width:600px; width:100%;">

## Contact Force Cone Detection

### `object_on_top` — Stable Support Detection

The `object_on_top` conditional uses physics-based contact force analysis to determine if an object is stably supported on a surface.

#### Mathematical Formulation

Let $\mathbf{f} = [f_x, f_y, f_z]^\top$ be the contact force from surface $B$ acting on object $A$, expressed in the **world frame** (Z-up).

For $A$ to be stably supported on $B$, the force must lie within an **upward cone**:

- **Cone axis**: $\hat{n} = [0, 0, 1]^\top$ (upward direction)
- **Cone half-angle**: $\theta_{\max}$ (default 45°)

**Conditions for stable support:**

```math
\begin{aligned}
\text{1. Meaningful contact:} \quad & \|\mathbf{f}\| > f_{\min} \\
\text{2. Upward force:} \quad & f_z > 0 \\
\text{3. Within cone:} \quad & f_z \geq \|\mathbf{f}\| \cdot \cos(\theta_{\max})
\end{aligned}
```

The cone constraint (3) can be derived from the dot product:

```math
\cos(\theta) = \frac{\mathbf{f} \cdot \hat{n}}{\|\mathbf{f}\|} = \frac{f_z}{\|\mathbf{f}\|} \geq \cos(\theta_{\max})
```

#### Comparison with Geometric Detection

| Function | Method | Use Case |
|----------|--------|----------|
| `object_on_top` | Contact force cone | Stable resting detection (terminations) |
| `object_above` | Bounding box geometry | Position-based checks (lifted above surface) |

#### Usage

```python
# Check if orange is stably resting on plate
object_on_top(env, object="orange", reference_object="plate", require_gripper_detached=True)

# Geometric check (e.g., for lifted detection)
object_above(env, object="orange", reference_object="table", z_margin=0.05)
```

---

## Details
### Logicals
For functions that support logicals, the available logicals are:
- `any`: if at least 1 object satisfies the condition
- `all`: All objects need to satisfy the condition
- `choose`: Given the set of `objects` with size `N`, exactly `K` objects must satisfy the condition.

### Function decorators

#### Atomic Functions
Base functions; can be used for task `Terminations` as well as `subtasks`.

#### Composite Functions
These expand into multiple atomic subtasks. These cannot be used for `Terminations`.
- `pick_and_place(object, container, logical)`: Picks up objects and places them in a container
  - Automatically creates the sequence: grab → move above → drop → verify in container
  - Supports multiple objects with "all" or "any" completion logic
