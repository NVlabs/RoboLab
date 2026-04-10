# USD Scene Format Reference

## Scene Structure

All scenes are USDA (ASCII USD) files in `assets/scenes/`. They extend `base_empty.usda` which provides:
- Physics scene configuration
- Table fixture at ~(0.55, 0, -0.35) — table top surface is at Z ≈ 0.0
- Franka table mount
- Ground plane

## Table Geometry

- **Table center**: approximately (0.55, 0.0) in XY
- **Safe placement bounds**: X=[0.30, 0.80], Y=[-0.40, 0.40]
- **Table top Z**: approximately 0.0 (objects placed at Z = half_height + small margin)
- **Table size**: ~0.7m width × ~1.0m depth
- Leave 5cm margin from table edges

## Coordinate System

- Units: meters
- Up axis: Z
- Front: +X direction (away from robot)
- Left: +Y direction
- Right: -Y direction

## Object Prim Format

Each object in a scene is an Xform with a payload reference to its USD file:

```usda
def "banana" (
    prepend payload = @../objects/ycb/banana.usd@
)
{
    quatf xformOp:orient = (1, 0, 0, 0)
    float3 xformOp:scale = (1, 1, 1)
    double3 xformOp:translate = (0.50, -0.12, 0.02)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
}
```

### Key fields:
- **Prim name** (`"banana"`) — must match the object name from the catalog
- **Payload path** — relative from `assets/scenes/` to the object USD (e.g., `@../objects/ycb/banana.usd@`)
- **xformOp:translate** — (x, y, z) position. Z should be `table_top_z + object_half_height + 0.002`
- **xformOp:orient** — quaternion (w, x, y, z). Use (1, 0, 0, 0) for upright
- **xformOp:scale** — usually (1, 1, 1)

## Complete Scene Template

A scene file starts with the base_empty content and adds object prims inside the `def Xform "world" { ... }` block, before the closing `}`.

```usda
#usda 1.0
(
    defaultPrim = "world"
    endTimeCode = 0
    kilogramsPerUnit = 1
    metersPerUnit = 1
    startTimeCode = -1
    upAxis = "Z"
)

def Xform "world"
{
    quatf xformOp:orient = (1, 0, 0, 0)
    float3 xformOp:scale = (1, 1, 1)
    double3 xformOp:translate = (0, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]

    ... (PhysicsScene, PhysicsMaterial, table, franka_table, GroundPlane from base_empty) ...

    def "bowl" (
        prepend payload = @../objects/ycb/bowl.usd@
    )
    {
        quatf xformOp:orient = (1, 0, 0, 0)
        float3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = (0.56, 0.15, 0.03)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    }

    def "banana" (
        prepend payload = @../objects/ycb/banana.usd@
    )
    {
        quatf xformOp:orient = (1, 0, 0, 0)
        float3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = (0.51, -0.12, 0.02)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    }
}
```

## Placement Guidelines

### Object Spacing
- **< 8 objects**: 25cm spacing (0.25m)
- **8-12 objects**: 20cm spacing (0.20m)
- **13-16 objects**: 15cm spacing (0.15m)
- **17+ objects**: 12cm spacing (0.12m)

### Z Height Calculation
For objects placed on the table:
```
z = 0.0 + (object_height / 2) + 0.002
```
where `object_height` is `dims[2]` from the catalog.

### Orientation
- Upright: `quatf xformOp:orient = (1, 0, 0, 0)`
- For random yaw rotation around Z, compute quaternion:
  - `w = cos(yaw/2)`, `z = sin(yaw/2)`, orient = `(w, 0, 0, z)`

## Existing Scene Examples

Reference existing scenes in `assets/scenes/` for patterns:
- `banana_bowl.usda` — 2 objects (simple)
- `rubiks_cube_bowl_banana.usda` — 3 objects
- `bananas_5_grey_bin.usda` — 6 objects (multiple same-type)
