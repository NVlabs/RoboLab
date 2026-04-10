# Object Catalog Reference

The full object catalog is at `assets/objects/object_catalog.json` (312 objects). Read it to get exact names, dimensions, and USD paths.

## Catalog Entry Format

Each entry has:
```json
{
  "name": "banana",           // prim name used in scene
  "usd_path": "assets/objects/ycb/banana.usd",  // relative path to USD file
  "class": "fruit",           // object category
  "description": "A yellow banana",
  "dims": [0.19, 0.04, 0.04] // bounding box [x, y, z] in meters
}
```

## Object Classes

| Class | Examples |
|-------|----------|
| block | blue_block, green_block, red_block, yellow_block, brick |
| fruit | avocado01, lemon_01, lime01, orange_01, banana, apple |
| vegetable | red_onion, garlic |
| food | bagel, sesame_bagel, spam, chips_can, coffee_can |
| container | bowl, grey_bin, crate, mug, plate, tray |
| dishware | plate_\*, bowl_\* |
| kitchenware | pot, pan, colander, cutting_board |
| utensil | spoon, fork, knife, ladle, measuring_cups, salad_tongs |
| cooking_utensil | spatula, whisk |
| tool | hammer, screwdriver, wrench, pliers |
| toy | rubiks_cube, toy_airplane, foam_brick |
| bottle | soft_scrub |
| condiment | bbq_sauce_bottle, mustard_bottle, mayo_bottle, ranch_dressing |
| stationery | marker, pen, scissors |
| electronics | remote_control, power_drill |
| vase | vase_\* |
| fixture | rack_l04, wireshelving_a01 (large — exclude from normal scenes) |

## Important Notes

- Object `name` must match exactly when used in scenes and tasks
- `dims` are in meters — use these to compute placement spacing
- `usd_path` is relative to the repo root
- For scene USDA, the payload path must be relative to `assets/scenes/` (e.g., `@../objects/ycb/banana.usd@`)
