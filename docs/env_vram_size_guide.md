# Per-Task `num_envs` Ceiling on 48 GB VRAM GPUs

This is a report on the maximum `num_envs` each RoboLab benchmark task can run with on a single L40 GPU (48GB VRAM) headlessly. Depending on how much load of the GPU is used for other applications, you may consult this guide as an  **upper bound** for `--num-envs` on RTX GPUs with 48GB memory.

## Summary

## Per-task ceiling (alphabetical within each bin)

### `num_envs = 100` (31 tasks)

- `BagelsOnPlateTask`
- `BananaInBowlTask`
- `BananaOnPlateTask`
- `BananaThenRubiksCubeTask`
- `BlockStackingOrderAgnosticTask`
- `BlockStackingSpecifiedOrderTask`
- `BowlInBinTask`
- `BowlStackingLeftOnRightTask`
- `BowlStackingRightOnLeftTask`
- `ButterAboveRaisinTask`
- `ClampInRightBinTask`
- `FoodPacking1BoxesTask`
- `FoodPacking1CansTask`
- `LargerObjectRaisinBoxInBinTask`
- `MustardAboveRaisinTask`
- `MustardInLeftBinTask`
- `MustardInRightBinTask`
- `NonHammerToolsInRightBinTask`
- `RedDishesInBinTask`
- `RedItemsInBinTask`
- `RubiksCubeAndBananaTask`
- `RubiksCubeBehindBowlTask`
- `RubiksCubeInFrontOfBowlTask`
- `RubiksCubeLeftOfBowlTask`
- `RubiksCubeOrBananaTask`
- `RubiksCubeRightOfBowlTask`
- `RubiksCubeThenBananaTask`
- `SauceBottlesCrateTask`
- `SmallerObjectButterInBinTask`
- `Stack3RubiksCubeTask`
- `ToolOrganizationTask`

### `num_envs = 90` (20 tasks)

- `AnimalsInBinTask`
- `BlocksInBinTask`
- `CleanUpToysTask`
- `CubesAndBlocksInBinTask`
- `GreenSpoonsInPotTask`
- `HammersInLeftBinTask`
- `PickUpBluePitcherTask`
- `PickUpGreenObjectTask`
- `PutBowlOnShelfTopTask`
- `PutMugsOnShelfTask`
- `PutTwoMugsOnShelfTask`
- `RecycleCartonTask`
- `RecycleCartonsVerticalCrateTask`
- `RubiksCubeTask`
- `RubiksCubesInBinTask`
- `StackYellowOnRedTask`
- `TakeMugsOffOfShelfTask`
- `ToolOrganizationBothTask`
- `UnstackRubiksCubeTask`
- `YellowAndWhiteObjectsInBinTask`

### `num_envs = 80` (42 tasks)

- `BananasInBinOneMoreTask`
- `BananasInBinThreeTotalTask`
- `BananasInCrateTask`
- `BananasOutOfBinTask`
- `BigPumpkinInBinTask`
- `BlackItemsInBinTask`
- `ClearOrganicObjectsTask`
- `ClutterPlasticTask`
- `ClutterPumpkinTask`
- `CookingClearPlateTask`
- `CookingPickPastaToolTask`
- `DishesInBinTask`
- `ElectronicsInBinTask`
- `FruitsMovingOrangeOrLimeTask`
- `FruitsMovingTask`
- `FruitsOnionTask`
- `FruitsOnionToPlateTask`
- `GrabABagelTask`
- `GrabAFruitTask`
- `KeyboardOutOfBinTask`
- `MarkerInMugTask`
- `MouseOnKeyboardTask`
- `MoveBananaToBagelPlateTask`
- `PhoneOrRemoteInBinTask`
- `PickDrillTask`
- `PickGlassesTask`
- `PickOrangeObjectTask`
- `ReorientAllMugsTask`
- `ReorientRedMugTask`
- `ReorientWhiteMugsTask`
- `SmallPumpkinInBinTask`
- `SmartphoneInBinTask`
- `SpoonInMugTask`
- `SpoonsInPotTask`
- `StackWhiteMugsTask`
- `TakeMeasuringSpoonOutTask`
- `TakeSpatulaOffShelfTask`
- `ToyInBinTask`
- `UtensilsInMugTask`
- `WhiteMugInCenterOfTableTask`
- `WhiteMugsInBinTask`
- `YogurtInBowlTask`

### `num_envs = 70` (26 tasks)

- `AppleAndYogurtInBowlTask`
- `BBQSauceInBinTask`
- `CannedFoodInBinTask`
- `CoffeePotInBinTask`
- `CondimentsInBinTask`
- `FoodPacking2BoxesTask`
- `FoodPacking2CansTask`
- `FoodPacking3BoxesTask`
- `FoodPacking3CansTask`
- `FoodPackingByColorTask`
- `FruitsGreenLimesOnPlateTask`
- `FruitsOnPlate3Task`
- `FruitsOrangesOnPlateTask`
- `JugsOnShelfTask`
- `OneBottleInSquarePailTask`
- `OneBottleOnShelfTask`
- `PinkSpoonInPotTask`
- `PlasticBottlesInSquarePailTask`
- `RecycleCartonsOnBoxTask`
- `ReorientJugTask`
- `ThrowAwayAppleTask`
- `ThrowAwaySnacksTask`
- `ToolsPickingAllHammersTask`
- `ToolsPickingDrillTask`
- `ToolsPickingHammerTask`
- `WoodSpatulaToBowlTask`

### `num_envs = 60` (1 task)

- `FruitsOnPlateTask`

## How this was measured

OOM signatures recognized by the monitor regex (`osmo workflow logs <id>` tail):

- `out of memory`, `OOMKilled`, `cudaErrorOutOfMemory`, `CUDA error: out of memory` (standard)
- `Warp CUDA error: Failed to get driver entry point.*cuDeviceGetUuid` (Warp init OOM, opaque)
- `ERROR_OUT_OF_DEVICE_MEMORY`, `vkAllocateMemory failed`, `Out of GPU memory allocating resource`, `Failed to execute RenderGraph`, `HydraEngine::render failed` (Vulkan render-graph silent-hang)
- `omni.physx.tensors.*CUDA error: an illegal memory access` (PhysX tensor corruption at high envs)
- `PhysX error: GPU.*Kernel fail to launch` (PhysX narrow-phase kernel launch failure)

## When to consult this guide

- Setting `--num-envs` on a L40 — pick the per-task ceiling.
- Sizing a **batch** of tasks under one workflow — use `min(ceiling)` across the batch.
- Sizing the same task on a different GPU — these numbers are L40-specific.
