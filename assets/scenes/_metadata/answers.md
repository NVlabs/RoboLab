### Shared conventions to confirm (applies to many tasks)
- Left/right/behind/ in-front checks use the robot frame: `frame_of_reference="robot"` and `mirrored=False`. Confirm this is correct globally.
- Shelf containers: confirm which prim names correspond to each shelf (e.g., `wireshelving_a01`, `rack_l04`, `sm_rack_m01`, `heavydutysteelshelving_a01`) and which one is “top shelf.”
- Tolerances: for “outside of” container checks, examples use `tolerance=0.00`. Confirm if that’s required globally.

### MugOnShelfVagueTask
- Scene: `wire_shelf_mugs_plate_spatula.usda`
- Instruction: “Put a mug on shelf”
- Key entities: `ceramic_mug`, shelf container `wireshelving_a01`
- Completion check: `object_placed_in_container(object=["ceramic_mug"], container="wireshelving_a01", logical="any")`
- Subtasks shape: `pick_and_place(object=["ceramic_mug"], container="wireshelving_a01", logical="any")`
- Questions:
  - Which mug ID(s) should count? Just `ceramic_mug`, or also `mug`, `mug_01`? ceramic_mug or mug any should be placed on the shelf, use or logical.
  - Confirm shelf container prim for this scene: is it `wireshelving_a01`? Yes.

### MugOnShelfSpecificTask
- Scene: `wire_shelf_mugs_plate_spatula.usda`
- Instruction: “Put a mug on shelf to the right of X”
- Key entities: `ceramic_mug`, reference object X (example uses `plate_small`), shelf container `wireshelving_a01`
- Completion check:
  - Termination: `object_placed_right_of(object="ceramic_mug", reference_object="<X>", frame_of_reference="robot", mirrored=False)`
  - Subtask conditions also include `object_in_container(..., container="wireshelving_a01")`
- Subtasks shape: one `Subtask` with conditions: grabbed → in shelf → right_of X → dropped
- Questions:
  - What exactly is “X” in this scene? Is it `plate_small`? No, it is fork_small. Remember to write the correct instruction. Remember any of mug or ceramic mug or mug_01 should be okay.
  - Confirm shelf container prim for “on shelf” check (still `wireshelving_a01`)? Yes.

### MugOnShelfMultipleTask (on shelf)
- Scene: `shelf_mugs_jug_bowl.usda`
- Instruction: “Put all mugs on shelf”
- Key entities: mugs `["ceramic_mug", "mug"]`, shelf container `rack_l04`
- Completion check: `object_placed_in_container(object=["ceramic_mug", "mug"], container="rack_l04", logical="all")`
- Subtasks shape: `pick_and_place(object=["ceramic_mug", "mug"], container="rack_l04", logical="all")`
- Questions:
  - Confirm which mug IDs exist/should count. mug and ceramic mug any should be placed on the shelf, use or logical.
  - Confirm which prim is “the shelf” for this scene: is it `rack_l04`? Yes.

### MugOnShelfMultipleTopTask (top shelf)
- Scene: `shelf_mugs_jug_bowl.usda`
- Instruction: “Put all mugs on top shelf”
- Key entities: mugs `["ceramic_mug", "mug"]`, top-shelf container (example uses `rack_l04`)
- Completion check: same as above, but ensure container is the “top shelf”
- Subtasks shape: same as above
- Questions:
  - Confirm which prim corresponds to the “top shelf.” Is it `rack_l04` in this scene? Yes.

### CutleryOutOfContainer2
- Scene: `cutlery_shelf.usda`
- Instruction: “Take 2 cutlery out of container”
- Key entities: `fork_big`, `fork_small`, container `sm_rack_m01`
- Completion check: `object_placed_outside_of(object=["fork_big", "fork_small"], container="sm_rack_m01", logical="all", tolerance=0.00)`
- Subtasks shape: one `Subtask` with `logical="choose", K=2`
- Questions:
  - Is “cutlery” restricted to forks here, or should `spoon_*`/`spatula_*` count? it should be any from fork_big or fork_small or spoon_big or spoon_small, use or logical. 
  - Confirm the container prim: `sm_rack_m01` and if `tolerance=0.00` is required. Yes.

### CutleryOutOfContainer1
- Scene: `cutlery_shelf.usda`
- Instruction: “Take cutlery out of container”
- Key entities: one cutlery item (example: `fork_big`), container `sm_rack_m01`
- Completion check: `object_placed_outside_of(object=["<chosen_cutlery>"], container="sm_rack_m01", logical="any", tolerance=0.00)`
- Subtasks shape: grabbed → outside_of → dropped for 1 item
- Questions:
  - Should any single cutlery item count, or specifically `fork_big`? fork_big or fork_small or spoon_big or spoon_small, use or logical.
  - Confirm container prim and `tolerance=0.00`. Yes.

### ToolOrganization (both bins)
- Scene: `tools_container.usda`
- Instruction: “Put hammers in one bin and everything else in the other bin”
- Key entities: hammers `["hammer_7", "hammer_8"]`, others `["cordless_drill", "spring_clamp"]`, bins `container_f24` and `container_b16`
- Completion check: two `object_placed_in_container` calls: hammers → `container_f24`, others → `container_b16`
- Subtasks shape: two `pick_and_place` subtasks
- Questions:
  - Confirm which bin is left/right and which ID is which (`container_b16` vs `container_f24`). container_f24 is the left bin and container_b16 is the right bin.
  - Confirm hammer IDs and “others” list for this scene. hammer_7 and hammer_8 are the hammers, and cordless_drill and spring_clamp are the others.

### ToolOrganization (left bin)
- Scene: `tools_container.usda`
- Instruction: “Put hammers in the left bin”
- Key entities: hammers `["hammer_7", "hammer_8"]`, left bin (example uses `container_b16`)
- Completion check: hammers in `container_b16`
- Subtasks shape: one `pick_and_place` subtask
- Questions:
  - Confirm which bin is the “left bin” and its prim name. container_b16 is the left bin. container_f24 is the right bin.

### ToolOrganization (specific two hammers to left bin)
- Scene: `tools_container.usda`
- Instruction: “Put the <hammer A> and <hammer B> in the left bin”
- Key entities: `hammer_7`, `hammer_8`, left bin `container_b16`
- Completion check: hammers in `container_b16`
- Subtasks shape: one `pick_and_place` subtask
- Questions:
  - Confirm the exact IDs for the two target hammers and left bin mapping. hammer_7 and hammer_8 are the hammers, and container_b16 is the left bin.

### FoodPackingDense
- Scene: `food_packing_dense.usda`
- Instruction: “Pack boxed foods into a container”
- Key entities: items `["cheez_it", "chocolate_pudding", "spam_can", "sugar_box"]`, destination `bin_a06`
- Completion check: `object_placed_in_container(object=[...], container="bin_a06", logical="all")`
- Subtasks shape: one `pick_and_place` subtask
- Questions:
  - Confirm the exact “boxed” set (example includes `spam_can`, which is canned—should it be included?). cheez_it, chocolate_pudding, sugar_box, coffee_can, tomato_soup_can, tuna_can should be included.
  - Confirm destination container prim (`bin_a06`). Yes.

### FoodPacking (by size)
- Scene: `food_packing.usda`
- Instruction: “Pack smaller objects in one container and larger objects in another”
- Key entities:
  - Small: `["cheez_it", "chocolate_pudding", "sugar_box"]` → `bin_a06`
  - Large: `["coffee_can", "tomato_soup_can"]` → `bin_b03`
- Completion check: two `object_placed_in_container` groups
- Subtasks shape: two `pick_and_place` subtasks
- Questions:
  - Confirm the small vs large sets and the two destination bins (`bin_a06`, `bin_b03`). You need to check all small are in one container and all large are in another container. small are: spam_can, chocolate_pudding, tomato_soup_can. large are: coffee_can, sugar_box, cheez_it, mustard, .

### ColorBlue
- Scene: `blue.usda`
- Instruction: “Pick blue object”
- Key entities: `pitcher`
- Completion check: `object_grabbed(object="pitcher")`
- Subtasks shape: one `Subtask` with grabbed condition
- Questions:
  - Confirm the intended blue target is `pitcher`. Yes.

### LargerBlueObject
- Scene: `blue.usda`
- Instruction: “Pick the larger object”
- Key entities: larger of the blue items (example targets `pitcher`)
- Completion check: `object_grabbed(object="pitcher")`
- Subtasks shape: one `Subtask` with grabbed condition
- Questions:
  - Confirm which is “larger” (is `pitcher` correct for this scene?). Yes. But make instruction to pick large blue object.

### ColorGreen
- Scene: `green.usda`
- Instruction: “Pick green object”
- Key entities: `green_beans_can`
- Completion check: `object_grabbed(object="green_beans_can")`
- Subtasks shape: one `Subtask` with grabbed condition
- Questions:
  - Confirm the intended green target is `green_beans_can`. Yes.

### WorkDeskClearing
- Scene: `workdesk.usda`
- Instruction: “Clear the space in front of X”
- Key entities: X=`keyboard`; example enforces moving `ceramic_mug` and `rubiks_cube` behind the keyboard
- Completion check: `object_placed_behind(object=["ceramic_mug", "rubiks_cube"], reference_object="keyboard", frame_of_reference="robot", mirrored=False, logical="all")`
- Subtasks shape: one `Subtask`, each object: grabbed → dropped (progress trace)
- Questions:
  - Confirm X is `keyboard`. Yes.
  - Should we enforce clearing specific objects (which ones), or any object found in front-of-keyboard? remote_control, glasses, ceramic_mug, marker, foam_roller, rubiks_cube, computer_mouse, lizard_figurine should be cleared.
  - Do we need to ensure no objects remain in-front of the keyboard (i.e., negative constraints)? Yes.

### ClutterToBin
- Scene: `clutter_bin.usda`
- Instruction: “Clear the table and put everything in bin”
- Key entities: bin `container_f14`, objects subset of fruit/produce (example: `["lemon_01", "lemon_02", "lime01", "orange_01", "pomegranate01", "pumpkinsmall"]`)
- Completion check: `object_placed_in_container(object=[...], container="container_f14", logical="all")`
- Subtasks shape: one `pick_and_place` subtask
- Questions:
  - Confirm the full list of objects to include (all clutter vs specific subset). lemon_01, lemon_02, lime01, lime01_01, orange_01, orange_02, pomegranate01, pumpkinlarge, pumpkinsmall, redonion, whitepackerbottle_a01, avocado01, crabbypenholder, milkjug_a01, serving_bowl, utilityjug_a03 should be included.
  - Confirm bin prim is `container_f14`. Yes.

### Shelf Clutter
- Scene: `clutter_shelf.usda`
- Instruction: “Put everything away onto the shelf”
- Key entities: shelf container `heavydutysteelshelving_a01`, object subset as above
- Completion check: `object_placed_in_container(object=[...], container="heavydutysteelshelving_a01", logical="all")`
- Subtasks shape: one `pick_and_place` subtask
- Questions:
  - Confirm the list of objects to move. lemon_01, lemon_02, lime01, lime01_01, orange_01, orange_02, pomegranate01, pumpkinlarge, pumpkinsmall, redonion, whitepackerbottle_a01, avocado01, crabbypenholder, milkjug_a01, serving_bowl, utilityjug_a03 should be included.
  - Confirm shelf container prim (`heavydutysteelshelving_a01`). Yes.

### LadleToPotTask
- Scene: `ladle_pot.usda`
- Instruction: “Grab ladle from holder and put in pot”
- Key entities: `ladle`, pot `anza_medium`
- Completion check: `object_placed_in_container(object="ladle", container="anza_medium")`
- Subtasks shape: one `pick_and_place` subtask
- Questions:
  - Confirm the pot prim is `anza_medium`. Yes.

### ClearFrontOfShelfTask
- Scene: `front_of_shelf.usda`
- Instruction: “Clear the table area in front of the shelf by moving those objects onto the shelf”
- Key entities: shelf `sm_rack_m01`; objects `["milkjug_a01", "blackandbrassbowl_large", "gardenplanter_large"]`
- Completion check: `object_placed_in_container(object=[...], container="sm_rack_m01", logical="all")`
- Subtasks shape: one `pick_and_place` subtask
- Questions:
  - Confirm the exact object list to clear and the shelf prim `sm_rack_m01`. blackandbrassbowl_large, milkjug_a01, gardenplanter_large should be cleared. Fix the instruction to clear the table area in front of the shelf.

### KetchupLeftMustardRightShelfTask
- Scene: `shelf2_without_condiments.usda`
- Instruction: “Place the ketchup on the left end of the shelf and the mustard on the right end”
- Key entities: `ketchup_bottle`, `mustard`, shelf `rack_l04`
- Completion check:
  - Termination: `object_placed_left_of(object="ketchup_bottle", reference_object="mustard", frame_of_reference="robot", mirrored=False)`
  - Subtasks also ensure both are on the shelf container
- Subtasks shape: one `Subtask`, each item: grabbed → in shelf → correct left/right relation → dropped
- Questions:
  - Confirm “middle shelf” vs any shelf and the shelf container prim to use (`rack_l04`?). sm_rack_m01 is left and rack_l04 is right shelf.
  - Do we need to check placement at the “ends” specifically or just relative left/right ordering? just check both or any shelf and check if they are left-right.

### SwapKetchupMustardMiddleShelfTask
- Scene: `shelf2_with_condiments.usda`
- Instruction: “On the middle shelf, swap ketchup and mustard so ketchup ends on the left and mustard on the right”
- Key entities: `ketchup_bottle`, `mustard`, target shelf container (example references `rack_l04` contextually)
- Completion check: `object_placed_left_of(object="ketchup_bottle", reference_object="mustard", frame_of_reference="robot", mirrored=False)`
- Subtasks shape: one `Subtask`, grabbed → left/right relation → dropped for each
- Questions:
  - Confirm the middle shelf’s prim (is it `rack_l04`?) and whether we must enforce “on middle shelf” in conditions with `object_in_container`.sm_rack_m01 is left and rack_l04 is right shelf. Fix the instruction to swap shelves currently ketchup is left and mustard is on right.

### MoveRedFromLeftShelfToRightShelfTask
- Scene: `shelf2_without_condiments.usda`
- Instruction: “Move all red objects from the left shelf to the right shelf; do not move non-red items or items already on the right shelf”
- Key entities: red items (example uses only `ketchup_bottle`), right shelf container `rack_l04`
- Completion check: `object_in_container(object="ketchup_bottle", container="rack_l04")`
- Subtasks shape: one `Subtask`, grabbed → `object_in_container` → dropped
- Questions:
  - What is the full set of “red objects” in this scene? ketchup
  - Confirm container prims for left vs right shelves. sm_rack_m01 is left and rack_l04 is right shelf.

### GlassesPhoneRemoteLeftToRightTask
- Scene: `workdesk.usda`
- Instruction: “Arrange from left to right: glasses, smartphone, remote control; keep other items where they are”
- Key entities: `glasses`, `smartphone`, `remote_control`
- Completion check: `object_placed_left_of(object=["glasses","smartphone"], reference_object="remote_control", frame_of_reference="robot", mirrored=False, logical="all")`
- Subtasks shape: one `Subtask`, conditions:
  - `glasses` left of `smartphone`
  - `smartphone` left of `remote_control`
- Questions:
  - Confirm left-right is the x-axis in robot frame. Yes.
  - Should we enforce “do not move other items” as an explicit negative constraint or leave it as instruction-only? Do not move other items.

### NestGardenPlantersBySizeTask
- Scene: `garden_planter.usda`
- Instruction: “Nest the small planter inside the medium planter, then nest the medium inside the large planter.”
- Key entities: `gardenplanter_small`, `gardenplanter_medium`, `gardenplanter_large`
- Completion check:
  - `object_placed_in_container(object=["gardenplanter_small"], container="gardenplanter_medium")`
  - `object_placed_in_container(object=["gardenplanter_medium"], container="gardenplanter_large")`
- Subtasks shape: two `pick_and_place` subtasks
- Questions:
  - Confirm these three prim names match the scene exactly. Yes.
  
Let me know the answers to the questions (especially the exact prim names for shelves/containers/targets and any object lists), and I’ll convert this plan into the actual task files using the exact example format.

- I read and extracted patterns from the example tasks, including scene names and conditionals, and drafted a precise plan with targeted questions to parameterize completion criteria for each new task.