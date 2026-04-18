# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from enum import IntEnum


def get_status_name(status_code: int) -> str:
    """Convert status code integer to its name."""
    try:
        return StatusCode(status_code).name
    except ValueError:
        return f"UNKNOWN({status_code})"

class StatusCode(IntEnum):
    """Error codes for task failure scenarios (uint16-compatible).

    Success codes: 100-199 (1xx range) - sequential
    Failure codes: 200-299 (2xx range) - categorized by function type:
        - 210-222: PLACED functions (requires gripper dropped) [X1X range]
        - 223-231: IN_CONTACT functions (contact checks) [X2X range]
        - 232-247: Spatial only functions (position/orientation) [X3X range]
        - 248-254: Other functions (misc/state checks) [X4X range]
    """
    OK = 0
    UNKNOWN_SUCCESS = 100
    UNKNOWN_FAILURE = 200

    # ============================================================
    # Success codes - sequential (100-199)
    # ============================================================

    # PLACED functions (success)
    OBJECT_PLACED_IN_CONTAINER_SUCCESS = 101
    OBJECT_PLACED_ON_TOP_SUCCESS = 102
    OBJECT_PLACED_ON_BOTTOM_SUCCESS = 103
    OBJECT_PLACED_ON_CENTER_SUCCESS = 104
    OBJECT_PLACED_LEFT_OF_SUCCESS = 105
    OBJECT_PLACED_RIGHT_OF_SUCCESS = 106
    OBJECT_PLACED_NEXT_TO_SUCCESS = 107
    OBJECT_PLACED_IN_FRONT_OF_SUCCESS = 108
    OBJECT_PLACED_BEHIND_SUCCESS = 109
    OBJECT_PLACED_BELOW_TOP_SUCCESS = 110
    OBJECT_PLACED_BELOW_SUCCESS = 111
    OBJECT_PLACED_ENCLOSED_SUCCESS = 112
    OBJECT_PLACED_OUTSIDE_OF_SUCCESS = 113

    # IN_CONTACT functions (success)
    OBJECT_IN_CONTACT_SUCCESS = 114
    OBJECT_IN_CONTAINER_AND_IN_CONTACT_SUCCESS = 115
    OBJECT_ON_TOP_AND_IN_CONTACT_SUCCESS = 116
    OBJECT_ON_BOTTOM_AND_IN_CONTACT_SUCCESS = 117
    OBJECT_ON_CENTER_AND_IN_CONTACT_SUCCESS = 118
    OBJECT_INSIDE_AND_IN_CONTACT_SUCCESS = 119
    OBJECT_ENCLOSED_AND_IN_CONTACT_SUCCESS = 120
    OBJECT_LEFT_OF_AND_IN_CONTACT_SUCCESS = 121
    OBJECT_RIGHT_OF_AND_IN_CONTACT_SUCCESS = 122

    # Spatial only functions (success)
    OBJECT_AT_SUCCESS = 123
    OBJECT_INSIDE_SUCCESS = 124
    OBJECT_IN_CONTAINER_SUCCESS = 125
    OBJECT_OUTSIDE_OF_SUCCESS = 126
    OBJECT_ENCLOSED_SUCCESS = 127
    OBJECT_ABOVE_SUCCESS = 128
    OBJECT_ABOVE_BOTTOM_SUCCESS = 129
    OBJECT_BELOW_SUCCESS = 130
    OBJECT_BELOW_TOP_SUCCESS = 131
    OBJECT_RIGHT_OF_SUCCESS = 132
    OBJECT_LEFT_OF_SUCCESS = 133
    OBJECT_IN_FRONT_OF_SUCCESS = 134
    OBJECT_BEHIND_SUCCESS = 135
    OBJECT_CENTER_OF_SUCCESS = 136
    OBJECT_NEXT_TO_SUCCESS = 137
    OBJECT_BETWEEN_SUCCESS = 138

    # Other functions (success)
    OBJECT_GRABBED_SUCCESS = 139
    OBJECT_DROPPED_SUCCESS = 140
    WRONG_OBJECT_GRABBED_SUCCESS = 141
    OBJECTS_IN_LINE_SUCCESS = 142
    OBJECTS_STATIONARY_SUCCESS = 143
    OBJECT_UPRIGHT_SUCCESS = 144
    STACKED_SUCCESS = 145

    # ============================================================
    # Failure codes - categorized by type (200-299)
    # ============================================================

    # X1X (210-222): PLACED functions - object placed and not held by gripper
    OBJECT_PLACED_IN_CONTAINER_FAILURE = 210
    OBJECT_PLACED_ON_TOP_FAILURE = 211
    OBJECT_PLACED_ON_BOTTOM_FAILURE = 212
    OBJECT_PLACED_ON_CENTER_FAILURE = 213
    OBJECT_PLACED_LEFT_OF_FAILURE = 214
    OBJECT_PLACED_RIGHT_OF_FAILURE = 215
    OBJECT_PLACED_NEXT_TO_FAILURE = 216
    OBJECT_PLACED_IN_FRONT_OF_FAILURE = 217
    OBJECT_PLACED_BEHIND_FAILURE = 218
    OBJECT_PLACED_BELOW_TOP_FAILURE = 219
    OBJECT_PLACED_BELOW_FAILURE = 220
    OBJECT_PLACED_ENCLOSED_FAILURE = 221
    OBJECT_PLACED_OUTSIDE_OF_FAILURE = 222

    # X2X (223-231): IN_CONTACT functions - spatial with contact requirement
    OBJECT_IN_CONTACT_FAILURE = 223
    OBJECT_IN_CONTAINER_AND_IN_CONTACT_FAILURE = 224
    OBJECT_ON_TOP_AND_IN_CONTACT_FAILURE = 225
    OBJECT_ON_BOTTOM_AND_IN_CONTACT_FAILURE = 226
    OBJECT_ON_CENTER_AND_IN_CONTACT_FAILURE = 227
    OBJECT_INSIDE_AND_IN_CONTACT_FAILURE = 228
    OBJECT_ENCLOSED_AND_IN_CONTACT_FAILURE = 229
    OBJECT_LEFT_OF_AND_IN_CONTACT_FAILURE = 230
    OBJECT_RIGHT_OF_AND_IN_CONTACT_FAILURE = 231

    # X3X (232-247): Spatial only - position/orientation checks without contact
    OBJECT_AT_FAILURE = 232
    OBJECT_INSIDE_FAILURE = 233
    OBJECT_IN_CONTAINER_FAILURE = 234
    OBJECT_OUTSIDE_OF_FAILURE = 235
    OBJECT_ENCLOSED_FAILURE = 236
    OBJECT_ABOVE_FAILURE = 237
    OBJECT_ABOVE_BOTTOM_FAILURE = 238
    OBJECT_BELOW_FAILURE = 239
    OBJECT_BELOW_TOP_FAILURE = 240
    OBJECT_RIGHT_OF_FAILURE = 241
    OBJECT_LEFT_OF_FAILURE = 242
    OBJECT_IN_FRONT_OF_FAILURE = 243
    OBJECT_BEHIND_FAILURE = 244
    OBJECT_CENTER_OF_FAILURE = 245
    OBJECT_NEXT_TO_FAILURE = 246
    OBJECT_BETWEEN_FAILURE = 247

    # X4X (248-259): Other functions - miscellaneous state checks
    OBJECT_GRABBED_FAILURE = 248
    OBJECT_DROPPED_FAILURE = 249
    WRONG_OBJECT_GRABBED_FAILURE = 250
    OBJECTS_IN_LINE_FAILURE = 251
    OBJECTS_STATIONARY_FAILURE = 252
    OBJECT_UPRIGHT_FAILURE = 253
    STACKED_FAILURE = 254
    GRIPPER_HIT_TABLE = 255
    GRIPPER_FULLY_CLOSED = 256
    WRONG_OBJECT_DETACHED = 257

    # Non-target object displacement events (by severity)
    OBJECT_BUMPED = 258          # Small movement (< 0.5m), minor collision
    OBJECT_MOVED = 259           # Medium movement (0.5m - 1.5m), significant displacement
    OBJECT_OUT_OF_SCENE = 260    # Large movement (> 1.5m), fell off table or knocked far

    # Object movement state transitions
    OBJECT_STARTED_MOVING = 261  # Object transitioned from stationary to moving

    # Additional grasp/manipulation events
    OBJECT_TIPPED_OVER = 262       # Object that should be upright has fallen over
    TARGET_OBJECT_DROPPED = 263    # Target object was grabbed but is now dropped mid-transport
    GRIPPER_HIT_OBJECT = 264       # Gripper collided with an object (not table)
    MULTIPLE_OBJECTS_GRABBED = 265 # Gripper is in contact with multiple objects

    # ============================================================
    # Legacy aliases for backward compatibility
    # ============================================================
    OBJECT_ABOVE_TOP_SURFACE_SUCCESS = 128  # Same as OBJECT_ABOVE_SUCCESS
    OBJECT_ABOVE_TOP_SURFACE_FAILURE = 237  # Same as OBJECT_ABOVE_FAILURE
    OBJECT_ABOVE_BOTTOM_SURFACE_SUCCESS = 129  # Same as OBJECT_ABOVE_BOTTOM_SUCCESS
    OBJECT_ABOVE_BOTTOM_SURFACE_FAILURE = 238  # Same as OBJECT_ABOVE_BOTTOM_FAILURE
    WRONG_OBJECT_GRABBED = 250  # Same as WRONG_OBJECT_GRABBED_FAILURE

    @classmethod
    def subtask_to_success(cls, subtask_name: str):
        if getattr(cls, f"{subtask_name.upper()}_SUCCESS", None) is not None:
            return cls[f"{subtask_name.upper()}_SUCCESS"]
        else:
            return cls.UNKNOWN_SUCCESS

    @classmethod
    def subtask_to_error(cls, subtask_name: str):
        if getattr(cls, f"{subtask_name.upper()}_FAILURE", None) is not None:
            return cls[f"{subtask_name.upper()}_FAILURE"]
        else:
            return cls.UNKNOWN_FAILURE


# Subset of StatusCodes classified as runtime events worth tallying in a
# run summary (wrong grabs, collisions, displacements, etc.). Consumed by
# ``robolab.eval.summarize.extract_events_from_log``.
EVENT_STATUS_CODES: set[StatusCode] = {
    StatusCode.WRONG_OBJECT_GRABBED_FAILURE,
    StatusCode.GRIPPER_HIT_TABLE,
    StatusCode.WRONG_OBJECT_DETACHED,
    StatusCode.OBJECT_BUMPED,
    StatusCode.OBJECT_MOVED,
    StatusCode.OBJECT_OUT_OF_SCENE,
    StatusCode.OBJECT_TIPPED_OVER,
    StatusCode.TARGET_OBJECT_DROPPED,
    StatusCode.GRIPPER_HIT_OBJECT,
    StatusCode.MULTIPLE_OBJECTS_GRABBED,
    StatusCode.GRIPPER_FULLY_CLOSED,
}
