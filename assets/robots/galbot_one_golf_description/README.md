# Galbot One Golf Description

USD description of the Galbot One Golf robot, used by the RoboLab Galbot Golf
embodiment (`robolab/robots/galbot_golf.py`).

## Checkout

This package was imported from the official Galbot description repository:

- Source: `https://github.com/GalaxyGeneralRobotics/galbot_one_golf_description`
- Revision: `b311f5ca1acf506e9b7026397e2c74fb2db11df6`

RoboLab retains only the USD assets, which are self-contained (meshes are
embedded in `usd/payloads/geometries.usd`; textures live under
`usd/Textures/`). The upstream URDF, MJCF, xacro, mesh, rviz, and launch
files are not used by RoboLab and were dropped. Use the source checkout at
the revision above when you need them (e.g. URDF for planning tools, or
regenerating descriptions from xacro).

## USD

The main USD entry point is `usd/galbot_one_golf.usda`. Related payloads and
textures are stored under `usd/`.

Variant sets on the root prim (defaults in bold): Physics (**physx**,
physics, none, mujoco), Robot (**none**, robot), Sensor (**none**, sensors).

The `d405` visuals use the forward `d405_sz073` assembly from description
revision `69cdfde0178f2eed10c4b2e612c0c3d3b2263c7a`. Its built-in stand replaces
the hidden original stand; collision geometry and physics are unchanged.
Wrist camera parameters match SynthNova's Yundonghui calibration at 640 x 360.
The standard IsaacLab camera implementation is retained, including its
runtime square-pixel assumption (`fy=fx`).

### Wrist camera comparison

These RGB captures use the same `BananaInBowlTask` reset state, robot root and
joint positions on Isaac Sim 5.1 / Isaac Lab 2.3.2.post1. Five render updates
were applied without stepping physics. Before uses `c6beb88` at 400 x 224;
after uses the forward D405 assembly and calibration at 640 x 360. The changed
framing comes from the camera calibration, rather than a different robot pose.
The standard camera reports `fx=fy=323.2`, `cx=320`, `cy=180` at runtime.

| Camera | Before | After |
|---|---|---|
| Left wrist | ![Left wrist before](../../../docs/images/robots/galbot_wrist_left_before.png) | ![Left wrist after](../../../docs/images/robots/galbot_wrist_left_after.png) |
| Right wrist | ![Right wrist before](../../../docs/images/robots/galbot_wrist_right_before.png) | ![Right wrist after](../../../docs/images/robots/galbot_wrist_right_after.png) |

## Fixed-base control profile

`robolab/robots/galbot_golf.py` uses the SynthNova
`galbot-one-golf-sim2real-v1` profile, from
`src/synthnova/robots/cfg/galbot_one_golf.toml` at revision
`24ae63bce2e5b5909f7e95228fb4d1cd176c6974`.
Arm stiffness and damping are averaged between the measured left/right values,
then rounded to the nearest integer. Other gains are rounded directly, with
half values rounded up (e.g. gripper stiffness 76.5 becomes 77).

| Arm joint | Source left stiffness / damping | Source right stiffness / damping | Applied to both arms |
|---|---|---|---|
| 1 | 3483.76 / 97.0301 | 3600.0 / 102.566 | 3542 / 100 |
| 2 | 3395.34 / 94.5673 | 3480.33 / 96.9345 | 3438 / 96 |
| 3 | 1144.9 / 31.8879 | 1144.52 / 31.8773 | 1145 / 32 |
| 4 | 1141.44 / 31.7915 | 1144.34 / 31.8723 | 1143 / 32 |
| 5 | 285.424 / 7.94966 | 286.78 / 7.98743 | 286 / 8 |
| 6 | 283.771 / 7.90362 | 283.84 / 7.90554 | 284 / 8 |
| 7 | 283.654 / 7.90037 | 283.654 / 7.90037 | 284 / 8 |

Stiffness is in N m/rad and damping in N m s/rad. Effort limits are 433/433/204/70/70
N m for the legs, 4 N m for the head, 180/180/90/90/30/30/30 N m per arm, and
1.5 N m per gripper. Specified profile velocity limits are retained. The source
does not specify head or gripper speed: the head retains its USD limit, and the
grippers retain RoboLab's existing 3.5 rad/s limit rather than falling back to
the USD's 0.5 rad/s. This preserves the existing closing-time limit; 3.5 rad/s
is not a speed calibration from the source profile. Wheels retain their USD
drive properties.

The robot root stays fixed. Robot-link gravity is disabled to approximate ideal
controller-side gravity compensation; gravity on task objects is unchanged.
This changes tracking/contact dynamics and can change benchmark results. It is
not a floating-base balancing or mobile-base gravity-compensation controller.

## LICENSE

This software is licensed under the Apache License 2.0. See `LICENSE` for details.

## Fingertip contact regions

Fingertip collisions use upstream revision `46c0b7447edc8254e8b44dd669df86fb6985d876`,
packed into the existing `usd/payloads/geometries.usd`. The 1 mm inner pads use
static/dynamic friction `1.5/1.5`; outer shells use `0.3/0.2`, with zero restitution.
The collision envelope and gripper opening are preserved.
