# EEF control and geometry

This is a single DROID Panda arm with a Robotiq gripper. Public measurements
`current_eef.position` and `quaternion_wxyz` describe the Robotiq **base_link**
in robot-root coordinates. This is not the camera frame or the separate,
rotated `eef_frame`. Position units are meters; quaternions are **wxyz**.

# Blog default gripper and camera calibration

Base + gripper + camera specification. Calibration only; action scheduling
remains the original Cosmos review harness.

GRIPPER GEOMETRY
The action pose and ee_pos/ee_quat locate the Robotiq base_link mounting flange. The finger pads extend along flange-local +X. The two opposing pads are separated along flange-local Y; closing reduces this separation. These local directions rotate with ee_quat, while action x/y/z remain robot-root axes.

The following approximate linkage geometry gives the midpoint between the pad centers, t_pad=[d,0,0] metres in the flange frame, and the inner-face gap. g is measured gripper_pos[0], normalized closure (0=open, 1=fully closed). It is different from the binary gripper action requesting open or closed.
g       d (metres)    inner-face gap (metres)
0.00    0.13010       0.0850
0.25    0.13651       0.0666
0.50    0.14101       0.0461
0.75    0.14343       0.0242
1.00    0.14366       0.0018
Interpolate for intermediate openings. Contact can deflect the linkage; these are geometric reference values, not exact contact locations or a grasp detector.

Use p_pad = p_flange + R(ee_quat) * t_pad, and choose a flange target with p_flange_target = desired_pad_position - R(target_quaternion) * t_pad. For the downward wxyz quaternion [0.7071,0,0.7071,0], local +X points downward and local Y points along robot-root +Y. With open fingers, the pads are about 0.13010 m below the flange and have essentially the same robot-root x/y. For example, an illustrative desired pad midpoint [0.40,0.10,0.05] m requires a flange target near [0.40,0.10,0.18010] m at that orientation and opening. At identity quaternion [1,0,0,0], the same local offset points along robot-root +X instead: the flange target for that illustrative pad position is [0.26990,0.10,0.05] m. Use the rotation for other orientations. These examples describe coordinate conversion, not the current object's pose.

CAMERA CALIBRATION AND TILT
These are the nominal fixed DROID RGB camera extrinsics. p_parent = t + R(q)*p_camera; each t is in metres, and each q is wxyz (normalize before computing R). OpenGL camera coordinates have +X image-right, +Y image-up, and -Z viewing-forward. Do not treat camera/image axes as robot-root action axes.

Camera                       Parent      t                       q (wxyz)
wrist_cam                    flange      [0.011,-0.031,-0.074]    [-0.420,0.570,0.576,-0.409]
over_shoulder_left_camera     robot_root  [0.05,0.57,0.66]         [-0.393,-0.195,0.399,0.805]
over_shoulder_right_camera    robot_root  [0.05,-0.57,0.66]        [0.805,0.399,-0.195,-0.393]

For wrist_cam: p_camera_root = ee_pos + R(ee_quat)*t; R_camera_root = R(ee_quat)*R(q). Its pose relative to the flange stays fixed. The shoulder cameras stay fixed in robot-root space, so their flange-relative extrinsics change with the arm: t_camera_flange = R(ee_quat)^T*(t_camera_root-ee_pos), R_camera_flange = R(ee_quat)^T*R(q_camera_root). Use the pose from the same image's observation, not a newer pose when interpreting a historical image.

The wrist viewing direction in flange coordinates is approximately [0.94976,-0.00763,0.31288]: 18.24 degrees away from flange-local +X axis. At the downward gripper quaternion [0.7071,0,0.7071,0], this viewing direction is approximately [0.31288,-0.00763,-0.94976] in robot-root coordinates. It looks forward as well as down; the camera also has a lateral mounting offset.

Projection uses normalized image coordinates u=0 at left, v=0 at top; the optical center is (0.5,0.5). For a point [X,Y,Z] in OpenGL camera coordinates with Z<0: u = 0.5 + fx_norm*X/(-Z); v = 0.5 - fy_norm*Y/(-Z). wrist_cam: fx_norm=0.520833333, fy_norm=0.925925926. Both shoulder cameras: fx_norm=0.390625, fy_norm=0.694444444. These normalized intrinsics apply to the full uncropped RGB images at any aspect-preserving resize. They use each individual camera image, not the video mosaic.

The relation between image coordinates and robot-root coordinates depends on depth and orientation. A pixel alone gives a ray, not a unique 3D position.


## Output modes

- {{student_eef_rule}}
- `eef`: give one fixed absolute base_link position, unit wxyz quaternion and
  boolean gripper command, for 1–5 controls. To express a relative move, compute
  the absolute target from the latest measured pose using this geometry. Omitted
  coordinates are not supported by the tool schema. Keep the measured
  quaternion if no rotation is intended.
- `edit`: modify the first 1–5 FK targets using a root-frame translation and
  rotation-vector delta. At row i=0..n-1 the offset scales by (i+1)/n. The
  gripper override (keep/open/closed) applies immediately, not gradually.

The full response also carries request_id, reason and outcome/intent assessment.
The schema includes edit and target fields for every mode; unused fields are
ignored. Final output is a call to robolab_execute, not a separately executed
shell script that sends actions.

## What happens after an EEF request

The requested absolute target must be within 0.05 m and 0.35 rad of the current
measured pose. The host then recomputes a local IK update after **each actual
control ACK**, using the fresh measured robot state and Jacobian:

    dq = J.T @ solve(J @ J.T + 0.05**2 * I, bounded_pose_error)

Per update, translation error is bounded to 0.02 m, rotation to 0.1 rad, each
joint increment to 0.05 rad and the native soft joint limits. The resulting
seven absolute joint targets and 0/1 gripper command go to the unchanged native
joint-position controller. There is no teleport, global path planner,
collision avoidance, or guaranteed arrival. One control is 1/15 s; five
controls are about 0.333 s, not five independent 5 cm displacements.

You receive RGB after the whole short segment, while low-level IK uses fresh
proprio inside it. Your target stays fixed during an `eef` segment. Between
segments another fresh Cosmos3 inference is required even if you intend another
correction. Review the attached observations between service calls.

Compare the post-execution measured pose with the target to compute remaining
translation/rotation errors. Linearized IK residual is not measured tracking
error. For example, run06 decision4 requested a 4.85 cm displacement but moved
only 2.58 cm in five controls. This illustrates why a valid ACK or small
kinematic residual does not prove arrival, contact or task progress.
