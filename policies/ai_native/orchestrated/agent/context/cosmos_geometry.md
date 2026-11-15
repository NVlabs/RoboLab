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
