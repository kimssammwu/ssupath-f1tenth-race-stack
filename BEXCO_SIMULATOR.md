# BEXCO PC Simulator

Branch: `mw_bexco_simulator`

This branch keeps the existing planner/controller/localization stack and replaces only the physical vehicle/sensor layer when `sim:=true`.

## Architecture

```text
/time_trials controller (PP, MPCC, ...)
             |
           /drive
             v
      f1tenth_gym dynamics
             |
       /car_state/odom_GT
             |
      bexco_sim_bridge
        |      |      |
      /odom   IMU   /livox/lidar
        |      |      |
        +------+------+
               v
     full_localization.launch.xml
               |
          estimated state
               |
       planner/controller
               +------> /drive
```

Ground-truth pose is used only to generate virtual hardware measurements. It is not used as the localization result.

## PCD placement

By default the simulator expects the PCD file next to the map files and with the same base name.

Example:

```text
stack_master/maps/hall_0821/
  hall_0821.yaml
  hall_0821.png
  hall_0821.pcd
  speed_scaling.yaml
  ot_sectors.yaml
```

A different PCD filename in the same map directory can be selected with `pcd_file:=...`.

Supported PCD DATA formats:

- `ascii`
- `binary`

`binary_compressed` is not supported in the first version. Convert it to `binary` or `ascii` before launch.

## Build

From the workspace root:

```bash
git checkout mw_bexco_simulator
colcon build --symlink-install
source install/setup.bash
```

## Run

Terminal 1:

```bash
ros2 launch stack_master base_system_3D_launch.xml \
  racecar_version:=NUC2 \
  map_dir:=hall_0821 \
  map_name:=hall_0821 \
  sim:=true \
  rviz:=true \
  imu_mode:=test
```

If the PCD name differs:

```bash
ros2 launch stack_master base_system_3D_launch.xml \
  racecar_version:=NUC2 \
  map_dir:=hall_0821 \
  map_name:=hall_0821 \
  pcd_file:=environment.pcd \
  sim:=true \
  rviz:=true \
  imu_mode:=test
```

Terminal 2:

```bash
ros2 launch stack_master time_trials_launch.xml \
  racecar_version:=NUC2 \
  ctrl_algo:=PP
```

## Expected simulator topics

Input to the virtual hardware bridge:

```text
/car_state/odom_GT
/vesc/sensors/imu/raw
```

Output toward the real 3D stack:

```text
/odom
/sensors/imu/raw
/livox/lidar
```

The controller output remains:

```text
/drive
```

## Quick checks

```bash
ros2 topic hz /car_state/odom_GT
ros2 topic hz /odom
ros2 topic hz /sensors/imu/raw
ros2 topic hz /livox/lidar
ros2 topic hz /drive
```

Check that the virtual cloud uses the expected frame:

```bash
ros2 topic echo /livox/lidar --once
```

Expected `header.frame_id`:

```text
livox_frame
```

## Virtual Livox model

The first implementation does the following for every virtual scan:

1. Load the static PCD once at startup.
2. Voxel-downsample it once.
3. Use F1TENTH Gym ground-truth pose only inside the hardware emulator.
4. Crop points by LiDAR range.
5. Transform map points into `livox_frame`.
6. Filter by height/FOV.
7. Divide the view into azimuth/elevation cells.
8. Keep the nearest point in each angular cell to approximate first-hit occlusion.
9. Publish the result as `sensor_msgs/PointCloud2` on `/livox/lidar`.

Default virtual sensor transform matches the existing localization launch:

```text
base_link -> livox_frame
x   = 0.27 m
y   = 0.00 m
z   = 0.07 m
yaw = 90 deg
```

## Current limitations

This is the first SIL version, not a full physical MID360 renderer.

- PCD map is treated as static.
- Moving obstacles are not simulated yet.
- Livox non-repetitive scan timing/pattern is not reproduced yet.
- Intensity is currently constant.
- Motion distortion during a scan is not reproduced yet.
- `binary_compressed` PCD is not supported yet.
- Visibility is approximated by nearest point per angular cell rather than mesh/voxel ray tracing.

The important property for this branch is that PP/planner/localization are not replaced by simplified simulator versions. The simulator boundary is kept below the control/localization algorithms.
