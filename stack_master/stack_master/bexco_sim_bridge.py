import math
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2, PointField
from std_msgs.msg import Header


def _quat_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _parse_pcd_header(fp) -> Tuple[Dict[str, List[str]], int]:
    header: Dict[str, List[str]] = {}
    while True:
        line = fp.readline()
        if not line:
            raise ValueError('PCD header ended before DATA line')
        text = line.decode('ascii', errors='strict').strip()
        if not text or text.startswith('#'):
            continue
        parts = text.split()
        key = parts[0].upper()
        header[key] = parts[1:]
        if key == 'DATA':
            return header, fp.tell()


def _pcd_dtype(header: Dict[str, List[str]]) -> np.dtype:
    fields = header.get('FIELDS', header.get('FIELD', []))
    sizes = [int(x) for x in header.get('SIZE', [])]
    types = header.get('TYPE', [])
    counts = [int(x) for x in header.get('COUNT', ['1'] * len(fields))]
    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ValueError('Unsupported or malformed PCD field definition')

    descr = []
    for name, size, pcd_type, count in zip(fields, sizes, types, counts):
        if pcd_type == 'F':
            code = {4: '<f4', 8: '<f8'}.get(size)
        elif pcd_type == 'I':
            code = {1: '<i1', 2: '<i2', 4: '<i4', 8: '<i8'}.get(size)
        elif pcd_type == 'U':
            code = {1: '<u1', 2: '<u2', 4: '<u4', 8: '<u8'}.get(size)
        else:
            code = None
        if code is None:
            raise ValueError(f'Unsupported PCD TYPE/SIZE: {pcd_type}/{size}')
        descr.append((name, code) if count == 1 else (name, code, (count,)))
    return np.dtype(descr)


def load_pcd_xyz(path: str) -> np.ndarray:
    with open(path, 'rb') as fp:
        header, _ = _parse_pcd_header(fp)
        mode = header['DATA'][0].lower()
        fields = header.get('FIELDS', header.get('FIELD', []))
        if not all(axis in fields for axis in ('x', 'y', 'z')):
            raise ValueError('PCD must contain x, y, z fields')

        points = int(header.get('POINTS', ['0'])[0])
        if points <= 0:
            width = int(header.get('WIDTH', ['0'])[0])
            height = int(header.get('HEIGHT', ['1'])[0])
            points = width * height

        if mode == 'ascii':
            raw = np.loadtxt(fp, dtype=np.float64)
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            ix, iy, iz = fields.index('x'), fields.index('y'), fields.index('z')
            xyz = raw[:, [ix, iy, iz]].astype(np.float32, copy=False)
        elif mode == 'binary':
            dtype = _pcd_dtype(header)
            raw = np.fromfile(fp, dtype=dtype, count=points if points > 0 else -1)
            xyz = np.column_stack((raw['x'], raw['y'], raw['z'])).astype(np.float32, copy=False)
        elif mode == 'binary_compressed':
            raise ValueError(
                'binary_compressed PCD is not supported yet; convert it to binary or ascii PCD')
        else:
            raise ValueError(f'Unsupported PCD DATA mode: {mode}')

    valid = np.isfinite(xyz).all(axis=1)
    return xyz[valid]


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    if voxel <= 0.0 or len(points) == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx.sort()
    return points[idx]


def nearest_angular_hits(points: np.ndarray, az_res: float, el_res: float) -> np.ndarray:
    if len(points) == 0 or az_res <= 0.0 or el_res <= 0.0:
        return points

    ranges = np.linalg.norm(points, axis=1)
    safe = np.maximum(ranges, 1e-6)
    az = np.arctan2(points[:, 1], points[:, 0])
    el = np.arcsin(np.clip(points[:, 2] / safe, -1.0, 1.0))
    az_bin = np.floor((az + math.pi) / az_res).astype(np.int64)
    el_bin = np.floor((el + math.pi / 2.0) / el_res).astype(np.int64)
    key = (az_bin << np.int64(32)) ^ (el_bin & np.int64(0xffffffff))
    order = np.argsort(ranges, kind='stable')
    _, first = np.unique(key[order], return_index=True)
    return points[order[first]]


def xyz_to_cloud(points: np.ndarray, stamp, frame_id: str) -> PointCloud2:
    msg = PointCloud2()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    msg.height = 1
    msg.width = int(points.shape[0])
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    packed = np.zeros((points.shape[0], 4), dtype=np.float32)
    if points.shape[0]:
        packed[:, :3] = points
        packed[:, 3] = 1.0
    msg.data = packed.tobytes()
    return msg


class BexcoSimBridge(Node):
    """Virtual VESC/IMU/Livox hardware for PC SIL.

    Ground truth is consumed only for virtual sensor rendering.  The /odom
    output exposes twist for the real odom_to_twist_converter and deliberately
    does not expose the map-frame ground-truth pose.
    """

    def __init__(self):
        super().__init__('bexco_sim_bridge')

        self.declare_parameter('pcd_path', '')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('lidar_frame', 'livox_frame')
        self.declare_parameter('range_min', 0.3)
        self.declare_parameter('range_max', 30.0)
        self.declare_parameter('z_min', -2.0)
        self.declare_parameter('z_max', 2.0)
        self.declare_parameter('horizontal_fov', 2.0 * math.pi)
        self.declare_parameter('cloud_rate', 10.0)
        self.declare_parameter('map_voxel_size', 0.04)
        self.declare_parameter('azimuth_resolution_deg', 0.4)
        self.declare_parameter('elevation_resolution_deg', 0.4)
        self.declare_parameter('max_points', 120000)
        self.declare_parameter('lidar_x', 0.27)
        self.declare_parameter('lidar_y', 0.0)
        self.declare_parameter('lidar_z', 0.07)
        self.declare_parameter('lidar_yaw', math.pi / 2.0)
        self.declare_parameter('odom_in', '/car_state/odom_GT')
        self.declare_parameter('imu_in', '/vesc/sensors/imu/raw')
        self.declare_parameter('odom_out', '/odom')
        self.declare_parameter('imu_out', '/sensors/imu/raw')
        self.declare_parameter('cloud_out', '/livox/lidar')

        self.map_frame = self.get_parameter('map_frame').value
        self.lidar_frame = self.get_parameter('lidar_frame').value
        self.range_min = float(self.get_parameter('range_min').value)
        self.range_max = float(self.get_parameter('range_max').value)
        self.z_min = float(self.get_parameter('z_min').value)
        self.z_max = float(self.get_parameter('z_max').value)
        self.horizontal_fov = float(self.get_parameter('horizontal_fov').value)
        self.cloud_period = 1.0 / max(0.1, float(self.get_parameter('cloud_rate').value))
        self.az_res = math.radians(float(self.get_parameter('azimuth_resolution_deg').value))
        self.el_res = math.radians(float(self.get_parameter('elevation_resolution_deg').value))
        self.max_points = int(self.get_parameter('max_points').value)
        self.lidar_xyz = np.array([
            float(self.get_parameter('lidar_x').value),
            float(self.get_parameter('lidar_y').value),
            float(self.get_parameter('lidar_z').value),
        ], dtype=np.float32)
        self.lidar_yaw = float(self.get_parameter('lidar_yaw').value)
        self._last_cloud_walltime = 0.0

        pcd_path = os.path.expanduser(str(self.get_parameter('pcd_path').value))
        if not pcd_path:
            raise RuntimeError('pcd_path is required for bexco_sim_bridge')
        if not os.path.isfile(pcd_path):
            raise RuntimeError(f'PCD file does not exist: {pcd_path}')

        self.get_logger().info(f'Loading PCD: {pcd_path}')
        points = load_pcd_xyz(pcd_path)
        before = len(points)
        voxel = float(self.get_parameter('map_voxel_size').value)
        self.map_points = voxel_downsample(points, voxel)
        self.get_logger().info(
            f'PCD ready: {before} -> {len(self.map_points)} points (voxel={voxel:.3f} m)')

        self.odom_pub = self.create_publisher(Odometry, self.get_parameter('odom_out').value, 10)
        self.imu_pub = self.create_publisher(Imu, self.get_parameter('imu_out').value, 20)
        self.cloud_pub = self.create_publisher(PointCloud2, self.get_parameter('cloud_out').value, 10)

        self.create_subscription(Odometry, self.get_parameter('odom_in').value, self._odom_cb, 10)
        self.create_subscription(Imu, self.get_parameter('imu_in').value, self._imu_cb, 20)

    def _imu_cb(self, msg: Imu):
        out = Imu()
        out.header = msg.header
        out.header.frame_id = 'imu'
        out.orientation = msg.orientation
        out.orientation_covariance = msg.orientation_covariance
        out.angular_velocity = msg.angular_velocity
        out.angular_velocity_covariance = msg.angular_velocity_covariance
        out.linear_acceleration = msg.linear_acceleration
        out.linear_acceleration_covariance = msg.linear_acceleration_covariance
        self.imu_pub.publish(out)

    def _odom_cb(self, msg: Odometry):
        odom = Odometry()
        odom.header.stamp = msg.header.stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.orientation.w = 1.0
        odom.pose.covariance = [1e6] * 36
        odom.twist = msg.twist
        self.odom_pub.publish(odom)

        now = time.monotonic()
        if now - self._last_cloud_walltime < self.cloud_period:
            return
        self._last_cloud_walltime = now
        self._publish_virtual_livox(msg)

    def _publish_virtual_livox(self, odom: Odometry):
        bx = float(odom.pose.pose.position.x)
        by = float(odom.pose.pose.position.y)
        bz = float(odom.pose.pose.position.z)
        yaw = _quat_to_yaw(odom.pose.pose.orientation)

        cy, sy = math.cos(yaw), math.sin(yaw)
        lx = bx + cy * self.lidar_xyz[0] - sy * self.lidar_xyz[1]
        ly = by + sy * self.lidar_xyz[0] + cy * self.lidar_xyz[1]
        lz = bz + self.lidar_xyz[2]

        dx = self.map_points[:, 0] - lx
        dy = self.map_points[:, 1] - ly
        dz = self.map_points[:, 2] - lz
        r2 = dx * dx + dy * dy + dz * dz
        mask = (r2 >= self.range_min * self.range_min) & (r2 <= self.range_max * self.range_max)

        if not np.any(mask):
            self.cloud_pub.publish(
                xyz_to_cloud(np.empty((0, 3), np.float32), odom.header.stamp, self.lidar_frame))
            return

        local = np.column_stack((dx[mask], dy[mask], dz[mask])).astype(np.float32, copy=False)

        c, s = math.cos(-yaw), math.sin(-yaw)
        x_b = c * local[:, 0] - s * local[:, 1]
        y_b = s * local[:, 0] + c * local[:, 1]
        z_b = local[:, 2]

        c, s = math.cos(-self.lidar_yaw), math.sin(-self.lidar_yaw)
        x_l = c * x_b - s * y_b
        y_l = s * x_b + c * y_b
        cloud = np.column_stack((x_l, y_l, z_b)).astype(np.float32, copy=False)

        mask = (cloud[:, 2] >= self.z_min) & (cloud[:, 2] <= self.z_max)
        if self.horizontal_fov < 2.0 * math.pi - 1e-3:
            angle = np.arctan2(cloud[:, 1], cloud[:, 0])
            mask &= np.abs(angle) <= self.horizontal_fov * 0.5
        cloud = cloud[mask]

        cloud = nearest_angular_hits(cloud, self.az_res, self.el_res)

        if self.max_points > 0 and len(cloud) > self.max_points:
            step = int(math.ceil(len(cloud) / self.max_points))
            cloud = cloud[::step]

        self.cloud_pub.publish(xyz_to_cloud(cloud, odom.header.stamp, self.lidar_frame))


def main(args=None):
    rclpy.init(args=args)
    node = BexcoSimBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
