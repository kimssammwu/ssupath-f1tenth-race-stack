import json
import math
from typing import Optional

import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped
from f110_msgs.msg import WpntArray
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _yaw_from_quaternion(q) -> float:
    # ROS quaternion -> yaw, without an extra tf dependency.
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class SimDriveDiagnostics(Node):
    """Simulation-only monitor for the BEXCO SIL loop.

    It intentionally does not change the controller command.  The purpose is to
    capture enough information to tell these failure modes apart after one run:

      * vehicle hit the occupancy-map wall (same TTC criterion used by f110_gym),
      * PP selected the wrong nearby branch of a self-close track,
      * PP lookahead moved behind the vehicle / steering flipped abruptly,
      * localization estimate departed from simulator ground truth,
      * initial vehicle yaw is inconsistent with the local raceline heading.

    Published topics:
      /sim/collision                 std_msgs/Bool, latched after Gym-equivalent TTC hit
      /sim/collision_warning         std_msgs/Bool, configurable pre-collision TTC warning
      /sim/pp_anomaly                std_msgs/Bool
      /sim/drive_diagnostics         std_msgs/String containing compact JSON

    The Gym ROS bridge currently does not expose obs['collisions'].  f110_gym's
    RaceCar.check_ttc() uses scan, longitudinal velocity, vehicle half extents,
    and a 0.005 s TTC threshold.  This node mirrors that calculation from the
    published scan.  If Gym has already zeroed the velocity on the collision
    step, the most recent non-zero GT velocity is used while a positive/negative
    drive command is still requested; this preserves the collision edge for
    ROS-side post-mortem logging.
    """

    def __init__(self):
        super().__init__('sim_drive_diagnostics')

        # Topics
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('gt_odom_topic', '/car_state/odom_GT')
        self.declare_parameter('est_pose_topic', '/car_state/pose')
        self.declare_parameter('est_odom_topic', '/car_state/odom')
        self.declare_parameter('frenet_odom_topic', '/car_state/frenet/odom')
        self.declare_parameter('local_waypoints_topic', '/local_waypoints')
        self.declare_parameter('drive_topic', '/drive')

        # Match f110_gym RaceCar defaults / current NUC2 sim parameters.
        self.declare_parameter('vehicle_width', 0.2032)
        self.declare_parameter('lf', 0.15875)
        self.declare_parameter('lr', 0.17145)
        self.declare_parameter('gym_collision_ttc', 0.005)
        self.declare_parameter('collision_warning_ttc', 0.15)

        # Mirror current PP parameters.  They are diagnostics only, not control.
        self.declare_parameter('pp_m_l1', 0.32)
        self.declare_parameter('pp_q_l1', 0.25)
        self.declare_parameter('pp_t_clip_min', 0.1)
        self.declare_parameter('pp_t_clip_max', 6.0)
        self.declare_parameter('waypoint_spacing', 0.1)
        self.declare_parameter('nearest_index_warn', 10)
        self.declare_parameter('ambiguity_index_gap', 10)
        self.declare_parameter('ambiguity_distance_margin', 0.25)
        self.declare_parameter('target_behind_tolerance', 0.05)
        self.declare_parameter('heading_warn_deg', 90.0)
        self.declare_parameter('startup_heading_warn_deg', 60.0)
        self.declare_parameter('startup_window_sec', 3.0)
        self.declare_parameter('sharp_steer_rad', 0.25)
        self.declare_parameter('steer_flip_rad', 0.15)
        self.declare_parameter('localization_pos_warn_m', 0.40)
        self.declare_parameter('localization_yaw_warn_deg', 30.0)
        self.declare_parameter('publish_rate_hz', 40.0)

        self.vehicle_width = float(self.get_parameter('vehicle_width').value)
        self.half_length = 0.5 * (
            float(self.get_parameter('lf').value)
            + float(self.get_parameter('lr').value)
        )
        self.gym_collision_ttc = float(self.get_parameter('gym_collision_ttc').value)
        self.warning_ttc = float(self.get_parameter('collision_warning_ttc').value)

        self.scan: Optional[LaserScan] = None
        self.gt_odom: Optional[Odometry] = None
        self.est_pose: Optional[PoseStamped] = None
        self.est_odom: Optional[Odometry] = None
        self.frenet_odom: Optional[Odometry] = None
        self.local_waypoints: Optional[WpntArray] = None
        self.drive: Optional[AckermannDriveStamped] = None

        self.prev_steer = 0.0
        self.prev_nearest_idx: Optional[int] = None
        self.last_nonzero_gt_speed = 0.0
        self.collision_latched = False
        self.last_pp_anomaly = False
        self.first_valid_diag_ns: Optional[int] = None

        self.collision_pub = self.create_publisher(Bool, '/sim/collision', 10)
        self.collision_warning_pub = self.create_publisher(Bool, '/sim/collision_warning', 10)
        self.pp_anomaly_pub = self.create_publisher(Bool, '/sim/pp_anomaly', 10)
        self.diag_pub = self.create_publisher(String, '/sim/drive_diagnostics', 10)

        self.create_subscription(
            LaserScan, self.get_parameter('scan_topic').value, self._scan_cb, 10)
        self.create_subscription(
            Odometry, self.get_parameter('gt_odom_topic').value, self._gt_odom_cb, 10)
        self.create_subscription(
            PoseStamped, self.get_parameter('est_pose_topic').value, self._est_pose_cb, 10)
        self.create_subscription(
            Odometry, self.get_parameter('est_odom_topic').value, self._est_odom_cb, 10)
        self.create_subscription(
            Odometry, self.get_parameter('frenet_odom_topic').value, self._frenet_cb, 10)
        self.create_subscription(
            WpntArray, self.get_parameter('local_waypoints_topic').value, self._wp_cb, 10)
        self.create_subscription(
            AckermannDriveStamped, self.get_parameter('drive_topic').value, self._drive_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self._reset_cb, 10)

        hz = max(1.0, float(self.get_parameter('publish_rate_hz').value))
        self.timer = self.create_timer(1.0 / hz, self._tick)

        self.get_logger().info(
            'BEXCO simulator diagnostics enabled: collision TTC + PP branch/heading/localization checks')

    def _scan_cb(self, msg: LaserScan):
        self.scan = msg

    def _gt_odom_cb(self, msg: Odometry):
        self.gt_odom = msg
        speed = float(msg.twist.twist.linear.x)
        if abs(speed) > 0.05:
            self.last_nonzero_gt_speed = speed

    def _est_pose_cb(self, msg: PoseStamped):
        self.est_pose = msg

    def _est_odom_cb(self, msg: Odometry):
        self.est_odom = msg

    def _frenet_cb(self, msg: Odometry):
        self.frenet_odom = msg

    def _wp_cb(self, msg: WpntArray):
        self.local_waypoints = msg

    def _drive_cb(self, msg: AckermannDriveStamped):
        self.drive = msg

    def _reset_cb(self, _msg: PoseWithCovarianceStamped):
        self.collision_latched = False
        self.last_nonzero_gt_speed = 0.0
        self.prev_nearest_idx = None
        self.prev_steer = 0.0
        self.first_valid_diag_ns = None
        self.get_logger().info('Diagnostics reset by /initialpose')

    @staticmethod
    def _ray_to_vehicle_edge(angle: float, half_length: float, half_width: float) -> float:
        """Distance from lidar origin to rectangular vehicle envelope along a beam.

        f110_gym uses min(front/rear intersection, side intersection) with
        half_length=(lf+lr)/2 and half_width=width/2.  This absolute-value form
        is algebraically equivalent for all quadrants and is numerically safer.
        """
        ca = abs(math.cos(angle))
        sa = abs(math.sin(angle))
        front = half_length / ca if ca > 1e-9 else math.inf
        side = half_width / sa if sa > 1e-9 else math.inf
        return min(front, side)

    def _collision_metrics(self):
        if self.scan is None or self.gt_odom is None:
            return math.inf, False, False

        current_speed = float(self.gt_odom.twist.twist.linear.x)
        requested_speed = float(self.drive.drive.speed) if self.drive is not None else 0.0

        # Gym sets velocity to zero in the same step that check_ttc reports a
        # collision. Use the last moving speed only when the controller is still
        # requesting motion, otherwise a deliberate stop near a wall could latch.
        speed_for_ttc = current_speed
        if abs(speed_for_ttc) <= 0.05 and abs(requested_speed) > 0.10:
            speed_for_ttc = self.last_nonzero_gt_speed

        if abs(speed_for_ttc) <= 1e-6:
            return math.inf, False, self.collision_latched

        min_ttc = math.inf
        half_width = 0.5 * self.vehicle_width
        angle = float(self.scan.angle_min)
        inc = float(self.scan.angle_increment)

        for r in self.scan.ranges:
            if not math.isfinite(r):
                angle += inc
                continue
            proj_vel = speed_for_ttc * math.cos(angle)
            if abs(proj_vel) > 1e-9:
                side_dist = self._ray_to_vehicle_edge(angle, self.half_length, half_width)
                ttc = (float(r) - side_dist) / proj_vel
                if 0.0 <= ttc < min_ttc:
                    min_ttc = ttc
            angle += inc

        warning = min_ttc < self.warning_ttc
        collision_now = min_ttc < self.gym_collision_ttc
        if collision_now:
            self.collision_latched = True
        return min_ttc, warning, self.collision_latched

    def _pp_metrics(self):
        result = {
            'ready': False,
            'nearest_idx': -1,
            'nearest_idx_jump': 0,
            'target_idx': -1,
            'nearest_distance_m': None,
            'second_branch_idx': -1,
            'second_branch_distance_m': None,
            'xy_branch_ambiguous': False,
            'target_forward_m': None,
            'target_lateral_m': None,
            'target_behind': False,
            'heading_error_deg': None,
            'heading_bad': False,
            'startup_heading_bad': False,
            'sharp_steer': False,
            'steer_flip': False,
            'localization_pos_error_m': None,
            'localization_yaw_error_deg': None,
            'localization_bad': False,
            'anomaly': False,
        }

        if self.est_pose is None or self.local_waypoints is None:
            return result
        wps = self.local_waypoints.wpnts
        if not wps:
            return result

        pose = self.est_pose.pose
        x = float(pose.position.x)
        y = float(pose.position.y)
        yaw = _yaw_from_quaternion(pose.orientation)

        distances = []
        for i, wp in enumerate(wps):
            dx = x - float(wp.x_m)
            dy = y - float(wp.y_m)
            distances.append((dx * dx + dy * dy, i))
        distances.sort(key=lambda p: p[0])
        nearest_d2, nearest_idx = distances[0]
        nearest_dist = math.sqrt(nearest_d2)

        idx_jump = 0
        if self.prev_nearest_idx is not None:
            idx_jump = nearest_idx - self.prev_nearest_idx
        self.prev_nearest_idx = nearest_idx

        # Find a geometrically competitive point that is far away in ordered
        # local-path index.  This is the signature expected at a self-close bend
        # if pure XY nearest-point selection jumps to another branch.
        gap = int(self.get_parameter('ambiguity_index_gap').value)
        second_idx = -1
        second_dist = None
        for d2, idx in distances[1:]:
            if abs(idx - nearest_idx) >= gap:
                second_idx = idx
                second_dist = math.sqrt(d2)
                break
        ambiguity_margin = float(self.get_parameter('ambiguity_distance_margin').value)
        xy_ambiguous = (
            second_dist is not None
            and second_dist <= nearest_dist + ambiguity_margin
        )

        wp_near = wps[nearest_idx]
        heading_error = abs(_wrap_angle(yaw - float(wp_near.psi_rad)))
        heading_warn = math.radians(float(self.get_parameter('heading_warn_deg').value))
        heading_bad = heading_error > heading_warn

        cross_track = (
            -math.sin(yaw) * (float(wp_near.x_m) - x)
            + math.cos(yaw) * (float(wp_near.y_m) - y)
        )
        speed = (
            float(self.est_odom.twist.twist.linear.x)
            if self.est_odom is not None
            else float(self.gt_odom.twist.twist.linear.x) if self.gt_odom is not None else 0.0
        )
        l1 = (
            float(self.get_parameter('pp_q_l1').value)
            + speed * float(self.get_parameter('pp_m_l1').value)
        )
        lower = max(
            float(self.get_parameter('pp_t_clip_min').value),
            math.sqrt(2.0) * abs(cross_track),
        )
        l1 = max(lower, min(float(self.get_parameter('pp_t_clip_max').value), l1))
        spacing = max(1e-3, float(self.get_parameter('waypoint_spacing').value))
        d_index = int(l1 / spacing + 0.5)
        target_idx = min(len(wps) - 1, nearest_idx + d_index)
        target = wps[target_idx]
        tx = float(target.x_m) - x
        ty = float(target.y_m) - y
        target_forward = math.cos(yaw) * tx + math.sin(yaw) * ty
        target_lateral = -math.sin(yaw) * tx + math.cos(yaw) * ty
        target_behind = target_forward < -float(self.get_parameter('target_behind_tolerance').value)

        steer = float(self.drive.drive.steering_angle) if self.drive is not None else 0.0
        sharp_steer = abs(steer) >= float(self.get_parameter('sharp_steer_rad').value)
        flip_thr = float(self.get_parameter('steer_flip_rad').value)
        steer_flip = (
            (self.prev_steer >= flip_thr and steer <= -flip_thr)
            or (self.prev_steer <= -flip_thr and steer >= flip_thr)
        )
        self.prev_steer = steer

        now_ns = self.get_clock().now().nanoseconds
        if self.first_valid_diag_ns is None:
            self.first_valid_diag_ns = now_ns
        startup_age = (now_ns - self.first_valid_diag_ns) * 1e-9
        startup_heading_bad = (
            startup_age <= float(self.get_parameter('startup_window_sec').value)
            and heading_error > math.radians(float(self.get_parameter('startup_heading_warn_deg').value))
        )

        loc_pos_err = None
        loc_yaw_err = None
        localization_bad = False
        if self.gt_odom is not None:
            gp = self.gt_odom.pose.pose
            gx = float(gp.position.x)
            gy = float(gp.position.y)
            gyaw = _yaw_from_quaternion(gp.orientation)
            loc_pos_err = math.hypot(x - gx, y - gy)
            loc_yaw_err = abs(_wrap_angle(yaw - gyaw))
            localization_bad = (
                loc_pos_err > float(self.get_parameter('localization_pos_warn_m').value)
                or loc_yaw_err > math.radians(float(self.get_parameter('localization_yaw_warn_deg').value))
            )

        nearest_idx_bad = nearest_idx > int(self.get_parameter('nearest_index_warn').value)
        idx_jump_bad = abs(idx_jump) > int(self.get_parameter('nearest_index_warn').value)

        # Sharp steering by itself is not an anomaly on a hairpin.  It becomes
        # diagnostic when accompanied by a flip, branch ambiguity, target/heading
        # inconsistency, or localization failure.
        anomaly = bool(
            target_behind
            or heading_bad
            or startup_heading_bad
            or xy_ambiguous
            or nearest_idx_bad
            or idx_jump_bad
            or steer_flip
            or localization_bad
        )

        result.update({
            'ready': True,
            'nearest_idx': nearest_idx,
            'nearest_idx_jump': idx_jump,
            'target_idx': target_idx,
            'nearest_distance_m': nearest_dist,
            'second_branch_idx': second_idx,
            'second_branch_distance_m': second_dist,
            'xy_branch_ambiguous': xy_ambiguous,
            'target_forward_m': target_forward,
            'target_lateral_m': target_lateral,
            'target_behind': target_behind,
            'heading_error_deg': math.degrees(heading_error),
            'heading_bad': heading_bad,
            'startup_heading_bad': startup_heading_bad,
            'sharp_steer': sharp_steer,
            'steer_flip': steer_flip,
            'localization_pos_error_m': loc_pos_err,
            'localization_yaw_error_deg': math.degrees(loc_yaw_err) if loc_yaw_err is not None else None,
            'localization_bad': localization_bad,
            'anomaly': anomaly,
        })
        return result

    def _tick(self):
        min_ttc, collision_warning, collision = self._collision_metrics()
        pp = self._pp_metrics()

        collision_msg = Bool()
        collision_msg.data = bool(collision)
        self.collision_pub.publish(collision_msg)

        warning_msg = Bool()
        warning_msg.data = bool(collision_warning)
        self.collision_warning_pub.publish(warning_msg)

        anomaly_msg = Bool()
        anomaly_msg.data = bool(pp['anomaly'])
        self.pp_anomaly_pub.publish(anomaly_msg)

        gt = None
        if self.gt_odom is not None:
            gp = self.gt_odom.pose.pose
            gt = {
                'x': float(gp.position.x),
                'y': float(gp.position.y),
                'yaw': _yaw_from_quaternion(gp.orientation),
                'speed': float(self.gt_odom.twist.twist.linear.x),
            }

        drive = None
        if self.drive is not None:
            drive = {
                'speed': float(self.drive.drive.speed),
                'steer': float(self.drive.drive.steering_angle),
            }

        frenet = None
        if self.frenet_odom is not None:
            frenet = {
                's': float(self.frenet_odom.pose.pose.position.x),
                'd': float(self.frenet_odom.pose.pose.position.y),
            }

        payload = {
            'stamp_ns': int(self.get_clock().now().nanoseconds),
            'collision': bool(collision),
            'collision_warning': bool(collision_warning),
            'min_ttc_sec': None if not math.isfinite(min_ttc) else float(min_ttc),
            'gt': gt,
            'drive': drive,
            'frenet': frenet,
            'pp': pp,
        }
        out = String()
        out.data = json.dumps(payload, separators=(',', ':'), allow_nan=False)
        self.diag_pub.publish(out)

        if collision and not getattr(self, '_collision_logged', False):
            self._collision_logged = True
            self.get_logger().error(
                'SIM COLLISION captured: %s', json.dumps(payload, separators=(',', ':')))

        if pp['anomaly'] and not self.last_pp_anomaly:
            self.get_logger().warn(
                'PP anomaly: nearest=%d target=%d forward=%s heading_err=%sdeg '
                'ambiguous=%s idx_jump=%d steer_flip=%s loc_bad=%s',
                pp['nearest_idx'],
                pp['target_idx'],
                'n/a' if pp['target_forward_m'] is None else f"{pp['target_forward_m']:.3f}",
                'n/a' if pp['heading_error_deg'] is None else f"{pp['heading_error_deg']:.1f}",
                pp['xy_branch_ambiguous'],
                pp['nearest_idx_jump'],
                pp['steer_flip'],
                pp['localization_bad'],
            )
        self.last_pp_anomaly = bool(pp['anomaly'])

        if not collision:
            # Permit a future collision log after an /initialpose reset.  A
            # latched collision itself stays true until that reset callback.
            if not self.collision_latched:
                self._collision_logged = False


def main(args=None):
    rclpy.init(args=args)
    node = SimDriveDiagnostics()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
