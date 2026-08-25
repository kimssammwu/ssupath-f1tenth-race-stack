import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Bool


class SimInitialPoseDriveGate(Node):
    """Block simulator drive commands until RViz 2D Pose Estimate is received.

    The real autonomy stack continues to publish /drive unchanged.  In sim mode
    f1tenth_gym subscribes to /sim/drive_to_gym instead, and this node is the only
    bridge between the two topics.

    A new /initialpose event closes the gate immediately, sends an explicit zero
    command, discards commands arriving during a short reset-settle window, then
    reopens the gate.  This prevents the command that existed before teleporting
    the car from being replayed at the new pose.
    """

    def __init__(self):
        super().__init__('sim_initialpose_drive_gate')

        self.declare_parameter('drive_in', '/drive')
        self.declare_parameter('drive_out', '/sim/drive_to_gym')
        self.declare_parameter('initialpose_topic', '/initialpose')
        self.declare_parameter('hold_after_initialpose_sec', 0.25)
        self.declare_parameter('require_initialpose', True)

        self.drive_in = str(self.get_parameter('drive_in').value)
        self.drive_out = str(self.get_parameter('drive_out').value)
        self.initialpose_topic = str(self.get_parameter('initialpose_topic').value)
        self.hold_sec = max(0.0, float(self.get_parameter('hold_after_initialpose_sec').value))
        self.require_initialpose = bool(self.get_parameter('require_initialpose').value)

        self.ready = not self.require_initialpose
        self.reopen_at_ns = None
        self._waiting_log_emitted = False

        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_out, 10)
        self.ready_pub = self.create_publisher(Bool, '/sim/initialpose_ready', 10)

        self.create_subscription(
            AckermannDriveStamped,
            self.drive_in,
            self._drive_cb,
            20,
        )
        self.create_subscription(
            PoseWithCovarianceStamped,
            self.initialpose_topic,
            self._initialpose_cb,
            10,
        )

        self.timer = self.create_timer(0.02, self._timer_cb)

        if self.ready:
            self.get_logger().warn(
                'Initial-pose requirement disabled; simulator drive gate starts OPEN.')
        else:
            self.get_logger().info(
                'Simulator drive gate CLOSED. Set RViz 2D Pose Estimate to enable motion.')

    def _publish_ready(self):
        msg = Bool()
        msg.data = bool(self.ready)
        self.ready_pub.publish(msg)

    def _zero_drive(self):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = 0.0
        msg.drive.acceleration = 0.0
        msg.drive.jerk = 0.0
        msg.drive.steering_angle = 0.0
        msg.drive.steering_angle_velocity = 0.0
        self.drive_pub.publish(msg)

    def _initialpose_cb(self, msg: PoseWithCovarianceStamped):
        # Close first.  The Gym bridge receives the same /initialpose and resets
        # its state; keeping this gate closed prevents a stale controller command
        # from moving the newly teleported vehicle during that reset.
        self.ready = False
        self._waiting_log_emitted = False
        self._zero_drive()

        now_ns = self.get_clock().now().nanoseconds
        self.reopen_at_ns = now_ns + int(self.hold_sec * 1e9)

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.get_logger().info(
            'RViz initial pose received: x=%.3f y=%.3f qz=%.4f qw=%.4f; '
            'drive held for %.3f s',
            p.x, p.y, q.z, q.w, self.hold_sec,
        )
        self._publish_ready()

    def _drive_cb(self, msg: AckermannDriveStamped):
        if not self.ready:
            if not self._waiting_log_emitted:
                self.get_logger().info(
                    'Ignoring /drive while waiting for RViz initial pose/reset settle.')
                self._waiting_log_emitted = True
            return
        # Do not cache or replay.  Only commands received while the gate is open
        # are allowed to reach the vehicle plant.
        self.drive_pub.publish(msg)

    def _timer_cb(self):
        if self.reopen_at_ns is not None:
            if self.get_clock().now().nanoseconds >= self.reopen_at_ns:
                self.reopen_at_ns = None
                self.ready = True
                self._waiting_log_emitted = False
                self.get_logger().info(
                    'Simulator drive gate OPEN. New /drive commands now control the vehicle.')
        self._publish_ready()


def main(args=None):
    rclpy.init(args=args)
    node = SimInitialPoseDriveGate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._zero_drive()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
