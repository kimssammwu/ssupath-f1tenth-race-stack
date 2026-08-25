#!/usr/bin/env python3
"""
mouse_estop_node
================
마우스 우클릭 한 번으로 정지, 다시 한 번으로 재출발하는 토글 e-stop.

조이스틱 e-stop(mux_controller 의 Circle=정지 / Triangle=해제)과 **같은 래치**를 공유하므로
둘 중 아무거나 써도 된다:
  - 마우스로 세운 차를 조이스틱 Triangle 로 출발시킬 수 있고,
  - 조이스틱 Circle 로 세운 차를 마우스 우클릭으로 출발시킬 수 있다.

동작:
  우클릭 -> /e_stop (std_msgs/Bool) 로 true(정지) / false(출발) 를 번갈아 발행.
  /e_stop_state 를 구독해서 mux_controller 가 들고 있는 '진짜' 래치와 상태를 맞춘다.
  (조이스틱으로 상태가 바뀌어도 다음 우클릭이 항상 현재 상태의 반대를 낸다.
   이 동기화가 없으면 조이스틱으로 푼 뒤 우클릭 한 번이 헛돈다.)

입력 백엔드:
  evdev  : /dev/input/eventX 를 직접 읽는다. X 서버와 무관하고 창 포커스도 필요 없다.
           /dev/input/event* 읽기 권한이 필요하다 (stack_master/scripts/99-f1tenth-input.rules).
  pynput : X11 전역 후킹. DISPLAY 가 필요하지만 별도 권한 설정은 필요 없다.
           rviz 창 위에서 우클릭하면 rviz 컨텍스트 메뉴도 같이 뜨지만 e-stop 은 정상 동작한다.
  auto   : evdev 가 열리면 evdev, 아니면 pynput (기본값)

주의: 이 노드는 마우스를 grab 하지 않는다. 우클릭은 평소처럼 데스크톱에도 그대로 전달된다.
"""

import os
import queue
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Bool


# e-stop 은 놓치면 안 되는 신호라 reliable + transient_local(마지막 값 유지).
# mux_controller 쪽 구독도 같은 프로필이어야 늦게 뜬 쪽이 현재 상태를 받는다.
def latched_qos(depth: int = 1) -> QoSProfile:
    return QoSProfile(
        depth=depth,
        history=QoSHistoryPolicy.KEEP_LAST,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )


class MouseEStop(Node):
    def __init__(self):
        super().__init__('mouse_estop')

        self.backend_param = self.declare_parameter('backend', 'auto').value
        self.device = self.declare_parameter('device', '').value
        self.button = self.declare_parameter('button', 'right').value
        self.debounce_ms = self.declare_parameter('debounce_ms', 250).value
        self.estop_topic = self.declare_parameter('estop_topic', '/e_stop').value
        self.state_topic = self.declare_parameter('state_topic', '/e_stop_state').value
        # true 로 두면 노드가 뜨자마자 정지 래치를 건다(우클릭 한 번으로 출발).
        self.start_stopped = self.declare_parameter('start_stopped', False).value
        # 비워 두면 환경변수 DISPLAY 를 그대로 쓴다. pynput 백엔드에서만 의미 있음.
        self.display = self.declare_parameter('display', '').value

        self.estop_pub = self.create_publisher(Bool, self.estop_topic, latched_qos())
        self.state_sub = self.create_subscription(
            Bool, self.state_topic, self.on_state, latched_qos())

        # 클릭은 리더 스레드에서 들어오고, 발행은 ROS 실행기 스레드에서만 한다.
        self.clicks = queue.Queue()
        self.latched = bool(self.start_stopped)
        self.last_click_t = 0.0
        self.backend_name = None

        self.reader_stop = threading.Event()
        self.reader = threading.Thread(target=self.reader_loop, daemon=True)
        self.reader.start()

        self.timer = self.create_timer(0.02, self.drain_clicks)

        if self.start_stopped:
            self.publish_estop(True, reason='start_stopped')

        self.get_logger().info(
            f"mouse_estop 시작: backend={self.backend_param} button={self.button} "
            f"-> {self.estop_topic} (우클릭 토글, debounce={self.debounce_ms} ms)")

    # ---------------- ROS 쪽 ----------------

    def on_state(self, msg: Bool):
        """mux_controller 의 실제 래치 상태로 내부 토글을 맞춘다."""
        if bool(msg.data) != self.latched:
            self.latched = bool(msg.data)
            self.get_logger().info(
                f"래치 상태 동기화: {'정지' if self.latched else '출발'} (조이스틱/외부에서 변경됨)")

    def drain_clicks(self):
        toggles = 0
        while True:
            try:
                self.clicks.get_nowait()
            except queue.Empty:
                break
            toggles += 1
        if toggles == 0:
            return
        # 한 틱에 여러 번 들어와도 토글은 홀짝만 의미가 있다.
        if toggles % 2 == 0:
            self.get_logger().warn(f"우클릭 {toggles}회가 한 틱에 몰려 상태 변화 없음")
            return
        self.publish_estop(not self.latched, reason='우클릭')

    def publish_estop(self, stop: bool, reason: str):
        self.latched = bool(stop)
        msg = Bool()
        msg.data = self.latched
        self.estop_pub.publish(msg)
        if self.latched:
            self.get_logger().error(f"마우스 E-STOP 걸림 ({reason}). 다시 우클릭하면 재출발.")
        else:
            self.get_logger().warn(f"마우스 E-STOP 해제 ({reason}). AUTO 주행 재개.")

    def on_click(self):
        """리더 스레드에서 호출. 디바운스만 하고 큐에 넣는다."""
        now = time.monotonic()
        if (now - self.last_click_t) * 1000.0 < self.debounce_ms:
            return
        self.last_click_t = now
        self.clicks.put(now)

    # ---------------- 입력 백엔드 ----------------

    def reader_loop(self):
        want = (self.backend_param or 'auto').lower()

        while not self.reader_stop.is_set():
            started = False
            if want in ('auto', 'evdev'):
                started = self.run_evdev()
                if started:
                    # run_evdev 는 장치가 빠질 때만 돌아온다. auto 면 재연결을 계속 시도.
                    continue
                if want == 'evdev':
                    self.get_logger().error(
                        "backend=evdev 인데 마우스 장치를 열 수 없다. "
                        "sudo cp stack_master/scripts/99-f1tenth-input.rules /etc/udev/rules.d/ && "
                        "sudo udevadm control --reload-rules && sudo udevadm trigger 실행 후 재시도.")
                    self.reader_stop.wait(5.0)
                    continue
                self.get_logger().warn("evdev 로 마우스를 열 수 없어 pynput(X11) 로 넘어간다.")

            if want in ('auto', 'pynput'):
                if self.run_pynput():
                    return
                self.get_logger().error("pynput(X11) 후킹도 실패. 5초 뒤 재시도.")

            self.reader_stop.wait(5.0)

    def find_evdev_device(self):
        import evdev

        if self.device:
            return self.device

        best = None
        for path in evdev.list_devices():
            try:
                dev = evdev.InputDevice(path)
            except OSError:
                continue
            keys = dev.capabilities().get(evdev.ecodes.EV_KEY, [])
            if evdev.ecodes.BTN_RIGHT not in keys:
                dev.close()
                continue
            # 상대 좌표(EV_REL)가 있으면 진짜 마우스일 가능성이 높다.
            has_rel = evdev.ecodes.EV_REL in dev.capabilities()
            dev.close()
            if has_rel:
                return path
            best = best or path
        return best

    def run_evdev(self) -> bool:
        """마우스를 열었으면 True(장치가 빠질 때까지 블록), 못 열었으면 즉시 False."""
        try:
            import evdev
        except ImportError:
            self.get_logger().warn("python3-evdev 가 없다 (pip3 install evdev).")
            return False

        btn_code = {
            'right': evdev.ecodes.BTN_RIGHT,
            'left': evdev.ecodes.BTN_LEFT,
            'middle': evdev.ecodes.BTN_MIDDLE,
        }.get((self.button or 'right').lower(), evdev.ecodes.BTN_RIGHT)

        try:
            path = self.find_evdev_device()
            if not path:
                return False
            dev = evdev.InputDevice(path)
        except OSError as exc:
            self.get_logger().warn(f"evdev 열기 실패: {exc}")
            return False

        self.backend_name = 'evdev'
        self.get_logger().info(f"evdev 백엔드 사용: {path} ({dev.name})")
        try:
            for event in dev.read_loop():
                if self.reader_stop.is_set():
                    break
                # value: 1=누름, 0=뗌, 2=오토리핏 -> 누름 에지만 센다.
                if event.type == evdev.ecodes.EV_KEY and event.code == btn_code and event.value == 1:
                    self.on_click()
        except OSError as exc:
            self.get_logger().error(f"마우스 연결 끊김 ({exc}). 재연결 대기.")
        finally:
            try:
                dev.close()
            except Exception:
                pass
        return True

    def run_pynput(self) -> bool:
        if self.display:
            os.environ['DISPLAY'] = self.display
        if not os.environ.get('DISPLAY'):
            self.get_logger().warn("DISPLAY 가 없어 pynput 을 쓸 수 없다 (display 파라미터로 지정 가능).")
            return False

        try:
            from pynput import mouse
        except Exception as exc:
            self.get_logger().warn(f"pynput 사용 불가: {exc}")
            return False

        target = {
            'right': mouse.Button.right,
            'left': mouse.Button.left,
            'middle': mouse.Button.middle,
        }.get((self.button or 'right').lower(), mouse.Button.right)

        def on_click(x, y, button, pressed):
            if pressed and button == target:
                self.on_click()

        try:
            listener = mouse.Listener(on_click=on_click)
            listener.start()
        except Exception as exc:
            self.get_logger().warn(f"pynput 리스너 시작 실패: {exc}")
            return False

        self.backend_name = 'pynput'
        self.get_logger().info(f"pynput(X11) 백엔드 사용: DISPLAY={os.environ['DISPLAY']}")
        self.reader_stop.wait()
        listener.stop()
        return True

    def destroy_node(self):
        self.reader_stop.set()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MouseEStop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
