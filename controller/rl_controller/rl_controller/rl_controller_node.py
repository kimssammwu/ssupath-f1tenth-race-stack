#!/usr/bin/env python3
"""DACER++ 강화학습 정책(dacerpp_isaaclab) 실차 컨트롤러.

Isaac Lab 에서 학습한 디퓨전 정책을 그대로 싣고, 학습 관측을 실차 토픽으로
재구성해 30Hz 로 /l1controller/control 을 발행한다. mux_controller 가 그 값을
/drive 로 내보내므로 조이스틱 오버라이드와 /e_stop 이 그대로 살아 있다.

관측 차원은 config 의 curv_lookahead 길이 k 로 정해진다: 32 + 4 + 2k + 4 + 5 + 2.
아래 인덱스는 **지금 실린 세대(k=6, obs_dim=60)** 기준이다. k=5 세대(obs_dim=58)를
실으면 [41] 이후가 두 칸씩 당겨진다 — 노드는 차원을 계산해서 검사하므로 불일치는
기동 시 에러로 막힌다(조용히 틀리지는 않는다).

관측 구성 (dacerpp_lab/racing_env.py `_observe_car` 와 순서/정규화 동일):
    [0:32]  스캔 32빔 / 10m,  전방 ±135도   <- 벽 + 장애물 + 상대차가 모두 섞여 들어온다
    [32]    속력 / v_max(10)                <- 학습 정규화 상수, 주행 v_max 와 무관
    [33:35] sin/cos(헤딩오차)
    [35]    횡오차 / 지역 반폭   (±1 = 벽)
    [36:42] 전방 곡률 6개  (+5/15/30/60/90/120 idx = 0.75/2.25/4.5/9/13.5/18m), ±3 클립
    [42:49] 현재+전방 반폭 7개 / 2.5m
    [49:53] 직전 2스텝의 '명령' 행동 (지연 하 Markov 복원용)
    [53:58] 상대차량 5개 = [rel_x/10, rel_y/10, (v_self-v_opp)/10, gap_s/10, visible]
    [58]    요레이트 / 4.0
    [59]    횡속도 / 3.0

장애물 vs 상대차 (학습 env_cfg.obstacles_enabled 주석 참조):
    학습은 두 가지를 '다른 채널'로 준다.
      - 장애물(맵에 없는 ≤50cm 물체): 별도 상태 채널이 없다. 32빔 스캔에 footprint
        를 오버레이할 뿐이다 -> 실차에서도 /livox/lidar 원본 클라우드에 물리적으로
        잡히므로 이 노드의 의사 스캔이 그대로 재현한다. **장애물 인지 노드 불필요.**
      - 상대차: 스캔에도 잡히고, 추가로 위 5개 특징으로 명시적으로 들어온다.
        -> 실차에서는 perception 의 tracking 노드가 EKF 로 추적한 '동적' 물체
           (/perception/obstacles 중 is_static=False)를 이 5개로 변환해 넣는다.
    즉 인지 노드를 고칠 필요는 없고, 그 출력에서 동적 객체만 골라 쓰면 된다.

행동 -> 명령: steer = a0 * max_steering_angle,
             speed = v_min + (a1+1)/2 * (v_max - v_min)   (학습의 속도 커리큘럼과 동일 형태)
"""
from __future__ import annotations

import math
import os
import time
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from f110_msgs.msg import L1controllerControl, ObstacleArray, WpntArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan, PointCloud2
from std_msgs.msg import Float32MultiArray

from .checkpoints import (DEFAULT_CKPT, load_run_config, resolve_checkpoint,
                          resolve_run)
from .global_path import GlobalPath
from .scan_builder import PseudoScanBuilder
from .track_reference import TrackReference


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class RLControllerNode(Node):

    def __init__(self):
        super().__init__("rl_controller")

        # ---------------- 파라미터 ---------------- #
        p = self.declare_parameter
        p("checkpoint", DEFAULT_CKPT)
        # 학습 런 폴더로 가중치를 고르는 경로 (launch 인자 ckpt_dir). 비어 있으면
        # checkpoint 파라미터를 그대로 쓴다. 런 폴더 안에 날짜 폴더가 한 겹 더
        # 있어도 알아서 찾는다 (models/20260822_5768/20260822_1/pow_healthy.pt).
        p("ckpt_dir", "")
        # 런 폴더의 run_config.json 에서 관측 규약(curv_lookahead / curv_clip)을
        # 읽어 자동으로 맞춘다. ★curv_clip 은 틀려도 차원이 안 바뀌어 노드가 못
        # 막는 값이라(= 조용히 틀린다), 체크포인트 옆의 학습 기록을 정답으로 삼는다.
        p("obs_from_run_config", True)
        p("device", "cpu")                  # cpu / cuda / auto (젯슨 pip torch 는 cuda 커널 없음)
        p("torch_threads", 1)
        p("policy_seed", 0)
        p("num_action_candidates", 1)       # >1 이면 QVN 으로 후보 선택 (학습 기본 1)
        p("explore_noise_std", 0.0)         # 실차는 탐험 노이즈 없음

        p("control_rate", 30.0)             # 학습 제어 주기 (decimation 4 @120Hz)
        p("v_max", 5.0)                     # 행동->속도 매핑 상한 [m/s]
        p("v_min", 1.0)                     # 학습 RacingCfg.v_min
        p("obs_v_max", 10.0)                # 학습 RacingCfg.v_max (관측 정규화 — 고정)
        p("max_steering_angle", 0.42)       # 학습 RacingCfg.max_steering_angle
        p("speed_mode", "scale")            # scale | clip

        p("n_beams", 32)
        p("scan_fov", 2.356)
        p("scan_max_range", 10.0)
        p("scan_z_min", 0.02)               # base_link 기준 높이 밴드 [m]
        p("scan_z_max", 0.30)
        p("scan_min_range", 0.05)
        p("self_box", [-0.40, 0.55, 0.20])  # 자차 반사 제외 박스 [x_min, x_max, |y|]
        p("scan_deskew", True)
        p("lidar_frame", "livox_frame")
        p("lidar_translation", [0.27, 0.0, 0.07])
        p("lidar_yaw", 1.5707963)

        # ★아래 두 기본값은 **학습 env_cfg.RacingCfg 의 현재 기본값**과 맞춰 둔다
        #   (2026-08-21 기준 (5,15,30,60,90) / 3.0). 실제 주행은 config yaml 이 항상
        #   덮어쓰므로, 이건 yaml 없이 띄웠을 때 조용히 구세대 규약으로 도는 것을
        #   막기 위한 것이다. curv_clip 은 차원이 안 바뀌어 틀려도 노드가 못 막는다.
        p("curv_lookahead", [5, 15, 30, 60, 90])
        p("curv_clip", 3.0)
        p("hw_ref", 2.5)                    # 학습 0.5*TrackParams.width_max
        p("yawrate_norm", 4.0)
        p("vy_norm", 3.0)
        p("act_hist_len", 2)

        # ---------------- 상대차량 관측 ---------------- #
        p("opponent_enabled", True)
        p("opponent_topic", "/perception/obstacles")
        p("global_wpnt_topic", "/global_waypoints")
        p("opp_timeout", 0.4)               # 이보다 오래 갱신이 없으면 미검출로 되돌림
        p("opp_min_speed", 0.3)             # 이보다 느리면 상대차로 인정 안 함(유령 방지)
        p("opp_require_dynamic", True)      # is_static=False 인 것만 상대차로 본다
        p("opp_gap_norm", 10.0)             # 학습 gap_s 정규화 상수(고정)
        # 대회 규정상 각 차량 뒷부분에 '감지 박스'(12x12cm, 지면 10~30cm)를 단다.
        # 실차 LiDAR 는 그 높이 밴드에서 이 박스만 보므로(섀시 축고 6cm 는 밴드 아래)
        # perception 이 주는 위치는 '박스 중심' = 차중심에서 뒤로 이만큼이다.
        # 학습 관측의 rel_x/rel_y/gap_s 는 '차중심' 기준이라 이 오프셋을 되돌린다
        # (학습 env_cfg.det_box_rear 와 같은 값). 0.0 = 보정 끔(박스 미장착 시).
        p("opp_det_box_rear", 0.20)

        p("odom_topic", "/car_state/odom")
        p("centerline_topic", "/centerline_waypoints")
        p("lidar_topic", "/livox/lidar")
        p("laserscan_topic", "/scan")
        p("scan_source", "cloud")           # cloud | laserscan
        p("imu_topic", "/sensors/imu/raw")
        p("yawrate_source", "auto")         # auto | odom | imu
        p("drive_topic", "/l1controller/control")

        p("odom_timeout", 0.2)
        p("scan_timeout", 0.5)
        p("publish_debug", True)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.rate = float(g("control_rate"))
        self.v_max = float(g("v_max"))
        self.v_min = float(g("v_min"))
        self.obs_v_max = float(g("obs_v_max"))
        self.max_steer = float(g("max_steering_angle"))
        self.speed_mode = str(g("speed_mode")).lower()
        self.hw_ref = float(g("hw_ref"))
        self.yawrate_norm = float(g("yawrate_norm"))
        self.vy_norm = float(g("vy_norm"))
        self.curv_off = [int(v) for v in g("curv_lookahead")]
        self.curv_clip = float(g("curv_clip"))
        self.width_off = [0] + self.curv_off
        self.act_hist_len = int(g("act_hist_len"))
        self.odom_timeout = float(g("odom_timeout"))
        self.scan_timeout = float(g("scan_timeout"))
        self.publish_debug = bool(g("publish_debug"))
        self.yawrate_source = str(g("yawrate_source")).lower()
        self.opp_enabled = bool(g("opponent_enabled"))
        self.opp_timeout = float(g("opp_timeout"))
        self.opp_min_speed = float(g("opp_min_speed"))
        self.opp_require_dynamic = bool(g("opp_require_dynamic"))
        self.opp_gap_norm = float(g("opp_gap_norm"))
        self.opp_det_box_rear = float(g("opp_det_box_rear"))

        if self.v_max <= self.v_min:
            raise RuntimeError(f"v_max({self.v_max}) 는 v_min({self.v_min}) 보다 커야 합니다")

        # ---------------- 정책 로드 ---------------- #
        from .torch_loader import import_torch, resolve_device
        torch = import_torch()
        torch.set_num_threads(max(1, int(g("torch_threads"))))
        device = resolve_device(torch, str(g("device")))
        ckpt_dir = str(g("ckpt_dir")).strip()
        if ckpt_dir:
            ckpt = resolve_run(ckpt_dir, str(g("checkpoint")))
            self.get_logger().info(f"ckpt_dir='{ckpt_dir}' -> {ckpt}")
        else:
            ckpt = resolve_checkpoint(str(g("checkpoint")))
        if not os.path.isfile(ckpt):
            raise RuntimeError(f"체크포인트를 찾을 수 없습니다: {ckpt}")
        if bool(g("obs_from_run_config")):
            self._sync_obs_contract(load_run_config(ckpt))

        from .policy import DacerPolicy
        t0 = time.perf_counter()
        self.policy = DacerPolicy.load(
            ckpt, device=device, num_candidates=int(g("num_action_candidates")),
            seed=int(g("policy_seed")), explore_noise_std=float(g("explore_noise_std")))
        self.policy.warmup(10)
        self.get_logger().info(
            f"정책 로드 완료: {os.path.basename(ckpt)} obs_dim={self.policy.obs_dim} "
            f"act_dim={self.policy.act_dim} risk={self.policy.risk_measure}"
            f"({self.policy.risk_eta}) T={self.policy._T} "
            f"cand={self.policy.num_candidates} device={device} "
            f"({time.perf_counter() - t0:.1f}s)")

        self.obs_dim = self.policy.obs_dim
        expected = (int(g("n_beams")) + 1 + 2 + 1 + len(self.curv_off)
                    + len(self.width_off) + 2 * self.act_hist_len + 5 + 2)
        if expected != self.obs_dim:
            raise RuntimeError(
                f"관측 차원 불일치: 체크포인트 {self.obs_dim} vs 구성 {expected}. "
                f"n_beams/curv_lookahead/act_hist_len 파라미터를 학습값과 맞추세요.")

        # ---------------- 스캔 빌더 ---------------- #
        self.scan = PseudoScanBuilder(
            n_beams=int(g("n_beams")), fov=float(g("scan_fov")),
            max_range=float(g("scan_max_range")), z_min=float(g("scan_z_min")),
            z_max=float(g("scan_z_max")), min_range=float(g("scan_min_range")),
            self_box=tuple(float(v) for v in g("self_box")), deskew=bool(g("scan_deskew")))
        self.scan.set_extrinsic(np.array([float(v) for v in g("lidar_translation")]),
                                float(g("lidar_yaw")))
        self.scan_source = str(g("scan_source")).lower()

        # ---------------- 상태 ---------------- #
        self.track: Optional[TrackReference] = None
        self.glob: Optional[GlobalPath] = None      # 상대차 Frenet -> 맵 좌표 변환용
        self.pose: Optional[Tuple[float, float, float]] = None    # x, y, yaw
        self.vx = self.vy = self.wz = 0.0
        self.imu_wz: Optional[float] = None
        self.t_odom = -1.0
        self.t_scan = -1.0
        self.opp_xy: Optional[Tuple[float, float]] = None         # 상대차 맵 좌표
        self.opp_speed = 0.0
        self.t_opp = -1.0
        self._opp_visible = False
        self.hist = np.zeros(2 * self.act_hist_len, dtype=np.float32)
        self.last_obs: Optional[np.ndarray] = None
        self._stopped = True
        self._odom_wz_zero_since: Optional[float] = None
        self._infer_ms = 0.0

        # 디스큐용 포즈 링버퍼 (2초 @100Hz)
        self._buf_n = 256
        self._buf_t = np.zeros(self._buf_n)
        self._buf_x = np.zeros(self._buf_n)
        self._buf_y = np.zeros(self._buf_n)
        self._buf_yaw = np.zeros(self._buf_n)
        self._buf_i = 0
        self._buf_filled = 0

        # ---------------- 통신 ---------------- #
        # global_republisher 는 5초마다 재발행한다(volatile) -> 일반 QoS 로 충분.
        self.create_subscription(WpntArray, str(g("centerline_topic")), self.wpnt_cb, 10)
        self.create_subscription(Odometry, str(g("odom_topic")), self.odom_cb, 10)
        self.create_subscription(Imu, str(g("imu_topic")), self.imu_cb, 10)
        if self.scan_source == "laserscan":
            self.create_subscription(LaserScan, str(g("laserscan_topic")), self.laserscan_cb, 1)
        else:
            self.create_subscription(PointCloud2, str(g("lidar_topic")), self.cloud_cb, 1)
        if self.opp_enabled:
            # tracking 노드의 s/d 는 '레이스라인'(/global_waypoints) 기준이라
            # 맵 좌표로 되돌리려면 그 웨이포인트가 필요하다 (중심선과 다른 곡선).
            self.create_subscription(WpntArray, str(g("global_wpnt_topic")),
                                     self.global_wpnt_cb, 10)
            self.create_subscription(ObstacleArray, str(g("opponent_topic")),
                                     self.obstacles_cb, 5)

        self.drive_pub = self.create_publisher(L1controllerControl, str(g("drive_topic")), 1)
        if self.publish_debug:
            self.scan_pub = self.create_publisher(LaserScan, "/rl_controller/scan", 1)
            self.obs_pub = self.create_publisher(Float32MultiArray, "/rl_controller/obs", 1)
            self.opp_pub = self.create_publisher(Float32MultiArray, "/rl_controller/opponent", 1)

        self.timer = self.create_timer(1.0 / self.rate, self.control_loop)
        self.get_logger().info(
            f"rl_controller 시작: {self.rate:.0f}Hz, v={self.v_min:.1f}~{self.v_max:.1f}m/s "
            f"({self.speed_mode}), |steer|<={self.max_steer:.2f}rad, "
            f"스캔소스={self.scan_source}, 곡률클립=±{self.curv_clip:.0f}, "
            f"상대차관측={'on' if self.opp_enabled else 'off'}")

    # ------------------------------------------------------------------ #
    # 콜백
    # ------------------------------------------------------------------ #
    def _sync_obs_contract(self, run_cfg: dict):
        """체크포인트 옆 run_config.json 의 관측 규약을 채택한다.

        yaml 과 다르면 **학습 기록 쪽을 쓰고** 크게 경고한다. 차원을 바꾸는
        curv_lookahead 는 어차피 아래 차원 검사가 막아 주지만, curv_clip 은
        차원이 그대로라 틀려도 아무도 못 막는다 — 그게 이 함수의 존재 이유다.
        """
        if not run_cfg:
            self.get_logger().warn(
                "체크포인트 옆에 run_config.json 이 없습니다 -> 관측 규약을 yaml 값 "
                "그대로 씁니다. curv_clip 이 학습값과 다르면 조용히 틀립니다.")
            return
        src = run_cfg.get("_path", "run_config.json")
        off = run_cfg.get("curv_lookahead")
        if off is not None:
            off = [int(v) for v in off]
            if off != self.curv_off:
                self.get_logger().warn(
                    f"curv_lookahead 를 학습값으로 교정: yaml {self.curv_off} -> {off} "
                    f"(출처 {src})")
                self.curv_off = off
                self.width_off = [0] + off
        clip = run_cfg.get("curv_clip")
        if clip is not None and abs(float(clip) - self.curv_clip) > 1e-9:
            self.get_logger().warn(
                f"curv_clip 을 학습값으로 교정: yaml {self.curv_clip} -> {float(clip)} "
                f"(출처 {src}) ★이 값은 차원이 안 바뀌어 검사로는 못 잡는다")
            self.curv_clip = float(clip)
        mu = run_cfg.get("mu_range")
        if mu is not None:
            self.get_logger().info(
                f"학습 노면 마찰 밴드 mu_range={mu} (실측 노면이 이 밴드보다 미끄러우면 "
                f"코너에서 언더스티어 — 배포 시점에 고칠 수단이 없다)")

    def wpnt_cb(self, msg: WpntArray):
        if self.track is not None or not msg.wpnts:
            return
        try:
            self.track = TrackReference.from_centerline_waypoints(
                [w.x_m for w in msg.wpnts], [w.y_m for w in msg.wpnts],
                [w.d_left for w in msg.wpnts], [w.d_right for w in msg.wpnts],
                logger=self.get_logger())
        except Exception as exc:                                   # noqa: BLE001
            self.get_logger().error(f"중심선 기준선 생성 실패: {exc}")
            return
        self.get_logger().info(f"중심선 기준선 생성: {self.track.summary()}")

    def global_wpnt_cb(self, msg: WpntArray):
        if self.glob is not None or len(msg.wpnts) < 4:
            return
        try:
            self.glob = GlobalPath.from_waypoints([w.s_m for w in msg.wpnts],
                                                  [w.x_m for w in msg.wpnts],
                                                  [w.y_m for w in msg.wpnts])
        except Exception as exc:                                   # noqa: BLE001
            self.get_logger().error(f"레이스라인(Frenet 변환용) 생성 실패: {exc}")
            return
        self.get_logger().info(f"레이스라인 수신: {self.glob.summary()} (상대차 s/d -> 맵 변환용)")

    def obstacles_cb(self, msg: ObstacleArray):
        """/perception/obstacles -> 상대차량 1대의 맵 좌표(차중심) + 속력.

        tracking 노드는 한 배열 안에 정적 장애물과 상대차를 섞어 보내지만
        구분 자체는 되어 있다:
          - is_static=True  : 정적/미정 장애물 (vs=vd=0) -> RL 은 스캔으로만 본다.
          - is_static=False : opponent EKF 가 추적 중인 동적 물체 = 상대차.
        여기서는 후자만 골라 쓴다. 정적 장애물을 상대차 채널에 넣으면 학습 분포와
        어긋난다 (학습에서 장애물은 스캔에만 존재한다).

        ★ 감지 박스 보정: 대회 규정상 LiDAR 에 잡히는 것은 상대차 뒷부분의
        12x12cm 감지 박스뿐이라(높이 10~30cm 밴드) perception 이 주는 위치는
        '박스 중심'이다. 학습 관측(rel_x/rel_y/gap_s)은 '차중심' 기준이므로
        상대차 헤딩 방향으로 opp_det_box_rear 만큼 앞으로 되돌려 준다.
        헤딩은 Frenet 속도에서 얻는다: heading = psi(s) + atan2(vd, vs).
        """
        if self.glob is None:
            return
        best = None
        for o in msg.obstacles:
            if self.opp_require_dynamic and o.is_static:
                continue
            vs, vd = float(o.vs), float(o.vd)
            speed = math.hypot(vs, vd)
            if speed < self.opp_min_speed:
                continue
            s_c, d_c = float(o.s_center), float(o.d_center)
            bx, by = self.glob.to_cartesian(s_c, d_c)          # 감지 박스 중심
            if self.opp_det_box_rear != 0.0:
                heading = self.glob.psi_at(s_c) + math.atan2(vd, vs)
                xy = (bx + self.opp_det_box_rear * math.cos(heading),
                      by + self.opp_det_box_rear * math.sin(heading))
            else:
                xy = (bx, by)
            if self.pose is not None:
                d = math.hypot(xy[0] - self.pose[0], xy[1] - self.pose[1])
            else:
                d = 0.0
            if best is None or d < best[0]:
                best = (d, xy, speed)
        if best is None:
            return
        _, self.opp_xy, self.opp_speed = best
        self.t_opp = self._now()

    def odom_cb(self, msg: Odometry):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        yaw = yaw_from_quat(msg.pose.pose.orientation)
        self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)
        self.vx = msg.twist.twist.linear.x
        self.vy = msg.twist.twist.linear.y
        self.wz = msg.twist.twist.angular.z
        self.t_odom = t

        i = self._buf_i
        self._buf_t[i] = t
        self._buf_x[i] = self.pose[0]
        self._buf_y[i] = self.pose[1]
        # 보간을 위해 yaw 를 연속(unwrap)으로 저장
        prev = self._buf_yaw[(i - 1) % self._buf_n] if self._buf_filled else yaw
        self._buf_yaw[i] = prev + wrap_to_pi(yaw - prev)
        self._buf_i = (i + 1) % self._buf_n
        self._buf_filled = min(self._buf_filled + 1, self._buf_n)

        # 요레이트 소스 자동 판정: EKF wz 가 계속 정확히 0 이면 IMU 로 전환
        if self.yawrate_source == "auto":
            now = self._now()
            if self.wz == 0.0:
                if self._odom_wz_zero_since is None:
                    self._odom_wz_zero_since = now
                elif (now - self._odom_wz_zero_since > 3.0 and self.imu_wz is not None
                        and abs(self.vx) > 0.2):
                    self.yawrate_source = "imu"
                    self.get_logger().warn(
                        "/car_state/odom 의 angular.z 가 계속 0 -> 요레이트를 IMU 에서 받습니다.")
            else:
                self._odom_wz_zero_since = None

    def imu_cb(self, msg: Imu):
        # base_link <- imu 는 yaw +90도 회전(static TF)이라 z축은 그대로다.
        self.imu_wz = msg.angular_velocity.z

    def cloud_cb(self, msg: PointCloud2):
        try:
            n = self.scan.ingest_cloud(msg)
        except Exception as exc:                                   # noqa: BLE001
            self.get_logger().error(f"포인트클라우드 처리 실패: {exc}", throttle_duration_sec=2.0)
            return
        self.t_scan = self._now()
        if n == 0:
            self.get_logger().warn("높이 밴드 안에 점이 하나도 없습니다 "
                                   "(scan_z_min/scan_z_max 확인)", throttle_duration_sec=5.0)

    def laserscan_cb(self, msg: LaserScan):
        self.scan.ingest_laserscan(msg)
        self.t_scan = self._now()

    # ------------------------------------------------------------------ #
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _pose_at(self, times: np.ndarray):
        """점 시각 배열 -> (x, y, yaw). 버퍼가 못 덮으면 (None, None, None)."""
        if self._buf_filled < 4:
            return None, None, None
        if self._buf_filled < self._buf_n:
            ts = self._buf_t[:self._buf_filled]
            xs, ys, yaws = (self._buf_x[:self._buf_filled], self._buf_y[:self._buf_filled],
                            self._buf_yaw[:self._buf_filled])
        else:
            order = np.arange(self._buf_i, self._buf_i + self._buf_n) % self._buf_n
            ts, xs, ys, yaws = (self._buf_t[order], self._buf_x[order],
                                self._buf_y[order], self._buf_yaw[order])
        # 라이다/EKF 시계가 어긋나 있으면 보간이 무의미하다 -> 디스큐 생략
        if times.max() < ts[0] - 0.5 or times.min() > ts[-1] + 0.5:
            self.get_logger().warn(
                "라이다 타임스탬프가 포즈 버퍼 범위 밖입니다 -> 디스큐 생략 "
                f"(scan {times.min():.3f}~{times.max():.3f}, pose {ts[0]:.3f}~{ts[-1]:.3f})",
                throttle_duration_sec=10.0)
            return None, None, None
        return (np.interp(times, ts, xs), np.interp(times, ts, ys),
                np.interp(times, ts, yaws))

    # ------------------------------------------------------------------ #
    def build_observation(self) -> Optional[np.ndarray]:
        assert self.track is not None and self.pose is not None
        x, y, yaw = self.pose
        proj = self.track.project(x, y)
        idx = proj["idx"]
        hw_local = float(self.track.hw[idx])
        herr = wrap_to_pi(yaw - proj["psi"])

        scan, n_beams_hit = self.scan.build(self.pose, self._pose_at)
        self._last_scan = scan
        self._last_beams_hit = n_beams_hit

        curv = self.track.lookahead_curvature(idx, self.curv_off)
        width = self.track.lookahead_width(idx, self.width_off)
        speed = math.hypot(self.vx, self.vy)
        wz = self.imu_wz if (self.yawrate_source == "imu" and self.imu_wz is not None) else self.wz
        opp = self.opponent_features(x, y, yaw, speed, proj["s"])

        obs = np.concatenate([
            np.clip(scan / self.scan.max_range, 0.0, 1.0),
            [speed / self.obs_v_max, math.sin(herr), math.cos(herr)],
            [np.clip(proj["lateral"] / max(hw_local, 1e-6), -2.0, 2.0)],
            np.clip(curv, -self.curv_clip, self.curv_clip),
            np.clip(width / self.hw_ref, 0.0, 1.0),
            np.clip(self.hist, -1.0, 1.0),
            opp,                              # 상대차량 5개 (미검출이면 전부 0)
            [np.clip(wz / self.yawrate_norm, -1.0, 1.0),
             np.clip(self.vy / self.vy_norm, -1.0, 1.0)],
        ]).astype(np.float32)

        self._dbg = dict(lat=proj["lateral"], hw=hw_local, herr=herr, speed=speed,
                         wz=wz, s=proj["s"], idx=idx)
        return obs

    # ------------------------------------------------------------------ #
    def opponent_features(self, x: float, y: float, yaw: float, speed: float,
                          s_self: float) -> np.ndarray:
        """학습 racing_env._observe_car 의 상대차 5개 특징을 그대로 만든다.

            [rel_x/R, rel_y/R, (v_self - v_opp)/v_max, gap_s/10, visible]  (R = 10m)

        학습의 visible 은 '실차 LiDAR 로 실제 보이는가'를 모사한 게이트였다
        (거리 < 10m, |방위각| <= 135도, 벽에 가리지 않음). 실차에서는 검출 자체가
        LiDAR 로 이뤄지므로 가림은 이미 반영되어 있고, 거리/시야각만 다시 건다.
        미검출이면 학습과 동일하게 5개 전부 0.
        """
        opp = np.zeros(5, dtype=np.float64)
        self._opp_visible = False
        if not self.opp_enabled or self.opp_xy is None or self.track is None:
            return opp
        if self._now() - self.t_opp > self.opp_timeout:
            return opp

        dx, dy = self.opp_xy[0] - x, self.opp_xy[1] - y
        c, s = math.cos(-yaw), math.sin(-yaw)
        rel_x = dx * c - dy * s
        rel_y = dx * s + dy * c
        rng = math.hypot(rel_x, rel_y)
        brg = math.atan2(rel_y, rel_x)
        if rng >= self.scan.max_range or abs(brg) > self.scan.fov:
            return opp

        gap = self._wrap_ds(self.track.project(*self.opp_xy)["s"] - s_self)
        opp[:] = [np.clip(rel_x / self.scan.max_range, -1.0, 1.0),
                  np.clip(rel_y / self.scan.max_range, -1.0, 1.0),
                  np.clip((speed - self.opp_speed) / self.obs_v_max, -1.0, 1.0),
                  np.clip(gap / self.opp_gap_norm, -1.0, 1.0),
                  1.0]
        self._opp_visible = True
        self._opp_dbg = (rel_x, rel_y, rng, gap, self.opp_speed)
        return opp

    def _wrap_ds(self, ds: float) -> float:
        """학습 racing_env._wrap_ds 와 동일: 랩 길이 기준 [-L/2, L/2) 로 접기."""
        tot = self.track.total_s
        if ds < -0.5 * tot:
            ds += tot
        elif ds > 0.5 * tot:
            ds -= tot
        return ds

    # ------------------------------------------------------------------ #
    def control_loop(self):
        reason = self._not_ready_reason()
        if reason is not None:
            if not self._stopped:
                self.get_logger().error(f"주행 중단 -> 정지 명령: {reason}")
            self._stopped = True
            self.hist[:] = 0.0
            self.publish_cmd(0.0, 0.0)
            self.get_logger().warn(f"대기 중: {reason}", throttle_duration_sec=2.0)
            return

        try:
            obs = self.build_observation()
            t0 = time.perf_counter()
            act = self.policy.act(obs)
            self._infer_ms = (time.perf_counter() - t0) * 1e3
        except Exception as exc:                                   # noqa: BLE001
            self.get_logger().error(f"추론 실패 -> 정지: {exc}", throttle_duration_sec=1.0)
            self._stopped = True
            self.publish_cmd(0.0, 0.0)
            return

        if self._stopped:
            self.get_logger().info("주행 시작 (관측/추론 정상)")
            self._stopped = False

        steer = float(np.clip(act[0], -1.0, 1.0)) * self.max_steer
        thr = float(np.clip(act[1], -1.0, 1.0))
        if self.speed_mode == "clip":
            speed = min(self.v_min + (thr + 1.0) * 0.5 * (self.obs_v_max - self.v_min),
                        self.v_max)
        else:
            speed = self.v_min + (thr + 1.0) * 0.5 * (self.v_max - self.v_min)
        self.publish_cmd(steer, speed)

        # 명령 히스토리 롤 (최신이 앞) — 학습 racing_env 와 동일
        self.hist = np.concatenate([[act[0], act[1]], self.hist[:-2]]).astype(np.float32)
        self.last_obs = obs

        if self.publish_debug:
            self.publish_debug_msgs(obs)
        d = self._dbg
        if self._opp_visible:
            rx, ry, rng, gap, vopp = self._opp_dbg
            opp_txt = f" opp=({rx:+.1f},{ry:+.1f})m d={rng:.1f} gap={gap:+.1f} v={vopp:.1f}"
        else:
            opp_txt = ""
        self.get_logger().info(
            f"v={d['speed']:.2f}->{speed:.2f} steer={steer:+.3f} lat={d['lat']:+.2f}/"
            f"{d['hw']:.2f} herr={d['herr']:+.2f} wz={d['wz']:+.2f} "
            f"beams={self._last_beams_hit}/{self.scan.n_beams} inf={self._infer_ms:.1f}ms"
            + opp_txt,
            throttle_duration_sec=1.0)

    def _not_ready_reason(self) -> Optional[str]:
        if self.track is None:
            return "중심선 웨이포인트 대기 중 (/centerline_waypoints)"
        if self.pose is None:
            return "측위 대기 중 (/car_state/odom)"
        now = self._now()
        if now - self.t_odom > self.odom_timeout:
            return f"측위 끊김 ({now - self.t_odom:.2f}s)"
        if self.t_scan < 0.0:
            return "라이다 대기 중"
        if now - self.t_scan > self.scan_timeout:
            return f"라이다 끊김 ({now - self.t_scan:.2f}s)"
        return None

    # ------------------------------------------------------------------ #
    def publish_cmd(self, steering_angle: float, speed: float):
        msg = L1controllerControl()
        msg.steering_angle = float(steering_angle)
        msg.speed = float(speed)
        self.drive_pub.publish(msg)

    def publish_debug_msgs(self, obs: np.ndarray):
        ls = LaserScan()
        ls.header.stamp = self.get_clock().now().to_msg()
        ls.header.frame_id = "base_link"
        ls.angle_min = -self.scan.fov
        ls.angle_max = self.scan.fov
        ls.angle_increment = self.scan.spacing
        ls.range_min = float(self.scan.min_range)
        ls.range_max = float(self.scan.max_range)
        ls.ranges = [float(v) for v in self._last_scan]
        self.scan_pub.publish(ls)

        m = Float32MultiArray()
        m.data = [float(v) for v in obs]
        self.obs_pub.publish(m)

        # 상대차 5개 특징만 따로 (rqt_plot 으로 검출 유무를 바로 보기 위함)
        o = Float32MultiArray()
        o.data = [float(v) for v in obs[-7:-2]]
        self.opp_pub.publish(o)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RLControllerNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            # 종료 직전 정지 명령 (mux 가 마지막 값을 붙들고 있지 않게)
            try:
                node.publish_cmd(0.0, 0.0)
            except Exception:                                      # noqa: BLE001
                pass
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
