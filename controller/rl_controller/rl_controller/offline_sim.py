"""맵을 실제로 주행해 보는 오프라인 폐루프 검증기.

주행 전에 "이 맵에서 이 정책이 도는가"를 직접 확인하기 위한 것이다. 임계값 몇 개로
맵 난이도를 점치는 것보다 훨씬 정확하다 — 2026-07-30 lobby_0730 사고 때, 곡률/폭
임계값으로는 학습 트랙과 구분되지 않았지만(절차 생성 트랙 60종을 재생성해 비교:
|kappa|max 중앙 1.44/최대 1.84, 급코너 반전 최소간격 중앙 1.05m 로 lobby_0730 보다
오히려 빡빡했다) 이 폐루프 검증은 실차와 같은 지점(s≈3.4m)에서 같은 방식으로
실패해 사고를 정확히 재현했다.

동역학은 dacerpp_lab/racing_env.py 의 `_apply_tire_forces` / `_drive_one` 을 그대로
옮긴 것이다(같은 수식, 물리 120Hz / 제어 30Hz). PhysX 강체 대신 평면 3자유도
(vx, vy, r)를 적분하고 전복/충돌은 다루지 않는다. 관성 Izz 는 학습 자산의
assets/f1tenth/f1tenth.urdf 링크 관성 합(≈0.1064, 2026-08-18 재계측 기준).

★ 순수 운동학 자전거 모델을 쓰면 안 된다: 그립 한계가 없어 풀락에서 요레이트가
  5.7rad/s(그립 한계 ~3.4의 1.7배)까지 나오고, 학습 분포 밖 관측이 되어 정책이
  발진한다. 정책이 고장난 것처럼 보이지만 하네스 문제다.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

# ---- 학습 파라미터 (dacerpp_lab/env_cfg.py TireModelCfg / RacingCfg) ----
# ★2026-08-19 동기화: 학습 레포(dacerpp-isaaclab @128d8dc)의 2026-08-18 실차 재계측을
#   그대로 옮겼다. 그 전까지 이 파일은 두 세대 뒤처져 있었다 (MASS 3.94 = 2026-07-16
#   코너웨이트, 학습은 4.24(0812) -> 4.987(0818)로 두 번 바뀌었다). 즉 검증기가 실차보다
#   27% 가벼운 차를 돌리고 있었다 — 완주하던 롤아웃이 실차에서 안 도는 방향의 오차다.
#
# 공칭 마찰. 학습 TireModelCfg.mu 와 같은 값을 쓴다 — 이 값이 학습과 다르면
# "정책이 자기 세계에서 도는가"라는 질문 자체가 성립하지 않는다.
# 이력: 0.75 -> 0.90(2026-08-07) -> 1.05(2026-08-09) -> 0.581(2026-08-21).
# ★★2026-08-21 동기화 (학습 레포 @3fc77db "fix: physics parameters").
#   test_0821 실차 실측으로 마찰/구동/제동/구름저항이 전부 바뀌었고, 구동계가
#   RWD 가 아니라 4WD 임이 확인됐다. 아래 상수는 그 커밋의 TireModelCfg 그대로다.
#   - 물리 실측 mu = 0.54 (a_lat_max/g). 다만 이 모델은 앞뒤 alpha_char 가 달라
#     두 축이 동시에 완전포화하지 못해 a_lat_max 가 mu*g 의 92.9% 밖에 안 나온다.
#     그래서 학습은 '천장'을 실측(5.3 m/s^2)에 맞춘 캘리브레이션 mu 를 쓴다:
#         mu_cal = 5.3 / (0.929 * 9.81) = 0.581
#   - 실측 mu 는 하루 안에도 크게 움직인다(0821 02시 0.66~0.67 / 06시 0.54).
#     학습 밴드는 mu_range=(0.58, 0.72) = 물리 0.54~0.67 을 /0.929 한 것.
# ★ 지금 실린 체크포인트(20260818_85110)는 **구 물리 + mu_range (0.85,1.10)** 에서
#   학습됐다. 즉 이 검증기를 기본값(0.581)으로 돌리면 '학습 세계'가 아니라
#   '측정된 실제 노면'에서 도는지를 보는 것이 된다 — 그게 지금 알아야 할 값이다.
#   학습 세계 쪽을 보려면 check_rl_setup.py --mu 0.85 ~ 1.10 으로 돌릴 것.
MU_NOM = 0.581
# 앞/뒤 특성 슬립각 분리 (2026-08-21 실차 원형스윕 7런 회귀). 같은 값을 쓰면 정상상태
# 요모멘트 평형에서 alpha_f = alpha_r 이 되어 언더스티어가 정확히 0 인 차가 된다 —
# 실차는 강한 언더스티어다. (ac_f - ac_r) = 0.086 rad 가 견고하게 식별된 값이고,
# 절대값은 mu 0.54 와 한 세트로만 의미가 있다.
ALPHA_CHAR_F, ALPHA_CHAR_R = 0.119, 0.033
# 코너웨이트 실측(2026-08-18, 배터리 장착 주행 상태): 총 4987g, 앞 2408(48.3%) / 뒤 2579(51.7%).
# lf = L x 뒤축하중비, lr = L x 앞축하중비 -> CoM 이 축거 중앙보다 5.7mm 뒤다(구 값은 앞).
MASS, COM_H, LF, LR = 4.987, 0.07, 0.171, 0.159
# ---- 종방향 (2026-08-21 실측, 학습 @3fc77db 와 동일) ----
# K_DRIVE: step 런 29계단에서 tau = m/k = 0.311s 로 식별 -> k = 4.987/0.312 = 16.0.
#   구 40.0 은 tau 0.125s 로 실차보다 2.5배 빨리 명령속도에 붙었다. 선형구간 폭이
#   f_drive_max/k = 0.78 -> 3.4 m/s 로 넓어져 일상 주행 대부분이 포화가 아니게 된다.
# F_DRIVE_MAX: 실측 가속은 마찰 한계라 분리 식별이 안 됐다(>=35N 만 확인). 4WD
#   마찰서클에서 종방향 캡이 mu*m*g 이므로, 힘 상한이 먼저 걸리지 않도록 '비구속'
#   으로 크게 둔다(55N). 즉 가감속을 정하는 것은 mu 다.
# F_BRAKE_MAX: 실측 제동 6.40 m/s^2 (accel 런 2개 0.4% 일치) ->
#   F = m*(a - c_roll*g) = 4.987*(6.40-1.02) = 26.8N.
#   구 14.2 의 근거였던 "실차 제동은 구동의 절반"은 '정책이 실제로 쓴 제동'이었지
#   '가능한 최대 제동'이 아니었다. 실측은 제동 6.40 vs 가속 5.6~6.3 으로 거의 같다.
K_DRIVE, F_DRIVE_MAX, F_BRAKE_MAX = 16.0, 55.0, 26.8
# C_ROLL: 코스트 감속이 3.2->0.1 m/s 전 구간 1.03 m/s^2 로 일정(= 순수 쿨롱 마찰,
#   모터 전류 p50 0.00A). 2런 편차 0.3% 로 이번 측정 중 신뢰도가 가장 높다.
#   상시 5.1N = f_drive_max 의 16%. 구 0.015 는 사실상 무저항이라 검증기 차가
#   타력으로 계속 굴렀는데 실차는 금방 선다.
C_ROLL, V_LAT_TAPER = 0.104, 0.3
# Izz: f1tenth.urdf 링크 관성 합(base 0.083920 + 바퀴/너클 평행축), CoM(x=-0.006) 기준.
IZZ, G = 0.1064, 9.81
MAX_STEER, STEER_LIMIT, STEER_K, STEER_VLIM = 0.42, 0.44, 10.0, 20.0
# 명령 -> 실제 조향각 배율 (학습 RacingCfg.steer_gain). 실차는 2026-08-21 실측에서
# 명령보다 12% 크게 꺾였는데(k 중앙값 1.123), 학습은 그것을 시뮬에 넣는 대신 실차
# vesc.yaml 의 steering_angle_to_servo_gain 을 -0.65 -> -0.58 로 고쳐 '명령 = 실제'
# 로 만드는 쪽을 택했다. ★둘 중 하나만 적용한다 — 실차 게인을 되돌리면 여기도 1.12.
STEER_GAIN = 1.0
PHYS_DT, DECIMATION = 1.0 / 120.0, 4
CTRL_DT = PHYS_DT * DECIMATION
N_BEAMS, FOV, RMAX = 32, 2.356, 10.0
HW_REF, OBS_VMAX, V_MIN = 2.5, 10.0, 1.0
# ★ 이 셋은 '관측 포맷'이라 **검증하려는 체크포인트**와 반드시 세트여야 한다.
#   기본값은 obs_dim=60 세대(20260818_*/20260820_* 런)다.
#   0821 물리로 재학습한 20260822_* 런은 다시 obs_dim=58 (5점, 120 제거) 이므로,
#   체크포인트를 바꿔 검증할 때는 손으로 고치지 말고 set_obs_contract() 를 쓸 것 —
#   check_rl_setup.py 가 run_config.json 을 읽어 자동으로 호출한다.
CURV_OFF = (5, 15, 30, 60, 90, 120)
WIDTH_OFF = (0,) + CURV_OFF
CURV_CLIP = 3.0


def set_obs_contract(curv_off=None, curv_clip=None):
    """관측 규약을 체크포인트의 run_config.json 값으로 맞춘다.

    curv_clip 은 차원을 안 바꾸므로 틀려도 아무 검사에 안 걸린다 — 검증기가
    학습과 다른 클립으로 돌면 '왜 실차에서만 안 되지'를 영원히 못 찾는다.
    """
    global CURV_OFF, WIDTH_OFF, CURV_CLIP
    if curv_off is not None:
        CURV_OFF = tuple(int(v) for v in curv_off)
        WIDTH_OFF = (0,) + CURV_OFF
    if curv_clip is not None:
        CURV_CLIP = float(curv_clip)
    return CURV_OFF, CURV_CLIP
OFFTRACK_MARGIN, SPIN_HERR = -0.20, 1.745
ANGLES = np.linspace(-FOV, FOV, N_BEAMS)


class CarSim:
    """평면 3자유도 + 학습 해석 타이어 모델."""

    def __init__(self, x, y, yaw, mu=MU_NOM, v0=1.0):
        self.x, self.y, self.yaw = x, y, yaw
        self.vx, self.vy, self.r, self.delta = v0, 0.0, 0.0, 0.0
        self.mu = mu

    @property
    def speed(self):
        return math.hypot(self.vx, self.vy)

    def step(self, act, v_cmd_max):
        delta_cmd = float(np.clip(act[0], -1, 1)) * MAX_STEER * STEER_GAIN
        v_cmd = V_MIN + (float(np.clip(act[1], -1, 1)) + 1.0) * 0.5 * (v_cmd_max - V_MIN)
        for _ in range(DECIMATION):
            self._substep(delta_cmd, v_cmd)

    def _substep(self, delta_cmd, v_cmd):
        d_dot = float(np.clip(STEER_K * (delta_cmd - self.delta), -STEER_VLIM, STEER_VLIM))
        self.delta = float(np.clip(self.delta + d_dot * PHYS_DT, -STEER_LIMIT, STEER_LIMIT))
        delta, vx, vy, r = self.delta, self.vx, self.vy, self.r
        L = LF + LR

        fx_des = float(np.clip(K_DRIVE * (v_cmd - vx), -F_BRAKE_MAX, F_DRIVE_MAX))
        ax_est = fx_des / MASS
        nf = max(MASS * G * LR / L - MASS * ax_est * COM_H / L, 0.0)
        nr = max(MASS * G * LF / L + MASS * ax_est * COM_H / L, 0.0)

        vx_eff = max(abs(vx), 0.5)
        taper = math.tanh(abs(vx) / V_LAT_TAPER)
        # 앞/뒤 특성 슬립각 분리 (2026-08-21) — 같은 값이면 언더스티어가 0 인 차가 된다.
        fyf = -self.mu * nf * math.tanh((math.atan2(vy + LF * r, vx_eff) - delta)
                                        / ALPHA_CHAR_F) * taper
        fyr = -self.mu * nr * math.tanh(math.atan2(vy - LR * r, vx_eff) / ALPHA_CHAR_R) * taper

        # ---- 4WD 축별 마찰서클 (2026-08-21, 섀시 4WD 확인) ----
        # 구 코드는 뒤축에만 걸었다(RWD 가정) -> 종방향 캡이 mu*N_r 뿐이라 정지가속이
        # 실측의 절반이었다. 구동력을 축하중 비례로 나눠 축별로 걸면 종방향 합 캡이
        # mu*m*g 가 되고(실측 일치), 횡력 잠식은 축별로 유지된다.
        n_tot = max(nf + nr, 1e-6)
        fx_f_des, fx_r_des = fx_des * nf / n_tot, fx_des * nr / n_tot
        scale_f = min(self.mu * nf / max(math.hypot(fx_f_des, fyf), 1e-6), 1.0)
        scale_r = min(self.mu * nr / max(math.hypot(fx_r_des, fyr), 1e-6), 1.0)
        fx_f, fyf = fx_f_des * scale_f, fyf * scale_f
        fx_r, fyr = fx_r_des * scale_r, fyr * scale_r

        f_roll = -C_ROLL * MASS * G * math.tanh(vx / 0.2)
        cd, sd = math.cos(delta), math.sin(delta)
        # 앞축 힘(종/횡)을 조향각만큼 차체 프레임으로 회전. 4WD 라 종방향 성분 fx_f 가
        # 생겼으므로 요모멘트에도 그 기여(LF * fx_f*sd)가 들어간다.
        fy_front_b = fx_f * sd + fyf * cd
        self.vx += ((fx_f * cd - fyf * sd + fx_r + f_roll) / MASS + vy * r) * PHYS_DT
        self.vy += ((fy_front_b + fyr) / MASS - vx * r) * PHYS_DT
        self.r += ((LF * fy_front_b - LR * fyr) / IZZ) * PHYS_DT
        self.yaw += self.r * PHYS_DT
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (self.vx * cy - self.vy * sy) * PHYS_DT
        self.y += (self.vx * sy + self.vy * cy) * PHYS_DT


def build_walls(tr) -> np.ndarray:
    """중심선 ± 반폭 벽 폴리라인 -> (S,2,2) 세그먼트(시작점, 방향)."""
    nrm = np.stack([-np.sin(tr.psi), np.cos(tr.psi)], axis=1)
    out = []
    for side in (+1.0, -1.0):
        poly = tr.pts + side * nrm * tr.hw[:, None]
        out.append(np.stack([poly, np.roll(poly, -1, axis=0) - poly], axis=1))
    return np.concatenate(out, axis=0)


def raycast(pos: np.ndarray, angles: np.ndarray, walls: np.ndarray,
            max_range: float = RMAX) -> np.ndarray:
    a, s = walls[:, 0, :], walls[:, 1, :]
    d = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    e = a - pos
    den = d[:, None, 0] * s[None, :, 1] - d[:, None, 1] * s[None, :, 0]
    ok = np.abs(den) > 1e-9
    q = np.where(ok, den, 1.0)
    t = (e[None, :, 0] * s[None, :, 1] - e[None, :, 1] * s[None, :, 0]) / q
    u = (e[None, :, 0] * d[:, None, 1] - e[None, :, 1] * d[:, None, 0]) / q
    hit = ok & (u >= -1e-6) & (u <= 1 + 1e-6) & (t >= 0)
    return np.minimum(np.where(hit, t, np.inf).min(axis=1), max_range)


def make_obs(tr, walls, car: CarSim, hist: np.ndarray, scan_noise: float = 0.0,
             rng: Optional[np.random.Generator] = None):
    proj = tr.project(car.x, car.y)
    idx = proj["idx"]
    hw = float(tr.hw[idx])
    herr = (car.yaw - proj["psi"] + math.pi) % (2 * math.pi) - math.pi
    scan = raycast(np.array([car.x, car.y]), car.yaw + ANGLES, walls)
    if scan_noise and rng is not None:
        scan = np.clip(scan + rng.normal(0.0, scan_noise, N_BEAMS), 0.02, RMAX)
    obs = np.concatenate([
        np.clip(scan / RMAX, 0, 1),
        [car.speed / OBS_VMAX, math.sin(herr), math.cos(herr)],
        [np.clip(proj["lateral"] / hw, -2, 2)],
        np.clip(tr.lookahead_curvature(idx, CURV_OFF), -CURV_CLIP, CURV_CLIP),
        np.clip(tr.lookahead_width(idx, WIDTH_OFF) / HW_REF, 0, 1),
        # 상대차 5개는 0 = 미검출. 이 검증기는 단독 주행만 본다.
        np.clip(hist, -1, 1), np.zeros(5),
        [np.clip(car.r / 4.0, -1, 1), np.clip(car.vy / 3.0, -1, 1)],
    ]).astype(np.float32)
    return obs, proj, hw, herr


def rollout(tr, walls, policy, v_max: float, seconds: float = 30.0, start_idx: int = 0,
            mu: float = MU_NOM, delay: int = 1, scan_noise: float = 0.02,
            seed: int = 0) -> dict:
    """한 번 주행. delay 는 명령 인가 지연(제어 스텝, 학습 act_delay 0~2 의 중앙값)."""
    rng = np.random.default_rng(seed)
    car = CarSim(float(tr.pts[start_idx, 0]), float(tr.pts[start_idx, 1]),
                 float(tr.psi[start_idx]), mu=mu, v0=1.0)
    hist = np.zeros(4, dtype=np.float32)
    queue = [np.zeros(2, dtype=np.float32) for _ in range(delay)]
    prev_s = tr.project(car.x, car.y)["s"]
    travelled, min_margin, steers = 0.0, 9.9, []

    for _ in range(int(seconds / CTRL_DT)):
        obs, proj, hw, herr = make_obs(tr, walls, car, hist, scan_noise, rng)
        min_margin = min(min_margin, hw - abs(proj["lateral"]))
        if abs(proj["lateral"]) > hw + OFFTRACK_MARGIN:
            return dict(ok=False, reason="이탈", s=proj["s"], travelled=travelled,
                        min_margin=min_margin, steers=np.array(steers))
        if abs(herr) > SPIN_HERR:
            return dict(ok=False, reason="스핀", s=proj["s"], travelled=travelled,
                        min_margin=min_margin, steers=np.array(steers))
        act = np.asarray(policy.act(obs), dtype=np.float32)
        queue.append(act)
        car.step(queue.pop(0), v_max)
        steers.append(float(act[0]) * MAX_STEER)
        hist = np.concatenate([act[:2], hist[:2]]).astype(np.float32)

        s_now = tr.project(car.x, car.y)["s"]
        ds = s_now - prev_s
        if ds < -0.5 * tr.total_s:
            ds += tr.total_s
        elif ds > 0.5 * tr.total_s:
            ds -= tr.total_s
        travelled += ds
        prev_s = s_now

    return dict(ok=True, reason="완주", s=prev_s, travelled=travelled,
                min_margin=min_margin, steers=np.array(steers))


def evaluate(tr, policy, v_max_list: Sequence[float] = (2.0, 3.0, 5.0),
             starts: int = 6, seconds: float = 30.0, mu: float = MU_NOM,
             delay: int = 1, log=print) -> dict:
    """여러 시작점 x 여러 v_max 로 돌려 완주율과 실패 지점을 보고."""
    walls = build_walls(tr)
    n = len(tr.pts)
    idxs = [int(n * j / starts) for j in range(starts)]
    result = {}
    for v_max in v_max_list:
        ok, prog, fails, sat = 0, [], [], []
        for si, s0 in enumerate(idxs):
            r = rollout(tr, walls, policy, v_max, seconds, s0, mu=mu, delay=delay, seed=si)
            ok += int(r["ok"])
            prog.append(r["travelled"])
            if len(r["steers"]):
                sat.append(float(np.mean(np.abs(r["steers"]) > 0.41)))
            if not r["ok"]:
                fails.append(f"{r['reason']}@s={r['s']:.1f}m")
        result[v_max] = dict(ok=ok, n=starts, travelled=float(np.mean(prog)), fails=fails)
        log(f"    v_max={v_max:4.1f}: 완주 {ok}/{starts}  "
            f"평균진행 {np.mean(prog):6.1f}m ({np.mean(prog) / tr.total_s:.2f}랩/{seconds:.0f}s)  "
            f"조향포화 {np.mean(sat) if sat else 0:.0%}")
        if fails:
            log(f"              실패: {', '.join(fails)}")
    return result
