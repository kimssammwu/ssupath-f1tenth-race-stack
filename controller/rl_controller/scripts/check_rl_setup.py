#!/usr/bin/env python3
"""rl_controller 사전 점검 도구 (주행 전에 오프라인으로 돌려볼 것).

  1) 체크포인트 로드 + 추론 지연 측정
  2) 맵의 global_waypoints.json 으로 중심선 기준선 재구성 (학습과 동일 전처리)
  3) (선택) rosbag2 의 /livox/lidar 로 의사 스캔 통계 -> 높이 밴드 튜닝

사용 예:
  ros2 run rl_controller check_rl_setup.py \
      --map ~/forza_ws/race_stack/stack_master/maps/lobby_0728 \
      --bag ~/forza_ws/race_stack/lobby_0728
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np


# --------------------------------------------------------------------------- #
# 트랙 형상 요약 (참고용)
#
# ★ 곡률/폭 임계값으로 '이 맵은 위험하다'를 판정하려 했으나 실패했다. 학습에 쓰인
#   절차 생성 트랙 60종을 실제로 재생성해 비교해 보면 |kappa|max 중앙 1.44/최대 1.84,
#   급코너 방향 반전 최소간격 중앙 1.05m 로, 사고가 난 lobby_0730(1.85 / 1.20m)보다
#   오히려 빡빡하다. 즉 이 지표들로는 학습 분포 안팎이 갈리지 않는다.
#   맵이 안전한지는 --rollout 으로 '실제로 돌려보는' 것이 유일하게 믿을 만하다.
# --------------------------------------------------------------------------- #
WHEELBASE = 0.33
MAX_STEER = 0.42
K_VEHICLE = math.tan(MAX_STEER) / WHEELBASE      # 차량이 낼 수 있는 최대 곡률 1.35


def describe_track(tr) -> None:
    """맵 형상 요약. 판정이 아니라 참고 정보다 (위 주석 참조)."""
    k = np.abs(tr.kappa)
    ds = tr.total_s / len(k)

    def spans(mask):
        out, start = [], None
        for i, v in enumerate(mask):
            if v and start is None:
                start = i
            elif not v and start is not None:
                out.append((tr.s[start], tr.s[i - 1] + ds))
                start = None
        if start is not None:
            out.append((tr.s[start], tr.s[-1]))
        return out

    hard = k > K_VEHICLE
    if hard.any():
        print(f"    중심선 추종 불가(|kappa|>{K_VEHICLE:.2f}, 최소회전반경 "
              f"{WHEELBASE / math.tan(MAX_STEER):.2f}m) {hard.sum() * ds:.1f}m "
              f"({hard.mean():.1%}) @ s = "
              + ", ".join(f"{a:.1f}~{b:.1f}" for a, b in spans(hard)))
        print("      -> 중심선을 그대로 못 따라가므로 코리도 폭을 쓰는 라인이 필요하다 "
              "(학습 절차 트랙에도 이런 구간은 있다: 60종 중 33종, 최대 2.5%)")
    # 곡률 관측 클립은 실린 체크포인트와 세트다 (offline_sim.CURV_CLIP =
    # config/rl_controller.yaml 의 curv_clip = run_config.json 의 curv_clip).
    from rl_controller.offline_sim import CURV_CLIP
    clipped = k > CURV_CLIP
    if clipped.any():
        print(f"    곡률 관측 포화(|kappa|>{CURV_CLIP:.0f} 은 관측에서 잘림) "
              f"{clipped.sum() * ds:.1f}m ({clipped.mean():.1%})")
    sig = np.where(k > 0.5, np.sign(tr.kappa), 0)
    idx = [i for i, v in enumerate(sig) if v != 0]
    gaps = [(b - a) * ds for a, b in zip(idx, idx[1:]) if sig[a] != sig[b]]
    if gaps:
        print(f"    급코너 방향 반전: {len(gaps)}회, 최소간격 {min(gaps):.2f}m "
              f"(학습 절차 트랙 중앙 1.05m)")
    if tr.hw.min() < 0.45:
        print(f"    ! 최소 반폭 {tr.hw.min():.2f}m < 0.45m (차 반폭 0.15m) — 여유가 거의 없다")


def main() -> int:
    ap = argparse.ArgumentParser()
    # 상대경로면 패키지의 models/ 기준으로 해석된다 (노드와 동일 규칙).
    ap.add_argument("--checkpoint", default="pow_healthy.pt")
    ap.add_argument("--ckpt-dir", default="",
                    help="학습 런 폴더로 가중치 선택 (예: 20260822_5768). launch 인자 "
                         "ckpt_dir 과 같은 규칙이고, 그 폴더의 run_config.json 에서 "
                         "관측 규약(curv_lookahead/curv_clip)을 자동으로 맞춘다")
    ap.add_argument("--map", default="", help="stack_master/maps/<name> 폴더")
    ap.add_argument("--bag", default="", help="rosbag2 폴더 (/livox/lidar 포함)")
    ap.add_argument("--z-bands", default="0.02:0.30,0.00:0.35,0.05:0.25",
                    help="검사할 높이 밴드 목록 (z_min:z_max, 쉼표 구분)")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--rollout", action="store_true",
                    help="맵에서 정책을 실제로 주행시켜 완주 여부/실패 지점을 본다 (수 분 소요)")
    ap.add_argument("--rollout-speeds", default="2.0,3.0,5.0")
    ap.add_argument("--rollout-starts", type=int, default=6)
    ap.add_argument("--rollout-seconds", type=float, default=30.0)
    # ★2026-08-21 기준 변경: 기본 mu 가 offline_sim.MU_NOM = 0.581 (test_0821 실측
    #   물리 mu 0.54 의 모델 캘리브레이션 값)이다. 구 1.05 는 폐기됐다 — 그 근거였던
    #   "실 노면 mu 0.85~1.07" 은 2026-08-18 의 간접 추정이었고, 0821 스키드패드
    #   직접 측정에서 0.54~0.67 로 뒤집혔다.
    # ★지금 실린 체크포인트(20260818_85110)는 mu_range (0.85,1.10) 에서 학습됐다.
    #   즉 기본값으로 돌리면 '정책이 자기 학습 세계에서 도는가'가 아니라
    #   '측정된 실제 노면에서 도는가'를 보는 것이다 — 배포 판단에는 이쪽이 맞다.
    #   학습 세계 쪽을 같이 보려면 --mu 0.85 / --mu 1.10 으로도 돌릴 것.
    #   두 결과가 갈리면 그 격차가 곧 sim2real 위험이다.
    ap.add_argument("--mu", type=float, default=None,
                    help="폐루프 시뮬 지면 마찰 (기본: 실측 캘리브레이션 0.581). "
                         "실린 체크포인트의 학습 밴드(run_config.json 의 mu_range)로도 "
                         "돌려 두 결과를 비교할 것")
    args = ap.parse_args()

    from rl_controller.checkpoints import (load_run_config, resolve_checkpoint,
                                           resolve_run)
    from rl_controller.policy import DacerPolicy
    from rl_controller.scan_builder import PseudoScanBuilder
    from rl_controller.track_reference import TrackReference

    ckpt = (resolve_run(args.ckpt_dir, args.checkpoint) if args.ckpt_dir
            else resolve_checkpoint(args.checkpoint))
    print("=" * 72)
    print(f"[1] 체크포인트: {ckpt}")
    if not os.path.isfile(ckpt):
        print("    ! 파일이 없습니다")
        return 1
    # 관측 규약을 학습 기록에 맞춘다 (노드가 하는 것과 같은 일).
    run_cfg = load_run_config(ckpt)
    if run_cfg:
        from rl_controller.offline_sim import set_obs_contract
        off, clip = set_obs_contract(run_cfg.get("curv_lookahead"), run_cfg.get("curv_clip"))
        mu_band = run_cfg.get("mu_range")
        print(f"    학습 규약: curv_lookahead={list(off)} curv_clip={clip} "
              f"obs_dim={run_cfg.get('obs_dim')} mu_range={mu_band}")
        if mu_band:
            print(f"    ※ 배포 yaml 의 curv_lookahead/curv_clip 도 이 값이어야 한다")
    else:
        print("    ! run_config.json 이 없어 관측 규약을 확인하지 못했습니다 "
              "(curv_clip 이 틀리면 조용히 틀립니다)")
    t0 = time.perf_counter()
    pol = DacerPolicy.load(ckpt, device="cpu")
    print(f"    obs_dim={pol.obs_dim} act_dim={pol.act_dim} "
          f"risk={pol.risk_measure}({pol.risk_eta}) T={pol._T} "
          f"(로드 {time.perf_counter() - t0:.1f}s)")
    pol.warmup(10)
    obs = np.zeros(pol.obs_dim, dtype=np.float32)
    lat = []
    for _ in range(100):
        t0 = time.perf_counter()
        pol.act(obs)
        lat.append((time.perf_counter() - t0) * 1e3)
    lat = np.array(lat)
    print(f"    추론 지연: 평균 {lat.mean():.2f}ms p95 {np.percentile(lat, 95):.2f}ms "
          f"max {lat.max():.2f}ms  (제어주기 33.3ms)")

    if args.map:
        print("=" * 72)
        jp = os.path.join(os.path.expanduser(args.map), "global_waypoints.json")
        print(f"[2] 맵: {jp}")
        if not os.path.isfile(jp):
            print("    ! global_waypoints.json 이 없습니다")
            return 1
        w = json.load(open(jp))["centerline_waypoints"]["wpnts"]
        tr = TrackReference.from_centerline_waypoints(
            [p["x_m"] for p in w], [p["y_m"] for p in w],
            [p["d_left"] for p in w], [p["d_right"] for p in w])
        print(f"    {tr.summary()}")
        describe_track(tr)

        if args.rollout:
            from rl_controller.offline_sim import MU_NOM, evaluate
            mu = MU_NOM if args.mu is None else args.mu
            print("    ---- 폐루프 시뮬 (학습 타이어 모델, 실차 사고를 재현하는 유일한 검사) ----")
            print(f"    지면 마찰 mu={mu:.3f}"
                  + ("  (2026-08-21 실측 캘리브레이션값 = 실제 노면)" if args.mu is None
                     else f"  (실측 캘리브레이션 {MU_NOM:.3f} 대비 지정)"))
            vs = [float(v) for v in args.rollout_speeds.split(",")]
            res = evaluate(tr, pol, v_max_list=vs, starts=args.rollout_starts,
                           seconds=args.rollout_seconds, mu=mu, log=lambda m: print(m))
            best = max(res.items(), key=lambda kv: (kv[1]["ok"], kv[1]["travelled"]))
            if best[1]["ok"] < best[1]["n"]:
                print("    ==> 어떤 속도에서도 완주하지 못했습니다. 실패 지점의 s 를 보고 "
                      "그 구간을 다시 깔거나(코너를 넓게), 그 맵에서는 RL 을 쓰지 마세요.")
            else:
                print(f"    ==> v_max={best[0]:.1f} 에서 {best[1]['ok']}/{best[1]['n']} 완주. "
                      "실차는 이보다 나쁠 수 있으니 낮은 쪽부터 시작하세요.")

    if args.bag:
        print("=" * 72)
        print(f"[3] bag: {args.bag}")
        import sqlite3
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import PointCloud2

        dbs = [f for f in os.listdir(os.path.expanduser(args.bag)) if f.endswith(".db3")]
        if not dbs:
            print("    ! .db3 파일이 없습니다 (sqlite3 백엔드만 지원)")
            return 1
        con = sqlite3.connect(os.path.join(os.path.expanduser(args.bag), sorted(dbs)[0]))
        cur = con.cursor()
        row = cur.execute("SELECT id FROM topics WHERE name='/livox/lidar'").fetchone()
        if row is None:
            print("    ! /livox/lidar 토픽이 없습니다")
            return 1
        total = cur.execute("SELECT COUNT(*) FROM messages WHERE topic_id=?",
                            (row[0],)).fetchone()[0]
        # 클라우드 한 프레임이 0.5MB 라 fetchall 하면 수백 MB 를 물고 있게 된다 ->
        # 커서를 흘려보내며 step 번째만 역직렬화한다.
        step = max(1, total // max(args.frames, 1))
        it = cur.execute("SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp",
                         (row[0],))
        msgs = []
        for i, r in enumerate(it):
            if i % step == 0:
                msgs.append(deserialize_message(bytes(r[0]), PointCloud2))
                if len(msgs) >= args.frames:
                    break
        print(f"    프레임 {len(msgs)}개 (전체 {total})")
        for band in args.z_bands.split(","):
            z0, z1 = (float(v) for v in band.split(":"))
            sb = PseudoScanBuilder(z_min=z0, z_max=z1, deskew=False)
            scans, hits = [], []
            for m in msgs:
                sb.ingest_cloud(m)
                sc, nh = sb.build()
                scans.append(sc)
                hits.append(nh)
            a = np.array(scans)
            print(f"    z[{z0:.2f},{z1:.2f}]: 빔점유 {np.mean(hits) / sb.n_beams:.1%}, "
                  f"거리 p5 {np.percentile(a, 5):.2f}m p50 {np.percentile(a, 50):.2f}m "
                  f"p95 {np.percentile(a, 95):.2f}m, "
                  f"무반사(=10m) {np.mean(a >= sb.max_range - 1e-6):.1%}, "
                  f"0.3m미만 {np.mean(a < 0.3):.2%}")
        print("    -> 빔점유가 높고 '0.3m미만'이 낮은 밴드를 고를 것 "
              "(0.3m 미만이 잦으면 자차/지면 반사가 섞인 것)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
