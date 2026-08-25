"""체크포인트 경로 해석.

가중치는 레포 안(`models/`)에 복사돼 있고 노드는 학습 워크스페이스
(`~/shared_dir/dacerpp_isaaclab`)를 참조하지 않는다. 설정에 상대경로를 쓰면
이 모듈이 패키지의 models/ 를 기준으로 절대경로를 만들어 준다.

ROS 의존이 없는 별도 모듈인 이유: check_rl_setup.py 같은 오프라인 도구가
rclpy/f110_msgs 없이도 같은 규칙을 쓸 수 있어야 하기 때문이다.
"""
from __future__ import annotations

import glob
import json
import os

# 배포 기본 가중치. 학습 car_b(pow) 가 실차 배포 대상이다 (models/README.md 참조).
# ★2026-08-19: *_healthy.pt 로 바꿨다. 학습 train.py 는 Pow 의 조향 다양성 pstd 가
#   붕괴 기준(0.02) 위인 동안만 *_healthy.pt 를 갱신한다 — 런 최종 pow.pt 는 붕괴
#   영역(pstd 0.013~0.018)이라 배포하면 안 된다.
# models/ 에는 배포 대상 두 개만 평평하게 둔다 (setup.py 가 models/*.pt 만 설치하므로
# 날짜 폴더 안의 파일은 install 공간에 들어가지 않는다).
DEFAULT_CKPT = "pow_healthy.pt"


def models_roots() -> list:
    """models/ 후보 디렉터리 — install 공간 우선, 없으면 소스 트리."""
    roots = []
    try:
        from ament_index_python.packages import get_package_share_directory
        roots.append(os.path.join(get_package_share_directory("rl_controller"), "models"))
    except Exception:                                              # noqa: BLE001
        pass
    roots.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "models"))
    return roots


def resolve_checkpoint(raw: str) -> str:
    """절대경로는 그대로, 상대경로는 패키지 models/ 기준으로 해석."""
    path = os.path.expanduser(str(raw))
    if os.path.isabs(path):
        return path
    roots = models_roots()
    for root in roots:
        cand = os.path.join(root, path)
        if os.path.isfile(cand):
            return cand
    return os.path.join(roots[0], path) if roots else path


def resolve_run(ckpt_dir: str, name: str = DEFAULT_CKPT) -> str:
    """런 폴더 이름 -> 그 안의 가중치 절대경로.

    학습 산출물은 `models/<런>/<날짜>/pow_healthy.pt` 처럼 날짜 폴더가 한 겹 더
    있어서, 런 이름만 주면 그 아래를 훑어 파일을 찾는다
    (`ckpt_dir:=20260822_5768` -> `.../20260822_5768/20260822_1/pow_healthy.pt`).

    같은 이름이 여러 개면 경로가 사전순으로 가장 뒤인 것(= 보통 최신 날짜)을 쓴다.
    """
    ckpt_dir = str(ckpt_dir).strip()
    if not ckpt_dir:
        raise ValueError("ckpt_dir 가 비어 있습니다")
    base = os.path.expanduser(ckpt_dir)
    bases = [base] if os.path.isabs(base) else [os.path.join(r, base) for r in models_roots()]
    tried = []
    for b in bases:
        tried.append(b)
        if not os.path.isdir(b):
            continue
        direct = os.path.join(b, name)
        if os.path.isfile(direct):
            return direct
        hits = sorted(glob.glob(os.path.join(b, "**", name), recursive=True))
        if hits:
            return hits[-1]
    raise FileNotFoundError(
        f"ckpt_dir='{ckpt_dir}' 안에서 '{name}' 을 찾지 못했습니다. 찾아본 곳: {tried}")


def load_run_config(ckpt_path: str) -> dict:
    """가중치 옆(또는 부모)의 run_config.json 을 읽는다. 없으면 {}.

    학습 설정 중 **관측 규약**(curv_lookahead / curv_clip)은 체크포인트와 한 세트라,
    배포 yaml 이 아니라 이 파일이 정답이다. 노드가 이걸 읽어 자동으로 맞춘다
    (README 의 '차원은 그대로인데 정규화/클립만 바뀌어 조용히 틀리는' 사고 방지).
    """
    d = os.path.dirname(os.path.abspath(ckpt_path))
    for cand in (d, os.path.dirname(d)):
        f = os.path.join(cand, "run_config.json")
        if os.path.isfile(f):
            try:
                with open(f, encoding="utf-8") as fh:
                    cfg = json.load(fh)
                cfg["_path"] = f
                return cfg
            except Exception:                                      # noqa: BLE001
                return {}
    return {}
