import os
from glob import glob

from setuptools import setup

package_name = 'rl_controller'


def _run_dir_data_files():
    """models/ 하위 학습 런 폴더를 install 공간에도 넣는다 (launch 인자 ckpt_dir 용).

    최상단의 평평한 *.pt 가 '기본 배포 대상'이라는 규칙은 그대로다. 여기서 추가로
    까는 것은 런 폴더뿐이고, 그중에서도 **`*_healthy.pt` 만** 깐다 —
    런 최종 `pow.pt`/`cvar.pt` 는 조향 다양성이 붕괴한 스냅샷이라 배포 대상이
    아니다(models/README.md). 실수로 고를 수 없게 아예 안 깐다.
    run_config.json 은 노드가 관측 규약(curv_lookahead/curv_clip)을 자동으로
    맞추는 데 쓰므로 반드시 같이 깔아야 한다.
    """
    out = {}
    for pat in ('*_healthy.pt', '*.json'):
        for f in glob(os.path.join('models', '*', '**', pat), recursive=True):
            out.setdefault(os.path.join('share', package_name, os.path.dirname(f)),
                           []).append(f)
    return sorted(out.items())

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),
        # 가중치를 install 공간에도 넣는다. rl_controller.yaml 의 checkpoint 가
        # 상대경로면 share/rl_controller/models/ 기준으로 해석되므로, 이게 빠지면
        # 학습 워크스페이스 경로에 다시 의존하게 된다.
        # models/ 는 최신 런의 pow.pt / cvar.pt 만 두는 평평한 디렉터리다 (날짜 폴더 없음)
        # -> 가중치를 갱신할 때 이 규칙을 고칠 일이 없다.
        (os.path.join('share', package_name, 'models'),
            glob(os.path.join('models', '*.pt')) + glob(os.path.join('models', '*.json'))
            + glob(os.path.join('models', '*.md'))),
    ] + _run_dir_data_files(),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='misys',
    maintainer_email='thdgkgus2001@gmail.com',
    description='DACER++ diffusion policy controller for the F1TENTH race stack',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rl_controller = rl_controller.rl_controller_node:main',
        ],
    },
    scripts=['scripts/check_rl_setup.py'],
)
