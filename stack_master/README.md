# Stack Master
Here is the `stack_master`, it is intended to be the main interface between the user and the PBL ForzaETH F110 system.

### Mapping (on the real car)
Run the mapping launch file, specifying the map name and the NUCX version:
```shell
ros2 launch stack_master mapping_launch.xml racecar_version:=<NUCX used> map_name:=<map name of choice>
```
  - `<map name of choice>` can be any name with no white space. Conventionally we use the location name (eg, 'hangar', 'ETZ', 'icra') followed by the day of the month followed by an incremental version number. For instance, `hangar_12_v0`.
  - `<NUCX>` depends on which car you are using. Parameters are available for NUC2, NUC5, NUC6, SIM (the latter represents a dummy car).

After completing a lap, a GUI will popup and pressing the requested button will start the global raceline generation. 
Then two GUIs will be shown, and within them a slider can be used to select the sectors. 
Be careful as once a sector is chosen it cannot be further subdivided. 

A ROS resourcing will be needed from here on. 

### Base System
```shell
ros2 launch stack_master base_system_launch.xml map_name:=<name of mapped track> sim:=<true/fasle> racecar_version:=<NUCX used>
```
  - `<name of mapped track>` is the name of the track you want to run on. It must belong to the list of maps available in the `stack_master/maps` folder.
  - `<true/false>` is a boolean value that indicates if you want to run the simulation or the real car. 
  - `<NUCX>` depends on which car you are using. Parameters are available for NUC2, NUC5, NUC6, SIM (the latter represents a dummy car).

### Time trials 
```shell
ros2 launch stack_master time_trials_launch.xml racecar_version:=<NUCx used> LU_table:=<Look-Up Table name> ctrl_algo:=<control algorithm> 
```
  - `<NUCx>` depends on which car you are using. Parameters are available for NUC2, NUC5, NUC6, SIM (the latter represents a dummy car).
  - `<Look-Up Table name>` is the name of the Look-Up Table you want to use. It must belong to the list of Look-Up Tables available in the `systm_identification/steering_lookup/cfg` folder.
  - `<control algorithm>` is the control algorithm you want to use. Current possibilities are MAP / PP.

### Head to Head
```shell
ros2 launch stack_master head_to_head_launch.xml racecar_version:=<NUCx used> LU_table:=<Look-Up Table name> ctrl_algo:=<control algorithm> overtake_mode:=spliner
```
- `<NUCx>` depends on which car you are using. Parameters are available for NUC2, NUC5, NUC6, SIM (the latter represents a dummy car).
- `<Look-Up Table name>` is the name of the Look-Up Table you want to use. It must belong to the list of Look-Up Tables available in the `systm_identification/steering_lookup/cfg` folder.
- `<control algorithm>` is the control algorithm you want to use. Current possibilities are MAP / PP.
- `<overtake_mode>` is the mode you want to use for overtaking. `spliner` is the only current possibility.

### 정지 / 재출발 (E-STOP)
조이스틱과 마우스 **둘 중 아무거나** 쓸 수 있고, 같은 래치를 공유한다.
마우스로 세운 차를 조이스틱으로 풀거나 그 반대도 된다.

| 입력 | 정지 | 재출발 |
| --- | --- | --- |
| 마우스 | 우클릭 | 우클릭 (토글) |
| 조이스틱 (DS4) | Circle (idx 2) | Triangle (idx 3) |
| 터미널 | 아래 명령의 `data: true` | 아래 명령의 `data: false` |

```shell
ros2 topic pub --once --keep-alive 1.0 /e_stop std_msgs/msg/Bool "data: true"
```
- 메시지 타입에 **앞 슬래시를 붙이면 안 된다** (`/std_msgs/msg/Bool` X → `std_msgs/msg/Bool` O).
- `--keep-alive` 를 빼면 `--once` 가 0.1 초 만에 죽어서 샘플이 전달 전에 사라지는 일이 있다
  (실측: 기본값으로 6회 중 1회 누락, `--keep-alive 1.0` 은 6회 중 0회). `-t 3` 도 같은 효과.
- QoS 는 따로 줄 필요 없다. `ros2 topic pub` 의 `--qos-durability` 기본값이 이미
  `transient_local` 이라 `/e_stop` 구독과 그대로 맞는다.

- 마우스 노드(`stack_master/mouse_estop`)는 `base_system_*_launch.xml` 이 같이 띄운다.
  끄려면 `mouse_estop:=false`.
- 실제 래치는 `mux_controller` 가 들고 있고 `/e_stop_state` 로 알린다. 걸려 있는 동안
  `/drive` 로 속도 0 과 duty 0 을 계속 내보낸다.
- 입력 백엔드는 `auto`: `/dev/input/event*` 를 읽을 수 있으면 evdev(포커스/X 불필요),
  아니면 pynput(X11 전역 후킹, `DISPLAY` 필요). evdev 를 쓰려면
  `scripts/99-f1tenth-input.rules` 를 `/etc/udev/rules.d/` 에 설치할 것.
- `/e_stop` 은 transient_local 이라, 마우스로 세워 둔 채 head_to_head 를 재기동하면
  새 `mux_controller` 도 정지 상태로 뜬다 (우클릭 한 번으로 출발).
