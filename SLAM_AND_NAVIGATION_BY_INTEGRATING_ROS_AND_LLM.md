# SLAM And Navigation By Integrating ROS And LLM
## 1. 환경설정
ubuntu 22.04...
## 2. 터틀봇 시뮬레이션
### 2.1 GAZEBO 실행
```bash 
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```
### 2.1.1 turtlebot 키보드 조작
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```
"i" "," 키로 전진과 후진

"j" "l" 키로 반시게방향, 시계방향 회전

"k" 키로 정지

"w" "x" 키로 전후진 속도 제어

"e" "c" 키로 회전 속도 제어

### 2.2 SLAM(cattograper) 실행
```bash
ros2 launch turtlebot3_cartographer cartographer.launch.py use_sim_time:=True
```
