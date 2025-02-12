# SLAM And Navigation By Integrating ROS And LLM
사용자의 자연어 명령을 LLM(Chat GPT)가 이해하고 분석하여 로봇명령으로 변환, 최종적으로 turtlebo3를 제어 하는 것이 목표이다.
기본 구조는 다음과 같이 설명할 수 있다.
```
사용자 입력 → ChatGPT (LLM) → ROS 2 노드 → TurtleBot3 제어 (`cmd_vel`)
```

## 1. 환경설정
ubuntu 22.04...
vscode...
pip...
자주 사용하는 코드 구조...
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

## 3. ROS와 Chat GPT 연동
### 3.1 Open API Key 설정
open ai 사이트에서 키를 발급받은 뒤 사용할 수 있다.
```bash
echo 'export OPENAI_API_KEY="your_api_key_here"' >> ~/.bashrc
source ~/.bashrc
```
### 3.2 ROS 2 워크스페이스
ROS 2의 기본 패키지는(ros-humble-desktop, colcon, gazebo_ros, nav2 등) /opt/ros/humble/에 설치되지만, 

사용자가 직접 만든 패키지 기본 패키지와 충돌 등의 문제를 일으킬 수 있기 때문에 따로 관리되어야 한다. 

따라서 패키지를 관리할 수 있는 경로를 /ros2_ws로 만들어 주자.

### 3.2.1 관련코드들
디렉토리 생성 및 이동
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

패키지 설치
```bash
ros2 pkg create llm_ros_bridge --build-type ament_python --dependencies rclpy std_msgs geometry_msgs
