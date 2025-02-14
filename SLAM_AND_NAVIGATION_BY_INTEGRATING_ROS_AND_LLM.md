# SLAM And Navigation By Integrating ROS And LLM
사용자의 자연어 명령을 LLM(Chat GPT)가 이해하고 분석하여 로봇명령으로 변환, 최종적으로 turtlebo3를 제어 하는 것이 목표이다.
기본 구조는 다음과 같이 설명할 수 있다.
```
사용자 입력 → ChatGPT (LLM) → ROS 2 노드 → TurtleBot3 제어 (`cmd_vel`)
```

## 생각나는 포인트들

**영어? 한국어? 어떤 언어 특화 대상 누구? 쓰임새**

**~~같은 명령어 계속 호출 -> 비효율적 수정~~ -> 해결완료**

**정의된 명령어가 아니라 llm이 스스로 cmd_vel값을 실시간으로 조작하게 만들 수 있을까?**

**환경과 llm 간 소통은? 장애물 인식 등** 

속도 조절 기능 추가

경로 계획 기능 연동

장애물 회피 추가

추가적인 자연어 이해 개선

---
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
---
### 2.1.1 turtlebot 키보드 조작
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```
"i" "," 키로 전진과 후진

"j" "l" 키로 반시게방향, 시계방향 회전

"k" 키로 정지

"w" "x" 키로 전후진 속도 제어

"e" "c" 키로 회전 속도 제어

---

### 2.2 SLAM(cattograper) 실행
```bash
ros2 launch turtlebot3_cartographer cartographer.launch.py use_sim_time:=True
```
---
## 3. ROS와 Chat GPT 연동
### 3.1 Open API Key 설정
open ai 사이트에서 키를 발급받은 뒤 사용할 수 있다.
```bash
echo 'export OPENAI_API_KEY="your_api_key_here"' >> ~/.bashrc
source ~/.bashrc
```

---
### 3.2 ROS 2 워크스페이스
ROS 2의 기본 패키지는(ros-humble-desktop, colcon, gazebo_ros, nav2 등) /opt/ros/humble/에 설치되지만, 

사용자가 직접 만든 패키지 기본 패키지와 충돌 등의 문제를 일으킬 수 있기 때문에 따로 관리되어야 한다. 

따라서 패키지를 관리할 수 있는 경로를 /ros2_ws로 만들어 주자.

---
3.2.1 관련코드들

디렉토리 생성 및 이동
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

패키지 설치 (워크스페이스 디렉토리에 설치 해야 함)
```bash
ros2 pkg create llm_ros_bridge --build-type ament_python --dependencies rclpy std_msgs geometry_msgs
```
---
### 3.2 LLM과 ROS를 연결
3.2.1 각종 패키지설치

openai 설치
```bash
pip3 install openai
```
colcon 설치 (ROS 2 패키지를 실행할 수 있도록 준비하게 해줌)
```bash
sudo apt update
sudo apt install python3-colcon-common-extensions
```

3.2.1 **chat gpt ROS 2 노드**(.py 파일)

파일 생성
```bash
cd ~/ros2_ws/src/llm_ros_bridge/llm_ros_bridge/
touch llm_ros_node.py
chmod +x llm_ros_node.py
```

코드 내용
```bash
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import openai
import os

class LLMROSBridge(Node):
    def __init__(self):
        super().__init__('llm_ros_bridge')

        # OpenAI API 키 설정
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            self.get_logger().error("❌ OpenAI API Key is missing! Set it with 'export OPENAI_API_KEY=your_key_here'")
            return

        # ROS 2 퍼블리셔 (`cmd_vel`을 통해 로봇 제어)
        self.vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # ROS 2 구독 (`llm_request` 토픽을 구독하여 사용자 명령 받기)
        self.subscription = self.create_subscription(
            String,
            'llm_request',
            self.llm_callback,
            10)

        self.get_logger().info("✅ LLM ROS Bridge Node Initialized (Supports Korean & English Commands)")

    def llm_callback(self, msg):
        """LLM을 호출하고, ChatGPT의 응답을 로봇 명령으로 변환"""
        user_input = msg.data
        self.get_logger().info(f"📩 User Command Received: {user_input}")

        # 최신 OpenAI API 방식 적용
        try:
            client = openai.OpenAI()  # 최신 API에서는 인스턴스 생성 필요
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": f"다음 명령을 로봇 이동 명령으로 변환하세요: {user_input}. 예제: '앞으로 가' → 'Move forward'"}]
            )

            chat_response = response.choices[0].message.content
            self.get_logger().info(f"🤖 LLM Response: {chat_response}")

            # 결과를 `cmd_vel` 토픽으로 퍼블리시
            self.publish_cmd_vel(chat_response)

        except Exception as e:
            self.get_logger().error(f"❌ OpenAI API Request Failed: {e}")

    def publish_cmd_vel(self, command):
        """ChatGPT의 응답을 기반으로 TurtleBot3 이동 명령 생성"""
        twist = Twist()

        if "forward" in command.lower() or "앞으로" in command:
            twist.linear.x = 0.2  # 전진
        elif "backward" in command.lower() or "뒤로" in command:
            twist.linear.x = -0.2  # 후진
        elif "left" in command.lower() or "왼쪽" in command:
            twist.angular.z = 0.5  # 좌회전
        elif "right" in command.lower() or "오른쪽" in command:
            twist.angular.z = -0.5  # 우회전
        else:
            twist.linear.x = 0.0
            twist.angular.z = 0.0  # 정지

        self.vel_publisher.publish(twist)
        self.get_logger().info(f"🚀 Published cmd_vel: linear={twist.linear.x}, angular={twist.angular.z}")

def main(args=None):
    rclpy.init(args=args)
    node = LLMROSBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

3.2.2 노드를 실행하기 위한 준비

setup.py (/ros2_ws/src/llm_ros_bridge/setup.py) 수정
```bash
from setuptools import find_packages, setup

package_name = 'llm_ros_bridge'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'openai'],  # openai 추가
    zip_safe=True,
    maintainer='deok',
    maintainer_email='deok@todo.todo',
    description='LLM-based ROS 2 bridge for TurtleBot3 control',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'llm_ros_node = llm_ros_bridge.llm_ros_node:main',  # 실행 가능한 노드 등록
        ],
    },
)
```
실행권한 부여
```bash
chmod +x ~/ros2_ws/src/llm_ros_bridge/llm_ros_bridge/llm_ros_node.py
```
환경변수 설정
```bash
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```
**colcon build 실행**

**py 코드를 수정할 때 마다 패키지 빌드는 다시 해줘야 한다.**

```bash
cd ~/ros2_ws
colcon build --packages-select llm_ros_bridge
```

노드 실행
```bash
ros2 run llm_ros_bridge llm_ros_node
```

출력의 의미

...

가제보 실행

...

### 3.3 chat gpt에게 명령 입력
```bash
ros2 topic pub /llm_request std_msgs/msg/String "{data: 'move forward'}"

ros2 topic pub /llm_request std_msgs/msg/String "{data: 'stop'}"

ros2 topic pub --once /llm_request std_msgs/msg/String "{data: 'move forward'}"

ros2 topic pub --once /llm_request std_msgs/msg/String "{data: 'stop'}"
```
명령을 계속 퍼블리시하여 토큰을 소모하는 문제 해결

여기까지 설명한 코드에서는 앞으로가, 멈춰 등의 제한적인 제어만 가능하다.

### 4. advanced code
```bash
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import openai
import os
import json
import time  # 시간 지연을 위한 라이브러리

class LLMROSBridge(Node):
    def __init__(self):
        super().__init__('llm_ros_bridge')

        # OpenAI API 클라이언트 생성
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        if not os.getenv("OPENAI_API_KEY"):
            self.get_logger().error("❌ OpenAI API Key is missing! Set it with 'export OPENAI_API_KEY=your_key_here'")
            return

        # ROS 2 퍼블리셔 (`cmd_vel`을 통해 로봇 제어)
        self.vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # ROS 2 구독 (`llm_request` 토픽을 구독하여 사용자 명령 받기)
        self.subscription = self.create_subscription(
            String,
            'llm_request',
            self.llm_callback,
            10)

        self.get_logger().info("✅ LLM ROS Bridge Node Initialized (Direct cmd_vel control enabled)")

    def llm_callback(self, msg):
        """LLM을 호출하여 자연어 명령을 cmd_vel 시퀀스로 변환"""
        user_input = msg.data.strip()
        self.get_logger().info(f"📩 User Command Received: {user_input}")

        try:
            # OpenAI API를 호출하여 cmd_vel 값을 직접 생성
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": f"""
                    Convert the following command into a sequence of ROS 2 Twist messages:
                    '{user_input}'.
                    Output in JSON format as a list of movement commands.
                    Example format:
                    [
                        {{"linear": {{"x": 0.2, "y": 0.0, "z": 0.0}}, "angular": {{"x": 0.0, "y": 0.0, "z": 0.0}}, "duration": 5}},
                        {{"linear": {{"x": 0.0, "y": 0.0, "z": 0.0}}, "angular": {{"x": 0.0, "y": 0.0, "z": 0.5}}, "duration": 2}}
                    ]
                    """
                }]
            )

            movement_sequence = json.loads(response.choices[0].message.content)
            self.execute_movement_sequence(movement_sequence)

        except Exception as e:
            self.get_logger().error(f"❌ OpenAI API Request Failed: {e}")

    def execute_movement_sequence(self, sequence):
        """cmd_vel 값을 받아서 일정 시간 동안 실행"""
        for step in sequence:
            twist = Twist()
            twist.linear.x = step["linear"]["x"]
            twist.linear.y = step["linear"]["y"]
            twist.linear.z = step["linear"]["z"]
            twist.angular.x = step["angular"]["x"]
            twist.angular.y = step["angular"]["y"]
            twist.angular.z = step["angular"]["z"]

            self.vel_publisher.publish(twist)
            self.get_logger().info(f"🚀 Executing cmd_vel: {twist}")

            # 일정 시간 동안 현재 동작 유지
            time.sleep(step["duration"])

        # 모든 동작이 끝난 후 정지
        self.stop_robot()

    def stop_robot(self):
        """모든 동작 후 로봇 정지"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.vel_publisher.publish(twist)
        self.get_logger().info("🛑 Stopping robot")

def main(args=None):
    rclpy.init(args=args)
    node = LLMROSBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```
위 코드를 노드로 이용하면 보다 복잡한 제어가 가능할 것으로 기대된다.
예를 들어 "앞으로 5초동안 가다가 좌회전하고 다시 앞으로 2초 동안 가"와 같은 제어가 가능하다.

**노드를 두 개 만들어서 여러 버전 동시에 관리**
```bash
touch ~/ros2_ws/src/llm_ros_bridge/llm_ros_bridge/llm_ros_node_2.py
```


setup.py 수정 필요!**
```bash
from setuptools import find_packages, setup

package_name = 'llm_ros_bridge'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'openai'],  # openai 추가
    zip_safe=True,
    maintainer='deok',
    maintainer_email='deok@todo.todo',
    description='LLM-based ROS 2 bridge for TurtleBot3 control',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'llm_ros_node = llm_ros_bridge.llm_ros_node:main',  # 기존 실행 가능한 노드
            'llm_ros_node_2 = llm_ros_bridge.llm_ros_node_2:main',  # 새로운 실행 가능한 노드 추가
        ],
    },
)
```
다시 빌드 
```bash
cd ~/ros2_ws
colcon build --packages-select llm_ros_bridge
```

새로운 노드 실행
```bash
ros2 run llm_ros_bridge llm_ros_node_2

ros2 run llm_ros_bridge llm_ros_node_3
```

새로운 명령어
```bash
ros2 topic pub --once /llm_request std_msgs/msg/String "{data: '3초 동안 앞으로 갔다가 90도 좌회전해서 다시 2초 동안 앞으로 가'}"

ros2 topic pub --once /llm_request std_msgs/msg/String "{data: '직경이 2m 정도 되는 원을 반시계 방향으로 두바퀴 그려'}"

ros2 topic pub --once /llm_request std_msgs/msg/String "{data: '시계 방향으로 135도 회전한 뒤 앞으로 구불구불한 곡선을 크게 그리며 15초 동안 가'}"

ros2 run llm_ros_bridge only_natural_language_command_node
```



빈 가제보 월드 실행
```bsah
ros2 launch turtlebot3_gazebo empty_world.launch.py
```
가제보 월드 종료 후에는 아래의 코드로 가제보를 초기화 시켜주면 불필요한 오류를 방지할 수 있다.
```bash
killall -9 gzserver gzclient
```

역동적이고 복잡한 움직임(ex 직경이 3m 정도 되게 반시계 방향으로 원을 두바퀴 그려, 우상단 대각선 방향으로 5초 동안 지그재그로 가 등)을 구현하기 위한 포인트

다양한 명령에 대한 프롬프트 준비
주어진 각도를 라디안각도로 변환하고 회전속도를 바탕으로 회전에 필요한 시간을 정확히 계산해서 출력해라 등

발생할 수 있는 오류에 대한 대응 스크립트 마련

```bash
ros2 topic pub --once /llm_request std_msgs/msg/String "{data: '현재 로봇이 바라보고 있는 방향을 기준으로 좌하단 45도 방향으로 회전한 뒤 지그재그로 곡선을 크게 그리며 7 초 동안 가'}"
```
위 명령은 현재로선 불가능

---


### 5. 자연어 명령 간단하게 입력하기
py 노드 생성
```bash
touch ~/ros2_ws/src/llm_ros_bridge/llm_ros_bridge/only_natural_language_command_node.py
```
py 코드
```bash
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class OnlyNaturalLanguageCommand(Node):  # 클래스 이름도 수정 (선택 사항)
    def __init__(self):
        super().__init__('only_natural_language_command_node')  # ✅ 노드 이름 수정
        self.publisher = self.create_publisher(String, 'llm_request', 10)

    def run(self):
        """터미널에서 자연어 입력을 받아 LLM 요청을 보내는 함수"""
        self.get_logger().info("✅ 자연어 입력을 받는 모드가 시작되었습니다! (종료하려면 Ctrl+C)")
        try:
            while rclpy.ok():
                # 사용자 입력 받기
                user_input = input("💬 명령을 입력하세요: ").strip()
                if not user_input:
                    continue  # 빈 입력 무시

                # 메시지 발행
                msg = String()
                msg.data = user_input
                self.publisher.publish(msg)
                self.get_logger().info(f"📡 Published: {msg.data}")

        except KeyboardInterrupt:
            self.get_logger().info("🛑 종료합니다.")

def main(args=None):
    rclpy.init(args=args)
    node = OnlyNaturalLanguageCommand()  # ✅ 새로운 클래스명 적용
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```
setup.py 수정

entry point에 아래의 코드 추가
```bash
'only_natural_language_command_node = llm_ros_bridge.only_natural_language_command_node:main'
```
빌드 실행

노드 실행
```bash
ros2 run llm_ros_bridge only_natural_language_command_node
```

### 5. 정확한 프롬프트
```bash
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import openai
import os
import json
import time
import re
import threading  # 멀티스레딩 사용

class LLMROSBridge(Node):
    def __init__(self):
        super().__init__('llm_ros_bridge')

        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        if not os.getenv("OPENAI_API_KEY"):
            self.get_logger().error("❌ OpenAI API Key is missing! Set it with 'export OPENAI_API_KEY=your_key_here'")
            return

        self.vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.subscription = self.create_subscription(String, 'llm_request', self.llm_callback, 10)

        # 현재 실행 중인 명령을 추적하는 변수
        self.running_command = False  
        self.stop_event = threading.Event()  # 현재 동작을 중단하는 이벤트

        # ✅ 시스템 프롬프트 설정 (토큰 절약 & 정확한 경로 생성)
        self.system_prompt = """
        너는 ROS에서 동작하는 터틀봇을 제어하는 AI야.
        자연어 명령을 받으면, cmd_vel 메시지의 시퀀스를 **JSON 형식**으로 출력해야 해.
        설명은 필요없고 오직 JSON 형식만 출력하도록 부탁해

        다음 규칙을 따르도록 해:
        1. **각도 회전 변환**
           - 각속도 angular.z와 각도를 기반으로 회전 시간 duration을 정확하게 계산하라.
           - 기본 각속도: ±0.5 rad/s (양수 = 반시계, 음수 = 시계)
           - 예: "90도 좌회전" → {angular.z: 0.5, duration: 1.57} (1.57초 동안 회전)
        
        2. **거리 이동 변환**
           - 선속도 linear.x와 거리를 기반으로 이동 시간을 계산하라.
           - 기본 선속도: 1.0 m/s
           - 예: "3초 동안 앞으로 가" → {linear.x: 1.0, duration: 3.0}
        
        3. **특정한 궤적 (예: 원, 곡선, 하트)**
           - 원을 그릴 때는 반지름과 회전 속도를 고려하여 linear.x와 angular.z를 조합하라.
           - linear.x = 반지름 × angular.z
           - 곡선은 적절한 샘플링을 통해 시퀀스로 분할하라.
           - 하트 같은 복잡한 모양은 궤적 방정식을 따라 계산하라.
        
        4. **자연스럽고 확실한 움직임을 보장**
           - 각 동작이 명확하게 구분될 수 있도록 cmd_vel을 구성하라.
           - 동작의 부드러움을 위해 속도를 적절히 조절하라.
        
        5. **지속되는 동작 처리** 
           -"계속",""지속" 등의 명령어가 포함되면 duration 값을 100 이상으로 줘 
        
        **출력 형식 (예제)**
        입력: "3초 동안 앞으로 갔다가 90도 좌회전해서 다시 2초 동안 앞으로 가"
        출력:
        ```json
        [
          {"linear": {"x": 1.0, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": 0.0}, "duration": 3.0},
          {"linear": {"x": 0.0, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": 0.5}, "duration": 1.57},
          {"linear": {"x": 1.0, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": 0.0}, "duration": 2.0}
        ]
        ```
        """

        self.get_logger().info("✅ LLM ROS Bridge Node Initialized (Immediate command switching enabled)")

    def llm_callback(self, msg):
        """ 새로운 명령을 받을 때 현재 동작을 중단하고 새로운 동작을 실행 """
        user_input = msg.data.strip()
        self.get_logger().info(f"📩 User Command Received: {user_input}")

        # 실행 중인 동작을 중단하도록 이벤트 설정
        if self.running_command:
            self.get_logger().info("🛑 Stopping current movement for new command...")
            self.stop_event.set()  

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_input}
                ]
            )

            raw_response = response.choices[0].message.content.strip()
            self.get_logger().info(f"Raw OpenAI Response: {raw_response}")
            cleaned_response = re.sub(r"```json\n(.*?)\n```", r"\1", raw_response, flags=re.DOTALL).strip()
            self.get_logger().info(f"Cleaned JSON Response: {cleaned_response}")

            movement_sequence = json.loads(cleaned_response)

            if not isinstance(movement_sequence, list):
                raise ValueError("Response is not a valid list of movement commands.")

            for step in movement_sequence:
                if not isinstance(step, dict) or "linear" not in step or "angular" not in step or "duration" not in step:
                    raise ValueError(f"Invalid step format: {step}")

            movement_thread = threading.Thread(target=self.execute_movement_sequence, args=(movement_sequence,))
            movement_thread.start()

        except json.JSONDecodeError as e:
            self.get_logger().error(f"❌ JSON Parsing Error: {e}")
        except ValueError as e:
            self.get_logger().error(f"❌ Invalid movement sequence: {e}")
        except Exception as e:
            self.get_logger().error(f"❌ OpenAI API Request Failed: {e}")

    def execute_movement_sequence(self, sequence):
        """cmd_vel 값을 받아서 일정 시간 동안 실행"""
        self.running_command = True
        self.stop_event.clear()

        for step in sequence:
            if self.stop_event.is_set():
                self.get_logger().info("⚠ Command interrupted by new input.")
                break

            if not isinstance(step, dict) or "linear" not in step or "angular" not in step or "duration" not in step:
                self.get_logger().error(f"❌ Invalid movement step: {step}")
                continue

            twist = Twist()
            twist.linear.x = step["linear"].get("x", 0.0)
            twist.linear.y = step["linear"].get("y", 0.0)
            twist.linear.z = step["linear"].get("z", 0.0)
            twist.angular.x = step["angular"].get("x", 0.0)
            twist.angular.y = step["angular"].get("y", 0.0)
            twist.angular.z = step["angular"].get("z", 0.0)

            self.vel_publisher.publish(twist)
            self.get_logger().info(f"🚀 Executing cmd_vel: {twist}")

            time.sleep(step["duration"])

        self.stop_robot()
        self.running_command = False

    def stop_robot(self):
        twist = Twist()
        self.vel_publisher.publish(twist)
        self.get_logger().info("🛑 Stopping robot")

def main(args=None):
    rclpy.init(args=args)
    node = LLMROSBridge()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```


가제보 상의 물리 조건들 예)마찰 속도 제한 등이 고려되지 않아 정확한 제어가 힘든 문제 발생

믈리조건을 주어주고 보정해야한다..

llm괴 로봇간 소통을 통해 실시간으로 제어 보정 vs 기존 제어기 사용 뭐가 더 이득?

