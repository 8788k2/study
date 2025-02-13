# LLM+ROS TURTLE BOT CONTROL
chat gpt를 통해 터틀봇을 제어하는 파이썬 코드에 대해 공부하는 것을 목표로 한다.
## 1. 정의된 명령어로부터 터틀봇을 제어하는 코드
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
### 1.1 ROS llm_request 토픽을 구독하여 유저 메시지를 받을 준비
```bash
self.subscription = self.create_subscription(
            String,
            'llm_request',
            self.llm_callback,
            10)
```


### 1.2 LLM을 부르고 유저 메시지를 미리 정의된 명령어로 변환
```bash
def llm_callback(self, msg):
        """LLM을 호출하고, ChatGPT의 응답을 로봇 명령으로 변환"""
        user_input = msg.data
        self.get_logger().info(f"📩 User Command Received: {user_input}")
``` 
```bash
try:
            client = openai.OpenAI()  # 최신 API에서는 인스턴스 생성 필요
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": f"다음 명령을 로봇 이동 명령으로 변환하세요: {user_input}. 예제: '앞으로 가' → 'Move forward'"}]
            )
```

### 1.3 chat gpt 응답 저장
```bash
chat_response = response.choices[0].message.content
            self.get_logger().info(f"🤖 LLM Response: {chat_response}")
```
### 1.4 chat gpt 응답으로부터 미리 정의된 명령어를 고르고 ros에 전달
```bash
# 결과를 `cmd_vel` 토픽으로 퍼블리시
            self.publish_cmd_vel(chat_response)
```
``` bash
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
```
publish_cmd_vel()함수가 cmd_vel메시지를 생성하고 self.vel_publisher.publish(twist)로 ros 2의 cmd_vel 토픽에 퍼블리시한다.
### 1.5 한계

정의된 몇 가지 명령어가 존재하고 LLM(chat gpt)는 유저가 입력한 자연어를 명령어로 변환하는데만 제한적으로 사용되기 때문에 LLM이 가진 포텐셜을 활용한다고 보기 힘들다.

## 2. LLM이 cmd_vel을 직접 생성하도록 조정된 코드
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

기존의 코드와의 차이점을 아래에 설명하겠다.
## 2.1 chat gpt에게 주는 명령어
```bash
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
```
1.에서 소개한 코드에서는 gpt에게 주는 명령이 아래와 같았다.
```
"다음 명령을 로봇 이동 명령으로 변환하세요: {user_input}. 예제: '앞으로 가' → 'Move forward'"
```
2.에서 수정된 코드에서는 아래와 같다.
```
"Convert the following command into a sequence of ROS 2 Twist messages:'{user_input}'. Output in JSON format as a list of movement commands.
Example format:
[{{"linear": {{"x": 0.2, "y": 0.0, "z": 0.0}}, "angular": {{"x": 0.0, "y": 0.0, "z": 0.0}}, "duration": 5}},
                        {{"linear": {{"x": 0.0, "y": 0.0, "z": 0.0}}, "angular": {{"x": 0.0, "y": 0.0, "z": 0.5}}, "duration": 2}}]"
                    
```
수정전 코드에서는 gpt가 user_input을 받아서 정의된 명령어로 변환하는 역할만 수행한 반면, 

**수정후 코드에서는 자연어인 user_input을 gpt가 이해하고 json형식의 cmd_vel을 출력하도록 요청하고 있다.**

## 2.2 gpt의 출력을 어떻게 처리?
```bash
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
```