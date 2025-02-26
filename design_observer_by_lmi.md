# LMI 
자동제어특강에서 배운 LMI 개념을 정리하고 이를 활용하여 inverted pendulum의 옵저버를 설계하는 것을 목적으로 한다.

**LMI를 통한 제어기, 혹은 옵저버 설계의 핵심은 다음과 같다.**

**주어진 시스템(제어방정식, 옵저버의 경우 오차방정식)으로부터 유도된 비선형 부등식을 LMI가 되게 만들면서도 구하길 원하는 gain matrix, 즉 옵저버의 l, 제어기의 k 등을 포함하는 변수($Y$)를 새롭게 잘 정의하고 LMI solver를 통해 구하는 것!**

**Y의 차원 설정에 신경써줘야 한다.**


## 1. LMI(Linear Matrix Inequality)
$$
F(x) := F_0 + \sum_i F_i x_i > 0
$$
where
$F_i = F_i^T$

변수 $x$(스칼라가 아니어도 됨)를 포함한 행렬 $F(x)$가 $x$에 대해 관계 없는 고정된 행렬$F_0$와 $x$와 곱해지는 행렬$F_i$의 합으로 표현될 수 있을 때,

**즉 $F(x)$가 linear matrix일 때, $F(x)$가 위 부등식을 만족하면, $x$공간은 convex하다.**

---
### 리니어 표현 예시 ($x$가 스칼라일 때)

$$
F(x) = \begin{bmatrix} 2 & x \\ 3 & 1 \end{bmatrix}
$$

$$
F(x) = \begin{bmatrix} 2 & 0 \\ 3 & 1 \end{bmatrix} + x \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix} > 0
$$


여러개의 다중 lmi를 하나의 lmi로 바꾸어 표현할 수도 있다. 블록 대각행렬로 묶어 새로은 $F(x)$를 정의하면 된다.

$$
F^{(1)}(x) > 0, \quad F^{(2)}(x) > 0, \quad \dots, \quad F^{(p)}(x) > 0
$$

$$
\begin{bmatrix}
F^{(1)}(x) & 0 & \cdots & 0 \\
0 & F^{(2)}(x) & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & F^{(p)}(x)
\end{bmatrix} > 0
$$

$F(x)$가 리니어하지 않을 수도 있는데, 이 경우엔 추가적인 변형을 가해 lmi꼴로 만들어줘야한다. 하지만 여러 제약조건이 따르고 불가능할 수도 있다.

## 2. Lyapunov stability
시스템이 $\dot{X} = f(X)$, $f: D \to \mathbb{R}^n$와 같이 주어졌을 때, 
$$
V(0) = 0
$$

$$
V(X) > 0, \quad \text{for } X \neq 0
$$

$$
\dot{V}(X) < 0, \quad \text{for } X \neq 0
$$
위의 식을 만족하는 $V(X)$, $V: D \to \mathbb{R}$ (상태벡터를 받아 스칼라로 변환하는 함수)가 존재하면,

**그 시스템은 asymptotically stable하다.** 

---
### Lyapunov와 LMI의 관계
어떤 시스템의 상태벡터를 $x$라고 하고, lyapunov 함수를 다음과 같이 정의하자.
$$
V(x) = x^T P x
$$
이때 $P$는 positive definite 행렬이다.
$$
P = P^T > 0
$$
미분을 적용하면 다음과 같다. (행렬식을 시그마로 표현하고 chain rule 적용하면 증명 가능, $x^T$와 $x$의 원소 따로 미분해주기 위해)
$$
\dot{V}(x) = \frac{d}{dt} (x^T P x)= \left( \frac{d}{dt} x^T \right) P x + x^T P \left( \frac{d}{dt} x \right)
$$
---
$\dot{x} = A x$로 정의된 시스템에서는 아래와 같이 쓸수 있다. (옵저버에서는 오차방정식을 대입할까?)

$$
\dot{V}(x) = (A x)^T P x + x^T P (A x)
$$
이는 행렬의 성질에 따라 다시 다음과 같이 쓸 수 있다.

$$
\dot{V}(x) = x^T A^T P x + x^T P A x = x^T (A^T P + P A) x
$$
$V(x)$가 리아프노프 함수라면 다음을 만족해야 한다.
$$
\dot{V}(x) = x^T (A^T P + P A) x < 0
$$
그러면 행렬의 부등식 정의에 의해 다음과 같은 P를 변수로 갖는 LMI가 유도된다.
$$
A^T P + P A < 0
$$
**이제 저 LMI를 풀어내는 것, 즉 p를 찾는 것이 $\dot{x} = A x$ 시스템의 안정성을 확인할 수 있는 문제와 동치가 된다!**


## 3. LMI를 이용한 제어기 설계
이번에는 $\dot{x}(t) = A x(t) + B u(t)$ 시스템의 제어기 설계를 LMI를 통해 해보자.
$$
u(t) = K x(t)
$$
$$
\dot{x}(t) = A x(t) + B K x(t)
$$
다음과 같은 리아프노브 함수의 미분에 
$$
\dot{V}(x(t)) = \dot{x}^T (t) P x(t) + x^T (t) P \dot{x}(t)
$$
제어방정식의 $\dot{x}(t)$를 대입하면 다음과 같이 유도된다.
$$
\dot{V}(x(t)) = (A x(t) + B K x(t))^T P x(t) + x^T (t) P (A x(t) + B K x(t))
$$
$$
= x^T (t) (A^T P + P A + K^T B^T P + P B K) x(t)
$$
윗 값이 0보다 작아야 하므로 행렬의 부등식 정의에 의해 아래의 행렬 부등식을 얻을수 있다.
$$
A^T P + P A + K^T B^T P + P B K < 0
$$

하지만 위 부등식은 LMI꼴이 아닌 bilinear하기 때문에 앞뒤로 $P^{-1}$을 곱해준다. (부등식은 여전히 성립한다.)
$$
P^{-1} A^T + A P^{-1} + P^{-1} K^T B^T + B K P^{-1} < 0
$$
여기서 위 부등식을 LMI로 만들어주면서 k를 포함하도록 새 변수 $X$, $Y$를 다음과 같이 정의하고 위 부등식에 대입한다. 
$$
P^{-1} = X, \quad K = YX^{-1}
$$
그러면 아래의 LMI를 만족하는 $X$와 $Y$를 LIM solver로 찾아 k값을 구하면 제어기 설계 완료!
$$
\quad X A^T + A X + Y^T B^T + B Y < 0
$$


## LMI를 이용한 inverted pendulum의 옵저버 설계 
다음과 같은 시스템이 있고 루엔버거 옵저버를 설계한다고 하자.
$$
\dot{x} = A x + B u, \quad y = C x
$$
$$
\dot{\hat{x}} = A \hat{x} + B u + L(y - C \hat{x})
$$
그러면 우리가 안정적으로 만들어야 하는 시스템은 오차방정식이다. 

$x$와 $\hat{x}$의 오차를 asymptotically stable하게 즉, 궁극적으로 0으로 만들 수 있기 때문이다.

따라서 오차 방정식을 주어진 시스템으로 보고 LMI를 유도해보자.


$$
\dot{e} = (A - L C) e
$$
아래와 같이 리아프노프 함수를 정의한다. 상태벡터가 $x$가 아닌 $e$로 바뀌었다.
$$
V(e) = e^T P e, \quad P = P^T > 0
$$
그리고 다음과 같이 얻어지는 리아프노브 함수의 미분에 
$$
\dot{V}(e(t)) = \dot{e}^T (t) P e(t) + e^T (t) P \dot{e}(t)
$$
$\dot{e}$를 대입하면 다음과 같은 $\dot{V}(e)를 얻을 수 있고 있고, 이는 0보다 작아야 한다.

$$
\dot{V}(e) = e^T ((A - L C)^T P + P (A - L C)) e < 0
$$
행렬 부등식의 정의에 의해 다음과 같은 행렬 부등식이 유도된다.
$$
(A - L C)^T P + P (A - L C) < 0
$$
앞뒤로 $p^{-1}$을 곱한다.
$$
p^{-1}(A - L C)^T+ (A - L C)p^{-1} = P^{-1} A^T - P^{-1}C^T L^T + AP^{-1} - L C P^{-1} < 0
$$


**여기서 위 부등식을 LMI 형태로 바꿔주면서도 옵저버의 L gain을 포함하도록 새 변수 X와 Y를 도입한다.**
$$
P^{-1} = X, L = Y X^{-1}
$$
이를 대입하면
$$
X A^T - X C^T (Y X^{-1})^T + A X - (Y X^{-1}) C X < 0
$$

**최종적으로 정리하면 다음과 같은 LMI를 얻을 수 있다. 이를 LMI solver로 풀어 L gain을 구하면 제어기 설계 완료!**
$$
X A^T + A X - Y C^T - C Y^T < 0
$$

---
### 적용 코드
```python
import numpy as np
import scipy.linalg
import scipy.signal
import matplotlib.pyplot as plt

# 시스템 매트릭스 정의
A = np.array([[0, 1, 0, 0],
              [0, 0, -9.8, 0],
              [0, 0, 0, 1],
              [0, 0, 19.6, 0]])

B = np.array([[0], [1], [0], [-1]])

C = np.array([[1, 0, 0, 0]])  # 위치 측정

# 🔹 LQR 게인 계산 (제어기 용)
Q = 10 * np.eye(4)
R = np.array([[1]])  
P = scipy.linalg.solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A  

# 🔹 옵저버 게인 L을 LMI로 계산
n = A.shape[0]
X = cp.Variable((n, n), symmetric=True)
Y = cp.Variable((n, 1))

constraints = [X >> 0]  # X는 양의 정부호 행렬
constraints.append(X @ A.T + A @ X - Y @ C.T - C @ Y.T << 0)  # LMI 제약 조건

# LMI 최적화 문제 정의
prob = cp.Problem(cp.Minimize(0), constraints)
prob.solve()

# 최적화된 값 가져오기
X_opt = X.value
Y_opt = Y.value
L = Y_opt @ np.linalg.inv(X_opt)  # L = Y * X^(-1)

# 룬지-쿠타 4차 (RK4) 적용
def rk4_step(f, x, u, dt, y=None):
    if y is not None:  
        k1 = f(x, u, y)
        k2 = f(x + dt * k1 / 2, u, y)
        k3 = f(x + dt * k2 / 2, u, y)
        k4 = f(x + dt * k3, u, y)
    else:
        k1 = f(x, u)
        k2 = f(x + dt * k1 / 2, u)
        k3 = f(x + dt * k2 / 2, u)
        k4 = f(x + dt * k3, u)

    return x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

# 시스템 동역학 정의 (제어기 포함)
def system_dynamics(x, u):
    return A @ x + B @ u

# 옵저버 동역학 정의
def observer_dynamics(x_hat, u, y):
    return A @ x_hat + B @ u + L @ (y - C @ x_hat)

# 시뮬레이션 설정
dt = 0.01
T = 10
N = int(T / dt)

# 🔹 초기 상태 설정 
x = np.array([[0], [0], [0.5], [0]])  
x_hat = np.array([[0], [0], [0.3], [0]])  # 옵저버 초기값

# 상태 기록용 배열
x_history = []
x_hat_history = []
u_history = []

# 시뮬레이션 루프
for i in range(N):
    u = -K @ x  

    x = rk4_step(system_dynamics, x, u, dt)

    y = C @ x

    x_hat = rk4_step(observer_dynamics, x_hat, u, dt, y=y)

    x_history.append(x.flatten())
    x_hat_history.append(x_hat.flatten())
    u_history.append(u.flatten())

# NumPy 배열 변환
x_history = np.array(x_history)
x_hat_history = np.array(x_hat_history)
u_history = np.array(u_history)

# 🔹 그래프 
plt.figure(figsize=(12, 8))

labels = ["Position x", "Velocity x_dot", "Angle θ", "Angular velocity θ_dot"]
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.plot(np.linspace(0, T, len(x_history)), x_history[:, i], label="True " + labels[i])
    plt.plot(np.linspace(0, T, len(x_hat_history)), x_hat_history[:, i], '--', label="Estimated " + labels[i])
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel(labels[i])
    plt.grid()

    plt.xlim([0, T])  
    plt.ylim([x_history[:, i].min() * 1.5, x_history[:, i].max() * 1.5])  

plt.suptitle("State Estimation using Pole Placement Observer with LQR Control")
plt.tight_layout()
plt.show()
```
```PY
import numpy as np
import scipy.linalg
import scipy.signal
import matplotlib.pyplot as plt
import cvxpy as cp  # LMI 최적화를 위한 라이브러리

# 시스템 매트릭스 정의
A = np.array([[0, 1, 0, 0],
              [0, 0, -9.8, 0],
              [0, 0, 0, 1],
              [0, 0, 19.6, 0]])

B = np.array([[0], [1], [0], [-1]])

C = np.array([[1, 0, 0, 0]])  # 위치 측정

# 🔹 LQR 게인 계산 (제어기 용)
Q = 10 * np.eye(4)
R = np.array([[1]])  
P = scipy.linalg.solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A  

# 🔹 옵저버 게인 L을 LMI로 계산
n = A.shape[0]
X = cp.Variable((n, n), symmetric=True)
Y = cp.Variable((1, 4))

constraints = [X >> 0]  # X는 양의 정부호 행렬
constraints.append(A.T @ X+ X @ A + C.T @ Y  + Y.T @ C << 0)  # LMI

# LMI 최적화 문제 정의
prob = cp.Problem(cp.Minimize(0), constraints)
prob.solve()

# 최적화된 값 가져오기
X_opt = X.value
Y_opt = Y.value
L = np.linalg.pinv(X_opt) @ Y_opt.T

# 룬지-쿠타 4차 (RK4) 적용
def rk4_step(f, x, u, dt, y=None):
    if y is not None:  
        k1 = f(x, u, y)
        k2 = f(x + dt * k1 / 2, u, y)
        k3 = f(x + dt * k2 / 2, u, y)
        k4 = f(x + dt * k3, u, y)
    else:
        k1 = f(x, u)
        k2 = f(x + dt * k1 / 2, u)
        k3 = f(x + dt * k2 / 2, u)
        k4 = f(x + dt * k3, u)

    return x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

# 시스템 동역학 정의 (제어기 포함)
def system_dynamics(x, u):
    return A @ x + B @ u

# 옵저버 동역학 정의
def observer_dynamics(x_hat, u, y):
    return A @ x_hat + B @ u + L @ (y - C @ x_hat)

# 시뮬레이션 설정
dt = 0.01
T = 10
N = int(T / dt)

# 🔹 초기 상태 설정 
x = np.array([[0], [0], [0.5], [0]])  
x_hat = np.array([[0], [0], [0.3], [0]])  # 옵저버 초기값

# 상태 기록용 배열
x_history = []
x_hat_history = []
u_history = []

# 시뮬레이션 루프
for i in range(N):
    u = -K @ x  

    x = rk4_step(system_dynamics, x, u, dt)

    y = C @ x

    x_hat = rk4_step(observer_dynamics, x_hat, u, dt, y=y)

    x_history.append(x.flatten())
    x_hat_history.append(x_hat.flatten())
    u_history.append(u.flatten())

# NumPy 배열 변환
x_history = np.array(x_history)
x_hat_history = np.array(x_hat_history)
u_history = np.array(u_history)

# 🔹 그래프 
plt.figure(figsize=(12, 8))

labels = ["Position x", "Velocity x_dot", "Angle θ", "Angular velocity θ_dot"]
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.plot(np.linspace(0, T, len(x_history)), x_history[:, i], label="True " + labels[i])
    plt.plot(np.linspace(0, T, len(x_hat_history)), x_hat_history[:, i], '--', label="Estimated " + labels[i])
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel(labels[i])
    plt.grid()

    plt.xlim([0, T])  
    plt.ylim([x_history[:, i].min() * 1.5, x_history[:, i].max() * 1.5])  

plt.suptitle("State Estimation using Pole Placement Observer with LQR Control")
plt.tight_layout()
plt.show()
```