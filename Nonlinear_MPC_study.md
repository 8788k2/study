# Nonlinear MPC
LG 과제를 수행하기위해 비선형 MPC에 대한 개념을 공부하고 정리하는 것을 목표로 한다.
## 기본 형태
비선형 시스템은 일반적으로 아래와 같이 비선형 이산 상태방정식을로 표현할 수 있다.
$$
\begin{cases}
x_{k+1} = f(x_k, u_k) \\
y_k = g(x_k, u_k)
\end{cases}
$$

아래와 같이 입력과 상태에 관한 제약 조건과 함께 정의된 MPC 문제가 주어질 수 있다.
$$
h(x_k, u_k) \leq 0
$$

미래 $N$개의 시간 스텝 동안 비용을 최소화하는 최적의 제어입력을 찾는 것이 목표이다.

따라서 아래와 같은 최적화 문제를 정의한다.

$$
\min_{z} F(z, x(t))
$$
$$
\text{s.t.} \quad G(z, x(t)) \leq 0, \quad H(z, x(t)) = 0
$$
$z$: 최적화 변수 벡터, 최적화 입력 벡터 $u$와 상태 벡터 $x$를 포함

$G(z, x(t))$, $H(z, x(t))$: 제약 조건


Nonlinear MPC를 위해서는 아래와 같이 $V_N(\{ x_k \}, \{ u_k \})$ 를 cost function을 정의할 수 있다.
$$
V_N (\{ x_k \}, \{ u_k \}) = F(x_N) + \sum_{k=0}^{N-1} L(x_k, u_k)
$$

$F(x_N)$: terminal cost 

$L (x_k,u_k)$: stage cost


참고로 여기서 terminal cost를 따로 정의해준 이유가 있다.

어떤 최적화 문제가 있으면, N스텝 너머에도 최적화 대상이 있을 것이다. 그러면 N을 초기조건으로 갖는 무한합으로 표현되는 cost function이 있을 것이고 해당 값이 수렴한다는 가정하에(애초에 수렴하지 않으면 최적화 불가능한 시스템이다.) $F(x_N)$이라고 써줄 수 있다.

해당 비용함수를 따로 정의하여 포함시키는 이유는 해당시스템의 stability, feasibility 등을 결정하는 중요한 변수가 되기 때문이다.



다시 주제로 돌아와서 비선형 MPC는 초기 상태 $x_0$에서부터 $N-1$ 번쨰 후 스텝까지의 입력 중 제약조건을 만족하면서 $V_N$이 최소가 되게 하는 입력 벡터 $u$를 찾는 문제로 볼 수 있다.

$$
V_N^{opt}(x_0) \triangleq \min V_N (\{ x_k \}, \{ u_k \})
$$

$$
x_{k+1} = f(x_k, u_k), \quad k = 0, \dots, N-1
$$

$$
x_0 = \text{현재 상태 (given)}
$$

$$
h_j(x_k, u_k) \leq 0, \quad j = 1, \dots, n_h
$$
그렇게 해서 구한 $u$의 해 중 첫번쨰 스텝의 입력만 가져와서 시스템의 입력으로 쓰고 다음 스텝부터 다시 MPC 문제를 풀어준다.
$$
u_{MPC}(x_0) = u_0^*
$$


## Nonlinear MPC의 종류
### Global optimizers 
특징
```
stochastic 방식이 포함될 수 있음
robust하고 global한 최적해를 찾을 수 있음
```
대표적 기법
```
Genetic Algorithms
Simulated Annealing
Pattern Search
Swarm Methods ex:PSO
```
계산 속도가 느려 MPC와 같은 실시간 시스템에 적용하기 어려움

### first order methods
특징
```
local 최적해에 수렴
미분정보만 필요
local 선형 수렴성을 가짐
```

대표적 기법
```
Projected Gradient 
Multiplicative updates (M3, PQP 등)
Multiplier 기법 (Augmented Lagrangian)
 -ADMM (Alternating Direction Method of Multipliers)
 -Dual Ascent
```

계산속도가 비교적 빠르고 선형 수렴성 가지므로 MPC 문제 해결에 적합

### Second order methods
특징
```
계산 속도가 빠름
불안정함(최적해에서 멀리 떨어진 경우 수렴이 어려울 수 있음)
Hessian 정보 활용
로컬에서 빠르게 수렴
```

대표적 기법
```
SQP (Sequential Quadratic Programming)
Interior Point Methods 
Complementarity Methods
 -Semi-smooth 
 -Non-interior Homotopy
```

가장 널리쓰이는 NMPC 기법임 

불안정성을 해결하기 위해 1차기법과 결합하여 사용되기도 함

파이썬에서 NMPC를 푸는 라이브러리인 CasADi는 
NMPC 문제를 정의할 수 있게 해주며 SQP와 Interior Point OPTimizer 둘 다 지원

IPOPT등의 내부점 최적화 솔버를 활용하여 MPC 문제를 해결함

## Interior Point Methods
MPC에서 최적화의 대상이 될 입력벡터 $U$와 상태벡터 $X$, 그리고 그에 대한 새로운 벡터 $z$를 아래와 같이 정의하자.
$$
U =
\begin{bmatrix} 
    u_0 \\ 
    \vdots \\ 
    u_{N-1} 
\end{bmatrix}, 
\quad
X =
\begin{bmatrix} 
    x_1 \\ 
    \vdots \\ 
    x_N 
\end{bmatrix}, 
\quad
z =
\begin{bmatrix} 
    X \\ 
    U 
\end{bmatrix}
$$
그러면 NMPC 문제를 비선형 최적화 문제로 바꾸어 아래와 같이 간단하게 표현할 수 있다.
$$
\min \Phi(z)
$$



비선형 상태방정식의 정의로부터 equation of motion을 제약조건으로서 유도할 수 있다.
$$
G(z) = 0 \quad \Rightarrow \quad x_{k+1} - f(x_k, u_k) = 0, \quad k = 0, \dots, N-1
$$

상태와 입력이 만족해야할 부등식 형태의 제약 조건도 아래와 같이 설정할 수 있다.
$$
H(z) \leq 0
$$

여기서 Interior Point Methods(IPM)은 아래와 같이 Slack 변수 $s$를 도입하고, 로그 장볍 함수를 추가하여 문제를 변형시킨다.
$$
\min \Phi(z) - \sigma \sum_{i=1}^{n_H} \log(s_i)
$$

$$
\mathcal{L}(z, s, \lambda, \mu) = \Phi(z) - \sigma \sum_{i=1}^{n_H} \log(s_i) + \lambda^T G(z) + \mu^T (H(z) + s)
$$

$$
\nabla_z \mathcal{L} = \nabla_z \Phi(z) + (\nabla_z G(z))^T \lambda + (\nabla_z H(z))^T \mu = 0
$$

$$
\nabla_s \mathcal{L} = -\sigma S^{-1} e + \mu = 0 \Rightarrow -\sigma e + S \mu = 0
$$
