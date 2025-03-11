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

$G(z, x(t))$와 $H(z, x(t))$는 제약 조건

다음으로 $F$를 잘 정의해보도록 하자

Nonlinear MPC를 위해서는 아래와 같이  
$$
\min \ell_N (x_N) + \sum_{k=0}^{N-1} \ell_N (x_N)
$$
$\ell_N (x_N)$: terminal cost 

$\ell_N (x_N)$: stage cost






$$
V_N (\{ x_k \}, \{ u_k \}) = F(x_N) + \sum_{k=0}^{N-1} L(x_k, u_k)
$$

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

$$
u_{MPC}(x_0) = u_0^*
$$
