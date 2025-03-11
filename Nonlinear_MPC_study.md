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

아래와 같은 입력과 상태에 관한 제약 조건과 함께 정의된 MPC 문제가 주어질 수 있다.
$$
h(x_k, u_k) \leq 0
$$

Nonlinear MPC를 위해서는 아래와 같이  성능지표를 
$$
\min \ell_N (x_N) + \sum_{k=0}^{N-1} \ell(x_k, u_k)
$$

$$
\min_{z} F(z, x(t))
$$
$$
\text{s.t.} \quad G(z, x(t)) \leq 0, \quad H(z, x(t)) = 0
$$

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
