# NMPC Matlab code practice

## background
Q, R, P, S는 각각 코스트 펑션에서 가중치를 의미

아래의 코드 조각에서 함수 y를 정의, f(x)는 어떤 입력을 받는지 명시하면서 이름을 지어주는 것

```matlab
function y = f(x)  
y = x^2
```

아래의 코드 조각은 y의 사이즈를 정하고 각 행을 어떻게 정의할지 명시
```matlab
function y = f(x) 
y= size(t, 2) % y의 사이즈 결정
for i = (1:t)
y(i,:)= [a_i(x), b_i(x)]   % y의 i번째 행이 [a(i), b(i)]
```

아래의 코드는 변수 x를 어떤 행렬 y의 i번째 행의 모든 열을 가져온다는 것을 명시
```matlab
x = y(i, :)
```

## 모바일로봇의 리더팔로우 문제 mpc 코드




```matlab
clc; clear; close

% Leader의 목표 제어 입력 생성, 원하는 reference 입력(v_goal, w_goal)을 생성, t_start 시점에서부터, t_count번 만큼 t_delta만큼의 변화량을 가지고 t를 샘플링해서 Delta(t) 값을 활용하여 입력을 생성 (총 t_count 개의 입력이 생김) 
function control_input_goal = generate_goal_trajectory(t_start, t_count, t_delta)
    Delta = @(t) 0.79 - (sin((2 * pi *(t +2)) / 3 + 1) + cos((4 * pi * (t+2)) / 3 + 1)) ./ sqrt(4 * t +12); 
    control_input_goal = zeros(t_count, 2); % control_input_goal(t_start, t_count, t_delta)는 t_count * 2 행렬
    for idx = 1:t_count % for idx(for 반복문 시작 idx는 인덱스로 반복에 사용할 변수)
        t = t_start + (idx -1) * t_delta;
        v_goal = 0 - Delta(t);
        w_goal = 1 - Delta(t);
        control_input_goal(idx,:) = [v_goal, w_goal]; % control_input_goal의 idx번째 행에, 현재 시점의 입력 [v_goal, w_goal]을 저장
    
    end
end
```

```matlab
% mobile_robot_dynamics를 통해 dq를 정의 u는 [v; w] 형태의 2×1 열 벡터, q는 [x; y; θ] 형태의 3×1 열 벡터
function dq = mobile_robot_dynamics(q, u)
    theta = q(3);
    S = [cos(theta), 0;
         sin(theta), 0;
         0, 1];
    dq = S * u;
end
```

```matlab
% 4차 룬지쿠타
function q_next = runge_kutta(q, u, dt)
    epsilon = 0;
    k1 = mobile_robot_dynamics(q, u) + epsilon;
    k2 = mobile_robot_dynamics(q + dt/2 * k1, u) + epsilon;
    k3 = mobile_robot_dynamics(q + dt/2 * k2, u) + epsilon;
    k4 = mobile_robot_dynamics(q + dt * k3, u ) + epsilon;
    q_next = q + dt/6 * (k1 + 2*k2 + 2*k3 + k4);
end
``` 


현재 방향 $\theta$ = $\pi$ - 0.01, 목표 방향 $\theta$ = - $\pi$ + 0.01 거의 같은 좌표임에도 수치적 차이 발생 오류 방지하기 위해 보정 필요 
```matlab
% 각도 정규화 함수 (어떤 theta에 대해서도 - pi ~ pi 범위를 갖도록 조정, 같은 각도에 대해 모바일 로봇이 다른 각도라고 인식하지 않도록) 
function theta_norm = normalize_angle(theta) 
    theta_norm = mod(theta + pi, 2 * pi) - pi; % (2 * pi 단위로 끊기)
end
```


uF_opt를 찾는 함수를 정의하는 하나의 코드 덩어리
function uF_opt = nmpc_path_tracking(qL, qF, uL_traj, k, Q, R, P, S, horizion, noise_std) -> 여러인자들 입력으로 받아 uF_opt를 출력으로 내는 함수를 선언


```matlab
% NMPC 최적 제어 입력 계산, 최적화 솔버로 U생성하고 정해진 U에 대해 계산된 코스트 펑션 값 최적화해 나가는 부분에 해당
function uF_opt = nmpc_path_tracking(qL, qF, uL_traj, k, Q, R, P, S, horizon, noise_std) 
    dt = 0.1; 
    % 초기 최적화 변수 설정 (2*horizon x 1 벡터), 논리니어 MPC 풀려면 U찾을 때 초기값을 정해줘야 한다, 0으로 시작
    U_init =  zeros(2 * horizon, 1);

    % 비용 함수 정의 (룬지 쿠타 통합 방식 반영)
    cost_function = @(U) compute_cost(U, qL, qF, uL_traj, k, dt, horizon, Q, R, P, S); % U_opt 계산하는 fmincon 함수는 입력으로 U에 대한 함수만 사용 가능, cost function 값을 넘겨줘야하므로 후에 정의할 compute_cost(U, qL, qF, uL_traj, k, dt, horizon, Q, R, P, S)값을 최적화의 대상인 U에 대한 익명 함수 @(U)의 값으로써 사용

    % 제어 입력 범위 설정 (constraint)
    lb = repmat([-2; -2], horizon, 1);
    ub = repmat([2; 2], horizon, 1);

    options = optimoptions('fmincon', 'Algorithm', 'sqp', 'Display', 'none'); % matlab의 비선형 최적화 함수인 fmincon에 사용할 옵션을 결정 (sqp 알고리즘 사용)
    U_opt = fmincon(cost_function, U_init, [], [], [], [], lb, ub, [], options);
    uF_opt = U_opt(1:2); % 첫 번째 제어 입력 반환 (u가 2 *1 벡터이므로)

    % uF에 정규분포 난수 추가 (평균 0, 표준편차 noise_std)
    uF_opt = uF_opt + noise_std * randn(2, 1);
end
```


```matlab
% 비용 함수, k시점에서의 어그먼티드 방정식에서의 u틸다, 초기값 x(k)에 의해 코스트펑션 값이 결정되는 부분에 해당  
function cost = compute_cost(U, qL, qF, uL_traj, k, dt, horizon, Q, R, P, S)
    U = reshape(U, 2, horizon); % 최적제어 입력 파트에서 U_init에서 시작하여 U를 생성하면 그 정해진 U에 대해 2 * horizon 행렬로 변환, 후에 uF_k 정할 때 i번째 입력을 쉽게 가져오기 위함
    cost = 0;

    % k시점부터 (k+ horizon -1)시점까지 그에 해당하는 u값과 초기상태를 이용해서 horizon*1 차원의 e, uF_k, delta_u 생성
    for i = 1:horizon
        % Leader의 미래 입력 선택
        if k + i - 1 <= size(uL_traj, 1) % k 시점에서부터 예측한 horizon 범위가 처음에 정해진 리더 입력 범위를 초과하는지 확인 
            uL_k = uL_traj(k + i - 1, :);  % uL_traj는 후에 control_input_goal(generate_goal_trajectory(t_start, t_count, t_delta))로 정의되므로 tcount*2 행렬, 거기서 k + i - 1번째 행을 가져오면 1*2 벡터
        else
            uL_k = uL_traj(end, :);
        end

        % follower의 미래 입력
        uF_k = U(:, i); % reshape한 U에서 i번째 열을 가져오면 2 *1 벡터

        % 룬지쿠타 4차 방법을 이용한 상태 업데이트, 리더의 경우 정해진 uL_k, 팔로워의 경우 최적화를 통해 얻어진 uF_K와 초기값로부터 다음 스테이트를 추출하는 과정
        qL = runge_kutta(qL, uL_k', dt); % qL 노테이션을 바꾸지 않고 그대로 엎어써서 최신 상태로 유지
        qF = runge_kutta(qF, uF_k, dt); 

        % 각도 정규화
        qL(3) = normalize_angle(qL(3));
        qF(3) = normalize_angle(qF(3));

        % 에러 상태 계산
        e = qF -qL;
        e(3) = normalize_angle(e(3))

        % 비용 함수 업데이트
        cost = cost + e' * Q * e + uF_k' * R * uF_k; 
        if i > 1 % i가 1보다 큰 시점부터 u 변화량 고려
            delta_u = U(:, i) - U(:, i -1);
            cost = cost + delta_u' * S * delta_u; % 각 예측 시점마다 상태 오차 e와 제어 입력 uF_k에 대해 비용을 누적
        end
    end
    cost = cost + e' * P * e; % 반복문이 끝나는 시점에서의 terminal cost에 가중치 P 곱해서 코스트 펑션에 더하기
end         
```

```matlab
% 시뮬레이션 실행, MPC에서 최적화 문제 풀고 즉, U틸다 구하고 그 중에서 첫번째 입력 뽑아 상태업데이트하고 반복하여 원하는 trajactory 그려나가도록 제어하는 부분에 해당 
dt = 0.1;
T = 30.1;
timesteps = T / dt;

% Leader 초기 상태 재설정
qL = [0; 0; 0];

% Follower 초기 상태 (난수 적용)
qF = [rand() * 8 - 4; % x: -4 ~ 4
      rand() * 8 - 4; % y: -4 ~ 4
      (rand() * (2 * pi / 3)) - (pi / 3)]; % theta: -pi/3 ~ pi/3

% 랜덤화된 MPC 파라미터 생성
horizon = randi([10, 15]); % 예측 지평 랜덤 (10 ~ 15)
P = diag([5*(1+0.3*(rand()-0.5)), 5*(1+0.3*(rand()-0.5)), 5*(1+0.3*(rand()-0.5))]);
Q = diag([10*(1+0.3*(rand()-0.5)), 10*(1+0.3*(rand()-0.5)), 10*(1+0.3*(rand()-0.5))]);
% R = diag([1*(1+0.2*(rand()-0.5)), 2*(1+0.2*(rand()-0.5))]);
% S = diag([1*(1+0.2*(rand()-0.5)), 3*(1+0.2*(rand()-0.5))]);
R = diag([2*(1+0.2*(rand()-0.5)), 2*(1+0.2*(rand()-0.5))]);
S = diag([2*(1+0.2*(rand()-0.5)), 2*(1+0.2*(rand()-0.5))]);

noise_std = 0.01 + 0.05 * rand();

% Leader 제어 입력 생성
uL_traj = generate_goal_trajectory(0, timesteps, dt);

% 결과 저장용 변수 초기화
leader_trajectory = zeros(timesteps, 3);
follower_trajectory = zeros(timesteps, 3);
error_trajectory = zeros(timesteps, 3);
mpc_input_trajectory = zeros(timesteps, 2);

for t = 1:timesteps 
    uL = uL_traj(t, :); % 첫 코드덩어리에서 정의된 generate_goal_trajectory(0, timesteps, dt)행렬 계산하고 t번째 값 가져오기  
    % NMPC 최적 제어 입력 계산 (랜덤화된 MPC 파라미터와 노이즈 포함)
    uF = nmpc_path_tracking(qL, qF, uL_traj, t, Q, R, P, S, horizon, noise_std); % uF_opt 값 가져와서 uF로 정의

    % Runge-Kutta 통합을 통한 상태 업데이트
    qL = runge_kutta(qL, uL', dt);
    qF = runge_kutta(qF, uF, dt);

    qL(3) = normalize_angle(qL(3));
    qF(3) = normalize_angle(qF(3));
    
    % 에러 상태 계산 (Follower - Leader)
    e = qF - qL;
    e(3) = normalize_angle(e(3));
    
    % 플롯을 위한 변수들 
    leader_trajectory(t, :) = qL';
    follower_trajectory(t, :) = qF';
    error_trajectory(t, :) = e';
    mpc_input_trajectory(t, :) = uF';
end
```
```matlab
if true
    figure;
    subplot(3, 1, 1);
    plot(leader_trajectory(:, 1), leader_trajectory(:, 2), 'b--', 'LineWidth', 1.5);
    hold on;
    plot(follower_trajectory(:, 1), follower_trajectory(:, 2), 'r-', 'LineWidth', 1.5);
    xlabel("X Position");
    ylabel("Y Position");
    legend("Leader Path", "Follower Path");
    title("Path Tracking using Nonlinear MPC (Runge-Kutta)");
    grid on;
    
    subplot(3, 1, 2);
    plot(error_trajectory(:, 1), 'LineWidth', 1.5);
    hold on;
    plot(error_trajectory(:, 2), 'LineWidth', 1.5);
    plot(error_trajectory(:, 3), 'LineWidth', 1.5);
    xlabel("Time step");
    ylabel("Error");
    legend("e_x", "e_y", "e_\theta");
    grid on;
    
    subplot(3, 1, 3);
    plot(mpc_input_trajectory(:, 1), 'LineWidth', 1.5);
    hold on;
    plot(mpc_input_trajectory(:, 2), 'LineWidth', 1.5);
    xlabel("Time step");
    ylabel("MPC input");
    legend("v_{mpc}", "w_{mpc}");
    grid on;
end
```

