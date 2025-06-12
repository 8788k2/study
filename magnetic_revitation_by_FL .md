# Magnetic Levitation by Feedback Linearization
비선형제어시스템의 과제인 마그네틱 레비테이션을 피드백 선형화를 통해 제어기를 설계하여 구동하는 것을 목표로한다.

2013 레퍼 참고


슬라이딩 모드 컨트롤과 피드백 션형화 차이

둘다 적용가능한 시스템 식은 비슷하나 슬라이딩모드는 어떤 계수의 바운디드된 범위와 부호만 알면 적용가능, 피드백 선형화는 정확히 시스템 알아야 함

다시말해 잘 알려진 시스템(마그네틱 레비테이션 같은)에선 FL이 유리할 수 있음

어떤 Uncertainty가 있는 모델의 바운디드 범위와 부호를 학습을 통해서 추론하고 슬라이딩 모드 컨트롤 활용해서 제어기 설게는 어때?? -> 추후 연구 주제로도 나쁘지 않을듯


## 피드백 리니어라이제이션 유도 및 설명
마그레브 시스템에서 볼의 동역학은 아래와 같이 쓸 수 있다.
$$
\ddot{x} = g - \frac{k}{m} \cdot \frac{i^2}{x^2}
\quad \Rightarrow \quad
v(t) = g - \frac{k}{m} \cdot \frac{i^2}{x^2}
$$

볼의 가속도를 새로운 입력 신호로 보고  전류 i와의 관계는 다시 아래와 같이 쓸 수 있다.
$$
\frac{k}{m} \cdot \frac{i^2}{x^2} = g - v(t)
\quad \Rightarrow \quad
i^2 = \frac{m}{k} \cdot (g - v(t)) \cdot x^2
\quad \Rightarrow \quad
i(t) = \sqrt{ \frac{m}{k} \cdot (g - v(t)) \cdot x^2 }
$$



## PD 컨트롤러 개념
에러와 에러의 미분으로 모델의 가속도를 결정할 수 있을 것이다. (이런 가속도로 볼이 움직여야 에러를 최소화한다.)

PD 컨트롤러의 역할은 에러에 대한 가속도의 미분 방정식의 계수를 찾는다고 할 수 있다.
$$
v(t) = K_p (x_{\text{ref}} - x) + K_d (\dot{x}_{\text{ref}} - \dot{x})
$$

$$
\ddot{x}_{\text{desired}} = v(t) = K_p e + K_d \dot{e}
$$


## 전체 시뮬레이션 흐름
```matlab
clc; clear; close all;

%% 시스템 파라미터
m = 0.02; g = 9.81; k = 2.4832e-5;

kp = 2000; % 제어기 게인
kd = 50;

%% MagLev 시스템 상수
k1 = 1.05; % A/V
k2 = 143.48; % V/m
offset = -2.8; % 전압 오프셋

%% 시뮬레이션 설정
tspan = [0 10]; % 시뮬레이션 시간
x0_init = [(-1.5 + 2.8) / 143.48; 0]; % 초기 위치 (전압 -1.5V에 대응되는 위치), 속도 0

% 참조 신호 선택: 'step' or 'sin'
ref_mode = 'sin'; % 변경 가능!

%% ODE 시뮬레이션
[t, X] = ode45(@(t, x) maglev_dynamics(t, x, ref_mode, kp, kd, m, g, k, k2, offset), tspan, x0_init);

%% 전류 계산
current = zeros(size(t));
x_ref_arr = zeros(size(t));
for idx = 1:length(t)
    % 참조 신호 계산
    if strcmp(ref_mode, 'step')
        x_v_ref = -1.5;
        dx_v_ref = 0;
    else
        x_v_ref = -1.5 + 0.3 * sin(2*pi*0.4*t(idx));
        dx_v_ref = 0.3 * 2*pi*0.4 * cos(2*pi*0.4*t(idx));
    end

    % 전압 → 위치 변환
    x_ref = (x_v_ref + 2.8) / 143.48;
    dx_ref = dx_v_ref / 143.48;

    v = kp * (x_ref - X(idx,1)) + kd * (dx_ref - X(idx,2));
    term = g - v;
    if term < 0
        term = 0;
    end
    current(idx) = sqrt((m/k) * term * X(idx,1)^2);
    x_ref_arr(idx) = x_ref;
end

%% 결과 그래프
figure('Position',[100 100 600 800]);

subplot(3,1,1);
plot(t, X(:,1), 'b', t, x_ref_arr, 'r--', 'LineWidth',1.5);
ylabel('Position (m)'); grid on; legend('Ball Position','Reference');

subplot(3,1,2);
plot(t, X(:,2), 'm', 'LineWidth',1.5);
ylabel('Velocity (m/s)'); grid on;

subplot(3,1,3);
plot(t, current, 'k', 'LineWidth',1.5);
xlabel('Time (s)'); ylabel('Current (A)'); grid on;
title('MagLev System: Position, Velocity, and Current');

%% Dynamics 함수
function dxdt = maglev_dynamics(t, x, ref_mode, kp, kd, m, g, k, k2, offset)
    if strcmp(ref_mode, 'step')
        x_v_ref = -1.5;
        dx_v_ref = 0;
    else
        x_v_ref = -1.5 + 0.3 * sin(2*pi*0.4*t);
        dx_v_ref = 0.3 * 2*pi*0.4 * cos(2*pi*0.4*t);
    end

    % 전압 → 위치 변환
    x_ref = (x_v_ref + 2.8) / 143.48;
    dx_ref = dx_v_ref / 143.48;

    % 제어기 (PD)
    v = kp * (x_ref - x(1)) + kd * (dx_ref - x(2));
    term = g - v;
    if term < 0
        term = 0;
    end
    i = sqrt((m/k) * term * x(1)^2);

    % 시스템 동역학
    dxdt = [x(2);
            g - (k/m) * (i^2) / (x(1)^2)];
end
```

## 전류 입력 계산기(v로 부터)
```matlab
function i = CurrentCalculator(v, x)
    m = 0.02; g = 9.81; k = 2.4832e-5;

    % Term 계산
    term = g - v;
    if term < 0
        term = 0;
    end

    % Debug: term 값 출력
    % disp(['term: ', num2str(term), ', v: ', num2str(v)]);

    % x가 너무 작으면 0 나누기 방지
    if abs(x) < 1e-6
        x = 1e-6;
    end

    % Current 계산
    i = sqrt((m/k) * term * x^2);
end
```

## 출력 (전압)과 상태(포지션)의 관계
$$
x_v = k_2 \cdot x + \text{Offset}
$$

$$
x_v = 143.48 \cdot x - 2.8
$$

| 위치 $x$ (m)         | 전압 $x_v$ (V)            |
| ------------------ | ----------------------- |
| $x=0$ (센서 기준점)     | -2.8V                   |
| $x > 0$ (구슬이 아래쪽)  | -2.8V보다 **큰 값**         |
| $x < 0$ (구슬이 자석 쪽) | -2.8V보다 **더 작은 값 (음수)** |

포지션은 센서와 구슬과의 거리

포지션이 음수이면 구슬이 센서보다 위에 존재

포지션이 양수이면 구슬이 센서보다 아래에 존재

-1.5면 0.906 cm 정도



## 결과
시뮬링크 상에서 성공!
![](images/전체%20시뮬링크%20구조.png)

제어기 구조
![](images/제어기%20시뮬링크%20구조.png)

ref vs act
![](images/ref_vs_act.png)


cost
![](images/cost.png)


## controller performance
```m
% === 1. 데이터 추출 ===
t = error.time;             % 시간 [s]
e = error.signals.values;   % 오차 e(t) = ref - actual
y = -e;                     % 응답 y(t) = actual

% === 2. 시간 응답 특성 계산 (stepinfo는 응답 기준이므로 -e 사용) ===
info = stepinfo(y, t, 'SettlingTimeThreshold', 0.01);

rise_time     = info.RiseTime;
peak_time     = info.PeakTime;

% === 3. 새 오버슈트 정의 (steady-state peak 기준) ===
N = length(y);
ss_start = floor(0.8 * N);     % 마지막 20%를 steady-state로 간주
y_ss = y(ss_start:end);

peak_ss  = max(y_ss);     % steady-state 오실레이션의 최대값
peak_all = max(y);        % 전체 응답의 최대값

% 예외 처리
if abs(peak_ss) < 1e-6
    overshoot = NaN;
    warning('Steady-state peak too small to compute overshoot reliably.');
else
    overshoot = ((peak_all - peak_ss) / abs(peak_ss)) * 100;
end

% === 4. 사용자 정의 Settling Time 계산 (steady-peak 이하로 진입 후 유지) ===
% e는 오차 → 0에 수렴하지만 steady-state는 진동하므로 별도 처리
steady_peak = max(abs(e(ss_start:end)));  % 오차의 steady peak

% 오차가 steady peak 이하로 진입한 이후부터 계속 유지되는지 확인
in_band = abs(e) <= steady_peak;

last_out_index = find(~in_band, 1, 'last');
if isempty(last_out_index) || last_out_index == length(e)
    settling_time_custom = 0;
else
    if all(in_band(last_out_index+1:end))
        settling_time_custom = t(last_out_index + 1);
    else
        settling_time_custom = NaN;
    end
end

% === 5. 누적 오차 성능 지표 ===
dt = mean(diff(t));
IAE  = sum(abs(e)) * dt;
ITAE = sum(t .* abs(e)) * dt;
ISE  = sum(e.^2) * dt;
ITSE = sum(t .* (e.^2)) * dt;

% === 6. 결과 출력 ===
fprintf('Overshoot             : %.2f %% (steady-state peak based)\n', overshoot);
fprintf('Rise Time             : %.4f s\n', rise_time);
fprintf('Peak Time             : %.4f s\n', peak_time);
fprintf('Settling Time (custom): %.4f s (steady-state peak based)\n', ...
        settling_time_custom);
fprintf('IAE                   : %.4f\n', IAE);
fprintf('ITAE                  : %.4f\n', ITAE);
fprintf('ISE                   : %.4f\n', ISE);
fprintf('ITSE                  : %.4f\n', ITSE);
```



세틀링 타임: 정상상태 피크의 절댓값 안에 계속 머무르기 시작한 시간을 기준으로 계산

오버슈트: 정상상태에서의 피크값과 가장 피크값의 비율로 계산

### performance of my Controller
```
Overshoot             : 181.88 % (steady-state peak based)
Rise Time             : 0.0000 s
Peak Time             : 0.0130 s
Settling Time (custom): 0.0890 s (steady-peak based)
IAE                   : 0.0065
ITAE                  : 0.0302
ISE                   : 0.0000
ITSE                  : 0.0000
```
### performace of linear PID Controller
```
Overshoot             : 0.45 % (steady-state peak based)
Rise Time             : 0.0517 s
Peak Time             : 1.7010 s
Settling Time (custom): 7.9770 s (steady-state peak based)
IAE                   : 0.2286
ITAE                  : 1.1474
ISE                   : 0.0065
ITSE                  : 0.0330
```