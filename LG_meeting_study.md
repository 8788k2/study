# LG 과제 3월 13일 미팅 준비
LG과제에 대한 전반적인 공부와 상기된 날짜의 미팅 준비를 목적으로 한다.


내일 오전 대화때 할 질문을 만드는 것을 서브목표로 해보자. 

타깃은 공항 공조 시스템 

재실자 몰리는 시간대와 아닌 시간대 나눔
희망온도에 맞추는 입력 달라져야 한다 MPC로 최적화 가능? 



# 논문 읽기
논문은 다음과 같이 구성됨.
```
Section 2: 연구 대상 건물 및 HVAC 시스템 설명.
Section 3: 건물 동역학 분석 및 ANN 모델 구축 방법 설명.
Section 4: 최적의 예측 제어(Optimal Start-Stop Control) 기법 소개.
Section 5: 시뮬레이션 및 모델 성능 비교.
Section 6: 연구 대상 시스템에 ANN 기반 예측 제어 적용.
```





## Abstract**

건물의 열역학적 특성 비선형적 

이 논문에서는 단일 구역 모델이 아니라 다중 구역(Multi-Zone) 모델을 사용하여 인접 구역 간의 열적 상호작용을 반영한 예측 제어 방법을 제안

ANN을 사용하여 건물의 온도 변화를 학습하고, 이를 바탕으로 다중 구역의 열적 상호작용을 반영한 모델을 구축하는 방법을 연구

기계적 냉각(Mechanical Cooling), 환기(Ventilation), 날씨 변화(Weather Change), 특히 인접 구역 간의 대류 열 전달(Convective Heat Transfer)을 고려 

제안된 모델을 기반으로 간단하지만 효과적인 모델 기반 예측 제어(Model-Based Predictive Control) 방법이 개발되었으며, 이를 통해 쾌적한 실내 온도를 유지하면서 에너지 소비를 줄일 수 있음이 확인되었다.

## Introduction**

### (1) 건물 에너지 소비와 HVAC 시스템의 중요성

건물은 전 세계 에너지 소비의 40%를 차지하며, CO₂ 배출의 33%를 담당한다.
이 중 절반 이상의 에너지가 HVAC 시스템(난방, 환기, 공조)에 사용된다.
에너지 비용 절감과 환경 문제 대응이 중요한 과제가 되었지만, 기존의 HVAC 시스템은 비효율적으로 운영되고 있다.
따라서, 상업용 건물에서 예측 제어(Predictive Control)를 적용할 수 있는 새로운 건물 모델링 방법이 필요하다.  

### (2) 기존의 PID 제어 방식과 한계점

현재 상업용 HVAC 시스템에서는 PID(Proportional-Integral-Derivative) 제어와 On/Off 제어가 주로 사용됨.
이러한 제어 방식은 실시간으로 측정된 온도를 기반으로 작동하지만, 건물의 열역학적 특성을 충분히 반영하지 못함.
건물의 열 관성(Thermal Inertia)이 크기 때문에 온도 변화가 지연되어 불필요한 에너지 낭비와 낮은 열적 쾌적성(Thermal Comfort) 문제가 발생함.
최근 연구에 따르면, 예측 제어 기법을 사용하면 HVAC 시스템의 에너지 비용을 크게 줄일 수 있음.

### (3)  예측 제어(Predictive Control)와 적용 가능성

예측 제어는 날씨 예측, 점유율(occupancy) 예측 등의 데이터를 활용하여 최적의 HVAC 운영 전략을 수립할 수 있음.
예측 제어는 기존의 여러 제어 기법(Optimal Start-Stop Control, Load Shifting Control, Demand Limiting Control 등)과 통합하여 운영 가능.
하지만 예측 제어를 구현하려면 건물의 열역학적 모델을 정확하게 구축해야 하며, 이를 위해 적절한 모델링 방법이 필요하다.

### (4) 건물 모델링의 어려움

건물의 열역학적 모델을 만드는 것은 쉽지 않다. 주요 도전 과제는 다음과 같다.

건물 환경은 시시각각 변하는 동적 시스템

점유자의 수 변화, 태양 복사열 변화 등이 실내 온도 변동을 유발함.
비선형 변수들이 존재

온도, 습도, 외부 공기 댐퍼(Damper) 작동 등의 요소들이 비선형적으로 작용하여 전통적인 물리 모델로는 정확한 예측이 어려움.
HVAC 시스템의 복잡한 상호 작용

예를 들어, AHU(Air Handling Unit) 시스템은 냉각수 온도 변화, 공기 흐름 속도 변화 등의 영향을 받아 개별적으로 독립적인 제어가 어려움.
건물 내부 공간이 다중 구역(Multi-Zone)으로 나뉘어 있음

각 구역이 서로 영향을 미치므로, 단순한 단일 구역(Single-Zone) 모델이 아니라 다중입력-다중출력(MIMO) 시스템 모델링이 필요.

### (5) 통계적 모델과 ANN(Artificial Neural Network) 모델의 필요성

기존의 ARX, ARMAX, Subspace 모델 등의 통계적 방법도 사용됨.
하지만 이러한 통계 모델은 선형 시스템을 가정하기 때문에 강한 비선형성과 불확실성이 있는 HVAC 시스템에서는 정확도가 떨어짐.
따라서, 최근에는 ANN(Artificial Neural Network) 기반 모델링이 주목받고 있음.
ANN 모델은 비선형 시스템을 학습할 수 있는 장점이 있어 HVAC 시스템과 같이 복잡한 건물 열역학을 모델링하는 데 적합함.

## Building**

...

## Modeling**

## Analytical model

### (1) Analytical Model (분석 모델) 설명

이 섹션에서는 다중 구역(Multi-Zone) 건물의 열적 거동을 설명하기 위한 물리 모델을 소개합니다. 특히, 구역 간 대류 열 전달(Convective Heat Transfer)의 중요성을 고려한 모델을 제시합니다.

### (2) 모델의 가정

모델을 단순화하기 위해, 각 구역의 공기는 완전히 혼합되며, 벽과 천장만이 열전달에 영향을 준다고 가정.

### (3) 모델의 열역학 방정식
$$
C_z^1 \frac{dT_1}{dt} = \dot{m}C_a (T_{sa,1} - T_1) + \frac{T_2 - T_1}{R_f} + \frac{T_{out} - T_1}{R_{g,1}} + \frac{T_{w,1} - T_1}{R_{w,1}} + Q_1
$$
$$
C_z^2 \frac{dT_2}{dt} = \dot{m}C_a (T_{sa,2} - T_2) + \frac{T_1 - T_2}{R_f} + \frac{T_{out} - T_2}{R_{g,2}} + \frac{T_{w,2} - T_2}{R_{w,2}} + Q_2
$$

### **(4) 모델의 주요 결론**

구역 간 열적 상호작용은 두 가지 요인에 의해 결정됨: 구역 간 온도 차이(Temperature Difference),
대류 열 전달 계수(Convective Heat Transfer Coefficient, $R_f$)

온도 차이는 센서를 통해 측정 가능하지만, 대류 열 전달 계수는 직접 측정이 어렵다.
→ 이 계수는 공기 흐름 및 구역 내 점유 상태 등에 따라 달라지므로, 모델링이 복잡해짐.


$Q_1, Q_2$  (태양 복사열, 점유자 열 등)은 불확실성이 커서, 전통적인 수학적 모델만으로 정확히 예측하기 어려움.
→ 이러한 불확실성을 해결하기 위해 ANN 모델이 필요.

**전통적인 수학적 모델만으로는 불확실성이 있는 대류 열 전달 및 추가적인 열원을 정확히 반영하기 어려움 → ANN 기반 모델링 필요성 강조.**


## NARX model for MIMO modelling
이 섹션에서는 NARX(Nonlinear AutoRegressive with eXogenous input) 모델을 활용하여 다중 입력-다중 출력(MIMO) HVAC 시스템을 모델링하는 방법을 설명한다.
이는 기존의 물리 모델이 복잡한 건물의 열역학적 특성을 충분히 반영하지 못하는 한계를 극복하기 위한 접근 방식이다. 건물의 열역학적 시스템은 비선형적(Nonlinear)이며, 확률적인 불확실성(Stochastic Uncertainty)이 포함됨.

### NARX 모델 개념
NARX는 반복적인 동적 네트워크(Recurrent Dynamic Network)로, 피드백 구조를 포함.

출력값이 현재 및 과거 입력값뿐만 아니라 과거 출력값에도 의존하는 구조를 가짐.

다중 입력-다중 출력(MIMO) 시스템에서 비선형성을 학습하는 데 적합.

$$
\hat{y}(k+1) = f(\phi(k), w) + e(k)
$$

$$
\phi(k) = [y(k), y(k-1), ..., y(k - n_y), u(k - n_k), ..., u(k - n_u - n_k)]
$$

입력-출력 데이터는 이전 시간의 정보를 포함해야 하므로, 데이터 준비 과정이 중요.

입력 지연 시간(𝑘)은 샘플링 시간(10분)보다 짧아야 함.



### 입력-출력 행렬 표현
NARX 모델을 학습하기 위해 입력-출력 행렬을 구성한다.
$$
U =
\begin{pmatrix}
T_i(r) & \cdots & T_i(r - n_a + 1) & T_{out}(r) & \cdots & T_{out}(r - n_b + 1) & T_{sa,i}(r) \\
T_i(r+1) & \cdots & T_i(r - n_a + 2) & T_{out}(r+1) & \cdots & T_{out}(r - n_b + 2) & T_{sa,i}(r+1) \\
\vdots & \ddots & \vdots & \vdots & \ddots & \vdots & \vdots \\
T_i(n) & \cdots & T_i(n - n_a + 1) & T_{out}(n) & \cdots & T_{out}(n - n_b + 1) & T_{sa,i}(n)
\end{pmatrix}
$$

$$
Y =
\begin{pmatrix}
\hat{T}_i(r+1) \\
\hat{T}_i(r+2) \\
\vdots \\
\hat{T}_i(r+n)
\end{pmatrix}
$$

### NARX 모델의 차수 선택 (Order Selection)
입력 차수와 출력 차수는 예측 성능에 큰 영향을 미침.
차수가 클수록 모델이 장기적인 패턴을 학습할 수 있지만, 복잡성이 증가하여 과적합(overfitting) 위험이 있음.
Forward Selection 방법을 사용하여 가장 적절한 입력 변수를 선택.

### Forward Selection을 통한 변수 선택
Forward Selection 방법은 다음 단계를 거쳐 최적의 입력 변수를 선택한다.

시스템의 물리적 특성을 고려하여 가장 중요한 입력 변수를 선택
초기 차수를 1로 설정

입력-출력 데이터셋을 사용하여 ANN 모델을 학습
RMSE(Root Mean Square Error)를 계산하여 모델 성능 평가

입력 차수를 증가시키고, 은닉층(hidden layer) 개수를 조정하여 성능 비교
최소 RMSE가 나올 때까지 반복

새로운 모델의 성능이 이전 모델보다 향상되었는지 평가
성능이 개선되면 차수를 유지, 그렇지 않으면 다른 변수를 추가하여 반복

### NARX 모델 정리
NARX 모델은 HVAC 시스템의 복잡한 비선형성을 모델링하는 데 효과적임.
과거의 입력과 출력을 모두 활용하여 예측 성능을 향상시킴.
입력 변수 선택 및 차수 조정이 중요하며, Forward Selection을 통해 최적화 가능.
결과적으로, ANN을 활용한 NARX 모델은 기존 물리 기반 모델보다 높은 예측 정확도를 제공할 것으로 기대됨.
## 용어
AHU(Air Handling Unit) 골기정화기

PID