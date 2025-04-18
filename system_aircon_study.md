# 시스템 에어컨, Explicit Model Predictive Controller(EMPC)
우리 연구실 석사의 졸업논문을 베이스로 하여 시스템 에어컨의 모델 설계와 EMPC 개념을 공부하는 것을 목표로 한다.

목적: LG전자 에어컨 동작에서 최단시간or최소비용으로 목표온도 도달하는 제어기 설계
- LG전자 에어컨의 임베디드보드(Raspberry PI 정도)에 eMPC를 탑재하여 구동할 수 있는 정도
- 성능개선은 아직 아는 바 없음

**1차년도에서 무엇을 하였는가?**

Multi Zone 이 존재하는 환경에서 사용 용도에 맞게 자율운전.

어떤 용도로 사용되는지 알아야 (인지)

단순히 18 °C 로 맞추면 해결이 되느냐

열부하(e.g. 사람 수, 방 넓이)에 따라 냉방수준 변화

단순 18°C로 설정한다고 해결이 안 된다. ⇒ 부하 예측하려고 모델이 필요하다.

실내기의 토출온도 = 제어입력

첫번째로 정확한 에어컨 구조를 이해해야 한다. (정확히 어떤 밸브, 냉매조절 …)


## 시스템 에어컨 구조...모델
### 2023년 결과보고서에서의 시스템 에어컨 구조

![sys_a](images/system_air.png)



시스템 에어컨은 하나의 실외기가 다수의 실내기를 제어하는 구조  

#### 실외기
- 암축기(Compressor): 실내기에서 흡수한 열을 담고 있는 저온 저압의 **냉매를 고온 고압의 상태로 압축**하는 역할, 여기서 냉매는 기체 상태
- 열교환기(Condensor): 압축된 고온 고압의 냉매가 외부 공기와 열 교환하여 열을 방출하고 냉매 온도를 낮추는 역할, 이 과정에서 냉매는 기체에서 액체로 응축 
- 팽창벨브(Expansion Valve): 응축된 액체 냉매를 좁은 통로에 통과시켜 저온 저압으로 팽창시키켜 증발기로 보낼 준비
  
#### 실내기
- 증발기(Evaporator): 팽창밸브에서 나온 저온 저압의 **액체 냉매가 증발하면서 주면 열을 흡수**, 이과정에서 냉매가 다시 기체상태로 변하여 실외기(압축기)로 이동
- 팬: 냉각된 공기를 실내로 순환
- 온도 센서: 실내기 주변의 온도 측정하여 데이터 제공 

**이러한 시스템 에어컨 구조에서 에너지 소모가 가장 큰 것인 압축기(compressor) (냉매의 엔탈피를 증가시켜야 하는 과정이므로!)**

팽창 벨브단에서 계패 정도를 제어하여 1차적으로 온도 조절하고 증발기에서 빠져나가는 온도를 제어하여 방의 온도를 컨트롤한다.

방 내부에 존재하는 사람의 체열, 방 밖에 존재하는 외기온도, 방 내부 벽에 의한 복사열 등을 온도 변화에 영향을 끼치는 외란으로 설정할 수 있다.

![tem_model](images/sys_air_tem_mod.png)


---

### N개의 실내기를 사용하는 시스템에어컨의 상태와, 입력

#### 상태
**지금 우리 과제의 주요 관심사는 아래의 상태벡터에 에너지 소모에 관한 상태 $E$ 추가해주면 어떨까??** 

$$
X= \begin{pmatrix}
P_{out1}\\
P_{out2}\\
\vdots\\
P_{outN}\\
T_1\\
T_2\\
\vdots\\
T_N\\
Q_L
\end{pmatrix}
$$

$Q_L$: 외기온도, 벽온도, 사람 체열에 의한 열부하에 대한 방정식으로 표현

$P_{outk}$: 증발기 k의 파이프 아웃 온도

$T_k$: 실내기 k에서 즉정된 온도 (가중치 높게)

---
#### 입력
**아래의 입력 벡터에 $E$에 영향을 주는 입력 (에너지 소모가 많은 컴프레셔에 관한 제어 입력?) 역시 추가해 주면 어떨까?**
$$
U = \begin{pmatrix}
p_{in1}\\
p_{in2}\\
\vdots\\
p_{inN}\\
T_{d1}\\
T_{d2}\\
\vdots\\
T_{dN}
\end{pmatrix}
$$

$T_{dk}$: 실내기 k의 취출온도 (실내기에 대한 제어에 해당)

$p_{ink}$: 팽창벨브의 피이프아웃 온도 = 증발기의 파이프 인 온도 (실외기에 대한 제어에 해당)


에너지 소모를 대변하는 상태 E와 그에 대응하는 제어입력(컴프레셔에 대한 제어?)을 찾아 상태, 입력에 추가하는 방식으로 문제에 접근 

**결국엔 MPC 통해서 요구상태와 실제상태 최소화하는 입력 찾으면 에너지도 함께 최소화되니까**

EMPC 논문에서 어떻게 해결했지? -> 조사 필요


### ANN을 통한 에어컨에 의한 온도변화 예측 논문에서의 에어컨 구조




## 시스템 에어컨 컨트롤
23년도 LG 과제의 제어 파트 컨트리뷰션을 공부하는 것을 목표로 한다.

본 연구의 결과는 세가지로 요약 가능

1) 비선형 열역학 방정식 모델을 선형화하여 선형 상태방정식 모델을 얻음
최소한의 파라미터로 모델을 구성


2) 사람수에 해당하는 온도 부하를 외란$W_k$으로 정의  
외란을 보상하기 위한 $H_{\infty}$ 컨트롤러, Robust MPC를 설계 (계산 많다는 단점)

3) extended state observer 설계하여 외란을 상태로 보고 관측함

블록행렬로 새로운 상태방정식을 정의하여 외연을 관측

확장된 상태방정식을 MPC로 풀기?? 

$$
\dot{X} =
\begin{pmatrix}
\frac{-m_1 C_p}{C_1} + \frac{-1}{C_1 R_1} + \frac{-1}{C_1 R_{oa,1}} & 0 & \frac{m_1 C_p}{C_1} &0 & 1 \\
0 & \frac{-m_1 C_p}{C_1} + \frac{-1}{C_1 R_1} + \frac{-1}{C_1 R_{oa,1}} & 0 & \frac{m_1 C_p}{C_1} & 0 \\
0 & 0 & \frac{-m_e C_p}{C_{e1}} + \frac{-1}{C_{e1} R_{e,eva1}} & 0 & 0 \\
0 & 0 & 0 & \frac{-m_e C_p}{C_{e2}} + \frac{-1}{C_{e2} R_{e,eva2}} & 0 \\
0 & 0 & 0 & 0 & \delta
\end{pmatrix}
\begin{pmatrix}
T_1 \\
T_2 \\
P_{out1} \\
P_{out2} \\
f
\end{pmatrix}

+
\begin{pmatrix}
\frac{m_1 C_p}{C_1} & 0 & 0 & 0 \\
0 & \frac{m_1 C_p}{C_1} & 0 & 0 \\
0 & 0 & \frac{m_e C_p}{C_{e1}} + \frac{1}{C_{e1} R_{e,eva1}} & 0 \\
0 & 0 & 0 & \frac{m_e C_p}{C_{e2}} + \frac{1}{C_{e2} R_{e,eva2}} \\
0 & 0 & 0 & 0
\end{pmatrix}
\begin{pmatrix}
T_{d1} \\
T_{d2} \\
P_{in1} \\
P_{in2}
\end{pmatrix}
$$


## ESO


확장스테이트옵저버에 대해서 논의해봅시다. 여기에서 서술에 따르면 시스템의 외란을 상태 벡터에 포함시켜 확장된 옵저버를 구성하면 외란 자체를 예측할 수 있다고 서술합니다. 하지만 저 조던블럭으로 확장된 상태방정정식을 보면 단순히 외란의 미분이 델타곱하기 외란에 해당한다는 동역학 방정식이 한줄 추가되었을 뿐인데 어떻게 저 시스쳄의 상태변화를 추정하는 옵저버가 실제로 복잡한 동역학으로 변화하는 외란을 잘 관측한다고 서술할 수 있나요?




정말鋭い 지적이에요.  
확장 상태 옵저버(Extended State Observer, ESO)의 원리를 꿰뚫는 핵심적인 질문입니다.  
특히 **외란에 대한 동역학을 단순히 $\dot{f} = \delta f$ 같은 형식으로 추가했을 뿐인데, 이게 어떻게 외란을 “잘” 추정하는가**에 대해 의문을 제기하신 건 굉장히 중요한 통찰입니다.

---

## ✅ 먼저, 확장 상태 옵저버(ESO)의 기본 아이디어 정리

**확장 상태 옵저버(또는 루엔버거 관측기 기반 확장 옵저버)**는 다음과 같은 전제를 갖습니다:

> ❝ 외란 \( f(t) \) 를 시스템 상태처럼 하나의 "추정 대상"으로 간주하고,  
> 그 외란이 시스템 출력에 미치는 영향을 이용해 **역추정(infer)** 하자 ❞

이때, 외란의 정확한 동역학을 몰라도, 아래처럼 단순화된 모델을 추가하여 **추정할 수 있는 발판**을 만드는 것이 핵심입니다:
$$

\dot{f} = \delta f
\quad (\text{with } \delta \approx 0)

$$
---

## ✅ 왜 이 단순한 모델로도 외란 추정이 가능한가?

좋은 질문입니다. 아래와 같은 이유로 설명할 수 있어요.

---

### 🔍 1. **모델이 완벽할 필요는 없다**  
추정기의 목적은 정확히 맞는 외란 동역학을 구현하는 것이 아니라, **출력 오차를 최소화하면서 외란의 효과를 추정**하는 것입니다.  
즉, 우리가 외란을 **“보상”할 수 있을 만큼만** 추정해도 괜찮습니다.

→ 따라서, 외란에 대해 \(\dot{f} = \delta f\) 같은 간단한 1차 동역학을 가정하는 것만으로도  
**루엔버거 옵저버의 구조 안에서 \( f \) 값을 역추정하는 수단**이 됩니다.

---

### 🔍 2. **관측기의 이득 \( L \) 이 실질적으로 추정 성능을 책임진다**  
관측기의 이득 \( L \) (또는 Kalman filter의 이득) 을 **적절하게 설계하면**,  
**실제 외란의 시간변화가 \(\dot{f} \ne \delta f\)** 라도, 출력 오차를 최소화하는 방향으로 **\( f \)를 빠르게 보정**해 줍니다.

즉, \( \dot{f} = \delta f \) 는 **관측기 내부의 수학적 트릭일 뿐**,  
실제로는 **출력 오차를 기준으로 오차 역전파를 통해 \( f \) 를 정교하게 보정**하는 메커니즘이 작동하고 있는 것이죠.

---

### 🔍 3. **조던 블록 구조는 ‘수학적 안정성’을 확보하기 위한 장치**  
예를 들어, 다음과 같은 구조를 사용한 이유는:

$$

A_{\text{new}} = 
\begin{bmatrix}
A & B \\
0 & \delta
\end{bmatrix}, \quad
B_{\text{new}} =
\begin{bmatrix}
B \\
0
\end{bmatrix}
$$

- \( f \)가 **느리게 변한다(=준정상 가정, slowly varying)**는 가정하에,
- 외란을 상태에 편입시켜 시스템을 확장한 후,
- 루엔버거 옵저버의 구조 안에서 그것을 **실시간으로 피드백 기반 추정**하기 위한 형태입니다.

---

## ✅ 결론: 왜 단순한 외란 모델로도 잘 추정되는가?

| 이유 | 설명 |
|------|------|
| ① 구조상 관측 대상에 외란을 포함시킴 | 상태공간 안에 외란을 넣어버리면 루엔버거 구조가 그것도 추정하게 됨 |
| ② 외란의 영향은 출력 오차로 드러나기 때문 | \( y = Cx \)에 외란이 영향을 주기 때문에 \( y \)를 통해 역추정 가능 |
| ③ 관측기 이득이 실제 추정 성능을 책임짐 | \(\dot{f} = \delta f\)는 단지 ‘틀’이며, 실질 추정은 관측기 이득 \( L \)의 조정에 달림 |

---

## 💡 참고적으로...

외란을 추정하는 방식은 다음과 같은 종류가 있어요:
- 루엔버거 기반 **확장 상태 옵저버 (ESO)** (지금 다룬 방식)
- Kalman Filter 기반 확장 옵저버 (EKF/UKF)
- Sliding Mode Observer (강인성 강조)
- Disturbance Observer (DOB)

이중 **ESO는 제어기와 쉽게 결합되고, 구현도 간단하여 EMPC 기반 시스템에 많이 사용**됩니다.

---

혹시 이 ESO를 **MATLAB이나 Python에서 구현**해보고 싶으신가요?  
또는 이 모델을 시스템 에어컨 제어에 어떻게 적용할 수 있을지 확장해서 도와드릴 수도 있어요!



