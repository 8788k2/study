# DDPM(Denoising Diffusion Probabilistic Models)
Diffusion 기법중 DDPM을 활용한 논문 (Denoising Diffusion Probabilistic Models)을 공부하고 정리하는 것을 목표로 한다.

![alt text](images/DDPM_diagram_1.png)

**Diffusion Probabilistic Models (DPMs)**: 확률적 과정(stochastic process)을 기반으로 한 생성 모델. 데이터에 점진적으로 노이즈를 추가하는 과정(Forward Process)과, 이를 제거하며 원본 이미지를 복원하는 과정(Reverse Process)으로 학습됨

**Denoising Score Matching**:
데이터의 score function(원본 이미지와 대조를 통해 FID 스코어를 매김)을 학습하여 노이즈를 제거하는 방법 

## Abstract
고품질의 이미지 생성 결과를 얻는데 집중

비평형 열역학에서 영감을 받아 잠재변수 (latent variable) 모델을 사용

DPMs와 **Langevin dynamics**(확률적 미분방정식을 통한 샘플링 방법) 사이에 새로운 연결 제안

**Denoising Score Matching** 기법을 활용하여 노이즈를 제거하고 이미지 품질을 개선.

**Weighted Variational Bound**를 적용하여 최적 모델 학습

**Progressive Lossy Decompression** 기법을 도입하여 점진적으로 손실이 발생하는 디코딩 구조를 적용

SOTA 대비 우수한 성능 달성

## Introduction
본 논문에서는 **DPM(Diffusion Probabilistic Model)** 을 활용한 새로운 방법을 제안.

DPM은 **마르코프 체인(Markov Chain)** 을 기반으로 학습되며, **변분 추론(Variational Inference)** 을 사용하여 샘플을 생성함.

Forward Process: 점진적으로 가우시안 노이즈를 추가하여 원본 데이터를 파괴하는 과정.

Reverse Process: 학습된 모델을 통해 Forward Process를 반전시켜 노이즈를 제거하고 원본 데이터를 복원하는 과정.

### DDPM의 장점
정의가 비교적 간단하고 훈련이 효율적임.

**기존에는 높은 품질의 샘플을 생성할 수 있다는 명확한 증거가 부족했으나, 본 연구에서는 이를 증명함**

기존 생성 모델보다 성능이 뛰어난 경우도 있음.

**특정한 파라미터화(parameterization) 방식이 Denoising Score Matching 및 Langevin Dynamics와 수학적으로 등가(equivalence) 관계를 가짐**

## Background 
논문에서 설명하는 수식들을 이해하기 위한 기본 개념들을 설명하는 것을 목표로 한다. 

$x_0$는 원본데이터, $[x_0, \dots, x_T]$는 foward process에서 노이즈를 추가해나간 데이터를 의미한다.

아래는 평균 $\mu$, 분산 $\Sigma$를 따르는 확률분포를 의미한다. 
$$
\mathcal{N}(\mu, \Sigma)
$$
어떤 확률분포를 간단히 $q$나 $p_\theta$로 쓸 수 있다.

아래는 $x_0$가 어떤 확률분포 $q$를 따르며 $q(x_0)$라는 확률밀도함수값을 가짐을 의미한다.
$$
x_0 \sim  q(x_0)
$$


### likelihood

Probability, $p(x | \theta)$: 주어진 확률분포는 고정으로 두고 관측되는 사건이 달라지는 경우 확률을 표현하는 단어 

likelihood, $L(\theta | x )$: 관측되는 사건은 고정으로 두고 확률분포의 변화를 보고 싶은 경우 확률을 표현하는 단어 

where, $\theta$: 확률분포를 구성하는 파라미터, $x$: 관측값

likelihood와 확률밀도 함수는 수식적으로는 같은 값을 가지나, 목적이 조금 다르다. 
예를 들어 $q(x_0)$는 어떤 확률분포 $q$에서 $x_0$가 샘플링될 확률밀도 함수값인과 동시에 라이클리후드로 해석할 수 있다. 
어떤 고정된 데이터를 두고 그 데이터를 포함하는 확률분포가 변화할때 얼마나 그 데이터를 샘플링할 확률이 높은지를 평가하는 최적화 도구로써 사용할 수 있기 때문이다. 

따라서 diffusion의 목적은 모델이 원본데이터 $x_0$를 내놓을 likelihood를 높이는 것 즉, 특정한 확률분포를 찾도록 학습하는 것이다. 





### foward process
$T$ 스텝 까지 데이터 $x_0$에 점진적으로 노이즈를 추가하는 과정

$$
q(x_{1:T} | x_0)
$$
마르코프 체인으로 정의되며 각 $t$스텝에서 가우시안 노이즈가 추가된다.  $q(x_t)$는 데이터 $x_t$가 샘플링되는 확률밀도 함수를 의미한다. 
$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)
$$
$\beta_t$가 커질 수록 노이즈도 커진다. 시간이 지나면 $x_0$는 완전한 가우시안 노이즈 상태로 변한다.

$$
x_T \sim  \mathcal{N}(0,I)
$$

### reverse process
foward process를 역전시켜, 노이즈가 있는 상태 $x_T$에서 원본 데이터 $x_0$를 복원하는 과정
$$
p_{\theta}(x_{0:T})
$$

$x_t$에서 $x_{t-1}$로 복원하는 과정은 아래와 같이 정의 된다.
$$
p_{\theta}(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_{\theta}(x_t, t), \Sigma_{\theta}(x_t, t))
$$
$\mu_{\theta}(x_t, t)$와 $\Sigma_{\theta}(x_t, t)$는 모델이 학습한 평균벡터와 공분산 행렬 

### 학습목표
전체 forward process의 확률 분포를 다음과 같이 정의
$$
q(x_{1:T} | x_0) = \prod_{t=1}^{T} q(x_t | x_{t-1})
$$
전체 reverse process의 확률 분포를 다음과 같이 정의
$$
p_{\theta}(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_{\theta}(x_{t-1} | x_t)
$$

**한편, 앞서 언급했듯이 우리의 학습 목표는 모델이 $x_0$를 내놓을 확률, likelihood를 $p_θ(x_0)$를 1에 가깝게 하는 것이다.** 

**$p_θ(x_0)$에 log를 씌우고 -를 붙이면 likelihood가 1에 가까울 수록 점점 더 0에 빠르게 수렴하는 훌륭한 성질의 nagative log likelihood를 잘 정의할 수 있고 그 기댓값을 loss function으로 사용한다.** 

**그러나 직접 loss function을 최적화하기 어려우므로 아래의 부등식이 성립함을 증명하여 우변을 최적화하는 방식으로 자연스럽게 loss function을 최적화하도록 한 것이 기존 2015년 논문의 핵심 컨트리뷰션이다.**
$$
\mathbb{E}[-\log p_{\theta}(x_0)] \leq \mathbb{E}_q \left[ -\log \frac{p_{\theta}(x_{0:T})}{q(x_{1:T} | x_0)} \right]
$$
위 식의 우변은 아래와 같이 KL Divergence의 합으로 표현될 수 있으므로 clsed form으로 잘 해석이 가능하다.
$$
L = \mathbb{E}_q \left[ D_{KL}(q(x_T | x_0) \parallel p(x_T)) + \sum_{t>1} D_{KL}(q(x_{t-1} | x_t, x_0) \parallel p_{\theta}(x_{t-1} | x_t)) - \log p_{\theta}(x_0 | x_1) \right]
$$


$$
\mathbb{E}_q \left[ 
\underbrace{D_{KL} ( q(\mathbf{x}_T | \mathbf{x}_0) \parallel p(\mathbf{x}_T) )}_{L_T}
+ \sum_{t>1} \underbrace{D_{KL} ( q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) \parallel p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_t) )}_{L_{t-1}}
- \underbrace{\log p_{\theta}(\mathbf{x}_0 | \mathbf{x}_1)}_{L_0}
\right]
$$
각항에 대해 설명을 해보자

$L_T$: 마지막 상태$(x_T)$에 대해 확률분포 q와 p의 KL Divergnece를 최소화하는 항

$q(x_T|x_0)$와 $p(x_T)$의 차이를 최소화해야 하는데 $p(x_T)$는 가우시안 분포이므로 앞서 언급했던 $\beta_t$를 조절함으로써 $q(x_T|x_0)$가 가우시안 분포에 가깝도록 조절할 수 있다. 

$L_{t-1}$: 현재 상태($x_t$)가 주어질 때, 이전 상태($x_{t-1}$)가 나올 확률 분포 








**$x_0$에서 $x_T$를 생성하는 프로세스의 마르코프 체인은 가우시간 분포를 따른다고 가정하고 $x_T$에서 $x_0$를 복원하는 프로세스에서 확률분포를 학습하여 모델이 $x_0$를 내놓을 확률을 최적화**

본 논문의 핵심 컨트리뷰선은 다음과 같다.
$$
q(x_t | x_0) = \mathcal{N} \left( x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I \right)
$$

특정 $t$시간에서의 상태 $x_t$를 한번에 샘플할 수 있는 공식이 아래와 같이 주어질 수 있다. 
$$
\alpha_t = 1 - \beta_t \quad \text{(\textbf{노이즈 첨가율})}
$$

$$
\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s \quad \text{(\textbf{누적 노이즈 첨가율})}
$$


---
가우시안 분포로 포워드 프로세스 정의했지만 현실의 노이즈 경향을 포함하여 최적화하고 싶다면 원본에서 노이즈로 가는 확률분포를 다른 모델을 통해 학습하면 어떨까?? 



