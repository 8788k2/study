# 시스템 아이덴티피케이션
스마트에어컨 로그 데이터를 이용하여 시스템 아이덴티피케이션을 적용하여 가장 적절한 데이터셋, 데이터 수, 모델을 찾는 것을 목표로 한다.


## 결과
가장 높은 정확도를 보이면서도 적은 데이터셋을 찾기위해 실험

4일간의 경남사무실 데이터가 SSM, OE 모델 적용에 가장 효과적이었음

```
입력: 실내기 두대의 희망온도, 풍량, on/off, 외기온도 (외란)

상태(출력): 5분 뒤의 실내온도
```



# OE(Output Error)
## 정확도 66.13% RMSE:0.6

![OE](images/oe_best_66.png)

## 풍량 없이 정확도: 67.3% RMSE: 0.586

![](images/풍량없이OE67.png)


## OE 모델에 대한 설명
$q^{-1}$: 시간지연 연산자 ex) $q^{-1}u(t)=u(t-1)$ or $q^{-1}y(t)=y(t-1)$

$(nb, nf, nk)=1, 1, 2$
$$
y(t) = \frac{B^{(i)}(q^{-1})}{F(q^{-1})} \cdot u^{(i)}(t-nk) + e(t)
$$

$$
B^{(i)}(q^{-1}) = b^{(i)}_1 \quad (\text{nb}=1\text{이므로})
$$

$$
F(q) = 1 + f_1 q^{-1} \quad (\text{nf}=1\text{이므로})
$$

$$
y(t) = -f_1 y(t-1) + b_1 u_1(t-2) + b_2 u_2(t-2) + \cdots + b_7 u_7(t-2) + e(t)
$$

---
$(nb, nf, nk)=2, 1, 2$
$$
y(t) = \frac{B^{(i)}(q^{-1})}{F(q^{-1})} \cdot u^{(i)}(t-nk) + e(t)
$$

$$
B^{(i)}(q^{-1}) = b^{(i)}_1+b^{(i)}_2q^{-1} \quad (\text{nb}=2\text{이므로})
$$

$$
F(q) = 1 + f_1 q^{-1} \quad (\text{nf}=1\text{이므로})
$$

$$
y(t) + f_1 y(t-1) = b^{(i)}_{1} u^{(i)}(t-2) + b^{(i)}_{2} u^{(i)}(t-3)+ e(t)
$$





# SSM
## 정확도 63.8% RMSE:0.65

![](images/ssm_best_64.png)


## 풍량없이 63.7 RMSE:0.65

![](images/풍량없이SSM63.png)


## SSM 설명
$$
\begin{align*}
x(t+1) &= A x(t) + B u(t) \\
y(t) &= C x(t) + D u(t)
\end{align*}
$$

order=1
$$
\begin{align*}
x(t+1) &= a \cdot x(t) + b_1 u_1(t) + b_2 u_2(t) + \cdots + b_7 u_7(t) \\
y(t) &= c \cdot x(t)
\end{align*}
$$

### 데이터 전처리 코드
```matlab
csv_files = dir('*.csv');

all_data = [];
prediction_step = 60; % (5초 × 60 = 5분)

for i = 1:length(csv_files)
    % 1. 파일별 데이터 불러오기
    fname = csv_files(i).name;
    T = readtable(fname);

    % 2. 컬럼명 공백 제거
    T.Properties.VariableNames = strtrim(T.Properties.VariableNames);

    % 3. 파일명에서 날짜(yyyymmdd) 추출, DateTime 문자열 생성
    date_str = regexp(fname, '\d{8}', 'match', 'once');
    if isempty(date_str)
        error('파일명에서 날짜(yyyymmdd)를 찾을 수 없습니다: %s', fname);
    end
    date_fmt = [date_str(1:4) '-' date_str(5:6) '-' date_str(7:8)];

    % 4. 실내기별 데이터 분리
    df1 = T(T.AutoId == 2, :);
    df2 = T(T.AutoId == 3, :);

    % 5. 컬럼명 통일
    df1.Properties.VariableNames{'Frun'} = 'frun1';
    df1.Properties.VariableNames{'Tcon'} = 'tcon1';
    df1.Properties.VariableNames{'Tid'}  = 'Tid1';
    df2.Properties.VariableNames{'Frun'} = 'frun2';
    df2.Properties.VariableNames{'Tcon'} = 'tcon2';

    % 6. ON/OFF 신호 생성
    df1.on_off1 = double(df1.tcon1 ~= 0 & ~isnan(df1.tcon1));
    df2.on_off2 = double(df2.tcon2 ~= 0 & ~isnan(df2.tcon2));

    % 7. 날짜+시간 결합해 DateTime 컬럼 생성
    df1.DateTime = datetime(strcat(repmat({date_fmt},height(df1),1), {' '}, string(df1.Time)), ...
        'InputFormat','yyyy-MM-dd HH:mm:ss');
    df2.DateTime = datetime(strcat(repmat({date_fmt},height(df2),1), {' '}, string(df2.Time)), ...
        'InputFormat','yyyy-MM-dd HH:mm:ss');

    % 8. DateTime 기준으로 병합
    merged = outerjoin(df1(:, {'DateTime', 'frun1', 'tcon1', 'on_off1', 'Tid1', 'Tod'}), ...
                       df2(:, {'DateTime', 'frun2', 'tcon2', 'on_off2'}), ...
                       'Keys', 'DateTime', 'MergeKeys', true);

    % 9. DateTime 기준 정렬
    merged = sortrows(merged, 'DateTime');

    % 10. 5분 뒤 온도(Tid1_next) 생성
    merged.Tid1_next = [merged.Tid1((1+prediction_step):end); NaN(prediction_step,1)];

    % 11. NaN 제거 (각 파일별로!)
    merged = rmmissing(merged);

    % 12. 누적 저장
    all_data = [all_data; merged];
end

% 13. 최종 입력(U), 출력(Y) 정의
U = [all_data.frun1, all_data.tcon1, all_data.on_off1, ...
     all_data.frun2, all_data.tcon2, all_data.on_off2, all_data.Tod];
Y = all_data.Tid1_next;

% 14. 샘플타임 및 iddata 생성
Ts = 5;
data = iddata(Y, U, Ts);

% 15. 최종 데이터 확인
disp(size(U))
disp(size(Y))
disp(['Min U: ', num2str(min(U(:))), ' / Max U: ', num2str(max(U(:)))])
disp(['Min Y: ', num2str(min(Y)),   ' / Max Y: ', num2str(max(Y))])
```

## OE 코드
```matlab
% 데이터 정규화(이미 했으면 생략)
mu_U = mean(U);
sigma_U = std(U);
U_norm = (U - mu_U) ./ sigma_U;
mu_Y = mean(Y);
sigma_Y = std(Y);
Y_norm = (Y - mu_Y) ./ sigma_Y;
Ts = 5;
data_norm = iddata(Y_norm, U_norm, Ts);

% 최적 fit 저장용 변수
best_fit = -Inf;
best_orders = [];

% 그리드서치 (nb, nf, nk 각 1~2 or 1~3까지 시도, 너무 넓게 하면 시간 오래 걸림)
nb_cand = 1:2;  % 입력 B차수 후보
nf_cand = 1:2;  % F(분모)차수 후보
nk_cand = 1:2;  % 지연 후보

for nbv = nb_cand
for nfv = nf_cand
for nkv = nk_cand
    nb = nbv * ones(1, 7);  % 입력 7개
    nf = nfv * ones(1, 7);
    nk = nkv * ones(1, 7);
    orders = [nb; nf; nk];
    orders = orders(:)';    % 1행 21열
    opt = oeOptions('EnforceStability', true);
    try
        sys_oe = oe(data_norm, orders, opt);
        [~, fit, ~] = compare(data_norm, sys_oe);
        fit_val = fit(1); % 첫 출력(단일출력) 기준
        if fit_val > best_fit
            best_fit = fit_val;
            best_orders = orders;
            disp(['New best fit: ', num2str(fit_val), ...
                  ' with [nb,nf,nk]=', num2str([nbv nfv nkv])])
        end
    catch
        % 오류 무시(불안정 등)
    end
end
end
end

disp('======== OE 최종 튜닝 결과 ========')
disp(['best accuracy: ', num2str(best_fit)])
disp(['optimal orders: ', mat2str(best_orders)])

% 최적 차수로 다시 학습/예측/시각화
opt = oeOptions('EnforceStability', true);
sys_oe_best = oe(data_norm, best_orders, opt);
compare(data_norm, sys_oe_best)
title('OE model (nomalized)')

% 역정규화 예측
Y_hat_norm = sim(sys_oe_best, U_norm);
Y_hat = Y_hat_norm * sigma_Y + mu_Y;
figure;
plot(Y, 'k'); hold on;
plot(Y_hat, 'r');
legend('real temp', 'optimal OE prediction');
xlabel('time');
ylabel('temp(℃)');
title('OE model (real unit)');

% RMSE/적합도(실제 단위)
rmse = sqrt(mean((Y - Y_hat).^2));
fit = 100 * (1 - norm(Y - Y_hat)/norm(Y - mean(Y)));
disp(['optimal OE model RMSE: ', num2str(rmse)]);
disp(['optimal OE model accuracy: ', num2str(fit), ' %']);
```


## SSM 코드
```matlab
% 1. 입력, 출력 정규화
mu_U = mean(U);
sigma_U = std(U);
U_norm = (U - mu_U) ./ sigma_U;
mu_Y = mean(Y);
sigma_Y = std(Y);
Y_norm = (Y - mu_Y) ./ sigma_Y;
Ts = 5; % 샘플타임

% 2. iddata 객체 생성
data_norm = iddata(Y_norm, U_norm, Ts);

% 3. SSM 차수 후보군
order_cand = 1:5;   % 보통 2~5에서 많이 사용, 더 넓게도 가능

% 4. 반복 튜닝
best_fit = -Inf;
best_order = NaN;
best_sys = [];
fit_history = [];

for order = order_cand
    opt = n4sidOptions('EnforceStability', true);
    try
        sys_ssm = n4sid(data_norm, order, opt);
        [~, fit, ~] = compare(data_norm, sys_ssm);
        fit_val = fit(1);
        fit_history(end+1,:) = [order, fit_val];
        disp(['order = ', num2str(order), ', fit = ', num2str(fit_val)]);
        if fit_val > best_fit
            best_fit = fit_val;
            best_order = order;
            best_sys = sys_ssm;
        end
    catch
        disp(['order = ', num2str(order), ' 실패 (불안정 등)']);
    end
end

disp('========= SSM(n4sid) 튜닝 결과 =========')
disp(['best order: ', num2str(best_order)]);
disp(['best accuracy: ', num2str(best_fit)]);

% 5. 최고 차수 모델로 예측 및 시각화
% compare (정규화 단위)
compare(data_norm, best_sys)
title(['optimal SSM (nomalized), order=', num2str(best_order)]);

% 실제 단위 예측/시각화
Y_hat_norm = sim(best_sys, U_norm);
Y_hat = Y_hat_norm * sigma_Y + mu_Y;
figure; plot(Y, 'k'); hold on; plot(Y_hat, 'b');
legend('real temp', 'optimal SSM prediction');
xlabel('time'); ylabel('temp(℃)');
title(['optimal SSM prediction (real unit, order=', num2str(best_order), ')']);

% 성능 지표
rmse = sqrt(mean((Y - Y_hat).^2));
fit = 100 * (1 - norm(Y - Y_hat)/norm(Y - mean(Y)));
disp(['optimal SSM RMSE: ', num2str(rmse)]);
disp(['optimal SSM real unit accuracy: ', num2str(fit), ' %']);
```



## 기타모델 테스트( arx, armax, BJ)
```matlab
% 예: ARX (전달함수)
na = 2; nb = [2 2 2 2 2 2 2]; nk = [1 1 1 1 1 1 1];
sys_arx = arx(data, [na nb nk]);
compare(data, sys_arx)
resid(data, sys_arx)


% (2) ARMAX
sys_armax = armax(data, [na nb nk 2]); % 마지막 2는 잡음 차수 nc (실험적으로 조정)
compare(data, sys_armax)

```

arx: 정확도 27% 대충 경향은 따라감

armax: 정확도 37%

bj: 정확도 47%








## Dwell time 고려

에어컨이 작동되고 온도가 떨어지는 구간의 외기온도, 실내온도 데이터

에어컨이 꺼지고 온도가 올라가는 구간의 외기온도, 실내온도 데이터 

각각 가져와서 플롯해보고, $\tau$ 산출해보기



온도변화 수식 (1차 시스템)
$$
T(t) = T_{\mathrm{env}} + (T_{\mathrm{init}} - T_{\mathrm{env}}) \cdot e^{-t/\tau}
$$

Dwell time 산정
중심 온도가 목표온도의 99%에 도달하는 시간을 구하려면 

$$
\frac{|T(t) - T_{\mathrm{env}}|}{|T_{\mathrm{init}} - T_{\mathrm{env}}|} = 0.01
$$

$$
e^{-t/\tau} = 0.01 \implies t = -\tau \ln(0.01) \approx 4.6\tau
$$

Dwell time= $4.6\tau$, 95% 도달은 $3\tau$
$$
T(t) = T_{\mathrm{env}} + (T_{\mathrm{init}} - T_{\mathrm{env}}) e^{-t/\tau}
$$

$$
\text{dwell time} \approx 4.6\tau
$$
