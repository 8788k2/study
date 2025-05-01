# TO DO LIST 
교수님 출장가기 전에 개인연구주제 제안내용 정리하기, LG 3차년도 과제내용 PPT 만들기 두가지 주제를 중심으로 할 일을 정리하는것을 목표로 한다.

## 공통할 일
내가 생각하는 개인 연구주제와 LG과제는 공통적으로 비선형적이고 시간 지연이 있는 타임바링 시스템(HVAC, 수처리 등)에 대해, 물리모델 없이 데이터 기반 머신러닝(LSTM/DeepSSM 등)으로 전이모델을 학습하고, 이를 제어 이론 또는 강화학습 환경에 연결하는 것으로 요약할 수 있다.

따라서 해당 개념을 블록다이어그램 등의 형식으로 PPT에 들어갈 그림으로 정리해두면 두 과제 모두에 사용할 수 있을 것으로 기대된다. 


## LG LSTM 시계열 학습 가능한지
유의미해보이는 데이터 컬럼 선택 완료


인풋 아웃풋 어떻게 설정할래

순수 현재 시점의 상태와 입력만으로 다음 상태 예측하는게 현실적으로 가능한가??


## 코드로 구상한거 구현해보기
콜랩환경에서 LSTM을 이용한 시계열 데이터학습 및 히든스테이트 추출

추출받은 히든 스테이트 추가해서 확장된 차원의 온도 예측기

강화학습 환경 마련까지 구현해보자



히든 스테이트 64 차원
loss는 MSE

이전 60개 오분전까지 타입스텝을 인풋으로
30초 단위로 10개 5분뒤까지의 방온도를 아웃풋으로 

## 모델 성능 결과
경남 사무실 9월 3일 데이터로 모델 테스트 결과
![0903_1](images/0903예시1.png)
![0903_2](images/0903예시2.png)
![0903_3](images/0903예시3.png)

경남 사무실 9월 4일 데이터로 모델 테스트 결과
![0904_1](images/0904예시1.png)
![0904_2](images/0904예시2.png)
![0904_3](images/0904예시3.png)

공대 7호관 4월 28일 데이터로 모델 테스트 결과
![0428_1](images/공대7호관예시1.png)
![0428_2](images/공대7호관예시2.png)
![0428_3](images/공대7호관예시3.png)



### 변수들 간 상관관계 분석
```py
# 구글 코랩용: 파일 업로드 라이브러리 호출
from google.colab import files
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 파일 업로드
uploaded = files.upload()

# 업로드된 파일 읽기
# uploaded은 딕셔너리 형태라 key()로 파일 이름 가져올 수 있음
filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)

# 사용할 컬럼만 선택
features = ['Thmo On', 'Tcon', 'Frun', 'Tpip_in', 'Tpip_out', 'Tod', 'Power', 'Tbdy']
df_selected = df[features]

# 상관계수 계산
corr_matrix = df_selected.corr()

# 히트맵 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()
```



### CSV 파일 업로드 및 데이터 준비
```PY
from google.colab import files
import pandas as pd
import numpy as np

# 파일 업로드
uploaded = files.upload()

# 파일 읽기
filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)

# 사용할 컬럼 선택
features = ['Auto Id', 'Thmo On', 'Tcon', 'Frun', 'Tpip_in', 'Tpip_out', 'Tod', 'Power', 'Tbdy']
df = df[features]

# 결측치 처리 (간단히 드랍)
df = df.dropna()

# 정규화 (MinMaxScaler)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
# Auto id는 정규화에서 제외
df_scaled = df.copy()
df_scaled[df.columns[1:]] = scaler.fit_transform(df[df.columns[1:]])
```

### 데이터셋 정의
```py
from torch.utils.data import Dataset, DataLoader
import torch

class HVACDatasetV2(Dataset):
    def __init__(self, data, input_window, output_window, stride, output_stride):
        self.data = data
        self.input_window = input_window
        self.output_window = output_window
        self.stride = stride
        self.output_stride = output_stride
        self.X, self.Y = self.create_sequences()

    def create_sequences(self):
        X = []
        Y = []
        for auto_id, group in self.data.groupby('Auto Id'):
            group = group.drop(columns=['Auto Id']).values
            L = len(group)
            for i in range(0, L - self.input_window - self.output_window * self.output_stride, self.stride):
                x_seq = group[i:i+self.input_window, :]
                # output_stride 만큼 건너뛴 Tbdy 예측
                y_seq = group[i+self.input_window : i+self.input_window+self.output_window*self.output_stride : self.output_stride, -1]
                X.append(x_seq)
                Y.append(y_seq)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
```

### 데이터 준비
```py
# 하이퍼파라미터
input_window = 60
output_window = 10
stride = 1
output_stride = 6

# 새로 만든 데이터셋 클래스 활용
dataset = HVACDatasetV2(df_scaled, input_window, output_window, stride, output_stride)

# 학습/검증 분할
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# 데이터로더 생성
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
```

### LSTM 인코더 디코더 모델
```PY
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers=1):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, hidden, cell, output_len):
        batch_size = hidden.size(1)
        outputs = []
        input_step = torch.zeros((batch_size, 1, hidden.size(2))).to(hidden.device)  # 제로 입력
        for _ in range(output_len):
            output, (hidden, cell) = self.lstm(input_step, (hidden, cell))
            pred = self.fc(output.squeeze(1))  # (batch, output_dim)
            outputs.append(pred.unsqueeze(1))
            input_step = output
        outputs = torch.cat(outputs, dim=1)  # (batch, output_len, output_dim)
        return outputs

class LSTMSeq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_len):
        super(LSTMSeq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, 1)
        self.output_len = output_len

    def forward(self, src):
        hidden, cell = self.encoder(src)
        output = self.decoder(hidden, cell, self.output_len)
        return output.squeeze(-1)  # (batch, output_len)
```

### 학습 설정
```PY
# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 생성
input_dim = 8  # 입력 피처 개수
hidden_dim = 64
output_len = output_window

model = LSTMSeq2Seq(input_dim, hidden_dim, output_len).to(device)

# 손실함수와 옵티마이저
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

### 학습 루프
```PY
from tqdm import tqdm
import matplotlib.pyplot as plt

n_epochs = 50
train_losses = []
val_losses = []

for epoch in range(n_epochs):
    model.train()
    train_loss = 0

    # tqdm progress bar
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)
    for X_batch, Y_batch in loop:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, Y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, Y_batch)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"[{epoch+1}/{n_epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
```

### 손실 시각화
```py
# 모델을 evaluation 모드로 전환
model.eval()

# 검증 데이터셋에서 하나의 배치를 가져오기
X_batch, Y_batch = next(iter(val_loader))
X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

# 예측
with torch.no_grad():
    y_pred = model(X_batch)  # (batch_size, output_len)

# CPU로 옮기고 numpy 변환
y_pred = y_pred.cpu().numpy()
y_true = Y_batch.cpu().numpy()

import random
# 하나의 샘플만 선택
sample_idx = random.randint(0, y_pred.shape[0] - 1)
predicted = y_pred[sample_idx]      # 정규화된 예측
ground_truth = y_true[sample_idx]   # 정규화된 실제

# 30초 간격 시간축 생성
time_axis = np.arange(30, 30 * (len(ground_truth) + 1), 30)

# ✅ 정규화 해제 (Tbdy는 마지막 feature이므로 열 index = -1)
# scaler는 원래 전체 feature (8개)에 대해 fit 되었으므로, Tbdy만 복원해야 함

tbdy_min = scaler.data_min_[-1]
tbdy_max = scaler.data_max_[-1]

predicted_real = predicted * (tbdy_max - tbdy_min) + tbdy_min
ground_truth_real = ground_truth * (tbdy_max - tbdy_min) + tbdy_min

# ✅ Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(time_axis, ground_truth_real, label='Ground Truth Tbdy (°C)', marker='o')
plt.plot(time_axis, predicted_real, label='Predicted Tbdy (°C)', marker='x')
plt.xlabel('Prediction Horizon (Seconds)')
plt.ylabel('Tbdy (°C)')
plt.title('Predicted vs Ground Truth Tbdy (Denormalized, 30s Interval)')
plt.ylim(tbdy_min - 1, tbdy_max + 1)  # y축 확대
plt.legend()
plt.grid(True)
plt.show()
```

### 다른 날짜 불러오기 및 전처리
```PY
from google.colab import files
import pandas as pd

# 📂 1. 파일 업로드
uploaded = files.upload()  # 파일 업로드 창이 뜸

# 📄 2. 업로드된 파일명 추출
filename = list(uploaded.keys())[0]

# 📊 3. 데이터 불러오기
new_df = pd.read_csv(filename)

# ✅ 4. Feature 선택 및 결측치 제거
features = ['Auto Id', 'Thmo On', 'Tcon', 'Frun', 'Tpip_in', 'Tpip_out', 'Tod', 'Power', 'Tbdy']
new_df = new_df[features].dropna()

# 🔄 5. 기존 scaler를 사용하여 정규화
new_df_scaled = new_df.copy()
new_df_scaled[new_df.columns[1:]] = scaler.transform(new_df[new_df.columns[1:]])
```
### 새로운 데이터셋 생성
```py
new_dataset = HVACDatasetV2(
    new_df_scaled,
    input_window=60,
    output_window=10,
    stride=1,
    output_stride=6
)

new_loader = DataLoader(new_dataset, batch_size=64, shuffle=False)
```

### 모델 예측 및 성능검증
```py
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# 1. 모델 평가 모드
model.eval()
all_preds = []
all_trues = []

# 2. 배치별 예측 수집
with torch.no_grad():
    for X_batch, Y_batch in new_loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        pred = model(X_batch)
        all_preds.append(pred.cpu().numpy())
        all_trues.append(Y_batch.cpu().numpy())

# 3. 전체 병합
preds = np.concatenate(all_preds, axis=0)  # (전체 샘플 수, output_len)
trues = np.concatenate(all_trues, axis=0)

# 4. 정규화 해제
tbdy_index = new_df.columns[1:].get_loc('Tbdy')
tbdy_min = scaler.data_min_[tbdy_index]
tbdy_max = scaler.data_max_[tbdy_index]

preds_real = preds * (tbdy_max - tbdy_min) + tbdy_min
trues_real = trues * (tbdy_max - tbdy_min) + tbdy_min

# 5. 성능 지표 계산
mae = mean_absolute_error(trues_real.flatten(), preds_real.flatten())
mse = mean_squared_error(trues_real.flatten(), preds_real.flatten())
rmse = np.sqrt(mse)

print(f"New Data Evaluation — MAE: {mae:.4f} °C, RMSE: {rmse:.4f} °C")

import random
# 6. 시각화 (0번째 샘플을 예시로 출력)
sample_idx = random.randint(0, len(preds_real) - 1)
pred_sample = preds_real[sample_idx]    # (output_len,)
true_sample = trues_real[sample_idx]    # (output_len,)

# 시간축 생성: 30초 간격, 예측 스텝 수만큼
time_axis = np.arange(30, 30 * (len(true_sample) + 1), 30)

# 7. Plot
plt.figure(figsize=(10, 5))
plt.plot(time_axis, true_sample, label='Ground Truth Tbdy (°C)', marker='o')
plt.plot(time_axis, pred_sample, label='Predicted Tbdy (°C)', marker='x')
plt.xlabel('Prediction Horizon (Seconds)')
plt.ylabel('Tbdy (°C)')
plt.title(f'Prediction vs Ground Truth on New Data (Sample #{sample_idx})\nMAE: {mae:.3f} °C, RMSE: {rmse:.3f} °C')
plt.ylim(tbdy_min - 1, tbdy_max + 1)
plt.grid(True)
plt.legend()
plt.show()
```


## 학습한 모델 Local에 다운로드
### 내 컴퓨터로 다운로드
```py
# 1. 모델 저장
torch.save(model.state_dict(), "lstm_seq2seq_tbdy_model.pt")

# 2. 확인
import os
print("Saved files:", os.listdir())

# 3. 다운로드
from google.colab import files
files.download("lstm_seq2seq_tbdy_model.pt")
```

### Colab에서 로컬 모델 파일 업로드하여 불러오기
```py
from google.colab import files
uploaded = files.upload()  # 파일 업로드 창이 뜸

# 예: lstm_seq2seq_tbdy_model.pt 업로드했다고 가정
model.load_state_dict(torch.load("lstm_seq2seq_tbdy_model.pt"))
model.eval()
```