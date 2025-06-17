# ANN 전이함수 구현
## 라이브러리 임포트
```PY
!pip install joblib
!pip install torch

# 2. 데이터 전처리 및 학습 코드 시작
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import joblib
from google.colab import files
import io
```

## 파일 업로드
```PY
# ✅ 파일 업로드
uploaded = files.upload()
```

## 전처리 및 학습
```PY
# ✅ 날짜-시간 보정
def attach_date_to_time(df, filename):
    date_str = filename.split('_')[-1].split('.')[0]  # 예: '20240909'
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce')
    df['Time'] = df['Time'].apply(lambda t: datetime.combine(date_obj.date(), t.time()) if pd.notnull(t) else pd.NaT)
    return df

# ✅ 전처리
merged_all = []
prediction_step = 120  # 10분 후 예측 (5초 간격 데이터 기준)

for fname in uploaded:
    df = pd.read_csv(io.BytesIO(uploaded[fname]))
    df = attach_date_to_time(df, fname)
    df.columns = df.columns.str.strip()
    df = df.sort_values(['Time', 'Auto Id'])

    df1 = df[df['Auto Id'] == 2].reset_index(drop=True)
    df2 = df[df['Auto Id'] == 3].reset_index(drop=True)

    df1 = df1.rename(columns={'Frun': 'frun1', 'Tcon': 'tcon1', 'Tid': 'Tid1'})
    df2 = df2.rename(columns={'Frun': 'frun2', 'Tcon': 'tcon2'})

    df1['on_off1'] = df1['tcon1'].apply(lambda x: 1 if x != 0 and not pd.isna(x) else 0)
    df2['on_off2'] = df2['tcon2'].apply(lambda x: 1 if x != 0 and not pd.isna(x) else 0)

    merged = pd.merge(df1[['Time', 'frun1', 'tcon1', 'on_off1', 'Tid1', 'Tod']],
                      df2[['Time', 'frun2', 'tcon2', 'on_off2']],
                      on='Time', how='inner')

    for col in ['frun1', 'tcon1', 'on_off1', 'frun2', 'tcon2', 'on_off2']:
        merged[f'{col}_prev'] = merged[col].shift(prediction_step)

    merged['Tid1_next'] = merged['Tid1'].shift(-prediction_step)
    merged = merged.dropna().reset_index(drop=True)
    merged_all.append(merged)

# ✅ 전체 병합
merged = pd.concat(merged_all).reset_index(drop=True)

# ✅ 입력/출력 정의
input_cols = ['Tid1', 'Tod',
              'frun1_prev', 'frun2_prev', 'tcon1_prev', 'tcon2_prev', 'on_off1_prev', 'on_off2_prev',
              'frun1', 'frun2', 'tcon1', 'tcon2', 'on_off1', 'on_off2']
output_cols = ['Tid1_next']

X = merged[input_cols].values
Y = merged[output_cols].values

# ✅ 정규화
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X_scaled = x_scaler.fit_transform(X)
Y_scaled = y_scaler.fit_transform(Y)

joblib.dump(x_scaler, 'x_scaler.pkl')
joblib.dump(y_scaler, 'y_scaler.pkl')

# ✅ 학습/검증 분할
X_train, X_val, Y_train, Y_val = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

# ✅ PyTorch Dataset
class AirconDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_loader = DataLoader(AirconDataset(X_train, Y_train), batch_size=16, shuffle=True)
val_loader = DataLoader(AirconDataset(X_val, Y_val), batch_size=16)

# ✅ ANN 모델 정의
class ANN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)

model = ANN(input_dim=len(input_cols), output_dim=len(output_cols))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ✅ 학습 루프 (평균 Loss 계산)
for epoch in range(200):
    model.train()
    train_loss = 0
    train_batches = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_batches += 1

    model.eval()
    val_loss = 0
    val_batches = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            val_loss += loss.item()
            val_batches += 1

    # 평균 Loss 계산
    avg_train_loss = train_loss / train_batches
    avg_val_loss = val_loss / val_batches

    print(f"Epoch {epoch+1:03d} | Avg Train MSE: {avg_train_loss:.4f} | Avg Val MSE: {avg_val_loss:.4f}")

# ✅ 모델 저장
torch.save(model.state_dict(), 'aircon_tid1_predictor_2.pt')
```

## 모델 평가
```py
import matplotlib.pyplot as plt
import numpy as np

# 1. 모델을 평가 모드로 설정
model.eval()

# 2. 검증 데이터셋에서 10개 무작위 샘플 선택
num_samples = 10
indices = np.random.choice(len(X_val), size=num_samples, replace=False)
X_sample = torch.tensor(X_val[indices], dtype=torch.float32)
Y_true_scaled = Y_val[indices]

# 3. 예측 수행 (정규화된 출력 → 역변환)
with torch.no_grad():
    Y_pred_scaled = model(X_sample).numpy()

# 4. 스케일링 복원 (정규화 해제)
Y_true = y_scaler.inverse_transform(Y_true_scaled)
Y_pred = y_scaler.inverse_transform(Y_pred_scaled)

# 5. 결과 출력
print(f"{'Sample':<6} | {'True Tid1(t+1)':<20} | {'Predicted Tid1(t+1)':<25}")
print("-" * 60)
for i in range(num_samples):
    print(f"{i:<6} | {Y_true[i][0]:<20.4f} | {Y_pred[i][0]:<25.4f}")

# 6. 시각화
plt.figure(figsize=(8, 4))
plt.plot(Y_true[:, 0], 'o-', label='True Tid1(t+1)')
plt.plot(Y_pred[:, 0], 'x--', label='Predicted Tid1(t+1)')
plt.title("Comparison of True vs Predicted Tid1(t+1)")
plt.xlabel("Sample Index")
plt.ylabel("Temperature")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

참고로 학습이 끝난 모델의 PT와 정규화 스케일러를 다운받아놓고 나중에 DQN환경 정의 할때 쓰시면 됩니다.

## 파일 다운로드
```py
files.download('OOOO')
```