import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

existing_data = pd.read_csv("flask/data/bitcoin2023.csv")

df = pd.DataFrame(existing_data)

# 데이터 전처리
df['candle_date_time_kst'] = pd.to_datetime(df['candle_date_time_kst'], format='%Y-%m-%dT%H:%M:%S')
df.set_index('candle_date_time_kst', inplace=True)

# 데이터 정규화
scaler = MinMaxScaler()
scale_cols = ['trade_price']
scaled = scaler.fit_transform(df[scale_cols])
test_scaled = scaler.transform(df[scale_cols])

# 데이터 분할
train_size = int(len(df) * 0.8)
train_data = df[:train_size]
test_data = df[train_size:]

# 시퀀스 길이
sequence_length = 30

# 데이터셋 생성
def create_dataset(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length][0])
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(scaled, sequence_length)
X_test, y_test = create_dataset(test_scaled, sequence_length)

# CNN 모델과 LSTM 모델

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(sequence_length, 1)))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(Dropout(0.1))
model.add(LSTM(units=64, activation='tanh', return_sequences=True))
model.add(LSTM(units=64, activation='tanh'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1))
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


# 모델 학습하기
model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping])

    # 모델 저장하기
model.save("../h5/my_model.h5")

'''
# 모델 평가하기
score = model.evaluate(X_test, y_test)

# 모델 예측하기
y_pred = model.predict(X_train)

def reverse_min_max_scaling(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()

# 예측 결과 역변환
y_pred_original = reverse_min_max_scaling(df['trade_price'], y_pred)

# 미래 날짜 계산
current_date = df.index[-1].to_pydatetime()
future_dates = [current_date + datetime.timedelta(hours=i+1) for i in range(1)]

# 예측 결과를 저장할 딕셔너리
y_pred_dict = {
    'trade_date': future_dates,
    'prediction': y_pred_original.flatten().tolist()
}

return y_pred_dict
'''