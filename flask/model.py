import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Dropout, Reshape

def model_prediction(data):
    logging.info("모델 시작")

    # 데이터 프레임으로 변환
    df = pd.DataFrame(data)

    # 데이터 전처리
    df['candle_date_time_kst'] = pd.to_datetime(df['candle_date_time_kst'], format='%Y-%m-%dT%H:%M:%S')
    df.set_index('candle_date_time_kst', inplace=True)

    # 데이터 정규화
    scaler = MinMaxScaler()
    scale_cols = ['opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_price', 'candle_acc_trade_volume']
    scaled = scaler.fit_transform(df[scale_cols])

    # 데이터셋 생성
    def create_dataset(data, target, time_steps=1):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:(i + time_steps)])
            y.append(target[i + time_steps])
        return np.array(X), np.array(y)

    # 데이터 분할 비율
    train_ratio = 0.6  # 학습 데이터 비율을 60%로 설정
    val_ratio = 0.2    # 검증 데이터 비율을 20%로 설정
    test_ratio = 0.2   # 테스트 데이터 비율을 20%로 설정

    # 학습 데이터와 테스트 데이터 분할
    test_data_size = int(len(scaled) * test_ratio)
    train_val_data = scaled[:-test_data_size]
    test_data = scaled[-test_data_size:]

    # 학습 데이터와 검증 데이터 분할
    val_data_size = int(len(train_val_data) * val_ratio / (train_ratio + val_ratio))
    train_data = train_val_data[:-val_data_size]
    val_data = train_val_data[-val_data_size:]

    n_timesteps = 4

    # 학습 데이터셋 생성
    X_train, y_train = create_dataset(train_data, train_data[:, 3], n_timesteps)

    # 검증 데이터셋 생성
    X_val, y_val = create_dataset(val_data, val_data[:, 3], n_timesteps)

    # 테스트 데이터셋 생성
    X_test, y_test = create_dataset(test_data, test_data[:, 3], n_timesteps)

    # CNN 모델과 LSTM 모델
    n_features = len(scale_cols)
    
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.025))
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
    model.add(Dropout(0.025))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))  # Add Dense layer to match the number of features
    model.add(Reshape((-1, 64)))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    # 모델 학습하기
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32)

    # 모델 평가하기
    score = model.evaluate(X_test, y_test)

    # 모델 예측하기
    y_pred = model.predict(X_test)
    def reverse_min_max_scaling(org_x, x):
        org_x_np = np.asarray(org_x)
        x_np = np.asarray(x)
        return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()

    # 예측 결과 역변환
    y_pred_reshaped = np.reshape(y_pred, (-1, 1))
    y_pred_original = reverse_min_max_scaling(df['trade_price'], y_pred_reshaped)

    # 미래 날짜 계산
    current_date = df.index[-1].to_pydatetime()
    future_dates = [current_date + datetime.timedelta(hours=i+1) for i in range(1)]

    # 예측 결과를 저장할 딕셔너리
    y_pred_dict = {
        'trade_date': future_dates,
        'prediction': [y_pred_original.flatten()[0]]
    }

    return y_pred_dict
