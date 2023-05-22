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

# logging 설정 추가
logging.basicConfig(level=logging.INFO)

def model_prediction(data):
    logging.info("모델 시작")
    data = pd.read_csv("./data.csv")

    if data.empty:
        logging.info("데이터 비어있습니다")
        return data

    # 데이터 프레임으로 변환
    df = pd.DataFrame(data)

    # 데이터 전처리
    df['candle_date_time_kst'] = pd.to_datetime(df['candle_date_time_kst'], format='%Y-%m-%dT%H:%M:%S')
    df.set_index('candle_date_time_kst', inplace=True)

    # 데이터 정규화
    scaler = MinMaxScaler()
    scale_cols = ['opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_price', 'candle_acc_trade_volume']
    scaled = scaler.fit_transform(df[scale_cols])

    # 데이터셋 분리 비율
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    # 학습 데이터와 테스트 데이터로 분리
    test_data_size = int(len(df) * test_ratio)
    train_val_data = df[:-test_data_size]
    test_data = df[-test_data_size:]

    # 학습 데이터와 검증 데이터로 분리
    val_data_size = int(len(train_val_data) * val_ratio / (train_ratio + val_ratio))
    train_data = train_val_data[:-val_data_size]
    val_data = train_val_data[-val_data_size:]

    # 데이터셋 생성 함수
    def create_dataset(X, y, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X.iloc[i:(i + time_steps)].values)
            ys.append(y.iloc[i + time_steps])
        return np.array(Xs), np.array(ys)

    n_timesteps = 24
    # 학습 데이터셋 생성
    X_train, y_train = create_dataset(train_data, train_data['trade_price'], n_timesteps)

    # 검증 데이터셋 생성
    X_val, y_val = create_dataset(val_data, val_data['trade_price'], n_timesteps)

    # 테스트 데이터셋 생성
    X_test, y_test = create_dataset(test_data, test_data['trade_price'], n_timesteps)

    # CNN 모델과 LSTM 모델
    n_features = len(scale_cols)

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.025))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.025))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Reshape((-1, 64)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mse',
                  optimizer=tf.keras.optimizers.Adam())

    # 모델 학습하기
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32)

    # 모델 평가하기
    score = model.evaluate(X_test, y_test)

    # 모델 예측하기
    y_pred = model.predict(X_test)

    last_sequence = scaled[-n_timesteps:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    future_pred = model.predict(last_sequence)

    # 미래 날짜 계산
    current_date = df.index[-1].to_pydatetime()
    future_dates = [current_date + datetime.timedelta(days=i+1) for i in range(1)]

    # 예측 결과를 저장할 딕셔너리
    y_pred_dict = {
        'dateday': future_dates,
        'prediction': future_pred.flatten().tolist()
    }
    return y_pred_dict