import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

def model_prediction(data):
    logging.info("모델 시작")

    # 기존 CSV 파일 읽어오기
    existing_data = pd.read_csv('flask/data/bitcoin2023.csv')
    # 신규 데이터
    new_data = pd.DataFrame(data)

    # 기존 데이터를 신규 데이터와 합치기
    merged_data = pd.concat([existing_data, new_data])

    # 데이터 프레임으로 변환
    df = pd.DataFrame(merged_data)

    # 데이터 전처리
    df['candle_date_time_kst'] = pd.to_datetime(df['candle_date_time_kst'], format='%Y-%m-%dT%H:%M:%S')
    df.set_index('candle_date_time_kst', inplace=True)

    # 데이터 정규화
    scaler = MinMaxScaler()
    scale_cols = ['trade_price']
    scaled = scaler.fit_transform(df[scale_cols])

    # 시퀀스 길이
    sequence_length = 30

    # 데이터셋 생성
    def create_dataset(data, sequence_length):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length][0])
        return np.array(X), np.array(y)

    model = keras.models.load_model("../h5/my_model.h5")  # 모델 로드

    X_new = scaled[-sequence_length:]  # 가장 최근의 30개 데이터를 사용하여 예측 수행
    X_test = np.array([X_new])  # 예측에 사용할 데이터 형식에 맞게 가공

    y_pred = model.predict(X_test)  # 예측 수행

    # 역정규화 함수 정의
    def reverse_min_max_scaling(org_x, x):
        org_x_np = np.asarray(org_x)
        x_np = np.asarray(x)
        return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()

    # 예측값을 원래 스케일로 변환
    y_pred_original = reverse_min_max_scaling(df['trade_price'], y_pred)

    current_date = df.index[-1].to_pydatetime()
    future_dates = [current_date + datetime.timedelta(hours=i+1) for i in range(1)]

    
    # 확률 계산
    probability = y_pred[0][0] * 100

    y_pred_dict = {
        'trade_date': future_dates,
        'prediction': y_pred_original.flatten().tolist(),
        'probability': probability
    }

    return y_pred_dict
