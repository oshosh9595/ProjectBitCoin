from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import schedule
import time
import numpy as np
import logging
import pandas as pd
from model import model_prediction
from database import mysql_connection
from datetime import datetime
from testapi import testapi
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Dropout, Reshape

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
#global model
#model = tf.keras.models.load_model("./save.h5")

def load_model():
    global model
    model = tf.keras.models.load_model("save.h5")

@app.route('/')
def test():
    logging.info("데이터 수집 시작")
    exec(open("testapi.py").read())
    collect_and_predict()

def collect_and_predict():
    #if model is None:
    #    load_model()
    exec(open("model.py").read())
    #model_prediction()
    mysql_connection()

def schedule_job():
    schedule.every(3).minutes.do(collect_and_predict)
    while True:
        schedule.run_pending() #schedule을 사용하여 작업을 예약하고 run_pending() 메서드를 호출하`면 현재 예약된 작업 중 대기 중인 작업을 실행할 수 있습니다.
        time.sleep(3) # 시간 지정
#@app.route('/')
#def job():
#    logging.info("데이터 수집 및 예측 시작")
#    collect_and_predict()
#    logging.info("데이터 수집 및 예측 완료")

# 스케줄링 설정 - 매 30분마다 job 함수 실행

if __name__ == "__main__":
    schedule_job()
    app.run(debug=True)